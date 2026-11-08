//! Ling-3.1-flash's MoE tail at decode and verify rows in one launch (`moeTail`). The served path ends a MoE layer in
//! five launches after its routed call's join: the source's routed sum (turboexl3_ling's streamed source:
//! `sum(multiply(astype(routed, f32), expand_dims(astype(weights, f32), -1)), -2)`, an astype and two ops), the MoE
//! block's astype of that sum to the activation dtype and its add of the shared expert, then the forward's tail: the
//! host's add_norm (mlx-serve sdk/blocks/norms/add_norm.zig), the residual add and the next layer's input norm (or the
//! final norm) in one dispatch. This launch is that add_norm's own kernel (one threadgroup of D / 4 threads a row, four
//! columns a thread) with its delta made in place: each column's routed sum in turboexl3's combine statements (each
//! product rounded to f32 on its own, MLX's col_reduce_small fold over the top-k in k order), rounded to the activation
//! dtype, plus the shared expert rounded again, so the residual and the normed rows are the chain's words.
//! `moeTailParts` is the same launch over the call's unjoined outputs (the device wave's rows and its miss parts', each
//! pair's row read in place at its (source, row)): the streamer's join (a concatenate and a take) folded in, so the
//! words of `moeTail` over that join.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const decode = @import("../../core/decode.zig");

// ── texts ──

/// mlx-serve's add_norm up to its delta (the formatted plain arm's statements, verbatim)
pub const tail_1 = @embedFile("kernels/tail/tail_1.metal");
/// the delta: the routed sum, its astype, the shared expert's add
pub const tail_delta = @embedFile("kernels/tail/tail_delta.metal");
/// the rest of mlx-serve's add_norm: the residual, the sum of squares, the norm (verbatim)
pub const tail_2 = @embedFile("kernels/tail/tail_2.metal");
/// The launch: inputs H [rows, D] (the residual), Y f32 [rows, K, D] (the call's joined rows, each pair at weight 1), WT
/// [rows, K] (the routes' weights), XS [rows, D] (the shared expert's output), W [D] (the norm's weight), eps (0-dim
/// f32); outputs H_NEW and HN in H's shape and dtype T; template T, D, TN, K
pub const text: [:0]const u8 = tail_1 ++ tail_delta ++ tail_2;
/// the delta over the call's unjoined outputs: each pair's row read in place at its (source, row), then tail_delta's
/// statements
pub const tail_delta_parts = @embedFile("kernels/tail/tail_delta_parts.metal");
/// The parts launch's header: the source choice
pub const pick = @embedFile("kernels/tail/pick.metal");
/// The launch over the call's unjoined outputs (`moeTailParts`): inputs H, s0..s7 (f32 [r_i, D], the call's waves'
/// outputs), loc int32 [rows, K, 2] (each pair's (source, row)), WT, XS, W, eps; outputs and template as `text`'s
pub const text_parts: [:0]const u8 = tail_1 ++ tail_delta_parts ++ tail_2;

/// The rows a call takes: decode's one to a verify's eight
pub const max_rows: c_int = 8;
/// The widest top-k (Ling's 8)
pub const max_topk: c_int = 8;
/// The widest row the launch takes: add_norm's single-row kernel, D / 4 threads in one pass
pub const max_width: c_int = 4096;
/// The sources a parts launch binds: a decode call's device wave and up to seven miss parts
pub const max_sources: usize = 8;

const in_names = [_][*:0]const u8{ "H", "Y", "WT", "XS", "W", "eps" };
const parts_in_names = [_][*:0]const u8{ "H", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "loc", "WT", "XS", "W", "eps" };
const out_names = [_][*:0]const u8{ "H_NEW", "HN" };

comptime {
    if (parts_in_names.len != max_sources + 6) @compileError("parts_in_names must name H, max_sources sources, loc, WT, XS, W and eps");
}

const lock = Global(std.atomic.Value(u32), "ling_tail_lock", std.atomic.Value(u32).init(0));
const kernel = Global(?mlx.mlx_fast_metal_kernel, "ling_tail_kernel", null);
const parts_kernel = Global(?mlx.mlx_fast_metal_kernel, "ling_tail_parts_kernel", null);

/// This file's module state (`Global`): what a host that shares its units' state registers (root's `shared_state`).
pub const shared_state = .{ lock, kernel, parts_kernel };

fn kernelOf() !mlx.mlx_fast_metal_kernel {
    while (lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer lock.get().*.store(0, .release);
    if (kernel.get().*) |k| return k;
    const k = ks.newKernel("turboexl3_ling_moe_tail", &in_names, &out_names, text, "", true) orelse return error.KernelCreateFailed;
    kernel.get().* = k;
    return k;
}

fn partsKernelOf() !mlx.mlx_fast_metal_kernel {
    while (lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer lock.get().*.store(0, .release);
    if (parts_kernel.get().*) |k| return k;
    const k = ks.newKernel("turboexl3_ling_moe_tail_parts", &parts_in_names, &out_names, text_parts, pick, true) orelse return error.KernelCreateFailed;
    parts_kernel.get().* = k;
    return k;
}

/// What `moeTail` returns: the residual and the normed rows, each in h's shape and dtype. The caller frees both.
pub const Out = struct { h: mlx.mlx_array, normed: mlx.mlx_array };

fn rowsOf(sh: []const c_int) c_int {
    var n: c_int = 1;
    for (sh[0 .. sh.len - 1]) |d| n *= d;
    return n;
}

/// Whether `moeTail` takes a call of `rows` rows of width `width` at top-k `topk` in `xd` (the caller keeps the chain
/// where it does not)
pub fn takes(rows: c_int, width: c_int, topk: c_int, xd: mlx.mlx_dtype) bool {
    return rows >= 1 and rows <= max_rows and topk >= 1 and topk <= max_topk and width >= 4 and width <= max_width and @rem(width, 4) == 0 and (xd == .bfloat16 or xd == .float16);
}

/// The MoE tail of a layer at decode or verify rows in one launch: `h` [..., D] (the residual before the MoE's add, in
/// the activation dtype T, bf16 or f16), `routed` f32 [rows, K, D] or [rows * K, D] (the routed call's joined rows in
/// routed order, each pair's expert output at weight 1), `weights` [rows, K] (the routes' weights: bf16, f16 or f32),
/// `shared` [..., D] in T (the shared expert's output, `rows` rows), `w` [D] in T (the next layer's input norm or the
/// final norm) and `eps` 0-dim f32. Returns h + (the routed sum in T + shared, in T) and its RMSNorm with w, the words of
/// the served chain (the source's three ops and astype, the block's astype and add, the host's add_norm). Rows 1 to
/// `max_rows`, K 1 to `max_topk`, D a multiple of 4 up to `max_width`; error.BadExl3Shape / error.BadExl3Dtype otherwise.
pub fn moeTail(s: mlx.mlx_stream, h: mlx.mlx_array, routed: mlx.mlx_array, weights: mlx.mlx_array, shared: mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array) !Out {
    const c = try callOf(h, weights);
    if (mlx.mlx_array_dtype(routed) != .float32) return error.BadExl3Dtype;
    if (mlx.mlx_array_size(routed) != @as(usize, @intCast(c.n * c.k * c.d))) return error.BadExl3Shape;
    const rsh = mlx.getShape(routed);
    if (rsh.len < 2 or rsh[rsh.len - 1] != c.d) return error.BadExl3Shape;
    try checkRest(c, shared, w, eps);
    return launch(s, try kernelOf(), c, &.{ h, routed, weights, shared, w, eps });
}

/// `moeTail` over the routed call's unjoined outputs, the decode join folded in: `sources` 1 to `max_sources` f32
/// [r_i, D] (the call's waves' outputs in the join's order, each pair's expert output at weight 1) and `loc` int32
/// [rows, K, 2] (each routed pair's (source, row): its last carrier in the join, so the row `take(concatenate(sources),
/// order)` gives it). Every entry must name a source below `sources.len` and a row below that source's rows: the
/// launch reads where loc points, unchecked. Each pair's row read in place, so the words of `moeTail` over that join;
/// the other operands, the checks and the errors as `moeTail`'s.
pub fn moeTailParts(s: mlx.mlx_stream, h: mlx.mlx_array, sources: []const mlx.mlx_array, loc: mlx.mlx_array, weights: mlx.mlx_array, shared: mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array) !Out {
    const c = try callOf(h, weights);
    if (sources.len < 1 or sources.len > max_sources) return error.BadExl3Shape;
    for (sources) |x| {
        if (mlx.mlx_array_dtype(x) != .float32) return error.BadExl3Dtype;
        const xsh = mlx.getShape(x);
        if (xsh.len != 2 or xsh[0] < 1 or xsh[1] != c.d) return error.BadExl3Shape;
    }
    if (mlx.mlx_array_dtype(loc) != .int32) return error.BadExl3Dtype;
    if (mlx.mlx_array_size(loc) != @as(usize, @intCast(2 * c.n * c.k))) return error.BadExl3Shape;
    try checkRest(c, shared, w, eps);
    var ins: [parts_in_names.len]mlx.mlx_array = undefined;
    ins[0] = h;
    // an unused slot aliases source 0 (no loc entry names it)
    for (ins[1 .. 1 + max_sources], 0..) |*x, i| x.* = if (i < sources.len) sources[i] else sources[0];
    ins[1 + max_sources] = loc;
    ins[2 + max_sources] = weights;
    ins[3 + max_sources] = shared;
    ins[4 + max_sources] = w;
    ins[5 + max_sources] = eps;
    return launch(s, try partsKernelOf(), c, &ins);
}

/// A tail call's checked shape: h's shape, width, rows and dtype, the top-k
const Call = struct { hsh: []const c_int, d: c_int, n: c_int, k: c_int, xd: mlx.mlx_dtype };

/// Both entries' first checks: `h` and `weights`
fn callOf(h: mlx.mlx_array, weights: mlx.mlx_array) !Call {
    const hsh = mlx.getShape(h);
    if (hsh.len < 1) return error.BadExl3Shape;
    const d = hsh[hsh.len - 1];
    const n = rowsOf(hsh);
    const xd = mlx.mlx_array_dtype(h);
    const wsh = mlx.getShape(weights);
    if (wsh.len != 2 or wsh[0] != n) return error.BadExl3Shape;
    const k = wsh[1];
    if (!takes(n, d, k, xd)) return error.BadExl3Shape;
    const wd = mlx.mlx_array_dtype(weights);
    if (wd != .bfloat16 and wd != .float16 and wd != .float32) return error.BadExl3Dtype;
    return .{ .hsh = hsh, .d = d, .n = n, .k = k, .xd = xd };
}

/// Both entries' last checks: `shared`, `w` and `eps` against the call
fn checkRest(c: Call, shared: mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array) !void {
    if (mlx.mlx_array_dtype(shared) != c.xd or mlx.mlx_array_dtype(w) != c.xd) return error.BadExl3Dtype;
    const ssh = mlx.getShape(shared);
    if (ssh.len < 1 or ssh[ssh.len - 1] != c.d or rowsOf(ssh) != c.n) return error.BadExl3Shape;
    if (mlx.getShape(w).len != 1 or mlx.getShape(w)[0] != c.d) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(eps) != .float32 or mlx.mlx_array_ndim(eps) != 0) return error.BadExl3Dtype;
}

/// One launch of `kern` over `ins`, in mlx-serve's add_norm config at the call's shape
fn launch(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, c: Call, ins: []const mlx.mlx_array) !Out {
    const tn: u32 = @intCast(@divExact(c.d, 4));
    const cfg = try ks.buildConfig(.{
        .grid = .{ tn * @as(u32, @intCast(c.n)), 1, 1 },
        .threadgroup = .{ tn, 1, 1 },
        .out_shapes = &.{ c.hsh, c.hsh },
        .out_dtypes = &.{ c.xd, c.xd },
        .template = &.{
            .{ .name = "T", .value = .{ .dtype = c.xd } },
            .{ .name = "D", .value = .{ .int = c.d } },
            .{ .name = "TN", .value = .{ .int = @intCast(tn) } },
            .{ .name = "K", .value = .{ .int = c.k } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [2]mlx.mlx_array = undefined;
    try ks.applyConfig(kern, cfg, ins, &o, s);
    return .{ .h = o[0], .normed = o[1] };
}
