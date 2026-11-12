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
/// The combine's source choice over its 24 sources, the wide launches' headers
const combine = @import("combine.zig");

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
/// The prompt-width launch's norm operands (`moeTailWide`): W and eps read from one buffer WE f32 [D + 1]
pub const tail_we = @embedFile("kernels/tail/tail_we.metal");
/// The delta at prompt widths over a wide call's unjoined outputs (JOINLESS, up to `wide_max_sources` f32 sources
/// through the combine's pick), then tail_delta's statements
pub const tail_delta_wide = @embedFile("kernels/tail/tail_delta_wide.metal");
/// ... over f16 sources (PF_F16_PAIRS): `tail_delta_wide` with the source pointer half and the combine's pick_h
pub const tail_delta_wide_h = @embedFile("kernels/tail/tail_delta_wide_h.metal");
/// The launch at prompt widths (`moeTailWide`): inputs H [rows, D], s0..s23 (f32 [r_i, D]), loc int32 [rows, K, 2], WT
/// [rows, K], XS [rows, D], WE f32 [D + 1]; outputs H_NEW and HN in H's shape and dtype T; template T, D, TN, K. The
/// add_norm's own statements (tail_1, tail_2) at any row count: one threadgroup a row, as MLX's fast rms_norm's single-row
/// kernel runs every row of a [rows, D] call
pub const text_wide: [:0]const u8 = tail_we ++ tail_1 ++ tail_delta_wide ++ tail_2;
/// ... over f16 sources
pub const text_wide_h: [:0]const u8 = tail_we ++ tail_1 ++ tail_delta_wide_h ++ tail_2;
/// The wide launches' headers: the combine's source choice over its 24 sources (f32, or f16 for `text_wide_h`); no
/// file-scope contract pragma, so the norm's statements compile as the add_norm's do (the delta's own block pragma
/// keeps the combine's rounding points)
pub const header_wide: [:0]const u8 = combine.pick;
pub const header_wide_h: [:0]const u8 = combine.pick_h;

/// The rows a call takes: decode's one to a verify's eight
pub const max_rows: c_int = 8;
/// The widest top-k (Ling's 8)
pub const max_topk: c_int = 8;
/// The widest row the launch takes: add_norm's single-row kernel, D / 4 threads in one pass
pub const max_width: c_int = 4096;
/// The sources a parts launch binds: a decode call's device wave and up to seven miss parts
pub const max_sources: usize = 8;
/// The sources a wide launch binds: the combine's (JOINLESS's 24; H, loc, WT, XS, WE and the two outputs fill MLX's 31
/// buffers)
pub const wide_max_sources: usize = combine.max_sources;

const in_names = [_][*:0]const u8{ "H", "Y", "WT", "XS", "W", "eps" };
const parts_in_names = [_][*:0]const u8{ "H", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "loc", "WT", "XS", "W", "eps" };
pub const wide_in_names = [_][*:0]const u8{ "H", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "loc", "WT", "XS", "WE" };
pub const out_names = [_][*:0]const u8{ "H_NEW", "HN" };

comptime {
    if (parts_in_names.len != max_sources + 6) @compileError("parts_in_names must name H, max_sources sources, loc, WT, XS, W and eps");
    if (wide_in_names.len != wide_max_sources + 5) @compileError("wide_in_names must name H, wide_max_sources sources, loc, WT, XS and WE");
    if (wide_in_names.len + out_names.len > 31) @compileError("a wide launch binds at most MLX's 31 buffers");
}

const lock = Global(std.atomic.Value(u32), "ling_tail_lock", std.atomic.Value(u32).init(0));
const kernel = Global(?mlx.mlx_fast_metal_kernel, "ling_tail_kernel", null);
const parts_kernel = Global(?mlx.mlx_fast_metal_kernel, "ling_tail_parts_kernel", null);
const WideKernels = struct { f32_src: mlx.mlx_fast_metal_kernel, f16_src: mlx.mlx_fast_metal_kernel };
const wide_kernels = Global(?WideKernels, "ling_tail_wide_kernels", null);

/// This file's module state (`Global`): what a host that shares its units' state registers (root's `shared_state`).
pub const shared_state = .{ lock, kernel, parts_kernel, wide_kernels };

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

/// The two wide launches (f32 and f16 sources), made at the first wide call
fn wideKernelsOf() !WideKernels {
    while (lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer lock.get().*.store(0, .release);
    if (wide_kernels.get().*) |k| return k;
    const k32 = ks.newKernel("turboexl3_ling_moe_tail_wide", &wide_in_names, &out_names, text_wide, header_wide, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(k32);
    const k16 = ks.newKernel("turboexl3_ling_moe_tail_wide_h", &wide_in_names, &out_names, text_wide_h, header_wide_h, true) orelse return error.KernelCreateFailed;
    wide_kernels.get().* = .{ .f32_src = k32, .f16_src = k16 };
    return wide_kernels.get().*.?;
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

/// Whether `moeTailWide` takes a call of `rows` rows (any count: one threadgroup a row) of width `width` at top-k `topk`
/// in `xd`: the add_norm's single-row statements, which are MLX's fast rms_norm's up to its looped limit (4096 wide)
pub fn wideTakes(rows: c_int, width: c_int, topk: c_int, xd: mlx.mlx_dtype) bool {
    return rows >= 1 and topk >= 1 and topk <= max_topk and width >= 4 and width <= max_width and @rem(width, 4) == 0 and (xd == .bfloat16 or xd == .float16);
}

/// The source dtype of a wide call: every source f32 or every source f16 (PF_F16_PAIRS), else error.BadExl3Dtype (the
/// caller keeps the served chain, whose combine widens a mix)
fn wideSources(sources: []const mlx.mlx_array, d: c_int) !mlx.mlx_dtype {
    if (sources.len < 1 or sources.len > wide_max_sources) return error.BadExl3Shape;
    const sd = mlx.mlx_array_dtype(sources[0]);
    if (sd != .float32 and sd != .float16) return error.BadExl3Dtype;
    for (sources) |x| {
        if (mlx.mlx_array_dtype(x) != sd) return error.BadExl3Dtype;
        const xsh = mlx.getShape(x);
        if (xsh.len != 2 or xsh[0] < 1 or xsh[1] != d) return error.BadExl3Shape;
    }
    return sd;
}

/// A MoE layer's tail at prompt widths in one launch, over its wide routed call's unjoined outputs (JOINLESS): `h`
/// [..., D] in T (bf16 or f16; the residual before the MoE's add, a span's rows), `sources` 1 to `wide_max_sources`
/// arrays [r_i, D] all f32 or all f16 (the call's outputs, each pair's expert output at weight 1; f16 under
/// PF_F16_PAIRS, each word the f16 value of the f32 word the served call stores), `loc` int32 [rows, K, 2] (each of the
/// span's pairs' (source, row): every entry must name a source below `sources.len` and one of its rows, unchecked),
/// `weights` [rows, K] (the routes' weights: bf16, f16 or f32), `shared` [..., D] in T (the span's shared expert
/// output), `we` f32 [D + 1] (the next layer's input norm, or the final norm, widened to f32, then eps). Returns
/// h + (the routed sum in T + shared, in T) and its RMSNorm: the served chain's words at prompt widths (the combine
/// over the parts, the block's astype and add, MLX's add, MLX's fast rms_norm, whose single-row kernel the add_norm's
/// statements mirror at every row count). Any row count; K 1 to `max_topk`; D a multiple of 4 up to `max_width`;
/// error.BadExl3Shape / error.BadExl3Dtype otherwise (the caller keeps the chain).
pub fn moeTailWide(s: mlx.mlx_stream, h: mlx.mlx_array, sources: []const mlx.mlx_array, loc: mlx.mlx_array, weights: mlx.mlx_array, shared: mlx.mlx_array, we: mlx.mlx_array) !Out {
    const c = try wideCallOf(h, weights, loc, shared, we);
    const sd = try wideSources(sources, c.d);
    const k = try wideKernelsOf();
    return wideLaunch(s, if (sd == .float16) k.f16_src else k.f32_src, c, h, sources, loc, weights, shared, we);
}

/// `moeTailWide` on a kernel the caller made from a wide launch's operand names (`wide_in_names`, `out_names`): the
/// red checks' planted texts. The checks, the config and the operands are `moeTailWide`'s.
pub fn moeTailWideWith(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, h: mlx.mlx_array, sources: []const mlx.mlx_array, loc: mlx.mlx_array, weights: mlx.mlx_array, shared: mlx.mlx_array, we: mlx.mlx_array) !Out {
    const c = try wideCallOf(h, weights, loc, shared, we);
    _ = try wideSources(sources, c.d);
    return wideLaunch(s, kern, c, h, sources, loc, weights, shared, we);
}

/// The wide entries' checks but the sources': h, weights, loc, shared and we against the call
fn wideCallOf(h: mlx.mlx_array, weights: mlx.mlx_array, loc: mlx.mlx_array, shared: mlx.mlx_array, we: mlx.mlx_array) !Call {
    const hsh = mlx.getShape(h);
    if (hsh.len < 1) return error.BadExl3Shape;
    const d = hsh[hsh.len - 1];
    const n = rowsOf(hsh);
    const xd = mlx.mlx_array_dtype(h);
    const wsh = mlx.getShape(weights);
    if (wsh.len != 2 or wsh[0] != n) return error.BadExl3Shape;
    const k = wsh[1];
    if (!wideTakes(n, d, k, xd)) return error.BadExl3Shape;
    const wd = mlx.mlx_array_dtype(weights);
    if (wd != .bfloat16 and wd != .float16 and wd != .float32) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(loc) != .int32) return error.BadExl3Dtype;
    const lsh = mlx.getShape(loc);
    if (lsh.len != 3 or lsh[0] != n or lsh[1] != k or lsh[2] != 2) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(shared) != xd) return error.BadExl3Dtype;
    const ssh = mlx.getShape(shared);
    if (ssh.len < 1 or ssh[ssh.len - 1] != d or rowsOf(ssh) != n) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(we) != .float32) return error.BadExl3Dtype;
    if (mlx.getShape(we).len != 1 or mlx.getShape(we)[0] != d + 1) return error.BadExl3Shape;
    return .{ .hsh = hsh, .d = d, .n = n, .k = k, .xd = xd };
}

/// One wide launch: the sources in s0..s23 (an unused slot aliases source 0, which no loc entry names), in the
/// add_norm's config at the call's shape
fn wideLaunch(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, c: Call, h: mlx.mlx_array, sources: []const mlx.mlx_array, loc: mlx.mlx_array, weights: mlx.mlx_array, shared: mlx.mlx_array, we: mlx.mlx_array) !Out {
    var ins: [wide_in_names.len]mlx.mlx_array = undefined;
    ins[0] = h;
    for (ins[1 .. 1 + wide_max_sources], 0..) |*x, i| x.* = if (i < sources.len) sources[i] else sources[0];
    ins[1 + wide_max_sources] = loc;
    ins[2 + wide_max_sources] = weights;
    ins[3 + wide_max_sources] = shared;
    ins[4 + wide_max_sources] = we;
    return launch(s, kern, c, &ins);
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
