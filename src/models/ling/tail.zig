//! Ling-3.1-flash's MoE tail at decode and verify rows in one launch (`moeTail`). The served path ends a MoE layer in
//! five launches after its routed call's join: the source's routed sum (turboexl3_ling's streamed source:
//! `sum(multiply(astype(routed, f32), expand_dims(astype(weights, f32), -1)), -2)`, an astype and two ops), the MoE
//! block's astype of that sum to the activation dtype and its add of the shared expert, then the forward's tail: the
//! host's add_norm (mlx-serve sdk/blocks/norms/add_norm.zig), the residual add and the next layer's input norm (or the
//! final norm) in one dispatch. This launch is that add_norm's own kernel (one threadgroup of D / 4 threads a row, four
//! columns a thread) with its delta made in place: each column's routed sum in turboexl3's combine statements (each
//! product rounded to f32 on its own, MLX's col_reduce_small fold over the top-k in k order), rounded to the activation
//! dtype, plus the shared expert rounded again, so the residual and the normed rows are the chain's words.
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

/// The rows a call takes: decode's one to a verify's eight
pub const max_rows: c_int = 8;
/// The widest top-k (Ling's 8)
pub const max_topk: c_int = 8;
/// The widest row the launch takes: add_norm's single-row kernel, D / 4 threads in one pass
pub const max_width: c_int = 4096;

const in_names = [_][*:0]const u8{ "H", "Y", "WT", "XS", "W", "eps" };
const out_names = [_][*:0]const u8{ "H_NEW", "HN" };

const lock = Global(std.atomic.Value(u32), "ling_tail_lock", std.atomic.Value(u32).init(0));
const kernel = Global(?mlx.mlx_fast_metal_kernel, "ling_tail_kernel", null);

/// This file's module state (`Global`): what a host that shares its units' state registers (root's `shared_state`).
pub const shared_state = .{ lock, kernel };

fn kernelOf() !mlx.mlx_fast_metal_kernel {
    while (lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer lock.get().*.store(0, .release);
    if (kernel.get().*) |k| return k;
    const k = ks.newKernel("turboexl3_ling_moe_tail", &in_names, &out_names, text, "", true) orelse return error.KernelCreateFailed;
    kernel.get().* = k;
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
    if (mlx.mlx_array_dtype(routed) != .float32) return error.BadExl3Dtype;
    if (mlx.mlx_array_size(routed) != @as(usize, @intCast(n * k * d))) return error.BadExl3Shape;
    const rsh = mlx.getShape(routed);
    if (rsh.len < 2 or rsh[rsh.len - 1] != d) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(shared) != xd or mlx.mlx_array_dtype(w) != xd) return error.BadExl3Dtype;
    const ssh = mlx.getShape(shared);
    if (ssh.len < 1 or ssh[ssh.len - 1] != d or rowsOf(ssh) != n) return error.BadExl3Shape;
    if (mlx.getShape(w).len != 1 or mlx.getShape(w)[0] != d) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(eps) != .float32 or mlx.mlx_array_ndim(eps) != 0) return error.BadExl3Dtype;
    const tn: u32 = @intCast(@divExact(d, 4));
    const cfg = try ks.buildConfig(.{
        .grid = .{ tn * @as(u32, @intCast(n)), 1, 1 },
        .threadgroup = .{ tn, 1, 1 },
        .out_shapes = &.{ hsh, hsh },
        .out_dtypes = &.{ xd, xd },
        .template = &.{
            .{ .name = "T", .value = .{ .dtype = xd } },
            .{ .name = "D", .value = .{ .int = d } },
            .{ .name = "TN", .value = .{ .int = @intCast(tn) } },
            .{ .name = "K", .value = .{ .int = k } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [2]mlx.mlx_array = undefined;
    try ks.applyConfig(try kernelOf(), cfg, &.{ h, routed, weights, shared, w, eps }, &o, s);
    return .{ .h = o[0], .normed = o[1] };
}
