//! Ling-3.1-flash's prefill SwiGLU epilogue (`upSwiglu`), on turboexl3's dense linear (`core/dense.zig`): at a prompt's
//! rows the up projection's finish runs the host's SwiGLU of the gate's finished value and its own, and stores the
//! activation h where the composed chain stores up and then runs the SwiGLU in launches of its own (mlx-serve's
//! fusedSwiGLU, or the plugin's swigluClampOut chain). The words are the chain's:
//! - the gate is the caller's `linear` of the gate projection;
//! - up's inner product is `linear`'s own launches at these rows (`Dense.innerOn`: the register path's GEMM at 17 to
//!   32 rows, the dequantize and MLX's matmul past them), and its value is text_finish's, statement for statement;
//! - h is the host's SwiGLU of the two, as the shared expert's decode launch runs it (`shared.sgu_swiglu`: limit 0,
//!   mlx-serve's fusedSwiGLU in the activation dtype over the host's table; `shared.sgu_swiglu_clamp_out`: limit L > 0,
//!   the plugin's swigluClampOut in f32 over MLX's f32 sigmoid of every 16-bit pattern).
//! Users: the outside plugin turboexl3_ling's MLX_SERVE_LING_SWIGLU_EPI, the shared expert (4096 -> 1280) and the dense
//! layers' MLP (4096 -> 10240) at prompt rows. Rows `linear` serves another way (the decode GEMVs up to 16 rows, the
//! fused-GEMM option, pg2, an un-rotated resident trellis) return null before any launch: the caller composes the chain.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const pgemm = @import("../../core/pgemm.zig");
const root = @import("../../root.zig");
const shared = @import("shared.zig");
const Proj = root.Proj;

fn freeArr(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

// ── texts ──

/// up's store loop: its opening with up's value kept (text_finish's store expression) and the gate's value at the same
/// element read, then its close with h stored where text_finish stores up
pub const epi_open = @embedFile("kernels/swiglu/epi_open.metal");
pub const epi_close = @embedFile("kernels/swiglu/epi_close.metal");

/// up's finish with the host's unclamped SwiGLU (fusedSwiGLU's statements): y T [rows, DIM] = h
pub const text_finish_sgu: [:0]const u8 = dense.finish_v ++ epi_open ++ shared.sgu_swiglu ++ epi_close;
/// up's finish with the host's SwiGLU clamped after the SiLU at `lim` (swigluClampOut's statements)
pub const text_finish_sguco: [:0]const u8 = dense.finish_v ++ epi_open ++ shared.sgu_swiglu_clamp_out ++ epi_close;

const sgu_in = [_][*:0]const u8{ "z", "svh", "gate", "sigtab" };
const sguco_in = [_][*:0]const u8{ "z", "svh", "gate", "sig32", "lim" };
const epi_out = [_][*:0]const u8{"y"};

// ── kernels: made at their first call ──

const Variant = enum(u8) { plain, clamp_out };
const kernel_lock = Global(std.atomic.Value(u32), "ling_swiglu_kernel_lock", std.atomic.Value(u32).init(0));
const kernels = Global([2]?mlx.mlx_fast_metal_kernel, "ling_swiglu_kernels", @splat(null));

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernels };

fn kernelOf(v: Variant) !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    const slot = &kernels.get().*[@intFromEnum(v)];
    if (slot.*) |made| return made;
    const made = switch (v) {
        .plain => ks.newKernel("turboexl3_ling_finish_sgu", &sgu_in, &epi_out, text_finish_sgu, decode.header, true),
        .clamp_out => ks.newKernel("turboexl3_ling_finish_sguco", &sguco_in, &epi_out, text_finish_sguco, decode.header, true),
    } orelse return error.KernelCreateFailed;
    slot.* = made;
    return made;
}

fn lim1(limit: u32) !mlx.mlx_array {
    const v: f32 = @floatFromInt(limit);
    const a = mlx.mlx_array_new_data(&v, &[_]c_int{1}, 1, .float32);
    if (a.ctx == null) return error.MlxError;
    return a;
}

// ── the call ──

/// Whether `upSwiglu` serves up's call at `rows` rows of x in `xd` on `d` (no launch): `linear`'s register path or its
/// matmul path, the latter with pg2 off and no un-rotated resident W_std for the trellis.
pub fn takes(d: *const dense.Dense, rows: usize, xd: mlx.mlx_dtype, trellis: mlx.mlx_array) bool {
    if (xd != .bfloat16 and xd != .float16) return false;
    switch (d.pathFor(rows)) {
        .small => return true,
        .matmul => {
            if (pgemm.mode.get().* != .off) return false;
            if (rows > dense.unrot_min_rows and dense.residentUnrot(trellis)) return false;
            return true;
        },
        .gemv, .fused => return false,
    }
}

/// h [..., out] in x's dtype for x [..., in] (bf16 or f16): the host's SwiGLU of `gate` (x's dtype, [..., out]: the
/// caller's `linear` of the gate projection of x) and `linear(x, up)` at `limit` (0 none, 1..128 Ling's clamp after the
/// SiLU), up's finish running the SwiGLU; `tab` the host's table at that limit (`shared.sigmoidTable`: [65536] in x's
/// dtype at limit 0, f32 above). Null where `takes` does not hold (the caller composes the chain); error.InvalidExl3Clamp
/// above 128 and `linear`'s shape and dtype errors, before any launch.
pub fn upSwiglu(s: mlx.mlx_stream, x: mlx.mlx_array, up: Proj, gate: mlx.mlx_array, limit: u32, tab: mlx.mlx_array, dec: format.Decode) !?mlx.mlx_array {
    if (limit > 128) return error.InvalidExl3Clamp;
    const xsh = mlx.getShape(x);
    if (xsh.len < 1) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    const d = try root.denseOf(up, in_dim, xd, dec);
    const out_dim = d.shape.out;
    const v = try root.rowsOf(x, in_dim, s);
    defer freeArr(v.x2);
    if (!takes(d, v.rows, xd, up.trellis)) return null;
    if (mlx.mlx_array_dtype(gate) != xd or mlx.mlx_array_size(gate) != v.rows * out_dim) return error.BadExl3Shape;
    const want_tab: mlx.mlx_dtype = if (limit == 0) xd else .float32;
    if (mlx.mlx_array_dtype(tab) != want_tab or mlx.mlx_array_size(tab) != 1 << 16) return error.BadExl3Dtype;
    const variant: Variant = if (limit == 0) .plain else .clamp_out;
    const k = try kernelOf(variant);
    const r: c_int = @intCast(v.rows);
    const OUT: c_int = @intCast(out_dim);
    var g2 = mlx.mlx_array_new();
    defer freeArr(g2);
    try mlx.check(mlx.mlx_reshape(&g2, gate, &[_]c_int{ r, OUT }, 2, s));
    // up's inner product by `linear`'s own launches at these rows, then its finish's config with the epilogue's text
    const z = try d.innerOn(s, s, v.x2, up.trellis, up.suh, v.rows);
    defer freeArr(z);
    const cfg = try ks.buildConfig(.{ .grid = .{ (out_dim / 128) * 32, @intCast(v.rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ r, OUT }}, .out_dtypes = &.{xd}, .template = &.{ .{ .name = "DIM", .value = .{ .int = OUT } }, .{ .name = "OUT_T", .value = .{ .dtype = xd } } } });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [1]mlx.mlx_array = undefined;
    if (variant == .plain) {
        try ks.applyConfig(k, cfg, &.{ z, up.svh, g2, tab }, &o, s);
    } else {
        const lim = try lim1(limit);
        defer freeArr(lim);
        try ks.applyConfig(k, cfg, &.{ z, up.svh, g2, tab, lim }, &o, s);
    }
    defer freeArr(o[0]);
    var osh: [8]c_int = undefined;
    if (xsh.len > osh.len) return error.BadExl3Shape;
    @memcpy(osh[0..xsh.len], xsh);
    osh[xsh.len - 1] = OUT;
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, o[0], osh[0..xsh.len].ptr, @intCast(xsh.len), s));
    return y;
}
