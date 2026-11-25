//! Hy4-flash's shared-expert and dense-MLP SwiGLU in the up linear's finish (`linearSwigluMul`; the hy4f model plugin's
//! `swiglu_fused`). The chain it replaces, at a prompt's rows (hy4f src/model.zig `mlp`, `swigluOut` at limit 0):
//!   gf = astype(linear(x, gate), f32)      uf = astype(linear(x, up), f32)       (no copy for an f32 x)
//!   h  = astype((gf * sigmoid(gf)) * uf, act)                                    MLX's Sigmoid and Multiply in f32
//!   down(h)
//! The gate stays the caller's `linear` (its y in x's dtype). `linearSwigluMul` runs up's inner product by `linear`'s own
//! launches at these rows (`Dense.innerOn`: the register path's GEMM at 17 to 32 rows, the dequantize and MLX's matmul
//! past them), then one finish launch: text_finish's loads, butterfly and value (up's y, rounded to x's dtype as its
//! store rounds it, then taken to f32), the gate's y at the same element taken to f32, MLX's Sigmoid of it (gate.zig's
//! gate_sig: unary_ops.h's statements), swigluOut's two f32 products in its order, stored in H_T (the down linear's input
//! dtype: MLX's astype). Every word is the chain's, at either activation dtype.
//! Per row of I outputs the finish reads z (f16) and the gate's y and writes h: 6 I bytes at bf16 (the chain: 54 I, up's
//! finish 4, the two f32 copies 12, the sigmoid 8, the two multiplies 24, the cast 6), 10 I at f32 (the chain: 38 I, up's
//! finish 6, the sigmoid 8, the multiplies 24).
//! With pg2 on at a tabled shape where MLX runs its regular kernel, up's product and this finish are one launch
//! (`text_pg2_swiglu`, gate.zig's `pg2Text` and `pg2Of`): the same words, z never in device memory.
//! Rows `linear` serves another way refuse with error.Exl3EngineNotImplemented before any launch (the caller keeps the
//! chain): the decode rows (<= 16, the GEMVs and DEC-ONE), the fused-GEMM option, trunk int8 and an un-rotated
//! resident trellis.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const pgemm = @import("../../core/pgemm.zig");
const root = @import("../../root.zig");
const gate = @import("gate.zig");
const Proj = root.Proj;

fn freeArr(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

// ── texts ──

/// up's store loop in two pieces around gate.zig's MLX Sigmoid (`gate.gate_sig`, of the gate's f32 value yv): its
/// opening with up's value kept (text_finish's store expression, uv) and the gate's y at the same element read, then
/// swigluOut's two products stored where text_finish stores y
pub const swiglu_open = @embedFile("kernels/swiglu/swiglu_open.metal");
pub const swiglu_close = @embedFile("kernels/swiglu/swiglu_close.metal");

/// up's finish with the SwiGLU: h H_T [rows, DIM] = H_T((g * sigmoid(g)) * f32(OUT_T(text_finish's value))), g the gate's
/// y [rows, DIM] (OUT_T) in f32; z half [rows, DIM], svh half [DIM]; one simdgroup per (128-block, row)
pub const text_finish_swiglu: [:0]const u8 = dense.finish_v ++ swiglu_open ++ gate.gate_sig ++ swiglu_close;

const swiglu_in = [_][*:0]const u8{ "z", "svh", "gate" };
const swiglu_out = [_][*:0]const u8{"h"};

/// up's product and this finish in one launch at a pg2 tile (gate.zig `pg2Text`): xp half [M, K], w half [K, N], svh,
/// gate as above
pub const text_pg2_swiglu: [:0]const u8 = gate.pg2Text(swiglu_open ++ gate.gate_sig ++ swiglu_close);

const pg2_swiglu_in = [_][*:0]const u8{ "x", "w", "svh", "gate" };

// ── kernels: made at their first call ──

const kernel_lock = Global(std.atomic.Value(u32), "hy4f_swiglu_kernel_lock", std.atomic.Value(u32).init(0));
const kernel = Global(?mlx.mlx_fast_metal_kernel, "hy4f_swiglu_kernel", null);
const kernel_pg2 = Global(?mlx.mlx_fast_metal_kernel, "hy4f_swiglu_kernel_pg2", null);

/// Test seam: the fault every launch carries in place of the served text (`plant`; null: the served one)
pub const planted = Global(?Fault, "hy4f_swiglu_planted", null);
const fault_kernels = Global([2][fault_count]?mlx.mlx_fast_metal_kernel, "hy4f_swiglu_fault_kernels_kinds", @splat(@splat(null)));

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernel, kernel_pg2, planted, fault_kernels };

fn acquire() void {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}

fn release() void {
    kernel_lock.get().*.store(0, .release);
}

/// The red checks' planted faults (`plant`), each the served text with one piece swapped: the gate and up swapped in
/// the opening (h = silu(up) * gate), the sigmoid's exp on the fast path (gate.zig's fault piece)
pub const Fault = enum(u8) { swapped, sig_fast_exp };
const fault_count = @typeInfo(Fault).@"enum".field_names.len;

const fault_open_swapped = @embedFile("kernels/test/fault/swiglu_open_swapped.metal");
const fault_sig_fast_exp = @embedFile("kernels/test/fault/gate_sig_fast_exp.metal");
const text_fault_swapped: [:0]const u8 = dense.finish_v ++ fault_open_swapped ++ gate.gate_sig ++ swiglu_close;
const text_fault_fast_exp: [:0]const u8 = dense.finish_v ++ swiglu_open ++ fault_sig_fast_exp ++ swiglu_close;
const text_pg2_fault_swapped: [:0]const u8 = gate.pg2Text(fault_open_swapped ++ gate.gate_sig ++ swiglu_close);
const text_pg2_fault_fast_exp: [:0]const u8 = gate.pg2Text(swiglu_open ++ fault_sig_fast_exp ++ swiglu_close);

fn makeFault(kind: gate.Kind, f: Fault) ?mlx.mlx_fast_metal_kernel {
    return switch (kind) {
        .finish => switch (f) {
            .swapped => ks.newKernel("turboexl3_test_hy4f_finish_swiglu_swapped_fault", &swiglu_in, &swiglu_out, text_fault_swapped, decode.header, true),
            .sig_fast_exp => ks.newKernel("turboexl3_test_hy4f_finish_swiglu_fast_exp_fault", &swiglu_in, &swiglu_out, text_fault_fast_exp, decode.header, true),
        },
        .pg2 => switch (f) {
            .swapped => ks.newKernel("turboexl3_test_hy4f_pg2_swiglu_swapped_fault", &pg2_swiglu_in, &swiglu_out, text_pg2_fault_swapped, pgemm.header, true),
            .sig_fast_exp => ks.newKernel("turboexl3_test_hy4f_pg2_swiglu_fast_exp_fault", &pg2_swiglu_in, &swiglu_out, text_pg2_fault_fast_exp, pgemm.header, true),
        },
    };
}

/// `kind`'s kernel, the planted fault's while one is planted; each made at its first call
fn kernelGet(kind: gate.Kind) !mlx.mlx_fast_metal_kernel {
    acquire();
    defer release();
    if (planted.get().*) |f| {
        const slot = &fault_kernels.get().*[@intFromEnum(kind)][@intFromEnum(f)];
        if (slot.*) |made| return made;
        const made = makeFault(kind, f) orelse return error.KernelCreateFailed;
        slot.* = made;
        return made;
    }
    const slot = if (kind == .pg2) kernel_pg2.get() else kernel.get();
    if (slot.*) |made| return made;
    const made = (if (kind == .pg2)
        ks.newKernel("turboexl3_hy4f_pg2_swiglu", &pg2_swiglu_in, &swiglu_out, text_pg2_swiglu, pgemm.header, true)
    else
        ks.newKernel("turboexl3_hy4f_finish_swiglu", &swiglu_in, &swiglu_out, text_finish_swiglu, decode.header, true)) orelse return error.KernelCreateFailed;
    slot.* = made;
    return made;
}

/// Test seam: fault `f` in every launch's text (the finish's and pg2's) until `unplant` (a red check: the words must
/// move). Each fault's kernel is made at its first call.
pub fn plant(f: Fault) !void {
    acquire();
    defer release();
    planted.get().* = f;
}

/// Test seam: the served kernels again
pub fn unplant() void {
    acquire();
    defer release();
    planted.get().* = null;
}

// ── the call ──

/// h [..., I] in `h_dtype` (bf16, f16 or f32: the down linear's input dtype) for x [..., in] (bf16, f16 or f32) and
/// gate_y [..., I] in x's dtype (the caller's `linear(x, gate)`): MLX's astype to `h_dtype` of (g * sigmoid(g)) *
/// f32(`linear(x, up)`), g = f32(gate_y), word for word (swigluOut at limit 0). The checks below and the refusals
/// (error.Exl3EngineNotImplemented: decode rows, and where `gate.takes` does not hold) come before any launch, the shape
/// and dtype checks before any engine is made.
pub fn linearSwigluMul(s: mlx.mlx_stream, x: mlx.mlx_array, up: Proj, gate_y: mlx.mlx_array, dec: format.Decode, h_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8) return error.BadExl3Shape;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    if (h_dtype != .bfloat16 and h_dtype != .float32 and h_dtype != .float16) return error.BadExl3Dtype;
    // the gate's y as `linear` gives it: in x's dtype
    if (mlx.mlx_array_dtype(gate_y) != xd) return error.BadExl3Dtype;
    var rows: usize = 1;
    for (xsh[0 .. xsh.len - 1]) |n| rows *= @as(usize, @intCast(n));
    if (rows < 1) return error.BadExl3Shape;
    // the decode rows: `linear`'s GEMVs, not this finish
    if (rows <= decode.Kernels.max_rows) return error.Exl3EngineNotImplemented;
    const tsh = mlx.getShape(up.trellis);
    if (tsh.len != 3) return error.BadExl3Shape;
    const out_dim: usize = @intCast(tsh[1] * 16);
    const gsh = mlx.getShape(gate_y);
    if (gsh.len < 1 or gsh[gsh.len - 1] != @as(c_int, @intCast(out_dim)) or mlx.mlx_array_size(gate_y) != rows * out_dim) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const d = try root.denseOf(up, in_dim, xd, dec);
    if (!gate.takes(d, rows, up.trellis)) return error.Exl3EngineNotImplemented;
    const v = try root.rowsOf(x, in_dim, s);
    defer freeArr(v.x2);
    const r: c_int = @intCast(rows);
    const OUT: c_int = @intCast(out_dim);
    var g2 = mlx.mlx_array_new();
    defer freeArr(g2);
    try mlx.check(mlx.mlx_reshape(&g2, gate_y, &[_]c_int{ r, OUT }, 2, s));
    const tmpl = [_]ks.TemplateArg{
        .{ .name = "DIM", .value = .{ .int = OUT } },
        .{ .name = "OUT_T", .value = .{ .dtype = xd } },
        .{ .name = "H_T", .value = .{ .dtype = h_dtype } },
    };
    // pg2 at its shapes: up's product and this finish in one launch
    if (try gate.pg2Of(d, s, v.x2, up.trellis, up.suh, rows)) |pg| {
        defer freeArr(pg.xp);
        defer freeArr(pg.w);
        const y2 = try pgemm.launchWith(s, try kernelGet(.pg2), &.{ pg.xp, pg.w, up.svh, g2 }, h_dtype, pg.plan, &tmpl);
        defer freeArr(y2);
        return gate.shapedAs(s, y2, xsh, OUT);
    }
    const k = try kernelGet(.finish);
    // up's inner product by `linear`'s own launches at these rows, then its finish's grid with the SwiGLU's text
    const z = try d.innerOn(s, s, v.x2, up.trellis, up.suh, rows);
    defer freeArr(z);
    const cfg = try ks.buildConfig(.{
        .grid = .{ @as(u32, @intCast(out_dim / 128)) * 32, @intCast(rows), 1 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{&[_]c_int{ r, OUT }},
        .out_dtypes = &.{h_dtype},
        .template = &tmpl,
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(k, cfg, &.{ z, up.svh, g2 }, &o, s);
    defer freeArr(o[0]);
    return gate.shapedAs(s, o[0], xsh, OUT);
}
