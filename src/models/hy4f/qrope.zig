//! Hy4-flash's q_b with its prompt rope in the finish (`linearQRope`; the hy4f model plugin's `qb_rope_fused`).
//! The chain it replaces at a prompt's rows (hy4f src/model.zig `attention`, `ropeGptj`):
//!   q    = transpose(reshape(astype(linear(x, q_b), f32), [1, S, H, HD]), 0 2 1 3)     (no copy for an f32 x)
//!   q_pe = mlx_fast_rope(q[..., HD - ROPE:], ROPE, traditional, base theta, scale 1, offset pos)   a roped copy
//! with q_nope = q[..., :HD - ROPE] a view. `linearQRope` runs q_b's inner product by `linear`'s own launches at these
//! rows (`Dense.innerOn`: the register path's GEMM at 17 to 32 rows, the dequantize and MLX's matmul past them), then one
//! finish launch: text_finish's loads, butterfly and value (q's y, rounded to x's dtype as its store rounds it), taken
//! to f32 (MLX's astype), and on each head's last ROPE dims MLX's rope of it with its traditional pair (the neighbouring
//! lane's value), MLX's statements at its float precision: inv_freq = precise exp2(-(i / (ROPE / 2)) log2(theta)) with
//! log2(theta) the host's f32 as MLX takes it (`std::log2` of a float), theta = (row + pos) inv_freq, the fast cosine
//! and sine, the rotation's two products and their sum or difference. q f32 [rows, H HD] in the layout the chain's q
//! has before its transpose, so the host's q_nope view is the chain's and its q_pe a view of the same words the rope
//! writes. Every word the chain's, at either activation dtype.
//! Per row of DIM = H HD outputs the finish reads z (f16) and writes q (f32): 6 DIM bytes (the chain at bf16: its finish
//! 4, the f32 copy 6, the rope's read and write of a quarter of q 2 at ROPE / HD = 1/4: 12 DIM; at f32 8 DIM).
//! With pg2 on at a tabled shape where MLX runs its regular kernel, q_b's product and this finish are one launch
//! (`text_pg2_qrope`, gate.zig's `pg2Text` and `pg2Of`): the same words, z never in device memory.
//! Rows `linear` serves another way refuse with error.Exl3EngineNotImplemented before any launch (the caller keeps the
//! chain): the decode rows (<= 16), the fused-GEMM option, trunk int8 and an un-rotated resident trellis (`gate.takes`).
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

/// q_b's store loop in four pieces: its opening with q's value kept in f32 (text_finish's store expression, qv) and
/// the element's dim in its head, the pair's other value (MLX's traditional pairs), MLX's rope statements, then the
/// store of the rotated value on the rope dims and q's elsewhere, in f32 where text_finish stores y
pub const qrope_open = @embedFile("kernels/qrope/qrope_open.metal");
pub const qrope_pair = @embedFile("kernels/qrope/qrope_pair.metal");
pub const qrope_rot = @embedFile("kernels/qrope/qrope_rot.metal");
pub const qrope_close = @embedFile("kernels/qrope/qrope_close.metal");

/// q_b's finish with the rope: q f32 [rows, DIM] = f32(OUT_T(text_finish's value)), each head's last ROPE dims roped at
/// position row + off[0] (MLX's fast::rope, traditional, base 2^as_float(LOG2B), scale 1); z half [rows, DIM], svh half
/// [DIM], off int32 [1]; one simdgroup per (128-block, row)
pub const text_finish_qrope: [:0]const u8 = dense.finish_v ++ qrope_open ++ qrope_pair ++ qrope_rot ++ qrope_close;

const qrope_in = [_][*:0]const u8{ "z", "svh", "off" };
const qrope_out = [_][*:0]const u8{"q"};

/// q_b's product and this finish in one launch at a pg2 tile (gate.zig `pg2Text`): xp half [M, K], w half [K, N], svh,
/// off as above
pub const text_pg2_qrope: [:0]const u8 = gate.pg2Text(qrope_open ++ qrope_pair ++ qrope_rot ++ qrope_close);

const pg2_qrope_in = [_][*:0]const u8{ "x", "w", "svh", "off" };

// ── kernels: made at their first call ──

/// The red checks' planted faults (`plant`), each the served text with one piece swapped: MLX's non-traditional (NeoX)
/// pairs (dims i and i + ROPE / 2; at ROPE 64 only) and the frequency by MLX's CPU fallback formula (exp of the natural
/// log, a near match whose angles feed the same cosine, sine and rotation statements)
pub const Fault = enum(u8) { pair_neox, freq_ln };
const fault_count = @typeInfo(Fault).@"enum".field_names.len;

const kernel_lock = Global(std.atomic.Value(u32), "hy4f_qrope_kernel_lock", std.atomic.Value(u32).init(0));
const kernel = Global(?mlx.mlx_fast_metal_kernel, "hy4f_qrope_kernel", null);
const kernel_pg2 = Global(?mlx.mlx_fast_metal_kernel, "hy4f_qrope_kernel_pg2", null);

/// Test seam: the fault the finish carries in place of the served text (`plant`; null: the served one)
pub const planted = Global(?Fault, "hy4f_qrope_planted", null);
const fault_kernels = Global([2][fault_count]?mlx.mlx_fast_metal_kernel, "hy4f_qrope_fault_kernels_kinds", @splat(@splat(null)));

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernel, kernel_pg2, planted, fault_kernels };

fn acquire() void {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}

fn release() void {
    kernel_lock.get().*.store(0, .release);
}

const fault_pair_neox = @embedFile("kernels/test/fault/qrope_pair_neox.metal");
const fault_rot_freq_ln = @embedFile("kernels/test/fault/qrope_rot_freq_ln.metal");
const text_fault_pair_neox: [:0]const u8 = dense.finish_v ++ qrope_open ++ fault_pair_neox ++ qrope_rot ++ qrope_close;
const text_fault_freq_ln: [:0]const u8 = dense.finish_v ++ qrope_open ++ qrope_pair ++ fault_rot_freq_ln ++ qrope_close;
const text_pg2_fault_pair_neox: [:0]const u8 = gate.pg2Text(qrope_open ++ fault_pair_neox ++ qrope_rot ++ qrope_close);
const text_pg2_fault_freq_ln: [:0]const u8 = gate.pg2Text(qrope_open ++ qrope_pair ++ fault_rot_freq_ln ++ qrope_close);

fn makeFault(kind: gate.Kind, f: Fault) ?mlx.mlx_fast_metal_kernel {
    return switch (kind) {
        .finish => switch (f) {
            .pair_neox => ks.newKernel("turboexl3_test_hy4f_finish_qrope_pair_neox_fault", &qrope_in, &qrope_out, text_fault_pair_neox, decode.header, true),
            .freq_ln => ks.newKernel("turboexl3_test_hy4f_finish_qrope_freq_ln_fault", &qrope_in, &qrope_out, text_fault_freq_ln, decode.header, true),
        },
        .pg2 => switch (f) {
            .pair_neox => ks.newKernel("turboexl3_test_hy4f_pg2_qrope_pair_neox_fault", &pg2_qrope_in, &qrope_out, text_pg2_fault_pair_neox, pgemm.header, true),
            .freq_ln => ks.newKernel("turboexl3_test_hy4f_pg2_qrope_freq_ln_fault", &pg2_qrope_in, &qrope_out, text_pg2_fault_freq_ln, pgemm.header, true),
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
        ks.newKernel("turboexl3_hy4f_pg2_qrope", &pg2_qrope_in, &qrope_out, text_pg2_qrope, pgemm.header, true)
    else
        ks.newKernel("turboexl3_hy4f_finish_qrope", &qrope_in, &qrope_out, text_finish_qrope, decode.header, true)) orelse return error.KernelCreateFailed;
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

/// Test seam: the served kernel again
pub fn unplant() void {
    acquire();
    defer release();
    planted.get().* = null;
}

// ── the call ──

/// log2(theta) as MLX's rope takes its base (mlx/backend/metal/rope.cpp: `float base = std::log2(base_)`, base_ a
/// float): the f32 nearest log2 of the f32 theta, from its f64 log2 (29 bits past f32's, so the f32 rounding is MLX's
/// libm's wherever that is correctly rounded; the GPU tests hold the words at Hy4's thetas and the fixture's)
pub fn log2Base(theta: f32) f32 {
    return @floatCast(@log2(@as(f64, theta)));
}

/// q [..., H HD] f32 for x [..., in] (bf16, f16 or f32) and q_b's projection: f32(`linear(x, q_b)`) with each head's
/// last `rope_dims` dims roped as MLX's `fast::rope` ropes them (traditional pairs, base `theta`, scale 1, offset `pos`
/// for the first row, row r at `pos + r`), word for word. `head_dim` divides q_b's outputs, `rope_dims` even and at most
/// `head_dim`, `head_dim - rope_dims` even. The checks below and the refusals (error.Exl3EngineNotImplemented: decode rows
/// and where `gate.takes` does not hold) come before any launch, the shape and dtype checks and the decode rows before
/// any engine is made.
pub fn linearQRope(s: mlx.mlx_stream, x: mlx.mlx_array, q_b: Proj, theta: f32, pos: i32, head_dim: u32, rope_dims: u32, dec: format.Decode) !mlx.mlx_array {
    return qropeOf(s, x, mlx.mlx_array_dtype(x), false, q_b, theta, pos, head_dim, rope_dims, dec);
}

/// `linearQRope` from rows a launch before prepared: xp f16 [..., in] in natural order with `linear`'s prepare
/// arithmetic of an x of dtype `xd` (rms.zig's `linearRmsPrep`: q_a's finish with the q norm), q_b's inner product by
/// `linear`'s own launches from there (`Dense.innerOf`) and the same finish at `xd`: the words `linearQRope` gives on that
/// x. The same checks and refusals, before any launch.
pub fn linearQRopePrepared(s: mlx.mlx_stream, xp: mlx.mlx_array, xd: mlx.mlx_dtype, q_b: Proj, theta: f32, pos: i32, head_dim: u32, rope_dims: u32, dec: format.Decode) !mlx.mlx_array {
    return qropeOf(s, xp, xd, true, q_b, theta, pos, head_dim, rope_dims, dec);
}

/// Whether `linearQRopePrepared` at `rows` rows of q_b's prepared rows (at `xd`) runs pg2's launch (no launch: the shape
/// checks `linearQRopePrepared` makes, `gate.takes`, then `gate.pg2PlanPrepared`): the host's switch for the q norm's
/// finish (rms.zig's linearRmsPrep), which runs q_a's finish with q_b's prepare only where q_b keeps pg2's launch
/// (`qnorm_fused`). Shape errors as `linearQRopePrepared`'s; false for the decode rows.
pub fn preparedPg2(q_b: Proj, rows: usize, xd: mlx.mlx_dtype, dec: format.Decode) !bool {
    if (rows <= decode.Kernels.max_rows) return false;
    const tsh = mlx.getShape(q_b.trellis);
    if (tsh.len != 3) return error.BadExl3Shape;
    const d = try root.denseOf(q_b, @intCast(tsh[0] * 16), xd, dec);
    if (!gate.takes(d, rows, q_b.trellis)) return false;
    return gate.pg2PlanPrepared(d, rows, q_b.trellis) != null;
}

/// The two calls' body: x (`prepared` false: its own dtype `xd`) or prepared rows xp f16 (`prepared`: the finish at `xd`)
fn qropeOf(s: mlx.mlx_stream, x: mlx.mlx_array, xd: mlx.mlx_dtype, prepared: bool, q_b: Proj, theta: f32, pos: i32, head_dim: u32, rope_dims: u32, dec: format.Decode) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8) return error.BadExl3Shape;
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    if (prepared and mlx.mlx_array_dtype(x) != .float16) return error.BadExl3Dtype;
    var rows: usize = 1;
    for (xsh[0 .. xsh.len - 1]) |n| rows *= @as(usize, @intCast(n));
    if (rows < 1) return error.BadExl3Shape;
    // the decode rows: `linear`'s GEMVs, not this finish
    if (rows <= decode.Kernels.max_rows) return error.Exl3EngineNotImplemented;
    if (!(theta > 0) or !std.math.isFinite(theta) or pos < 0) return error.BadExl3Shape;
    const tsh = mlx.getShape(q_b.trellis);
    if (tsh.len != 3) return error.BadExl3Shape;
    const out_dim: usize = @intCast(tsh[1] * 16);
    if (head_dim == 0 or out_dim % head_dim != 0 or rope_dims == 0 or rope_dims % 2 != 0 or rope_dims > head_dim or (head_dim - rope_dims) % 2 != 0) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const d = try root.denseOf(q_b, in_dim, xd, dec);
    if (!gate.takes(d, rows, q_b.trellis)) return error.Exl3EngineNotImplemented;
    const v = try root.rowsOf(x, in_dim, s);
    defer freeArr(v.x2);
    // the row's position offset as MLX's rope reads it (an int32 it adds to the row)
    const off_host = [_]i32{pos};
    const off = mlx.mlx_array_new_data(&off_host, &[_]c_int{1}, 1, .int32);
    if (off.ctx == null) return error.MlxError;
    defer freeArr(off);
    const r: c_int = @intCast(rows);
    const OUT: c_int = @intCast(out_dim);
    const tmpl = [_]ks.TemplateArg{
        .{ .name = "DIM", .value = .{ .int = OUT } },
        .{ .name = "HD", .value = .{ .int = @intCast(head_dim) } },
        .{ .name = "ROPE", .value = .{ .int = @intCast(rope_dims) } },
        .{ .name = "LOG2B", .value = .{ .int = @bitCast(log2Base(theta)) } },
        .{ .name = "OUT_T", .value = .{ .dtype = xd } },
    };
    // pg2 at its shapes: q_b's product and this finish in one launch
    // pg2 from prepared rows: the caller's xp and W_hat by `linear`'s own dequantize (`gate.pg2OfPrepared`)
    if (prepared) if (try gate.pg2OfPrepared(d, s, q_b.trellis, rows)) |pg| {
        defer freeArr(pg.w);
        const y2 = try pgemm.launchWith(s, try kernelGet(.pg2), &.{ v.x2, pg.w, q_b.svh, off }, .float32, pg.plan, &tmpl);
        defer freeArr(y2);
        return gate.shapedAs(s, y2, xsh, OUT);
    };
    // from x: pg2's operands by `linear`'s own prepare
    if (!prepared) if (try gate.pg2Of(d, s, v.x2, q_b.trellis, q_b.suh, rows)) |pg| {
        defer freeArr(pg.xp);
        defer freeArr(pg.w);
        const y2 = try pgemm.launchWith(s, try kernelGet(.pg2), &.{ pg.xp, pg.w, q_b.svh, off }, .float32, pg.plan, &tmpl);
        defer freeArr(y2);
        return gate.shapedAs(s, y2, xsh, OUT);
    };
    const k = try kernelGet(.finish);
    // q_b's inner product by `linear`'s own launches at these rows (from x, or from the prepared rows on `linear`'s path
    // there), then its finish's grid with the rope's text
    const z = if (prepared) try d.innerOf(s, v.x2, q_b.trellis, rows, d.pathFor(rows)) else try d.innerOn(s, s, v.x2, q_b.trellis, q_b.suh, rows);
    defer freeArr(z);
    const cfg = try ks.buildConfig(.{
        .grid = .{ @as(u32, @intCast(out_dim / 128)) * 32, @intCast(rows), 1 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{&[_]c_int{ r, OUT }},
        .out_dtypes = &.{.float32},
        .template = &tmpl,
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(k, cfg, &.{ z, q_b.svh, off }, &o, s);
    defer freeArr(o[0]);
    return gate.shapedAs(s, o[0], xsh, OUT);
}
