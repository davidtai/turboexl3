//! Hy4-flash's attention output gate in its linear's finish (`linearGateMul`; the hy4f model plugin's `gate_fused`). The
//! chain it replaces, at a prompt's rows (hy4f src/model.zig `attention`):
//!   flat  = reshape(transpose(o, 0 2 1 3), [1, S, H VH])       o f32 [1, H, S, VH]: a transpose copy (none where o is
//!                                                              the transposed view of rows [1, S, H, VH])
//!   gated = flat * sigmoid(astype(linear(x, gate), f32))        MLX's Sigmoid and Multiply in f32
//!   o_proj(astype(gated, act))
//! `linearGateMul` runs the gate's inner product by `linear`'s own launches at these rows (`Dense.innerOn`: the
//! register path's GEMM at 17 to 32 rows, the dequantize and MLX's matmul past them), then one finish launch:
//! text_finish's loads, butterfly and value (the gate's y, rounded to x's dtype as its store rounds it), MLX's Sigmoid
//! of y in f32 (unary_ops.h's statements), o read by index in its stored layout (no transpose copy) and one f32
//! product, stored in G_T (o_proj's input dtype: MLX's astype). o's layout is the caller's (`OLayout`, the template's
//! O_FLAT): [1, H, S, VH] as MLX's matmul writes the un-absorb, or the rows [1, S, H, VH] the host's kv_b kernel writes
//! (of which the host's o is the transposed view): read where it lies either way, so no launch copies it. Every word
//! is the chain's, at either activation dtype.
//! Per row of DIM = H VH outputs the finish reads z (f16) and o (f32) and writes the product in G_T: 8 DIM bytes at
//! bf16 (the chain: 44 DIM, its finish, the f32 copy, the sigmoid, the transpose copy, the multiply and the cast), 10 DIM
//! at f32 (the chain: 34 DIM).
//! With pg2 on (core/pgemm.zig, the host's switch) at a tabled shape where MLX runs its regular kernel, the gate's
//! product and this finish are one launch (`text_pg2_gate`): pg2's GEMM, its staged halves read by text_finish's
//! statements and this store loop in pg2's store loop's place, z never in device memory; the same words, since pg2's
//! GEMM gives MLX's z word for word and every statement after it is this finish's. Elsewhere with pg2 on, the call
//! runs the launches above.
//! Rows `linear` serves another way refuse with error.Exl3EngineNotImplemented before any launch (the caller keeps the
//! chain): the decode rows (<= 16, the GEMVs and DEC-ONE), the fused-GEMM option, trunk int8 and an un-rotated resident
//! trellis.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const pgemm = @import("../../core/pgemm.zig");
const trunk_q8 = @import("../../core/trunk_q8.zig");
const root = @import("../../root.zig");
const Proj = root.Proj;

fn freeArr(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

// ── texts ──

/// The gate's store loop in four pieces: its opening with the gate's value kept (text_finish's store expression), MLX's
/// Sigmoid of that value, o's element read in o's layout, then the product stored where text_finish stores y
pub const gate_open = gate_open_1 ++ gate_open_2 ++ gate_open_3;
/// gate_open in pieces: its constants, the rows (the finish grid's y; a pg2 launch takes them from its GEMM's M,
/// `gate_open_2_pg2`), then the loop's opening with the gate's value kept
pub const gate_open_1 = @embedFile("kernels/gate/gate_open_1.metal");
pub const gate_open_2 = @embedFile("kernels/gate/gate_open_2.metal");
pub const gate_open_2_pg2 = @embedFile("kernels/gate/gate_open_2_pg2.metal");
pub const gate_open_3 = @embedFile("kernels/gate/gate_open_3.metal");
pub const gate_sig = @embedFile("kernels/gate/gate_sig.metal");
pub const gate_src = @embedFile("kernels/gate/gate_src.metal");
pub const gate_close = @embedFile("kernels/gate/gate_close.metal");

/// The gate's finish: gated G_T [rows, DIM] = G_T(o[h, row, d] * sigmoid(f32(OUT_T(text_finish's value)))) at element
/// h VH + d; z half [rows, DIM], svh half [DIM], o_src f32 [1, H, rows, VH] (O_FLAT 0) or [1, rows, H, VH] (O_FLAT 1);
/// one simdgroup per (128-block, row)
pub const text_finish_gate: [:0]const u8 = dense.finish_v ++ gate_open ++ gate_sig ++ gate_src ++ gate_close;

const gate_in = [_][*:0]const u8{ "z", "svh", "o_src" };
const gate_out = [_][*:0]const u8{"gated"};

/// The gate's product and finish in one launch at a pg2 tile (pgemm.zig `launchWith`): pg2's GEMM to z's staged
/// halves and text_finish's statements over them (`pgemm.fused_1 ++ pgemm.fused_2_1`), this finish's store loop with
/// the rows from the GEMM's M, the row loop's close; xp half [M, K], w half [K, N], svh, o_src as above
pub fn pg2Text(comptime store: []const u8) [:0]const u8 {
    return pgemm.sentinel(pgemm.fused_1 ++ pgemm.fused_2_1 ++ store ++ pgemm.fused_2_2_2 ++ pgemm.undef);
}
pub const text_pg2_gate: [:0]const u8 = pg2Text(gate_open_1 ++ gate_open_2_pg2 ++ gate_open_3 ++ gate_sig ++ gate_src ++ gate_close);

const pg2_gate_in = [_][*:0]const u8{ "x", "w", "svh", "o_src" };

/// The prep finish (`linearGateMulPrep`; the hy4f model plugin's `gate_prep_fused`): the gate's finish writing o_proj's prepared rows
/// in place of `gated`. The store loop's value is gate_close's G_T(ov * sg); o_proj's prepare runs on it (dense
/// prep_2_1's expression with o_proj's suh `suh_o`, then prep_2_2's butterfly and prep_3's store at LANE_ORDER 0): xp half
/// [rows, DIM], the bytes o_proj's own prepare of `gated` writes. Lane L's value j is column b 128 + L + 32 j in the
/// gate's store loop and in o_proj's prepare alike (one simdgroup a (128-block, row) in either launch, the finish's grid
/// and pg2's store loop), so the block's butterfly runs on the values each lane holds.
pub const gate_prep_open = @embedFile("kernels/gate/gate_prep_open.metal");
pub const gate_close_prep = @embedFile("kernels/gate/gate_close_prep.metal");

/// The prep store loop with `sig`, `src` and `close` in their pieces' places (the planted faults' texts), its rows the
/// finish grid's (`open2` gate_open_2) or pg2's GEMM's M (gate_open_2_pg2)
fn prepStore(comptime open2: []const u8, comptime sig: []const u8, comptime src: []const u8, comptime close: []const u8) []const u8 {
    return gate_open_1 ++ open2 ++ gate_prep_open ++ gate_open_3 ++ sig ++ src ++ close ++ dense.prep_2_2 ++ dense.prep_3;
}

fn prepFinishText(comptime sig: []const u8, comptime src: []const u8, comptime close: []const u8) [:0]const u8 {
    return pgemm.sentinel(dense.finish_v ++ prepStore(gate_open_2, sig, src, close));
}

fn prepPg2Text(comptime sig: []const u8, comptime src: []const u8, comptime close: []const u8) [:0]const u8 {
    return pg2Text(prepStore(gate_open_2_pg2, sig, src, close));
}

/// xp half [rows, DIM] = o_proj's prepare of the gate's finish's G_T product; z, svh, o_src as text_finish_gate's, suh_o
/// half [DIM] (o_proj's suh)
pub const text_finish_gate_prep: [:0]const u8 = prepFinishText(gate_sig, gate_src, gate_close_prep);
/// The same in pg2's launch (text_pg2_gate's GEMM and store loop)
pub const text_pg2_gate_prep: [:0]const u8 = prepPg2Text(gate_sig, gate_src, gate_close_prep);

const gate_prep_in = [_][*:0]const u8{ "z", "svh", "o_src", "suh_o" };
const pg2_gate_prep_in = [_][*:0]const u8{ "x", "w", "svh", "o_src", "suh_o" };
const prep_out = [_][*:0]const u8{"xp"};

// ── kernels: made at their first call ──

const kernel_lock = Global(std.atomic.Value(u32), "hy4f_gate_kernel_lock", std.atomic.Value(u32).init(0));
const kernel = Global(?mlx.mlx_fast_metal_kernel, "hy4f_gate_kernel", null);
const kernel_pg2 = Global(?mlx.mlx_fast_metal_kernel, "hy4f_gate_kernel_pg2", null);
/// The prep finish's two launches: the finish's and pg2's
const kernels_prep = Global([2]?mlx.mlx_fast_metal_kernel, "hy4f_gate_kernels_prep", @splat(null));

/// Test seam: the fault every launch carries in place of the served text (`plant`; null: the served one)
pub const planted = Global(?Fault, "hy4f_gate_planted", null);
const fault_kernels = Global([launch_count][fault_count]?mlx.mlx_fast_metal_kernel, "hy4f_gate_fault_kernels_kinds", @splat(@splat(null)));

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernel, kernel_pg2, kernels_prep, planted, fault_kernels };

fn acquire() void {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}

fn release() void {
    kernel_lock.get().*.store(0, .release);
}

/// The launch a call makes: the finish over `linear`'s z, or pg2's GEMM with this finish (`text_pg2_gate`)
pub const Kind = enum(u1) { finish, pg2 };

/// This file's launches: `Kind`'s, each storing `gated`, and the prep finish's, storing o_proj's prepared rows (`_prep`). Its own:
/// qrope.zig and swiglu.zig name their launches by `Kind`.
const Launch = enum(u2) { finish, pg2, finish_prep, pg2_prep };
const launch_count = @typeInfo(Launch).@"enum".field_names.len;

/// The red checks' planted faults (`plant`), each the served text with one piece swapped: o read in the other layout
/// than the call's (`o_layout`: the rows' order on a heads-layout o, the heads' on a rows-layout one), the sigmoid's exp
/// on the fast path, the one-sided sigmoid 1 / (1 + exp(-y)); and in the prep finish's launches alone, o_proj's prepare without its
/// suh (`prep_no_suh`)
pub const Fault = enum(u8) { o_layout, sig_fast_exp, sig_one_sided, prep_no_suh };
const fault_count = @typeInfo(Fault).@"enum".field_names.len;

const fault_src_swapped = @embedFile("kernels/test/fault/gate_src_swapped.metal");
const fault_sig_fast_exp = @embedFile("kernels/test/fault/gate_sig_fast_exp.metal");
const fault_sig_one_sided = @embedFile("kernels/test/fault/gate_sig_one_sided.metal");
const text_fault_o_layout: [:0]const u8 = dense.finish_v ++ gate_open ++ gate_sig ++ fault_src_swapped ++ gate_close;
const text_fault_fast_exp: [:0]const u8 = dense.finish_v ++ gate_open ++ fault_sig_fast_exp ++ gate_src ++ gate_close;
const text_fault_one_sided: [:0]const u8 = dense.finish_v ++ gate_open ++ fault_sig_one_sided ++ gate_src ++ gate_close;
const pg2_open = gate_open_1 ++ gate_open_2_pg2 ++ gate_open_3;
const text_pg2_fault_o_layout: [:0]const u8 = pg2Text(pg2_open ++ gate_sig ++ fault_src_swapped ++ gate_close);
const text_pg2_fault_fast_exp: [:0]const u8 = pg2Text(pg2_open ++ fault_sig_fast_exp ++ gate_src ++ gate_close);
const text_pg2_fault_one_sided: [:0]const u8 = pg2Text(pg2_open ++ fault_sig_one_sided ++ gate_src ++ gate_close);

const fault_close_prep_nosuh = @embedFile("kernels/test/fault/gate_close_prep_nosuh.metal");

/// The prep finish's launch text with fault `f` (pg2's launch or the finish's)
fn prepFaultText(comptime pg2: bool, comptime f: Fault) [:0]const u8 {
    const sig = switch (f) {
        .sig_fast_exp => fault_sig_fast_exp,
        .sig_one_sided => fault_sig_one_sided,
        else => gate_sig,
    };
    const src = if (f == .o_layout) fault_src_swapped else gate_src;
    const close = if (f == .prep_no_suh) fault_close_prep_nosuh else gate_close_prep;
    return if (pg2) prepPg2Text(sig, src, close) else prepFinishText(sig, src, close);
}

fn makePrepFault(pg2: bool, f: Fault) ?mlx.mlx_fast_metal_kernel {
    return switch (f) {
        inline else => |cf| if (pg2)
            ks.newKernel("turboexl3_test_hy4f_pg2_gate_prep_" ++ @tagName(cf) ++ "_fault", &pg2_gate_prep_in, &prep_out, comptime prepFaultText(true, cf), pgemm.header, true)
        else
            ks.newKernel("turboexl3_test_hy4f_finish_gate_prep_" ++ @tagName(cf) ++ "_fault", &gate_prep_in, &prep_out, comptime prepFaultText(false, cf), decode.header, true),
    };
}

fn makeFault(kind: Launch, f: Fault) ?mlx.mlx_fast_metal_kernel {
    return switch (kind) {
        .finish_prep => makePrepFault(false, f),
        .pg2_prep => makePrepFault(true, f),
        .finish => switch (f) {
            .o_layout => ks.newKernel("turboexl3_test_hy4f_finish_gate_o_layout_fault", &gate_in, &gate_out, text_fault_o_layout, decode.header, true),
            .sig_fast_exp => ks.newKernel("turboexl3_test_hy4f_finish_gate_fast_exp_fault", &gate_in, &gate_out, text_fault_fast_exp, decode.header, true),
            .sig_one_sided => ks.newKernel("turboexl3_test_hy4f_finish_gate_one_sided_fault", &gate_in, &gate_out, text_fault_one_sided, decode.header, true),
            .prep_no_suh => null,
        },
        .pg2 => switch (f) {
            .o_layout => ks.newKernel("turboexl3_test_hy4f_pg2_gate_o_layout_fault", &pg2_gate_in, &gate_out, text_pg2_fault_o_layout, pgemm.header, true),
            .sig_fast_exp => ks.newKernel("turboexl3_test_hy4f_pg2_gate_fast_exp_fault", &pg2_gate_in, &gate_out, text_pg2_fault_fast_exp, pgemm.header, true),
            .sig_one_sided => ks.newKernel("turboexl3_test_hy4f_pg2_gate_one_sided_fault", &pg2_gate_in, &gate_out, text_pg2_fault_one_sided, pgemm.header, true),
            .prep_no_suh => null,
        },
    };
}

/// `kind`'s kernel, the planted fault's while one is planted; each made at its first call
fn kernelGet(kind: Launch) !mlx.mlx_fast_metal_kernel {
    acquire();
    defer release();
    if (planted.get().*) |f| {
        const slot = &fault_kernels.get().*[@intFromEnum(kind)][@intFromEnum(f)];
        if (slot.*) |made| return made;
        const made = makeFault(kind, f) orelse return error.KernelCreateFailed;
        slot.* = made;
        return made;
    }
    const slot = switch (kind) {
        .finish => kernel.get(),
        .pg2 => kernel_pg2.get(),
        .finish_prep => &kernels_prep.get().*[0],
        .pg2_prep => &kernels_prep.get().*[1],
    };
    if (slot.*) |made| return made;
    const made = (switch (kind) {
        .pg2 => ks.newKernel("turboexl3_hy4f_pg2_gate", &pg2_gate_in, &gate_out, text_pg2_gate, pgemm.header, true),
        .finish => ks.newKernel("turboexl3_hy4f_finish_gate", &gate_in, &gate_out, text_finish_gate, decode.header, true),
        .finish_prep => ks.newKernel("turboexl3_hy4f_finish_gate_prep", &gate_prep_in, &prep_out, text_finish_gate_prep, decode.header, true),
        .pg2_prep => ks.newKernel("turboexl3_hy4f_pg2_gate_prep", &pg2_gate_prep_in, &prep_out, text_pg2_gate_prep, pgemm.header, true),
    }) orelse return error.KernelCreateFailed;
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

/// Whether `linearGateMul` serves the gate's call (and swiglu.zig's `linearSwigluMul` up's, qrope.zig's
/// `linearQRope` q_b's) at `rows` rows on `d` (no launch): `linear`'s register path, or its matmul path with no trunk
/// int8 and no un-rotated resident W_std for the trellis. pg2 is no refusal: with it on, a call at a pg2 shape runs
/// pg2's GEMM with its finish (`pg2Of`), any other the launches without it.
pub fn takes(d: *const dense.Dense, rows: usize, trellis: mlx.mlx_array) bool {
    switch (d.pathFor(rows)) {
        .small => return true,
        .matmul => {
            if (trunk_q8.takes(d)) return false;
            if (rows > dense.unrot_min_rows and dense.residentUnrot(trellis)) return false;
            return true;
        },
        .gemv, .fused => return false,
    }
}

/// pg2's operands for an arm's call (this file's, swiglu.zig's, qrope.zig's) of x2 [rows, in] on `d`: with pg2 on, on
/// `linear`'s matmul path, where pg2 has a tile at (rows, in, out) and MLX runs its regular kernel (`pgemm.planAt`
/// without split K), the prepared rows xp and W_hat by `linear`'s own launches (`Dense.prefillOperands`) and the plan;
/// null elsewhere (the arm's launches without pg2). The caller frees xp and w.
pub const Pg2 = struct { xp: mlx.mlx_array, w: mlx.mlx_array, plan: pgemm.Plan };

pub fn pg2Of(d: *dense.Dense, s: mlx.mlx_stream, x2: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, rows: usize) !?Pg2 {
    if (pgemm.mode.get().* != .on or d.pathFor(rows) != .matmul) return null;
    const p = pgemm.planAt(@intCast(rows), d.shape.in, d.shape.out) orelse return null;
    if (p.sk != null) return null;
    const ops = (try d.prefillOperands(s, x2, trellis, suh, rows)) orelse return null;
    return .{ .xp = ops[0], .w = ops[1], .plan = p };
}

/// `pg2Of`'s plan for a call on rows a launch before prepared (no launch): pg2 on, `linear`'s matmul path, a tile at
/// (rows, in, out) where MLX runs its regular kernel, and none of `prefillOperands`' refusals (an un-rotated resident
/// trellis); null elsewhere
pub fn pg2PlanPrepared(d: *const dense.Dense, rows: usize, trellis: mlx.mlx_array) ?pgemm.Plan {
    if (pgemm.mode.get().* != .on or d.pathFor(rows) != .matmul) return null;
    if (rows > dense.unrot_min_rows and dense.residentUnrot(trellis)) return null;
    const p = pgemm.planAt(@intCast(rows), d.shape.in, d.shape.out) orelse return null;
    if (p.sk != null) return null;
    return p;
}

/// pg2's operands for an arm's call on rows a launch before prepared (xp half [rows, in], natural order: `linear`'s
/// prepare arithmetic, the caller's): W_hat by `linear`'s own dequantize and the plan where `pg2PlanPrepared` holds;
/// null elsewhere (the arm's launches without pg2). The caller frees w (xp stays the caller's).
pub const Pg2W = struct { w: mlx.mlx_array, plan: pgemm.Plan };

pub fn pg2OfPrepared(d: *dense.Dense, s: mlx.mlx_stream, trellis: mlx.mlx_array, rows: usize) !?Pg2W {
    const p = pg2PlanPrepared(d, rows, trellis) orelse return null;
    return .{ .w = try d.dequantize(s, trellis), .plan = p };
}

/// y [rows, out] as x's leading dims [..., out]; the caller owns the result (y stays the caller's)
pub fn shapedAs(s: mlx.mlx_stream, y: mlx.mlx_array, xsh: []const c_int, out: c_int) !mlx.mlx_array {
    var ysh: [8]c_int = undefined;
    @memcpy(ysh[0..xsh.len], xsh);
    ysh[xsh.len - 1] = out;
    var r = mlx.mlx_array_new();
    errdefer freeArr(r);
    try mlx.check(mlx.mlx_reshape(&r, y, ysh[0..xsh.len].ptr, @intCast(xsh.len), s));
    return r;
}

/// o's layout in a `linearGateMulAt` call (the template's O_FLAT): `heads` [1, H, rows, VH], the layout MLX's matmul
/// writes the un-absorb in; `flat` [1, rows, H, VH], the rows the host's kv_b kernel writes it in (hy4f's kvb_mix, of
/// which the host's o is the transposed view): element h VH + d of a row at h VH + d. Each is read where it lies, so a
/// caller hands o's own buffer and no launch copies it (a launch copies an input that is not row-contiguous).
pub const OLayout = enum(u8) { heads, flat };

/// `linearGateMulAt` on o in its heads layout (o_src f32 [1, H, rows, VH]).
pub fn linearGateMul(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Proj, o_src: mlx.mlx_array, dec: format.Decode, g_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    return linearGateMulAt(s, x, gate, o_src, .heads, dec, g_dtype);
}

/// gated [..., H VH] in `g_dtype` (bf16, f16 or f32: o_proj's input dtype) for x [..., in] (bf16, f16 or f32) and
/// o_src f32 in `layout` ([1, H, rows, VH] or [1, rows, H, VH], rows the product of x's leading dims): MLX's astype to
/// `g_dtype` of o (as [rows, H VH]) times MLX's sigmoid of the gate's `linear(x)` in f32, word for word, whichever
/// layout o arrives in. The checks below and the refusals (error.Exl3EngineNotImplemented: decode rows, and where
/// `takes` does not hold) come before any launch, the shape and dtype checks before any engine is made.
pub fn linearGateMulAt(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Proj, o_src: mlx.mlx_array, layout: OLayout, dec: format.Decode, g_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    return gateOf(s, x, gate, o_src, layout, null, dec, g_dtype);
}

/// The prep finish: `linearGateMulAt`'s product as o_proj (`o`) prepares it, xp half [rows, H VH] (natural order, o_proj's
/// suh: `linear`'s prepare arithmetic of the G_T product), in place of the product itself: o_proj continues from xp by
/// `linear`'s own prefill tail (`root.linearOnPrepared`: pg2's launch and split K where the host's switch tables o_proj's
/// shape), so its words are o_proj(`linearGateMulAt`)'s. The checks and refusals are `linearGateMulAt`'s, and o_proj's
/// operands as `root.denseOf` checks them, and `takes` on o_proj's own `Dense` (at `g_dtype`), before any launch.
pub fn linearGateMulPrep(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Proj, o_src: mlx.mlx_array, layout: OLayout, o: Proj, dec: format.Decode, g_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    return gateOf(s, x, gate, o_src, layout, o, dec, g_dtype);
}

/// The two calls' body: `next` null stores the product, else `next`'s prepared rows of it
fn gateOf(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Proj, o_src: mlx.mlx_array, layout: OLayout, next: ?Proj, dec: format.Decode, g_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8) return error.BadExl3Shape;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    if (g_dtype != .bfloat16 and g_dtype != .float32 and g_dtype != .float16) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(o_src) != .float32) return error.BadExl3Dtype;
    var rows: usize = 1;
    for (xsh[0 .. xsh.len - 1]) |n| rows *= @as(usize, @intCast(n));
    if (rows < 1) return error.BadExl3Shape;
    // the decode rows: `linear`'s GEMVs, not this finish
    if (rows <= decode.Kernels.max_rows) return error.Exl3EngineNotImplemented;
    const tsh = mlx.getShape(gate.trellis);
    if (tsh.len != 3) return error.BadExl3Shape;
    const out_dim: usize = @intCast(tsh[1] * 16);
    const osh = mlx.getShape(o_src);
    // the axes of o's heads and rows in its layout
    const ha: usize = if (layout == .flat) 2 else 1;
    const ra: usize = if (layout == .flat) 1 else 2;
    if (osh.len != 4 or osh[0] != 1 or osh[ha] < 1 or osh[3] < 1 or osh[ra] != @as(c_int, @intCast(rows))) return error.BadExl3Shape;
    const heads: usize = @intCast(osh[ha]);
    const o_flat: i32 = @intFromBool(layout == .flat);
    const vh: usize = @intCast(osh[3]);
    if (heads * vh != out_dim) return error.BadExl3Shape;
    if (next) |n| {
        // o_proj's operands as `root.denseOf` checks them (its inputs the gate's outputs), before any engine
        const nsh = mlx.getShape(n.trellis);
        if (nsh.len != 3 or @as(usize, @intCast(nsh[0])) * 16 != out_dim) return error.BadExl3Shape;
        const ush = mlx.getShape(n.suh);
        if (ush.len != 1 or ush[0] != @as(c_int, @intCast(out_dim))) return error.BadExl3Shape;
        if (mlx.mlx_array_dtype(n.suh) != .float16) return error.BadExl3Dtype;
    }
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const d = try root.denseOf(gate, in_dim, xd, dec);
    if (!takes(d, rows, gate.trellis)) return error.Exl3EngineNotImplemented;
    // o_proj reads the prepared rows as `linear` reads its own prepare's (`root.linearOnPrepared`'s paths)
    if (next) |n| if (!takes(try root.denseOf(n, @intCast(out_dim), g_dtype, dec), rows, n.trellis)) return error.Exl3EngineNotImplemented;
    const v = try root.rowsOf(x, in_dim, s);
    defer freeArr(v.x2);
    const r: c_int = @intCast(rows);
    const OUT: c_int = @intCast(out_dim);
    const tmpl_all = [_]ks.TemplateArg{
        .{ .name = "DIM", .value = .{ .int = OUT } },
        .{ .name = "VH", .value = .{ .int = @intCast(vh) } },
        .{ .name = "H", .value = .{ .int = @intCast(heads) } },
        .{ .name = "OUT_T", .value = .{ .dtype = xd } },
        .{ .name = "G_T", .value = .{ .dtype = g_dtype } },
        .{ .name = "O_FLAT", .value = .{ .int = o_flat } },
        .{ .name = "LANE_ORDER", .value = .{ .int = 0 } },
    };
    // the prep launches name o_proj's prepare's LANE_ORDER (natural); the product's do not
    const tmpl: []const ks.TemplateArg = if (next != null) &tmpl_all else tmpl_all[0 .. tmpl_all.len - 1];
    const out_dt: mlx.mlx_dtype = if (next != null) .float16 else g_dtype;
    // pg2 at its shapes: the gate's product and this finish in one launch
    if (try pg2Of(d, s, v.x2, gate.trellis, gate.suh, rows)) |pg| {
        defer freeArr(pg.xp);
        defer freeArr(pg.w);
        if (next) |n| return pgemm.launchWith(s, try kernelGet(.pg2_prep), &.{ pg.xp, pg.w, gate.svh, o_src, n.suh }, out_dt, pg.plan, tmpl);
        const y2 = try pgemm.launchWith(s, try kernelGet(.pg2), &.{ pg.xp, pg.w, gate.svh, o_src }, g_dtype, pg.plan, tmpl);
        defer freeArr(y2);
        return shapedAs(s, y2, xsh, OUT);
    }
    const k = try kernelGet(if (next != null) .finish_prep else .finish);
    // the gate's inner product by `linear`'s own launches at these rows, then its finish's grid with the gate's text
    const z = try d.innerOn(s, s, v.x2, gate.trellis, gate.suh, rows);
    defer freeArr(z);
    const cfg = try ks.buildConfig(.{
        .grid = .{ @as(u32, @intCast(out_dim / 128)) * 32, @intCast(rows), 1 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{&[_]c_int{ r, OUT }},
        .out_dtypes = &.{out_dt},
        .template = tmpl,
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [1]mlx.mlx_array = undefined;
    if (next) |n| {
        // xp [rows, DIM], as `root.linearOnPrepared` takes it
        try ks.applyConfig(k, cfg, &.{ z, gate.svh, o_src, n.suh }, &o, s);
        return o[0];
    }
    try ks.applyConfig(k, cfg, &.{ z, gate.svh, o_src }, &o, s);
    defer freeArr(o[0]);
    return shapedAs(s, o[0], xsh, OUT);
}
