//! Hy4-flash's RMSNorms and the next linear's prepare in a linear's prompt finish (the hy4f model plugin's
//! `addnorm_fused`, `qnorm_fused`, `kvnorm_fused`, `down_prep_fused`). The chains they replace, at a prompt's
//! rows (hy4f src/model.zig):
//!   linearRms         yn = rmsNorm(astype(up, f32) + astype(linear(x), f32), w)    the shared down, the add of the
//!                                                                                    routed experts' up, the MLP post-norm
//!   linearRmsPrep     q = rmsNorm(astype(linear(x), f32), w); xp = prepare(astype(q, Q_T))   q_a, q_norm, q_b's prepare
//!   linearKv          c_kv = rmsNorm(astype(linear(x), f32)[..., :LAT], w);
//!                     k_pe = fast::rope(astype(linear(x), f32)[..., LAT:LAT + ROPE])         kv_a, kv_norm, k_pe's rope
//!   linearSwigluPrep  xp = prepare(h) of swiglu.zig's linearSwigluMul h             the down linear's prepare
//! Each runs the linear's inner product by `linear`'s own launches at these rows (`Dense.innerOn` from x, or
//! `Dense.innerOf` from rows a launch before prepared: `Src`), then one finish launch a row: one threadgroup a row, one
//! simdgroup a 128-column block running text_finish's loads, butterfly and value (finish_v's statements with the block
//! the simdgroup's), the row staged by element in threadgroup memory as text_finish's store rounds it (OUT_T, then f32:
//! MLX's astype), then MLX's rms_single_row statements over it (rms_norm.metal at the server's MLX: its threadgroup of
//! axis / 4 threads is this kernel's axis / 128 simdgroups, thread lid taking values 4 lid .. 4 lid + 3, its sums in its
//! order), its output loop, and the epilogue's own tail (the next linear's prepare: dense prep_2_2 / prep_3 on the
//! staged Q_T row; qrope's rope statements in the simdgroup holding k_pe). linearSwigluPrep is text_finish_swiglu's grid
//! (one simdgroup per (128-block, row)) with the down linear's prepare of h in place of h's store. Every word the
//! chain's; the prepared rows are the bytes the next linear's own prepare writes, so it continues as `linear`
//! (`root.linearPrepared`).
//! Rows `linear` serves another way refuse with error.Exl3EngineNotImplemented before any launch (the caller keeps the
//! chain): the decode rows (<= 16), the fused-GEMM option, pg2, trunk int8 and an un-rotated resident trellis
//! (`gate.takes`, on the next linear too where the call prepares its rows), and rows of more than 4096 outputs (one
//! threadgroup a row: MLX's single-row kernel's limit).
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const root = @import("../../root.zig");
const gate = @import("gate.zig");
const swiglu = @import("swiglu.zig");
const qrope = @import("qrope.zig");
const Proj = root.Proj;

fn freeArr(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

fn ci(v: anytype) c_int {
    return @intCast(v);
}

// ── texts ──

/// finish_v's statements with the 128-column block the simdgroup's (one threadgroup a row): dense/finish_v with its
/// block line `b = threadgroup_position_in_grid.x` as `b = simdgroup_index_in_threadgroup` (a CPU test holds the rest
/// equal)
pub const finish_v_row = @embedFile("kernels/rms/finish_v_row.metal");
/// The row staged as text_finish's store gives it (OUT_T, then f32), alone (the prepare's and kv's texts) or added to the
/// other branch's y (`add`: linearRms)
pub const rms_stage = @embedFile("kernels/rms/rms_stage.metal");
pub const rms_stage_add = @embedFile("kernels/rms/rms_stage_add.metal");
/// MLX's rms_single_row up to its inverse mean, then its output loop (rms_store) or the normed row in Q_T staged for
/// the next linear's prepare (rms_qx) and that prepare's load (rms_prep_load)
pub const rms_sum = @embedFile("kernels/rms/rms_sum.metal");
pub const rms_store = @embedFile("kernels/rms/rms_store.metal");
pub const rms_qx = @embedFile("kernels/rms/rms_qx.metal");
pub const rms_prep_load = @embedFile("kernels/rms/rms_prep_load.metal");
/// k_pe's rope in the simdgroup holding it: its opening and close around qrope's pair and rope statements
pub const kv_rope_open = @embedFile("kernels/rms/kv_rope_open.metal");
pub const kv_rope_close = @embedFile("kernels/rms/kv_rope_close.metal");
/// The SwiGLU finish's store as the down linear's prepare (swiglu.zig's texts around it)
pub const swp_open = @embedFile("kernels/rms/swp_open.metal");
pub const swp_close = @embedFile("kernels/rms/swp_close.metal");

/// y f32 [rows, NORM] = MLX's rms_norm(f32(add) + f32(OUT_T(text_finish's value)), w) of the row's first NORM = nrm[0]
/// values (all DIM here); z half [rows, DIM], svh half [DIM], add [rows, DIM] (any float dtype), w f32 [NORM], nrm
/// uint32 [1], eps_in f32 [1]
pub const text_add_rms: [:0]const u8 = finish_v_row ++ rms_stage_add ++ rms_sum ++ rms_store;
/// xp half [rows, DIM] = the next linear's prepare (suh half [DIM], natural order) of Q_T(the normed row), and q f32
/// [rows, DIM] the normed row where QF is 1
pub const text_rms_prep: [:0]const u8 = finish_v_row ++ rms_stage ++ rms_sum ++ rms_qx ++ rms_prep_load ++ dense.prep_2_2 ++ dense.prep_3;
/// y f32 [rows, LAT] = the norm of the row's first LAT values (nrm[0] = LAT), k_pe f32 [rows, ROPE] = MLX's fast::rope
/// (traditional, base 2^as_float(LOG2B), scale 1, the row at off[0] + row) of values LAT .. LAT + ROPE
pub const text_kv: [:0]const u8 = finish_v_row ++ rms_stage ++ rms_sum ++ rms_store ++ kv_rope_open ++ qrope.qrope_pair ++ qrope.qrope_rot ++ kv_rope_close;
/// xp half [rows, DIM] = the down linear's prepare (suh half [DIM]) of linearSwigluMul's h; one simdgroup per
/// (128-block, row) as text_finish_swiglu
pub const text_swiglu_prep: [:0]const u8 = dense.finish_v ++ swp_open ++ swiglu.swiglu_open ++ gate.gate_sig ++ swp_close ++ dense.prep_2_2 ++ dense.prep_3;

/// The launches, one kernel each
pub const Kind = enum(u8) { add_rms, rms_prep, kv, swiglu_prep };
const kind_count = @typeInfo(Kind).@"enum".field_names.len;

const add_rms_in = [_][*:0]const u8{ "z", "svh", "add", "w", "nrm", "eps_in" };
const rms_prep_in = [_][*:0]const u8{ "z", "svh", "w", "nrm", "eps_in", "suh" };
const kv_in = [_][*:0]const u8{ "z", "svh", "w", "nrm", "eps_in", "off" };
const swiglu_prep_in = [_][*:0]const u8{ "z", "svh", "gate", "suh" };
const y_out = [_][*:0]const u8{"y"};
const prep_out = [_][*:0]const u8{ "xp", "q" };
const kv_out = [_][*:0]const u8{ "y", "k_pe" };
const xp_out = [_][*:0]const u8{"xp"};

/// The red checks' planted faults (`plant`), each the served text of one launch with one piece swapped: the norm's
/// output without its weight (`kv_no_weight`: linearKv's c_kv), the other branch's y left out of the sum (`no_add`:
/// linearRms), the next linear's prepare without its suh (`prep_no_suh`, `swiglu_no_suh`), k_pe's frequency by
/// MLX's CPU fallback formula (`kv_freq_ln`, qrope's fault piece)
pub const Fault = enum(u8) { no_add, prep_no_suh, kv_no_weight, kv_freq_ln, swiglu_no_suh };
const fault_count = @typeInfo(Fault).@"enum".field_names.len;

fn faultKind(f: Fault) Kind {
    return switch (f) {
        .no_add => .add_rms,
        .prep_no_suh => .rms_prep,
        .kv_no_weight, .kv_freq_ln => .kv,
        .swiglu_no_suh => .swiglu_prep,
    };
}

const fault_noweight = @embedFile("kernels/test/fault/rms_store_noweight.metal");
const fault_prep_nosuh = @embedFile("kernels/test/fault/rms_prep_load_nosuh.metal");
const fault_swp_nosuh = @embedFile("kernels/test/fault/swp_close_nosuh.metal");
const fault_freq_ln = @embedFile("kernels/test/fault/qrope_rot_freq_ln.metal");

// the add dropped: rms_stage's statements under another comment, its own text (the server's kernel cache keys a text)
const text_fault_no_add: [:0]const u8 = finish_v_row ++ @embedFile("kernels/test/fault/rms_stage_add_dropped.metal") ++ rms_sum ++ rms_store;
const text_fault_prep_no_suh: [:0]const u8 = finish_v_row ++ rms_stage ++ rms_sum ++ rms_qx ++ fault_prep_nosuh ++ dense.prep_2_2 ++ dense.prep_3;
const text_fault_kv_no_weight: [:0]const u8 = finish_v_row ++ rms_stage ++ rms_sum ++ fault_noweight ++ kv_rope_open ++ qrope.qrope_pair ++ qrope.qrope_rot ++ kv_rope_close;
const text_fault_kv_freq_ln: [:0]const u8 = finish_v_row ++ rms_stage ++ rms_sum ++ rms_store ++ kv_rope_open ++ qrope.qrope_pair ++ fault_freq_ln ++ kv_rope_close;
const text_fault_swiglu_no_suh: [:0]const u8 = dense.finish_v ++ swp_open ++ swiglu.swiglu_open ++ gate.gate_sig ++ fault_swp_nosuh ++ dense.prep_2_2 ++ dense.prep_3;

// ── kernels: made at their first call ──

const no_kernels: [kind_count]?mlx.mlx_fast_metal_kernel = @splat(null);
const kernel_lock = Global(std.atomic.Value(u32), "hy4f_rms_kernel_lock", std.atomic.Value(u32).init(0));
const kernels = Global([kind_count]?mlx.mlx_fast_metal_kernel, "hy4f_rms_kernels", no_kernels);
const fault_kernels = Global([fault_count]?mlx.mlx_fast_metal_kernel, "hy4f_rms_fault_kernels", @splat(null));

/// Test seam: the fault its launch carries in place of the served text (`plant`; null: the served ones)
pub const planted = Global(?Fault, "hy4f_rms_planted", null);

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernels, fault_kernels, planted };

fn acquire() void {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}

fn release() void {
    kernel_lock.get().*.store(0, .release);
}

fn makeServed(k: Kind) ?mlx.mlx_fast_metal_kernel {
    return switch (k) {
        .add_rms => ks.newKernel("turboexl3_hy4f_finish_add_rms", &add_rms_in, &y_out, text_add_rms, decode.header, true),
        .rms_prep => ks.newKernel("turboexl3_hy4f_finish_rms_prep", &rms_prep_in, &prep_out, text_rms_prep, decode.header, true),
        .kv => ks.newKernel("turboexl3_hy4f_finish_kv", &kv_in, &kv_out, text_kv, decode.header, true),
        .swiglu_prep => ks.newKernel("turboexl3_hy4f_finish_swiglu_prep", &swiglu_prep_in, &xp_out, text_swiglu_prep, decode.header, true),
    };
}

fn makeFault(f: Fault) ?mlx.mlx_fast_metal_kernel {
    return switch (f) {
        .no_add => ks.newKernel("turboexl3_test_hy4f_finish_add_rms_no_add_fault", &add_rms_in, &y_out, text_fault_no_add, decode.header, true),
        .prep_no_suh => ks.newKernel("turboexl3_test_hy4f_finish_rms_prep_no_suh_fault", &rms_prep_in, &prep_out, text_fault_prep_no_suh, decode.header, true),
        .kv_no_weight => ks.newKernel("turboexl3_test_hy4f_finish_kv_no_weight_fault", &kv_in, &kv_out, text_fault_kv_no_weight, decode.header, true),
        .kv_freq_ln => ks.newKernel("turboexl3_test_hy4f_finish_kv_freq_ln_fault", &kv_in, &kv_out, text_fault_kv_freq_ln, decode.header, true),
        .swiglu_no_suh => ks.newKernel("turboexl3_test_hy4f_finish_swiglu_prep_no_suh_fault", &swiglu_prep_in, &xp_out, text_fault_swiglu_no_suh, decode.header, true),
    };
}

/// The kernel of launch `k`: the planted fault's where one of `k`'s is planted, else the served text's
fn kernelFor(k: Kind) !mlx.mlx_fast_metal_kernel {
    acquire();
    defer release();
    if (planted.get().*) |f| if (faultKind(f) == k) {
        const slot = &fault_kernels.get().*[@intFromEnum(f)];
        if (slot.*) |made| return made;
        const made = makeFault(f) orelse return error.KernelCreateFailed;
        slot.* = made;
        return made;
    };
    const slot = &kernels.get().*[@intFromEnum(k)];
    if (slot.*) |made| return made;
    const made = makeServed(k) orelse return error.KernelCreateFailed;
    slot.* = made;
    return made;
}

/// Test seam: fault `f`'s kernel in its launch's served one's place for every call until `unplant` (a red check: the
/// words must move). The kernel is made at its first call.
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

// ── the call's pieces ──

/// What a finish reads its linear's inner product from: x [..., in] (bf16, f16 or f32; `linear`'s prepare inside,
/// `Dense.innerOn`), or rows a launch before prepared, xp f16 [..., in] in natural order with `linear`'s prepare
/// arithmetic of an x of the call's dtype (`Dense.innerOf`: a prepared epilogue's output, `linearRmsPrep`'s and
/// `linearSwigluPrep`'s)
pub const Src = union(enum) { x: mlx.mlx_array, xp: mlx.mlx_array };

/// The most outputs a row's threadgroup takes (MLX's single-row rms kernel's limit: RMS_LOOPED_LIMIT)
pub const max_dim: usize = 4096;

/// A call's geometry, from its operands' shapes and dtypes alone (no engine, no launch): the source array, the rows,
/// the linear's out width, the source's leading shape, the dtype the linear's finish rounds to (x's, or the caller's for
/// prepared rows)
const Shape = struct { a: mlx.mlx_array, x: bool, rows: usize, in: u32, out: usize, lead: [8]c_int, nlead: usize, out_t: mlx.mlx_dtype };

fn floatDtype(d: mlx.mlx_dtype) bool {
    return d == .bfloat16 or d == .float32 or d == .float16;
}

/// The checks every call makes before any engine: src's shape and dtype (`xd` the finish's dtype for prepared rows), the
/// decode rows (`linear`'s GEMVs: error.Exl3EngineNotImplemented), the trellis' shape, whole 128-blocks up to `max_dim`
/// outputs (one threadgroup a row; wider refuses)
fn shapeOf(src: Src, xd: mlx.mlx_dtype, p: Proj) !Shape {
    const is_x = src == .x;
    const a = switch (src) {
        .x, .xp => |v| v,
    };
    const sh = mlx.getShape(a);
    if (sh.len < 1 or sh.len > 8) return error.BadExl3Shape;
    const ad = mlx.mlx_array_dtype(a);
    const out_t = if (is_x) ad else xd;
    if (is_x and !floatDtype(ad)) return error.BadExl3Dtype;
    if (!is_x and ad != .float16) return error.BadExl3Dtype;
    if (!floatDtype(out_t)) return error.BadExl3Dtype;
    var rows: usize = 1;
    for (sh[0 .. sh.len - 1]) |n| rows *= @as(usize, @intCast(n));
    if (rows < 1) return error.BadExl3Shape;
    if (rows <= decode.Kernels.max_rows) return error.Exl3EngineNotImplemented;
    const tsh = mlx.getShape(p.trellis);
    if (tsh.len != 3) return error.BadExl3Shape;
    const out_dim: usize = @intCast(tsh[1] * 16);
    if (out_dim % 128 != 0) return error.BadExl3Shape;
    if (out_dim > max_dim) return error.Exl3EngineNotImplemented;
    var lead: [8]c_int = undefined;
    @memcpy(lead[0 .. sh.len - 1], sh[0 .. sh.len - 1]);
    return .{ .a = a, .x = is_x, .rows = rows, .in = @intCast(sh[sh.len - 1]), .out = out_dim, .lead = lead, .nlead = sh.len - 1, .out_t = out_t };
}

/// The `Dense` `linear` takes for the call, where `gate.takes` holds (else error.Exl3EngineNotImplemented): no launch
fn denseFor(sh: Shape, p: Proj, dec: format.Decode) !*dense.Dense {
    const d = try root.denseOf(p, sh.in, sh.out_t, dec);
    if (!gate.takes(d, sh.rows, p.trellis)) return error.Exl3EngineNotImplemented;
    return d;
}

/// The next linear's operands as `root.denseOf` checks them (its trellis [in_dim / 16, out / 16, n], suh f16 [in_dim],
/// svh f16 [out]), before any engine
fn nextShape(next: Proj, in_dim: usize) !void {
    const tsh = mlx.getShape(next.trellis);
    if (tsh.len != 3 or @as(usize, @intCast(tsh[0])) * 16 != in_dim) return error.BadExl3Shape;
    const ush = mlx.getShape(next.suh);
    const vsh = mlx.getShape(next.svh);
    if (ush.len != 1 or ush[0] != ci(in_dim) or vsh.len != 1 or vsh[0] != tsh[1] * 16) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(next.suh) != .float16 or mlx.mlx_array_dtype(next.svh) != .float16) return error.BadExl3Dtype;
}

/// Whether the next linear (`next`, its inputs `in_dim` wide, its finish at `xd`) reads prepared rows at `rows` as
/// `linear` reads its own prepare's (`root.linearPrepared`'s paths with `gate.takes`' conditions): error.Exl3EngineNotImplemented
/// where it would not; no launch
fn checkNext(next: Proj, in_dim: usize, xd: mlx.mlx_dtype, rows: usize, dec: format.Decode) !void {
    const dn = try root.denseOf(next, @intCast(in_dim), xd, dec);
    if (!gate.takes(dn, rows, next.trellis)) return error.Exl3EngineNotImplemented;
}

/// The inner product z half [rows, out] by `linear`'s own launches at these rows (from x, or from the prepared rows on
/// `linear`'s path there); the caller frees it
fn innerOf(s: mlx.mlx_stream, sh: Shape, d: *dense.Dense, p: Proj) !mlx.mlx_array {
    const v = try root.rowsOf(sh.a, sh.in, s);
    defer freeArr(v.x2);
    if (sh.x) return d.innerOn(s, s, v.x2, p.trellis, p.suh, sh.rows);
    return d.innerOf(s, v.x2, p.trellis, sh.rows, d.pathFor(sh.rows));
}

/// An array of `y` [rows, n] reshaped to the call's leading shape and n; the caller frees it
fn shaped(s: mlx.mlx_stream, y: mlx.mlx_array, sh: Shape, n: usize) !mlx.mlx_array {
    var osh: [8]c_int = undefined;
    @memcpy(osh[0..sh.nlead], sh.lead[0..sh.nlead]);
    osh[sh.nlead] = ci(n);
    var out = mlx.mlx_array_new();
    errdefer freeArr(out);
    try mlx.check(mlx.mlx_reshape(&out, y, &osh, @intCast(sh.nlead + 1), s));
    return out;
}

/// An operand of the call's rows and `n` values a row (`add`, the gate's y): its dtype one of the float three (`want`
/// where given) and its last dim n, rows x n values
fn checkRows(a: mlx.mlx_array, sh: Shape, n: usize, want: ?mlx.mlx_dtype) !void {
    const d = mlx.mlx_array_dtype(a);
    if (want) |w| {
        if (d != w) return error.BadExl3Dtype;
    } else if (!floatDtype(d)) return error.BadExl3Dtype;
    const ash = mlx.getShape(a);
    if (ash.len < 1 or ash[ash.len - 1] != ci(n) or mlx.mlx_array_size(a) != sh.rows * n) return error.BadExl3Shape;
}

/// A norm's weight: f32 [n]
fn checkWeight(w: mlx.mlx_array, n: usize) !void {
    const sh = mlx.getShape(w);
    if (mlx.mlx_array_dtype(w) != .float32) return error.BadExl3Dtype;
    if (sh.len != 1 or sh[0] != ci(n)) return error.BadExl3Shape;
}

/// The norm's scalars as MLX's kernel takes them: the axis size (uint32) and eps (f32), one-element arrays the caller
/// frees
fn scalars(norm: usize, eps: f32) ![2]mlx.mlx_array {
    const n_host = [_]u32{@intCast(norm)};
    const e_host = [_]f32{eps};
    const n = mlx.mlx_array_new_data(&n_host, &[_]c_int{1}, 1, .uint32);
    if (n.ctx == null) return error.MlxError;
    errdefer freeArr(n);
    const e = mlx.mlx_array_new_data(&e_host, &[_]c_int{1}, 1, .float32);
    if (e.ctx == null) return error.MlxError;
    return .{ n, e };
}

/// The row kernels' launch: one threadgroup a row of out / 128 simdgroups
fn rowLaunch(sh: Shape, outs: []const []const c_int, dts: []const mlx.mlx_dtype, tmpl: []const ks.TemplateArg) !mlx.mlx_fast_metal_kernel_config {
    const nb: u32 = @intCast(sh.out / 128);
    return ks.buildConfig(.{ .grid = .{ nb * 32, @intCast(sh.rows), 1 }, .threadgroup = .{ nb * 32, 1, 1 }, .out_shapes = outs, .out_dtypes = dts, .template = tmpl });
}

// ── the calls ──

/// f32 [..., out] = with `add` [..., out] (bf16, f16 or f32) MLX's rms_norm(f32(add) + f32(`linear(src)`), w, eps), word
/// for word; `xd` the finish's dtype for prepared rows (ignored for
/// an x: its own). w f32 [out]. The checks and the refusals (error.Exl3EngineNotImplemented: decode rows, more than
/// `max_dim` outputs, and where `gate.takes` does not hold) come before any launch, the shape and dtype checks before any
/// engine is made.
pub fn linearRms(s: mlx.mlx_stream, src: Src, xd: mlx.mlx_dtype, p: Proj, add: mlx.mlx_array, w: mlx.mlx_array, eps: f32, dec: format.Decode) !mlx.mlx_array {
    const sh = try shapeOf(src, xd, p);
    try checkWeight(w, sh.out);
    try checkRows(add, sh, sh.out, null);
    const d = try denseFor(sh, p, dec);
    const k = try kernelFor(.add_rms);
    const sc = try scalars(sh.out, eps);
    defer for (sc) |a| freeArr(a);
    const r = ci(sh.rows);
    const OUT = ci(sh.out);
    const cfg = try rowLaunch(sh, &.{&[_]c_int{ r, OUT }}, &.{.float32}, &.{
        .{ .name = "DIM", .value = .{ .int = OUT } },
        .{ .name = "OUT_T", .value = .{ .dtype = sh.out_t } },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const z = try innerOf(s, sh, d, p);
    defer freeArr(z);
    var o: [1]mlx.mlx_array = undefined;
    var a2 = mlx.mlx_array_new();
    defer freeArr(a2);
    try mlx.check(mlx.mlx_reshape(&a2, add, &[_]c_int{ r, OUT }, 2, s));
    try ks.applyConfig(k, cfg, &.{ z, p.svh, a2, w, sc[0], sc[1] }, &o, s);
    defer freeArr(o[0]);
    return shaped(s, o[0], sh, sh.out);
}

/// `linearRmsPrep`'s output: the next linear's prepared rows xp f16 [rows, out] (natural order) and, where asked, the
/// normed row q f32 [..., out]; the caller frees both
pub const Prepped = struct { xp: mlx.mlx_array, q: ?mlx.mlx_array };

/// xp = the next linear's prepare (`next`'s suh, `linear`'s prepare arithmetic) of astype(q, q_t), q = MLX's
/// rms_norm(f32(`linear(src)`), w, eps), and q itself in f32 where `want_q`, word for word; `xd` as `linearRms`. The next
/// linear reads xp as prepared rows (`root.linearPrepared`, qrope.zig's `linearQRopePrepared`, at `q_t`); where it would
/// not read them as its own prepare's (`checkNext`) the call refuses with the others, before any launch.
pub fn linearRmsPrep(s: mlx.mlx_stream, src: Src, xd: mlx.mlx_dtype, p: Proj, w: mlx.mlx_array, eps: f32, next: Proj, q_t: mlx.mlx_dtype, want_q: bool, dec: format.Decode) !Prepped {
    const sh = try shapeOf(src, xd, p);
    try checkWeight(w, sh.out);
    if (!floatDtype(q_t)) return error.BadExl3Dtype;
    try nextShape(next, sh.out);
    const d = try denseFor(sh, p, dec);
    try checkNext(next, sh.out, q_t, sh.rows, dec);
    const k = try kernelFor(.rms_prep);
    const sc = try scalars(sh.out, eps);
    defer for (sc) |a| freeArr(a);
    const r = ci(sh.rows);
    const OUT = ci(sh.out);
    // q's array at one value where it is not asked for (the kernel stores none: QF 0)
    const q_shape = if (want_q) [_]c_int{ r, OUT } else [_]c_int{ 1, 1 };
    const cfg = try rowLaunch(sh, &.{ &[_]c_int{ r, OUT }, &q_shape }, &.{ .float16, .float32 }, &.{
        .{ .name = "DIM", .value = .{ .int = OUT } },
        .{ .name = "OUT_T", .value = .{ .dtype = sh.out_t } },
        .{ .name = "Q_T", .value = .{ .dtype = q_t } },
        .{ .name = "QF", .value = .{ .int = @intFromBool(want_q) } },
        .{ .name = "LANE_ORDER", .value = .{ .int = 0 } },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const z = try innerOf(s, sh, d, p);
    defer freeArr(z);
    var o: [2]mlx.mlx_array = undefined;
    try ks.applyConfig(k, cfg, &.{ z, p.svh, w, sc[0], sc[1], next.suh }, &o, s);
    if (!want_q) {
        freeArr(o[1]);
        return .{ .xp = o[0], .q = null };
    }
    defer freeArr(o[1]);
    errdefer freeArr(o[0]);
    return .{ .xp = o[0], .q = try shaped(s, o[1], sh, sh.out) };
}

/// `linearKv`'s output: c_kv f32 [..., lat] and k_pe f32 [..., rope]; the caller frees both
pub const Kv = struct { c_kv: mlx.mlx_array, k_pe: mlx.mlx_array };

/// c_kv = MLX's rms_norm(f32(`linear(src)`)[..., :lat], w, eps) and k_pe = MLX's fast::rope of f32(`linear(src)`)[...,
/// lat : lat + rope] (traditional pairs, base `theta`, scale 1, offset `pos` for the first row, row r at `pos + r`), word
/// for word; `xd` as `linearRms`. The linear's outputs (whole 128-blocks up to `max_dim`) hold lat + rope (lat a whole
/// number of 128, rope even and at most 128: one simdgroup's); the rest are not read. The checks and refusals come
/// before any launch, the shape and dtype checks before any engine is made.
pub fn linearKv(s: mlx.mlx_stream, src: Src, xd: mlx.mlx_dtype, p: Proj, w: mlx.mlx_array, eps: f32, lat: u32, rope: u32, theta: f32, pos: i32, dec: format.Decode) !Kv {
    const sh = try shapeOf(src, xd, p);
    if (lat == 0 or lat % 128 != 0 or rope == 0 or rope % 2 != 0 or rope > 128 or lat + rope > sh.out) return error.BadExl3Shape;
    if (!(theta > 0) or !std.math.isFinite(theta) or pos < 0) return error.BadExl3Shape;
    try checkWeight(w, lat);
    const d = try denseFor(sh, p, dec);
    const k = try kernelFor(.kv);
    const sc = try scalars(lat, eps);
    defer for (sc) |a| freeArr(a);
    const off_host = [_]i32{pos};
    const off = mlx.mlx_array_new_data(&off_host, &[_]c_int{1}, 1, .int32);
    if (off.ctx == null) return error.MlxError;
    defer freeArr(off);
    const r = ci(sh.rows);
    const cfg = try rowLaunch(sh, &.{ &[_]c_int{ r, ci(lat) }, &[_]c_int{ r, ci(rope) } }, &.{ .float32, .float32 }, &.{
        .{ .name = "DIM", .value = .{ .int = ci(sh.out) } },
        .{ .name = "OUT_T", .value = .{ .dtype = sh.out_t } },
        .{ .name = "LAT", .value = .{ .int = ci(lat) } },
        .{ .name = "ROPE", .value = .{ .int = ci(rope) } },
        .{ .name = "LOG2B", .value = .{ .int = @bitCast(qrope.log2Base(theta)) } },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const z = try innerOf(s, sh, d, p);
    defer freeArr(z);
    var o: [2]mlx.mlx_array = undefined;
    try ks.applyConfig(k, cfg, &.{ z, p.svh, w, sc[0], sc[1], off }, &o, s);
    defer for (o) |a| freeArr(a);
    const c_kv = try shaped(s, o[0], sh, lat);
    errdefer freeArr(c_kv);
    return .{ .c_kv = c_kv, .k_pe = try shaped(s, o[1], sh, rope) };
}

/// xp f16 [rows, I] = the down linear's prepare (`down`'s suh, `linear`'s prepare arithmetic) of swiglu.zig's
/// `linearSwigluMul(x, up, gate_y, h_dtype)`, word for word; the down linear reads it as prepared rows at `h_dtype`
/// (`root.linearPrepared`, `linearRms` from `.xp`). The checks and refusals are `linearSwigluMul`'s and `checkNext`'s,
/// before any launch; up's outputs need not fit a threadgroup (one simdgroup per (128-block, row), as
/// text_finish_swiglu).
pub fn linearSwigluPrep(s: mlx.mlx_stream, x: mlx.mlx_array, up: Proj, gate_y: mlx.mlx_array, down: Proj, dec: format.Decode, h_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8) return error.BadExl3Shape;
    const xd = mlx.mlx_array_dtype(x);
    if (!floatDtype(xd) or !floatDtype(h_dtype)) return error.BadExl3Dtype;
    var rows: usize = 1;
    for (xsh[0 .. xsh.len - 1]) |n| rows *= @as(usize, @intCast(n));
    if (rows < 1) return error.BadExl3Shape;
    if (rows <= decode.Kernels.max_rows) return error.Exl3EngineNotImplemented;
    const tsh = mlx.getShape(up.trellis);
    if (tsh.len != 3) return error.BadExl3Shape;
    const out_dim: usize = @intCast(tsh[1] * 16);
    if (out_dim % 128 != 0) return error.BadExl3Shape;
    var lead: [8]c_int = undefined;
    @memcpy(lead[0 .. xsh.len - 1], xsh[0 .. xsh.len - 1]);
    const sh: Shape = .{ .a = x, .x = true, .rows = rows, .in = @intCast(xsh[xsh.len - 1]), .out = out_dim, .lead = lead, .nlead = xsh.len - 1, .out_t = xd };
    // the gate's y as `linear` gives it: in x's dtype
    try checkRows(gate_y, sh, out_dim, xd);
    try nextShape(down, out_dim);
    const d = try denseFor(sh, up, dec);
    try checkNext(down, out_dim, h_dtype, rows, dec);
    const k = try kernelFor(.swiglu_prep);
    const r = ci(rows);
    const OUT = ci(out_dim);
    const cfg = try ks.buildConfig(.{
        .grid = .{ @as(u32, @intCast(out_dim / 128)) * 32, @intCast(rows), 1 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{&[_]c_int{ r, OUT }},
        .out_dtypes = &.{.float16},
        .template = &.{
            .{ .name = "DIM", .value = .{ .int = OUT } },
            .{ .name = "OUT_T", .value = .{ .dtype = xd } },
            .{ .name = "H_T", .value = .{ .dtype = h_dtype } },
            .{ .name = "LANE_ORDER", .value = .{ .int = 0 } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var g2 = mlx.mlx_array_new();
    defer freeArr(g2);
    try mlx.check(mlx.mlx_reshape(&g2, gate_y, &[_]c_int{ r, OUT }, 2, s));
    const z = try innerOf(s, sh, d, up);
    defer freeArr(z);
    var o: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(k, cfg, &.{ z, up.svh, g2, down.suh }, &o, s);
    return o[0];
}
