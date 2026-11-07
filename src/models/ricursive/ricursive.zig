//! Ricursive Ripple-65B's EXL3 linears on turboexl3's dense kernels. Ripple is dense: 80 layers, hidden 5120, 80 query
//! heads and 8 kv heads of 128, a sigmoid output gate g_proj (5120 -> 10240) beside q (5120 -> 10240), k and v (5120 ->
//! 1024; v's input the value residual lerp(x0, x, sigmoid(lambda))), o (10240 -> 5120), an MLP gate, up (5120 -> 40960)
//! and down (40960 -> 5120) with the SwiGLU clamped at 20 (silu(min(gate, 20)) * clip(up, -20, 20)), and an untied head
//! (5120 -> 100352). Its pack is uniform K3 (MUL1), the head K6. Every projection runs `root.linear`; this file adds
//! what folds launches where several linears read one input or one linear's input is a cheap function of others':
//!
//! - `jointLinear`: two linears of one input in one DEC-ONE launch (q | g, gate | up): their trellises joined end to end
//!   and their svh along OUT once at load (`join`), each threadgroup's prepare scaling x by its member's own suh and its
//!   GEMV reading its member's trellis (`text_jdec1`: dense's DEC-ONE text with the prepare inside, `jsel.metal`
//!   choosing the member, `jtile.metal` its tile in place of `dense.dec_tile`). Each member's K split is its own
//!   linear's (`Dense.ksFor(in, out)` of one member, so the members must share it), so y [rows, out0 + out1] holds each
//!   member's `linear` bytes. The joint is the one resident copy: each member is a view of it (`Joint.members`, no copy),
//!   which every per-member call (`root.linear`, `valueLinear`, `mlp`'s two linears) takes where the joined launch
//!   does not serve.
//! - `jointLinearBlend`: the same for k | v with v's input the value residual, formed in the prepare (`text_jdec1_blend`):
//!   v's columns are `linear(v, T(x0 + blend (x - x0)))`'s bytes, the blend in f32 as the host's ops form it.
//! - `valueLinear`: v alone at prefill rows, the blend in its prepare (`text_blend_prep`): the bytes of the host's blend
//!   then `linear`.
//! - `mlp`: the MLP. Decode rows: gate | up through `jointLinear` where it serves (else two `linear`), then one launch
//!   of the host's clamped SwiGLU in the activation dtype with the down's prepare (`text_swiglu_prep_sc`: dense's
//!   prep_sc reading the SwiGLU through `swiglu_x.metal`), then the down's DEC-ONE GEMV (`dec0`) or, past DEC-ONE's
//!   rows, its GEMV and finish. Prefill rows: gate and up through `linear`, the SwiGLU in the down's prepare
//!   (`text_swiglu_prep`), then the down's inner product and finish (`root.linearPrepared`). The bytes of `linear(gate)`,
//!   `linear(up)`, the host's clamped SwiGLU (MLX's minimum, clip, sigmoid and two multiplies; the sigmoid read from
//!   the host's table of MLX's own results) and `linear(down)`.
//!
//! o and the head are `root.linear` (o's input, the gated attention output, is the host's). Rows: `jointLinear` and
//! `jointLinearBlend` serve 1 to 16 rows where the joined prepare fits threadgroup memory (`Dense.decPrepFits`: q | g and
//! gate | up where a row chunk is at most 3 rows, so 1, 2, 3 and 6; k | v every decode row); a call they do not serve returns error.Exl3EngineNotImplemented before any
//! launch, and the caller runs `root.linear` per member.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const root = @import("../../root.zig");

/// Ripple-65B's shapes
pub const hidden: u32 = 5120;
pub const q_out: u32 = 10240;
pub const kv_out: u32 = 1024;
pub const ffn: u32 = 40960;
pub const vocab: u32 = 100352;
pub const layers: u32 = 80;
/// `ffn_clamp_threshold`
pub const clamp: f32 = 20;
/// the pack's rates: K3 for every layer linear, K6 for the head
pub const n_linear: u32 = 48;
pub const n_head: u32 = 96;

// ── texts ──

/// the joined launches' member choice (the suh, trellis base and output tiles of the threadgroup's member) and the
/// GEMV's tile address in that member's trellis (in place of `dense.dec_tile`)
pub const jsel = @embedFile("kernels/ricursive/jsel.metal");
pub const jtile = @embedFile("kernels/ricursive/jtile.metal");
/// the value residual's blend as the prepare reads x, and its two uses: member 1 of a joined launch, every element
pub const blend_struct = @embedFile("kernels/ricursive/blend_struct.metal");
pub const blend_member = @embedFile("kernels/ricursive/blend_member.metal");
const blend_all = @embedFile("kernels/ricursive/blend_all.metal");
/// the clamped SwiGLU of the gate's and up's outputs as the down's prepare reads x
const swiglu_x = @embedFile("kernels/ricursive/swiglu_x.metal");

/// DEC-ONE with the prepare inside (`dense.text_dec1`) after its head: the prepare, the GEMV, the arrival and the finish,
/// the GEMV's tile read in the threadgroup's member's trellis (`jtile`)
const dec1_body = dense.dec_prep1 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive_1 ++ jtile ++ dense.dec_gemv_arrive_2 ++ dense.dec_xsum1 ++ dense.dec_finish_v ++ dense.dec_finish_store;
/// two members of one input x in one DEC-ONE launch: `dense.text_dec1` with the member chosen after its head
pub const text_jdec1: [:0]const u8 = dense.dec_head ++ jsel ++ dec1_body;
/// the same with member 1's input the value residual's blend of xa and x0
pub const text_jdec1_blend: [:0]const u8 = dense.dec_head ++ jsel ++ blend_struct ++ blend_member ++ dec1_body;
/// `dense.text_prep` (natural order, prefill rows) of the value residual's blend
pub const text_blend_prep: [:0]const u8 = blend_struct ++ blend_all ++ dense.text_prep;
/// `dense.text_prep_sc` (lane order, block sums, DEC-ONE's counters zeroed) of the clamped SwiGLU: the down's prepare
pub const text_swiglu_prep_sc: [:0]const u8 = swiglu_x ++ dense.text_prep_sc;
/// `dense.text_prep` (natural order) of the clamped SwiGLU: the down's prepare at prefill rows
pub const text_swiglu_prep: [:0]const u8 = swiglu_x ++ dense.text_prep;

const jdec1_in = [_][*:0]const u8{ "x", "suh0", "suh1", "w", "svh", "cnt" };
const jdec1_blend_in = [_][*:0]const u8{ "xa", "x0", "blend", "suh0", "suh1", "w", "svh", "cnt" };
const jdec1_out = [_][*:0]const u8{ "y", "z", "xsx" };
const blend_prep_in = [_][*:0]const u8{ "xa", "x0", "blend", "suh" };
const swiglu_in = [_][*:0]const u8{ "gate", "up", "sigtab", "lim", "suh" };
const prep_out = [_][*:0]const u8{"xp"};
const prep_sc_out = [_][*:0]const u8{ "xp", "xs", "cnt" };

// ── kernels: made at their first call ──

const Kind = enum(u8) { jdec1, jdec1_blend, blend_prep, swiglu_prep_sc, swiglu_prep };
const kernel_lock = Global(std.atomic.Value(u32), "ricursive_kernel_lock", std.atomic.Value(u32).init(0));
const kernels = Global([5]?mlx.mlx_fast_metal_kernel, "ricursive_kernels", @splat(null));

fn kernelOf(k: Kind) !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    const slot = &kernels.get().*[@intFromEnum(k)];
    if (slot.*) |made| return made;
    const made = switch (k) {
        .jdec1 => ks.newKernel("turboexl3_ric_jdec1", &jdec1_in, &jdec1_out, text_jdec1, decode.header5, true),
        .jdec1_blend => ks.newKernel("turboexl3_ric_jdec1_blend", &jdec1_blend_in, &jdec1_out, text_jdec1_blend, decode.header5, true),
        .blend_prep => ks.newKernel("turboexl3_ric_blend_prep", &blend_prep_in, &prep_out, text_blend_prep, decode.header, true),
        .swiglu_prep_sc => ks.newKernel("turboexl3_ric_swiglu_prep_sc", &swiglu_in, &prep_sc_out, text_swiglu_prep_sc, decode.header, true),
        .swiglu_prep => ks.newKernel("turboexl3_ric_swiglu_prep", &swiglu_in, &prep_out, text_swiglu_prep, decode.header, true),
    } orelse return error.KernelCreateFailed;
    slot.* = made;
    return made;
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernels };

// ── joined members ──

/// Two linears of one input joined: the trellis flat, member 0's [in/16, out0/16, n] then member 1's [in/16, out1/16, n]
/// (in/16 (out0 + out1)/16 n words), svh [out0 + out1] (member 0's columns first), each member's own suh [in], and the
/// members as views of the joint (`members`: trellis [in/16, out_m/16, n] a contiguous range of `trellis`, svh a range
/// of `svh`, the member's suh), so the joint is the one copy of its members' weights. Made once per layer at load
/// (`join`); `deinit` frees what it holds, the views included.
pub const Joint = struct {
    trellis: mlx.mlx_array,
    svh: mlx.mlx_array,
    suh0: mlx.mlx_array,
    suh1: mlx.mlx_array,
    members: [2]root.Proj,
    in: u32,
    out0: u32,
    out1: u32,
    n: u32,

    pub fn deinit(j: *Joint) void {
        for (j.members) |m| for ([_]mlx.mlx_array{ m.trellis, m.svh }) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for ([_]mlx.mlx_array{ j.trellis, j.svh, j.suh0, j.suh1 }) |a| _ = mlx.mlx_array_free(a);
        j.* = undefined;
    }
};

/// The view of `a` from element `off`, `shape` (a contiguous range of a row-contiguous array: no copy), evaluated
fn view(s: mlx.mlx_stream, a: mlx.mlx_array, off: usize, shape: []const c_int) !mlx.mlx_array {
    var size: c_int = 1;
    for (shape) |d| size *= d;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_slice(&flat, a, &[_]c_int{@intCast(off)}, 1, &[_]c_int{@as(c_int, @intCast(off)) + size}, 1, &[_]c_int{1}, 1, s));
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, flat, shape.ptr, shape.len, s));
    try mlx.check(mlx.mlx_array_eval(r));
    return r;
}

fn flat1(s: mlx.mlx_stream, a: mlx.mlx_array) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, a, &[_]c_int{@intCast(mlx.mlx_array_size(a))}, 1, s));
    return r;
}

fn concat2(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array, axis: c_int) !mlx.mlx_array {
    const both = [_]mlx.mlx_array{ a, b };
    const vec = mlx.mlx_vector_array_new_data(&both, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_concatenate_axis(&r, vec, axis, s));
    try mlx.check(mlx.mlx_array_eval(r));
    return r;
}

/// `a` and `b` (one input width, one rate, U16 or I16 trellises alike) joined, evaluated (`Joint`): the caller frees the
/// members' trellises and svh after, and runs each member through `Joint.members` (each suh is a new reference).
pub fn join(s: mlx.mlx_stream, a: root.Proj, b: root.Proj) !Joint {
    const ta = mlx.getShape(a.trellis);
    const tb = mlx.getShape(b.trellis);
    if (ta.len != 3 or tb.len != 3 or ta[0] != tb[0] or ta[2] != tb[2]) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(a.trellis) != mlx.mlx_array_dtype(b.trellis)) return error.BadExl3Dtype;
    const in: u32 = @intCast(ta[0] * 16);
    const out0: u32 = @intCast(ta[1] * 16);
    const out1: u32 = @intCast(tb[1] * 16);
    for ([_]root.Proj{ a, b }, [_]u32{ out0, out1 }) |p, o| {
        if (mlx.getShape(p.suh).len != 1 or mlx.getShape(p.suh)[0] != @as(c_int, @intCast(in))) return error.BadExl3Shape;
        if (mlx.getShape(p.svh).len != 1 or mlx.getShape(p.svh)[0] != @as(c_int, @intCast(o))) return error.BadExl3Shape;
        if (mlx.mlx_array_dtype(p.suh) != .float16 or mlx.mlx_array_dtype(p.svh) != .float16) return error.BadExl3Dtype;
    }
    const fa = try flat1(s, a.trellis);
    defer _ = mlx.mlx_array_free(fa);
    const fb = try flat1(s, b.trellis);
    defer _ = mlx.mlx_array_free(fb);
    const t = try concat2(s, fa, fb, 0);
    errdefer _ = mlx.mlx_array_free(t);
    const v = try concat2(s, a.svh, b.svh, 0);
    errdefer _ = mlx.mlx_array_free(v);
    var s0 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(s0);
    try mlx.check(mlx.mlx_array_set(&s0, a.suh));
    var s1 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(s1);
    try mlx.check(mlx.mlx_array_set(&s1, b.suh));
    const len0 = mlx.mlx_array_size(fa);
    const t0 = try view(s, t, 0, &.{ ta[0], ta[1], ta[2] });
    errdefer _ = mlx.mlx_array_free(t0);
    const t1 = try view(s, t, len0, &.{ tb[0], tb[1], tb[2] });
    errdefer _ = mlx.mlx_array_free(t1);
    const v0 = try view(s, v, 0, &.{@intCast(out0)});
    errdefer _ = mlx.mlx_array_free(v0);
    const v1 = try view(s, v, out0, &.{@intCast(out1)});
    return .{
        .trellis = t,
        .svh = v,
        .suh0 = s0,
        .suh1 = s1,
        .members = .{ .{ .trellis = t0, .suh = s0, .svh = v0 }, .{ .trellis = t1, .suh = s1, .svh = v1 } },
        .in = in,
        .out0 = out0,
        .out1 = out1,
        .n = @intCast(ta[2]),
    };
}

/// The joined launch's plan at `rows` rows, or null where it does not serve (the caller runs `linear` per member):
/// decode rows, the members' one K split, the prepare inside fitting threadgroup memory.
pub const JointPlan = struct { ks: u32, rb: u32, bt: u32, chunks: u32, ncnt: u32, nx: u32 };

pub fn jointPlan(j: Joint, dec: format.Decode, rows: usize) ?JointPlan {
    if (rows < 1 or rows > decode.Kernels.max_rows) return null;
    if (j.out0 % 128 != 0 or j.out1 % 128 != 0 or j.in % 128 != 0 or (j.in / 16) % 4 != 0) return null;
    const ksv = dense.Dense.ksFor(j.in, j.out0);
    if (dense.Dense.ksFor(j.in, j.out1) != ksv) return null;
    const rows_u: u32 = @intCast(rows);
    const rb = dense.Dense.rbFor(rows_u);
    const bt = dense.Dense.btFor(rb);
    const rate = format.Rate.fromN(j.n) orelse return null;
    const sh: dense.Shape = .{ .in = j.in, .out = j.out0 + j.out1, .rate = rate, .dec = dec, .out_dtype = .float32 };
    if (!dense.Dense.decPrepFits(sh, ksv, rb, bt)) return null;
    const chunks = (rows_u + rb - 1) / rb;
    const nob = (j.out0 + j.out1) / 128;
    const nb = j.in / 128;
    const defer_affine = dense.Dense.deferFor(dec);
    return .{
        .ks = ksv,
        .rb = rb,
        .bt = bt,
        .chunks = chunks,
        .ncnt = @max(nob * chunks, 8),
        .nx = if (defer_affine) @max(nob * chunks * rb * nb, 8) else 8,
    };
}

fn rows2(s: mlx.mlx_stream, x: mlx.mlx_array, in: u32) !struct { rows: usize, x2: mlx.mlx_array } {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh[xsh.len - 1] != @as(c_int, @intCast(in))) return error.BadExl3Shape;
    var rows: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows *= d;
    if (rows < 1) return error.BadExl3Shape;
    var x2 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, @intCast(in) }, 2, s));
    return .{ .rows = @intCast(rows), .x2 = x2 };
}

/// a one-element operand (the blend, the limit) as [1]: MLX binds a 0-d input as a scalar reference, not a pointer
fn one1(s: mlx.mlx_stream, a: mlx.mlx_array) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, a, &[_]c_int{1}, 1, s));
    return r;
}

fn actDtype(d: mlx.mlx_dtype) !void {
    if (d != .bfloat16 and d != .float16 and d != .float32) return error.BadExl3Dtype;
}

/// The joined launch: `dec_zero`'s counters (a launch that waits on nothing) then `text_jdec1` (or `_blend`)
fn jointRun(s: mlx.mlx_stream, kind: Kind, inputs: []const mlx.mlx_array, j: Joint, dec: format.Decode, rows: usize, xd: mlx.mlx_dtype, p: JointPlan) !mlx.mlx_array {
    return jointLaunch(s, try kernelOf(kind), kind == .jdec1_blend, inputs, j, dec, rows, xd, p);
}

/// `jointRun` with kernel `k` (`blend`: its inputs and template are `text_jdec1_blend`'s): the tests run the joint's
/// earlier text, over the members interleaved along OUT, through the same configs
pub fn jointLaunch(s: mlx.mlx_stream, k: mlx.mlx_fast_metal_kernel, blend: bool, inputs: []const mlx.mlx_array, j: Joint, dec: format.Decode, rows: usize, xd: mlx.mlx_dtype, p: JointPlan) !mlx.mlx_array {
    const rate = format.Rate.fromN(j.n) orelse return error.BadExl3Shape;
    // DEC-ONE's counter zeroing, dense's own kernel (the member shape's Dense: its kernels only)
    const d = try root.denseCached(.{ .in = j.in, .out = j.out0, .rate = rate, .dec = dec, .out_dtype = xd });
    const r: c_int = @intCast(rows);
    const out: u32 = j.out0 + j.out1;
    const ncnt_shape = [_]c_int{@intCast(p.ncnt)};
    const zcfg = try ks.buildConfig(.{ .grid = .{ ((p.ncnt + 127) / 128) * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&ncnt_shape}, .out_dtypes = &.{.uint32}, .template = &.{.{ .name = "NCNT", .value = .{ .int = @intCast(p.ncnt) } }} });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(zcfg);
    var cnt: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(d.dec_zero, zcfg, &.{j.svh}, &cnt, s);
    defer _ = mlx.mlx_array_free(cnt[0]);
    const tmpl = [_]ks.TemplateArg{
        .{ .name = "NHW", .value = .{ .int = @intCast(j.n) } },
        .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(dec.codebook) } },
        .{ .name = "WBITS", .value = .{ .int = @intCast(dec.window.bits()) } },
        .{ .name = "IN", .value = .{ .int = @intCast(j.in) } },
        .{ .name = "OUT", .value = .{ .int = @intCast(out) } },
        .{ .name = "KS", .value = .{ .int = @intCast(p.ks) } },
        .{ .name = "BT", .value = .{ .int = @intCast(p.bt) } },
        .{ .name = "RB", .value = .{ .int = @intCast(p.rb) } },
        .{ .name = "ROWS", .value = .{ .int = r } },
        .{ .name = "DEFER", .value = .{ .int = if (dense.Dense.deferFor(dec)) 1 else 0 } },
        .{ .name = "OUT_T", .value = .{ .dtype = xd } },
        .{ .name = "OB1", .value = .{ .int = @intCast(j.out0 / 128) } },
        .{ .name = "T", .value = .{ .dtype = xd } },
    };
    const cfg = try ks.buildConfig(.{
        .grid = .{ (out / 16 / p.bt) * 128, p.chunks, p.ks },
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ r, @intCast(out) }, &[_]c_int{ @as(c_int, @intCast(p.ks)) * r, @intCast(out) }, &[_]c_int{@intCast(p.nx)} },
        .out_dtypes = &.{ xd, .float32, .float32 },
        // the plain launch has no T: its x is the input itself
        .template = if (!blend) tmpl[0 .. tmpl.len - 1] else &tmpl,
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [3]mlx.mlx_array = undefined;
    var all: [10]mlx.mlx_array = undefined;
    @memcpy(all[0..inputs.len], inputs);
    all[inputs.len] = cnt[0];
    try ks.applyConfig(k, cfg, all[0 .. inputs.len + 1], &o, s);
    _ = mlx.mlx_array_free(o[1]);
    _ = mlx.mlx_array_free(o[2]);
    return o[0];
}

/// Two linears of x [..., in] in one launch: y [rows, out0 + out1] in x's dtype, member m's columns `linear`'s bytes of
/// member m. error.Exl3EngineNotImplemented where `jointPlan` declines, before any launch.
pub fn jointLinear(s: mlx.mlx_stream, x: mlx.mlx_array, j: Joint, dec: format.Decode) !mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x);
    try actDtype(xd);
    const v = try rows2(s, x, j.in);
    defer _ = mlx.mlx_array_free(v.x2);
    const p = jointPlan(j, dec, v.rows) orelse return error.Exl3EngineNotImplemented;
    return jointRun(s, .jdec1, &.{ v.x2, j.suh0, j.suh1, j.trellis, j.svh }, j, dec, v.rows, xd, p);
}

/// `jointLinear` with member 1's input Ripple's value residual T(x0 + blend (x - x0)): x and x0 [..., in] in one dtype
/// (bf16, f16 or f32), blend f32 of one element (sigmoid(lambda), 0-d or [1]); member 1's columns are the bytes of the host's blend then `linear`.
pub fn jointLinearBlend(s: mlx.mlx_stream, x: mlx.mlx_array, x0: mlx.mlx_array, blend: mlx.mlx_array, j: Joint, dec: format.Decode) !mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x);
    try actDtype(xd);
    if (mlx.mlx_array_dtype(x0) != xd) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(blend) != .float32 or mlx.mlx_array_size(blend) != 1) return error.BadExl3Shape;
    const v = try rows2(s, x, j.in);
    defer _ = mlx.mlx_array_free(v.x2);
    const v0 = try rows2(s, x0, j.in);
    defer _ = mlx.mlx_array_free(v0.x2);
    if (v0.rows != v.rows) return error.BadExl3Shape;
    const p = jointPlan(j, dec, v.rows) orelse return error.Exl3EngineNotImplemented;
    const b1 = try one1(s, blend);
    defer _ = mlx.mlx_array_free(b1);
    return jointRun(s, .jdec1_blend, &.{ v.x2, v0.x2, b1, j.suh0, j.suh1, j.trellis, j.svh }, j, dec, v.rows, xd, p);
}

/// v at prefill rows (more than DECODE_ROWS_MAX) with Ripple's value residual formed in its prepare: y [rows, out], the
/// bytes of the host's blend then `linear(v)`. Decode rows: `jointLinearBlend` (k | v); error.Exl3EngineNotImplemented.
pub fn valueLinear(s: mlx.mlx_stream, x: mlx.mlx_array, x0: mlx.mlx_array, blend: mlx.mlx_array, vp: root.Proj, dec: format.Decode) !mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x);
    try actDtype(xd);
    if (mlx.mlx_array_dtype(x0) != xd) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(blend) != .float32 or mlx.mlx_array_size(blend) != 1) return error.BadExl3Shape;
    const tsh = mlx.getShape(vp.trellis);
    if (tsh.len != 3) return error.BadExl3Shape;
    const in: u32 = @intCast(tsh[0] * 16);
    const v = try rows2(s, x, in);
    defer _ = mlx.mlx_array_free(v.x2);
    const v0 = try rows2(s, x0, in);
    defer _ = mlx.mlx_array_free(v0.x2);
    if (v0.rows != v.rows) return error.BadExl3Shape;
    if (v.rows <= decode.Kernels.max_rows) return error.Exl3EngineNotImplemented;
    const k = try kernelOf(.blend_prep);
    const r: c_int = @intCast(v.rows);
    const cfg = try ks.buildConfig(.{
        .grid = .{ (in / 128) * 32, @intCast(v.rows), 1 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{&[_]c_int{ r, @intCast(in) }},
        .out_dtypes = &.{.float16},
        .template = &.{
            .{ .name = "DIM", .value = .{ .int = @intCast(in) } },
            .{ .name = "LANE_ORDER", .value = .{ .int = 0 } },
            .{ .name = "T", .value = .{ .dtype = xd } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const b1 = try one1(s, blend);
    defer _ = mlx.mlx_array_free(b1);
    var xp: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(k, cfg, &.{ v.x2, v0.x2, b1, vp.suh }, &xp, s);
    defer _ = mlx.mlx_array_free(xp[0]);
    return root.linearPrepared(s, xp[0], vp, dec, xd);
}

// ── the MLP ──

/// The clamped SwiGLU's operands: the gate's and up's outputs (one joined array or two), the limit, the table.
const Gu = struct { gate: mlx.mlx_array, up: mlx.mlx_array, stride: u32, uoff: u32 };

/// Ripple's MLP of x [..., hidden]: down(clampedSwiGLU(gate(x), up(x), limit)) in x's dtype (bf16 or f16), [rows, out].
/// `gu` is gate | up joined (`join`): one launch where `jointLinear` serves, else `linear` of each member's view
/// (`gu.members`); `lim` x's dtype, one element (the limit, 20; 0-d or [1]);
/// `sigtab` x's dtype [65536] (MLX's sigmoid of every 16-bit pattern, the host's table). The bytes of `linear(gate)`,
/// `linear(up)`, the host's clamped SwiGLU and `linear(down)`. error.Exl3EngineNotImplemented where a step does not
/// serve (the down's prepare inside its DEC-ONE launch), before its launches.
pub fn mlp(s: mlx.mlx_stream, x: mlx.mlx_array, gu: Joint, down: root.Proj, lim: mlx.mlx_array, sigtab: mlx.mlx_array, dec: format.Decode) !mlx.mlx_array {
    const gate = gu.members[0];
    const up = gu.members[1];
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(lim) != xd or mlx.mlx_array_size(lim) != 1) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(sigtab) != xd or mlx.mlx_array_size(sigtab) != 1 << 16) return error.BadExl3Dtype;
    const dsh = mlx.getShape(down.trellis);
    if (dsh.len != 3) return error.BadExl3Shape;
    const f: u32 = @intCast(dsh[0] * 16);
    const out: u32 = @intCast(dsh[1] * 16);
    const rate = format.Rate.fromN(@intCast(dsh[2])) orelse return error.BadExl3Shape;
    const gsh = mlx.getShape(gate.trellis);
    if (gsh.len != 3) return error.BadExl3Shape;
    const in: u32 = @intCast(gsh[0] * 16);
    const v = try rows2(s, x, in);
    defer _ = mlx.mlx_array_free(v.x2);
    const rows = v.rows;
    const lim1 = try one1(s, lim);
    defer _ = mlx.mlx_array_free(lim1);
    const d = try root.denseCached(.{ .in = f, .out = out, .rate = rate, .dec = dec, .out_dtype = xd });
    const decode_rows = rows <= decode.Kernels.max_rows;
    // the down's DEC-ONE plan: its prepare must run in a launch of its own (ours)
    const dc: ?dense.Dense.DecCfg = if (decode_rows and d.decTakes(rows)) try d.decCfgs(rows) else null;
    if (dc) |c| if (c.prep_in) return error.Exl3EngineNotImplemented;
    // gate and up
    var held: [2]?mlx.mlx_array = .{ null, null };
    defer for (held) |h| {
        if (h) |a| _ = mlx.mlx_array_free(a);
    };
    var ops: Gu = undefined;
    if (gu.out0 != f or gu.out1 != f) return error.BadExl3Shape;
    if (decode_rows and jointPlan(gu, dec, rows) != null) {
        held[0] = try jointRun(s, .jdec1, &.{ v.x2, gu.suh0, gu.suh1, gu.trellis, gu.svh }, gu, dec, rows, xd, jointPlan(gu, dec, rows).?);
        ops = .{ .gate = held[0].?, .up = held[0].?, .stride = 2 * f, .uoff = f };
    } else {
        held[0] = try root.linear(s, v.x2, gate.trellis, gate.suh, gate.svh, dec);
        held[1] = try root.linear(s, v.x2, up.trellis, up.suh, up.svh, dec);
        ops = .{ .gate = held[0].?, .up = held[1].?, .stride = f, .uoff = 0 };
    }
    const r: c_int = @intCast(rows);
    const nb: u32 = f / 128;
    if (!decode_rows) {
        // prefill rows: the SwiGLU in the down's prepare (natural order), then its inner product and finish
        const k = try kernelOf(.swiglu_prep);
        const cfg = try ks.buildConfig(.{
            .grid = .{ nb * 32, @intCast(rows), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{&[_]c_int{ r, @intCast(f) }},
            .out_dtypes = &.{.float16},
            .template = &.{
                .{ .name = "DIM", .value = .{ .int = @intCast(f) } },
                .{ .name = "LANE_ORDER", .value = .{ .int = 0 } },
                .{ .name = "T", .value = .{ .dtype = xd } },
                .{ .name = "GSTRIDE", .value = .{ .int = @intCast(ops.stride) } },
                .{ .name = "UOFF", .value = .{ .int = @intCast(ops.uoff) } },
            },
        });
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        var xp: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, cfg, &.{ ops.gate, ops.up, sigtab, lim1, down.suh }, &xp, s);
        defer _ = mlx.mlx_array_free(xp[0]);
        return root.linearPrepared(s, xp[0], down, dec, xd);
    }
    // decode rows: the SwiGLU in the down's prepare with its block sums (lane order) and, for DEC-ONE, its counters
    const rows_u: u32 = @intCast(rows);
    const rb = dense.Dense.rbFor(rows_u);
    const chunks = (rows_u + rb - 1) / rb;
    const ncnt: u32 = if (dc != null) @max((out / 128) * chunks, 8) else 8;
    const k = try kernelOf(.swiglu_prep_sc);
    const cfg = try ks.buildConfig(.{
        .grid = .{ nb * 32, rows_u, 1 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ r, @intCast(f) }, &[_]c_int{ r, @intCast(nb) }, &[_]c_int{@intCast(ncnt)} },
        .out_dtypes = &.{ .float16, .float32, .uint32 },
        .template = &.{
            .{ .name = "DIM", .value = .{ .int = @intCast(f) } },
            .{ .name = "LANE_ORDER", .value = .{ .int = 1 } },
            .{ .name = "NCNT", .value = .{ .int = @intCast(ncnt) } },
            .{ .name = "NTH", .value = .{ .int = @intCast(nb * 32 * rows_u) } },
            .{ .name = "T", .value = .{ .dtype = xd } },
            .{ .name = "GSTRIDE", .value = .{ .int = @intCast(ops.stride) } },
            .{ .name = "UOFF", .value = .{ .int = @intCast(ops.uoff) } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var po: [3]mlx.mlx_array = undefined;
    try ks.applyConfig(k, cfg, &.{ ops.gate, ops.up, sigtab, lim1, down.suh }, &po, s);
    defer for (po) |a| {
        _ = mlx.mlx_array_free(a);
    };
    if (dc) |c| {
        var o: [2]mlx.mlx_array = undefined;
        try ks.applyConfig(d.dec0, c.gemv, &.{ po[0], po[1], down.trellis, down.svh, po[2] }, &o, s);
        _ = mlx.mlx_array_free(o[1]);
        return o[0];
    }
    return d.linearPrepared(s, po[0], po[1], down.trellis, down.svh, rows);
}

// ── CPU tests ──

const testing = std.testing;

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

/// a joint's shape alone (`jointPlan` reads no array)
fn fakeJoint(in: u32, out0: u32, out1: u32) Joint {
    const fake: mlx.mlx_array = .{ .ctx = null };
    const fp: root.Proj = .{ .trellis = fake, .suh = fake, .svh = fake };
    return .{ .trellis = fake, .svh = fake, .suh0 = fake, .suh1 = fake, .members = .{ fp, fp }, .in = in, .out0 = out0, .out1 = out1, .n = n_linear };
}

test "turboexl3 ricursive texts: dense's DEC-ONE and prepare statements verbatim around the member choice, the blend and the clamped SwiGLU" {
    // the joined launches are DEC-ONE with the prepare inside, the member's suh chosen after its head
    try testing.expect(std.mem.startsWith(u8, text_jdec1, dense.dec_head ++ jsel));
    try testing.expect(std.mem.endsWith(u8, text_jdec1, dense.dec_gemv_arrive_2 ++ dense.dec_xsum1 ++ dense.dec_finish_v ++ dense.dec_finish_store));
    // dense's DEC-ONE with the prepare inside but for its one tile address, which reads the member's trellis
    try testing.expectEqual(dense.text_dec1.len + jsel.len - dense.dec_tile.len + jtile.len, text_jdec1.len);
    try testing.expectEqualStrings(dense.dec_gemv_arrive, dense.dec_gemv_arrive_1 ++ dense.dec_tile ++ dense.dec_gemv_arrive_2);
    try testing.expectEqualStrings(dense.text_dec1, dense.dec_head ++ dense.dec_prep1 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive ++ dense.dec_xsum1 ++ dense.dec_finish_v ++ dense.dec_finish_store);
    try testing.expectEqual(@as(usize, 1), count(text_jdec1, jtile));
    try testing.expectEqual(@as(usize, 0), count(text_jdec1, dense.dec_tile));
    try testing.expectEqualStrings(text_jdec1_blend, dense.dec_head ++ jsel ++ blend_struct ++ blend_member ++ dec1_body);
    // the prepares are dense's, x read through the accessor
    try testing.expect(std.mem.endsWith(u8, text_swiglu_prep_sc, dense.text_prep_sc));
    try testing.expect(std.mem.endsWith(u8, text_swiglu_prep, dense.text_prep));
    try testing.expect(std.mem.endsWith(u8, text_blend_prep, dense.text_prep));
    // the pieces name only what their launches pass, and declare the one `suh` or `x` the statements after them read
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/ricursive_texts.metal"));
    try testing.expectEqual(@as(usize, 1), count(jsel, pin[0]));
    try testing.expectEqual(@as(usize, 1), count(blend_member, pin[1]));
    try testing.expectEqual(@as(usize, 1), count(blend_all, pin[1]));
    try testing.expectEqual(@as(usize, 1), count(swiglu_x, pin[2]));
    for (pin[5..8]) |n| try testing.expect(count(jsel, n) > 0);
    for (pin[8..11]) |n| try testing.expect(count(blend_member, n) > 0);
    for (pin[11..17]) |n| try testing.expect(count(swiglu_x, n) > 0);
    // dense's statements read x and suh as these pieces define them
    try testing.expectEqual(@as(usize, 1), count(dense.dec_prep1, pin[3]));
    try testing.expectEqual(@as(usize, 1), count(dense.text_prep, pin[4]));
    try testing.expectEqual(@as(usize, 1), count(dense.text_prep_s, pin[4]));
}

test "turboexl3 ricursive plan: q | g and gate | up join where the row chunks are at most 3 rows (1, 2, 3, 6), k | v at every decode row" {
    // the members' own K splits, which the joined launch keeps
    try testing.expectEqual(@as(u32, 1), dense.Dense.ksFor(hidden, q_out));
    try testing.expectEqual(@as(u32, 10), dense.Dense.ksFor(hidden, kv_out));
    try testing.expectEqual(@as(u32, 1), dense.Dense.ksFor(hidden, ffn));
    const qg = fakeJoint(hidden, q_out, q_out);
    const kv = fakeJoint(hidden, kv_out, kv_out);
    const gu = fakeJoint(hidden, ffn, ffn);
    for (1..17) |rows| {
        // the joined prepare holds RB rows of the K slice: 3 at most at q | g's and gate | up's K split of 1
        const small = dense.Dense.rbFor(@intCast(rows)) <= 3;
        try testing.expectEqual(small, jointPlan(qg, .mul1, rows) != null);
        try testing.expect(jointPlan(kv, .mul1, rows) != null);
        try testing.expectEqual(small, jointPlan(gu, .mul1, rows) != null);
        if (jointPlan(kv, .mul1, rows)) |p| try testing.expectEqual(@as(u32, 10), p.ks);
    }
    try testing.expect(jointPlan(qg, .mul1, 6) != null and jointPlan(qg, .mul1, 4) == null);
    try testing.expect(jointPlan(qg, .mul1, 17) == null);
    // members whose own K splits differ do not join (q with k)
    const qk = fakeJoint(hidden, q_out, kv_out);
    try testing.expect(jointPlan(qk, .mul1, 1) == null);
    // every Ripple shape is a width the dense kernels admit
    for ([_][2]u32{ .{ hidden, q_out }, .{ hidden, kv_out }, .{ q_out, hidden }, .{ hidden, ffn }, .{ ffn, hidden }, .{ hidden, vocab } }) |io| {
        try testing.expect(io[0] % 128 == 0 and io[1] % 128 == 0 and (io[0] / 16) % 4 == 0);
        const k = dense.Dense.ksFor(io[0], io[1]);
        try testing.expect(((io[0] / 16) / 4) % k == 0);
    }
}
