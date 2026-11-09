//! Ling-3.1-flash's KDA decode projections in one GEMV launch (`kdaProj`): a KDA layer's q|k|v (4096 -> 24576), its
//! forget gate f and its output gate g (4096 -> 8192 each), three projections of one input, each in its own buffers
//! (nothing joined at load) and each at its own `linear`'s K split (q|k|v KS 1, f and g KS 2). A call of 1 to 16 rows
//! is two launches:
//! - `turboexl3_ling_kdaproj_prep`: dense's `prep_sc` text with member m on the grid's z, preparing x with its own suh
//!   into rows m * ROWS.. of the joined prepared rows, and zeroing the GEMV's arrival counters;
//! - `turboexl3_ling_kdaproj_dec0`: dense's DEC-ONE text over prepared rows (`dense.text_dec0`) under a head that reads
//!   the threadgroup's member, K slice and block off the grid and gives the member its own widths, K split, arrivals
//!   and simdgroup span (`kdaproj/head`), with the member's prepared rows, trellis, svh, output, partial sums and
//!   arrival counters chosen after it (`kdaproj/member`).
//! A host that prepares the input itself (a norm launch that also writes its readers' prepared rows) calls the GEMV
//! alone (`kdaProjPrepared`) over that layout: the joined rows, their block sums and zeroed counters (`ncntFor`).
//! `kdaProjB` adds the layer's b_proj (bf16, 4096 -> 64) as a fourth member of the GEMV launch
//! (`turboexl3_ling_kdaproj_dec0b`: `kdaproj/bmember` before the head): the grid's threadgroups past the three
//! members' run MLX's gemv over each row (the kernel MLX's matmul picks for b's one-row product), so b's output is the
//! served one-row matmul's bytes at every row, and the three members' are `kdaProj`'s.
//! Member m's output is its own `linear`'s bytes at every row count: each output is the same products in the same order
//! (its K split's partial sums, summed in slice order by its block's last arrival), and neither the tiles per
//! threadgroup nor where the prepare runs moves a byte of it (dense's DEC-ONE forms all give the three-launch chain's).
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const root = @import("../../root.zig");

// ── texts ──

/// the prepare's member choice: threadgroup z's suh and its rows of the joined prepared rows
pub const kprep = @embedFile("kernels/kdaproj/prep.metal");
/// DEC-ONE's head over three members: the threadgroup's member, K slice and block, the member's widths and K split
pub const khead = @embedFile("kernels/kdaproj/head.metal");
/// the member's prepared rows, trellis, svh, output, partial sums and arrival counters
pub const kmember = @embedFile("kernels/kdaproj/member.metal");

/// `dense.text_prep_sc` (lane order, block sums, DEC-ONE's counters zeroed) for the three members in one launch
pub const text_kprep_sc: [:0]const u8 = kprep ++ dense.text_prep_sc;
/// `dense.text_dec0` with its head (`dense.dec_head`) in the three-member head and the member choice
pub const text_kdec0: [:0]const u8 = khead ++ kmember ++ dense.dec_gemv0 ++ dense.dec_gemv_arrive ++ dense.dec_xsum0 ++ dense.dec_finish_v ++ dense.dec_finish_store;
/// b_proj as a bf16 member: the grid's threadgroups from FB on run MLX's gemv over b's weight, each row of their chunk
pub const kbmember = kbmember_1 ++ kbmember_2;
/// `text_kdec0` with b_proj's threadgroups before its head (`kdaProjB`)
pub const text_kdec0b: [:0]const u8 = kbmember ++ text_kdec0;

const kprep_in = [_][*:0]const u8{ "x", "suh0", "suh1", "suh2" };
const kprep_out = [_][*:0]const u8{ "jxp", "jxs", "cnt" };
const kdec0_in = [_][*:0]const u8{ "jxp", "jxs", "w0", "w1", "w2", "svh0", "svh1", "svh2", "jcnt" };
const kdec0_out = [_][*:0]const u8{ "y0", "y1", "y2", "z0", "z1", "z2" };
const kdec0b_in = kdec0_in ++ [_][*:0]const u8{ "xb", "wb" };
const kdec0b_out = kdec0_out ++ [_][*:0]const u8{"yb"};

// ── kernels: made at their first call ──

const Kind = enum(u8) { kprep_sc, kdec0, kdec0b, kdec0bv, kdec0bv_t4, kdec0bv_pf, kdec0bv_pft4 };
const kernel_lock = Global(std.atomic.Value(u32), "ling_kdaproj_kernel_lock", std.atomic.Value(u32).init(0));
const kernels = Global([7]?mlx.mlx_fast_metal_kernel, "ling_kdaproj_kernels", @splat(null));

fn kernelOf(k: Kind) !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    const slot = &kernels.get().*[@intFromEnum(k)];
    if (slot.*) |made| return made;
    const made = switch (k) {
        .kprep_sc => ks.newKernel("turboexl3_ling_kdaproj_prep", &kprep_in, &kprep_out, text_kprep_sc, decode.header, true),
        .kdec0 => ks.newKernel("turboexl3_ling_kdaproj_dec0", &kdec0_in, &kdec0_out, text_kdec0, decode.header5, true),
        .kdec0b => ks.newKernel("turboexl3_ling_kdaproj_dec0b", &kdec0b_in, &kdec0b_out, text_kdec0b, decode.header5, true),
        // the variant launch (`gemvVariant`): its text, or pf's, each over header5 or t4's
        .kdec0bv => ks.newKernel("turboexl3_ling_kdaproj_dec0bv", &kdec0b_in, &kdec0b_out, text_kdec0bv, decode.header5, true),
        .kdec0bv_t4 => ks.newKernel("turboexl3_ling_kdaproj_dec0bv_t4", &kdec0b_in, &kdec0b_out, text_kdec0bv, dense.header5_t4, true),
        .kdec0bv_pf => ks.newKernel("turboexl3_ling_kdaproj_dec0bv_pf", &kdec0b_in, &kdec0b_out, text_kdec0bv_pf, decode.header5, true),
        .kdec0bv_pft4 => ks.newKernel("turboexl3_ling_kdaproj_dec0bv_pft4", &kdec0b_in, &kdec0b_out, text_kdec0bv_pf, dense.header5_t4, true),
    } orelse return error.KernelCreateFailed;
    slot.* = made;
    return made;
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernels, variant_said, variant_runs };

// ── the plan ──

/// The three members' shape: one input width and rate, each member's output width (the trellises [in/16, out_m/16, n])
pub const Shape = struct { in: u32, out: [3]u32, n: u32 };

/// A call's launch: member 0's K split and members 1 and 2's (each its own `linear`'s), DEC-ONE's row chunks
/// (`dense.Dense.rbFor`), the output tiles per threadgroup, the arrival counters and the GEMV's threadgroups.
pub const Plan = struct { ks0: u32, ks1: u32, rb: u32, bt: u32, chunks: u32, ncnt: u32, tgs: u32 };

/// The members as `linear` takes each (trellis U16 or I16 [in/16, out/16, n], suh F16 [in], svh F16 [out]): one input
/// width, one rate, one trellis dtype; else an error.
pub fn shapeOf(p: [3]root.Proj) !Shape {
    const t0 = mlx.getShape(p[0].trellis);
    if (t0.len != 3 or t0[0] < 1) return error.BadExl3Shape;
    const td = mlx.mlx_array_dtype(p[0].trellis);
    if (td != .uint16 and td != .int16) return error.BadExl3Dtype;
    var sh: Shape = .{ .in = @intCast(t0[0] * 16), .out = undefined, .n = @intCast(t0[2]) };
    for (p, &sh.out) |m, *o| {
        const t = mlx.getShape(m.trellis);
        if (t.len != 3 or t[0] != t0[0] or t[2] != t0[2] or t[1] < 1) return error.BadExl3Shape;
        if (mlx.mlx_array_dtype(m.trellis) != td) return error.BadExl3Dtype;
        o.* = @intCast(t[1] * 16);
        if (mlx.getShape(m.suh).len != 1 or mlx.getShape(m.suh)[0] != @as(c_int, @intCast(sh.in))) return error.BadExl3Shape;
        if (mlx.getShape(m.svh).len != 1 or mlx.getShape(m.svh)[0] != @as(c_int, @intCast(o.*))) return error.BadExl3Shape;
        if (mlx.mlx_array_dtype(m.suh) != .float16 or mlx.mlx_array_dtype(m.svh) != .float16) return error.BadExl3Dtype;
    }
    return sh;
}

/// What `plan` reads off each member's cached Dense at a call: its K split, whether its finish applies the affine
/// (`defer_affine`), and member 0's output tiles per threadgroup at the call's row chunk (`dense.Dense.btDec`)
pub const Facts = struct { ks: [3]u32, defer_affine: [3]bool, bt0: u32 };

/// The arrival counters a call of `rows` rows over `sh` zeroes before its GEMV: every member's output blocks times
/// DEC-ONE's row chunks (member m's after the members before it), at least 8.
pub fn ncntFor(sh: Shape, rows: usize) u32 {
    const rows_u: u32 = @intCast(rows);
    const rb = dense.Dense.rbFor(rows_u);
    return @max((sh.out[0] + sh.out[1] + sh.out[2]) / 128 * ((rows_u + rb - 1) / rb), 8);
}

/// The plan of a call of `rows` rows over members of shape `sh` with facts `f`, or null where it does not serve (the
/// caller runs `linear` per member): past the decode rows, an input or output width DEC-ONE does not take, members 1
/// and 2 at two K splits, the members at two affine deferrals.
pub fn planFor(sh: Shape, f: Facts, rows: usize) ?Plan {
    if (rows < 1 or rows > decode.Kernels.max_rows) return null;
    if (sh.in % 128 != 0 or (sh.in / 16) % 4 != 0) return null;
    for (sh.out) |o| if (o == 0 or o % 128 != 0) return null;
    if (f.ks[1] != f.ks[2]) return null;
    if (f.defer_affine[0] != f.defer_affine[1] or f.defer_affine[0] != f.defer_affine[2]) return null;
    for (f.ks) |k| if (k == 0 or (sh.in / 16) % (k * 4) != 0) return null;
    const rows_u: u32 = @intCast(rows);
    const rb = dense.Dense.rbFor(rows_u);
    // the tiles per threadgroup move no output byte; member 0's own count (the widest), within a lane's accumulators
    const bt = f.bt0;
    if (bt == 0 or 8 % bt != 0 or 2 * bt * rb > 32) return null;
    const gc = 8 / bt;
    const tgs = (sh.out[0] / 128) * gc * f.ks[0] + ((sh.out[1] + sh.out[2]) / 128) * gc * f.ks[1];
    return .{ .ks0 = f.ks[0], .ks1 = f.ks[1], .rb = rb, .bt = bt, .chunks = (rows_u + rb - 1) / rb, .ncnt = ncntFor(sh, rows), .tgs = tgs };
}

/// `planFor` on the members' cached Dense (`root.denseCached` at `dec`, output in `xd`)
pub fn plan(sh: Shape, dec: format.Decode, rows: usize, xd: mlx.mlx_dtype) !?Plan {
    if (rows < 1 or rows > decode.Kernels.max_rows or sh.in % 128 != 0) return null;
    const rate = format.Rate.fromN(sh.n) orelse return null;
    var f: Facts = .{ .ks = undefined, .defer_affine = undefined, .bt0 = 0 };
    for (sh.out, 0..) |o, m| {
        if (o == 0 or o % 128 != 0) return null;
        const d = try root.denseCached(.{ .in = sh.in, .out = o, .rate = rate, .dec = dec, .out_dtype = xd });
        f.ks[m] = d.ks_split;
        f.defer_affine[m] = d.defer_affine;
        if (m == 0) f.bt0 = d.btDec(rows, dense.Dense.rbFor(@intCast(rows)));
    }
    return planFor(sh, f, rows);
}

// ── the calls ──

/// A KDA layer's three projections of x [..., in] (bf16, f16 or f32), `p` = q|k|v, f, g as `linear` takes each: one
/// prepare launch, then one GEMV launch; each member's output [rows, out_m] in x's dtype, its own `linear`'s bytes; the
/// caller frees the three. Refusals before any launch: the members' shapes and dtypes as `shapeOf` checks them,
/// error.Exl3EngineNotImplemented where `plan` declines (the caller runs `linear` per member).
pub fn kdaProj(s: mlx.mlx_stream, x: mlx.mlx_array, p: [3]root.Proj, dec: format.Decode) ![3]mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return error.BadExl3Dtype;
    const sh = try shapeOf(p);
    const v = try root.rowsOf(x, sh.in, s);
    defer root.freeArr(v.x2);
    const pl = (try plan(sh, dec, v.rows, xd)) orelse return error.Exl3EngineNotImplemented;
    const pre = try prepare(s, v.x2, p, sh, v.rows, pl);
    defer for (pre) |arr| root.freeArr(arr);
    return gemv(s, pre[0], pre[1], pre[2], p, sh, dec, v.rows, xd, pl);
}

/// The width of b_proj's output where `kdaProjB` takes its weight `wb` beside members of shape `sh` for x in `xd`, else
/// null: x and w bf16, w [OUTB, in] (the host's loaded [in, OUTB] column-major view transposed back), OUTB a multiple
/// of 4 below 4096, in a multiple of 1024 and at least 16 OUTB. There MLX's matmul runs b's one-row product as its gemv
/// at BM 1, BN 8 over whole 1024-blocks, the statements `kdaproj/bmember` holds.
pub fn bTakes(sh: Shape, xd: mlx.mlx_dtype, wb: mlx.mlx_array) ?u32 {
    if (xd != .bfloat16 or mlx.mlx_array_dtype(wb) != .bfloat16) return null;
    const ws = mlx.getShape(wb);
    if (ws.len != 2 or ws[0] < 4 or ws[1] != @as(c_int, @intCast(sh.in))) return null;
    const ob: u32 = @intCast(ws[0]);
    if (ob % 4 != 0 or ob >= 4096 or sh.in % 1024 != 0 or sh.in < 16 * ob) return null;
    return ob;
}

/// `kdaProj` with the layer's b_proj as a fourth member of its GEMV launch (`turboexl3_ling_kdaproj_dec0b`): b's weight
/// `wb` bf16 [OUTB, in] row-major (`bTakes`), x bf16 [..., in]. b's output [rows, OUTB] bf16 is MLX's matmul's bytes of
/// each row (the served one-row matmul's, and so each row of the exact verify's denseRows); the three members' outputs
/// are `kdaProj`'s. The caller frees the four. Refusals before any launch: `kdaProj`'s, and
/// error.Exl3EngineNotImplemented where `bTakes` does not hold (the caller runs `kdaProj` and b_proj apart). Where the
/// configuration asks a variant of the launch (`variantFor`: Config.kda_bfirst, kda_bt0 / kda_bt1, dec_variant), the
/// variant launch runs instead (`turboexl3_ling_kdaproj_dec0bv` and its t4 and pf texts), every output the same bytes.
pub fn kdaProjB(s: mlx.mlx_stream, x: mlx.mlx_array, p: [3]root.Proj, wb: mlx.mlx_array, dec: format.Decode) ![4]mlx.mlx_array {
    return kdaProjBWith(s, x, p, wb, dec, config.get().*);
}

/// `kdaProjB` under configuration `c`: the launch `variantFor(c, ...)` picks (the tests' and the timing arms' seam)
pub fn kdaProjBWith(s: mlx.mlx_stream, x: mlx.mlx_array, p: [3]root.Proj, wb: mlx.mlx_array, dec: format.Decode, c: config.Config) ![4]mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return error.BadExl3Dtype;
    const sh = try shapeOf(p);
    const ob = bTakes(sh, xd, wb) orelse return error.Exl3EngineNotImplemented;
    const v = try root.rowsOf(x, sh.in, s);
    defer root.freeArr(v.x2);
    const pl = (try plan(sh, dec, v.rows, xd)) orelse return error.Exl3EngineNotImplemented;
    const pre = try prepare(s, v.x2, p, sh, v.rows, pl);
    defer for (pre) |arr| root.freeArr(arr);
    return gemvWith(s, pre[0], pre[1], pre[2], p, sh, dec, v.rows, xd, pl, .{ .x = v.x2, .w = wb, .out = ob, .v = variantFor(c, pl, v.rows) });
}

/// The prepare launch (`turboexl3_ling_kdaproj_prep`) over x's rows x2 [rows, in]: the joined prepared rows, their
/// block sums and the GEMV's zeroed counters; the caller frees the three
fn prepare(s: mlx.mlx_stream, x2: mlx.mlx_array, p: [3]root.Proj, sh: Shape, rows: usize, pl: Plan) ![3]mlx.mlx_array {
    const nb = sh.in / 128;
    const r: c_int = @intCast(rows);
    const rows_u: u32 = @intCast(rows);
    // dense's prep_sc for each member, the members on the grid's z; the GEMV's counters zeroed
    const pcfg = try ks.buildConfig(.{
        .grid = .{ nb * 32, rows_u, 3 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ 3 * r, @intCast(sh.in) }, &[_]c_int{ 3 * r, @intCast(nb) }, &[_]c_int{@intCast(pl.ncnt)} },
        .out_dtypes = &.{ .float16, .float32, .uint32 },
        .template = &.{
            .{ .name = "DIM", .value = .{ .int = @intCast(sh.in) } },
            .{ .name = "LANE_ORDER", .value = .{ .int = 1 } },
            .{ .name = "NCNT", .value = .{ .int = @intCast(pl.ncnt) } },
            .{ .name = "NTH", .value = .{ .int = @intCast(nb * 32 * rows_u) } },
            .{ .name = "ROWS", .value = .{ .int = r } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(pcfg);
    var pre: [3]mlx.mlx_array = @splat(.{ .ctx = null });
    try ks.applyConfig(try kernelOf(.kprep_sc), pcfg, &.{ x2, p[0].suh, p[1].suh, p[2].suh }, &pre, s);
    return pre;
}

/// `kdaProj`'s GEMV alone over an input the host prepared: `jxp` f16 [3 * rows, in] (member m's rows m * rows.., each
/// the values and lane order its own prep_sc writes), `jxs` f32 [3 * rows, in / 128] (their 128-block sums at the same
/// rows), `cnt` u32 [ncntFor(shape, rows)] zeroed; each member's output [rows, out_m] in `xd`, its own `linear`'s bytes
/// on that input. The refusals are `kdaProj`'s, and the prepared arrays' shapes and dtypes.
pub fn kdaProjPrepared(s: mlx.mlx_stream, jxp: mlx.mlx_array, jxs: mlx.mlx_array, cnt: mlx.mlx_array, rows: usize, p: [3]root.Proj, dec: format.Decode, xd: mlx.mlx_dtype) ![3]mlx.mlx_array {
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return error.BadExl3Dtype;
    const sh = try shapeOf(p);
    const pl = (try plan(sh, dec, rows, xd)) orelse return error.Exl3EngineNotImplemented;
    const r: c_int = @intCast(rows);
    if (!std.mem.eql(c_int, mlx.getShape(jxp), &.{ 3 * r, @intCast(sh.in) }) or mlx.mlx_array_dtype(jxp) != .float16) return error.BadExl3Shape;
    if (!std.mem.eql(c_int, mlx.getShape(jxs), &.{ 3 * r, @intCast(sh.in / 128) }) or mlx.mlx_array_dtype(jxs) != .float32) return error.BadExl3Shape;
    if (mlx.getShape(cnt).len != 1 or mlx.getShape(cnt)[0] < @as(c_int, @intCast(pl.ncnt)) or mlx.mlx_array_dtype(cnt) != .uint32) return error.BadExl3Shape;
    return gemv(s, jxp, jxs, cnt, p, sh, dec, rows, xd, pl);
}

fn gemv(s: mlx.mlx_stream, jxp: mlx.mlx_array, jxs: mlx.mlx_array, cnt: mlx.mlx_array, p: [3]root.Proj, sh: Shape, dec: format.Decode, rows: usize, xd: mlx.mlx_dtype, pl: Plan) ![3]mlx.mlx_array {
    const o = try gemvWith(s, jxp, jxs, cnt, p, sh, dec, rows, xd, pl, null);
    return .{ o[0], o[1], o[2] };
}

/// b_proj's operands in the GEMV launch: x's rows [rows, in], its weight [out, in] (`bTakes`) and its width; and the
/// variant launch the call takes (`variantFor`; null: the served launch)
const BMember = struct { x: mlx.mlx_array, w: mlx.mlx_array, out: u32, v: ?Variant = null };

/// The GEMV launch over the three members, and with `b` b_proj's threadgroups after theirs (`kdaProjB`): the members'
/// outputs, then b's [rows, out] in `xd` (null without `b`)
fn gemvWith(s: mlx.mlx_stream, jxp: mlx.mlx_array, jxs: mlx.mlx_array, cnt: mlx.mlx_array, p: [3]root.Proj, sh: Shape, dec: format.Decode, rows: usize, xd: mlx.mlx_dtype, pl: Plan, b: ?BMember) ![4]mlx.mlx_array {
    const rate = format.Rate.fromN(sh.n) orelse return error.BadExl3Shape;
    const d0 = try root.denseCached(.{ .in = sh.in, .out = sh.out[0], .rate = rate, .dec = dec, .out_dtype = xd });
    const r: c_int = @intCast(rows);
    // dense's DEC-ONE GEMV for the three members, each at its own widths and K split
    const t = d0.templ();
    const tmpl = [_]ks.TemplateArg{
        t[0], t[1], t[2],
        .{ .name = "IN", .value = .{ .int = @intCast(sh.in) } },
        .{ .name = "OUT0", .value = .{ .int = @intCast(sh.out[0]) } },
        .{ .name = "OUT1", .value = .{ .int = @intCast(sh.out[1]) } },
        .{ .name = "OUT2", .value = .{ .int = @intCast(sh.out[2]) } },
        .{ .name = "KS0", .value = .{ .int = @intCast(pl.ks0) } },
        .{ .name = "KS1", .value = .{ .int = @intCast(pl.ks1) } },
        .{ .name = "BT", .value = .{ .int = @intCast(pl.bt) } },
        .{ .name = "RB", .value = .{ .int = @intCast(pl.rb) } },
        .{ .name = "ROWS", .value = .{ .int = r } },
        .{ .name = "DEFER", .value = .{ .int = if (d0.defer_affine) 1 else 0 } },
        .{ .name = "OUT_T", .value = .{ .dtype = xd } },
        // b_proj's (`kdaProjB`): its first threadgroup and its width
        .{ .name = "FB", .value = .{ .int = @intCast(pl.tgs) } },
        .{ .name = "OUTB", .value = .{ .int = if (b) |m| @intCast(m.out) else 0 } },
    };
    const k0: c_int = @intCast(pl.ks0);
    const k1: c_int = @intCast(pl.ks1);
    const o0: c_int = @intCast(sh.out[0]);
    const o1: c_int = @intCast(sh.out[1]);
    const o2: c_int = @intCast(sh.out[2]);
    if (b) |m| if (m.v) |vv| return gemvVariant(s, jxp, jxs, cnt, p, sh, d0, rows, xd, pl, m, vv);
    if (b) |m| {
        // the members' threadgroups, then b's: OUTB / 4 a row chunk (MLX's gemv's four outputs a threadgroup)
        const ob: c_int = @intCast(m.out);
        const cfg = try ks.buildConfig(.{
            .grid = .{ (pl.tgs + m.out / 4) * 128, pl.chunks, 1 },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ r, o0 }, &[_]c_int{ r, o1 }, &[_]c_int{ r, o2 }, &[_]c_int{ k0 * r, o0 }, &[_]c_int{ k1 * r, o1 }, &[_]c_int{ k1 * r, o2 }, &[_]c_int{ r, ob } },
            .out_dtypes = &.{ xd, xd, xd, .float32, .float32, .float32, xd },
            .template = &tmpl,
        });
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        var o: [7]mlx.mlx_array = undefined;
        try ks.applyConfig(try kernelOf(.kdec0b), cfg, &.{ jxp, jxs, p[0].trellis, p[1].trellis, p[2].trellis, p[0].svh, p[1].svh, p[2].svh, cnt, m.x, m.w }, &o, s);
        for (o[3..6]) |zm| _ = mlx.mlx_array_free(zm);
        return .{ o[0], o[1], o[2], o[6] };
    }
    const cfg = try ks.buildConfig(.{
        .grid = .{ pl.tgs * 128, pl.chunks, 1 },
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ r, o0 }, &[_]c_int{ r, o1 }, &[_]c_int{ r, o2 }, &[_]c_int{ k0 * r, o0 }, &[_]c_int{ k1 * r, o1 }, &[_]c_int{ k1 * r, o2 } },
        .out_dtypes = &.{ xd, xd, xd, .float32, .float32, .float32 },
        .template = tmpl[0 .. tmpl.len - 2],
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [6]mlx.mlx_array = undefined;
    try ks.applyConfig(try kernelOf(.kdec0), cfg, &.{ jxp, jxs, p[0].trellis, p[1].trellis, p[2].trellis, p[0].svh, p[1].svh, p[2].svh, cnt }, &o, s);
    for (o[3..]) |zm| _ = mlx.mlx_array_free(zm);
    return .{ o[0], o[1], o[2], .{ .ctx = null } };
}

// ── the variant launch (D35: Config.kda_bfirst, kda_bt0 / kda_bt1, dec_variant), each output the b launch's bytes ──
//
// `kdaProjB`'s launch with b's threadgroups first (`kda_bfirst`: they run beside the members' instead of after them, each
// row of a chunk on its own simdgroup), member 0's and members 1 and 2's output tiles a threadgroup apart (`kda_bt0`,
// `kda_bt1`, at a verify's rows: at one BT a threadgroup of f or g, at KS 2, decodes half the tiles of q|k|v's, at KS
// 1; BT 1 for q|k|v and 2 for f and g gives each threadgroup the same work), and DEC-ONE's GEMV
// variants (`dense.dec_variants`: t4, pf). Each member's threadgroup runs DEC-ONE's statements over its prepared rows
// under its own BTC (a branch on the threadgroup's member; one branch at one BT), so its outputs are its `linear`'s as
// the served launch's are; b's rows run kdaproj/bmember_2's statements whichever threadgroup and simdgroup run them.

const config = @import("../../core/config.zig");

/// `kbmember`'s pieces: b's threadgroups and rows (from FB on, simdgroup 0 alone: each row of the chunk in turn), then
/// MLX's gemv statements over one row, its store and the closes
pub const kbmember_1 = @embedFile("kernels/kdaproj/bmember_1.metal");
pub const kbmember_2 = @embedFile("kernels/kdaproj/bmember_2.metal");
/// The variant launch's pieces: b's threadgroups and rows under BF (in `kbmember_1`'s place); the head with each
/// member's own tiles a threadgroup and the members' threadgroups after b's under BF (in `khead`'s place); and the
/// branch on the threadgroup's member that runs DEC-ONE's statements under that member's BTC
pub const kbfirst = @embedFile("kernels/kdaproj/bfirst.metal");
pub const khead_v = @embedFile("kernels/kdaproj/head_v.metal");
pub const kbt_0 = @embedFile("kernels/kdaproj/bt_0.metal");
pub const kbt_1 = @embedFile("kernels/kdaproj/bt_1.metal");
pub const kbt_close = @embedFile("kernels/kdaproj/bt_close.metal");
/// DEC-ONE's statements over prepared rows after the member choice (`text_kdec0` after `khead` ++ `kmember`)
pub const kbody = dense.dec_gemv0 ++ dense.dec_gemv_arrive ++ dense.dec_xsum0 ++ dense.dec_finish_v ++ dense.dec_finish_store;
/// The variant launch (`gemvVariant`): b's threadgroups and rows under BF, then the variant head, the member choice and
/// DEC-ONE's statements in each member's branch
pub const text_kdec0bv: [:0]const u8 = kbfirst ++ kbmember_2 ++ khead_v ++ kmember ++ kbt_0 ++ kbody ++ kbt_1 ++ kbody ++ kbt_close;
/// `text_kdec0bv` with pf's statements in each branch (`dense.dec_body0_pf`: each input tile's words a pass ahead)
pub const text_kdec0bv_pf: [:0]const u8 = kbfirst ++ kbmember_2 ++ khead_v ++ kmember ++ kbt_0 ++ dense.dec_body0_pf ++ kbt_1 ++ dense.dec_body0_pf ++ kbt_close;

/// A call's variant launch (`variantFor`): b's threadgroups first, member 0's and members 1 and 2's output tiles a
/// threadgroup, DEC-ONE's GEMV variant (`dense.dec_variants`, 0 DEC-ONE's own)
pub const Variant = struct { bfirst: bool, bt0: u32, bt1: u32, dv: u8 };

/// Whether a member's chunks of `rb` rows take `bt` output tiles a threadgroup: 1, 2, 4 or 8, its 2 BT RB accumulators
/// within 32 floats a lane
fn btFits(bt: u8, rb: u32) bool {
    return bt != 0 and 8 % bt == 0 and 2 * @as(u32, bt) * rb <= 32;
}

/// The variant launch `c` asks of a `kdaProjB` call of `rows` rows on plan `pl`, null where it asks none (the served
/// launch): `kda_bt0` and `kda_bt1` at 2 rows or more where they fit (a tile count unset, at one row or past the chunk's
/// room keeps the plan's), `kda_bfirst` and `dec_variant` at every row count
pub fn variantFor(c: config.Config, pl: Plan, rows: usize) ?Variant {
    const verify = rows >= 2;
    const bt0: u32 = if (verify and btFits(c.kda_bt0, pl.rb)) c.kda_bt0 else pl.bt;
    const bt1: u32 = if (verify and btFits(c.kda_bt1, pl.rb)) c.kda_bt1 else pl.bt;
    const dv: u8 = if (c.dec_variant < dense.dec_variants.len) c.dec_variant else 0;
    if (!c.kda_bfirst and bt0 == pl.bt and bt1 == pl.bt and dv == 0) return null;
    return .{ .bfirst = c.kda_bfirst, .bt0 = bt0, .bt1 = bt1, .dv = dv };
}

/// The members' threadgroups of plan `pl` with member 0 at `bt0` output tiles a threadgroup and members 1 and 2 at `bt1`
/// (the plan's `tgs` at its own BT for both)
pub fn tgsAt(sh: Shape, pl: Plan, bt0: u32, bt1: u32) u32 {
    return (sh.out[0] / 128) * (8 / bt0) * pl.ks0 + ((sh.out[1] + sh.out[2]) / 128) * (8 / bt1) * pl.ks1;
}

/// The variant launch's engagement lines: bit 0 at its first one-row call, bit 1 at its first of 2 rows or more
const variant_said = Global(u8, "ling_kdaproj_variant_said", 0);
/// The variant launches this process has run (`variantRuns`)
const variant_runs = Global(u64, "ling_kdaproj_variant_runs", 0);

fn sayVariant(v: Variant, rows: usize) void {
    const bit: u8 = if (rows >= 2) 2 else 1;
    if ((variant_said.get().* & bit) != 0) return;
    variant_said.get().* |= bit;
    std.debug.print("[turboexl3] kda proj variant engaged: rows={d} bfirst={s} bt0={d} bt1={d} gemv={s}\n", .{ rows, if (v.bfirst) "on" else "off", v.bt0, v.bt1, if (v.dv == 0) "dec_one" else dense.dec_variants[v.dv].name });
}

/// The variant launches this process has run (the tests' check that an arm ran it)
pub fn variantRuns() u64 {
    return variant_runs.get().*;
}

/// The variant launch (`text_kdec0bv`, or `text_kdec0bv_pf` under pf; over header5, or t4's under t4): member 0 at
/// `v.bt0` output tiles a threadgroup and members 1 and 2 at `v.bt1`, b's threadgroups first under `v.bfirst`; the four
/// outputs as `gemvWith`'s b launch returns them
fn gemvVariant(s: mlx.mlx_stream, jxp: mlx.mlx_array, jxs: mlx.mlx_array, cnt: mlx.mlx_array, p: [3]root.Proj, sh: Shape, d0: *const dense.Dense, rows: usize, xd: mlx.mlx_dtype, pl: Plan, m: BMember, v: Variant) ![4]mlx.mlx_array {
    const r: c_int = @intCast(rows);
    const tgs = tgsAt(sh, pl, v.bt0, v.bt1);
    const t = d0.templ();
    const tmpl = [_]ks.TemplateArg{
        t[0], t[1], t[2],
        .{ .name = "IN", .value = .{ .int = @intCast(sh.in) } },
        .{ .name = "OUT0", .value = .{ .int = @intCast(sh.out[0]) } },
        .{ .name = "OUT1", .value = .{ .int = @intCast(sh.out[1]) } },
        .{ .name = "OUT2", .value = .{ .int = @intCast(sh.out[2]) } },
        .{ .name = "KS0", .value = .{ .int = @intCast(pl.ks0) } },
        .{ .name = "KS1", .value = .{ .int = @intCast(pl.ks1) } },
        .{ .name = "BT0", .value = .{ .int = @intCast(v.bt0) } },
        .{ .name = "BT1", .value = .{ .int = @intCast(v.bt1) } },
        .{ .name = "RB", .value = .{ .int = @intCast(pl.rb) } },
        .{ .name = "ROWS", .value = .{ .int = r } },
        .{ .name = "DEFER", .value = .{ .int = if (d0.defer_affine) 1 else 0 } },
        .{ .name = "OUT_T", .value = .{ .dtype = xd } },
        // b's: the members' threadgroups (b's first at BF 0), b's width, and b first (BF 1)
        .{ .name = "FB", .value = .{ .int = @intCast(tgs) } },
        .{ .name = "OUTB", .value = .{ .int = @intCast(m.out) } },
        .{ .name = "BF", .value = .{ .int = if (v.bfirst) 1 else 0 } },
    };
    const k0: c_int = @intCast(pl.ks0);
    const k1: c_int = @intCast(pl.ks1);
    const o0: c_int = @intCast(sh.out[0]);
    const o1: c_int = @intCast(sh.out[1]);
    const o2: c_int = @intCast(sh.out[2]);
    const ob: c_int = @intCast(m.out);
    const cfg = try ks.buildConfig(.{
        .grid = .{ (tgs + m.out / 4) * 128, pl.chunks, 1 },
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ r, o0 }, &[_]c_int{ r, o1 }, &[_]c_int{ r, o2 }, &[_]c_int{ k0 * r, o0 }, &[_]c_int{ k1 * r, o1 }, &[_]c_int{ k1 * r, o2 }, &[_]c_int{ r, ob } },
        .out_dtypes = &.{ xd, xd, xd, .float32, .float32, .float32, xd },
        .template = &tmpl,
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const dv = dense.dec_variants[v.dv];
    const kind: Kind = if (dv.pf) (if (dv.t4) .kdec0bv_pft4 else .kdec0bv_pf) else (if (dv.t4) .kdec0bv_t4 else .kdec0bv);
    var o: [7]mlx.mlx_array = undefined;
    try ks.applyConfig(try kernelOf(kind), cfg, &.{ jxp, jxs, p[0].trellis, p[1].trellis, p[2].trellis, p[0].svh, p[1].svh, p[2].svh, cnt, m.x, m.w }, &o, s);
    for (o[3..6]) |zm| _ = mlx.mlx_array_free(zm);
    variant_runs.get().* += 1;
    sayVariant(v, rows);
    return .{ o[0], o[1], o[2], o[6] };
}

// ── tests (CPU) ──

const testing = std.testing;

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

test "turboexl3 ling kdaproj texts: dense's prep_sc verbatim after the member choice; dense's DEC-ONE text over prepared rows verbatim after its head, under the three-member head and the member choice; each operand bound by name" {
    try testing.expectEqualStrings(text_kprep_sc, kprep ++ dense.text_prep_sc);
    try testing.expectEqualStrings(dense.text_dec0[dense.dec_head.len..], text_kdec0[khead.len + kmember.len ..]);
    try testing.expect(std.mem.startsWith(u8, text_kdec0, khead ++ kmember ++ dense.dec_gemv0));
    try testing.expectEqual(@as(usize, 0), count(text_kdec0, dense.dec_head));
    for (kprep_in ++ kprep_out) |n| try testing.expect(count(text_kprep_sc, std.mem.span(n)) > 0);
    for (kdec0_in ++ kdec0_out) |n| try testing.expect(count(text_kdec0, std.mem.span(n)) > 0);
    for ([_][]const u8{ "DIM", "LANE_ORDER", "NCNT", "NTH", "ROWS" }) |n| try testing.expect(count(text_kprep_sc, n) > 0);
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "IN", "OUT0", "OUT1", "OUT2", "KS0", "KS1", "BT", "RB", "ROWS", "DEFER", "OUT_T" }) |n| try testing.expect(count(text_kdec0, n) > 0);
    // the head and the member choice name everything dense's DEC-ONE body reads off dense's head and its operands
    for ([_][]const u8{ "NW", "WMASK", "INC", "OUTC", "IN_TILES", "OUT_TILES", "KSC", "BTC", "RBC", "ROWSC", "NCHC", "GC", "ARR", "PER", "HAD_SCALE", "blk", "chunk", "r0", "ksl", "tid", "red", "arrived_last" }) |n| try testing.expect(count(khead, n) > 0);
    for ([_][]const u8{ "xp", "xs", "svh", "cnt", "zc" }) |n| try testing.expect(count(kmember, n) > 0);
    // MLX adds a shape / strides / ndim buffer for an input name followed by those suffixes
    for ([_][:0]const u8{ text_kprep_sc, text_kdec0 }) |txt| for (kprep_in ++ kprep_out ++ kdec0_in ++ kdec0_out) |n| for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| {
        var buf: [24]u8 = undefined;
        try testing.expectEqual(@as(usize, 0), count(txt, std.fmt.bufPrint(&buf, "{s}{s}", .{ std.mem.span(n), suf }) catch unreachable));
    };
}

/// Ling-3.1-flash's KDA projections: q|k|v 4096 -> 24576, f and g 4096 -> 8192, K6
const ling_shape: Shape = .{ .in = 4096, .out = .{ 24576, 8192, 8192 }, .n = 96 };

test "turboexl3 ling kdaproj plan: Ling's members keep their own K splits (q|k|v 1, f and g 2) at decode rows 1 to 16; past the decode rows, two K splits on f and g, two affine deferrals and widths DEC-ONE does not take decline" {
    try testing.expectEqual(@as(u32, 1), dense.Dense.ksFor(4096, 24576));
    try testing.expectEqual(@as(u32, 2), dense.Dense.ksFor(4096, 8192));
    const f: Facts = .{ .ks = .{ 1, 2, 2 }, .defer_affine = .{ true, true, true }, .bt0 = 4 };
    for (1..decode.Kernels.max_rows + 1) |rows| {
        const rb = dense.Dense.rbFor(@intCast(rows));
        for ([_]u32{ 1, 2, 4, 8 }) |bt| {
            const fb: Facts = .{ .ks = f.ks, .defer_affine = f.defer_affine, .bt0 = bt };
            if (2 * bt * rb > 32) {
                try testing.expect(planFor(ling_shape, fb, rows) == null);
                continue;
            }
            const p = planFor(ling_shape, fb, rows).?;
            try testing.expectEqual(@as(u32, 1), p.ks0);
            try testing.expectEqual(@as(u32, 2), p.ks1);
            try testing.expectEqual(rb, p.rb);
            try testing.expectEqual(bt, p.bt);
            const gc = 8 / bt;
            // member 0's 192 128-blocks at one K slice, members 1 and 2's 64 each at two
            try testing.expectEqual(192 * gc + 2 * 64 * gc * 2, p.tgs);
            try testing.expectEqual((@as(u32, @intCast(rows)) + rb - 1) / rb, p.chunks);
            try testing.expectEqual(@max(320 * p.chunks, 8), p.ncnt);
            try testing.expectEqual(p.ncnt, ncntFor(ling_shape, rows));
        }
    }
    try testing.expect(planFor(ling_shape, f, 0) == null);
    try testing.expect(planFor(ling_shape, f, decode.Kernels.max_rows + 1) == null);
    try testing.expect(planFor(ling_shape, .{ .ks = .{ 1, 2, 4 }, .defer_affine = f.defer_affine, .bt0 = 4 }, 1) == null);
    try testing.expect(planFor(ling_shape, .{ .ks = f.ks, .defer_affine = .{ true, false, true }, .bt0 = 4 }, 1) == null);
    try testing.expect(planFor(.{ .in = 4096, .out = .{ 24576, 8192, 8000 }, .n = 96 }, f, 1) == null);
    try testing.expect(planFor(.{ .in = 4000, .out = ling_shape.out, .n = 96 }, f, 1) == null);
    try testing.expect(planFor(ling_shape, .{ .ks = f.ks, .defer_affine = f.defer_affine, .bt0 = 3 }, 1) == null);
}

test "turboexl3 ling kdaproj grid: the head's member, K slice and block of every threadgroup cover each member's (K slice, block) once, at every tile count" {
    for ([_]u32{ 1, 2, 4, 8 }) |bt| {
        const gc = 8 / bt;
        const kss = [3]u32{ 1, 2, 2 };
        // the head's arithmetic (kernels/kdaproj/head.metal)
        const nt = [3]u32{ ling_shape.out[0] / 128 * gc, ling_shape.out[1] / 128 * gc, ling_shape.out[2] / 128 * gc };
        const f1 = nt[0] * kss[0];
        const f2 = f1 + nt[1] * kss[1];
        const tgs = f2 + nt[2] * kss[2];
        try testing.expectEqual(planFor(ling_shape, .{ .ks = kss, .defer_affine = .{ true, true, true }, .bt0 = bt }, 1).?.tgs, tgs);
        var seen: [3][2][192 * 8]u8 = @splat(@splat(@splat(0)));
        for (0..tgs) |fl| {
            const flat: u32 = @intCast(fl);
            const jm: u32 = if (flat < f1) 0 else if (flat < f2) 1 else 2;
            const mflat = flat - (if (jm == 0) 0 else if (jm == 1) f1 else f2);
            const blk = mflat % nt[jm];
            const ksl = mflat / nt[jm];
            try testing.expect(ksl < kss[jm]);
            seen[jm][ksl][blk] += 1;
        }
        for (0..3) |m| for (0..2) |k| for (0..192 * 8) |b| {
            const want: u8 = if (k < kss[m] and b < nt[m]) 1 else 0;
            try testing.expectEqual(want, seen[m][k][b]);
        };
    }
}

test "turboexl3 ling kdaproj variant texts: the b launch's member is its two pieces; the variant launch is kbfirst in bmember_1's place, its head, the member choice, then DEC-ONE's statements over prepared rows in each member's branch (pf's in the pf text) and nothing else; every operand and argument named, no shape buffers" {
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/kdaproj_variants.metal"));
    // the served texts keep their values
    try testing.expectEqualStrings(kbmember, kbmember_1 ++ kbmember_2);
    try testing.expectEqualStrings(text_kdec0b, kbmember_1 ++ kbmember_2 ++ khead ++ kmember ++ kbody);
    try testing.expectEqualStrings(text_kdec0, khead ++ kmember ++ kbody);
    // the variant texts
    try testing.expectEqualStrings(text_kdec0bv, kbfirst ++ kbmember_2 ++ khead_v ++ kmember ++ kbt_0 ++ kbody ++ kbt_1 ++ kbody ++ kbt_close);
    try testing.expectEqualStrings(dense.text_dec0_pf, dense.dec_head ++ dense.dec_body0_pf);
    try testing.expectEqualStrings(text_kdec0bv_pf, kbfirst ++ kbmember_2 ++ khead_v ++ kmember ++ kbt_0 ++ dense.dec_body0_pf ++ kbt_1 ++ dense.dec_body0_pf ++ kbt_close);
    // the variant head keeps the head's member-independent lines, once each, and reads each member's own tiles
    for (pin[0..26]) |l| {
        try testing.expectEqual(@as(usize, 1), count(khead, l));
        try testing.expectEqual(@as(usize, 1), count(khead_v, l));
    }
    for (pin[29..33]) |l| {
        try testing.expectEqual(@as(usize, 1), count(khead_v, l));
        try testing.expectEqual(@as(usize, 0), count(khead, l));
    }
    // each branch sets its member's BTC; one branch where the tile counts agree
    try testing.expectEqual(@as(usize, 1), count(kbt_0, pin[26]));
    try testing.expectEqual(@as(usize, 1), count(kbt_0, pin[28]));
    try testing.expectEqual(@as(usize, 1), count(kbt_1, pin[27]));
    try testing.expectEqual(@as(usize, 1), count(text_kdec0bv, pin[26]) * count(text_kdec0bv, pin[27]));
    // b's row statements are bmember_2's, its choice keeps bmember_1's lines; no BT outside the branches' own
    for (pin[33..37]) |l| {
        try testing.expectEqual(@as(usize, 1), count(kbmember_1, l));
        try testing.expectEqual(@as(usize, 1), count(kbfirst, l));
    }
    for ([_][:0]const u8{ text_kdec0bv, text_kdec0bv_pf }) |txt| {
        try testing.expectEqual(@as(usize, 0), count(txt, "uint(BT)"));
        try testing.expectEqual(@as(usize, 2), count(txt, dense.dec_gemv0));
        for (kdec0b_in ++ kdec0b_out) |n| try testing.expect(count(txt, std.mem.span(n)) > 0);
        for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "IN", "OUT0", "OUT1", "OUT2", "KS0", "KS1", "BT0", "BT1", "RB", "ROWS", "DEFER", "OUT_T", "FB", "OUTB", "BF" }) |n| try testing.expect(count(txt, n) > 0);
        for (kdec0b_in ++ kdec0b_out) |n| for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| {
            var buf: [24]u8 = undefined;
            try testing.expectEqual(@as(usize, 0), count(txt, std.fmt.bufPrint(&buf, "{s}{s}", .{ std.mem.span(n), suf }) catch unreachable));
        };
    }
}

test "turboexl3 ling kdaproj variant plan: the configuration's variant launch per row count (the members' tiles apart at a verify's rows where they fit, b first and DEC-ONE's GEMV variant at every row count); the variant grid covers each member's (K slice, block) and b's output rows once, at every pair of tile counts, b first or last" {
    const f: Facts = .{ .ks = .{ 1, 2, 2 }, .defer_affine = .{ true, true, true }, .bt0 = 2 };
    const one = planFor(ling_shape, .{ .ks = f.ks, .defer_affine = f.defer_affine, .bt0 = 1 }, 1).?;
    const three = planFor(ling_shape, f, 3).?;
    const five = planFor(ling_shape, .{ .ks = f.ks, .defer_affine = f.defer_affine, .bt0 = 1 }, 5).?;
    const none: config.Config = .{};
    try testing.expect(variantFor(none, one, 1) == null and variantFor(none, three, 3) == null);
    // the members' tiles apart: at a verify's rows only
    const bts: config.Config = .{ .kda_bt0 = 1, .kda_bt1 = 2 };
    try testing.expect(variantFor(bts, one, 1) == null);
    const v3 = variantFor(bts, three, 3).?;
    try testing.expect(!v3.bfirst and v3.bt0 == 1 and v3.bt1 == 2 and v3.dv == 0);
    // a tile count past the chunk's room (2 BT RB > 32) keeps the plan's: at 5 rows 8 and 4 do not fit, so nothing does
    const wide: config.Config = .{ .kda_bt0 = 8, .kda_bt1 = 4 };
    try testing.expect(variantFor(wide, five, 5) == null);
    try testing.expectEqual(@as(u32, 4), variantFor(wide, three, 3).?.bt1);
    try testing.expectEqual(three.bt, variantFor(wide, three, 3).?.bt0);
    // b first and the GEMV variants at every row count, at the plan's tiles
    const bf: config.Config = .{ .kda_bfirst = true };
    const b1 = variantFor(bf, one, 1).?;
    try testing.expect(b1.bfirst and b1.bt0 == one.bt and b1.bt1 == one.bt and b1.dv == 0);
    const t4: config.Config = .{ .dec_variant = 1 };
    try testing.expectEqual(@as(u8, 1), variantFor(t4, one, 1).?.dv);
    try testing.expectEqual(@as(u8, 3), variantFor(.{ .dec_variant = 3, .kda_bt1 = 1 }, three, 3).?.dv);
    // the plan's threadgroups are tgsAt's at its own tiles
    for (1..decode.Kernels.max_rows + 1) |rows| for ([_]u32{ 1, 2, 4, 8 }) |bt| {
        const p = planFor(ling_shape, .{ .ks = f.ks, .defer_affine = f.defer_affine, .bt0 = bt }, rows) orelse continue;
        try testing.expectEqual(p.tgs, tgsAt(ling_shape, p, p.bt, p.bt));
    };
    // the variant head's arithmetic (kernels/kdaproj/head_v.metal, bfirst.metal) over the grid of tgs + OUTB / 4
    // threadgroups: b's first (BF 1) or last (BF 0), each member at its own tiles
    const outb: u32 = 64;
    const kss = [3]u32{ 1, 2, 2 };
    for ([_]u32{ 1, 2, 4, 8 }) |bt0| for ([_]u32{ 1, 2, 4, 8 }) |bt1| for ([_]bool{ false, true }) |bfirst| {
        const gcs = [3]u32{ 8 / bt0, 8 / bt1, 8 / bt1 };
        const nt = [3]u32{ ling_shape.out[0] / 128 * gcs[0], ling_shape.out[1] / 128 * gcs[1], ling_shape.out[2] / 128 * gcs[2] };
        const f1 = nt[0] * kss[0];
        const f2 = f1 + nt[1] * kss[1];
        const tgs = f2 + nt[2] * kss[2];
        try testing.expectEqual(tgs, tgsAt(ling_shape, three, bt0, bt1));
        const nbf: u32 = if (bfirst) outb / 4 else 0;
        var seen: [3][2][192 * 8]u8 = @splat(@splat(@splat(0)));
        var b_rows: [64]u8 = @splat(0);
        for (0..tgs + outb / 4) |xi| {
            const x: u32 = @intCast(xi);
            const is_b = if (bfirst) x < outb / 4 else x >= tgs;
            if (is_b) {
                const out_row = (if (bfirst) x else x - tgs) * 4;
                for (0..4) |tm| b_rows[@as(usize, out_row) + tm] += 1;
                continue;
            }
            const flat = x - nbf;
            const jm: u32 = if (flat < f1) 0 else if (flat < f2) 1 else 2;
            const mflat = flat - (if (jm == 0) 0 else if (jm == 1) f1 else f2);
            const blk = mflat % nt[jm];
            const ksl = mflat / nt[jm];
            try testing.expect(ksl < kss[jm]);
            // the member's output block is a whole 128-wide block of its own
            try testing.expect(blk / gcs[jm] < ling_shape.out[jm] / 128);
            seen[jm][ksl][blk] += 1;
        }
        for (b_rows) |c| try testing.expectEqual(@as(u8, 1), c);
        for (0..3) |m| for (0..2) |k| for (0..192 * 8) |b| {
            const want: u8 = if (k < kss[m] and b < nt[m]) 1 else 0;
            try testing.expectEqual(want, seen[m][k][b]);
        };
    };
}
