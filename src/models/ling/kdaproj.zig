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

const kprep_in = [_][*:0]const u8{ "x", "suh0", "suh1", "suh2" };
const kprep_out = [_][*:0]const u8{ "jxp", "jxs", "cnt" };
const kdec0_in = [_][*:0]const u8{ "jxp", "jxs", "w0", "w1", "w2", "svh0", "svh1", "svh2", "jcnt" };
const kdec0_out = [_][*:0]const u8{ "y0", "y1", "y2", "z0", "z1", "z2" };

// ── kernels: made at their first call ──

const Kind = enum(u8) { kprep_sc, kdec0 };
const kernel_lock = Global(std.atomic.Value(u32), "ling_kdaproj_kernel_lock", std.atomic.Value(u32).init(0));
const kernels = Global([2]?mlx.mlx_fast_metal_kernel, "ling_kdaproj_kernels", @splat(null));

fn kernelOf(k: Kind) !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    const slot = &kernels.get().*[@intFromEnum(k)];
    if (slot.*) |made| return made;
    const made = switch (k) {
        .kprep_sc => ks.newKernel("turboexl3_ling_kdaproj_prep", &kprep_in, &kprep_out, text_kprep_sc, decode.header, true),
        .kdec0 => ks.newKernel("turboexl3_ling_kdaproj_dec0", &kdec0_in, &kdec0_out, text_kdec0, decode.header5, true),
    } orelse return error.KernelCreateFailed;
    slot.* = made;
    return made;
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernels };

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
    const nb = sh.in / 128;
    const r: c_int = @intCast(v.rows);
    const rows_u: u32 = @intCast(v.rows);
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
    defer for (pre) |arr| if (arr.ctx != null) {
        _ = mlx.mlx_array_free(arr);
    };
    try ks.applyConfig(try kernelOf(.kprep_sc), pcfg, &.{ v.x2, p[0].suh, p[1].suh, p[2].suh }, &pre, s);
    return gemv(s, pre[0], pre[1], pre[2], p, sh, dec, v.rows, xd, pl);
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
    };
    const k0: c_int = @intCast(pl.ks0);
    const k1: c_int = @intCast(pl.ks1);
    const o0: c_int = @intCast(sh.out[0]);
    const o1: c_int = @intCast(sh.out[1]);
    const o2: c_int = @intCast(sh.out[2]);
    const cfg = try ks.buildConfig(.{
        .grid = .{ pl.tgs * 128, pl.chunks, 1 },
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ r, o0 }, &[_]c_int{ r, o1 }, &[_]c_int{ r, o2 }, &[_]c_int{ k0 * r, o0 }, &[_]c_int{ k1 * r, o1 }, &[_]c_int{ k1 * r, o2 } },
        .out_dtypes = &.{ xd, xd, xd, .float32, .float32, .float32 },
        .template = &tmpl,
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [6]mlx.mlx_array = undefined;
    try ks.applyConfig(try kernelOf(.kdec0), cfg, &.{ jxp, jxs, p[0].trellis, p[1].trellis, p[2].trellis, p[0].svh, p[1].svh, p[2].svh, cnt }, &o, s);
    for (o[3..]) |zm| _ = mlx.mlx_array_free(zm);
    return .{ o[0], o[1], o[2] };
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
