//! Ling-3.1-flash's MLA decode projections of the layer's input in one GEMV launch (`mlaProjG`): an MLA layer's q_a
//! (4096 -> 1536) and kv_a (4096 -> 640), two EXL3 projections of one input, each in its own buffers (nothing joined at
//! load) and each at its own `linear`'s K split (q_a KS 8, kv_a KS 16), and the layer's head gate g_proj (bf16,
//! 4096 -> 64) as a third member. A call of 1 to 16 rows is two launches:
//! - `turboexl3_ling_mlaproj_prep`: dense's `prep_sc` text with member m on the grid's z, preparing x with its own suh
//!   into rows m * ROWS.. of the joined prepared rows, and zeroing the GEMV's arrival counters;
//! - `turboexl3_ling_mlaproj_dec0g`: `kdaproj/bmember` (the gate's threadgroups, past the two members', run MLX's gemv
//!   over each row: the kernel MLX's matmul picks for g's one-row product), then dense's DEC-ONE text over prepared rows
//!   (`dense.text_dec0`) under a head that reads the threadgroup's member, K slice and block off the grid and gives the
//!   member its own widths, K split, arrivals and simdgroup span (`mlaproj/head`), with the member's prepared rows,
//!   trellis, svh, output, partial sums and arrival counters chosen after it (`mlaproj/member`).
//! Member m's output is its own `linear`'s bytes at every row count: each output is the same products in the same order
//! (its K split's partial sums, summed in slice order by its block's last arrival), and neither the tiles per
//! threadgroup nor where the prepare runs moves a byte of it. The gate's output is the served one-row matmul's bytes at
//! every row (MLX's gemv statement for statement), so each row of an exact verify's denseRows too.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const root = @import("../../root.zig");
const kdaproj = @import("kdaproj.zig");

// ── texts ──

/// the prepare's member choice: threadgroup z's suh and its rows of the joined prepared rows
pub const mprep = @embedFile("kernels/mlaproj/prep.metal");
/// DEC-ONE's head over two members: the threadgroup's member, K slice and block, the member's widths and K split
pub const mhead = @embedFile("kernels/mlaproj/head.metal");
/// the member's prepared rows, trellis, svh, output, partial sums and arrival counters
pub const mmember = @embedFile("kernels/mlaproj/member.metal");

/// `dense.text_prep_sc` (lane order, block sums, DEC-ONE's counters zeroed) for the two members in one launch
pub const text_mprep_sc: [:0]const u8 = mprep ++ dense.text_prep_sc;
/// `dense.text_dec0` with its head (`dense.dec_head`) in the two-member head and the member choice, the gate's
/// threadgroups (`kdaproj/bmember`, MLX's gemv over each row) before it
pub const text_mdec0g: [:0]const u8 = kdaproj.kbmember ++ mhead ++ mmember ++ dense.dec_gemv0 ++ dense.dec_gemv_arrive ++ dense.dec_xsum0 ++ dense.dec_finish_v ++ dense.dec_finish_store;

const mprep_in = [_][*:0]const u8{ "x", "suh0", "suh1" };
const mprep_out = [_][*:0]const u8{ "jxp", "jxs", "cnt" };
const mdec0g_in = [_][*:0]const u8{ "jxp", "jxs", "w0", "w1", "svh0", "svh1", "jcnt", "xb", "wb" };
const mdec0g_out = [_][*:0]const u8{ "y0", "y1", "z0", "z1", "yb" };

// ── kernels: made at their first call ──

const Kind = enum(u8) { mprep_sc, mdec0g };
const kernel_lock = Global(std.atomic.Value(u32), "ling_mlaproj_kernel_lock", std.atomic.Value(u32).init(0));
const kernels = Global([2]?mlx.mlx_fast_metal_kernel, "ling_mlaproj_kernels", @splat(null));

fn kernelOf(k: Kind) !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    const slot = &kernels.get().*[@intFromEnum(k)];
    if (slot.*) |made| return made;
    const made = switch (k) {
        .mprep_sc => ks.newKernel("turboexl3_ling_mlaproj_prep", &mprep_in, &mprep_out, text_mprep_sc, decode.header, true),
        .mdec0g => ks.newKernel("turboexl3_ling_mlaproj_dec0g", &mdec0g_in, &mdec0g_out, text_mdec0g, decode.header5, true),
    } orelse return error.KernelCreateFailed;
    slot.* = made;
    return made;
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernels };

// ── the plan ──

/// The two members' shape: one input width and rate, each member's output width (the trellises [in/16, out_m/16, n])
pub const Shape = struct { in: u32, out: [2]u32, n: u32 };

/// A call's launch: each member's own `linear`'s K split, DEC-ONE's row chunks (`dense.Dense.rbFor`), the output tiles
/// per threadgroup, the arrival counters and the members' GEMV threadgroups (the gate's come after them).
pub const Plan = struct { ks0: u32, ks1: u32, rb: u32, bt: u32, chunks: u32, ncnt: u32, tgs: u32 };

/// The members as `linear` takes each (trellis U16 or I16 [in/16, out/16, n], suh F16 [in], svh F16 [out]): one input
/// width, one rate, one trellis dtype; else an error.
pub fn shapeOf(p: [2]root.Proj) !Shape {
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
pub const Facts = struct { ks: [2]u32, defer_affine: [2]bool, bt0: u32 };

/// The arrival counters a call of `rows` rows over `sh` zeroes before its GEMV: both members' output blocks times
/// DEC-ONE's row chunks (member 1's after member 0's), at least 8.
pub fn ncntFor(sh: Shape, rows: usize) u32 {
    const rows_u: u32 = @intCast(rows);
    const rb = dense.Dense.rbFor(rows_u);
    return @max((sh.out[0] + sh.out[1]) / 128 * ((rows_u + rb - 1) / rb), 8);
}

/// The plan of a call of `rows` rows over members of shape `sh` with facts `f`, or null where it does not serve (the
/// caller runs each projection): past the decode rows, an input or output width DEC-ONE does not take, the members at
/// two affine deferrals.
pub fn planFor(sh: Shape, f: Facts, rows: usize) ?Plan {
    if (rows < 1 or rows > decode.Kernels.max_rows) return null;
    if (sh.in % 128 != 0 or (sh.in / 16) % 4 != 0) return null;
    for (sh.out) |o| if (o == 0 or o % 128 != 0) return null;
    if (f.defer_affine[0] != f.defer_affine[1]) return null;
    for (f.ks) |k| if (k == 0 or (sh.in / 16) % (k * 4) != 0) return null;
    const rows_u: u32 = @intCast(rows);
    const rb = dense.Dense.rbFor(rows_u);
    // the tiles per threadgroup move no output byte; member 0's own count (the wider), within a lane's accumulators
    const bt = f.bt0;
    if (bt == 0 or 8 % bt != 0 or 2 * bt * rb > 32) return null;
    const gc = 8 / bt;
    const tgs = (sh.out[0] / 128) * gc * f.ks[0] + (sh.out[1] / 128) * gc * f.ks[1];
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

/// The width of the gate's output where `mlaProjG` takes its weight `wg` for an input of width `in` in `xd`, else null:
/// x and w bf16, w [OUTG, in] (the host's loaded [in, OUTG] column-major view transposed back), OUTG a multiple of 4
/// below 4096, in a multiple of 1024 and at least 16 OUTG. There MLX's matmul runs g's one-row product as its gemv at
/// BM 1, BN 8 over whole 1024-blocks, the statements `kdaproj/bmember` holds.
pub fn gTakes(in: u32, xd: mlx.mlx_dtype, wg: mlx.mlx_array) ?u32 {
    if (xd != .bfloat16 or mlx.mlx_array_dtype(wg) != .bfloat16) return null;
    const ws = mlx.getShape(wg);
    if (ws.len != 2 or ws[0] < 4 or ws[1] != @as(c_int, @intCast(in))) return null;
    const og: u32 = @intCast(ws[0]);
    if (og % 4 != 0 or og >= 4096 or in % 1024 != 0 or in < 16 * og) return null;
    return og;
}

// ── the call ──

/// An MLA layer's q_a and kv_a of x [..., in] (bf16), `p` = q_a, kv_a as `linear` takes each, and its head gate
/// g_proj (weight `wg` bf16 [OUTG, in] row-major, `gTakes`) in two launches: one prepare launch, then one GEMV launch.
/// The outputs: q_a's [rows, out_0] and kv_a's [rows, out_1] (each its own `linear`'s bytes), then g's [rows, OUTG]
/// bf16 (MLX's matmul's bytes of each row: the served one-row matmul's, so each row of the exact verify's denseRows);
/// the caller frees the three. Refusals before any launch: the members' shapes and dtypes as `shapeOf` checks them,
/// error.Exl3EngineNotImplemented where `gTakes` does not hold or `plan` declines (the caller runs each projection).
pub fn mlaProjG(s: mlx.mlx_stream, x: mlx.mlx_array, p: [2]root.Proj, wg: mlx.mlx_array, dec: format.Decode) ![3]mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return error.BadExl3Dtype;
    const sh = try shapeOf(p);
    const og = gTakes(sh.in, xd, wg) orelse return error.Exl3EngineNotImplemented;
    const v = try root.rowsOf(x, sh.in, s);
    defer root.freeArr(v.x2);
    const pl = (try plan(sh, dec, v.rows, xd)) orelse return error.Exl3EngineNotImplemented;
    const pre = try prepare(s, v.x2, p, sh, v.rows, pl);
    defer for (pre) |arr| root.freeArr(arr);
    return gemvG(s, pre, p, sh, dec, v.rows, xd, pl, v.x2, wg, og);
}

/// The prepare launch (`turboexl3_ling_mlaproj_prep`) over x's rows x2 [rows, in]: the joined prepared rows, their
/// block sums and the GEMV's zeroed counters; the caller frees the three
fn prepare(s: mlx.mlx_stream, x2: mlx.mlx_array, p: [2]root.Proj, sh: Shape, rows: usize, pl: Plan) ![3]mlx.mlx_array {
    const nb = sh.in / 128;
    const r: c_int = @intCast(rows);
    const rows_u: u32 = @intCast(rows);
    // dense's prep_sc for each member, the members on the grid's z; the GEMV's counters zeroed
    const pcfg = try ks.buildConfig(.{
        .grid = .{ nb * 32, rows_u, 2 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ 2 * r, @intCast(sh.in) }, &[_]c_int{ 2 * r, @intCast(nb) }, &[_]c_int{@intCast(pl.ncnt)} },
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
    try ks.applyConfig(try kernelOf(.mprep_sc), pcfg, &.{ x2, p[0].suh, p[1].suh }, &pre, s);
    return pre;
}

/// The GEMV launch over the two members, then the gate's threadgroups (`OUTG / 4` a row chunk, MLX's gemv's four
/// outputs a threadgroup): q_a's, kv_a's and g's outputs
fn gemvG(s: mlx.mlx_stream, pre: [3]mlx.mlx_array, p: [2]root.Proj, sh: Shape, dec: format.Decode, rows: usize, xd: mlx.mlx_dtype, pl: Plan, x2: mlx.mlx_array, wg: mlx.mlx_array, og: u32) ![3]mlx.mlx_array {
    const rate = format.Rate.fromN(sh.n) orelse return error.BadExl3Shape;
    const d0 = try root.denseCached(.{ .in = sh.in, .out = sh.out[0], .rate = rate, .dec = dec, .out_dtype = xd });
    const r: c_int = @intCast(rows);
    // dense's DEC-ONE GEMV for the two members, each at its own widths and K split; the gate's first threadgroup and width
    const t = d0.templ();
    const tmpl = [_]ks.TemplateArg{
        t[0], t[1], t[2],
        .{ .name = "IN", .value = .{ .int = @intCast(sh.in) } },
        .{ .name = "OUT0", .value = .{ .int = @intCast(sh.out[0]) } },
        .{ .name = "OUT1", .value = .{ .int = @intCast(sh.out[1]) } },
        .{ .name = "KS0", .value = .{ .int = @intCast(pl.ks0) } },
        .{ .name = "KS1", .value = .{ .int = @intCast(pl.ks1) } },
        .{ .name = "BT", .value = .{ .int = @intCast(pl.bt) } },
        .{ .name = "RB", .value = .{ .int = @intCast(pl.rb) } },
        .{ .name = "ROWS", .value = .{ .int = r } },
        .{ .name = "DEFER", .value = .{ .int = if (d0.defer_affine) 1 else 0 } },
        .{ .name = "OUT_T", .value = .{ .dtype = xd } },
        .{ .name = "FB", .value = .{ .int = @intCast(pl.tgs) } },
        .{ .name = "OUTB", .value = .{ .int = @intCast(og) } },
    };
    const o0: c_int = @intCast(sh.out[0]);
    const o1: c_int = @intCast(sh.out[1]);
    const cfg = try ks.buildConfig(.{
        .grid = .{ (pl.tgs + og / 4) * 128, pl.chunks, 1 },
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ r, o0 }, &[_]c_int{ r, o1 }, &[_]c_int{ @as(c_int, @intCast(pl.ks0)) * r, o0 }, &[_]c_int{ @as(c_int, @intCast(pl.ks1)) * r, o1 }, &[_]c_int{ r, @intCast(og) } },
        .out_dtypes = &.{ xd, xd, .float32, .float32, xd },
        .template = &tmpl,
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [5]mlx.mlx_array = undefined;
    try ks.applyConfig(try kernelOf(.mdec0g), cfg, &.{ pre[0], pre[1], p[0].trellis, p[1].trellis, p[0].svh, p[1].svh, pre[2], x2, wg }, &o, s);
    _ = mlx.mlx_array_free(o[2]);
    _ = mlx.mlx_array_free(o[3]);
    return .{ o[0], o[1], o[4] };
}

// ── tests (CPU) ──

const testing = std.testing;

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

test "turboexl3 ling mlaproj texts: dense's prep_sc verbatim after the member choice; the gate's member, then dense's DEC-ONE text over prepared rows verbatim after its head, under the two-member head and the member choice; each operand bound by name" {
    try testing.expectEqualStrings(text_mprep_sc, mprep ++ dense.text_prep_sc);
    try testing.expect(std.mem.startsWith(u8, text_mdec0g, kdaproj.kbmember ++ mhead ++ mmember ++ dense.dec_gemv0));
    try testing.expectEqualStrings(dense.text_dec0[dense.dec_head.len..], text_mdec0g[kdaproj.kbmember.len + mhead.len + mmember.len ..]);
    try testing.expectEqual(@as(usize, 0), count(text_mdec0g, dense.dec_head));
    for (mprep_in ++ mprep_out) |n| try testing.expect(count(text_mprep_sc, std.mem.span(n)) > 0);
    for (mdec0g_in ++ mdec0g_out) |n| try testing.expect(count(text_mdec0g, std.mem.span(n)) > 0);
    for ([_][]const u8{ "DIM", "LANE_ORDER", "NCNT", "NTH", "ROWS" }) |n| try testing.expect(count(text_mprep_sc, n) > 0);
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "IN", "OUT0", "OUT1", "KS0", "KS1", "BT", "RB", "ROWS", "DEFER", "OUT_T", "FB", "OUTB" }) |n| try testing.expect(count(text_mdec0g, n) > 0);
    // the head and the member choice name everything dense's DEC-ONE body reads off dense's head and its operands
    for ([_][]const u8{ "NW", "WMASK", "INC", "OUTC", "IN_TILES", "OUT_TILES", "KSC", "BTC", "RBC", "ROWSC", "NCHC", "GC", "ARR", "PER", "HAD_SCALE", "blk", "chunk", "r0", "ksl", "tid", "red", "arrived_last" }) |n| try testing.expect(count(mhead, n) > 0);
    for ([_][]const u8{ "xp", "xs", "svh", "cnt", "zc" }) |n| try testing.expect(count(mmember, n) > 0);
    // MLX adds a shape / strides / ndim buffer for an input name followed by those suffixes
    for ([_][:0]const u8{ text_mprep_sc, text_mdec0g }) |txt| for (mprep_in ++ mprep_out ++ mdec0g_in ++ mdec0g_out) |n| for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| {
        var buf: [24]u8 = undefined;
        try testing.expectEqual(@as(usize, 0), count(txt, std.fmt.bufPrint(&buf, "{s}{s}", .{ std.mem.span(n), suf }) catch unreachable));
    };
}

/// Ling-3.1-flash's MLA projections of the layer's input: q_a 4096 -> 1536 and kv_a 4096 -> 640, K6
const ling_shape: Shape = .{ .in = 4096, .out = .{ 1536, 640 }, .n = 96 };

test "turboexl3 ling mlaproj plan: Ling's members keep their own K splits (q_a 8, kv_a 16) at decode rows 1 to 16; past the decode rows, two affine deferrals and widths DEC-ONE does not take decline" {
    try testing.expectEqual(@as(u32, 8), dense.Dense.ksFor(4096, 1536));
    try testing.expectEqual(@as(u32, 16), dense.Dense.ksFor(4096, 640));
    const f: Facts = .{ .ks = .{ 8, 16 }, .defer_affine = .{ true, true }, .bt0 = 4 };
    for (1..decode.Kernels.max_rows + 1) |rows| {
        const rb = dense.Dense.rbFor(@intCast(rows));
        for ([_]u32{ 1, 2, 4, 8 }) |bt| {
            const fb: Facts = .{ .ks = f.ks, .defer_affine = f.defer_affine, .bt0 = bt };
            if (2 * bt * rb > 32) {
                try testing.expect(planFor(ling_shape, fb, rows) == null);
                continue;
            }
            const p = planFor(ling_shape, fb, rows).?;
            try testing.expectEqual(@as(u32, 8), p.ks0);
            try testing.expectEqual(@as(u32, 16), p.ks1);
            try testing.expectEqual(rb, p.rb);
            try testing.expectEqual(bt, p.bt);
            const gc = 8 / bt;
            // member 0's 12 128-blocks at eight K slices, member 1's 5 at sixteen
            try testing.expectEqual(12 * gc * 8 + 5 * gc * 16, p.tgs);
            try testing.expectEqual((@as(u32, @intCast(rows)) + rb - 1) / rb, p.chunks);
            try testing.expectEqual(@max(17 * p.chunks, 8), p.ncnt);
            try testing.expectEqual(p.ncnt, ncntFor(ling_shape, rows));
        }
    }
    try testing.expect(planFor(ling_shape, f, 0) == null);
    try testing.expect(planFor(ling_shape, f, decode.Kernels.max_rows + 1) == null);
    try testing.expect(planFor(ling_shape, .{ .ks = f.ks, .defer_affine = .{ true, false }, .bt0 = 4 }, 1) == null);
    try testing.expect(planFor(.{ .in = 4096, .out = .{ 1536, 600 }, .n = 96 }, f, 1) == null);
    try testing.expect(planFor(.{ .in = 4000, .out = ling_shape.out, .n = 96 }, f, 1) == null);
    try testing.expect(planFor(ling_shape, .{ .ks = f.ks, .defer_affine = f.defer_affine, .bt0 = 3 }, 1) == null);
    try testing.expect(planFor(ling_shape, .{ .ks = .{ 8, 0 }, .defer_affine = f.defer_affine, .bt0 = 4 }, 1) == null);
}

test "turboexl3 ling mlaproj grid: the head's member, K slice and block of every member threadgroup cover each member's (K slice, block) once, at every tile count; the gate's threadgroups follow them" {
    for ([_]u32{ 1, 2, 4, 8 }) |bt| {
        const gc = 8 / bt;
        const kss = [2]u32{ 8, 16 };
        // the head's arithmetic (kernels/mlaproj/head.metal)
        const nt = [2]u32{ ling_shape.out[0] / 128 * gc, ling_shape.out[1] / 128 * gc };
        const f1 = nt[0] * kss[0];
        const tgs = f1 + nt[1] * kss[1];
        try testing.expectEqual(planFor(ling_shape, .{ .ks = kss, .defer_affine = .{ true, true }, .bt0 = bt }, 1).?.tgs, tgs);
        var seen: [2][16][12 * 8]u8 = @splat(@splat(@splat(0)));
        for (0..tgs) |fl| {
            const flat: u32 = @intCast(fl);
            const jm: u32 = if (flat < f1) 0 else 1;
            const mflat = flat - (if (jm == 0) 0 else f1);
            const blk = mflat % nt[jm];
            const ksl = mflat / nt[jm];
            try testing.expect(ksl < kss[jm]);
            seen[jm][ksl][blk] += 1;
        }
        for (0..2) |m| for (0..16) |k| for (0..12 * 8) |b| {
            const want: u8 = if (k < kss[m] and b < nt[m]) 1 else 0;
            try testing.expectEqual(want, seen[m][k][b]);
        };
    }
}
