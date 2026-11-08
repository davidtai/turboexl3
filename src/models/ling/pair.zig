//! Ling-3.1-flash's same-input linear pairs in one DEC-ONE launch (`linearPair`): a KDA layer's z | a (its output gate
//! and its forget gate, 4096 -> 8192 each), the shared expert's gate | up (4096 -> 1280) and the dense FFN's gate | up
//! (4096 -> 10240). Each member stays in its own buffers (no joined copy, nothing done at load): the launch binds both
//! members' trellises and svh and chooses a threadgroup's member by its output block. A call runs both members in the
//! DEC-ONE form each member's own `linear` takes at that row count (the member's cached `dense.Dense`: its prepare inside
//! the launch or in a launch of its own, its output tiles per threadgroup):
//! - the prepare inside: dense's counter zeroing (`turboexl3_dense_dec_zero`), then `turboexl3_ling_pair_dec1` (dense's
//!   DEC-ONE text with the prepare inside, the member chosen after its head: its suh, its trellis, its svh);
//! - the prepare in its own launch: one prepare launch for both members (`turboexl3_ling_pair_prep`: dense's `prep_sc`
//!   text with member m, on the grid's z, preparing x with its own suh into rows m * ROWS.. of the pair's prepared
//!   rows), then `turboexl3_ling_pair_dec0` (dense's DEC-ONE text over prep_sc's rows, the member chosen after its head: its
//!   prepared rows, its trellis, its svh).
//! The GEMV reads a tile at Ripple's member tile address (`ricursive.jtile`, each trellis at base 0). Each member's K
//! split is its own linear's (the members must share it), so member m's columns are `linear`'s bytes of member m at
//! every decode row count the pair serves (1 to 16; DEC-ONE gives the three-launch chain's bytes at each).
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const root = @import("../../root.zig");
const ricursive = @import("../ricursive/ricursive.zig");

// ── texts ──

/// the pair's prepare: member threadgroup z's suh and its rows of the pair's prepared rows
pub const jprep = @embedFile("kernels/pair/jprep.metal");
/// the GEMV over prepared rows: the threadgroup's member and its prepared rows
pub const jrows = @embedFile("kernels/pair/jrows.metal");
/// the GEMV with the prepare inside: the threadgroup's member and its suh
pub const jsuh = @embedFile("kernels/pair/jsuh.metal");
/// the member's trellis, output tiles and svh (after `jrows` or `jsuh`)
pub const jw = @embedFile("kernels/pair/jw.metal");

/// `dense.text_prep_sc` (lane order, block sums, DEC-ONE's counters zeroed) for both members of a pair in one launch
pub const text_jprep_sc: [:0]const u8 = jprep ++ dense.text_prep_sc;
/// DEC-ONE over prep_sc's rows (`dense.text_dec0`) with the member chosen after its head (`jrows`, `jw`) and the GEMV's
/// tile read in its trellis (`ricursive.jtile` in place of `dense.dec_tile`)
pub const text_jdec0: [:0]const u8 = dense.dec_head ++ jrows ++ jw ++ dense.dec_gemv0 ++ dense.dec_gemv_arrive_1 ++ ricursive.jtile ++ dense.dec_gemv_arrive_2 ++ dense.dec_xsum0 ++ dense.dec_finish_v ++ dense.dec_finish_store;
/// DEC-ONE with the prepare inside (`dense.text_dec1`) with the member chosen after its head (`jsuh`, `jw`) and the
/// GEMV's tile read in its trellis (`ricursive.jtile`)
pub const text_jdec1: [:0]const u8 = dense.dec_head ++ jsuh ++ jw ++ dense.dec_prep1 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive_1 ++ ricursive.jtile ++ dense.dec_gemv_arrive_2 ++ dense.dec_xsum1 ++ dense.dec_finish_v ++ dense.dec_finish_store;

const jprep_in = [_][*:0]const u8{ "x", "suh0", "suh1" };
const jprep_out = [_][*:0]const u8{ "jxp", "jxs", "cnt" };
const jdec0_in = [_][*:0]const u8{ "jxp", "jxs", "w0", "w1", "svh0", "svh1", "cnt" };
const jdec0_out = [_][*:0]const u8{ "y", "z" };
const jdec1_in = [_][*:0]const u8{ "x", "suh0", "suh1", "w0", "w1", "svh0", "svh1", "cnt" };
const jdec1_out = [_][*:0]const u8{ "y", "z", "xsx" };

// ── kernels: made at their first call ──

const Kind = enum(u8) { jprep_sc, jdec0, jdec1 };
const kernel_lock = Global(std.atomic.Value(u32), "ling_pair_kernel_lock", std.atomic.Value(u32).init(0));
const kernels = Global([3]?mlx.mlx_fast_metal_kernel, "ling_pair_kernels", @splat(null));

fn kernelOf(k: Kind) !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    const slot = &kernels.get().*[@intFromEnum(k)];
    if (slot.*) |made| return made;
    const made = switch (k) {
        .jprep_sc => ks.newKernel("turboexl3_ling_pair_prep", &jprep_in, &jprep_out, text_jprep_sc, decode.header, true),
        .jdec0 => ks.newKernel("turboexl3_ling_pair_dec0", &jdec0_in, &jdec0_out, text_jdec0, decode.header5, true),
        .jdec1 => ks.newKernel("turboexl3_ling_pair_dec1", &jdec1_in, &jdec1_out, text_jdec1, decode.header5, true),
    } orelse return error.KernelCreateFailed;
    slot.* = made;
    return made;
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernels };

// ── the plan ──

/// A pair's shape: the members' input width, output widths and rate (the trellises [in/16, out_m/16, n])
pub const PairShape = struct { in: u32, out0: u32, out1: u32, n: u32 };

/// A pair call's launch: the form each member's own `linear` takes at its row count (`inside`: the prepare in the GEMV
/// launch, `dense.Dense.decPrepInside`), the members' one K split, DEC-ONE's row chunks (`dense.Dense.rbFor`), the
/// members' output tiles per threadgroup (`dense.Dense.btDec`), the arrival counters and (the prepare inside) the
/// published input sums.
pub const Plan = struct { inside: bool, ks: u32, rb: u32, bt: u32, chunks: u32, ncnt: u32, nx: u32 };

/// The pair `a`, `b` as `linear` takes each (trellis U16 or I16 [in/16, out/16, n], suh F16 [in], svh F16 [out]): one
/// input width, one rate, one trellis dtype; else an error.
pub fn pairShape(a: root.Proj, b: root.Proj) !PairShape {
    const ta = mlx.getShape(a.trellis);
    const tb = mlx.getShape(b.trellis);
    if (ta.len != 3 or tb.len != 3 or ta[0] != tb[0] or ta[2] != tb[2] or ta[0] < 1 or ta[1] < 1 or tb[1] < 1) return error.BadExl3Shape;
    const td = mlx.mlx_array_dtype(a.trellis);
    if ((td != .uint16 and td != .int16) or mlx.mlx_array_dtype(b.trellis) != td) return error.BadExl3Dtype;
    const sh: PairShape = .{ .in = @intCast(ta[0] * 16), .out0 = @intCast(ta[1] * 16), .out1 = @intCast(tb[1] * 16), .n = @intCast(ta[2]) };
    for ([_]root.Proj{ a, b }, [_]u32{ sh.out0, sh.out1 }) |p, o| {
        if (mlx.getShape(p.suh).len != 1 or mlx.getShape(p.suh)[0] != @as(c_int, @intCast(sh.in))) return error.BadExl3Shape;
        if (mlx.getShape(p.svh).len != 1 or mlx.getShape(p.svh)[0] != @as(c_int, @intCast(o))) return error.BadExl3Shape;
        if (mlx.mlx_array_dtype(p.suh) != .float16 or mlx.mlx_array_dtype(p.svh) != .float16) return error.BadExl3Dtype;
    }
    return sh;
}

/// The plan of a call of `rows` rows in `xd` over a pair of shape `sh`, or null where the pair does not serve (the caller
/// runs `linear` per member): past the decode rows, or members whose own calls differ in K split, form or output tiles.
pub fn plan(sh: PairShape, dec: format.Decode, rows: usize, xd: mlx.mlx_dtype) !?Plan {
    if (rows < 1 or rows > decode.Kernels.max_rows) return null;
    if (sh.in % 128 != 0 or sh.out0 % 128 != 0 or sh.out1 % 128 != 0 or (sh.in / 16) % 4 != 0) return null;
    const rate = format.Rate.fromN(sh.n) orelse return null;
    const d0 = try root.denseCached(.{ .in = sh.in, .out = sh.out0, .rate = rate, .dec = dec, .out_dtype = xd });
    const d1 = try root.denseCached(.{ .in = sh.in, .out = sh.out1, .rate = rate, .dec = dec, .out_dtype = xd });
    if (d0.ks_split != d1.ks_split or d0.defer_affine != d1.defer_affine) return null;
    const inside = try d0.decPrepInside(rows);
    if (inside != try d1.decPrepInside(rows)) return null;
    const rows_u: u32 = @intCast(rows);
    const rb = dense.Dense.rbFor(rows_u);
    const bt = d0.btDec(rows, rb);
    if (bt != d1.btDec(rows, rb) or 8 % bt != 0) return null;
    const chunks = (rows_u + rb - 1) / rb;
    const nob = (sh.out0 + sh.out1) / 128;
    const nb = sh.in / 128;
    return .{
        .inside = inside,
        .ks = d0.ks_split,
        .rb = rb,
        .bt = bt,
        .chunks = chunks,
        .ncnt = @max(nob * chunks, 8),
        .nx = if (inside and d0.defer_affine) @max(nob * chunks * rb * nb, 8) else 8,
    };
}

// ── the call ──

/// Two linears of x [..., in] (bf16, f16 or f32), `a` and `b` as `linear` takes each, in one GEMV launch: y [rows,
/// out_a + out_b] in x's dtype, `a`'s columns then `b`'s, each `linear`'s bytes of its member. Refusals before any
/// launch: the pair's shapes and dtypes as `pairShape` checks them, error.Exl3EngineNotImplemented where `plan`
/// declines (the caller runs `linear` per member).
pub fn linearPair(s: mlx.mlx_stream, x: mlx.mlx_array, a: root.Proj, b: root.Proj, dec: format.Decode) !mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return error.BadExl3Dtype;
    const sh = try pairShape(a, b);
    const v = try root.rowsOf(x, sh.in, s);
    defer root.freeArr(v.x2);
    const p = (try plan(sh, dec, v.rows, xd)) orelse return error.Exl3EngineNotImplemented;
    return launch(s, v.x2, a, b, sh, dec, v.rows, xd, p);
}

fn launch(s: mlx.mlx_stream, x2: mlx.mlx_array, a: root.Proj, b: root.Proj, sh: PairShape, dec: format.Decode, rows: usize, xd: mlx.mlx_dtype, p: Plan) !mlx.mlx_array {
    const rate = format.Rate.fromN(sh.n) orelse return error.BadExl3Shape;
    const d0 = try root.denseCached(.{ .in = sh.in, .out = sh.out0, .rate = rate, .dec = dec, .out_dtype = xd });
    const r: c_int = @intCast(rows);
    const rows_u: u32 = @intCast(rows);
    const nb = sh.in / 128;
    const out = sh.out0 + sh.out1;
    const ncnt_shape = [_]c_int{@intCast(p.ncnt)};
    // the counters (and, the prepare in its own launch, both members' prepared rows): one launch before the GEMV
    var pre: [3]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (pre) |arr| if (arr.ctx != null) {
        _ = mlx.mlx_array_free(arr);
    };
    if (p.inside) {
        const zcfg = try ks.buildConfig(.{ .grid = .{ ((p.ncnt + 127) / 128) * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&ncnt_shape}, .out_dtypes = &.{.uint32}, .template = &.{.{ .name = "NCNT", .value = .{ .int = @intCast(p.ncnt) } }} });
        defer _ = mlx.mlx_fast_metal_kernel_config_free(zcfg);
        try ks.applyConfig(d0.dec_zero, zcfg, &.{a.svh}, pre[0..1], s);
    } else {
        // dense's prep_sc launch for each member, the members on the grid's z
        const pcfg = try ks.buildConfig(.{
            .grid = .{ nb * 32, rows_u, 2 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ 2 * r, @intCast(sh.in) }, &[_]c_int{ 2 * r, @intCast(nb) }, &ncnt_shape },
            .out_dtypes = &.{ .float16, .float32, .uint32 },
            .template = &.{
                .{ .name = "DIM", .value = .{ .int = @intCast(sh.in) } },
                .{ .name = "LANE_ORDER", .value = .{ .int = 1 } },
                .{ .name = "NCNT", .value = .{ .int = @intCast(p.ncnt) } },
                .{ .name = "NTH", .value = .{ .int = @intCast(nb * 32 * rows_u) } },
                .{ .name = "ROWS", .value = .{ .int = r } },
            },
        });
        defer _ = mlx.mlx_fast_metal_kernel_config_free(pcfg);
        try ks.applyConfig(try kernelOf(.jprep_sc), pcfg, &.{ x2, a.suh, b.suh }, &pre, s);
    }
    // dense's DEC-ONE GEMV launch over the pair's OUT (its grid, K split, chunks and tiles), the member boundary OB1
    const t = d0.templ();
    const tmpl = [_]ks.TemplateArg{
        t[0], t[1], t[2],
        .{ .name = "IN", .value = .{ .int = @intCast(sh.in) } },
        .{ .name = "OUT", .value = .{ .int = @intCast(out) } },
        .{ .name = "KS", .value = .{ .int = @intCast(p.ks) } },
        .{ .name = "BT", .value = .{ .int = @intCast(p.bt) } },
        .{ .name = "RB", .value = .{ .int = @intCast(p.rb) } },
        .{ .name = "ROWS", .value = .{ .int = r } },
        .{ .name = "DEFER", .value = .{ .int = if (d0.defer_affine) 1 else 0 } },
        .{ .name = "OUT_T", .value = .{ .dtype = xd } },
        .{ .name = "OB1", .value = .{ .int = @intCast(sh.out0 / 128) } },
    };
    const y_shape = [_]c_int{ r, @intCast(out) };
    const z_shape = [_]c_int{ @as(c_int, @intCast(p.ks)) * r, @intCast(out) };
    const grid = [3]u32{ (out / 16 / p.bt) * 128, p.chunks, p.ks };
    if (p.inside) {
        const cfg = try ks.buildConfig(.{ .grid = grid, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &y_shape, &z_shape, &[_]c_int{@intCast(p.nx)} }, .out_dtypes = &.{ xd, .float32, .float32 }, .template = &tmpl });
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        var o: [3]mlx.mlx_array = undefined;
        try ks.applyConfig(try kernelOf(.jdec1), cfg, &.{ x2, a.suh, b.suh, a.trellis, b.trellis, a.svh, b.svh, pre[0] }, &o, s);
        _ = mlx.mlx_array_free(o[1]);
        _ = mlx.mlx_array_free(o[2]);
        return o[0];
    }
    const cfg = try ks.buildConfig(.{ .grid = grid, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &y_shape, &z_shape }, .out_dtypes = &.{ xd, .float32 }, .template = &tmpl });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [2]mlx.mlx_array = undefined;
    try ks.applyConfig(try kernelOf(.jdec0), cfg, &.{ pre[0], pre[1], a.trellis, b.trellis, a.svh, b.svh, pre[2] }, &o, s);
    _ = mlx.mlx_array_free(o[1]);
    return o[0];
}

// ── tests (CPU) ──

const testing = std.testing;

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

test "turboexl3 ling pair texts: dense's prep_sc and DEC-ONE texts verbatim around the member choice, the tile read in the member's trellis; each operand bound by name" {
    // the pair's prepare: the member choice, then prep_sc's text verbatim
    try testing.expectEqualStrings(text_jprep_sc, jprep ++ dense.text_prep_sc);
    // the GEMVs: dense's DEC-ONE texts with the member chosen after their head, Ripple's member tile address in place of
    // dense's tile
    try testing.expect(std.mem.startsWith(u8, text_jdec0, dense.dec_head ++ jrows ++ jw ++ dense.dec_gemv0));
    try testing.expect(std.mem.startsWith(u8, text_jdec1, dense.dec_head ++ jsuh ++ jw ++ dense.dec_prep1 ++ dense.dec_gemv1));
    try testing.expectEqual(dense.text_dec0.len + jrows.len + jw.len - dense.dec_tile.len + ricursive.jtile.len, text_jdec0.len);
    try testing.expectEqual(dense.text_dec1.len + jsuh.len + jw.len - dense.dec_tile.len + ricursive.jtile.len, text_jdec1.len);
    for ([_][:0]const u8{ text_jdec0, text_jdec1 }) |txt| {
        try testing.expectEqual(@as(usize, 0), count(txt, dense.dec_tile));
        try testing.expectEqual(@as(usize, 1), count(txt, ricursive.jtile));
    }
    // every operand and template constant the launches bind is named in the text
    for (jprep_in ++ jprep_out) |n| try testing.expect(count(text_jprep_sc, std.mem.span(n)) > 0);
    for (jdec0_in ++ jdec0_out) |n| try testing.expect(count(text_jdec0, std.mem.span(n)) > 0);
    for (jdec1_in ++ jdec1_out) |n| try testing.expect(count(text_jdec1, std.mem.span(n)) > 0);
    for ([_][]const u8{ "DIM", "LANE_ORDER", "NCNT", "NTH", "ROWS" }) |n| try testing.expect(count(text_jprep_sc, n) > 0);
    for ([_][:0]const u8{ text_jdec0, text_jdec1 }) |txt| for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "IN", "OUT", "KS", "BT", "RB", "ROWS", "DEFER", "OUT_T", "OB1" }) |n| try testing.expect(count(txt, n) > 0);
    // MLX adds a shape / strides / ndim buffer for an input name followed by those suffixes
    for ([_][:0]const u8{ text_jprep_sc, text_jdec0, text_jdec1 }) |txt| for (jprep_in ++ jprep_out ++ jdec1_in ++ jdec1_out ++ jdec0_in) |n| for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| {
        var buf: [24]u8 = undefined;
        try testing.expectEqual(@as(usize, 0), count(txt, std.fmt.bufPrint(&buf, "{s}{s}", .{ std.mem.span(n), suf }) catch unreachable));
    };
}

test "turboexl3 ling pair: Ling's pairs share their members' K split (z | a 2, the shared gate | up 8, the dense gate | up 1); qkv beside z or a and q_a beside kv_a do not" {
    try testing.expectEqual(@as(u32, 2), dense.Dense.ksFor(4096, 8192));
    try testing.expectEqual(@as(u32, 8), dense.Dense.ksFor(4096, 1280));
    try testing.expectEqual(@as(u32, 1), dense.Dense.ksFor(4096, 10240));
    try testing.expectEqual(@as(u32, 1), dense.Dense.ksFor(4096, 24576));
    try testing.expectEqual(@as(u32, 8), dense.Dense.ksFor(4096, 1536));
    try testing.expectEqual(@as(u32, 16), dense.Dense.ksFor(4096, 640));
}
