//! Two linears of one input x in one DEC-ONE launch, each member at its own K split (GLM-5.3-Flash's DSA q_a and kv_a:
//! 4096 -> 1536 at K split 8 and 4096 -> 512 at 32; the plugin's TURBOEXL3_GLM53F_SMALLOP_FUSE). `shared_gu`'s launch
//! (dense's DEC-ONE text with the prepare inside, the member chosen over the two trellises as they lie, ricursive's tile
//! address in the member's trellis, the finish's value stored in the member's own output) with two pieces in place:
//! - `kernels/pair_ks/ks_head.metal` (for `dense.dec_head`): the members' threadgroups lie flat along x, member 0's
//!   (its output blocks' column groups, slice by slice) then member 1's, and each takes its member's slice, slice count,
//!   tiles a slice, inputs a slice and arrivals;
//! - `kernels/pair_ks/ks_prep1_2.metal` (for `dense.dec_prep1_2`, dense's prepare's store): each prepared input at its
//!   offset in the member's slice, in a buffer whose row stride is the wider member's slice.
//! The rest is dense's pieces as they are. So each member's threadgroups run its own launch's prepare, K slices, slice
//! order and finish (its own K split, suh and svh) and each output holds its member's `root.linear` bytes. Rows: decode
//! rows (1 to `decode.Kernels.max_rows`) where each member's prepare fits threadgroup memory (`Dense.decPrepFits` at its
//! own K split; the buffer is the wider member's); a call it does not serve returns error.Exl3EngineNotImplemented before
//! any launch, and the caller runs `root.linear` per member.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const root = @import("../../root.zig");
const ricursive = @import("../ricursive/ricursive.zig");
const shared_gu = @import("shared_gu.zig");

/// `dense.dec_head` for two members at their own K splits, laid out flat along x
pub const ks_head = @embedFile("kernels/pair_ks/ks_head.metal");
/// `dense.dec_prep1`'s store with the input's offset in the member's slice (`MSPAN`), the buffer's row stride the wider
/// slice's (`SPAN`)
pub const ks_prep1_2 = @embedFile("kernels/pair_ks/ks_prep1_2.metal");
/// `dense.dec_prep1` with that store
pub const ks_prep1 = dense.dec_prep1_1 ++ ks_prep1_2 ++ dense.dec_prep1_3;
/// two linears of one input x over two trellises, each at its own K split, in one DEC-ONE launch: `shared_gu.text_gu`
/// with `ks_head` and `ks_prep1` in place of dense's head and prepare
pub const text_ks: [:0]const u8 = ks_head ++ shared_gu.jsel_pair ++ ks_prep1 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive_1 ++ ricursive.jtile ++ dense.dec_gemv_arrive_2 ++ dense.dec_xsum1 ++ dense.dec_finish_v ++ shared_gu.jstore_pair;

const ks_in = [_][*:0]const u8{ "x", "suh0", "suh1", "w0", "w1", "svh", "cnt" };
const ks_out = [_][*:0]const u8{ "y0", "y1", "z", "xsx" };

const kernel_lock = Global(std.atomic.Value(u32), "glm53f_pair_ks_lock", std.atomic.Value(u32).init(0));
const kernel = Global(?mlx.mlx_fast_metal_kernel, "glm53f_pair_ks_kernel", null);

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernel };

/// The launch's kernel, made at its first call
fn kernelOf() !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    if (kernel.get().*) |made| return made;
    const made = ks.newKernel("turboexl3_glm53f_pair_ks", &ks_in, &ks_out, text_ks, decode.header5, true) orelse return error.KernelCreateFailed;
    kernel.get().* = made;
    return made;
}

/// The launch's plan at `rows` rows, or null where it does not serve: decode rows, each member's prepare fitting
/// threadgroup memory at its own K split (`ks0`, `ks1`: each member's `Dense.ksFor`); `ng0`, `ng1` each member's
/// threadgroups a K slice and row chunk
pub const Plan = struct { ks0: u32, ks1: u32, rb: u32, bt: u32, chunks: u32, ng0: u32, ng1: u32, ncnt: u32, nx: u32 };

pub fn plan(p: shared_gu.Pair, dec: format.Decode, rows: usize) ?Plan {
    if (rows < 1 or rows > decode.Kernels.max_rows) return null;
    if (p.out0 % 128 != 0 or p.out1 % 128 != 0 or p.in % 128 != 0 or (p.in / 16) % 4 != 0) return null;
    const rows_u: u32 = @intCast(rows);
    const rb = dense.Dense.rbFor(rows_u);
    const bt = dense.Dense.btFor(rb);
    const rate = format.Rate.fromN(p.n) orelse return null;
    const ks0 = dense.Dense.ksFor(p.in, p.out0);
    const ks1 = dense.Dense.ksFor(p.in, p.out1);
    // each member's prepare as its own launch takes it (its slice whole 128-blocks, its rows in threadgroup memory); the
    // one buffer is the wider member's (the smaller K split's), which its own check covers
    for ([_]u32{ p.out0, p.out1 }, [_]u32{ ks0, ks1 }) |o, k| {
        const sh: dense.Shape = .{ .in = p.in, .out = o, .rate = rate, .dec = dec, .out_dtype = .float32 };
        if (!dense.Dense.decPrepFits(sh, k, rb, bt)) return null;
    }
    const chunks = (rows_u + rb - 1) / rb;
    const nob = (p.out0 + p.out1) / 128;
    const nb = p.in / 128;
    const gc = 8 / bt;
    return .{
        .ks0 = ks0,
        .ks1 = ks1,
        .rb = rb,
        .bt = bt,
        .chunks = chunks,
        .ng0 = (p.out0 / 128) * gc,
        .ng1 = (p.out1 / 128) * gc,
        .ncnt = @max(nob * chunks, 8),
        .nx = if (dense.Dense.deferFor(dec)) @max(nob * chunks * rb * nb, 8) else 8,
    };
}

/// m0 and m1 of x [..., in] in one launch: [m0's y [..., out0], m1's y [..., out1]] in x's dtype (bf16, f16 or f32),
/// each row-contiguous and `root.linear`'s bytes of its member; `svh` is `shared_gu.joinSvh(m0.svh, m1.svh)`. DEC-ONE's
/// counters (`dec_zero`, member 0's shape's kernel: a launch that waits on nothing) then `text_ks`.
/// error.Exl3EngineNotImplemented where `plan` declines, before any launch.
pub fn linear2(s: mlx.mlx_stream, x: mlx.mlx_array, m0: root.Proj, m1: root.Proj, svh: mlx.mlx_array, dec: format.Decode) ![2]mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return error.BadExl3Dtype;
    const p = try shared_gu.pairOf(m0, m1);
    if (mlx.getShape(svh).len != 1 or mlx.getShape(svh)[0] != @as(c_int, @intCast(p.out0 + p.out1))) return error.BadExl3Shape;
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8 or xsh[xsh.len - 1] != @as(c_int, @intCast(p.in))) return error.BadExl3Shape;
    var rows: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows *= d;
    if (rows < 1) return error.BadExl3Shape;
    const pl = plan(p, dec, @intCast(rows)) orelse return error.Exl3EngineNotImplemented;
    const rate = format.Rate.fromN(p.n) orelse return error.BadExl3Shape;
    const d = try root.denseCached(.{ .in = p.in, .out = p.out0, .rate = rate, .dec = dec, .out_dtype = xd });
    const k = try kernelOf();
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, @intCast(p.in) }, 2, s));
    const out: u32 = p.out0 + p.out1;
    const ncnt_shape = [_]c_int{@intCast(pl.ncnt)};
    const zcfg = try ks.buildConfig(.{ .grid = .{ ((pl.ncnt + 127) / 128) * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&ncnt_shape}, .out_dtypes = &.{.uint32}, .template = &.{.{ .name = "NCNT", .value = .{ .int = @intCast(pl.ncnt) } }} });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(zcfg);
    var cnt: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(d.dec_zero, zcfg, &.{svh}, &cnt, s);
    defer _ = mlx.mlx_array_free(cnt[0]);
    const ks_max = @max(pl.ks0, pl.ks1);
    const cfg = try ks.buildConfig(.{
        .grid = .{ (pl.ng0 * pl.ks0 + pl.ng1 * pl.ks1) * 128, pl.chunks, 1 },
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ rows, @intCast(p.out0) }, &[_]c_int{ rows, @intCast(p.out1) }, &[_]c_int{ @as(c_int, @intCast(ks_max)) * rows, @intCast(out) }, &[_]c_int{@intCast(pl.nx)} },
        .out_dtypes = &.{ xd, xd, .float32, .float32 },
        .template = &.{
            .{ .name = "NHW", .value = .{ .int = @intCast(p.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(dec.window.bits()) } },
            .{ .name = "IN", .value = .{ .int = @intCast(p.in) } },
            .{ .name = "OUT", .value = .{ .int = @intCast(out) } },
            .{ .name = "KS0", .value = .{ .int = @intCast(pl.ks0) } },
            .{ .name = "KS1", .value = .{ .int = @intCast(pl.ks1) } },
            .{ .name = "BT", .value = .{ .int = @intCast(pl.bt) } },
            .{ .name = "RB", .value = .{ .int = @intCast(pl.rb) } },
            .{ .name = "ROWS", .value = .{ .int = rows } },
            .{ .name = "DEFER", .value = .{ .int = if (dense.Dense.deferFor(dec)) 1 else 0 } },
            .{ .name = "OUT_T", .value = .{ .dtype = xd } },
            .{ .name = "OB1", .value = .{ .int = @intCast(p.out0 / 128) } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [4]mlx.mlx_array = undefined;
    try ks.applyConfig(k, cfg, &.{ x2, m0.suh, m1.suh, m0.trellis, m1.trellis, svh, cnt[0] }, &o, s);
    _ = mlx.mlx_array_free(o[2]);
    _ = mlx.mlx_array_free(o[3]);
    defer for (o[0..2]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var osh: [8]c_int = undefined;
    for (xsh, 0..) |v, i| osh[i] = v;
    var ys: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (ys) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&ys, o[0..2], [_]u32{ p.out0, p.out1 }) |*y, ym, om| {
        osh[xsh.len - 1] = @intCast(om);
        try mlx.check(mlx.mlx_reshape(y, ym, osh[0..xsh.len].ptr, @intCast(xsh.len), s));
    }
    return ys;
}

const testing = std.testing;

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

test "turboexl3 glm53f pair ks texts: shared_gu's launch with the head laying the members out flat at their own K splits and the prepare's store at the member's slice; dense's other pieces as they are" {
    // shared_gu's text with ks_head for dense's head and ks_prep1_2 for dense's prepare's store, the rest the same pieces
    // in order
    try testing.expectEqualStrings(dense.dec_prep1, dense.dec_prep1_1 ++ dense.dec_prep1_2 ++ dense.dec_prep1_3);
    try testing.expectEqualStrings(shared_gu.text_gu, dense.dec_head ++ shared_gu.jsel_pair ++ dense.dec_prep1_1 ++ dense.dec_prep1_2 ++ dense.dec_prep1_3 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive_1 ++ ricursive.jtile ++ dense.dec_gemv_arrive_2 ++ dense.dec_xsum1 ++ dense.dec_finish_v ++ shared_gu.jstore_pair);
    try testing.expectEqualStrings(text_ks, ks_head ++ shared_gu.jsel_pair ++ dense.dec_prep1_1 ++ ks_prep1_2 ++ dense.dec_prep1_3 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive_1 ++ ricursive.jtile ++ dense.dec_gemv_arrive_2 ++ dense.dec_xsum1 ++ dense.dec_finish_v ++ shared_gu.jstore_pair);
    // the pin file: [0] dense's store at the slice's span and [1] the pair's at the member's, each once in its piece and
    // nowhere else in the texts; [2..] the head's per-member names, each defined once in ks_head; dense's head defines a
    // single K split ([last]) and reads the slice from the grid's z, ks_head neither
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/pair_ks_texts.metal"));
    try testing.expectEqual(@as(usize, 1), count(dense.dec_prep1_2, pin[0]));
    try testing.expectEqual(@as(usize, 1), count(ks_prep1_2, pin[1]));
    try testing.expectEqual(@as(usize, 0), count(text_ks, pin[0]));
    try testing.expectEqual(@as(usize, 0), count(shared_gu.text_gu, pin[1]));
    try testing.expectEqual(dense.dec_prep1_2.len, ks_prep1_2.len - 1);
    for (pin[2 .. pin.len - 2]) |p| try testing.expectEqual(@as(usize, 1), count(ks_head, p));
    for (pin[pin.len - 2 ..]) |p| {
        try testing.expectEqual(@as(usize, 1), count(dense.dec_head, p));
        try testing.expectEqual(@as(usize, 0), count(ks_head, p));
    }
    // one tile address, the member's
    try testing.expectEqual(@as(usize, 1), count(text_ks, ricursive.jtile));
    try testing.expectEqual(@as(usize, 0), count(text_ks, dense.dec_tile));
}

test "turboexl3 glm53f pair ks plan: a DSA layer's q_a and kv_a at their own K splits 8 and 32 at every decode row; 0 and 17 rows decline" {
    try testing.expectEqual(@as(u32, 8), dense.Dense.ksFor(4096, 1536));
    try testing.expectEqual(@as(u32, 32), dense.Dense.ksFor(4096, 512));
    for ([_]u32{ 96, 64 }) |n| {
        const p: shared_gu.Pair = .{ .in = 4096, .out0 = 1536, .out1 = 512, .n = n };
        for (1..decode.Kernels.max_rows + 1) |rows| {
            const pl = plan(p, .mul1, rows) orelse return error.TestUnexpectedResult;
            try testing.expectEqual(@as(u32, 8), pl.ks0);
            try testing.expectEqual(@as(u32, 32), pl.ks1);
            try testing.expectEqual(dense.Dense.rbFor(@intCast(rows)), pl.rb);
            const gc = 8 / pl.bt;
            try testing.expectEqual(12 * gc, pl.ng0);
            try testing.expectEqual(4 * gc, pl.ng1);
        }
        try testing.expectEqual(@as(?Plan, null), plan(p, .mul1, 0));
        try testing.expectEqual(@as(?Plan, null), plan(p, .mul1, decode.Kernels.max_rows + 1));
    }
    // equal K splits (shared_gu's gate | up) take it too: the layout is shared_gu's grid's, flat
    const pe = plan(.{ .in = 4096, .out0 = 2048, .out1 = 2048, .n = 96 }, .mul1, 3) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(pe.ks0, pe.ks1);
}
