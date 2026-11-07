//! GLM-5.3-Flash's shared expert gate | up in one DEC-ONE launch (the plugin's TURBOEXL3_GLM53F_SHARED_GU): the gate and
//! the up read one input x, so one launch of DEC-ONE with the prepare inside covers both (`text_gu`: dense's
//! `text_dec1` with the member chosen after its head, `jsel_pair.metal`, ricursive's `jtile.metal` reading the member's
//! own trellis, and its finish storing into the member's own output, `jstore_pair.metal`). Unlike ricursive's joint,
//! each trellis stays where it lies (two inputs, no copy, nothing to join at load), only the two svh are joined along
//! OUT (`joinSvh`: 2 x 2048 halves a layer), and the gate's and the up's outputs come back as two row-contiguous arrays,
//! as each member's `linear` returns it (the SwiGLU after them reads them as they are, with no copy).
//!
//! Each member keeps its own K split (`Dense.ksFor(in, out)` of one member: 8 at 4096 -> 2048, where the joined width
//! 4096 -> 4096 takes 4, another sum order), its own suh in its prepare and its own svh in its finish, so each output
//! holds its member's `root.linear` bytes (DEC-ONE's text after the member is chosen: the same prepare, K split, slice
//! order and finish per output block, the same value stored). Rows: decode rows (1 to `decode.Kernels.max_rows`) where
//! the joined prepare fits threadgroup memory (`Dense.decPrepFits`); a call it does not serve returns
//! error.Exl3EngineNotImplemented before any launch, and the caller runs `root.linear` per member.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const root = @import("../../root.zig");
const ricursive = @import("../ricursive/ricursive.zig");

/// the member choice over two trellises (ricursive's `jsel` over its joint, with each member's own trellis)
pub const jsel_pair = @embedFile("kernels/shared_gu/jsel_pair.metal");
/// `dense.dec_finish_store`'s value stored in the threadgroup's member's own output (y0 the gate's, y1 the up's)
pub const jstore_pair = @embedFile("kernels/shared_gu/jstore_pair.metal");
/// two linears of one input x over two trellises in one DEC-ONE launch: `dense.text_dec1` with the member chosen after its
/// head, the GEMV's tile read in that member's trellis (ricursive's `jtile`, from the trellis's start) and the finish's
/// value stored in that member's output
pub const text_gu: [:0]const u8 = dense.dec_head ++ jsel_pair ++ dense.dec_prep1 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive_1 ++ ricursive.jtile ++ dense.dec_gemv_arrive_2 ++ dense.dec_xsum1 ++ dense.dec_finish_v ++ jstore_pair;

const gu_in = [_][*:0]const u8{ "x", "suh0", "suh1", "w0", "w1", "svh", "cnt" };
const gu_out = [_][*:0]const u8{ "y0", "y1", "z", "xsx" };

const kernel_lock = Global(std.atomic.Value(u32), "glm53f_shared_gu_lock", std.atomic.Value(u32).init(0));
const kernel = Global(?mlx.mlx_fast_metal_kernel, "glm53f_shared_gu_kernel", null);

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernel };

/// The launch's kernel, made at its first call
fn kernelOf() !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    if (kernel.get().*) |made| return made;
    const made = ks.newKernel("turboexl3_glm53f_shared_gu", &gu_in, &gu_out, text_gu, decode.header5, true) orelse return error.KernelCreateFailed;
    kernel.get().* = made;
    return made;
}

/// The two members' svh joined along OUT (member 0's then member 1's), evaluated: the launch's `svh`. Made once a layer
/// by the caller, who frees it; the members' own svh stay theirs.
pub fn joinSvh(s: mlx.mlx_stream, svh0: mlx.mlx_array, svh1: mlx.mlx_array) !mlx.mlx_array {
    if (mlx.getShape(svh0).len != 1 or mlx.getShape(svh1).len != 1) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(svh0) != .float16 or mlx.mlx_array_dtype(svh1) != .float16) return error.BadExl3Dtype;
    const both = [_]mlx.mlx_array{ svh0, svh1 };
    const vec = mlx.mlx_vector_array_new_data(&both, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_concatenate_axis(&r, vec, 0, s));
    try mlx.check(mlx.mlx_array_eval(r));
    return r;
}

/// The two members' geometry: one input width, one rate, one trellis dtype, each suh [in] and svh [out_m] f16
pub const Pair = struct { in: u32, out0: u32, out1: u32, n: u32 };

/// `gate` and `up`'s `Pair`, or the error that says why they do not pair
pub fn pairOf(gate: root.Proj, up: root.Proj) !Pair {
    const ta = mlx.getShape(gate.trellis);
    const tb = mlx.getShape(up.trellis);
    if (ta.len != 3 or tb.len != 3 or ta[0] != tb[0] or ta[2] != tb[2]) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(gate.trellis) != mlx.mlx_array_dtype(up.trellis)) return error.BadExl3Dtype;
    const p: Pair = .{ .in = @intCast(ta[0] * 16), .out0 = @intCast(ta[1] * 16), .out1 = @intCast(tb[1] * 16), .n = @intCast(ta[2]) };
    for ([_]root.Proj{ gate, up }, [_]u32{ p.out0, p.out1 }) |m, o| {
        if (mlx.getShape(m.suh).len != 1 or mlx.getShape(m.suh)[0] != @as(c_int, @intCast(p.in))) return error.BadExl3Shape;
        if (mlx.getShape(m.svh).len != 1 or mlx.getShape(m.svh)[0] != @as(c_int, @intCast(o))) return error.BadExl3Shape;
        if (mlx.mlx_array_dtype(m.suh) != .float16 or mlx.mlx_array_dtype(m.svh) != .float16) return error.BadExl3Dtype;
    }
    return p;
}

/// The launch's plan at `rows` rows (ricursive's `jointPlan` for two trellises), or null where it does not serve: decode
/// rows, both members at one K split (each its own `ksFor`), the prepare inside fitting threadgroup memory.
pub const Plan = struct { ks: u32, rb: u32, bt: u32, chunks: u32, ncnt: u32, nx: u32 };

pub fn plan(p: Pair, dec: format.Decode, rows: usize) ?Plan {
    if (rows < 1 or rows > decode.Kernels.max_rows) return null;
    if (p.out0 % 128 != 0 or p.out1 % 128 != 0 or p.in % 128 != 0 or (p.in / 16) % 4 != 0) return null;
    const ksv = dense.Dense.ksFor(p.in, p.out0);
    if (dense.Dense.ksFor(p.in, p.out1) != ksv) return null;
    const rows_u: u32 = @intCast(rows);
    const rb = dense.Dense.rbFor(rows_u);
    const bt = dense.Dense.btFor(rb);
    const rate = format.Rate.fromN(p.n) orelse return null;
    const sh: dense.Shape = .{ .in = p.in, .out = p.out0 + p.out1, .rate = rate, .dec = dec, .out_dtype = .float32 };
    if (!dense.Dense.decPrepFits(sh, ksv, rb, bt)) return null;
    const chunks = (rows_u + rb - 1) / rb;
    const nob = (p.out0 + p.out1) / 128;
    const nb = p.in / 128;
    return .{
        .ks = ksv,
        .rb = rb,
        .bt = bt,
        .chunks = chunks,
        .ncnt = @max(nob * chunks, 8),
        .nx = if (dense.Dense.deferFor(dec)) @max(nob * chunks * rb * nb, 8) else 8,
    };
}

/// gate and up of x [..., in] in one launch: [gate's y [..., out0], up's y [..., out1]] in x's dtype (bf16, f16 or f32),
/// each row-contiguous and `root.linear`'s bytes of its member; `svh` is `joinSvh(gate.svh, up.svh)`. DEC-ONE's counters
/// (`dec_zero`, the member shape's own kernel: a launch that waits on nothing) then `text_gu`.
/// error.Exl3EngineNotImplemented where `plan` declines, before any launch.
pub fn gateUp(s: mlx.mlx_stream, x: mlx.mlx_array, gate: root.Proj, up: root.Proj, svh: mlx.mlx_array, dec: format.Decode) ![2]mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return error.BadExl3Dtype;
    const p = try pairOf(gate, up);
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
    const cfg = try ks.buildConfig(.{
        .grid = .{ (out / 16 / pl.bt) * 128, pl.chunks, pl.ks },
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ rows, @intCast(p.out0) }, &[_]c_int{ rows, @intCast(p.out1) }, &[_]c_int{ @as(c_int, @intCast(pl.ks)) * rows, @intCast(out) }, &[_]c_int{@intCast(pl.nx)} },
        .out_dtypes = &.{ xd, xd, .float32, .float32 },
        .template = &.{
            .{ .name = "NHW", .value = .{ .int = @intCast(p.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(dec.window.bits()) } },
            .{ .name = "IN", .value = .{ .int = @intCast(p.in) } },
            .{ .name = "OUT", .value = .{ .int = @intCast(out) } },
            .{ .name = "KS", .value = .{ .int = @intCast(pl.ks) } },
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
    try ks.applyConfig(k, cfg, &.{ x2, gate.suh, up.suh, gate.trellis, up.trellis, svh, cnt[0] }, &o, s);
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

test "turboexl3 glm53f shared gu texts: dense's DEC-ONE with the prepare inside, the member chosen over two trellises after its head, the tile read in the member's own trellis and the finish's value stored in the member's own output" {
    // ricursive's joined text with jsel_pair in place of jsel and jstore_pair in place of dense's store: the same pieces
    // in the same order otherwise
    const head = dense.dec_head.len;
    try testing.expectEqual(ricursive.text_jdec1.len - ricursive.jsel.len + jsel_pair.len - dense.dec_finish_store.len + jstore_pair.len, text_gu.len);
    try testing.expectEqualStrings(dense.dec_head ++ jsel_pair, text_gu[0 .. head + jsel_pair.len]);
    const jbody = ricursive.text_jdec1[head + ricursive.jsel.len .. ricursive.text_jdec1.len - dense.dec_finish_store.len];
    try testing.expectEqualStrings(jbody, text_gu[head + jsel_pair.len .. text_gu.len - jstore_pair.len]);
    try testing.expect(std.mem.endsWith(u8, ricursive.text_jdec1, dense.dec_finish_store));
    try testing.expect(std.mem.endsWith(u8, text_gu, jstore_pair));
    // one tile address, the member's; dense's own (the one trellis's) nowhere
    try testing.expectEqual(@as(usize, 1), count(text_gu, ricursive.jtile));
    try testing.expectEqual(@as(usize, 0), count(text_gu, dense.dec_tile));
    // the pin file: the member's trellis chosen with its suh and read from its own start, and the stored value, which is
    // dense's store's
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/shared_gu_texts.metal"));
    for (pin[0..3]) |p| try testing.expectEqual(@as(usize, 1), count(jsel_pair, p));
    try testing.expectEqual(@as(usize, 1), count(dense.dec_finish_store, pin[3]));
    try testing.expectEqual(@as(usize, 1), count(jstore_pair, pin[3]));
}

test "turboexl3 glm53f shared gu plan: the shared expert's gate | up at each member's K split 8 (the joined width's own would be 4), every decode row where the prepare fits; 17 rows and members of other K splits decline" {
    try testing.expectEqual(@as(u32, 8), dense.Dense.ksFor(4096, 2048));
    try testing.expectEqual(@as(u32, 4), dense.Dense.ksFor(4096, 4096));
    for ([_]u32{ 96, 64 }) |n| {
        const p: Pair = .{ .in = 4096, .out0 = 2048, .out1 = 2048, .n = n };
        for (1..decode.Kernels.max_rows + 1) |rows| {
            const pl = plan(p, .mul1, rows) orelse return error.TestUnexpectedResult;
            try testing.expectEqual(@as(u32, 8), pl.ks);
            try testing.expectEqual(dense.Dense.rbFor(@intCast(rows)), pl.rb);
        }
        try testing.expectEqual(@as(?Plan, null), plan(p, .mul1, 0));
        try testing.expectEqual(@as(?Plan, null), plan(p, .mul1, decode.Kernels.max_rows + 1));
    }
    // a member of another width at another K split: no joined launch
    try testing.expectEqual(@as(?Plan, null), plan(.{ .in = 4096, .out0 = 2048, .out1 = 12288, .n = 96 }, .mul1, 1));
}
