//! The banked decode (`root.moeClampedBanked`): a call's routed pairs over an expert streamer's three slot banks (base,
//! ext, transient), each pair a top-1 row-slot whose id is packed (bank << 24 | slot row), y f32 [rows, hidden] the
//! pairs' expert outputs at weight 1. CPU (always): each banked text is its base text with the bank's lines in place of
//! the base's and nothing else; the refusals, all made before a launch. GPU (TURBOEXL3_GPU=1, else
//! skipped), at DeepSeek-V4.1's routed geometry (hidden 5120, expert width 2304, swiglu_limit 10, MUL1) over three
//! synthetic banks of 20, 12 and 8 experts at K3 and K2:
//! - every pair's row is `root.moeClamped`'s on its bank at top-1 with score 1, byte for byte, at rows 1, 2, 6, 7, 16, 36
//!   and 48; x bf16 and f32, the trellis U16 and I16 (an expert streamer's code arrays); the pairs in tokens of six (pair p
//!   on token p / 6's row, as the streamer's device wave lays a call out), their ids drawn over the banks with, from 4
//!   rows, a repeated id (the GEMVs pair those row-slots) and one slot row in two banks (never paired), and from 8 rows a
//!   run of base row 0 (DEVROUTE's rows for its misses);
//! - the sum the streamer serves, each full token's pairs weighed and summed, equals the same sum of the clamped host
//!   reference's pairs (`ref.moeRowClamped`, one slot at score 1) by clamped_test's criterion for moeClamped's rows at
//!   this geometry (f32: `ref.compare`, 5e-3 of the reference's rms); x ~ N(0, 8^2) so that the clamp acts.
//! TURBOEXL3_BANKED_N (rates in halfwords per tile; default 48,32) and TURBOEXL3_BANKED_ROWS narrow it. With
//! TURBOEXL3_BANKED_TIME=1 a second test times the banked call, and each half, against `moeClamped`'s launches over the
//! same number of pairs (top-1, today's per-bank calls; top-6, the routed layout) on banks of 48, 16 and 8 experts.
//! Lines: TURBOEXL3_BANKED_BYTES, TURBOEXL3_BANKED_PARITY, TURBOEXL3_BANKED_TIME.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");

const testing = std.testing;

const hidden: u32 = 5120;
const inter: u32 = 2304;
const limit: c_int = 10;
const sigma: f32 = 8;
/// a token's routed pairs (DeepSeek-V4.1's top-6)
const top_k: u32 = 6;
/// the parity test's banks: base, ext, transient
const bank_experts = [3]u32{ 20, 12, 8 };

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// `v` rounded to bf16 (as MLX rounds), back in f32: a value both sides read alike.
fn bfExact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
}

/// A comma list of numbers from the environment (at most buf.len of them), else `default`.
fn envList(name: [*:0]const u8, default: []const u32, buf: []u32) []const u32 {
    const v = std.c.getenv(name) orelse return default;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
    while (it.next()) |tok| {
        if (n == buf.len) break;
        buf[n] = std.fmt.parseInt(u32, std.mem.trim(u8, tok, " "), 10) catch continue;
        n += 1;
    }
    return if (n == 0) default else buf[0..n];
}

test "turboexl3 banked decode texts: each is its base text with the bank's lines in place of the expert line or the weight base and nothing else; every banked input is read, none asks MLX for a shape buffer" {
    const decode = @import("../../core/decode.zig");
    const expert_line = @embedFile("../../core/kernels/decode/expert_line.metal");
    const Pair = struct { base: []const u8, banked: []const u8, line: []const u8, piece: []const u8, inputs: []const [*:0]const u8 };
    const pairs = [_]Pair{
        .{ .base = decode.text_prep2, .banked = decode.text_prep2b, .line = expert_line, .piece = @embedFile("../../core/kernels/decode/prep2b_expert.metal"), .inputs = &decode.prep2b_inputs },
        .{ .base = decode.text_gu3, .banked = decode.text_gu3b, .line = @embedFile("../../core/kernels/decode/gu3_w.metal"), .piece = @embedFile("../../core/kernels/decode/gu3b_w.metal"), .inputs = &decode.gu3b_inputs },
        .{ .base = decode.text_fin7c, .banked = decode.text_fin7cb, .line = expert_line, .piece = @embedFile("../../core/kernels/decode/fin7b_expert.metal"), .inputs = &decode.fin7b_inputs },
        .{ .base = decode.text_down3, .banked = decode.text_down3b, .line = @embedFile("../../core/kernels/decode/down3_w.metal"), .piece = @embedFile("../../core/kernels/decode/down3b_w.metal"), .inputs = &decode.down3b_inputs },
        .{ .base = decode.text_dfin3, .banked = decode.text_dfin3b, .line = @embedFile("../../core/kernels/decode/dfin3_expert.metal"), .piece = @embedFile("../../core/kernels/decode/dfin3b_expert.metal"), .inputs = &decode.dfin3b_inputs },
    };
    for (pairs) |p| {
        // the base reads the line once; the banked text has the piece once in its place and the base's bytes around it
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, p.base, p.line));
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, p.banked, p.piece));
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, p.banked, p.line));
        const i = std.mem.indexOf(u8, p.base, p.line).?;
        try testing.expectEqual(i, std.mem.indexOf(u8, p.banked, p.piece).?);
        try testing.expectEqualStrings(p.base[0..i], p.banked[0..i]);
        try testing.expectEqualStrings(p.base[i + p.line.len ..], p.banked[i + p.piece.len ..]);
        // every input is named in the text; MLX adds a shape / strides / ndim buffer for an input name followed by those
        // suffixes, and the banked texts take none
        for (p.inputs) |n| {
            try testing.expect(std.mem.indexOf(u8, p.banked, std.mem.span(n)) != null);
            for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| {
                var buf: [32]u8 = undefined;
                try testing.expect(std.mem.indexOf(u8, p.banked, try std.fmt.bufPrint(&buf, "{s}{s}", .{ std.mem.span(n), suf })) == null);
            }
        }
        // within Metal's 31 buffers (at most 4 outputs a launch)
        try testing.expect(p.inputs.len + 4 <= 31);
    }
    // the slot-row lines take the id's low 24 bits and its bank from the top byte; the GEMVs keep the packed id for their
    // members (two row-slots share a decode only on one bank's one slot) and read the trellis at the slot row
    const pin = comptime support.pins(@embedFile("kernels/test/pins/decode_banked.metal"));
    for ([_][]const u8{ pairs[0].piece, pairs[2].piece, pairs[4].piece }) |piece| {
        try testing.expect(std.mem.indexOf(u8, piece, pin[0]) != null);
        try testing.expect(std.mem.indexOf(u8, piece, pin[1]) != null);
    }
    for ([_][]const u8{ pairs[1].piece, pairs[3].piece }) |piece| {
        try testing.expect(std.mem.indexOf(u8, piece, pin[2]) != null);
        try testing.expect(std.mem.indexOf(u8, piece, pin[3]) != null);
    }
    try testing.expect(std.mem.indexOf(u8, @embedFile("../../core/kernels/decode/gu3_2.metal"), pin[4]) != null);
    try testing.expect(std.mem.indexOf(u8, @embedFile("../../core/kernels/decode/down3_2.metal"), pin[4]) != null);
}

/// A zero host array of `shape` in `dt` (the refusals read shapes and dtypes only).
fn zeroArray(a: std.mem.Allocator, g: *G, shape: []const c_int, dt: mlx.mlx_dtype) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const size: usize = switch (dt) {
        .uint16, .int16, .float16, .bfloat16 => 2,
        else => 4,
    };
    const bytes = try a.alloc(u8, n * size);
    defer a.free(bytes);
    @memset(bytes, 0);
    return g.hostArray(bytes, shape, dt);
}

/// A bank of `experts` at hidden `h`, width `w` and `n` halfwords per tile, its trellis in `td`, suh / svh in `sd`.
fn zeroBank(a: std.mem.Allocator, g: *G, experts: c_int, h: c_int, w: c_int, n: c_int, td: mlx.mlx_dtype, sd: mlx.mlx_dtype) !root.Bank {
    return .{
        .gate = .{ .trellis = try zeroArray(a, g, &.{ experts, @divExact(h, 16), @divExact(w, 16), n }, td), .suh = try zeroArray(a, g, &.{ experts, h }, sd), .svh = try zeroArray(a, g, &.{ experts, w }, sd) },
        .up = .{ .trellis = try zeroArray(a, g, &.{ experts, @divExact(h, 16), @divExact(w, 16), n }, td), .suh = try zeroArray(a, g, &.{ experts, h }, sd), .svh = try zeroArray(a, g, &.{ experts, w }, sd) },
        .down = .{ .trellis = try zeroArray(a, g, &.{ experts, @divExact(w, 16), @divExact(h, 16), n }, td), .suh = try zeroArray(a, g, &.{ experts, w }, sd), .svh = try zeroArray(a, g, &.{ experts, h }, sd) },
    };
}

test "turboexl3 banked decode refusals: the clamp, the ids' shape and dtype, the banks' dtypes, widths and rates, x's and the half's shapes and dtypes, each before a launch" {
    const a = testing.allocator;
    var g = try G.init(a, .{ .ctx = null });
    defer g.deinit();
    const s0: mlx.mlx_stream = .{ .ctx = null };
    const H: c_int = 256;
    const W: c_int = 128;
    const ok = try zeroBank(a, &g, 2, H, W, 48, .uint16, .float16);
    const ok_i = try zeroBank(a, &g, 1, H, W, 48, .int16, .float16);
    const banks = [3]root.Bank{ ok, ok, ok };
    const ids = try zeroArray(a, &g, &.{4}, .uint32);
    const x = try zeroArray(a, &g, &.{ 4, H }, .bfloat16);
    const hd = try zeroArray(a, &g, &.{ 4, W }, .float16);
    const hs = try zeroArray(a, &g, &.{ 4, @divExact(W, 128) }, .float32);
    // the clamp, before any operand is read
    const none: mlx.mlx_array = .{ .ctx = null };
    for ([_]c_int{ 0, -10, 129 }) |l| {
        try testing.expectError(error.InvalidExl3Clamp, root.moeClampedBanked(s0, none, &banks, none, .mul1, l));
        try testing.expectError(error.InvalidExl3Clamp, root.moeClampedBankedDown(s0, .{ .hd = none, .hs = none }, &banks, none, .mul1, l));
    }
    // the ids: one dimension of 1..banked_rows_max, u32 or i32
    for ([_][]const c_int{ &.{ 1, 4 }, &.{0}, &.{root.banked_rows_max + 1} }) |sh| {
        try testing.expectError(error.BadExl3Shape, root.moeClampedBankedGateUp(s0, x, &banks, try zeroArray(a, &g, sh, .uint32), .mul1, limit));
    }
    try testing.expectError(error.BadExl3Dtype, root.moeClampedBankedGateUp(s0, x, &banks, try zeroArray(a, &g, &.{4}, .float32), .mul1, limit));
    // the banks: one trellis dtype, U16 or I16 (bank 0's gate sets it), suh / svh F16, one width and rate
    const f16_trellis = try zeroBank(a, &g, 1, H, W, 48, .float16, .float16);
    const f32_scales = try zeroBank(a, &g, 1, H, W, 48, .uint16, .float32);
    const wide = try zeroBank(a, &g, 1, H, 2 * W, 48, .uint16, .float16);
    const k2 = try zeroBank(a, &g, 1, H, W, 32, .uint16, .float16);
    const cases = [_]struct { b: [3]root.Bank, e: anyerror }{
        .{ .b = .{ f16_trellis, ok, ok }, .e = error.BadExl3Dtype },
        .{ .b = .{ ok, f16_trellis, ok }, .e = error.BadExl3Dtype },
        .{ .b = .{ ok, ok_i, ok }, .e = error.BadExl3Dtype },
        .{ .b = .{ ok, ok, f32_scales }, .e = error.BadExl3Dtype },
        .{ .b = .{ ok, wide, ok }, .e = error.BadExl3Shape },
        .{ .b = .{ ok, ok, k2 }, .e = error.BadExl3Shape },
    };
    for (cases) |c| try testing.expectError(c.e, root.moeClampedBankedGateUp(s0, x, &c.b, ids, .mul1, limit));
    // x: [rows, hidden] in bf16, f16 or f32
    try testing.expectError(error.BadExl3Shape, root.moeClampedBankedGateUp(s0, try zeroArray(a, &g, &.{ 3, H }, .bfloat16), &banks, ids, .mul1, limit));
    try testing.expectError(error.BadExl3Shape, root.moeClampedBankedGateUp(s0, try zeroArray(a, &g, &.{ 4, 2 * H }, .bfloat16), &banks, ids, .mul1, limit));
    try testing.expectError(error.BadExl3Shape, root.moeClampedBankedGateUp(s0, try zeroArray(a, &g, &.{ 1, 4, H }, .bfloat16), &banks, ids, .mul1, limit));
    try testing.expectError(error.BadExl3Dtype, root.moeClampedBankedGateUp(s0, try zeroArray(a, &g, &.{ 4, H }, .int32), &banks, ids, .mul1, limit));
    // the half: hd f16 [rows, width], hs f32 [rows, width / 128]
    try testing.expectError(error.BadExl3Shape, root.moeClampedBankedDown(s0, .{ .hd = try zeroArray(a, &g, &.{ 4, 2 * W }, .float16), .hs = hs }, &banks, ids, .mul1, limit));
    try testing.expectError(error.BadExl3Shape, root.moeClampedBankedDown(s0, .{ .hd = hd, .hs = try zeroArray(a, &g, &.{ 3, 1 }, .float32) }, &banks, ids, .mul1, limit));
    try testing.expectError(error.BadExl3Dtype, root.moeClampedBankedDown(s0, .{ .hd = try zeroArray(a, &g, &.{ 4, W }, .float32), .hs = hs }, &banks, ids, .mul1, limit));
    try testing.expectError(error.BadExl3Dtype, root.moeClampedBankedDown(s0, .{ .hd = hd, .hs = try zeroArray(a, &g, &.{ 4, 1 }, .float16) }, &banks, ids, .mul1, limit));
}

/// The bank as `root` takes it, its trellis in `td` (U16, or I16 as an expert streamer's code arrays come: the same words).
fn rootBank(g: *G, b: *const ref.Bank, td: mlx.mlx_dtype) !root.Bank {
    const E = ci(b.experts);
    const ht = ci(b.hidden / 16);
    const it = ci(b.inter / 16);
    const n = ci(b.rate.n);
    return .{
        .gate = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(b.gate_t), &.{ E, ht, it, n }, td), .suh = try g.hostArray(std.mem.sliceAsBytes(b.gate_suh), &.{ E, ci(b.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.gate_svh), &.{ E, ci(b.inter) }, .float16) },
        .up = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(b.up_t), &.{ E, ht, it, n }, td), .suh = try g.hostArray(std.mem.sliceAsBytes(b.up_suh), &.{ E, ci(b.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.up_svh), &.{ E, ci(b.inter) }, .float16) },
        .down = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(b.down_t), &.{ E, it, ht, n }, td), .suh = try g.hostArray(std.mem.sliceAsBytes(b.down_suh), &.{ E, ci(b.inter) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.down_svh), &.{ E, ci(b.hidden) }, .float16) },
    };
}

/// A call's pairs: row-slot r on bank `bank[r]` at slot row `slot[r]`, drawn over the banks; from 4 rows row-slot 1
/// repeats row-slot 0's id (the GEMVs pair them) and row-slot 2 takes row-slot 0's slot row in the next bank (never
/// paired); from 8 rows the last three take base row 0 (DEVROUTE's rows for its misses).
fn drawPairs(rnd: std.Random, experts: [3]u32, bank: []u32, slot: []u32) void {
    for (bank, slot) |*b, *s| {
        b.* = rnd.uintLessThan(u32, 3);
        s.* = rnd.uintLessThan(u32, experts[b.*]);
    }
    const rows = bank.len;
    if (rows >= 4) {
        // a slot row every bank has
        slot[0] = rnd.uintLessThan(u32, @min(experts[0], @min(experts[1], experts[2])));
        bank[1] = bank[0];
        slot[1] = slot[0];
        bank[2] = (bank[0] + 1) % 3;
        slot[2] = slot[0];
    }
    if (rows >= 8) for (rows - 3..rows) |r| {
        bank[r] = 0;
        slot[r] = 0;
    };
}

/// The elements whose bits differ between two f32 slices of one length, and the non-finite ones in `got`.
fn bitDiff(got: []const f32, want: []const f32) struct { differing: usize, nonfinite: usize } {
    var d: usize = 0;
    var bad: usize = 0;
    for (got, want) |u, v| {
        d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
        bad += @intFromBool(!std.math.isFinite(u));
    }
    return .{ .differing = d, .nonfinite = bad };
}

/// The rows a call's reference covers (it decodes every weight of their experts on the CPU): the first, the middle and
/// the last.
fn sampleRows(rows: u32, buf: *[3]u32) []const u32 {
    buf.* = .{ 0, rows / 2, rows - 1 };
    if (rows <= 2) return buf[0..rows];
    return buf;
}

// DeepSeek-V4.1's routed experts (hidden 5120, width 2304, swiglu_limit 10) over three banks of 20, 12 and 8 experts at K3
// and K2: each pair's row against `moeClamped` on its bank at top-1 with score 1 (16 rows a call, so its decode launches),
// byte for byte, in four arms (x bf16 / f32, trellis I16 / U16); then the bf16 / I16 arm's rows as the streamer serves
// them, each token's pairs weighed (positive, summing to one, bf16-exact: clamped_test's draw) and summed, against the
// same sum of the clamped reference's pairs: clamped_test's criterion for moeClamped's top-6 rows, on its first, middle and
// last full token. Each pair alone against its reference is reported as well (pair_worst, beyond its f16 store), not held to
// that bar: one expert's output is not a sum over slots, and its largest error runs near 5e-3 of its rms at this width.
// x ~ N(0, 8^2), as the clamped test's: about 12% of the gates pass 10 and 25% of the ups pass +-10.
test "turboexl3 gpu banked decode: every pair of moeClampedBanked over three banks is moeClamped's on its bank at top-1 with score 1, byte for byte, and sampled pairs the clamped reference, at DeepSeek-V4.1's routed geometry (K3, K2) (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_BANKED_N", &.{ 48, 32 }, &nbuf);
    var rbuf: [16]u32 = undefined;
    const row_counts = envList("TURBOEXL3_BANKED_ROWS", &.{ 1, 2, 6, 7, 16, 36, 48 }, &rbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0xBA4C3D ^ @as(u64, n));
        const rnd = prng.random();
        var hb: [3]ref.Bank = undefined;
        var made: usize = 0;
        defer for (hb[0..made]) |*b| b.deinit(a);
        for (&hb, bank_experts) |*b, e| {
            b.* = try ref.synthBank(a, rnd, e, hidden, inter, rate);
            made += 1;
        }
        var rb_u: [3]root.Bank = undefined;
        var rb_i: [3]root.Bank = undefined;
        for (&hb, &rb_u, &rb_i) |*b, *u, *i| {
            u.* = try rootBank(&g, b, .uint16);
            i.* = try rootBank(&g, b, .int16);
        }
        for (row_counts) |rows| {
            if (rows == 0 or rows > root.banked_rows_max) continue;
            const m2 = g.mark();
            defer g.resetTo(m2);
            // the pairs in tokens of six: pair p on token p / 6's row, its weight one of the token's
            const tokens = (rows + top_k - 1) / top_k;
            const xt = try a.alloc(f32, tokens * hidden);
            defer a.free(xt);
            for (xt) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
            const x = try a.alloc(f32, rows * hidden);
            defer a.free(x);
            for (0..rows) |p| @memcpy(x[p * hidden ..][0..hidden], xt[(p / top_k) * hidden ..][0..hidden]);
            var wts: [root.banked_rows_max]f32 = undefined;
            for (0..tokens) |t| {
                const tw = wts[t * top_k .. @min(rows, (t + 1) * top_k)];
                var sum: f32 = 0;
                for (tw) |*v| {
                    v.* = 0.05 + rnd.float(f32);
                    sum += v.*;
                }
                for (tw) |*v| v.* = bfExact(v.* / sum);
            }
            var bank_of: [root.banked_rows_max]u32 = undefined;
            var slot_of: [root.banked_rows_max]u32 = undefined;
            drawPairs(rnd, bank_experts, bank_of[0..rows], slot_of[0..rows]);
            var ids: [root.banked_rows_max]u32 = undefined;
            for (ids[0..rows], bank_of[0..rows], slot_of[0..rows]) |*d, b, sl| d.* = (b << 24) | sl;
            const x32 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ ci(rows), ci(hidden) }, .float32);
            const xbf = try g.astype(x32, .bfloat16);
            const ids_a = try g.hostArray(std.mem.sliceAsBytes(ids[0..rows]), &.{ci(rows)}, .uint32);
            // the reference bytes: each bank's pairs through moeClamped at top-1 with score 1, 16 rows a call
            const want = try a.alloc(f32, rows * hidden);
            defer a.free(want);
            const buf = try a.alloc(f32, 16 * hidden);
            defer a.free(buf);
            const ones: [16]f32 = @splat(1.0);
            for (0..3) |b| {
                var pos: [root.banked_rows_max]u32 = undefined;
                var np: usize = 0;
                for (bank_of[0..rows], 0..) |bk, r| if (bk == b) {
                    pos[np] = @intCast(r);
                    np += 1;
                };
                var c0: usize = 0;
                while (c0 < np) : (c0 += 16) {
                    const c = @min(16, np - c0);
                    const P = pos[c0 .. c0 + c];
                    var sl: [16]i32 = undefined;
                    for (P, sl[0..c]) |p, *d| d.* = @intCast(slot_of[p]);
                    const xr = try g.take(x32, try g.hostArray(std.mem.sliceAsBytes(P), &.{ci(c)}, .uint32), 0);
                    const x3 = try g.reshape(xr, &.{ 1, ci(c), ci(hidden) });
                    const inds = try g.hostArray(std.mem.sliceAsBytes(sl[0..c]), &.{ 1, ci(c), 1 }, .int32);
                    const sc = try g.hostArray(std.mem.sliceAsBytes(ones[0..c]), &.{ 1, ci(c), 1 }, .float32);
                    const y = try g.adopt(try root.moeClamped(s, x3, rb_u[b], inds, sc, .mul1, limit));
                    _ = try g.hostF32(y, buf[0 .. c * hidden]);
                    for (P, 0..) |p, i| @memcpy(want[p * hidden ..][0..hidden], buf[i * hidden ..][0..hidden]);
                }
            }
            const got = try a.alloc(f32, rows * hidden);
            defer a.free(got);
            const served = try a.alloc(f32, rows * hidden);
            defer a.free(served);
            const Arm = struct { x: mlx.mlx_array, banks: *const [3]root.Bank, name: []const u8 };
            const arms = [_]Arm{
                .{ .x = xbf, .banks = &rb_i, .name = "x=bf16 trellis=i16" },
                .{ .x = x32, .banks = &rb_u, .name = "x=f32 trellis=u16" },
                .{ .x = xbf, .banks = &rb_u, .name = "x=bf16 trellis=u16" },
                .{ .x = x32, .banks = &rb_i, .name = "x=f32 trellis=i16" },
            };
            for (arms, 0..) |arm, ai| {
                const y = try g.adopt(try root.moeClampedBanked(s, arm.x, arm.banks, ids_a, .mul1, limit));
                try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(hidden) }, mlx.getShape(y));
                try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(y));
                _ = try g.hostF32(y, got);
                if (ai == 0) @memcpy(served, got);
                const d = bitDiff(got, want);
                const ok = d.differing == 0 and d.nonfinite == 0;
                std.debug.print("TURBOEXL3_BANKED_BYTES n={d} rows={d} {s} differing={d}/{d} nonfinite={d} {s}\n", .{ n, rows, arm.name, d.differing, rows * hidden, d.nonfinite, if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            }
            // the served sum: each full token's pair rows (the bf16 / I16 arm's) weighed and summed in f32, as the streamer
            // sums a call's pair rows, against the same sum of the pairs' clamped references
            const full = rows / top_k;
            if (full == 0) {
                std.debug.print("TURBOEXL3_BANKED_PARITY n={d} rows={d} tokens=0 (no full token of {d} pairs: bytes only)\n", .{ n, rows, top_k });
                continue;
            }
            var tbuf: [3]u32 = undefined;
            const toks = sampleRows(full, &tbuf);
            const pair_ref = try a.alloc(f32, hidden);
            defer a.free(pair_ref);
            const want_t = try a.alloc(f32, hidden);
            defer a.free(want_t);
            const got_t = try a.alloc(f32, hidden);
            defer a.free(got_t);
            var hits: ref.Hits = .{};
            var worst: f64 = 0;
            var pair_worst: f64 = 0;
            for (toks) |t| {
                @memset(want_t, 0);
                @memset(got_t, 0);
                for (t * top_k..(t + 1) * top_k) |p| {
                    const h = try ref.moeRowClamped(a, &hb[bank_of[p]], .mul1, @floatFromInt(limit), x[p * hidden ..][0..hidden], &.{slot_of[p]}, &.{1.0}, pair_ref);
                    hits.gate += h.gate;
                    hits.up += h.up;
                    hits.total += h.total;
                    const gp = served[p * hidden ..][0..hidden];
                    pair_worst = @max(pair_worst, ref.compareRounded(gp, pair_ref, 0x1p-11).max_ratio);
                    for (want_t, got_t, pair_ref, gp) |*wv, *gv, rv, sv| {
                        wv.* += wts[p] * rv;
                        gv.* += wts[p] * sv;
                    }
                }
                worst = @max(worst, ref.compare(got_t, want_t).max_ratio);
            }
            const bar: f64 = 5e-3;
            const hit = hits.gate * 100 >= hits.total and hits.up * 100 >= hits.total;
            const ok = hit and worst <= bar;
            std.debug.print("TURBOEXL3_BANKED_PARITY n={d} rows={d} tokens={d} worst_vs_ref={e:.3} bar={e:.1} pair_worst={e:.3} gates_clamped={d}/{d} ups_clamped={d}/{d} {s}\n", .{ n, rows, toks.len, worst, bar, pair_worst, hits.gate, hits.total, hits.up, hits.total, if (ok) "PASS" else "FAIL" });
            if (!ok) failed = true;
        }
    }
    try testing.expect(!failed);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// The timing test's banks: base, ext, transient (72 experts: 48 distinct pairs never repeat one)
const time_experts = [3]u32{ 48, 16, 8 };

/// What a timed arm runs: the banked call, its gate|up alone, its down alone (from an evaluated gate|up), and
/// `moeClamped` over the base bank at the same distinct experts, top-1 (a row per pair: the per-bank calls before the
/// banked route) and top-6 (rows / 6 rows of six pairs: the routed layout, where rows % 6 == 0).
const TimeArm = enum { banked, banked_gu, banked_dn, top1, top6 };

// TURBOEXL3_BANKED_TIME=1: at rows 1, 6, 12, 18, 24, 36, 48 (TURBOEXL3_BANKED_ROWS), the pairs distinct or each id twice
// (the first half's ids again), x bf16 and the trellis U16, 3 warm then 15 timed rounds, the arms interleaved with a
// rotated start; each sample is one call built and evaluated (the host's encode with it). K3 (TURBOEXL3_BANKED_N).
test "turboexl3 gpu banked decode time: the banked call and its halves against moeClamped's top-1 and top-6 launches over the same distinct experts (TURBOEXL3_GPU=1 TURBOEXL3_BANKED_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_BANKED_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_BANKED_N", &.{48}, &nbuf);
    var rbuf: [16]u32 = undefined;
    const row_counts = envList("TURBOEXL3_BANKED_ROWS", &.{ 1, 6, 12, 18, 24, 36, 48 }, &rbuf);
    const warm: usize = 3;
    const timed: usize = 15;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x7BA4C3D ^ @as(u64, n));
        const rnd = prng.random();
        var rb: [3]root.Bank = undefined;
        for (&rb, time_experts) |*b, e| {
            var hb = try ref.synthBank(a, rnd, e, hidden, inter, rate);
            defer hb.deinit(a);
            b.* = try rootBank(&g, &hb, .uint16);
            // the host copy goes with `hb`: the device arrays are made now
            try g.evalAll(&.{ b.gate.trellis, b.gate.suh, b.gate.svh, b.up.trellis, b.up.suh, b.up.svh, b.down.trellis, b.down.suh, b.down.svh });
        }
        for (row_counts) |rows| {
            if (rows == 0 or rows > root.banked_rows_max or rows > time_experts[0]) continue;
            for ([_]bool{ false, true }) |paired| {
                const m2 = g.mark();
                defer g.resetTo(m2);
                // the distinct experts: global index e over the banks (base 0..47, ext 48..63, transient 64..71)
                var order: [72]u32 = undefined;
                for (&order, 0..) |*o, i| o.* = @intCast(i);
                rnd.shuffle(u32, &order);
                const half = (rows + 1) / 2;
                var ids: [root.banked_rows_max]u32 = undefined;
                var single: [root.banked_rows_max]i32 = undefined;
                for (0..rows) |p| {
                    const d = if (paired and p >= half) p - half else p;
                    const e = order[d];
                    const bk: u32 = if (e < 48) 0 else if (e < 64) 1 else 2;
                    const sl: u32 = if (bk == 0) e else if (bk == 1) e - 48 else e - 64;
                    ids[p] = (bk << 24) | sl;
                    // the single-bank arms: the same pattern of distinct experts, on the base bank
                    single[p] = @intCast(d);
                }
                const xh = try a.alloc(f32, rows * hidden);
                defer a.free(xh);
                for (xh) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
                const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(hidden) }, .float32), .bfloat16);
                const ids_a = try g.hostArray(std.mem.sliceAsBytes(ids[0..rows]), &.{ci(rows)}, .uint32);
                const x_top1 = try g.reshape(x, &.{ 1, ci(rows), ci(hidden) });
                const ids_top1 = try g.hostArray(std.mem.sliceAsBytes(single[0..rows]), &.{ 1, ci(rows), 1 }, .int32);
                const ones: [root.banked_rows_max]f32 = @splat(1.0);
                const sc_top1 = try g.hostArray(std.mem.sliceAsBytes(ones[0..rows]), &.{ 1, ci(rows), 1 }, .float32);
                const r6 = rows / 6;
                const six = rows % 6 == 0 and r6 <= root.kernels.DECODE_ROWS_MAX;
                const x_top6 = if (six) try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh[0 .. r6 * hidden]), &.{ 1, ci(r6), ci(hidden) }, .float32), .bfloat16) else x_top1;
                const ids_top6 = if (six) try g.hostArray(std.mem.sliceAsBytes(single[0..rows]), &.{ 1, ci(r6), 6 }, .int32) else ids_top1;
                var sixth: [root.banked_rows_max]f32 = undefined;
                for (&sixth) |*v| v.* = bfExact(1.0 / 6.0);
                const sc_top6 = if (six) try g.hostArray(std.mem.sliceAsBytes(sixth[0..rows]), &.{ 1, ci(r6), 6 }, .float32) else sc_top1;
                // the down arm's input: the gate|up, evaluated once
                const h = try root.moeClampedBankedGateUp(s, x, &rb, ids_a, .mul1, limit);
                const hd = try g.adopt(h.hd);
                const hs = try g.adopt(h.hs);
                try g.evalAll(&.{ x, ids_a, x_top1, ids_top1, sc_top1, x_top6, ids_top6, sc_top6, hd, hs });
                const arms = [_]TimeArm{ .banked, .banked_gu, .banked_dn, .top1, .top6 };
                var run: [arms.len]bool = undefined;
                for (arms, &run) |arm, *r| r.* = switch (arm) {
                    .top1 => rows <= root.kernels.DECODE_ROWS_MAX,
                    .top6 => six,
                    else => true,
                };
                var samples: [arms.len][timed]u64 = undefined;
                for (0..warm + timed) |round| {
                    for (0..arms.len) |k| {
                        const ai = (k + round) % arms.len;
                        if (!run[ai]) continue;
                        const t0 = nowNs();
                        switch (arms[ai]) {
                            .banked => {
                                const y = try root.moeClampedBanked(s, x, &rb, ids_a, .mul1, limit);
                                defer _ = mlx.mlx_array_free(y);
                                try mlx.check(mlx.mlx_array_eval(y));
                            },
                            .banked_gu => {
                                const hh = try root.moeClampedBankedGateUp(s, x, &rb, ids_a, .mul1, limit);
                                defer _ = mlx.mlx_array_free(hh.hd);
                                defer _ = mlx.mlx_array_free(hh.hs);
                                try g.evalAll(&.{ hh.hd, hh.hs });
                            },
                            .banked_dn => {
                                const y = try root.moeClampedBankedDown(s, .{ .hd = hd, .hs = hs }, &rb, ids_a, .mul1, limit);
                                defer _ = mlx.mlx_array_free(y);
                                try mlx.check(mlx.mlx_array_eval(y));
                            },
                            .top1 => {
                                const y = try root.moeClamped(s, x_top1, rb[0], ids_top1, sc_top1, .mul1, limit);
                                defer _ = mlx.mlx_array_free(y);
                                try mlx.check(mlx.mlx_array_eval(y));
                            },
                            .top6 => {
                                const y = try root.moeClamped(s, x_top6, rb[0], ids_top6, sc_top6, .mul1, limit);
                                defer _ = mlx.mlx_array_free(y);
                                try mlx.check(mlx.mlx_array_eval(y));
                            },
                        }
                        const dt = nowNs() - t0;
                        if (round >= warm) samples[ai][round - warm] = dt;
                    }
                }
                const distinct = if (paired) half else rows;
                for (arms, 0..) |arm, ai| {
                    if (!run[ai]) continue;
                    std.mem.sort(u64, &samples[ai], {}, std.sort.asc(u64));
                    const min_us = @as(f64, @floatFromInt(samples[ai][0])) / 1000.0;
                    const med_us = @as(f64, @floatFromInt(samples[ai][timed / 2])) / 1000.0;
                    std.debug.print("TURBOEXL3_BANKED_TIME n={d} rows={d} ids={s} distinct={d} arm={s} min_us={d:.1} median_us={d:.1} median_us_per_distinct={d:.2}\n", .{ n, rows, if (paired) "paired" else "distinct", distinct, @tagName(arm), min_us, med_us, med_us / @as(f64, @floatFromInt(distinct)) });
                }
            }
        }
    }
}

// ── The grouped banked decode (`Config.banked_group` 4) ──

test "turboexl3 grouped banked decode texts: prep2bg, fin7cbg and dfin3bg are the banked texts with the skip line before their bank lines; gu3bg and down3bg take the banked weight base between the four-member pieces" {
    const decode = @import("../../core/decode.zig");
    const skip = @embedFile("../../core/kernels/decode/skip_rk.metal");
    const skip_next = @embedFile("../../core/kernels/decode/skip_rk_next.metal");
    const Ins = struct { banked: []const u8, grouped: []const u8, piece: []const u8, line: []const u8 };
    for ([_]Ins{
        .{ .banked = decode.text_prep2b, .grouped = decode.text_prep2bg, .piece = @embedFile("../../core/kernels/decode/prep2b_expert.metal"), .line = skip },
        .{ .banked = decode.text_fin7cb, .grouped = decode.text_fin7cbg, .piece = @embedFile("../../core/kernels/decode/fin7b_expert.metal"), .line = skip },
        .{ .banked = decode.text_dfin3b, .grouped = decode.text_dfin3bg, .piece = @embedFile("../../core/kernels/decode/dfin3b_expert.metal"), .line = skip_next },
    }) |c| {
        const i = std.mem.indexOf(u8, c.banked, c.piece).?;
        try testing.expectEqualStrings(c.banked[0..i], c.grouped[0..i]);
        try testing.expectEqualStrings(c.line, c.grouped[i .. i + c.line.len]);
        try testing.expectEqualStrings(c.banked[i..], c.grouped[i + c.line.len ..]);
    }
    const Gemv = struct { banked: []const u8, grouped: []const u8, w: []const u8, head: []const u8 };
    for ([_]Gemv{
        .{ .banked = decode.text_gu3b, .grouped = decode.text_gu3bg, .w = @embedFile("../../core/kernels/decode/gu3b_w.metal"), .head = @embedFile("../../core/kernels/decode/gu3g_2.metal") },
        .{ .banked = decode.text_down3b, .grouped = decode.text_down3bg, .w = @embedFile("../../core/kernels/decode/down3b_w.metal"), .head = @embedFile("../../core/kernels/decode/down3g_2.metal") },
    }) |c| {
        // the constants and grid line are the banked text's, then the member head, the weight base, the member tail
        const i = std.mem.indexOf(u8, c.grouped, c.head).?;
        try testing.expectEqualStrings(c.banked[0..i], c.grouped[0..i]);
        try testing.expectEqual(i + c.head.len, std.mem.indexOf(u8, c.grouped, c.w).?);
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, c.grouped, c.w));
    }
}

/// A Kernels3 at DeepSeek-V4.1's routed geometry (top-1, clamped, MUL1) as `root` builds it for a banked call, its
/// banked GEMVs at `group` row-slots per lead.
fn bankedKernels3(rate: format.Rate, group: u32) !@import("../../core/decode.zig").Kernels3 {
    var k = try @import("../../core/decode.zig").Kernels3.initWith(.{ .hidden = hidden, .inter = inter, .topk = 1, .rate = rate, .dec = .mul1, .limit = @intCast(limit) }, 1, 4, 2);
    k.out_dtype = .float32;
    k.banked_group = group;
    return k;
}

/// The banked launches' banks of three root banks.
fn kernelBanks(rb: *const [3]root.Bank) [3]@import("../../core/decode.zig").Kernels.Bank {
    var kb: [3]@import("../../core/decode.zig").Kernels.Bank = undefined;
    for (rb, &kb) |b, *o| o.* = .{ .gate_t = b.gate.trellis, .gate_suh = b.gate.suh, .gate_svh = b.gate.svh, .up_t = b.up.trellis, .up_suh = b.up.suh, .up_svh = b.up.svh, .down_t = b.down.trellis, .down_suh = b.down.suh, .down_svh = b.down.svh };
    return kb;
}

/// One banked call (gate|up, then down) on `k`: y f32 [rows, hidden]; the caller frees it.
fn groupCall(k: *@import("../../core/decode.zig").Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, ids: mlx.mlx_array, kb: *const [3]@import("../../core/decode.zig").Kernels.Bank, rows: usize) !mlx.mlx_array {
    const h = try k.bankedGateUp(s, x, ids, kb, rows);
    defer _ = mlx.mlx_array_free(h[0]);
    defer _ = mlx.mlx_array_free(h[1]);
    return k.bankedDown(s, h[0], h[1], ids, kb, rows);
}

// TURBOEXL3_GPU=1: the grouped banked call (banked_group 4) against the banked call (2) over the same three banks (20, 12
// and 8 experts), K3 and K2 (TURBOEXL3_BANKED_N), at rows 6, 12, 36 and 48 (TURBOEXL3_BANKED_ROWS), x bf16, trellis I16.
// The pairs in tokens of six: one expert in every token (up to eight row-slots of one id: two leads at four), one of three
// in each, the rest drawn; two skipped row-slots (`banked_skip_id` in the grouped call, base row 0 in the banked call, as
// DEVROUTE sends a miss today). Every other row-slot's y row byte for byte, a skipped one's 0.
test "turboexl3 gpu banked group: the grouped banked call's rows are the banked call's byte for byte, a skipped row-slot's 0, at DeepSeek-V4.1's routed geometry (K3, K2) (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_BANKED_N", &.{ 48, 32 }, &nbuf);
    var rbuf: [16]u32 = undefined;
    const row_counts = envList("TURBOEXL3_BANKED_ROWS", &.{ 6, 12, 36, 48 }, &rbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x96A0C4 ^ @as(u64, n));
        const rnd = prng.random();
        var rb: [3]root.Bank = undefined;
        for (&rb, bank_experts) |*b, e| {
            var hb = try ref.synthBank(a, rnd, e, hidden, inter, rate);
            defer hb.deinit(a);
            b.* = try rootBank(&g, &hb, .int16);
            try g.evalAll(&.{ b.gate.trellis, b.gate.suh, b.gate.svh, b.up.trellis, b.up.suh, b.up.svh, b.down.trellis, b.down.suh, b.down.svh });
        }
        const kb = kernelBanks(&rb);
        var k2 = try bankedKernels3(rate, 2);
        defer k2.deinit();
        var k4 = try bankedKernels3(rate, 4);
        defer k4.deinit();
        for (row_counts) |rows| {
            if (rows < 6 or rows % 6 != 0 or rows > root.banked_rows_max) continue;
            const m2 = g.mark();
            defer g.resetTo(m2);
            var ids2: [root.banked_rows_max]u32 = undefined;
            var ids4: [root.banked_rows_max]u32 = undefined;
            for (0..rows) |p| {
                // every token's expert (base row 1), one of three (ext rows 0..2), the rest drawn
                var bk: u32 = 0;
                var sl: u32 = 1;
                if (p % 6 == 1) {
                    bk = 1;
                    sl = @intCast((p / 6) % 3);
                } else if (p % 6 != 0) {
                    bk = rnd.uintLessThan(u32, 3);
                    sl = rnd.uintLessThan(u32, bank_experts[bk]);
                }
                ids2[p] = (bk << 24) | sl;
                ids4[p] = ids2[p];
            }
            const skipped = [2]usize{ rows - 1, rows - 4 };
            for (skipped) |p| {
                ids2[p] = 0;
                ids4[p] = root.banked_skip_id;
            }
            const xh = try a.alloc(f32, rows * hidden);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(hidden) }, .float32), .bfloat16);
            const a2 = try g.hostArray(std.mem.sliceAsBytes(ids2[0..rows]), &.{ci(rows)}, .uint32);
            const a4 = try g.hostArray(std.mem.sliceAsBytes(ids4[0..rows]), &.{ci(rows)}, .uint32);
            const y2 = try g.adopt(try groupCall(&k2, s, x, a2, &kb, rows));
            const y4 = try g.adopt(try groupCall(&k4, s, x, a4, &kb, rows));
            try g.evalAll(&.{ y2, y4 });
            const b2 = try a.alloc(f32, rows * hidden);
            defer a.free(b2);
            const b4 = try a.alloc(f32, rows * hidden);
            defer a.free(b4);
            const v2 = try g.hostF32(y2, b2);
            const v4 = try g.hostF32(y4, b4);
            var differing: usize = 0;
            var nonfinite: usize = 0;
            var skipped_nonzero: usize = 0;
            for (0..rows) |r| {
                const row4 = v4[r * hidden .. (r + 1) * hidden];
                if (r == skipped[0] or r == skipped[1]) {
                    for (row4) |v| skipped_nonzero += @intFromBool(@as(u32, @bitCast(v)) != 0);
                    continue;
                }
                const d = bitDiff(row4, v2[r * hidden .. (r + 1) * hidden]);
                differing += d.differing;
                nonfinite += d.nonfinite;
            }
            const ok = differing == 0 and nonfinite == 0 and skipped_nonzero == 0;
            if (!ok) failed = true;
            std.debug.print("TURBOEXL3_BANKED_GROUP_BYTES n={d} rows={d} differing={d} nonfinite={d} skipped_nonzero={d} {s}\n", .{ n, rows, differing, nonfinite, skipped_nonzero, if (ok) "OK" else "FAIL" });
        }
    }
    try testing.expect(!failed);
}

/// A timed call's hit multiplicities: `count[c]` experts named by c row-slots each (34 hit row-slots), as a 36-pair
/// verify call of DeepSeek-V4.1's names them (recorded routes: p10, about the median and p90 of the distinct experts).
const Profile = struct { name: []const u8, count: [7]u32 };
const profiles = [_]Profile{
    .{ .name = "d18", .count = .{ 0, 10, 4, 2, 1, 0, 1 } },
    .{ .name = "d22", .count = .{ 0, 14, 5, 2, 1, 0, 0 } },
    .{ .name = "d29", .count = .{ 0, 25, 3, 1, 0, 0, 0 } },
};
const GroupArm = enum { g2, g4, g4m };

// TURBOEXL3_BANKED_TIME=1: 36 row-slots (34 hits by a profile, 2 misses) on banks of 48, 16 and 8 experts, K3
// (TURBOEXL3_BANKED_N), x bf16, trellis I16; the arms: g2 the banked call with the misses on base row 0 (today's device
// wave), g4 the grouped call with the misses skipped, g4m the grouped call with the misses on base row 0. 3 warm then 21
// timed rounds, the arms interleaved with a rotated start; each sample one call built and evaluated.
test "turboexl3 gpu banked group time: the grouped banked call against the banked call at a verify's 36 pairs (TURBOEXL3_GPU=1 TURBOEXL3_BANKED_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_BANKED_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_BANKED_N", &.{48}, &nbuf);
    const rows: usize = 36;
    const warm: usize = 3;
    const timed: usize = 21;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x7A1E5 ^ @as(u64, n));
        const rnd = prng.random();
        var rb: [3]root.Bank = undefined;
        for (&rb, time_experts) |*b, e| {
            var hb = try ref.synthBank(a, rnd, e, hidden, inter, rate);
            defer hb.deinit(a);
            b.* = try rootBank(&g, &hb, .int16);
            try g.evalAll(&.{ b.gate.trellis, b.gate.suh, b.gate.svh, b.up.trellis, b.up.suh, b.up.svh, b.down.trellis, b.down.suh, b.down.svh });
        }
        const kb = kernelBanks(&rb);
        var k2 = try bankedKernels3(rate, 2);
        defer k2.deinit();
        var k4 = try bankedKernels3(rate, 4);
        defer k4.deinit();
        for (profiles) |pf| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            // the hit experts: distinct global indexes over base rows 1..47 and the ext bank, never base row 0
            var order: [63]u32 = undefined;
            for (&order, 0..) |*o, i| o.* = @intCast(i + 1);
            rnd.shuffle(u32, &order);
            var hit: [rows]u32 = undefined;
            var nh: usize = 0;
            var d: usize = 0;
            for (pf.count, 0..) |cnt, c| for (0..cnt) |_| {
                const e = order[d];
                d += 1;
                const id: u32 = if (e < 48) e else (1 << 24) | (e - 48);
                for (0..c) |_| {
                    hit[nh] = id;
                    nh += 1;
                }
            };
            std.debug.assert(nh == rows - 2);
            rnd.shuffle(u32, hit[0..nh]);
            var ids_row0: [rows]u32 = undefined;
            var ids_skip: [rows]u32 = undefined;
            const miss = [2]usize{ 7, 29 };
            var hi: usize = 0;
            for (0..rows) |p| {
                if (p == miss[0] or p == miss[1]) {
                    ids_row0[p] = 0;
                    ids_skip[p] = root.banked_skip_id;
                } else {
                    ids_row0[p] = hit[hi];
                    ids_skip[p] = hit[hi];
                    hi += 1;
                }
            }
            const xh = try a.alloc(f32, rows * hidden);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(hidden) }, .float32), .bfloat16);
            const a_row0 = try g.hostArray(std.mem.sliceAsBytes(&ids_row0), &.{ci(rows)}, .uint32);
            const a_skip = try g.hostArray(std.mem.sliceAsBytes(&ids_skip), &.{ci(rows)}, .uint32);
            try g.evalAll(&.{ x, a_row0, a_skip });
            const arms = [_]GroupArm{ .g2, .g4, .g4m };
            var samples: [arms.len][timed]u64 = undefined;
            for (0..warm + timed) |round| {
                for (0..arms.len) |k| {
                    const ai = (k + round) % arms.len;
                    const t0 = nowNs();
                    const y = switch (arms[ai]) {
                        .g2 => try groupCall(&k2, s, x, a_row0, &kb, rows),
                        .g4 => try groupCall(&k4, s, x, a_skip, &kb, rows),
                        .g4m => try groupCall(&k4, s, x, a_row0, &kb, rows),
                    };
                    defer _ = mlx.mlx_array_free(y);
                    try mlx.check(mlx.mlx_array_eval(y));
                    const dt = nowNs() - t0;
                    if (round >= warm) samples[ai][round - warm] = dt;
                }
            }
            for (arms, 0..) |arm, ai| {
                std.mem.sort(u64, &samples[ai], {}, std.sort.asc(u64));
                const min_us = @as(f64, @floatFromInt(samples[ai][0])) / 1000.0;
                const med_us = @as(f64, @floatFromInt(samples[ai][timed / 2])) / 1000.0;
                std.debug.print("TURBOEXL3_BANKED_GROUP_TIME n={d} profile={s} distinct={d} arm={s} min_us={d:.1} median_us={d:.1}\n", .{ n, pf.name, d, @tagName(arm), min_us, med_us });
            }
        }
    }
}
