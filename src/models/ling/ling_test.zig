//! Ling-3.1-flash (`models/ling/api.zig`). CPU (always): each Ling text is its base text with the SwiGLU line swapped for
//! Ling's clamp piece (or, banked, DeepSeek's clamp piece swapped for fin7's own line or Ling's) and nothing else; the
//! reference clamp's rule; the refusals; every trunk shape a width the dense kernels admit. GPU (TURBOEXL3_GPU=1, else
//! skipped), at Ling's shapes (hidden 4096, expert width 1280: ten 128-blocks, 512 experts, top-8, MUL1 at n 40 = K 2.5
//! and n 32 = K 2, the 2.60 and 2.10 packs' routed rates):
//! - `root.moeLing` at a limit no value reaches gives `root.moe`'s bytes (decode and prefill rows), and at limits 0 and 10
//!   equals `ref.moeRowClampedAt` (`silu_out`) on sampled rows of decode rows 1, 4, 16 and prefill rows 32, 1328;
//! - every pair of `root.moeLingBanked` over three banks is `moeLing`'s on its bank at top-1 with score 1, byte for byte,
//!   at limits 0 and 10, banked groups 2 and 4, rows 1, 2, 8, 16, 48;
//! - `root.linear` at every Ling trunk shape (K6; the dense FFN K4) equals the host reference.
//! TURBOEXL3_LING_N (rates in halfwords per tile; default 40,32) narrows the routed tests. Lines: TURBOEXL3_LING_BYTES,
//! TURBOEXL3_LING_PARITY, TURBOEXL3_LING_BANKED, TURBOEXL3_LING_DENSE.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const prefill = @import("../../core/prefill.zig");
const dense = @import("../../core/dense.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const config = @import("../../core/config.zig");
const banked = @import("../deepseek_v41/banked.zig");

const testing = std.testing;

const hidden: u32 = 4096;
const inter: u32 = 1280;
const experts: u32 = 512;
const top_k: u32 = 8;
const limit: c_int = 10;
const sigma: f32 = 8;

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

/// `piece` stands once where `base` holds `line` once, and the texts are equal around it.
fn expectSwapped(base: []const u8, text: []const u8, line: []const u8, piece: []const u8) !void {
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, base, line));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, piece));
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, text, line));
    const i = std.mem.indexOf(u8, base, line).?;
    try testing.expectEqual(i, std.mem.indexOf(u8, text, piece).?);
    try testing.expectEqualStrings(base[0..i], text[0..i]);
    try testing.expectEqualStrings(base[i + line.len ..], text[i + piece.len ..]);
}

test "turboexl3 ling texts: each Ling kernel is its base text with the SwiGLU line swapped for Ling's clamp (or fin7's own line) and nothing else; only a clamp reads LIMIT" {
    const fin7_line = @embedFile("../../core/kernels/decode/fin7_swiglu.metal");
    const fin7_ds = @embedFile("../../core/kernels/decode/fin7_swiglu_clamp.metal");
    const fin7_ling = @embedFile("../../core/kernels/decode/fin7_swiglu_clamp_out.metal");
    const mid_line = @embedFile("../../core/kernels/prefill/mid_swiglu.metal");
    const mid_ling = @embedFile("../../core/kernels/prefill/mid_swiglu_clamp_out.metal");
    try expectSwapped(decode.text_fin7, decode.text_fin7co, fin7_line, fin7_ling);
    try expectSwapped(prefill.text_mid, prefill.text_midco, mid_line, mid_ling);
    try expectSwapped(prefill.text_mid_v, prefill.text_mid_vco, mid_line, mid_ling);
    try expectSwapped(prefill.text_mid_t128, prefill.text_mid_t128co, mid_line, mid_ling);
    // the banked fin7s: DeepSeek's clamp piece swapped for fin7's own line (unclamped) or for Ling's clamp
    try expectSwapped(banked.text_fin7cb, banked.text_fin7b, fin7_ds, fin7_line);
    try expectSwapped(banked.text_fin7cb, banked.text_fin7cob, fin7_ds, fin7_ling);
    try expectSwapped(banked.text_fin7cbg, banked.text_fin7bg, fin7_ds, fin7_line);
    try expectSwapped(banked.text_fin7cbg, banked.text_fin7cobg, fin7_ds, fin7_ling);
    for ([_][]const u8{ decode.text_fin7, prefill.text_mid, banked.text_fin7b, banked.text_fin7bg }) |t| try testing.expect(std.mem.indexOf(u8, t, "LIMIT") == null);
    for ([_][]const u8{ fin7_ling, mid_ling }) |piece| try testing.expect(std.mem.indexOf(u8, piece, "LIMIT") != null);
}

test "turboexl3 ling reference: the gate's SiLU clamps from above only and up both ways; a NaN passes; limit 0 is the plain SwiGLU" {
    const silu = struct {
        fn f(v: f32) f32 {
            return v / (1.0 + @exp(-v));
        }
    }.f;
    const tol: f32 = 1e-6;
    try testing.expectApproxEqRel(@as(f32, 10) * 10, ref.swigluClampedOut(20, 20, 10), tol); // silu(20) > 10: clamped to 10
    try testing.expectApproxEqRel(silu(10) * -10, ref.swigluClampedOut(10, -12, 10), tol); // silu(10) < 10: kept
    try testing.expectApproxEqRel(silu(-20) * 3, ref.swigluClampedOut(-20, 3, 10), tol);
    try testing.expectApproxEqRel(silu(20) * 20, ref.swigluClampedOut(20, 20, 0), tol);
    try testing.expectEqual(ref.swigluClamped(3.5, -2.0, 0), ref.swigluClampedOut(3.5, -2.0, 0));
    // where the two clamps part: a gate past the limit is silu(L) * u at the gate, L * u after the SiLU
    try testing.expect(ref.swigluClamped(30, 1, 10) < ref.swigluClampedOut(30, 1, 10));
    try testing.expect(std.math.isNan(ref.swigluClampedOut(std.math.nan(f32), 1, 10)));
    try testing.expect(std.math.isNan(ref.swigluClampedOut(1, std.math.nan(f32), 10)));
}

test "turboexl3 ling refusals: moeLing and the banked entries take a limit in 0..128, before any operand is read" {
    const none: mlx.mlx_array = .{ .ctx = null };
    const p: root.Proj = .{ .trellis = none, .suh = none, .svh = none };
    const b: root.Bank = .{ .gate = p, .up = p, .down = p };
    const banks = [3]root.Bank{ b, b, b };
    const s0: mlx.mlx_stream = .{ .ctx = null };
    for ([_]c_int{ -1, 129 }) |l| {
        try testing.expectError(error.InvalidExl3Clamp, root.moeLing(s0, none, b, none, none, .mul1, l));
        try testing.expectError(error.InvalidExl3Clamp, root.moeLingBanked(s0, none, &banks, none, .mul1, l));
    }
}

/// Every distinct Ling-3.1-flash trunk shape (`n` the pack's rate: K6, the dense FFN K4); a name lists the linears that
/// share it. kv_a is stored at 640 outputs (576 used, exllamav3 pads a width to 128).
const DenseCase = struct { name: []const u8, in: u32, out: u32, n: u32, head: bool = false };
const dense_cases = [_]DenseCase{
    .{ .name = "kda.qkv", .in = 4096, .out = 24576, .n = 96 },
    .{ .name = "kda.f/g", .in = 4096, .out = 8192, .n = 96 },
    .{ .name = "kda.o/mla.dense", .in = 8192, .out = 4096, .n = 96 },
    .{ .name = "mla.q_a", .in = 4096, .out = 1536, .n = 96 },
    .{ .name = "mla.q_b", .in = 1536, .out = 12288, .n = 96 },
    .{ .name = "mla.kv_a", .in = 4096, .out = 640, .n = 96 },
    .{ .name = "shared.gate/up", .in = 4096, .out = 1280, .n = 96 },
    .{ .name = "shared.down", .in = 1280, .out = 4096, .n = 96 },
    .{ .name = "dense.gate/up", .in = 4096, .out = 10240, .n = 64 },
    .{ .name = "dense.down", .in = 10240, .out = 4096, .n = 64 },
    .{ .name = "head", .in = 4096, .out = 157184, .n = 96, .head = true },
};

test "turboexl3 ling: every Ling-3.1-flash trunk shape is a width the dense kernels admit; the routed geometry is a decode and prefill geometry" {
    for (dense_cases) |c| {
        try testing.expect(c.in % 128 == 0 and c.out % 128 == 0 and (c.in / 16) % 4 == 0);
        const k = dense.Dense.ksFor(c.in, c.out);
        try testing.expect(k >= 1 and ((c.in / 16) / 4) % k == 0);
    }
    for ([_]u32{ 40, 32 }) |n| {
        const rate = format.Rate.fromN(n).?;
        for ([_]u32{ 1, top_k }) |k| {
            try testing.expect((decode.Geometry{ .hidden = hidden, .inter = inter, .topk = k, .rate = rate, .dec = .mul1 }).admitted());
            try testing.expect((prefill.Geometry{ .hidden = hidden, .inter = inter, .topk = k, .experts = experts, .rate = rate, .dec = .mul1 }).admitted());
        }
    }
}

/// The bank as `root` takes it: each projection's arrays from the host bank, the trellis in `td`.
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

/// One call's routing: each row's top-k distinct experts by Gumbel top-k over lognormal popularity (clamped_test's draw),
/// scores positive and bf16-exact, summing to one per row before rounding; row 0 takes the bank's last expert.
fn route(a: std.mem.Allocator, rnd: std.Random, rows: u32, n_experts: u32, topk: u32, skew: f64, slots: []u32, scores: []f32) !void {
    const logw = try a.alloc(f64, n_experts);
    defer a.free(logw);
    for (logw) |*w| w.* = skew * rnd.floatNorm(f64);
    const keys = try a.alloc(f64, n_experts);
    defer a.free(keys);
    for (0..rows) |r| {
        for (keys, logw) |*kv, lw| kv.* = lw - @log(-@log(@max(rnd.float(f64), 1e-300)));
        for (slots[r * topk ..][0..topk]) |*sl| {
            var best: usize = 0;
            for (keys, 0..) |kv, e| {
                if (kv > keys[best]) best = e;
            }
            sl.* = @intCast(best);
            keys[best] = -std.math.inf(f64);
        }
        const ws = scores[r * topk ..][0..topk];
        var sum: f32 = 0;
        for (ws) |*w| {
            w.* = 0.05 + rnd.float(f32);
            sum += w.*;
        }
        for (ws) |*w| w.* = bfExact(w.* / sum);
    }
    if (std.mem.indexOfScalar(u32, slots[0..topk], n_experts - 1) == null) slots[topk - 1] = n_experts - 1;
}

/// One call's host values and its operands as a host passes them: x [1, rows, hidden] bf16, the slots i32 and the scores
/// bf16 [1, rows, topk].
const Call = struct {
    x: []f32,
    slots: []u32,
    scores: []f32,
    x3: mlx.mlx_array,
    inds3: mlx.mlx_array,
    sc3: mlx.mlx_array,

    fn make(a: std.mem.Allocator, g: *G, rnd: std.Random, rows: u32, sd: f32) !Call {
        const x = try a.alloc(f32, rows * hidden);
        errdefer a.free(x);
        for (x) |*v| v.* = bfExact(sd * rnd.floatNorm(f32));
        const slots = try a.alloc(u32, rows * top_k);
        errdefer a.free(slots);
        const scores = try a.alloc(f32, rows * top_k);
        errdefer a.free(scores);
        try route(a, rnd, rows, experts, top_k, 1.5, slots, scores);
        const inds = try a.alloc(i32, rows * top_k);
        defer a.free(inds);
        for (slots, inds) |sl, *d| d.* = @intCast(sl);
        return .{
            .x = x,
            .slots = slots,
            .scores = scores,
            .x3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(hidden) }, .float32), .bfloat16),
            .inds3 = try g.hostArray(std.mem.sliceAsBytes(inds), &.{ 1, ci(rows), ci(top_k) }, .int32),
            .sc3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, ci(rows), ci(top_k) }, .float32), .bfloat16),
        };
    }

    fn deinit(c: *Call, a: std.mem.Allocator) void {
        a.free(c.x);
        a.free(c.slots);
        a.free(c.scores);
    }
};

/// The elements whose bits differ between two arrays of one size (both read as f32).
fn bitDiffArr(a: std.mem.Allocator, g: *G, x: mlx.mlx_array, y: mlx.mlx_array) !usize {
    const n = mlx.mlx_array_size(x);
    if (mlx.mlx_array_size(y) != n) return error.SizeMismatch;
    const p = try a.alloc(f32, n);
    defer a.free(p);
    const q = try a.alloc(f32, n);
    defer a.free(q);
    _ = try g.hostF32(try g.astype(x, .float32), p);
    _ = try g.hostF32(try g.astype(y, .float32), q);
    var d: usize = 0;
    for (p, q) |u, v| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
    return d;
}

/// The rows a call's reference covers: the first and the last, and the middle one past 32 rows.
fn sampleRows(rows: u32, buf: *[3]u32) []const u32 {
    buf.* = .{ 0, rows - 1, rows / 2 };
    if (rows == 1) return buf[0..1];
    return if (rows <= 32) buf[0..2] else buf;
}

// Ling's routed geometry on synthetic MUL1 banks of 512 experts at each rate. First, x ~ N(0, 1): every gate's SiLU and up
// stays far below 128, so moeLing at 128 must give moe's bytes (rows 1, 16: decode form 9; 400: prefill). Then x ~ N(0, 8^2)
// (gate and up ~ N(0, 8.6^2): about 12% of the gates' SiLUs pass 10 and a quarter of the ups pass +-10; the test requires 1%
// of each on the sampled rows at limit 10): moeLing at limits 0 and 10, bf16 and f32 x, against ref.moeRowClampedAt at
// `silu_out` on sampled rows, clamped_test's bar (5e-3 of the reference's rms; bf16 beyond its store's rounding).
test "turboexl3 gpu ling: root.moeLing at Ling-3.1-flash's routed geometry (4096 x 1280, 512 experts, top-8, n 40 and 32) gives moe's bytes where no value clamps and equals the reference at limits 0 and 10, decode and prefill rows (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_LING_N", &.{ 40, 32 }, &nbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x11A6 ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
        defer bank.deinit(a);
        const rb = try rootBank(&g, &bank, .uint16);
        for ([_]u32{ 1, 16, 400 }) |rows| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            var c = try Call.make(a, &g, rnd, rows, 1);
            defer c.deinit(a);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const x = try g.astype(c.x3, dt);
                const yu = try g.adopt(try root.moe(s, x, rb, c.inds3, c.sc3, .mul1, false));
                const yc = try g.adopt(try root.moeLing(s, x, rb, c.inds3, c.sc3, .mul1, 128));
                const d = try bitDiffArr(a, &g, yu, yc);
                std.debug.print("TURBOEXL3_LING_BYTES n={d} rows={d} out={s} differing={d}/{d} {s}\n", .{ n, rows, @tagName(dt), d, rows * hidden, if (d == 0) "PASS" else "FAIL" });
                if (d != 0) failed = true;
            }
        }
        for ([_]u32{ 1, 4, 16, 32, 1328 }) |rows| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            var c = try Call.make(a, &g, rnd, rows, sigma);
            defer c.deinit(a);
            var sbuf: [3]u32 = undefined;
            const sample = sampleRows(rows, &sbuf);
            const want = try a.alloc(f32, sample.len * hidden);
            defer a.free(want);
            const got = try a.alloc(f32, rows * hidden);
            defer a.free(got);
            for ([_]c_int{ 0, limit }) |lim| {
                var hits: ref.Hits = .{};
                for (sample, 0..) |r, i| {
                    const h = try ref.moeRowClampedAt(a, &bank, .mul1, @floatFromInt(lim), .silu_out, c.x[r * hidden ..][0..hidden], c.slots[r * top_k ..][0..top_k], c.scores[r * top_k ..][0..top_k], want[i * hidden ..][0..hidden]);
                    hits.gate += h.gate;
                    hits.up += h.up;
                    hits.total += h.total;
                }
                const hit = lim == 0 or (hits.gate * 100 >= hits.total and hits.up * 100 >= hits.total);
                for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                    const y = try g.adopt(try root.moeLing(s, try g.astype(c.x3, dt), rb, c.inds3, c.sc3, .mul1, lim));
                    try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(hidden) }, mlx.getShape(y));
                    try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
                    _ = try g.hostF32(try g.astype(y, .float32), got);
                    var bad: usize = 0;
                    for (got) |v| bad += @intFromBool(!std.math.isFinite(v));
                    var worst: f64 = 0;
                    for (sample, 0..) |r, i| {
                        const gr = got[r * hidden ..][0..hidden];
                        const wr = want[i * hidden ..][0..hidden];
                        worst = @max(worst, (if (dt == .float32) ref.compare(gr, wr) else ref.compareRounded(gr, wr, 0x1p-8)).max_ratio);
                    }
                    const bar: f64 = 5e-3;
                    const ok = hit and bad == 0 and worst <= bar;
                    const path: []const u8 = if (rows <= root.kernels.DECODE_ROWS_MAX) "decode" else "prefill";
                    std.debug.print("TURBOEXL3_LING_PARITY n={d} rows={d} limit={d} path={s} out={s} worst_vs_ref={e:.3} bar={e:.1} nonfinite={d} gates_clamped={d}/{d} ups_clamped={d}/{d} {s}\n", .{ n, rows, lim, path, @tagName(dt), worst, bar, bad, hits.gate, hits.total, hits.up, hits.total, if (ok) "PASS" else "FAIL" });
                    if (!ok) failed = true;
                }
            }
        }
    }
    try testing.expect(!failed);
}

/// The banked test's banks: base, ext, transient
const bank_experts = [3]u32{ 20, 12, 8 };

/// A call's pairs as banked_test draws them: from 4 rows row-slot 1 repeats row-slot 0's id and row-slot 2 takes its slot
/// row in the next bank; from 8 rows the last three take base row 0.
fn drawPairs(rnd: std.Random, bank: []u32, slot: []u32) void {
    for (bank, slot) |*b, *sl| {
        b.* = rnd.uintLessThan(u32, 3);
        sl.* = rnd.uintLessThan(u32, bank_experts[b.*]);
    }
    const rows = bank.len;
    if (rows >= 4) {
        slot[0] = rnd.uintLessThan(u32, bank_experts[2]);
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

// Ling's routed geometry over three banks of 20, 12 and 8 experts at each rate: every pair's row of moeLingBanked against
// moeLing on its bank at top-1 with score 1 (at most 16 rows a call, so its decode launches), byte for byte, x bf16 and the
// trellis I16 (the streamer's code arrays) and U16, at limits 0 (fin7b / fin7bg) and 10 (fin7cob / fin7cobg), under
// `banked_group` 2 and 4. x ~ N(0, 8^2), so the clamp acts at 10.
test "turboexl3 gpu ling banked: every pair of moeLingBanked over three banks is moeLing's on its bank at top-1 with score 1, byte for byte, at limits 0 and 10 and banked groups 2 and 4 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = config.get().*;
    defer config.configure(saved) catch {};
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_LING_N", &.{ 40, 32 }, &nbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x11A6BA ^ @as(u64, n));
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
        for ([_]u32{ 1, 2, 8, 16, 48 }) |rows| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            const x = try a.alloc(f32, rows * hidden);
            defer a.free(x);
            for (x) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
            var bank_of: [root.banked_rows_max]u32 = undefined;
            var slot_of: [root.banked_rows_max]u32 = undefined;
            drawPairs(rnd, bank_of[0..rows], slot_of[0..rows]);
            var ids: [root.banked_rows_max]u32 = undefined;
            for (ids[0..rows], bank_of[0..rows], slot_of[0..rows]) |*d, b, sl| d.* = (b << 24) | sl;
            const x32 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ ci(rows), ci(hidden) }, .float32);
            const xbf = try g.astype(x32, .bfloat16);
            const ids_a = try g.hostArray(std.mem.sliceAsBytes(ids[0..rows]), &.{ci(rows)}, .uint32);
            const want = try a.alloc(f32, rows * hidden);
            defer a.free(want);
            const got = try a.alloc(f32, rows * hidden);
            defer a.free(got);
            const buf = try a.alloc(f32, 16 * hidden);
            defer a.free(buf);
            const ones: [16]f32 = @splat(1.0);
            for ([_]c_int{ 0, limit }) |lim| {
                // the reference bytes: each bank's pairs through moeLing at top-1 with score 1, 16 rows a call
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
                        const xr = try g.take(xbf, try g.hostArray(std.mem.sliceAsBytes(P), &.{ci(c)}, .uint32), 0);
                        const x3 = try g.reshape(xr, &.{ 1, ci(c), ci(hidden) });
                        const inds = try g.hostArray(std.mem.sliceAsBytes(sl[0..c]), &.{ 1, ci(c), 1 }, .int32);
                        const sc = try g.hostArray(std.mem.sliceAsBytes(ones[0..c]), &.{ 1, ci(c), 1 }, .float32);
                        const y = try g.adopt(try root.moeLing(s, try g.astype(x3, .float32), rb_u[b], inds, sc, .mul1, lim));
                        _ = try g.hostF32(y, buf[0 .. c * hidden]);
                        for (P, 0..) |p, i| @memcpy(want[p * hidden ..][0..hidden], buf[i * hidden ..][0..hidden]);
                    }
                }
                for ([_]u32{ 2, 4 }) |group| {
                    var cfg = saved;
                    cfg.banked_group = group;
                    try config.configure(cfg);
                    const Arm = struct { banks: *const [3]root.Bank, name: []const u8 };
                    for ([_]Arm{ .{ .banks = &rb_i, .name = "trellis=i16" }, .{ .banks = &rb_u, .name = "trellis=u16" } }) |arm| {
                        const y = try g.adopt(try root.moeLingBanked(s, x32, arm.banks, ids_a, .mul1, lim));
                        try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(hidden) }, mlx.getShape(y));
                        _ = try g.hostF32(y, got);
                        var d: usize = 0;
                        var bad: usize = 0;
                        for (got, want) |u, v| {
                            d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
                            bad += @intFromBool(!std.math.isFinite(u));
                        }
                        const ok = d == 0 and bad == 0;
                        std.debug.print("TURBOEXL3_LING_BANKED n={d} rows={d} limit={d} group={d} {s} differing={d}/{d} nonfinite={d} {s}\n", .{ n, rows, lim, group, arm.name, d, rows * hidden, bad, if (ok) "PASS" else "FAIL" });
                        if (!ok) failed = true;
                    }
                }
                try config.configure(saved);
            }
        }
    }
    try testing.expect(!failed);
}

/// DEC-ONE (1, 4), the GEMV chain's two chunks (16), the register path (32), dequantize + matmul (512)
const dense_rows = [_]u32{ 1, 4, 16, 32, 512 };
const head_rows = [_]u32{ 1, 4 };

/// The rows a case compares: every row up to 16; past that both ends and either side of the middle
fn pickRows(rows: u32, buf: *[16]u32) []const u32 {
    if (rows <= 16) {
        for (buf[0..rows], 0..) |*r, i| r.* = @intCast(i);
        return buf[0..rows];
    }
    buf[0..6].* = .{ 0, 1, rows / 2 - 1, rows / 2, rows - 2, rows - 1 };
    return buf[0..6];
}

fn report(name: []const u8, rows: u32, xd: []const u8, width: usize, got: []const f32, pick: []const u32, want: []const f32, sel: []f32, bar: f64) usize {
    for (pick, 0..) |r, i| @memcpy(sel[i * width ..][0..width], got[r * width ..][0..width]);
    const e = ref.compare(sel[0 .. pick.len * width], want[0 .. pick.len * width]);
    const ok = std.math.isFinite(e.max_abs) and e.max_ratio <= bar;
    std.debug.print("TURBOEXL3_LING_DENSE {s} rows={d} x={s} compared_rows={d} max_abs={e:.3} rms={e:.3} ratio={e:.3} bar={e:.1} {s}\n", .{ name, rows, xd, pick.len, e.max_abs, e.rms_ref, e.max_ratio, bar, if (ok) "PASS" else "FAIL" });
    return @intFromBool(!ok);
}

fn denseCase(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, c: DenseCase) !usize {
    var prng = std.Random.DefaultPrng.init(0x11A6DE ^ (@as(u64, c.in) << 24) ^ c.out);
    const rnd = prng.random();
    const rate = format.Rate.fromN(c.n).?;
    var lin = try ref.synthLinear(a, rnd, 1, c.in, c.out, rate, .mul1);
    defer lin.deinit(a);
    const mark = g.mark();
    defer g.resetTo(mark);
    const trellis = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(c.in / 16), ci(c.out / 16), ci(c.n) }, .uint16);
    const suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(c.in)}, .float16);
    const svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(c.out)}, .float16);
    const transformed = try a.alloc(f32, c.in);
    defer a.free(transformed);
    const inner = try a.alloc(f32, c.out);
    defer a.free(inner);
    var failed: usize = 0;
    for (if (c.head) head_rows[0..] else dense_rows[0..]) |rows| {
        const rmark = g.mark();
        defer g.resetTo(rmark);
        const x = try a.alloc(f32, rows * c.in);
        defer a.free(x);
        for (x) |*v| v.* = bfExact(rnd.floatNorm(f32));
        var pick_buf: [16]u32 = undefined;
        const pick = pickRows(rows, &pick_buf);
        const want = try a.alloc(f32, pick.len * c.out);
        defer a.free(want);
        for (pick, 0..) |r, i| lin.row(0, x[r * c.in ..][0..c.in], transformed, inner, want[i * c.out ..][0..c.out]);
        const got = try a.alloc(f32, rows * c.out);
        defer a.free(got);
        const sel = try a.alloc(f32, pick.len * c.out);
        defer a.free(sel);
        const x3 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(c.in) }, .float32);
        const y = try g.adopt(try root.linear(s, x3, trellis, suh, svh, .mul1));
        try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(c.out) }, mlx.getShape(y));
        _ = try g.hostF32(y, got);
        failed += report(c.name, rows, "f32", c.out, got, pick, want, sel, 5e-3);
        if (rows == 4 or rows == 512) {
            const yb = try g.adopt(try root.linear(s, try g.astype(x3, .bfloat16), trellis, suh, svh, .mul1));
            try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(yb));
            _ = try g.hostF32(try g.astype(yb, .float32), got);
            failed += report(c.name, rows, "bf16", c.out, got, pick, want, sel, 2.5e-2);
        }
    }
    return failed;
}

test "turboexl3 gpu ling dense: root.linear equals the host reference at every Ling-3.1-flash trunk shape, K6 and the dense FFN's K4 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var failed: usize = 0;
    for (dense_cases) |c| failed += try denseCase(a, &g, s, c);
    try testing.expectEqual(@as(usize, 0), failed);
}
