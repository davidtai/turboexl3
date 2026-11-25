//! Inkling 2 (`models/inkling2/api.zig`). CPU (always): each softcap text is its base text with the SwiGLU line swapped
//! for the softcap piece and nothing else; the reference softcap's values; the refusals; Inkling 2's trunk and routed
//! shapes are widths the dense, decode and prefill kernels admit. GPU (TURBOEXL3_GPU=1, else skipped), at Inkling 2's
//! routed geometry (hidden 4608, expert width 2304: eighteen 128-blocks, 384 experts, top-8), MCG (the pack's codebook)
//! at n 16 = K1 and n 32 = K2 (TURBOEXL3_INKLING2_N narrows or widens), caps 4 and 25:
//! - `root.moeInkling2` equals `ref.moeRowSoftcap` on sampled rows of decode rows 1, 4, 16 and prefill rows 32, 1328, x
//!   bf16 and f32, with at least 1% of the sampled gates and ups past their caps; `root.moe` on the same calls equals
//!   `ref.moeRowDec` (the unmodified chain at K1 and K2 at this geometry);
//! - every pair of `root.moeInkling2Banked` over three banks is `moeInkling2`'s on its bank at top-1 with score 1, byte
//!   for byte, on the configured texts, the grouped texts (g4) and the M = 3 path (g4m3), each selected and checked to
//!   have run, trellis I16 and U16;
//! - `root.moeInkling2Rows` gives, row for row, `moeInkling2`'s bytes at top-1 over the gathered rows.
//! Lines: TURBOEXL3_INKLING2_PARITY, TURBOEXL3_INKLING2_PLAIN, TURBOEXL3_INKLING2_BANKED, TURBOEXL3_INKLING2_ROWS.
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

const hidden: u32 = 4608;
const inter: u32 = 2304;
const experts: u32 = 384;
const top_k: u32 = 8;
const caps: format.Softcap = .{ .gate = 4, .up = 25 };
/// x ~ N(0, 16^2): the synthetic gates and ups land near N(0, 17^2), so most gates and about 15% of the ups pass the caps
const sigma: f32 = 16;
const dec: format.Decode = .mcg;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn bfExact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
}

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

test "turboexl3 inkling2 texts: each softcap kernel is its base text with the SwiGLU line swapped for the softcap piece and nothing else; the piece reads LIMIT's two bytes" {
    const fin7_line = @embedFile("../../core/kernels/decode/fin7_swiglu.metal");
    const fin7_ds = @embedFile("../../core/kernels/decode/fin7_swiglu_clamp.metal");
    const fin7_cs = @embedFile("../../core/kernels/decode/fin7_swiglu_softcap.metal");
    const mid_line = @embedFile("../../core/kernels/prefill/mid_swiglu.metal");
    const mid_cs = @embedFile("../../core/kernels/prefill/mid_swiglu_softcap.metal");
    try expectSwapped(decode.text_fin7, decode.text_fin7cs, fin7_line, fin7_cs);
    try expectSwapped(prefill.text_midc, prefill.text_midcs, @embedFile("../../core/kernels/prefill/mid_swiglu_clamp.metal"), mid_cs);
    try expectSwapped(prefill.text_mid, prefill.text_midcs, mid_line, mid_cs);
    try expectSwapped(prefill.text_mid_v, prefill.text_mid_vcs, mid_line, mid_cs);
    try expectSwapped(prefill.text_mid_t128, prefill.text_mid_t128cs, mid_line, mid_cs);
    try expectSwapped(banked.text_fin7cb, banked.text_fin7csb, fin7_ds, fin7_cs);
    try expectSwapped(banked.text_fin7cbg, banked.text_fin7csbg, fin7_ds, fin7_cs);
    for ([_][]const u8{ fin7_cs, mid_cs }) |piece| {
        try testing.expect(std.mem.indexOf(u8, piece, "LIMIT & 255") != null);
        try testing.expect(std.mem.indexOf(u8, piece, "(LIMIT >> 8) & 255") != null);
        try testing.expect(std.mem.indexOf(u8, piece, "tanh") != null);
    }
    // the pieces assign the name the next statement reads: fin7's `hv`, the mid's `h`
    try testing.expect(std.mem.indexOf(u8, fin7_cs, "const float hv = ") != null);
    try testing.expect(std.mem.indexOf(u8, mid_cs, "const float h = ") != null);
}

test "turboexl3 inkling2 reference: the softcap is cg tanh(g / cg) sigmoid(g) * cu tanh(u / cu), the sigmoid on the raw gate; near zero it is the plain SwiGLU; a NaN passes" {
    const silu = struct {
        fn f(v: f32) f32 {
            return v / (1.0 + @exp(-v));
        }
    }.f;
    // small values: tanh(v / c) ~ v / c, so the plain SwiGLU within a few 1e-3
    try testing.expectApproxEqRel(silu(0.05) * 0.1, ref.swigluSoftcap(0.05, 0.1, caps), 1e-3);
    // a large gate saturates at cg * sigmoid(g) ~ 4, a large up at cu = 25
    try testing.expectApproxEqRel(@as(f32, 4) * 25, ref.swigluSoftcap(100, 1000, caps), 1e-4);
    try testing.expectApproxEqRel(@as(f32, -4) * 25 * (1.0 / (1.0 + @exp(@as(f32, 30)))), ref.swigluSoftcap(-30, 1000, caps), 1e-3);
    // the sigmoid reads the raw gate, not the capped one: at g = 8, sigmoid(8) and not sigmoid(4 tanh(2))
    const g: f32 = 8;
    const want: f64 = 4.0 * std.math.tanh(@as(f64, 2)) / (1.0 + @exp(-@as(f64, 8))) * 25.0 * std.math.tanh(@as(f64, 0.4));
    try testing.expectApproxEqRel(@as(f32, @floatCast(want)), ref.swigluSoftcap(g, 10, caps), 1e-6);
    // odd in u, and a negative gate past -cg is bounded by cg too
    try testing.expectEqual(-ref.swigluSoftcap(3, 7, caps), ref.swigluSoftcap(3, -7, caps));
    try testing.expect(@abs(ref.swigluSoftcap(-6, 30, caps)) < 4 * 25);
    try testing.expect(std.math.isNan(ref.swigluSoftcap(std.math.nan(f32), 1, caps)));
    try testing.expect(std.math.isNan(ref.swigluSoftcap(1, std.math.nan(f32), caps)));
}

test "turboexl3 inkling2 refusals: every entry takes whole caps in 1..255, before any operand is read" {
    const none: mlx.mlx_array = .{ .ctx = null };
    const p: root.Proj = .{ .trellis = none, .suh = none, .svh = none };
    const b: root.Bank = .{ .gate = p, .up = p, .down = p };
    const banks = [3]root.Bank{ b, b, b };
    const s0: mlx.mlx_stream = .{ .ctx = null };
    for ([_]format.Softcap{ .{ .gate = 0, .up = 25 }, .{ .gate = 4, .up = 0 }, .{ .gate = 4.5, .up = 25 }, .{ .gate = 4, .up = 256 }, .{ .gate = std.math.nan(f32), .up = 25 } }) |c| {
        try testing.expectError(error.InvalidExl3Clamp, root.moeInkling2(s0, none, b, none, none, dec, c));
        try testing.expectError(error.InvalidExl3Clamp, root.moeInkling2Rows(s0, none, none, b, none, none, dec, c, .float32));
        try testing.expectError(error.InvalidExl3Clamp, root.moeInkling2Banked(s0, none, &banks, none, dec, c));
    }
    // a softcap limit read as a clamp, and a clamp read as a softcap, are refused by the shared entries
    const l = format.softcapLimit(caps).?;
    try testing.expectError(error.InvalidExl3Clamp, root.moeSwiglu(s0, none, b, none, none, dec, l, .silu_out));
    try testing.expectError(error.InvalidExl3Clamp, root.moeSwiglu(s0, none, b, none, none, dec, 10, .softcap));
}

/// Every distinct Inkling 2 trunk shape (layers.L.attn.*, the head), at the 2.99bpw pack's trunk rate (K6) and the
/// 1.99bpw's and 1.02bpw's (K5, K3). unembed is stored at 201088 rows (vocab 201024 padded).
const DenseCase = struct { name: []const u8, in: u32, out: u32 };
const dense_cases = [_]DenseCase{
    .{ .name = "wq_du/gate_attn", .in = 4608, .out = 6144 },
    .{ .name = "wk_dv/wv_dv", .in = 4608, .out = 1024 },
    .{ .name = "wr_du", .in = 4608, .out = 768 },
    .{ .name = "wo_ud", .in = 6144, .out = 4608 },
    .{ .name = "unembed", .in = 4608, .out = 201088 },
};

test "turboexl3 inkling2: every Inkling 2 trunk shape is a width the dense kernels admit; the routed geometry is a decode and prefill geometry at K1 to K4 and the softcap" {
    for (dense_cases) |c| {
        try testing.expect(c.in % 128 == 0 and c.out % 128 == 0 and (c.in / 16) % 4 == 0);
        const k = dense.Dense.ksFor(c.in, c.out);
        try testing.expect(k >= 1 and ((c.in / 16) / 4) % k == 0);
    }
    const l = format.softcapLimit(caps).?;
    for ([_]u32{ 16, 32, 48, 64 }) |n| {
        const rate = format.Rate.fromN(n).?;
        for ([_]u32{ 1, top_k }) |k| {
            try testing.expect((decode.Geometry{ .hidden = hidden, .inter = inter, .topk = k, .rate = rate, .dec = dec, .limit = l, .clamp = .softcap }).admitted());
            try testing.expect((prefill.Geometry{ .hidden = hidden, .inter = inter, .topk = k, .experts = experts, .rate = rate, .dec = dec, .limit = l, .clamp = .softcap }).admitted());
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

/// One call's routing: each row's top-k distinct experts by Gumbel top-k over lognormal popularity, scores positive and
/// bf16-exact, summing to one per row before rounding; row 0 takes the bank's last expert.
fn route(a: std.mem.Allocator, rnd: std.Random, rows: u32, n_experts: u32, topk: u32, slots: []u32, scores: []f32) !void {
    const logw = try a.alloc(f64, n_experts);
    defer a.free(logw);
    for (logw) |*w| w.* = 1.5 * rnd.floatNorm(f64);
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

/// The rows a call's reference covers: the first and the last.
fn sampleRows(rows: u32, buf: *[2]u32) []const u32 {
    buf.* = .{ 0, rows - 1 };
    return if (rows == 1) buf[0..1] else buf;
}

fn bitDiff(p: []const f32, q: []const f32) usize {
    var d: usize = 0;
    for (p, q) |u, v| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
    return d;
}

// Inkling 2's routed geometry on synthetic MCG banks of 384 experts at each rate, x ~ N(0, 16^2) in bf16 values: each
// call through moeInkling2 (x bf16 and f32) against ref.moeRowSoftcap on its first and last rows at the decode chain's bar
// (5e-3 of the reference's rms; bf16 beyond its store's rounding), the softcap bending at least 1% of the sampled gates
// and of the ups; then root.moe on the same call against ref.moeRowDec, the unmodified chain at this geometry and rate.
test "turboexl3 gpu inkling2: root.moeInkling2 at Inkling 2's routed geometry (4608 x 2304, 384 experts, top-8, MCG at n 16 and 32) equals the softcap reference, and root.moe the plain one, at decode and prefill rows (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_INKLING2_N", &.{ 16, 32 }, &nbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x1A4C2 ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
        defer bank.deinit(a);
        const rb = try rootBank(&g, &bank, .uint16);
        for ([_]u32{ 1, 4, 16, 32, 1328 }) |rows| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            const x = try a.alloc(f32, rows * hidden);
            defer a.free(x);
            for (x) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
            // The plain control reads x / 16 (exact in bf16): at the softcap's input scale an uncapped SwiGLU overflows
            // the kernels' half-precision intermediates, which tests nothing of the plain path.
            const xs = try a.alloc(f32, rows * hidden);
            defer a.free(xs);
            for (xs, x) |*d, v| d.* = v * 0.0625;
            const slots = try a.alloc(u32, rows * top_k);
            defer a.free(slots);
            const scores = try a.alloc(f32, rows * top_k);
            defer a.free(scores);
            try route(a, rnd, rows, experts, top_k, slots, scores);
            const inds = try a.alloc(i32, rows * top_k);
            defer a.free(inds);
            for (slots, inds) |sl, *d| d.* = @intCast(sl);
            const x3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(hidden) }, .float32), .bfloat16);
            const xs3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs), &.{ 1, ci(rows), ci(hidden) }, .float32), .bfloat16);
            const inds3 = try g.hostArray(std.mem.sliceAsBytes(inds), &.{ 1, ci(rows), ci(top_k) }, .int32);
            const sc3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, ci(rows), ci(top_k) }, .float32), .bfloat16);
            var sbuf: [2]u32 = undefined;
            const sample = sampleRows(rows, &sbuf);
            const want = try a.alloc(f32, sample.len * hidden);
            defer a.free(want);
            const want_plain = try a.alloc(f32, sample.len * hidden);
            defer a.free(want_plain);
            const got = try a.alloc(f32, rows * hidden);
            defer a.free(got);
            var hits: ref.Hits = .{};
            for (sample, 0..) |r, i| {
                const xr = x[r * hidden ..][0..hidden];
                const sl = slots[r * top_k ..][0..top_k];
                const sc = scores[r * top_k ..][0..top_k];
                const h = try ref.moeRowSoftcap(a, &bank, dec, caps, xr, sl, sc, want[i * hidden ..][0..hidden]);
                hits.gate += h.gate;
                hits.up += h.up;
                hits.total += h.total;
                try ref.moeRowDec(a, &bank, dec, xs[r * hidden ..][0..hidden], sl, sc, want_plain[i * hidden ..][0..hidden]);
            }
            const hit = hits.gate * 100 >= hits.total and hits.up * 100 >= hits.total;
            const path: []const u8 = if (rows <= root.kernels.DECODE_ROWS_MAX) "decode" else "prefill";
            const bar: f64 = 5e-3;
            const Arm = enum { softcap, plain };
            for ([_]Arm{ .softcap, .plain }) |arm| for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const xa = try g.astype(if (arm == .softcap) x3 else xs3, dt);
                const y = try g.adopt(switch (arm) {
                    .softcap => try root.moeInkling2(s, xa, rb, inds3, sc3, dec, caps),
                    .plain => try root.moe(s, xa, rb, inds3, sc3, dec, false),
                });
                try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(hidden) }, mlx.getShape(y));
                try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
                _ = try g.hostF32(try g.astype(y, .float32), got);
                var bad: usize = 0;
                for (got) |v| bad += @intFromBool(!std.math.isFinite(v));
                const ref_rows = if (arm == .softcap) want else want_plain;
                var worst: f64 = 0;
                for (sample, 0..) |r, i| {
                    const gr = got[r * hidden ..][0..hidden];
                    const wr = ref_rows[i * hidden ..][0..hidden];
                    worst = @max(worst, (if (dt == .float32) ref.compare(gr, wr) else ref.compareRounded(gr, wr, 0x1p-8)).max_ratio);
                }
                const ok = (arm == .plain or hit) and bad == 0 and worst <= bar;
                const line: []const u8 = if (arm == .softcap) "PARITY" else "PLAIN";
                std.debug.print("TURBOEXL3_INKLING2_{s} n={d} cb=mcg rows={d} path={s} out={s} worst_vs_ref={e:.3} bar={e:.1} nonfinite={d} gates_capped={d}/{d} ups_capped={d}/{d} {s}\n", .{ line, n, rows, path, @tagName(dt), worst, bar, bad, hits.gate, hits.total, hits.up, hits.total, if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            };
        }
    }
    try testing.expect(!failed);
}

/// The banked test's banks: base, ext, transient
const bank_experts = [3]u32{ 20, 12, 8 };

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
}

// Inkling 2's routed geometry over three MCG banks of 20, 12 and 8 experts at each rate: every pair's row of
// moeInkling2Banked against moeInkling2 on its bank at top-1 with score 1 (at most 16 rows a call, so its decode
// launches), byte for byte, x bf16 values, the trellis I16 (the streamer's code arrays) and U16, on each selection of the
// texts, each line checked to have run the texts it names.
test "turboexl3 gpu inkling2 banked: every pair of moeInkling2Banked over three banks is moeInkling2's on its bank at top-1 with score 1, byte for byte, on the configured, g4 and g4m3 texts, at K1 and K2 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = config.get().*;
    defer config.configure(saved) catch {};
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_INKLING2_N", &.{ 16, 32 }, &nbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x1A4CBA ^ @as(u64, n));
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
                    const y = try g.adopt(try root.moeInkling2(s, try g.astype(x3, .float32), rb_u[b], inds, sc, dec, caps));
                    _ = try g.hostF32(y, buf[0 .. c * hidden]);
                    for (P, 0..) |p, i| @memcpy(want[p * hidden ..][0..hidden], buf[i * hidden ..][0..hidden]);
                }
            }
            const geo: decode.Geometry = .{ .hidden = hidden, .inter = inter, .topk = 1, .rate = rate, .dec = dec, .limit = format.softcapLimit(caps).?, .clamp = .softcap };
            const k3 = try root.kernelsFor(geo, .float32);
            const Sel = struct { sel: root.BankedSel, want: banked.Banked.Texts };
            const sels = [_]Sel{
                .{ .sel = .configured, .want = .banked },
                .{ .sel = .g4, .want = .g4 },
                .{ .sel = .g4m3, .want = .g4m3 },
            };
            const Arm = struct { banks: *const [3]root.Bank, name: []const u8 };
            for ([_]Arm{ .{ .banks = &rb_i, .name = "trellis=i16" }, .{ .banks = &rb_u, .name = "trellis=u16" } }) |arm| {
                for (sels) |sa| {
                    const y = try g.adopt(try root.moeInkling2BankedSel(s, x32, arm.banks, ids_a, dec, caps, sa.sel));
                    try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(hidden) }, mlx.getShape(y));
                    const ran = (try banked.forKernels(k3)).selTexts(sa.sel);
                    _ = try g.hostF32(y, got);
                    const d = bitDiff(got, want);
                    var bad: usize = 0;
                    for (got) |u| bad += @intFromBool(!std.math.isFinite(u));
                    const ok = d == 0 and bad == 0 and ran == sa.want;
                    std.debug.print("TURBOEXL3_INKLING2_BANKED n={d} rows={d} arm={s} texts={s} {s} differing={d}/{d} nonfinite={d} {s}\n", .{ n, rows, @tagName(sa.sel), @tagName(ran), arm.name, d, rows * hidden, bad, if (ok) "PASS" else "FAIL" });
                    if (!ok) failed = true;
                }
            }
        }
    }
    try testing.expect(!failed);
}

// A streamer's prompt call over one MCG bank of 24 experts: S row-slots each reading x's row rows[p] (x [64, hidden] bf16,
// rows drawn with repeats) at slot inds[p] and a bf16-exact weight. moeInkling2Rows (y f32) against moeInkling2 at top-1
// over take(x, rows) as f32 [1, S, hidden], word for word, at S 17 (the smallest prompt call), 48 and 400.
test "turboexl3 gpu inkling2 rows: moeInkling2Rows gives moeInkling2's bytes at top-1 over the gathered rows, at K1 and K2 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    if (!config.get().prefill or !config.get().nax) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_INKLING2_N", &.{ 16, 32 }, &nbuf);
    const n_x: u32 = 64;
    const n_e: u32 = 24;
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x1A4C0 ^ @as(u64, n));
        const rnd = prng.random();
        var hbank = try ref.synthBank(a, rnd, n_e, hidden, inter, rate);
        defer hbank.deinit(a);
        const rb = try rootBank(&g, &hbank, .uint16);
        const x = try a.alloc(f32, n_x * hidden);
        defer a.free(x);
        for (x) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
        const xbf = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ ci(n_x), ci(hidden) }, .float32), .bfloat16);
        for ([_]u32{ 17, 48, 400 }) |S| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            const rows = try a.alloc(u32, S);
            defer a.free(rows);
            const slots = try a.alloc(u32, S);
            defer a.free(slots);
            const w = try a.alloc(f32, S);
            defer a.free(w);
            for (rows, slots, w) |*r, *sl, *wv| {
                r.* = rnd.uintLessThan(u32, n_x);
                sl.* = rnd.uintLessThan(u32, n_e);
                wv.* = bfExact(0.05 + rnd.float(f32));
            }
            const rows_a = try g.hostArray(std.mem.sliceAsBytes(rows), &.{ci(S)}, .uint32);
            const inds_a = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(S)}, .uint32);
            const w_a = try g.hostArray(std.mem.sliceAsBytes(w), &.{ci(S)}, .float32);
            const y = try g.adopt(try root.moeInkling2Rows(s, xbf, rows_a, rb, inds_a, w_a, dec, caps, .float32));
            try testing.expectEqualSlices(c_int, &.{ ci(S), ci(hidden) }, mlx.getShape(y));
            const xt = try g.reshape(try g.astype(try g.take(xbf, rows_a, 0), .float32), &.{ 1, ci(S), ci(hidden) });
            const yw = try g.adopt(try root.moeInkling2(s, xt, rb, try g.reshape(inds_a, &.{ 1, ci(S), 1 }), try g.reshape(w_a, &.{ 1, ci(S), 1 }), dec, caps));
            const got = try a.alloc(f32, S * hidden);
            defer a.free(got);
            const want = try a.alloc(f32, S * hidden);
            defer a.free(want);
            _ = try g.hostF32(y, got);
            _ = try g.hostF32(yw, want);
            const d = bitDiff(got, want);
            std.debug.print("TURBOEXL3_INKLING2_ROWS n={d} S={d} differing={d}/{d} {s}\n", .{ n, S, d, S * hidden, if (d == 0) "PASS" else "FAIL" });
            if (d != 0) failed = true;
        }
    }
    try testing.expect(!failed);
}
