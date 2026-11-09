//! Ling-3.1-flash (`models/ling/api.zig`). CPU (always): each Ling text is its base text with the SwiGLU line swapped for
//! Ling's clamp piece (or, banked, DeepSeek's clamp piece swapped for fin7's own line or Ling's) and nothing else; the
//! reference clamp's rule; the refusals; every trunk shape a width the dense kernels admit; the one-row table's launch
//! per trunk shape. GPU (TURBOEXL3_GPU=1, else skipped), at Ling's shapes (hidden 4096, expert width 1280: ten
//! 128-blocks, 512 experts, top-8, MUL1 at n 40 = K 2.5 and n 32 = K 2, the 2.60 and 2.10 packs' routed rates):
//! - `root.moeLing` at a limit no value reaches gives `root.moe`'s bytes (decode and prefill rows), and at limits 0 and 10
//!   equals `ref.moeRowClampedAt` (`silu_out`) on sampled rows of decode rows 1, 4, 16 and prefill rows 32, 1328;
//! - `root.moeLing` at top-4 on the operands the plugin's MTP head hands it (x f32, the slots u32, the weights f32
//!   rounded through bf16, limit 0) equals `ref.moeRowClampedAt` on sampled rows of decode rows 1, 4 and prefill rows
//!   17, 1328, 4096, at n 40;
//! - every pair of `root.moeLingBanked` over three banks is `moeLing`'s on its bank at top-1 with score 1, byte for byte,
//!   at limits 0 and 10, rows 1, 2, 8, 16, 48, on the configured texts (banked group 2), the grouped texts (g4) and the
//!   M = 3 member path (g4m3), each selected (`moeLingBankedSel`) and checked to have run;
//! - the 17 routed geometries those three tests make in one process have their kernels in root's cache at once;
//! - `root.linear` at every Ling trunk shape (K6; the dense FFN K4) equals the host reference;
//! - a one-row call on its shape's `dense.one_row_table` launch (`Config.dec_one_row_table`) gives one row's own bytes and
//!   the three-launch chain's at every trunk shape, on that launch;
//! - `root.ling.linearPair` (z | a, the shared and the dense gate | up) gives each member's `linear` bytes at decode
//!   rows 1 to 16, the prepare inside and in its own launch;
//! - `root.ling.kdaProj` (a KDA layer's q|k|v, f and g in one launch) gives each member's `linear` bytes at decode rows
//!   1 to 8 and 16, after its own prepare and over each member's own prep_sc rows joined (`kdaProjPrepared`);
//! - each row of a prompt call's `root.linear` (bf16 x) has a whole-prompt call's bytes where both calls run MLX's
//!   regular matmul, at every EXL3 linear the 2.60 pack serves at prompt rows (`dense_cases` but the head, and the MTP
//!   layer's eh_proj): the chunk loop's spans of the standard prompt's 18,380 rows, calls over its first 1 to 8,192
//!   rows and an 8,192-row call's halves; the decode GEMVs (16 rows or fewer), the register-path GEMM (17 to 32) and
//!   MLX's split-K matmul are reported, not asserted;
//! - on operands as the plugin serves them (the pack's I16 trellis words; `[1, rows, in]` bf16 activations, whole or a
//!   row view of a wider array), `root.lingSharedExpert` gives the composed shared expert's words (rows 1 to 4),
//!   `root.ling.linearPair` each member's `linear` words (rows 1, 2, 4, 6), and the one-row table one row's own words
//!   (every trunk shape), each also the words of the same call on U16 words and a contiguous `[rows, in]`;
//! - D35: each DEC-ONE GEMV variant (`Config.dec_variant`: t4, pf, pft4) gives the served GEMV's words at every trunk
//!   shape and the LM head, rows 1 to 4, through `linear` and over prepared rows; with red checks (a flipped word,
//!   another K split).
//! TURBOEXL3_LING_N (rates in halfwords per tile; default 40,32) narrows the routed tests. Lines: TURBOEXL3_LING_BYTES,
//! TURBOEXL3_LING_PARITY, TURBOEXL3_LING_TOP4, TURBOEXL3_LING_BANKED, TURBOEXL3_LING_CACHE, TURBOEXL3_LING_DENSE,
//! TURBOEXL3_LING_ONE_ROW_TABLE, TURBOEXL3_LING_PAIR, TURBOEXL3_LING_KDA_PROJ, TURBOEXL3_LING_ROWINV,
//! TURBOEXL3_LING_SERVED, TURBOEXL3_LING_DEC_VARIANT (_RED, _DONE).
//! Timing (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; asserts nothing, guidance only): every trunk linear, `moeLing`
//! and `moeLingBanked` at Ling's shapes and rows, the decode forward's linears chained over the 72 layers, and the dense
//! decode GEMV's launch shapes swept per trunk shape on that chain at rows 1 to 8 (asserting bytes only); D35's arms:
//! the DEC-ONE GEMV variants and the router's rows launch (router_test.zig). Lines: TURBOEXL3_LING_TIME,
//! TURBOEXL3_LING_CHAIN, TURBOEXL3_LING_GEMV (_BEST, _TABLE, _DONE).
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
        for ([_]u32{ 1, 4, top_k }) |k| {
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

// The plugin's MTP head at top-4 (turboexl3_ling's MLX_SERVE_LING_MTP_HEAD_TOP4, rejected as R7 and removed from the
// plugin; this case stays as moeLing's top-4 coverage) called moeLing on its resident bank: the 512 experts' I16
// trellis words stacked on axis 0 and viewed as u16, suh and svh f16 (rootBank at .uint16), with x f32 [1, rows,
// hidden], the slots u32 and the weights f32 [1, rows, 4]. The weights are rounded through bf16 (the block's cast to
// the activation dtype, then the head's to f32), and the limit is 0. Decode rows 1 and 4, then prefill rows 17, 1328
// and 4096 (the head's history pass). x ~ N(0, 8^2), with the f32 output held to ref.moeRowClampedAt on sampled rows at
// the top-8 test's bar.
test "turboexl3 gpu ling top4: root.moeLing at top-4 on the MTP head's operands (x f32, u32 slots, bf16-rounded f32 weights, limit 0) equals the reference at Ling-3.1-flash's routed geometry, decode rows 1 and 4 and prefill rows 17, 1328 and 4096 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    const k: u32 = 4;
    const n: u32 = 40;
    const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
    var prng = std.Random.DefaultPrng.init(0x70B4 ^ @as(u64, n));
    const rnd = prng.random();
    var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
    defer bank.deinit(a);
    const rb = try rootBank(&g, &bank, .uint16);
    var failed = false;
    for ([_]u32{ 1, 4, 17, 1328, 4096 }) |rows| {
        const m2 = g.mark();
        defer g.resetTo(m2);
        const x = try a.alloc(f32, rows * hidden);
        defer a.free(x);
        for (x) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
        const slots = try a.alloc(u32, rows * k);
        defer a.free(slots);
        const scores = try a.alloc(f32, rows * k);
        defer a.free(scores);
        try route(a, rnd, rows, experts, k, 1.5, slots, scores);
        // the head's weights sum to routed_scaling_factor 2.5, then land in bf16
        for (scores) |*w| w.* = bfExact(2.5 * w.*);
        const x3 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(hidden) }, .float32);
        const inds3 = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ 1, ci(rows), ci(k) }, .uint32);
        const sc3 = try g.astype(try g.astype(try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, ci(rows), ci(k) }, .float32), .bfloat16), .float32);
        var sbuf: [3]u32 = undefined;
        const sample = sampleRows(rows, &sbuf);
        const want = try a.alloc(f32, sample.len * hidden);
        defer a.free(want);
        for (sample, 0..) |r, i| _ = try ref.moeRowClampedAt(a, &bank, .mul1, 0, .silu_out, x[r * hidden ..][0..hidden], slots[r * k ..][0..k], scores[r * k ..][0..k], want[i * hidden ..][0..hidden]);
        const y = try g.adopt(try root.moeLing(s, x3, rb, inds3, sc3, .mul1, 0));
        try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(hidden) }, mlx.getShape(y));
        try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(y));
        const got = try a.alloc(f32, rows * hidden);
        defer a.free(got);
        _ = try g.hostF32(try g.astype(y, .float32), got);
        var bad: usize = 0;
        for (got) |v| bad += @intFromBool(!std.math.isFinite(v));
        var worst: f64 = 0;
        for (sample, 0..) |r, i| worst = @max(worst, ref.compare(got[r * hidden ..][0..hidden], want[i * hidden ..][0..hidden]).max_ratio);
        const bar: f64 = 5e-3;
        const ok = bad == 0 and worst <= bar;
        const path: []const u8 = if (rows <= root.kernels.DECODE_ROWS_MAX) "decode" else "prefill";
        std.debug.print("TURBOEXL3_LING_TOP4 n={d} rows={d} k={d} path={s} x=float32 worst_vs_ref={e:.3} bar={e:.1} nonfinite={d} {s}\n", .{ n, rows, k, path, worst, bar, bad, if (ok) "PASS" else "FAIL" });
        if (!ok) failed = true;
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
// trellis I16 (the streamer's code arrays) and U16, at limits 0 (fin7b / fin7bg) and 10 (fin7cob / fin7cobg), on each
// selection of the texts (`moeLingBankedSel`): the configured ones at the default configuration (banked group 2), the
// grouped texts (g4) and the grouped texts with the M = 3 member path (g4m3, the served set), each line checked to have
// run the texts it names (the kernel cache fixes the configured texts at a geometry's first banked call, so setting
// `Config.banked_group` in the process selects nothing). x ~ N(0, 8^2), so the clamp acts at 10.
test "turboexl3 gpu ling banked: every pair of moeLingBanked over three banks is moeLing's on its bank at top-1 with score 1, byte for byte, at limits 0 and 10, on the configured texts (group 2), the grouped texts (g4) and the M = 3 path (g4m3), each selected (TURBOEXL3_GPU=1)" {
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
                // the banked state the calls run on (the geometry `moeLingBanked` builds), whose texts each line names
                const geo: decode.Geometry = .{ .hidden = hidden, .inter = inter, .topk = 1, .rate = rate, .dec = .mul1, .limit = @intCast(lim), .clamp = .silu_out };
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
                        const y = try g.adopt(try root.moeLingBankedSel(s, x32, arm.banks, ids_a, .mul1, lim, sa.sel));
                        try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(hidden) }, mlx.getShape(y));
                        const ran = (try banked.forKernels(k3)).selTexts(sa.sel);
                        _ = try g.hostF32(y, got);
                        var d: usize = 0;
                        var bad: usize = 0;
                        for (got, want) |u, v| {
                            d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
                            bad += @intFromBool(!std.math.isFinite(u));
                        }
                        const ok = d == 0 and bad == 0 and ran == sa.want;
                        std.debug.print("TURBOEXL3_LING_BANKED n={d} rows={d} limit={d} arm={s} texts={s} {s} differing={d}/{d} nonfinite={d} {s}\n", .{ n, rows, lim, @tagName(sa.sel), @tagName(ran), arm.name, d, rows * hidden, bad, if (ok) "PASS" else "FAIL" });
                        if (!ok) failed = true;
                    }
                }
            }
        }
    }
    try testing.expect(!failed);
}

// The routed geometries this file's GPU tests make, each one's kernels in root's cache at once (`root.kernelsFor`): the
// cache keeps a geometry's kernels for the process's life and evicts none, and a test binary runs every test in one
// process. The moeLing test makes 12 (top-8 at n 40 and 32, limits 0, 10 and 128, out bf16 and f32; root.moe at limit 0
// shares moeLing's), the top-4 test 1 (n 40, limit 0, f32) and the banked test 4 (top-1 at n 40 and 32, limits 0 and
// 10, f32): 17 in all. A cache of 16 failed the banked test's last geometry with Exl3EngineCacheFull after its byte
// lines. Kernel objects only: nothing launches.
test "turboexl3 gpu ling kernel cache: the 17 routed geometries the Ling GPU tests make (top-8, top-4 and top-1, n 40 and 32, limits 0, 10 and 128, out bf16 and f32) have their kernels in root's cache at once (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const Key = struct { topk: u32, n: u32, lim: u32, out: mlx.mlx_dtype };
    var keys: [17]Key = undefined;
    var nk: usize = 0;
    for ([_]u32{ 40, 32 }) |n| for ([_]u32{ 0, @intCast(limit), 128 }) |lim| for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
        keys[nk] = .{ .topk = top_k, .n = n, .lim = lim, .out = dt };
        nk += 1;
    };
    keys[nk] = .{ .topk = 4, .n = 40, .lim = 0, .out = .float32 };
    nk += 1;
    for ([_]u32{ 40, 32 }) |n| for ([_]u32{ 0, @intCast(limit) }) |lim| {
        keys[nk] = .{ .topk = 1, .n = n, .lim = lim, .out = .float32 };
        nk += 1;
    };
    try testing.expectEqual(keys.len, nk);
    var got: [keys.len]*decode.Kernels3 = undefined;
    for (keys, &got, 0..) |k, *p, i| {
        const rate = format.Rate.fromN(k.n) orelse return error.Exl3RateUnsupported;
        p.* = try root.kernelsFor(.{ .hidden = hidden, .inter = inter, .topk = k.topk, .rate = rate, .dec = .mul1, .limit = k.lim, .clamp = .silu_out }, k.out);
        std.debug.print("TURBOEXL3_LING_CACHE geometry={d}/{d} topk={d} n={d} limit={d} out={s} cached\n", .{ i + 1, keys.len, k.topk, k.n, k.lim, @tagName(k.out) });
    }
    // each geometry its own kernels, every one still held after the last was made
    for (keys, got, 0..) |k, p, i| {
        try testing.expect(p.geo.topk == k.topk and p.geo.rate.n == k.n and p.geo.limit == k.lim and p.out_dtype == k.out);
        for (got[0..i]) |q| try testing.expect(p != q);
    }
    std.debug.print("TURBOEXL3_LING_CACHE {d} geometries held at once PASS\n", .{keys.len});
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

/// `dense.one_row_table`'s one-row launch for each trunk shape (`dense_cases` by name): output tiles per threadgroup (0 =
/// the served launch's) and the prepare inside or in its own launch (null = the served placement)
const OneRowWant = struct { name: []const u8, bt: u8, prep_in: ?bool };
const one_row_want = [_]OneRowWant{
    .{ .name = "kda.qkv", .bt = 1, .prep_in = false },
    .{ .name = "kda.f/g", .bt = 1, .prep_in = false },
    .{ .name = "kda.o/mla.dense", .bt = 1, .prep_in = false },
    .{ .name = "mla.q_a", .bt = 0, .prep_in = null },
    .{ .name = "mla.q_b", .bt = 8, .prep_in = false },
    .{ .name = "mla.kv_a", .bt = 2, .prep_in = false },
    .{ .name = "shared.gate/up", .bt = 2, .prep_in = null },
    .{ .name = "shared.down", .bt = 1, .prep_in = null },
    .{ .name = "dense.gate/up", .bt = 0, .prep_in = null },
    .{ .name = "dense.down", .bt = 0, .prep_in = null },
    .{ .name = "head", .bt = 0, .prep_in = null },
};

fn oneRowWant(name: []const u8) OneRowWant {
    for (one_row_want) |w| if (std.mem.eql(u8, w.name, name)) return w;
    unreachable;
}

test "turboexl3 ling: Config.dec_one_row_table gives each Ling-3.1-flash trunk shape its one_row_table launch at one row only, and every other shape the rest of the configuration's" {
    try testing.expectEqual(dense_cases.len, one_row_want.len);
    const off: config.Config = .{};
    const on: config.Config = .{ .dec_one_row_table = true };
    for (dense_cases) |c| {
        const w = oneRowWant(c.name);
        const shape: dense.Shape = .{ .in = c.in, .out = c.out, .rate = format.Rate.fromN(c.n).?, .dec = .mul1, .out_dtype = .bfloat16 };
        try testing.expectEqual(w.bt != 0, dense.oneRowFor(shape) != null);
        const b_off = dense.Dense.btSetFor(off, shape);
        const b_on = dense.Dense.btSetFor(on, shape);
        try testing.expectEqual(@as(u8, 0), b_off[1]);
        try testing.expectEqual(w.bt, b_on[1]);
        for (b_off[2..], b_on[2..]) |x, y| try testing.expect(x == 0 and y == 0);
        try testing.expectEqual(@as(?bool, null), dense.Dense.onePrepFor(off, shape));
        try testing.expectEqual(w.prep_in, dense.Dense.onePrepFor(on, shape));
    }
    // a table shape at another rate is not a table shape
    const k4: dense.Shape = .{ .in = 4096, .out = 24576, .rate = format.Rate.fromN(64).?, .dec = .mul1, .out_dtype = .bfloat16 };
    try testing.expect(dense.oneRowFor(k4) == null);
    try testing.expectEqual(@as(u8, 0), dense.Dense.btSetFor(on, k4)[1]);
}

test "turboexl3 gpu ling dense: the one-row launch table (Config.dec_one_row_table) gives one row's own bytes and the three-launch chain's at every Ling-3.1-flash trunk shape, on its table launch (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = config.get().*;
    defer config.configure(saved) catch unreachable;
    var served_cfg = saved;
    served_cfg.dec_one_row_table = false;
    var table_cfg = served_cfg;
    table_cfg.dec_one_row_table = true;
    var failed: usize = 0;
    for (dense_cases) |c| {
        const w = oneRowWant(c.name);
        var prng = std.Random.DefaultPrng.init(0x1E0B ^ (@as(u64, c.in) << 24) ^ c.out);
        const rnd = prng.random();
        const rate = format.Rate.fromN(c.n).?;
        var lin = try ref.synthLinear(a, rnd, 1, c.in, c.out, rate, .mul1);
        defer lin.deinit(a);
        const mark = g.mark();
        defer g.resetTo(mark);
        const trellis = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(c.in / 16), ci(c.out / 16), ci(c.n) }, .uint16);
        const suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(c.in)}, .float16);
        const svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(c.out)}, .float16);
        const x = try a.alloc(f32, c.in);
        defer a.free(x);
        for (x) |*v| v.* = bfExact(rnd.floatNorm(f32));
        const got = try a.alloc(f32, 3 * c.out);
        defer a.free(got);
        for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dt| {
            const geo: dense.Shape = .{ .in = c.in, .out = c.out, .rate = rate, .dec = .mul1, .out_dtype = dt };
            try config.configure(served_cfg);
            var chain = try dense.Dense.init(geo);
            defer chain.deinit();
            chain.dec_on = false;
            var own = try dense.Dense.init(geo);
            defer own.deinit();
            try config.configure(table_cfg);
            var table = try dense.Dense.init(geo);
            defer table.deinit();
            try config.configure(served_cfg);
            const xr = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(c.in) }, .float32), dt);
            for ([_]*dense.Dense{ &chain, &own, &table }, 0..) |d, k| {
                const y = try g.adopt(try d.linear(s, xr, trellis, suh, svh, 1));
                _ = try g.hostF32(try g.astype(y, .float32), got[k * c.out ..][0..c.out]);
            }
            const own_in = try own.decPrepInside(1);
            const table_in = try table.decPrepInside(1);
            // the table's launch: its tiles per threadgroup, its prepare placement (null: the served one)
            const shape_ok = table.bt_set[1] == w.bt and table_in == (w.prep_in orelse own_in);
            var d_own: usize = 0;
            var d_table: usize = 0;
            for (got[c.out..][0..c.out], got[2 * c.out ..][0..c.out], got[0..c.out]) |o, t, ch| {
                d_own += @intFromBool(@as(u32, @bitCast(o)) != @as(u32, @bitCast(ch)));
                d_table += @intFromBool(@as(u32, @bitCast(t)) != @as(u32, @bitCast(ch)));
            }
            const ok = shape_ok and d_own == 0 and d_table == 0;
            std.debug.print("TURBOEXL3_LING_ONE_ROW_TABLE {s} in={d} out={d} n={d} x={s} own_prepare={s} table_prepare={s} table_bt={d} own_vs_chain={d}/{d} table_vs_chain={d}/{d} {s}\n", .{ c.name, c.in, c.out, c.n, @tagName(dt), if (own_in) "inside" else "own_launch", if (table_in) "inside" else "own_launch", table.bt_set[1], d_own, c.out, d_table, c.out, if (ok) "PASS" else "FAIL" });
            failed += @intFromBool(!ok);
        }
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

/// Ling-3.1-flash's same-input pairs (`root.ling.linearPair`): the KDA layers' z | a, the shared expert's and the dense
/// FFN's gate | up
const PairCase = struct { name: []const u8, in: u32, out: u32, n: u32 };
const pair_cases = [_]PairCase{
    .{ .name = "kda.z|a", .in = 4096, .out = 8192, .n = 96 },
    .{ .name = "shared.gate|up", .in = 4096, .out = 1280, .n = 96 },
    .{ .name = "dense.gate|up", .in = 4096, .out = 10240, .n = 64 },
};
/// DEC-ONE's rows (1 to 4), a verify's widest decode-lane call (6) and the GEMV chain's chunks past them (5, 8, 16)
const pair_rows = [_]u32{ 1, 2, 3, 4, 5, 6, 8, 16 };

fn synthProj(a: std.mem.Allocator, g: *G, rnd: std.Random, in: u32, out: u32, n: u32) !struct { lin: ref.Linear, p: root.Proj } {
    var lin = try ref.synthLinear(a, rnd, 1, in, out, format.Rate.fromN(n).?, .mul1);
    errdefer lin.deinit(a);
    const p: root.Proj = .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(out)}, .float16),
    };
    return .{ .lin = lin, .p = p };
}

test "turboexl3 gpu ling pair: z | a and the shared and dense gate | up in one launch give each member's linear bytes, the prepare inside or in its own launch, at decode rows 1 to 16 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var failed: usize = 0;
    var checks: usize = 0;
    var forms = [2]usize{ 0, 0 };
    for (pair_cases) |c| {
        var prng = std.Random.DefaultPrng.init(0x7017 ^ (@as(u64, c.in) << 24) ^ c.out);
        const rnd = prng.random();
        const mark = g.mark();
        defer g.resetTo(mark);
        var m0 = try synthProj(a, &g, rnd, c.in, c.out, c.n);
        defer m0.lin.deinit(a);
        var m1 = try synthProj(a, &g, rnd, c.in, c.out, c.n);
        defer m1.lin.deinit(a);
        const sh = try root.ling.pair.pairShape(m0.p, m1.p);
        const max_rows: u32 = 16;
        const xs = try a.alloc(f32, max_rows * c.in);
        defer a.free(xs);
        for (xs) |*v| v.* = bfExact(rnd.floatNorm(f32));
        const want = try a.alloc(f32, 2 * max_rows * c.out);
        defer a.free(want);
        const got = try a.alloc(f32, 2 * max_rows * c.out);
        defer a.free(got);
        for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dt| {
            // the member's cached Dense decides the form; the wide arm gives one row two rows' launch shape
            // (dec_prep_rows 0, bt_set[1] 2), so both forms run at one row
            const d = try root.denseCached(.{ .in = c.in, .out = c.out, .rate = format.Rate.fromN(c.n).?, .dec = .mul1, .out_dtype = dt });
            const saved_prep = d.dec_prep_rows;
            const saved_bt = d.bt_set[1];
            defer {
                d.dec_prep_rows = saved_prep;
                d.bt_set[1] = saved_bt;
                d.resetCfgs();
            }
            for ([_]bool{ false, true }) |wide| {
                d.dec_prep_rows = if (wide) 0 else 1;
                d.bt_set[1] = if (wide) 2 else 0;
                d.resetCfgs();
                for (pair_rows) |rows| {
                    const rm = g.mark();
                    defer g.resetTo(rm);
                    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. rows * c.in]), &.{ ci(rows), ci(c.in) }, .float32), dt);
                    const n_out = rows * c.out;
                    for ([_]root.Proj{ m0.p, m1.p }, 0..) |mp, k| {
                        const y = try g.adopt(try root.linear(s, x, mp.trellis, mp.suh, mp.svh, .mul1));
                        _ = try g.hostF32(try g.astype(y, .float32), want[k * n_out ..][0..n_out]);
                    }
                    const p = (try root.ling.pair.plan(sh, .mul1, rows, dt)) orelse {
                        std.debug.print("TURBOEXL3_LING_PAIR {s} x={s} wide={} rows={d} not served FAIL\n", .{ c.name, @tagName(dt), wide, rows });
                        failed += 1;
                        continue;
                    };
                    const yp = try g.adopt(try root.ling.linearPair(s, x, m0.p, m1.p, .mul1));
                    try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(2 * c.out) }, mlx.getShape(yp));
                    _ = try g.hostF32(try g.astype(yp, .float32), got[0 .. 2 * n_out]);
                    // member m's columns of each row against member m's linear
                    var differing: usize = 0;
                    for (0..rows) |r| for (0..2) |k| {
                        const gr = got[r * 2 * c.out + k * c.out ..][0..c.out];
                        const wr = want[k * n_out + r * c.out ..][0..c.out];
                        for (gr, wr) |u, w| differing += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
                    };
                    const form_ok = p.inside == try d.decPrepInside(rows);
                    const ok = form_ok and differing == 0;
                    checks += 1;
                    forms[@intFromBool(p.inside)] += 1;
                    std.debug.print("TURBOEXL3_LING_PAIR {s} in={d} out={d}+{d} n={d} x={s} wide={} rows={d} prepare={s} ks={d} rb={d} bt={d} differing={d}/{d} {s}\n", .{ c.name, c.in, c.out, c.out, c.n, @tagName(dt), wide, rows, if (p.inside) "inside" else "own_launch", p.ks, p.rb, p.bt, differing, 2 * n_out, if (ok) "PASS" else "FAIL" });
                    if (!ok) failed += 1;
                }
            }
        }
    }
    std.debug.print("TURBOEXL3_LING_PAIR_DONE checks={d} inside={d} own_launch={d} {s}\n", .{ checks, forms[1], forms[0], if (failed == 0) "PASS" else "FAIL" });
    try testing.expect(forms[0] > 0 and forms[1] > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}

/// A projection's trellis as the plugin's loader hands it over: the pack's I16 words (the same bits as `p`'s U16).
fn packTrellis(g: *G, p: root.Proj) !root.Proj {
    var v = mlx.mlx_array_new();
    mlx.check(mlx.mlx_view(&v, p.trellis, .int16, g.s)) catch |e| {
        _ = mlx.mlx_array_free(v);
        return e;
    };
    return .{ .trellis = try g.adopt(v), .suh = p.suh, .svh = p.svh };
}

/// Rows 1 to `rows` of `xs` ((rows + 2) x `in` values) in `dt`, as the plugin hands an activation over (`served`:
/// `[1, rows, in]`, the whole array, or with `view` a slice of a wider array's rows, as a verify's row loop cuts one)
/// and as the byte tests build it (`flat`: a contiguous `[rows, in]`).
fn servedAct(g: *G, xs: []const f32, rows: u32, in: u32, dt: mlx.mlx_dtype, view: bool) !struct { served: mlx.mlx_array, flat: mlx.mlx_array } {
    const n = @as(usize, rows) * in;
    const flat = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[in..][0..n]), &.{ ci(rows), ci(in) }, .float32), dt);
    const served = if (view) blk: {
        const wide = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. n + 2 * @as(usize, in)]), &.{ 1, ci(rows + 2), ci(in) }, .float32), dt);
        break :blk try g.slice(wide, &.{ 0, 1, 0 }, &.{ 1, ci(rows + 1), ci(in) }, &.{ 1, 1, 1 });
    } else try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[in..][0..n]), &.{ 1, ci(rows), ci(in) }, .float32), dt);
    return .{ .served = served, .flat = flat };
}

fn wordsDiffer(p: []const f32, q: []const f32) usize {
    var d: usize = 0;
    for (p, q) |u, w| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return d;
}

test "turboexl3 gpu ling served operands: the shared expert, the pairs and the one-row table on the pack's I16 trellis words and [1, rows, in] activations (whole or a row view) give their composed chains' words and those of U16 words on [rows, in] rows (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var checks: usize = 0;
    var failed: usize = 0;
    // the shared expert (SHARED_F3): turboexl3's two launches against the composed chain the block runs (each `linear`,
    // MLX's sigmoid, multiply and multiply in bf16, the words of the host's table SwiGLU), both on served operands
    {
        var prng = std.Random.DefaultPrng.init(0x5E2D_F3);
        const rnd = prng.random();
        const mark = g.mark();
        defer g.resetTo(mark);
        var gm = try synthProj(a, &g, rnd, 4096, 1280, 96);
        defer gm.lin.deinit(a);
        var um = try synthProj(a, &g, rnd, 4096, 1280, 96);
        defer um.lin.deinit(a);
        var dm = try synthProj(a, &g, rnd, 1280, 4096, 96);
        defer dm.lin.deinit(a);
        const gate = try packTrellis(&g, gm.p);
        const up = try packTrellis(&g, um.p);
        const down = try packTrellis(&g, dm.p);
        const tab = try g.adopt(try root.lingSigmoidTable(s, .bfloat16, false));
        const xs = try a.alloc(f32, 6 * 4096);
        defer a.free(xs);
        for (xs) |*v| v.* = bfExact(rnd.floatNorm(f32));
        const want = try a.alloc(f32, 4 * 4096);
        defer a.free(want);
        const got = try a.alloc(f32, 4 * 4096);
        defer a.free(got);
        const flat = try a.alloc(f32, 4 * 4096);
        defer a.free(flat);
        for ([_]u32{ 1, 2, 3, 4 }) |rows| for ([_]bool{ false, true }) |view| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const x = try servedAct(&g, xs, rows, 4096, .bfloat16, view);
            const n = @as(usize, rows) * 4096;
            if (!(try root.lingSharedTakes(x.served, gate, up, down, .mul1))) {
                std.debug.print("TURBOEXL3_LING_SERVED shared rows={d} view={} not served FAIL\n", .{ rows, view });
                failed += 1;
                continue;
            }
            const gv = try g.adopt(try root.linear(s, x.served, gate.trellis, gate.suh, gate.svh, .mul1));
            const uv = try g.adopt(try root.linear(s, x.served, up.trellis, up.suh, up.svh, .mul1));
            const h = try g.mul(try g.mul(gv, try g.sigmoid(gv)), uv);
            _ = try g.hostF32(try g.astype(try g.adopt(try root.linear(s, h, down.trellis, down.suh, down.svh, .mul1)), .float32), want[0..n]);
            _ = try g.hostF32(try g.astype(try g.adopt(try root.lingSharedExpert(s, x.served, gate, up, down, 0, tab, .mul1)), .float32), got[0..n]);
            _ = try g.hostF32(try g.astype(try g.adopt(try root.lingSharedExpert(s, x.flat, gm.p, um.p, dm.p, 0, tab, .mul1)), .float32), flat[0..n]);
            const d_chain = wordsDiffer(got[0..n], want[0..n]);
            const d_flat = wordsDiffer(got[0..n], flat[0..n]);
            const ok = d_chain == 0 and d_flat == 0;
            checks += 1;
            failed += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_LING_SERVED shared rows={d} view={} vs_chain={d}/{d} vs_u16_flat={d}/{d} {s}\n", .{ rows, view, d_chain, n, d_flat, n, if (ok) "PASS" else "FAIL" });
        };
    }
    // the pairs (PAIR): each member's columns of the pair against its own `linear`, both on served operands, at the
    // decode and verify widths
    for (pair_cases) |c| {
        var prng = std.Random.DefaultPrng.init(0x5E2D_9A ^ (@as(u64, c.in) << 24) ^ c.out);
        const rnd = prng.random();
        const mark = g.mark();
        defer g.resetTo(mark);
        var m0 = try synthProj(a, &g, rnd, c.in, c.out, c.n);
        defer m0.lin.deinit(a);
        var m1 = try synthProj(a, &g, rnd, c.in, c.out, c.n);
        defer m1.lin.deinit(a);
        const p0 = try packTrellis(&g, m0.p);
        const p1 = try packTrellis(&g, m1.p);
        const xs = try a.alloc(f32, 8 * c.in);
        defer a.free(xs);
        for (xs) |*v| v.* = bfExact(rnd.floatNorm(f32));
        const want = try a.alloc(f32, 2 * 6 * c.out);
        defer a.free(want);
        const got = try a.alloc(f32, 2 * 6 * c.out);
        defer a.free(got);
        const flat = try a.alloc(f32, 2 * 6 * c.out);
        defer a.free(flat);
        for ([_]u32{ 1, 2, 4, 6 }) |rows| for ([_]bool{ false, true }) |view| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const x = try servedAct(&g, xs, rows, c.in, .bfloat16, view);
            const n_out = @as(usize, rows) * c.out;
            for ([_]root.Proj{ p0, p1 }, 0..) |mp, k| {
                _ = try g.hostF32(try g.astype(try g.adopt(try root.linear(s, x.served, mp.trellis, mp.suh, mp.svh, .mul1)), .float32), want[k * n_out ..][0..n_out]);
            }
            _ = try g.hostF32(try g.astype(try g.adopt(try root.ling.linearPair(s, x.served, p0, p1, .mul1)), .float32), got[0 .. 2 * n_out]);
            _ = try g.hostF32(try g.astype(try g.adopt(try root.ling.linearPair(s, x.flat, m0.p, m1.p, .mul1)), .float32), flat[0 .. 2 * n_out]);
            var d_chain: usize = 0;
            for (0..rows) |r| for (0..2) |k| {
                d_chain += wordsDiffer(got[r * 2 * c.out + k * c.out ..][0..c.out], want[k * n_out + r * c.out ..][0..c.out]);
            };
            const d_flat = wordsDiffer(got[0 .. 2 * n_out], flat[0 .. 2 * n_out]);
            const ok = d_chain == 0 and d_flat == 0;
            checks += 1;
            failed += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_LING_SERVED pair {s} rows={d} view={} vs_linears={d}/{d} vs_u16_flat={d}/{d} {s}\n", .{ c.name, rows, view, d_chain, 2 * n_out, d_flat, 2 * n_out, if (ok) "PASS" else "FAIL" });
        };
    }
    // the one-row table (ONE_ROW_TABLE): a one-row call on its table launch against one row's own, both on served
    // operands (one row viewed out of three and reshaped to [1, in] as `rowsOf` does), at every trunk shape
    {
        const saved = config.get().*;
        defer config.configure(saved) catch unreachable;
        var served_cfg = saved;
        served_cfg.dec_one_row_table = false;
        var table_cfg = served_cfg;
        table_cfg.dec_one_row_table = true;
        for (dense_cases) |c| {
            var prng = std.Random.DefaultPrng.init(0x5E2D_1E ^ (@as(u64, c.in) << 24) ^ c.out);
            const rnd = prng.random();
            const mark = g.mark();
            defer g.resetTo(mark);
            var m = try synthProj(a, &g, rnd, c.in, c.out, c.n);
            defer m.lin.deinit(a);
            const p = try packTrellis(&g, m.p);
            const xs = try a.alloc(f32, 3 * c.in);
            defer a.free(xs);
            for (xs) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const got = try a.alloc(f32, 3 * c.out);
            defer a.free(got);
            const geo: dense.Shape = .{ .in = c.in, .out = c.out, .rate = format.Rate.fromN(c.n).?, .dec = .mul1, .out_dtype = .bfloat16 };
            try config.configure(served_cfg);
            var own = try dense.Dense.init(geo);
            defer own.deinit();
            try config.configure(table_cfg);
            var table = try dense.Dense.init(geo);
            defer table.deinit();
            try config.configure(served_cfg);
            const x = try servedAct(&g, xs, 1, c.in, .bfloat16, true);
            const x2 = try g.reshape(x.served, &.{ 1, ci(c.in) });
            _ = try g.hostF32(try g.astype(try g.adopt(try table.linear(s, x2, p.trellis, p.suh, p.svh, 1)), .float32), got[0..c.out]);
            _ = try g.hostF32(try g.astype(try g.adopt(try own.linear(s, x2, p.trellis, p.suh, p.svh, 1)), .float32), got[c.out..][0..c.out]);
            _ = try g.hostF32(try g.astype(try g.adopt(try own.linear(s, x.flat, m.p.trellis, m.p.suh, m.p.svh, 1)), .float32), got[2 * c.out ..][0..c.out]);
            const d_own = wordsDiffer(got[0..c.out], got[c.out..][0..c.out]);
            const d_flat = wordsDiffer(got[0..c.out], got[2 * c.out ..][0..c.out]);
            const ok = d_own == 0 and d_flat == 0;
            checks += 1;
            failed += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_LING_SERVED one_row_table {s} in={d} out={d} table_bt={d} vs_own={d}/{d} vs_u16_flat={d}/{d} {s}\n", .{ c.name, c.in, c.out, table.bt_set[1], d_own, c.out, d_flat, c.out, if (ok) "PASS" else "FAIL" });
        }
    }
    std.debug.print("TURBOEXL3_LING_SERVED_DONE checks={d} failed={d} {s}\n", .{ checks, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 0), failed);
}

/// A KDA layer's three projections of one input (`root.ling.kdaProj`): q|k|v, f and g at Ling's widths, K6
const kdaproj_outs = [3]u32{ 24576, 8192, 8192 };
/// the decode rows a KDA layer's call takes (a verify's up to 8) and DEC-ONE's widest
const kdaproj_rows = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 16 };

test "turboexl3 gpu ling kda proj: q|k|v, f and g in one launch give each member's linear bytes, after its own prepare and over each member's own prep_sc rows joined, at decode rows 1 to 8 and 16 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    const ks = sdk.kernel_launch;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xe1e1_4096);
    const rnd = prng.random();
    const in: u32 = 4096;
    var m0 = try synthProj(a, &g, rnd, in, kdaproj_outs[0], 96);
    defer m0.lin.deinit(a);
    var m1 = try synthProj(a, &g, rnd, in, kdaproj_outs[1], 96);
    defer m1.lin.deinit(a);
    var m2 = try synthProj(a, &g, rnd, in, kdaproj_outs[2], 96);
    defer m2.lin.deinit(a);
    const P = [3]root.Proj{ m0.p, m1.p, m2.p };
    const sh = try root.ling.kdaproj.shapeOf(P);
    const max_rows: u32 = 16;
    const xs = try a.alloc(f32, max_rows * in);
    defer a.free(xs);
    for (xs) |*v| v.* = bfExact(rnd.floatNorm(f32));
    var want: [3][]f32 = undefined;
    var got: [3][]f32 = undefined;
    for (&want, &got, kdaproj_outs) |*w, *gt, o| {
        w.* = try a.alloc(f32, max_rows * o);
        gt.* = try a.alloc(f32, max_rows * o);
    }
    defer for (want, got) |w, gt| {
        a.free(w);
        a.free(gt);
    };
    var failed: usize = 0;
    var checks: usize = 0;
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dt| {
        var d: [3]*dense.Dense = undefined;
        var saved: [3]struct { prep: u32, bt1: u8 } = undefined;
        for (&d, &saved, kdaproj_outs) |*dm, *sv, o| {
            dm.* = try root.denseCached(.{ .in = in, .out = o, .rate = format.Rate.fromN(96).?, .dec = .mul1, .out_dtype = dt });
            sv.* = .{ .prep = dm.*.dec_prep_rows, .bt1 = dm.*.bt_set[1] };
        }
        defer for (d, saved) |dm, sv| {
            dm.dec_prep_rows = sv.prep;
            dm.bt_set[1] = sv.bt1;
            dm.resetCfgs();
        };
        // the members' forms at one row: served; one row on two rows' launch shape (dec_prep_rows 0, bt_set[1] 2);
        // the prepare in its own launch at one output tile a threadgroup. The call's tiles follow member 0's.
        const arms = [_]struct { name: []const u8, prep: u32, bt1: u8 }{ .{ .name = "served", .prep = 1, .bt1 = 0 }, .{ .name = "wide", .prep = 0, .bt1 = 2 }, .{ .name = "bt1", .prep = 0, .bt1 = 1 } };
        for (arms) |arm| {
            for (d) |dm| {
                dm.dec_prep_rows = arm.prep;
                dm.bt_set[1] = arm.bt1;
                dm.resetCfgs();
            }
            for (kdaproj_rows) |rows| {
                const rm = g.mark();
                defer g.resetTo(rm);
                const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. rows * in]), &.{ ci(rows), ci(in) }, .float32), dt);
                for (P, want, kdaproj_outs) |mp, w, o| {
                    const y = try g.adopt(try root.linear(s, x, mp.trellis, mp.suh, mp.svh, .mul1));
                    _ = try g.hostF32(try g.astype(y, .float32), w[0 .. rows * o]);
                }
                const p = (try root.ling.kdaproj.plan(sh, .mul1, rows, dt)) orelse {
                    std.debug.print("TURBOEXL3_LING_KDA_PROJ x={s} arm={s} rows={d} not served FAIL\n", .{ @tagName(dt), arm.name, rows });
                    failed += 1;
                    continue;
                };
                // after its own prepare
                const ys = try root.ling.kdaProj(s, x, P, .mul1);
                var differing_own: usize = 0;
                for (ys, got, want, kdaproj_outs) |y, gt, w, o| {
                    const ya = try g.adopt(y);
                    try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(o) }, mlx.getShape(ya));
                    _ = try g.hostF32(try g.astype(ya, .float32), gt[0 .. rows * o]);
                    for (gt[0 .. rows * o], w[0 .. rows * o]) |u, v| differing_own += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
                }
                // over each member's own prep_sc rows (its prepare in its own launch), joined as the prepared entry takes them
                var xps: [3]mlx.mlx_array = undefined;
                var xss: [3]mlx.mlx_array = undefined;
                for (d, P, &xps, &xss) |dm, mp, *xpm, *xsm| {
                    const sv_prep = dm.dec_prep_rows;
                    dm.dec_prep_rows = 0;
                    dm.resetCfgs();
                    defer {
                        dm.dec_prep_rows = sv_prep;
                        dm.resetCfgs();
                    }
                    const c = try dm.decCfgs(rows);
                    var po: [3]mlx.mlx_array = undefined;
                    try ks.applyConfig(dm.prep_sc, c.prep, &.{ x, mp.suh }, &po, s);
                    xpm.* = try g.adopt(po[0]);
                    xsm.* = try g.adopt(po[1]);
                    _ = mlx.mlx_array_free(po[2]);
                }
                const jxp = try g.concat(&xps, 0);
                const jxs = try g.concat(&xss, 0);
                const cnt = try g.zeros(&.{ci(root.ling.kdaproj.ncntFor(sh, rows))}, .uint32);
                const yp = try root.ling.kdaProjPrepared(s, jxp, jxs, cnt, rows, P, .mul1, dt);
                var differing_prep: usize = 0;
                for (yp, got, want, kdaproj_outs) |y, gt, w, o| {
                    const ya = try g.adopt(y);
                    _ = try g.hostF32(try g.astype(ya, .float32), gt[0 .. rows * o]);
                    for (gt[0 .. rows * o], w[0 .. rows * o]) |u, v| differing_prep += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
                }
                const ok = differing_own == 0 and differing_prep == 0;
                checks += 1;
                const n_out = rows * (kdaproj_outs[0] + kdaproj_outs[1] + kdaproj_outs[2]);
                std.debug.print("TURBOEXL3_LING_KDA_PROJ x={s} arm={s} rows={d} ks={d},{d},{d} rb={d} bt={d} own_prepare_differing={d}/{d} prepared_differing={d}/{d} {s}\n", .{ @tagName(dt), arm.name, rows, p.ks0, p.ks1, p.ks1, p.rb, p.bt, differing_own, n_out, differing_prep, n_out, if (ok) "PASS" else "FAIL" });
                if (!ok) failed += 1;
            }
        }
    }
    // the refusals, before any launch
    {
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. 17 * 128]), &.{ 17, 128 }, .float32), .bfloat16);
        try testing.expectError(error.BadExl3Shape, root.ling.kdaProj(s, x, P, .mul1));
        const big = try g.zeros(&.{ ci(max_rows + 1), ci(in) }, .bfloat16);
        try testing.expectError(error.Exl3EngineNotImplemented, root.ling.kdaProj(s, big, P, .mul1));
        const narrow = [3]root.Proj{ m0.p, m1.p, .{ .trellis = m2.p.trellis, .suh = m2.p.suh, .svh = m1.p.suh } };
        try testing.expectError(error.BadExl3Shape, root.ling.kdaProj(s, try g.zeros(&.{ 1, ci(in) }, .bfloat16), narrow, .mul1));
        const bad = try g.zeros(&.{ 2, ci(in) }, .float16);
        try testing.expectError(error.BadExl3Shape, root.ling.kdaProjPrepared(s, bad, bad, bad, 1, P, .mul1, .bfloat16));
    }
    std.debug.print("TURBOEXL3_LING_KDA_PROJ_DONE checks={d} {s}\n", .{ checks, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── DEC-ONE's GEMV variants (`Config.dec_variant`, `dense.dec_variants`: t4, pf, both) at Ling-3.1-flash's shapes ──

/// The variants past DEC-ONE's own (`dense.dec_variants` 1 to 3: t4, pf, pft4), each a byte test's arm
const dec_variant_arms = [_]u8{ 1, 2, 3 };

/// A K split of a linear of `in` inputs other than `ks` (the smallest divisor of its input tiles over the four
/// simdgroups that is not `ks`): the red checks' reordering of the same products
fn otherKs(in: u32, ks: u32) u32 {
    const q = (in / 16) / 4;
    var dv: u32 = 1;
    while (dv <= q) : (dv += 1) {
        if (dv != ks and q % dv == 0) return dv;
    }
    return ks;
}

/// A trellis as the loader hands it (the pack's I16 words) from its U16 words
fn i16Trellis(g: *G, s: mlx.mlx_stream, words: []const u16, in: u32, out: u32, n: u32) !mlx.mlx_array {
    const u = try g.hostArray(std.mem.sliceAsBytes(words), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16);
    var v = mlx.mlx_array_new();
    mlx.check(mlx.mlx_view(&v, u, .int16, s)) catch |e| {
        _ = mlx.mlx_array_free(v);
        return e;
    };
    return g.adopt(v);
}

/// A linear's operands as `ref.synthLinear` draws them (the trellis words, then suh, then svh, from the same generator in
/// the same order), without the reference weights it reconstructs on the host, which the variant tests do not read
/// (4096 x 157184 of them at the LM head)
const Drawn = struct {
    trellis: []u16,
    suh: []u16,
    svh: []u16,

    fn deinit(self: Drawn, a: std.mem.Allocator) void {
        a.free(self.trellis);
        a.free(self.suh);
        a.free(self.svh);
    }
};

fn drawLinear(a: std.mem.Allocator, rnd: std.Random, in: u32, out: u32, n: u32) !Drawn {
    const trellis = try a.alloc(u16, @as(usize, in / 16) * (out / 16) * n);
    errdefer a.free(trellis);
    const suh = try a.alloc(u16, in);
    errdefer a.free(suh);
    const svh = try a.alloc(u16, out);
    for (trellis) |*v| v.* = rnd.int(u16);
    const out_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(in)));
    for (suh) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    for (svh) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * out_scale);
    return .{ .trellis = trellis, .suh = suh, .svh = svh };
}

// Every DEC-ONE GEMV variant against the served GEMV, word for word, where it runs: a Dense whose prepare runs in its own
// launch at every row count (`dec_prep_rows` and `dec_prep_out` 0), so each call at rows 1 to 4 takes the GEMV over
// prepared rows: through `linear` (prep_sc, then the GEMV) and over prep_sc's rows with zeroed counters
// (`decLinearPrepared`, KDA_EPILOGUE's o_proj). Every Ling-3.1-flash trunk shape (K6; the dense FFN at K4) at x bf16
// and f32, and the LM head (4096 -> 157184 at the pack's K6) at x bf16: the variants act on the head at a verify's rows,
// where its prepare runs in its own launch. The trellis as I16 words; the operands synthLinear's draws without its
// reference weights (`drawLinear`). Each arm also checks that it ran its variant (`dense.dec0VariantRuns`).
// Red checks: one word flipped in the served output is exactly one differing word; at o_proj's and q|k|v's shapes each
// variant on a Dense at another K split (the same products summed in another order) must differ from the served words.
// Lines: TURBOEXL3_LING_DEC_VARIANT (252: 240 at the trunk shapes, 12 at the head), _RED (8), _DONE.
test "turboexl3 gpu ling dense variants: each DEC-ONE GEMV variant (t4, pf, pft4) gives the served GEMV's words at every Ling-3.1-flash trunk shape and the LM head, rows 1 to 4, x bf16 and f32 (the head bf16), through linear and over prepared rows; red checks: a flipped word, each variant at another K split (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    const ks = sdk.kernel_launch;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = config.get().*;
    defer config.configure(saved) catch unreachable;
    var cfg = saved;
    cfg.dec_variant = 0;
    var failed: usize = 0;
    var checks: usize = 0;
    var reds: usize = 0;
    for (dense_cases) |c| {
        var prng = std.Random.DefaultPrng.init(0xD35A ^ (@as(u64, c.in) << 24) ^ c.out);
        const rnd = prng.random();
        const rate = format.Rate.fromN(c.n).?;
        const lin = try drawLinear(a, rnd, c.in, c.out, c.n);
        defer lin.deinit(a);
        const mark = g.mark();
        defer g.resetTo(mark);
        const trellis = try i16Trellis(&g, s, lin.trellis, c.in, c.out, c.n);
        const suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(c.in)}, .float16);
        const svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(c.out)}, .float16);
        const xs = try a.alloc(f32, 4 * c.in);
        defer a.free(xs);
        for (xs) |*v| v.* = bfExact(rnd.floatNorm(f32));
        const want = try a.alloc(f32, 4 * c.out);
        defer a.free(want);
        const got = try a.alloc(f32, 4 * c.out);
        defer a.free(got);
        // the red checks' shapes: o_proj (KDA o and the MLA output projection) and q|k|v
        const red = std.mem.eql(u8, c.name, "kda.o/mla.dense") or std.mem.eql(u8, c.name, "kda.qkv");
        // the LM head at bf16 x only, the dtype its verify rows take
        const dts: []const mlx.mlx_dtype = if (c.head) &[_]mlx.mlx_dtype{.bfloat16} else &[_]mlx.mlx_dtype{ .bfloat16, .float32 };
        for (dts) |dt| {
            try config.configure(cfg);
            const geo: dense.Shape = .{ .in = c.in, .out = c.out, .rate = rate, .dec = .mul1, .out_dtype = dt };
            var d = try dense.Dense.init(geo);
            defer d.deinit();
            d.dec_prep_rows = 0;
            d.dec_prep_out = 0;
            for (1..5) |ru| {
                const rows: u32 = @intCast(ru);
                const rm = g.mark();
                defer g.resetTo(rm);
                const n = rows * c.out;
                const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. rows * c.in]), &.{ ci(rows), ci(c.in) }, .float32), dt);
                // prep_sc's rows, once; each GEMV over them reads counters zeroed for it
                const dc = try d.decCfgs(rows);
                var po: [3]mlx.mlx_array = undefined;
                try ks.applyConfig(d.prep_sc, dc.prep, &.{ x, suh }, &po, s);
                const xp = try g.adopt(po[0]);
                const xsum = try g.adopt(po[1]);
                _ = mlx.mlx_array_free(po[2]);
                const ncnt: c_int = @intCast(try d.decCounters(rows));
                // the served GEMV through linear, and over the prepared rows
                try config.configure(cfg);
                _ = try g.hostF32(try g.astype(try g.adopt(try d.linear(s, x, trellis, suh, svh, rows)), .float32), want[0..n]);
                _ = try g.hostF32(try g.astype(try g.adopt(try d.decLinearPrepared(s, xp, xsum, try g.zeros(&.{ncnt}, .uint32), trellis, svh, rows)), .float32), got[0..n]);
                const d_served = bytesDifferF32(got[0..n], want[0..n]);
                for (dec_variant_arms) |v| {
                    var vc = cfg;
                    vc.dec_variant = v;
                    try config.configure(vc);
                    const runs0 = dense.dec0VariantRuns(v);
                    _ = try g.hostF32(try g.astype(try g.adopt(try d.linear(s, x, trellis, suh, svh, rows)), .float32), got[0..n]);
                    const d_lin = bytesDifferF32(got[0..n], want[0..n]);
                    _ = try g.hostF32(try g.astype(try g.adopt(try d.decLinearPrepared(s, xp, xsum, try g.zeros(&.{ncnt}, .uint32), trellis, svh, rows)), .float32), got[0..n]);
                    const d_prep = bytesDifferF32(got[0..n], want[0..n]);
                    const ran = dense.dec0VariantRuns(v) - runs0;
                    try config.configure(cfg);
                    const ok = d_served == 0 and d_lin == 0 and d_prep == 0 and ran == 2;
                    checks += 1;
                    std.debug.print("TURBOEXL3_LING_DEC_VARIANT shape={s} in={d} out={d} n={d} rows={d} x={s} ks={d} bt={d} variant={s} linear_differing={d}/{d} prepared_differing={d}/{d} served_prepared_differing={d}/{d} variant_calls={d}/2 {s}\n", .{ c.name, c.in, c.out, c.n, rows, @tagName(dt), d.ks_split, d.btDec(rows, dense.Dense.rbFor(rows)), dense.dec_variants[v].name, d_lin, n, d_prep, n, d_served, n, ran, if (ok) "PASS" else "FAIL" });
                    if (!ok) failed += 1;
                }
                // red: one flipped word of the served output is one differing word
                if (red and rows == 3 and dt == .bfloat16) {
                    @memcpy(got[0..n], want[0..n]);
                    got[n / 2] = @bitCast(@as(u32, @bitCast(got[n / 2])) ^ 1);
                    const d_flip = bytesDifferF32(got[0..n], want[0..n]);
                    const ok_flip = d_flip == 1;
                    reds += 1;
                    std.debug.print("TURBOEXL3_LING_DEC_VARIANT_RED check=flipped_word shape={s} rows={d} differing={d}/{d} expect=1 {s}\n", .{ c.name, rows, d_flip, n, if (ok_flip) "PASS" else "FAIL" });
                    if (!ok_flip) failed += 1;
                    // red: each variant at another K split must differ from the served words (a reordering is caught)
                    var dk = try dense.Dense.init(geo);
                    defer dk.deinit();
                    dk.dec_prep_rows = 0;
                    dk.dec_prep_out = 0;
                    const ks_other = otherKs(c.in, d.ks_split);
                    dk.ks_split = ks_other;
                    for (dec_variant_arms) |v| {
                        var vc = cfg;
                        vc.dec_variant = v;
                        try config.configure(vc);
                        _ = try g.hostF32(try g.astype(try g.adopt(try dk.linear(s, x, trellis, suh, svh, rows)), .float32), got[0..n]);
                        try config.configure(cfg);
                        const d_ks = bytesDifferF32(got[0..n], want[0..n]);
                        const ok_ks = ks_other != d.ks_split and d_ks > 0;
                        reds += 1;
                        std.debug.print("TURBOEXL3_LING_DEC_VARIANT_RED check=other_ks shape={s} rows={d} ks={d}->{d} variant={s} differing={d}/{d} expect=nonzero {s}\n", .{ c.name, rows, d.ks_split, ks_other, dense.dec_variants[v].name, d_ks, n, if (ok_ks) "PASS" else "FAIL" });
                        if (!ok_ks) failed += 1;
                    }
                }
            }
        }
    }
    std.debug.print("TURBOEXL3_LING_DEC_VARIANT_DONE checks={d} red={d} {s}\n", .{ checks, reds, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 252), checks);
    try testing.expectEqual(@as(usize, 8), reds);
    try testing.expectEqual(@as(usize, 0), failed);
}

/// TURBOEXL3_LING_TIME's arm for DEC-ONE's GEMV variants (`dense.dec_variants`), guidance only: each trunk shape at rows
/// 1 to 4 on a Dense made under the outside plugin's launch shapes (`Config.dec_one_row_table`), the served GEMV and each
/// variant (`linear`: prep_sc, then the GEMV): min and median us a call (8 calls an evaluation), the weights' GB/s, the
/// variant's min over the served GEMV's, and the calls that ran the variant (0 where the call's prepare runs inside its
/// launch: no variant acts there). Its own draw. Lines: TURBOEXL3_LING_TIME dec_variant.
fn timeDecVariants(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, base: config.Config) !void {
    defer config.configure(base) catch {};
    var tc = base;
    tc.dec_one_row_table = true;
    tc.dec_variant = 0;
    var prng = std.Random.DefaultPrng.init(0xD35_71AE);
    const rnd = prng.random();
    const Ctx = struct {
        s: mlx.mlx_stream,
        x: mlx.mlx_array,
        p: root.Proj,
        d: *dense.Dense,
        rows: usize,
        pub fn call(cx: @This()) !mlx.mlx_array {
            return cx.d.linear(cx.s, cx.x, cx.p.trellis, cx.p.suh, cx.p.svh, cx.rows);
        }
    };
    for (dense_cases) |c| {
        const mark = g.mark();
        defer g.resetTo(mark);
        defer _ = mlx.mlx_clear_cache();
        const p = try ds_test.randProj(a, g, rnd, null, c.in, c.out, c.n);
        try config.configure(tc);
        var d = try dense.Dense.init(.{ .in = c.in, .out = c.out, .rate = format.Rate.fromN(c.n).?, .dec = .mul1, .out_dtype = .bfloat16 });
        defer d.deinit();
        const wbytes = @as(f64, @floatFromInt(c.in)) * @as(f64, @floatFromInt(c.out)) * @as(f64, @floatFromInt(c.n)) / 128.0;
        for ([_]u32{ 1, 2, 3, 4 }) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const ctx: Ctx = .{ .s = s, .x = try ds_test.randBf16(a, g, rnd, &.{ ci(rows), ci(c.in) }), .p = p, .d = &d, .rows = rows };
            var served_min: f64 = 0;
            for (0..dense.dec_variants.len) |vi| {
                const v: u8 = @intCast(vi);
                var vc = tc;
                vc.dec_variant = v;
                try config.configure(vc);
                const runs0 = dense.dec0VariantRuns(v);
                const st = try ds_test.timeIt(g, ctx, 8);
                const calls = dense.dec0VariantRuns(v) - runs0;
                if (v == 0) served_min = st.min;
                std.debug.print("TURBOEXL3_LING_TIME dec_variant {s} {d}x{d} n={d} rows={d} variant={s} min_us={d:.1} med_us={d:.1} GBps={d:.0} of_served={d:.3} variant_calls={d}\n", .{ c.name, c.in, c.out, c.n, rows, if (v == 0) "served" else dense.dec_variants[v].name, st.min, st.med, wbytes / (st.min * 1e3), if (served_min > 0) st.min / served_min else 0, calls });
            }
        }
    }
}


// ── a prompt call's rows at any call width (a layer-major prompt pass runs the trunk linears over the whole prompt) ──

/// The MTP layer's input projection (the 2.60 pack's `eh_proj`, K4): with `dense_cases` but the head, every EXL3 linear
/// the pack's headers serve at prompt rows
const eh_proj: DenseCase = .{ .name = "mtp.eh_proj", .in = 8192, .out = 4096, .n = 64 };
/// The standard prompt's rows and the ends of the chunk loop's spans over them: 4,096-row chunks, the SSM snapshot
/// backoff's span, then the last 31 rows; without checkpoints its last two spans end at 18,379 and 18,380
const prompt_rows: u32 = 18380;
const prompt_ends = [_]u32{ 4096, 8192, 12288, 16384, 18349, 18380 };
const prompt_ends_nocp = [_]u32{ 18379, 18380 };
/// Calls over the prompt's first rows: the decode GEMVs (1, 16), the register-path GEMM (17, 24, 32), and MLX's matmul on
/// both sides of each threshold of its dispatch at these shapes (split K up to max(M, N) 1365 at K 4096; bk 64 below
/// M + N = K at K 8192 and 10240)
const prompt_firsts = [_]u32{ 1, 16, 17, 24, 32, 33, 64, 100, 512, 1024, 1365, 1366, 1965, 1995, 2048, 4095, 4096, 4097, 6143, 6144, 8192 };

const PromptPath = enum { decode, register, matmul, split_k };

/// MLX's matmul sums an M x N x K product's K in partitions (its NAX dispatch's split K) at K >= 3 max(M, N), or at
/// max(M, N) <= 1024 with K > 2 max(M, N); its regular kernel sums K in one pass of 32-deep steps whatever its tiles.
fn mlxSplitK(m: u32, n: u32, k: u32) bool {
    const mx = @max(m, n);
    return k >= 3 * mx or (mx <= 1024 and k > 2 * mx);
}

/// The path `root.linear` runs a call of `m` rows of `c` on (dense.zig's: the decode GEMVs, the register-path GEMM, then
/// MLX's matmul against the dequantized trellis)
fn promptPath(d: *const dense.Dense, c: DenseCase, m: u32) PromptPath {
    if (m <= decode.Kernels.max_rows) return .decode;
    if (m <= d.small_rows and !d.fused_prefill) return .register;
    return if (mlxSplitK(m, c.out, c.in)) .split_k else .matmul;
}

/// Elements of `x` and `y` (one shape, a 2-byte dtype) whose bits differ, counted on the GPU
fn wordDiff(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, y: mlx.mlx_array) !usize {
    if (!std.mem.eql(c_int, mlx.getShape(x), mlx.getShape(y)) or mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(y)) return error.ShapeMismatch;
    var xv = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_view(&xv, x, .uint16, s));
    xv = try g.adopt(xv);
    var yv = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_view(&yv, y, .uint16, s));
    yv = try g.adopt(yv);
    var eq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_equal(&eq, xv, yv, s));
    eq = try g.adopt(eq);
    var same_a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum(&same_a, try g.astype(eq, .int32), false, s));
    same_a = try g.adopt(same_a);
    try g.evalAll(&.{same_a});
    var same: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&same, same_a));
    return mlx.mlx_array_size(x) - @as(usize, @intCast(same));
}

/// One prompt linear's inputs and the whole-prompt call's path
const PromptLinear = struct {
    g: *G,
    s: mlx.mlx_stream,
    c: DenseCase,
    d: *const dense.Dense,
    x: mlx.mlx_array,
    trellis: mlx.mlx_array,
    suh: mlx.mlx_array,
    svh: mlx.mlx_array,

    /// `root.linear` over rows [lo, hi) of x
    fn call(p: PromptLinear, lo: u32, hi: u32) !mlx.mlx_array {
        const xs = try p.g.slice(p.x, &.{ ci(lo), 0 }, &.{ ci(hi), ci(p.c.in) }, &.{ 1, 1 });
        return p.g.adopt(try root.linear(p.s, xs, p.trellis, p.suh, p.svh, .mul1));
    }

    /// `got` (a call over rows [lo, hi) of x) against those rows of `whole` (a call over x's first `whole_rows`), word
    /// for word: its line, and false where both calls run MLX's regular matmul and a word differs
    fn check(p: PromptLinear, got: mlx.mlx_array, whole: mlx.mlx_array, whole_rows: u32, lo: u32, hi: u32) !bool {
        const want = try p.g.slice(whole, &.{ ci(lo), 0 }, &.{ ci(hi), ci(p.c.out) }, &.{ 1, 1 });
        const diff = try wordDiff(p.g, p.s, got, want);
        const path = promptPath(p.d, p.c, hi - lo);
        const asserted = path == .matmul and promptPath(p.d, p.c, whole_rows) == .matmul;
        const status: []const u8 = if (asserted) (if (diff == 0) "PASS" else "FAIL") else if (diff == 0) "SAME" else "DIFFERS";
        std.debug.print("TURBOEXL3_LING_ROWINV shape={s} K={d} N={d} n={d} rows=[{d},{d}) m={d} whole={d} path={s} differing={d}/{d} {s}\n", .{ p.c.name, p.c.in, p.c.out, p.c.n, lo, hi, hi - lo, whole_rows, @tagName(path), diff, (hi - lo) * p.c.out, status });
        return !asserted or diff == 0;
    }
};

// A layer-major prompt pass (an outside plugin's: each layer over the whole prompt, the chunk loop's spans kept where a
// row's bytes need them) can run a trunk linear over the whole prompt only where each row's bytes do not depend on the
// call's row count. At every EXL3 linear the 2.60 pack serves at prompt rows, bf16 x ~ N(0, 1): one call over the
// standard prompt's 18,380 rows against the chunk loop's spans over them (with and without its checkpoints) and calls
// over its first rows at widths across the paths' and MLX's dispatch thresholds, and an 8,192-row call against it and
// against its two 4,096-row halves, word for word. Asserted where both calls run MLX's regular matmul; the decode GEMVs,
// the register-path GEMM and MLX's split-K matmul sum in other orders, and their lines say whether the words still match.
test "turboexl3 gpu ling prompt linears: a prompt call's rows of root.linear have the whole-prompt call's bytes at every EXL3 trunk shape the 2.60 pack serves at prompt rows, where both run MLX's regular matmul (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var cases: [dense_cases.len]DenseCase = undefined;
    var nc: usize = 0;
    for (dense_cases) |c| if (!c.head) {
        cases[nc] = c;
        nc += 1;
    };
    cases[nc] = eh_proj;
    nc += 1;
    const nokey = mlx.mlx_array_new();
    var failed: usize = 0;
    var lines: usize = 0;
    for (cases[0..nc]) |c| {
        defer _ = mlx.mlx_clear_cache();
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x20A91B ^ (@as(u64, c.in) << 24) ^ (@as(u64, c.out) << 4) ^ c.n);
        const rnd = prng.random();
        const tbuf = try a.alloc(u16, (c.in / 16) * (c.out / 16) * c.n);
        defer a.free(tbuf);
        for (tbuf) |*v| v.* = rnd.int(u16);
        const sbuf = try a.alloc(u16, c.in + c.out);
        defer a.free(sbuf);
        const out_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.in)));
        for (sbuf[0..c.in]) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
        for (sbuf[c.in..]) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * out_scale);
        var xr = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&xr, &[_]c_int{ ci(prompt_rows), ci(c.in) }, 2, .bfloat16, 0, 1, nokey, s));
        const p: PromptLinear = .{
            .g = &g,
            .s = s,
            .c = c,
            .d = try root.denseCached(.{ .in = c.in, .out = c.out, .rate = format.Rate.fromN(c.n).?, .dec = .mul1, .out_dtype = .bfloat16 }),
            .x = try g.adopt(xr),
            .trellis = try g.hostArray(std.mem.sliceAsBytes(tbuf), &.{ ci(c.in / 16), ci(c.out / 16), ci(c.n) }, .int16),
            .suh = try g.hostArray(std.mem.sliceAsBytes(sbuf[0..c.in]), &.{ci(c.in)}, .float16),
            .svh = try g.hostArray(std.mem.sliceAsBytes(sbuf[c.in..]), &.{ci(c.out)}, .float16),
        };
        const whole = try p.call(0, prompt_rows);
        try g.evalAll(&.{ p.x, whole });
        var checks: [prompt_ends.len + prompt_ends_nocp.len + prompt_firsts.len + 3]bool = undefined;
        var n_checks: usize = 0;
        // the chunk loop's spans, with its checkpoints, then the last two without
        var lo: u32 = 0;
        for (prompt_ends) |hi| {
            const m = g.mark();
            defer g.resetTo(m);
            checks[n_checks] = try p.check(try p.call(lo, hi), whole, prompt_rows, lo, hi);
            n_checks += 1;
            lo = hi;
        }
        lo = prompt_ends[3];
        for (prompt_ends_nocp) |hi| {
            const m = g.mark();
            defer g.resetTo(m);
            checks[n_checks] = try p.check(try p.call(lo, hi), whole, prompt_rows, lo, hi);
            n_checks += 1;
            lo = hi;
        }
        // calls over the prompt's first rows
        for (prompt_firsts) |hi| {
            const m = g.mark();
            defer g.resetTo(m);
            checks[n_checks] = try p.check(try p.call(0, hi), whole, prompt_rows, 0, hi);
            n_checks += 1;
        }
        // an 8,192-row call against the whole prompt's, and its two halves against it
        {
            const m = g.mark();
            defer g.resetTo(m);
            const half = try p.call(0, 8192);
            try g.evalAll(&.{half});
            checks[n_checks] = try p.check(half, whole, prompt_rows, 0, 8192);
            n_checks += 1;
            for ([_][2]u32{ .{ 0, 4096 }, .{ 4096, 8192 } }) |r| {
                const m2 = g.mark();
                defer g.resetTo(m2);
                checks[n_checks] = try p.check(try p.call(r[0], r[1]), half, 8192, r[0], r[1]);
                n_checks += 1;
            }
        }
        for (checks[0..n_checks]) |ok| failed += @intFromBool(!ok);
        lines += n_checks;
    }
    std.debug.print("TURBOEXL3_LING_ROWINV_DONE shapes={d} lines={d} failed={d} {s}\n", .{ nc, lines, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── timing (TURBOEXL3_LING_TIME=1): guidance for the serving plan, never a speed verdict ──

const ds_test = @import("../deepseek_v41/deepseek_test.zig");
const mla_latent = @import("mla_latent.zig");
const mla_latent_test = @import("mla_latent_test.zig");
const mla_flash = @import("mla_flash.zig");
const router_test = @import("router_test.zig");

/// A serial decode step and the verify widths of 2 to 4 rows, the engine's prompt chunk and a layer-major call
const time_rows = [_]u32{ 1, 2, 3, 4, 4096, 16384 };

/// The outside plugin's prompt call: a wide call's pairs sorted by slot, at most this many a `moeLing` call at top-1
/// (turboexl3_ling's `prefill_rows`)
const plugin_call_pairs: u32 = 4096;

/// Distinct values among `xs` (each below 2^16)
fn distinctOf(xs: []const u32) usize {
    var seen = std.StaticBitSet(1 << 16).empty;
    for (xs) |v| seen.set(v);
    return seen.count();
}

/// Each of `rows` rows' `k` distinct slots drawn from `pool`, written as banked ids (bank 0) to `ids` and each pair's
/// row to `prow`
fn drawDecodePairs(rnd: std.Random, rows: u32, k: u32, pool: []const u32, ids: []u32, prow: []u32) void {
    for (0..rows) |r| {
        var used: u64 = 0;
        for (0..k) |j| {
            var i = rnd.uintLessThan(usize, pool.len);
            while ((used >> @as(u6, @intCast(i))) & 1 != 0) i = rnd.uintLessThan(usize, pool.len);
            used |= @as(u64, 1) << @as(u6, @intCast(i));
            ids[r * k + j] = pool[i];
            prow[r * k + j] = @intCast(r);
        }
    }
}

// Ling-3.1-flash's projections and routed calls timed on the device (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; it
// asserts nothing): guidance for the serving plan, never a speed verdict (the standard test through the outside plugin
// decides speed). Random trellises (deepseek_test's draw: every pattern a codeword) at the packs' rates:
// - `root.linear` at every trunk shape (`dense_cases`: K6, the dense FFN K4) at rows 1 to 4, 4096 and 16384, the head at
//   rows 1 to 4; the weights' GB/s at decode rows (the 8 calls of an evaluation read one weight, so a shape that fits the
//   GPU's cache reads part of it from there), TFLOPS at prompt rows;
// - `root.moeLing` over a stacked bank of 512 experts at top-8, rows 1 to 4 and 4096 (`route`'s draw, sigma 1.5), and at
//   top-1 with score 1 over the first 4096 of a call's pairs sorted by slot, the call of a 4096-token chunk and of a
//   16384-token one (the outside plugin's prompt calls: about 64 and about 16 experts a call);
// - `root.moeLingBanked` over one bank of 200 slot rows (the 2.60 pack's decode rows) at the pairs of 1, 2 and 3 decode
//   rows, each row's 8 slots drawn from 24 of them (8, about 13 and about 17 distinct slots), banked group 4 with the M = 3
//   texts (the plugin's configuration); the distinct records' GB/s;
// - first, `mlaLatentDecode` (the MTP head's latent attention, `mla_latent.zig`) at the standard request's head cache
//   (19,404 keys of bf16 latents and rope keys, the first rows of a wider buffer; kv_b bf16 [16384, 512]; the query halves
//   as views of the transposed heads, as the plugin passes them) at rows 1 to 4, on launch 2 as it is and on its
//   variant (`mla_latent.Attn.stage`, launch 2's words): us a row, the variant's min over
//   launch 2's, the attention's TFLOPS (64 heads x (576 + 512) multiply-adds a key and row);
// - second, the MLA prompt attention at FL-1's served calls (Ling's 64 heads in bf16, 4096 query rows over 12,288 and
//   16,384 keys, 1965 over 18,349, 31 and 17 over 18,380; caches of 18,432 tokens): `mlaFlashPrefill` over views of
//   the served cache's expanded keys and values against `mlaFlashPrefillRope1` over views of the rope-once cache's
//   (kv_b's rows a token, the roped key once a token): the min over the served one's.
// Each line: min and median us per call (deepseek_test's timeIt). TURBOEXL3_LING_N (default 40,32) narrows the rates.
test "turboexl3 gpu ling timing: every trunk linear, moeLing and moeLingBanked at Ling-3.1-flash's shapes (TURBOEXL3_LING_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_LING_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = config.get().*;
    defer config.configure(saved) catch {};
    var cfg = saved;
    cfg.banked_group = 4;
    cfg.banked_m3 = true;
    try config.configure(cfg);
    {
        const mark = g.mark();
        defer g.resetTo(mark);
        defer _ = mlx.mlx_clear_cache();
        // a variant MLX refuses to launch prints its error line and the arms go on
        mla_latent_test.reportMlxErrors(true);
        defer mla_latent_test.reportMlxErrors(false);
        // its own draw, so the arms after it draw as before
        var mla_prng = std.Random.DefaultPrng.init(0x4C47_4BC0);
        const mr = mla_prng.random();
        const keys: u32 = 19404;
        const heads: u32 = 64;
        const lat = try g.slice(try ds_test.randBf16(a, &g, mr, &.{ 1, 1, ci(keys + 64), 512 }), &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(keys), 512 }, &.{ 1, 1, 1, 1 });
        const rot = try g.slice(try ds_test.randBf16(a, &g, mr, &.{ 1, 1, ci(keys + 64), 64 }), &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(keys), 64 }, &.{ 1, 1, 1, 1 });
        const ws: f32 = 1.0 / @sqrt(512.0);
        const kv_b = try g.astype(try g.mul(try g.astype(try ds_test.randBf16(a, &g, mr, &.{ ci(heads * 256), 512 }), .float32), try g.hostArray(std.mem.asBytes(&ws), &.{1}, .float32)), .bfloat16);
        try g.evalAll(&.{ lat, rot, kv_b });
        const MlaCtx = struct {
            s: mlx.mlx_stream,
            qn: mlx.mlx_array,
            qr: mlx.mlx_array,
            lat: mlx.mlx_array,
            rot: mlx.mlx_array,
            w: mlx.mlx_array,
            attn: mla_latent.Attn,
            pub fn call(cx: @This()) !mlx.mlx_array {
                return mla_latent.decodeLatentWith(cx.s, cx.qn, cx.qr, cx.lat, cx.rot, cx.w, 1.0 / @sqrt(192.0), cx.attn);
            }
        };
        const flops_row = 2.0 * @as(f64, @floatFromInt(keys)) * @as(f64, @floatFromInt(heads)) * @as(f64, @floatFromInt(mla_latent.dl + mla_latent.dr + mla_latent.dl));
        for ([_]u32{ 1, 2, 3, 4 }) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const qt = try g.transposeAxes(try ds_test.randBf16(a, &g, mr, &.{ 1, ci(rows), ci(heads), 192 }), &.{ 0, 2, 1, 3 });
            const qn = try g.slice(qt, &.{ 0, 0, 0, 0 }, &.{ 1, ci(heads), ci(rows), 128 }, &.{ 1, 1, 1, 1 });
            const qr = try g.slice(qt, &.{ 0, 0, 0, 128 }, &.{ 1, ci(heads), ci(rows), 192 }, &.{ 1, 1, 1, 1 });
            var base_min: f64 = 0;
            for ([_]mla_latent.Attn{ .base, .stage }) |v| {
                const ctx: MlaCtx = .{ .s = s, .qn = qn, .qr = qr, .lat = lat, .rot = rot, .w = kv_b, .attn = v };
                const st = ds_test.timeIt(&g, ctx, 8) catch |e| {
                    std.debug.print("TURBOEXL3_LING_TIME mla_latent keys={d} rows={d} arm={s}: {s}\n", .{ keys, rows, @tagName(v), @errorName(e) });
                    continue;
                };
                if (v == .base) base_min = st.min;
                const r: f64 = @floatFromInt(rows);
                std.debug.print("TURBOEXL3_LING_TIME mla_latent keys={d} rows={d} arm={s} min_us={d:.1} med_us={d:.1} us_per_row={d:.1} of_base={d:.3} tflops={d:.2}\n", .{ keys, rows, @tagName(v), st.min, st.med, st.min / r, if (base_min > 0) st.min / base_min else 0, r * flops_row / (st.min * 1e6) });
            }
        }
    }
    {
        const mark = g.mark();
        defer g.resetTo(mark);
        defer _ = mlx.mlx_clear_cache();
        mla_latent_test.reportMlxErrors(true);
        defer mla_latent_test.reportMlxErrors(false);
        const r1_saved = config.get().*;
        defer config.configure(r1_saved) catch {};
        var r1_cfg = r1_saved;
        r1_cfg.mla_rope_once = true;
        try config.configure(r1_cfg);
        // its own draw, so the arms after it draw as before
        var r1_prng = std.Random.DefaultPrng.init(0x4D1A_201E);
        const r1r = r1_prng.random();
        const heads: u32 = 64;
        const cap: u32 = 18432;
        // the served cache's buffers (the expanded key, the values) and the rope-once cache's (kv_b's rows a token: each
        // head's nope keys then its values; the roped key once a token)
        const kbuf = try ds_test.randBf16(a, &g, r1r, &.{ 1, ci(heads), ci(cap), 192 });
        const vbuf = try ds_test.randBf16(a, &g, r1r, &.{ 1, ci(heads), ci(cap), 128 });
        const rows_buf = try ds_test.randBf16(a, &g, r1r, &.{ 1, ci(cap), ci(heads), 256 });
        const rope_buf = try ds_test.randBf16(a, &g, r1r, &.{ 1, 1, ci(cap), 64 });
        try g.evalAll(&.{ kbuf, vbuf, rows_buf, rope_buf });
        const rows_heads = try g.transposeAxes(rows_buf, &.{ 0, 2, 1, 3 });
        const FlashCtx = struct {
            s: mlx.mlx_stream,
            q: mlx.mlx_array,
            k: mlx.mlx_array,
            v: mlx.mlx_array,
            pub fn call(cx: @This()) !mlx.mlx_array {
                return mla_flash.mlaFlashPrefill(cx.s, cx.q, cx.k, cx.v, 1.0 / @sqrt(192.0));
            }
        };
        const Rope1Ctx = struct {
            s: mlx.mlx_stream,
            q: mlx.mlx_array,
            k: mlx.mlx_array,
            kr: mlx.mlx_array,
            v: mlx.mlx_array,
            pub fn call(cx: @This()) !mlx.mlx_array {
                return mla_flash.mlaFlashPrefillRope1(cx.s, cx.q, cx.k, cx.kr, cx.v, 1.0 / @sqrt(192.0));
            }
        };
        for ([_][2]u32{ .{ 4096, 12288 }, .{ 4096, 16384 }, .{ 1965, 18349 }, .{ 31, 18380 }, .{ 17, 18380 } }) |qk_rows| {
            const ql = qk_rows[0];
            const kl = qk_rows[1];
            const rm = g.mark();
            defer g.resetTo(rm);
            const q = try ds_test.randBf16(a, &g, r1r, &.{ 1, ci(heads), ci(ql), 192 });
            const k = try g.slice(kbuf, &.{ 0, 0, 0, 0 }, &.{ 1, ci(heads), ci(kl), 192 }, &.{ 1, 1, 1, 1 });
            const v = try g.slice(vbuf, &.{ 0, 0, 0, 0 }, &.{ 1, ci(heads), ci(kl), 128 }, &.{ 1, 1, 1, 1 });
            const kn = try g.slice(rows_heads, &.{ 0, 0, 0, 0 }, &.{ 1, ci(heads), ci(kl), 128 }, &.{ 1, 1, 1, 1 });
            const vr = try g.slice(rows_heads, &.{ 0, 0, 0, 128 }, &.{ 1, ci(heads), ci(kl), 256 }, &.{ 1, 1, 1, 1 });
            const kr = try g.slice(rope_buf, &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(kl), 64 }, &.{ 1, 1, 1, 1 });
            try g.evalAll(&.{ q, k, v, kn, vr, kr });
            const served = ds_test.timeIt(&g, FlashCtx{ .s = s, .q = q, .k = k, .v = v }, 8) catch |e| {
                std.debug.print("TURBOEXL3_LING_TIME mla_flash ql={d} kl={d} arm=served: {s}\n", .{ ql, kl, @errorName(e) });
                continue;
            };
            std.debug.print("TURBOEXL3_LING_TIME mla_flash ql={d} kl={d} arm=served min_us={d:.1} med_us={d:.1}\n", .{ ql, kl, served.min, served.med });
            const st = ds_test.timeIt(&g, Rope1Ctx{ .s = s, .q = q, .k = kn, .kr = kr, .v = vr }, 8) catch |e| {
                std.debug.print("TURBOEXL3_LING_TIME mla_flash ql={d} kl={d} arm=rope_once: {s}\n", .{ ql, kl, @errorName(e) });
                continue;
            };
            std.debug.print("TURBOEXL3_LING_TIME mla_flash ql={d} kl={d} arm=rope_once min_us={d:.1} med_us={d:.1} of_served={d:.3}\n", .{ ql, kl, st.min, st.med, st.min / served.min });
        }
    }
    var prng = std.Random.DefaultPrng.init(0x11A671);
    const rnd = prng.random();
    for (dense_cases) |c| {
        const mark = g.mark();
        defer g.resetTo(mark);
        defer _ = mlx.mlx_clear_cache();
        const p = try ds_test.randProj(a, &g, rnd, null, c.in, c.out, c.n);
        const wbytes = @as(f64, @floatFromInt(c.in)) * @as(f64, @floatFromInt(c.out)) * @as(f64, @floatFromInt(c.n)) / 128.0;
        for (time_rows) |rows| {
            if (c.head and rows > 4) continue;
            const rm = g.mark();
            defer g.resetTo(rm);
            const Ctx = struct {
                s: mlx.mlx_stream,
                x: mlx.mlx_array,
                p: root.Proj,
                pub fn call(cx: @This()) !mlx.mlx_array {
                    return root.linear(cx.s, cx.x, cx.p.trellis, cx.p.suh, cx.p.svh, .mul1);
                }
            };
            const ctx: Ctx = .{ .s = s, .x = try ds_test.randBf16(a, &g, rnd, &.{ ci(rows), ci(c.in) }), .p = p };
            const st = try ds_test.timeIt(&g, ctx, if (rows <= 16) 8 else 2);
            const flops = 2.0 * @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(c.in)) * @as(f64, @floatFromInt(c.out));
            std.debug.print("TURBOEXL3_LING_TIME linear {s} {d}x{d} n={d} rows={d} min_us={d:.1} med_us={d:.1} GBps={d:.0} tflops={d:.2}\n", .{ c.name, c.in, c.out, c.n, rows, st.min, st.med, wbytes / (st.min * 1e3), flops / (st.min * 1e6) });
        }
    }
    try timeDecVariants(a, &g, s, cfg);
    try router_test.timeRouterRows(a, &g, s);
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_LING_N", &.{ 40, 32 }, &nbuf);
    for (rates) |n| {
        const mark = g.mark();
        defer g.resetTo(mark);
        defer _ = mlx.mlx_clear_cache();
        // one expert's three records (in * out * n / 128 bytes each)
        const record = 3.0 * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter)) * @as(f64, @floatFromInt(n)) / 128.0;
        const expert_flops = 2.0 * 3.0 * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter));
        const bank: root.Bank = .{
            .gate = try ds_test.randProj(a, &g, rnd, experts, hidden, inter, n),
            .up = try ds_test.randProj(a, &g, rnd, experts, hidden, inter, n),
            .down = try ds_test.randProj(a, &g, rnd, experts, inter, hidden, n),
        };
        const MoeCtx = struct {
            s: mlx.mlx_stream,
            x: mlx.mlx_array,
            bank: root.Bank,
            inds: mlx.mlx_array,
            sc: mlx.mlx_array,
            pub fn call(cx: @This()) !mlx.mlx_array {
                return root.moeLing(cx.s, cx.x, cx.bank, cx.inds, cx.sc, .mul1, 0);
            }
        };
        for ([_]u32{ 1, 2, 3, 4, 4096 }) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const slots = try a.alloc(u32, rows * top_k);
            defer a.free(slots);
            const scores = try a.alloc(f32, rows * top_k);
            defer a.free(scores);
            try route(a, rnd, rows, experts, top_k, 1.5, slots, scores);
            const inds = try a.alloc(i32, rows * top_k);
            defer a.free(inds);
            for (slots, inds) |sl, *d| d.* = @intCast(sl);
            const ctx: MoeCtx = .{
                .s = s,
                .x = try g.astype(try ds_test.randBf16(a, &g, rnd, &.{ 1, ci(rows), ci(hidden) }), .float32),
                .bank = bank,
                .inds = try g.hostArray(std.mem.sliceAsBytes(inds), &.{ 1, ci(rows), ci(top_k) }, .int32),
                .sc = try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, ci(rows), ci(top_k) }, .float32),
            };
            const st = try ds_test.timeIt(&g, ctx, if (rows <= 16) 8 else 2);
            const distinct = distinctOf(slots);
            std.debug.print("TURBOEXL3_LING_TIME moeLing n={d} topk=8 rows={d} distinct={d} min_us={d:.1} med_us={d:.1} GBps={d:.0} tflops={d:.2}\n", .{ n, rows, distinct, st.min, st.med, @as(f64, @floatFromInt(distinct)) * record / (st.min * 1e3), @as(f64, @floatFromInt(rows * top_k)) * expert_flops / (st.min * 1e6) });
        }
        for ([_]u32{ 4096, 16384 }) |tokens| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const slots = try a.alloc(u32, tokens * top_k);
            defer a.free(slots);
            const scores = try a.alloc(f32, tokens * top_k);
            defer a.free(scores);
            try route(a, rnd, tokens, experts, top_k, 1.5, slots, scores);
            // the call's pairs by slot (a counting sort), its first plugin_call_pairs
            var starts: [experts + 1]u32 = @splat(0);
            for (slots) |sl| starts[sl + 1] += 1;
            for (1..starts.len) |i| starts[i] += starts[i - 1];
            const sorted = try a.alloc(i32, slots.len);
            defer a.free(sorted);
            for (slots) |sl| {
                sorted[starts[sl]] = @intCast(sl);
                starts[sl] += 1;
            }
            const call_slots = sorted[0..plugin_call_pairs];
            const ones: [plugin_call_pairs]f32 = @splat(1.0);
            var cs: [plugin_call_pairs]u32 = undefined;
            for (call_slots, &cs) |sl, *d| d.* = @intCast(sl);
            const ctx: MoeCtx = .{
                .s = s,
                .x = try g.astype(try ds_test.randBf16(a, &g, rnd, &.{ 1, ci(plugin_call_pairs), ci(hidden) }), .float32),
                .bank = bank,
                .inds = try g.hostArray(std.mem.sliceAsBytes(call_slots), &.{ 1, ci(plugin_call_pairs), 1 }, .int32),
                .sc = try g.hostArray(std.mem.sliceAsBytes(ones[0..]), &.{ 1, ci(plugin_call_pairs), 1 }, .float32),
            };
            const st = try ds_test.timeIt(&g, ctx, 2);
            std.debug.print("TURBOEXL3_LING_TIME moeLing n={d} topk=1 plugin_call pairs={d} of_tokens={d} distinct={d} min_us={d:.1} med_us={d:.1} tflops={d:.2}\n", .{ n, plugin_call_pairs, tokens, distinctOf(&cs), st.min, st.med, @as(f64, @floatFromInt(plugin_call_pairs)) * expert_flops / (st.min * 1e6) });
        }
        {
            const rm = g.mark();
            defer g.resetTo(rm);
            const slot_rows: u32 = 200;
            const sb: root.Bank = .{
                .gate = try ds_test.randProj(a, &g, rnd, slot_rows, hidden, inter, n),
                .up = try ds_test.randProj(a, &g, rnd, slot_rows, hidden, inter, n),
                .down = try ds_test.randProj(a, &g, rnd, slot_rows, inter, hidden, n),
            };
            const banks = [3]root.Bank{ sb, sb, sb };
            var pool: [24]u32 = undefined;
            {
                var used = std.StaticBitSet(256).empty;
                for (&pool) |*p| {
                    var v = rnd.uintLessThan(u32, slot_rows);
                    while (used.isSet(v)) v = rnd.uintLessThan(u32, slot_rows);
                    used.set(v);
                    p.* = v;
                }
            }
            const BankedCtx = struct {
                s: mlx.mlx_stream,
                x: mlx.mlx_array,
                banks: *const [3]root.Bank,
                ids: mlx.mlx_array,
                pub fn call(cx: @This()) !mlx.mlx_array {
                    return root.moeLingBanked(cx.s, cx.x, cx.banks, cx.ids, .mul1, 0);
                }
            };
            for ([_]u32{ 1, 2, 3 }) |rows| {
                const pm = g.mark();
                defer g.resetTo(pm);
                var ids: [24]u32 = undefined;
                var prow: [24]u32 = undefined;
                const np = rows * top_k;
                drawDecodePairs(rnd, rows, top_k, &pool, ids[0..np], prow[0..np]);
                const ctx: BankedCtx = .{
                    .s = s,
                    .x = try ds_test.randBf16(a, &g, rnd, &.{ ci(np), ci(hidden) }),
                    .banks = &banks,
                    .ids = try g.hostArray(std.mem.sliceAsBytes(ids[0..np]), &.{ci(np)}, .uint32),
                };
                const st = try ds_test.timeIt(&g, ctx, 8);
                const distinct = distinctOf(ids[0..np]);
                std.debug.print("TURBOEXL3_LING_TIME moeLingBanked n={d} rows={d} pairs={d} distinct={d} min_us={d:.1} med_us={d:.1} GBps={d:.0}\n", .{ n, rows, np, distinct, st.min, st.med, @as(f64, @floatFromInt(distinct)) * record / (st.min * 1e3) });
            }
        }
    }
}

/// One layer of the chained decode forward: its mixer's projections, its dense FFN or shared expert, its routed bank
const ChainLayer = struct {
    mixer: union(enum) {
        kda: struct { qkv: root.Proj, f: root.Proj, g: root.Proj, o: root.Proj },
        mla: struct { q_a: root.Proj, q_b: root.Proj, kv_a: root.Proj, o: root.Proj },
    },
    ffn: ?[3]root.Proj = null,
    shared: ?[3]root.Proj = null,
    bank: ?root.Bank = null,
};

const ChainArm = enum { trunk, routed, layer };

fn chainLin(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj) !mlx.mlx_array {
    return g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
}

/// `x` [R, 8192]: the first 8192 columns of `y` [R, >= 8192]
fn firstCols(g: *G, y: mlx.mlx_array, rows: u32) !mlx.mlx_array {
    return g.slice(y, &.{ 0, 0 }, &.{ ci(rows), 8192 }, &.{ 1, 1 });
}

/// `silu(gate) * up` of the two projections of `h`, then the third
fn chainSwiglu(g: *G, s: mlx.mlx_stream, h: mlx.mlx_array, p: [3]root.Proj) !mlx.mlx_array {
    const gt = try chainLin(g, s, h, p[0]);
    const up = try chainLin(g, s, h, p[1]);
    return chainLin(g, s, try g.mul(try g.mul(try g.sigmoid(gt), gt), up), p[2]);
}

/// One decode forward of `rows` rows through `layers`, chained through the residual as the model's layers are: a KDA
/// layer's qkv, f and g of h, then o on qkv's first 8192 columns (the attention output it stands in for); an MLA layer's
/// q_a, then q_b on it, kv_a, then the output projection on q_b's first 8192 columns; each output that feeds nothing
/// folded into h by its row mean; then the dense FFN or the shared expert; then the routed pairs (`ids[L]`, each pair's
/// row `prow`) through moeLingBanked on the layer's bank, each row's pairs reduced (mean) into h. The attention cores, the
/// KDA recurrence, the norms and the router are not in it.
fn chainForward(g: *G, s: mlx.mlx_stream, layers: []const ChainLayer, ids: []const ?mlx.mlx_array, prow: mlx.mlx_array, x: mlx.mlx_array, rows: u32, arm: ChainArm, head: ?root.Proj) !mlx.mlx_array {
    const R = ci(rows);
    var h = x;
    for (layers, ids) |ly, lids| {
        if (arm != .routed) {
            switch (ly.mixer) {
                .kda => |k| {
                    const qkv = try chainLin(g, s, h, k.qkv);
                    const extra = try g.add(try g.mean(try chainLin(g, s, h, k.f), 1, true), try g.mean(try chainLin(g, s, h, k.g), 1, true));
                    h = try g.add(try g.add(h, try chainLin(g, s, try firstCols(g, qkv, rows), k.o)), extra);
                },
                .mla => |m| {
                    const qb = try chainLin(g, s, try chainLin(g, s, h, m.q_a), m.q_b);
                    const extra = try g.mean(try chainLin(g, s, h, m.kv_a), 1, true);
                    h = try g.add(try g.add(h, try chainLin(g, s, try firstCols(g, qb, rows), m.o)), extra);
                },
            }
            if (ly.ffn) |p| h = try g.add(h, try chainSwiglu(g, s, h, p));
            if (ly.shared) |p| h = try g.add(h, try chainSwiglu(g, s, h, p));
        }
        if (arm != .trunk) if (ly.bank) |b| {
            const banks = [3]root.Bank{ b, b, b };
            const xp = try g.take(h, prow, 0);
            const y = try g.adopt(try root.moeLingBanked(s, xp, &banks, lids.?, .mul1, 0));
            const yr = try g.mean(try g.reshape(y, &.{ R, ci(top_k), ci(hidden) }), 1, false);
            h = try g.add(h, try g.astype(yr, mlx.mlx_array_dtype(h)));
        };
    }
    if (head) |p| return chainLin(g, s, h, p);
    return h;
}

// Ling-3.1-flash's decode forward's linears chained over its 72 layers in one evaluation (TURBOEXL3_LING_TIME=1 with
// TURBOEXL3_GPU=1; it asserts nothing; guidance only): `chainForward` with each layer's weights in its own buffers (one
// draw per shape, copied per layer): layers 7, 15, ..., 71 MLA, the others KDA; layers 0-3 the dense FFN (K4), 4-71 the
// shared expert and a routed bank of 24 slot rows (K 2.5: the 2.60 pack), each row's 8 slots drawn from them (8, about 13
// and about 17 distinct a layer at rows 1, 2 and 3); then the head. Arms: the trunk alone, the routed experts alone, the
// whole layer plus the head; rows 1, 2 and 3 (a serial step and the MTP verify widths), banked group 4 with the M = 3
// texts. Each line: the token's min and median ms (host graph build plus evaluation, as a decode step pays it), the
// per-layer us at the min, and the median host graph-build ms. The served forward adds the attention cores, the KDA
// recurrence and its gates, the norms, the router and the streamer's routing barriers.
test "turboexl3 gpu ling timing: the decode forward's linears over 72 layers chained in one evaluation (TURBOEXL3_LING_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_LING_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    const saved = config.get().*;
    defer config.configure(saved) catch {};
    var cfg = saved;
    cfg.banked_group = 4;
    cfg.banked_m3 = true;
    try config.configure(cfg);
    var prng = std.Random.DefaultPrng.init(0x11A6C4);
    const rnd = prng.random();
    const nl = 72;
    const bank_rows: u32 = 24;
    const HostProj = ds_test.HostProj;
    var hp = [_]HostProj{
        try HostProj.draw(a, rnd, null, 4096, 24576, 96), // 0 KDA qkv
        try HostProj.draw(a, rnd, null, 4096, 8192, 96), // 1 KDA f and g
        try HostProj.draw(a, rnd, null, 8192, 4096, 96), // 2 KDA o and the MLA output projection
        try HostProj.draw(a, rnd, null, 4096, 1536, 96), // 3 MLA q_a
        try HostProj.draw(a, rnd, null, 1536, 12288, 96), // 4 MLA q_b
        try HostProj.draw(a, rnd, null, 4096, 640, 96), // 5 MLA kv_a
        try HostProj.draw(a, rnd, null, 4096, 10240, 64), // 6 dense FFN gate and up
        try HostProj.draw(a, rnd, null, 10240, 4096, 64), // 7 dense FFN down
        try HostProj.draw(a, rnd, null, 4096, 1280, 96), // 8 shared gate and up
        try HostProj.draw(a, rnd, null, 1280, 4096, 96), // 9 shared down
        try HostProj.draw(a, rnd, bank_rows, 4096, 1280, 40), // 10 routed gate and up
        try HostProj.draw(a, rnd, bank_rows, 1280, 4096, 40), // 11 routed down
        try HostProj.draw(a, rnd, null, 4096, 157184, 96), // 12 the head
    };
    defer for (&hp) |*h| h.deinit(a);
    var layers: [nl]ChainLayer = undefined;
    for (&layers, 0..) |*ly, li| {
        ly.* = .{ .mixer = if (li % 8 == 7)
            .{ .mla = .{ .q_a = try hp[3].upload(&g), .q_b = try hp[4].upload(&g), .kv_a = try hp[5].upload(&g), .o = try hp[2].upload(&g) } }
        else
            .{ .kda = .{ .qkv = try hp[0].upload(&g), .f = try hp[1].upload(&g), .g = try hp[1].upload(&g), .o = try hp[2].upload(&g) } } };
        if (li < 4) {
            ly.ffn = .{ try hp[6].upload(&g), try hp[6].upload(&g), try hp[7].upload(&g) };
        } else {
            ly.shared = .{ try hp[8].upload(&g), try hp[8].upload(&g), try hp[9].upload(&g) };
            ly.bank = .{ .gate = try hp[10].upload(&g), .up = try hp[10].upload(&g), .down = try hp[11].upload(&g) };
        }
    }
    const head = try hp[12].upload(&g);
    var pool: [bank_rows]u32 = undefined;
    for (&pool, 0..) |*p, i| p.* = @intCast(i);
    for ([_]u32{ 1, 2, 3 }) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        const np = rows * top_k;
        var ids: [nl]?mlx.mlx_array = @splat(null);
        var idb: [24]u32 = undefined;
        var prb: [24]u32 = undefined;
        var distinct: usize = 0;
        for (layers, &ids) |ly, *d| if (ly.bank != null) {
            drawDecodePairs(rnd, rows, top_k, &pool, idb[0..np], prb[0..np]);
            distinct += distinctOf(idb[0..np]);
            d.* = try g.hostArray(std.mem.sliceAsBytes(idb[0..np]), &.{ci(np)}, .uint32);
        };
        const prow = try g.hostArray(std.mem.sliceAsBytes(prb[0..np]), &.{ci(np)}, .uint32);
        const x = try ds_test.randBf16(a, &g, rnd, &.{ ci(rows), 4096 });
        for ([_]ChainArm{ .trunk, .routed, .layer }) |arm| {
            var build: [5]f64 = undefined;
            var total: [5]f64 = undefined;
            for (0..build.len + 2) |round| {
                const fm = g.mark();
                defer g.resetTo(fm);
                const t0 = ds_test.nowNs();
                const y = try chainForward(&g, s, &layers, &ids, prow, x, rows, arm, if (arm == .layer) head else null);
                const t1 = ds_test.nowNs();
                try g.evalAll(&.{y});
                const t2 = ds_test.nowNs();
                if (round >= 2) {
                    build[round - 2] = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
                    total[round - 2] = @as(f64, @floatFromInt(t2 - t0)) / 1000.0;
                }
            }
            std.mem.sort(f64, &build, {}, std.sort.asc(f64));
            std.mem.sort(f64, &total, {}, std.sort.asc(f64));
            std.debug.print("TURBOEXL3_LING_CHAIN arm={s} rows={d} distinct_per_layer={d:.1} token_min_ms={d:.2} token_med_ms={d:.2} per_layer_us={d:.1} graph_build_med_ms={d:.2}{s}\n", .{ @tagName(arm), rows, @as(f64, @floatFromInt(distinct)) / 68.0, total[0] / 1000.0, total[total.len / 2] / 1000.0, total[0] / @as(f64, nl), build[build.len / 2] / 1000.0, if (arm == .layer) " (the head included)" else "" });
        }
    }
}

// ── TURBOEXL3_LING_TIME: the dense decode GEMV's launch shapes per Ling trunk shape (a sweep; asserts bytes only) ──

/// One launch-shape choice for a trunk shape class at a row count; null fields keep the served value (deepseek_gemv_test's
/// GemvConf without the grouped arm): tiles per threadgroup, DEC-ONE's prepare inside or in its own launch, DEC-ONE
/// against the three-launch chain, DEC-ONE's row cap, the lean text.
const LingGemvConf = struct {
    bt: u8 = 0,
    dec_on: ?bool = null,
    dec_rows: ?u32 = null,
    prep_in: ?bool = null,
    lean: ?bool = null,

    fn isDefault(c: LingGemvConf) bool {
        return c.bt == 0 and c.dec_on == null and c.dec_rows == null and c.prep_in == null and c.lean == null;
    }

    fn name(c: LingGemvConf, buf: []u8) []const u8 {
        if (c.isDefault()) return "served";
        const tri = struct {
            fn f(v: ?bool) []const u8 {
                return if (v) |b| (if (b) "1" else "0") else "-";
            }
        }.f;
        return std.fmt.bufPrint(buf, "bt{d}_dec{s}_decrows{d}_prepin{s}_lean{s}", .{ c.bt, tri(c.dec_on), c.dec_rows orelse 0, tri(c.prep_in), tri(c.lean) }) catch "?";
    }
};

const LingDenseSaved = struct { bt: u8, dec_on: bool, dec_rows: u32, dec_prep_rows: u32, dec_prep_out: u32, lean: bool };

fn lingApplyDense(d: *dense.Dense, rows: u32, c: LingGemvConf) LingDenseSaved {
    const sv: LingDenseSaved = .{ .bt = d.bt_set[rows], .dec_on = d.dec_on, .dec_rows = d.dec_rows, .dec_prep_rows = d.dec_prep_rows, .dec_prep_out = d.dec_prep_out, .lean = d.lean };
    if (c.bt != 0) d.bt_set[rows] = c.bt;
    if (c.dec_on) |v| d.dec_on = v;
    if (c.dec_rows) |v| d.dec_rows = v;
    if (c.lean) |v| d.lean = v;
    if (c.prep_in) |v| {
        // inside: every row count and width that fits; outside: none
        d.dec_prep_rows = if (v) @intCast(decode.Kernels.max_rows) else 0;
        d.dec_prep_out = if (v) std.math.maxInt(u32) else 0;
    }
    d.resetCfgs();
    return sv;
}

fn lingRestoreDense(d: *dense.Dense, rows: u32, sv: LingDenseSaved) void {
    d.bt_set[rows] = sv.bt;
    d.dec_on = sv.dec_on;
    d.dec_rows = sv.dec_rows;
    d.dec_prep_rows = sv.dec_prep_rows;
    d.dec_prep_out = sv.dec_prep_out;
    d.lean = sv.lean;
    d.resetCfgs();
}

/// The launch-shape choices tried for a class's Dense at `rows` (the served one first): each other tile count whose
/// accumulators fit 32 floats a lane; where DEC-ONE takes the call, its prepare in the other place (where it fits) and the
/// three-launch chain; past DEC-ONE's rows, the chain off the lean text and DEC-ONE at up to 8 rows
fn lingGemvConfs(d: *dense.Dense, rows: u32, buf: []LingGemvConf) ![]const LingGemvConf {
    var n: usize = 0;
    buf[n] = .{};
    n += 1;
    const dec = d.decTakes(rows);
    const rb = if (dec) dense.Dense.rbFor(rows) else d.chainRb(rows);
    const served_bt = if (dec) d.btDec(rows, rb) else d.btAt(rows, rb);
    for ([_]u8{ 1, 2, 4, 8 }) |bt| {
        if (bt == served_bt or 2 * @as(u32, bt) * rb > 32) continue;
        buf[n] = .{ .bt = bt };
        n += 1;
    }
    if (dec) {
        const inside = try d.decPrepInside(rows);
        const fits = dense.Dense.decPrepFits(d.shape, d.ks_split, rb, served_bt);
        if (inside or fits) {
            buf[n] = .{ .prep_in = !inside };
            n += 1;
            // the prepare moved together with each other tile count (T 3a's two-row shape at one row: own launch, BT 2)
            for ([_]u8{ 1, 2, 4, 8 }) |bt| {
                if (bt == served_bt or 2 * @as(u32, bt) * rb > 32) continue;
                if (inside == false and !dense.Dense.decPrepFits(d.shape, d.ks_split, rb, bt)) continue;
                buf[n] = .{ .bt = bt, .prep_in = !inside };
                n += 1;
            }
        }
        buf[n] = .{ .dec_on = false };
        n += 1;
    } else {
        if (d.lean) {
            buf[n] = .{ .lean = false };
            n += 1;
        }
        buf[n] = .{ .dec_rows = 8 };
        n += 1;
    }
    return buf[0..n];
}

/// The 72-layer chain's trunk (and the head) at `rows`: min ms of 5 evaluations after 2, and the output's bytes (f32) in
/// `out` when given
fn lingTrunkMin(g: *G, s: mlx.mlx_stream, layers: []const ChainLayer, head: root.Proj, x: mlx.mlx_array, rows: u32, out: ?[]f32) !f64 {
    var no_ids: [72]?mlx.mlx_array = @splat(null);
    var total: [5]f64 = undefined;
    for (0..total.len + 2) |round| {
        const fm = g.mark();
        defer g.resetTo(fm);
        const t0 = ds_test.nowNs();
        const y = try chainForward(g, s, layers, no_ids[0..layers.len], .{ .ctx = null }, x, rows, .trunk, head);
        try g.evalAll(&.{y});
        const dt = ds_test.nowNs() - t0;
        if (round >= 2) total[round - 2] = @as(f64, @floatFromInt(dt)) / 1e6;
        if (round == 0) if (out) |o| {
            _ = try g.hostF32(try g.astype(y, .float32), o);
        };
    }
    std.mem.sort(f64, &total, {}, std.sort.asc(f64));
    return total[0];
}

// TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1 (a diagnostic: guidance, never a verdict; it asserts bytes only): the dense
// decode GEMV's launch shapes swept per Ling-3.1-flash trunk shape class on the 72-layer chain's trunk arm and the head
// (`chainForward`, the served kernels around each class), rows 1..8. Per row count: the chain as served, then each
// launch-shape choice (`lingGemvConfs`) on one class's linears alone, the chain's ms and its delta from the served chain's,
// and the chain output's bytes against the served chain's (every choice is a launch shape: the bytes must not move). Then
// the fastest choice per (row count, class) past 0.3% of the served chain, and all of a row count's choices together
// against the served chain in interleaved rounds. Lines: TURBOEXL3_LING_GEMV, _BEST, _TABLE.
test "turboexl3 gpu ling timing: the dense decode GEMV's launch shapes per trunk shape on the 72-layer chain, rows 1..8 (TURBOEXL3_LING_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_LING_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    var prng = std.Random.DefaultPrng.init(0x11A6C5);
    const rnd = prng.random();
    const nl = 72;
    const HostProj = ds_test.HostProj;
    var hp = [_]HostProj{
        try HostProj.draw(a, rnd, null, 4096, 24576, 96), // 0 KDA qkv
        try HostProj.draw(a, rnd, null, 4096, 8192, 96), // 1 KDA f and g
        try HostProj.draw(a, rnd, null, 8192, 4096, 96), // 2 KDA o and the MLA output projection
        try HostProj.draw(a, rnd, null, 4096, 1536, 96), // 3 MLA q_a
        try HostProj.draw(a, rnd, null, 1536, 12288, 96), // 4 MLA q_b
        try HostProj.draw(a, rnd, null, 4096, 640, 96), // 5 MLA kv_a
        try HostProj.draw(a, rnd, null, 4096, 10240, 64), // 6 dense FFN gate and up
        try HostProj.draw(a, rnd, null, 10240, 4096, 64), // 7 dense FFN down
        try HostProj.draw(a, rnd, null, 4096, 1280, 96), // 8 shared gate and up
        try HostProj.draw(a, rnd, null, 1280, 4096, 96), // 9 shared down
        try HostProj.draw(a, rnd, null, 4096, 157184, 96), // 10 the head
    };
    defer for (&hp) |*h| h.deinit(a);
    var layers: [nl]ChainLayer = undefined;
    for (&layers, 0..) |*ly, li| {
        ly.* = .{ .mixer = if (li % 8 == 7)
            .{ .mla = .{ .q_a = try hp[3].upload(&g), .q_b = try hp[4].upload(&g), .kv_a = try hp[5].upload(&g), .o = try hp[2].upload(&g) } }
        else
            .{ .kda = .{ .qkv = try hp[0].upload(&g), .f = try hp[1].upload(&g), .g = try hp[1].upload(&g), .o = try hp[2].upload(&g) } } };
        if (li < 4) {
            ly.ffn = .{ try hp[6].upload(&g), try hp[6].upload(&g), try hp[7].upload(&g) };
        } else {
            ly.shared = .{ try hp[8].upload(&g), try hp[8].upload(&g), try hp[9].upload(&g) };
        }
    }
    const head = try hp[10].upload(&g);
    // each class's cached Dense (the object `root.linear` runs the class's linears on), bf16 x
    var dense_of: [dense_cases.len]*dense.Dense = undefined;
    for (dense_cases, &dense_of) |c, *d| d.* = try root.denseCached(.{ .in = c.in, .out = c.out, .rate = format.Rate.fromN(c.n).?, .dec = .mul1, .out_dtype = .bfloat16 });
    var failed: usize = 0;
    var name_buf: [96]u8 = undefined;
    for (1..9) |ri| {
        const rows: u32 = @intCast(ri);
        const rm = g.mark();
        defer g.resetTo(rm);
        const x = try ds_test.randBf16(a, &g, rnd, &.{ ci(rows), 4096 });
        const n_out = rows * 157184;
        const want = try a.alloc(f32, n_out);
        defer a.free(want);
        const got = try a.alloc(f32, n_out);
        defer a.free(got);
        const served0 = try lingTrunkMin(&g, s, &layers, head, x, rows, want);
        var best: [dense_cases.len]LingGemvConf = @splat(.{});
        var best_ms: [dense_cases.len]f64 = undefined;
        for (dense_cases, dense_of, 0..) |c, d, k| {
            var cbuf: [16]LingGemvConf = undefined;
            const confs = try lingGemvConfs(d, rows, &cbuf);
            best_ms[k] = served0;
            for (confs[1..]) |cf| {
                const sv = lingApplyDense(d, rows, cf);
                const ms = try lingTrunkMin(&g, s, &layers, head, x, rows, got);
                lingRestoreDense(d, rows, sv);
                const differing = bytesDifferF32(got, want);
                if (differing != 0) failed += 1;
                std.debug.print("TURBOEXL3_LING_GEMV rows={d} class={s} conf={s} chain_ms={d:.3} served_ms={d:.3} delta_ms={d:.3} differing={d}/{d} {s}\n", .{ rows, c.name, cf.name(&name_buf), ms, served0, ms - served0, differing, n_out, if (differing == 0) "BYTES_OK" else "BYTES_FAIL" });
                if (ms < best_ms[k] and ms < served0 * (1.0 - 0.003)) {
                    best_ms[k] = ms;
                    best[k] = cf;
                }
            }
            std.debug.print("TURBOEXL3_LING_GEMV_BEST rows={d} class={s} conf={s} chain_ms={d:.3} served_ms={d:.3} delta_ms={d:.3}\n", .{ rows, c.name, best[k].name(&name_buf), best_ms[k], served0, best_ms[k] - served0 });
        }
        // the row count's choices together against the served chain, interleaved
        var sv_all: [dense_cases.len]LingDenseSaved = undefined;
        var served_min: f64 = std.math.inf(f64);
        var table_min: f64 = std.math.inf(f64);
        var differing_all: usize = 0;
        for (0..3) |_| {
            served_min = @min(served_min, try lingTrunkMin(&g, s, &layers, head, x, rows, null));
            for (dense_of, best, &sv_all) |d, cf, *sv| sv.* = lingApplyDense(d, rows, cf);
            table_min = @min(table_min, try lingTrunkMin(&g, s, &layers, head, x, rows, got));
            for (dense_of, sv_all) |d, sv| lingRestoreDense(d, rows, sv);
            differing_all = @max(differing_all, bytesDifferF32(got, want));
        }
        if (differing_all != 0) failed += 1;
        std.debug.print("TURBOEXL3_LING_GEMV_TABLE rows={d} served_ms={d:.3} table_ms={d:.3} delta_ms={d:.3} differing={d}/{d} {s}\n", .{ rows, served_min, table_min, table_min - served_min, differing_all, n_out, if (differing_all == 0) "BYTES_OK" else "BYTES_FAIL" });
    }
    std.debug.print("TURBOEXL3_LING_GEMV_DONE bytes_failures={d} {s}\n", .{ failed, if (failed == 0) "BYTES_OK" else "BYTES_FAIL" });
    try testing.expectEqual(@as(usize, 0), failed);
}

fn bytesDifferF32(p: []const f32, q: []const f32) usize {
    var n: usize = 0;
    for (p, q) |u, w| n += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return n;
}

// D1: launch 2 of the MTP head's latent attention on the matrix units (`mla_latent.Attn.nax_*`), TIME's arms only: each
// NAX shape (16, 32 or 64 heads a threadgroup; 32 or 64 keys a step; the scores whole a simdgroup or split over an M
// tile's 4) against the served stage form (S40, `Attn.stage`) and launch 2 as it is, at rows 1 to 5 over the standard
// request's 18,380 keys, the operands as the plugin passes them (the cache the first rows of a wider bf16 buffer, kv_b
// bf16 [16384, 512], the query halves as views of the transposed heads). Guidance that ranks the shapes; the standard
// test decides the switch.
test "turboexl3 gpu ling timing: D1, the MTP head's latent attention on the matrix units at rows 1 to 5 over 18,380 keys (TURBOEXL3_LING_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_LING_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    // a shape MLX refuses to launch prints its error line and the arms go on
    mla_latent_test.reportMlxErrors(true);
    defer mla_latent_test.reportMlxErrors(false);
    var d1_prng = std.Random.DefaultPrng.init(0x0D1_71AE);
    const mr = d1_prng.random();
    const keys: u32 = 18380;
    const heads: u32 = 64;
    const lat = try g.slice(try ds_test.randBf16(a, &g, mr, &.{ 1, 1, ci(keys + 64), 512 }), &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(keys), 512 }, &.{ 1, 1, 1, 1 });
    const rot = try g.slice(try ds_test.randBf16(a, &g, mr, &.{ 1, 1, ci(keys + 64), 64 }), &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(keys), 64 }, &.{ 1, 1, 1, 1 });
    const ws: f32 = 1.0 / @sqrt(512.0);
    const kv_b = try g.astype(try g.mul(try g.astype(try ds_test.randBf16(a, &g, mr, &.{ ci(heads * 256), 512 }), .float32), try g.hostArray(std.mem.asBytes(&ws), &.{1}, .float32)), .bfloat16);
    try g.evalAll(&.{ lat, rot, kv_b });
    const NaxCtx = struct {
        s: mlx.mlx_stream,
        qn: mlx.mlx_array,
        qr: mlx.mlx_array,
        lat: mlx.mlx_array,
        rot: mlx.mlx_array,
        w: mlx.mlx_array,
        attn: mla_latent.Attn,
        pub fn call(cx: @This()) !mlx.mlx_array {
            return mla_latent.decodeLatentWith(cx.s, cx.qn, cx.qr, cx.lat, cx.rot, cx.w, 1.0 / @sqrt(192.0), cx.attn);
        }
    };
    // the attention's multiply-adds a key and row: 64 heads x (576 + 512)
    const flops_row = 2.0 * @as(f64, @floatFromInt(keys)) * @as(f64, @floatFromInt(heads)) * @as(f64, @floatFromInt(mla_latent.dl + mla_latent.dr + mla_latent.dl));
    const arms = [_]mla_latent.Attn{ .stage, .base } ++ mla_latent.nax_shapes;
    for ([_]u32{ 1, 2, 3, 4, 5 }) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        const qt = try g.transposeAxes(try ds_test.randBf16(a, &g, mr, &.{ 1, ci(rows), ci(heads), 192 }), &.{ 0, 2, 1, 3 });
        const qn = try g.slice(qt, &.{ 0, 0, 0, 0 }, &.{ 1, ci(heads), ci(rows), 128 }, &.{ 1, 1, 1, 1 });
        const qr = try g.slice(qt, &.{ 0, 0, 0, 128 }, &.{ 1, ci(heads), ci(rows), 192 }, &.{ 1, 1, 1, 1 });
        var stage_min: f64 = 0;
        for (arms) |v| {
            const ctx: NaxCtx = .{ .s = s, .qn = qn, .qr = qr, .lat = lat, .rot = rot, .w = kv_b, .attn = v };
            const st = ds_test.timeIt(&g, ctx, 8) catch |e| {
                std.debug.print("TURBOEXL3_LING_TIME mla_latent_nax keys={d} rows={d} arm={s}: {s}\n", .{ keys, rows, @tagName(v), @errorName(e) });
                continue;
            };
            if (v == .stage) stage_min = st.min;
            const r: f64 = @floatFromInt(rows);
            std.debug.print("TURBOEXL3_LING_TIME mla_latent_nax keys={d} rows={d} arm={s} min_us={d:.1} med_us={d:.1} us_per_row={d:.1} of_stage={d:.3} tflops={d:.2}\n", .{ keys, rows, @tagName(v), st.min, st.med, st.min / r, if (stage_min > 0) st.min / stage_min else 0, r * flops_row / (st.min * 1e6) });
        }
    }
}
