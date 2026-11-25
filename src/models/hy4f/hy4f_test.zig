//! Hy4-flash (tencent/Hy4-flash; its EXL3 packs, MCG): tests at Hy4's geometry. turboexl3 has no Hy4 entry: the quant
//! plugin turboexl3_hy4f runs the routed experts on `root.moeSwiglu` at `silu_out` (exllamav3's `act_limit` with SiLU,
//! min(silu(g), 10) * clamp(u, -10, 10)) and every trunk linear and the head on `root.linear`.
//! CPU (always): the routed geometry is a decode and prefill geometry at K1 to K4; every trunk shape a dense width.
//! GPU (TURBOEXL3_GPU=1, else skipped):
//! - routed: 512 experts, hidden = expert width 1536 (the latent MoE between routed_expert_down_proj and
//!   routed_expert_up_proj), top-16, weights summing to 3.998 (routed_scaling_factor), limit 10 at `silu_out`, K1 to K4
//!   (n 16..64; TURBOEXL3_HY4F_N narrows), x ~ N(0, 8^2) in bf16 and f32: `root.moeSwiglu` equals `ref.moeRowClampedAt`
//!   on every row of decode calls of 1 to 8 rows and on both ends of a 1328-row prefill call, with at least 1% of the
//!   sampled SiLUs and ups clamped. Red: the same output against the unclamped reference must miss the bar. Lines:
//!   TURBOEXL3_HY4F_MOE, TURBOEXL3_HY4F_MOE_RED.
//! - dense: `root.linear` at every Hy4 trunk shape and the head at K6 equals `ref.Linear.row` at rows 1, 4, 8, 32, 512
//!   (the head 1, 4). Red: at kv_a's stored shape (3072 -> 640), the reference decoded as MUL1 must miss the bar.
//!   Lines: TURBOEXL3_HY4F_DENSE, TURBOEXL3_HY4F_DENSE_RED.
//! - routed tiles: form 9 at 4 output tiles from two rows (`Config.decode_bt9` 4) and the x texts (`Config.decode_iop`)
//!   against the served engine byte for byte at decode rows 1 to 8, K1 to K4; a changed input and the x texts' planted
//!   defect must move words. Lines: TURBOEXL3_HY4F_ROUTED_TILES.
//! - verify rows: the chain at 2 output tiles (`Config.chain_bt` 2) against the chain byte for byte; the lean chunks on
//!   the matrix units (`Config.dense_mma`, reorder-class) within the float64 summation bound on their K-slice planes
//!   beside today's chain (its rms gap against today's printed), three planted faults past the bound (the red checks),
//!   and their words against today's counted. Lines: TURBOEXL3_HY4F_VERIFY_ROWS, TURBOEXL3_HY4F_MMA_BAR,
//!   TURBOEXL3_HY4F_MMA_RED.
//! - gatemul: `linearGateMul` (gate.zig, the hy4f model plugin's `gate_fused`) at linear_gate's widths (3072 -> 32
//!   and 64 heads of 256), K6 MCG, rows 20 (the register path), 33, 1024 and 3584, x bf16 with the product in bf16 and
//!   x f32 with it in f32: every word the chain's (`root.linear`, MLX's astype to f32, sigmoid, o's transpose, multiply,
//!   astype). Red: at the sliding width and 1024 rows, o read in the flat order, the sigmoid's exp on the fast path and
//!   the one-sided sigmoid must each move words. CPU: the decode rows and wrong operands refuse before any engine.
//!   Lines: TURBOEXL3_HY4F_GATEMUL, TURBOEXL3_HY4F_GATEMUL_RED, then TURBOEXL3_HY4F_GATEMUL_DONE.
//! - swiglumul: `linearSwigluMul` (swiglu.zig, the hy4f model plugin's `swiglu_fused`) at the shared expert's width
//!   (3072 -> 1536) and the dense MLP's (3072 -> 13824), K6 MCG, rows 20, 33, 1024 and 3584, x bf16 with h in bf16 and
//!   x f32 with h in f32: every word the chain's (`root.linear` for the gate and up, MLX's astype to f32, sigmoid,
//!   multiply, multiply, astype). Red: at the dense width and 1024 rows, the gate and up swapped and the sigmoid's exp on
//!   the fast path must each move words. CPU: the decode rows and wrong operands refuse before any engine. Lines:
//!   TURBOEXL3_HY4F_SWIGLUMUL, TURBOEXL3_HY4F_SWIGLUMUL_RED, then TURBOEXL3_HY4F_SWIGLUMUL_DONE.
//! - qrope: `linearQRope` (qrope.zig, the hy4f model plugin's `qb_rope_fused`) at q_b's widths (1024 -> 32 heads of
//!   256 at theta 1e7, 64 heads at 1e4; the last 64 dims of each head roped), K6 MCG, rows 33 at positions 0 and 17459,
//!   1024 at 2048 and 3584 at 13905, x bf16 and f32: every word the chain's (`root.linear`, MLX's astype to f32, the
//!   transpose, MLX's fast::rope of the strided slice, traditional, scale 1). Red: at the sliding width and 1024 rows,
//!   the NeoX pairs and MLX's CPU fallback frequency (exp of the natural log) must each move the rope dims' words and
//!   no other. CPU: the decode rows, wrong dtypes, thetas, positions and head geometries refuse before any engine.
//!   Lines: TURBOEXL3_HY4F_QROPE, TURBOEXL3_HY4F_QROPE_RED, then TURBOEXL3_HY4F_QROPE_DONE.
//! - pg2arms (pg2, core/pgemm.zig, the host's switch; the tests' names carry "pg2arms"): the gatemul, swiglumul and qrope
//!   tests (rows 3125 too) run each call again with pg2 on, the table naming another shape (the arm's finish stays on, no
//!   pg2 launch) and at each of `pgemm.candidates` (pg2's GEMM with the arm's finish where the tile's N divides the
//!   width): every word the chain's, and the planted faults move them in pg2's launch too. At every trunk shape
//!   (`hy4f_pg2.trunk`), rows 33, 1024, 3125 and 3584, `root.linear` with pg2 on (each candidate tile at bf16, the served
//!   table at bf16 and f32) gives the chain's words, split K in registers at o_proj.sliding and dense.down; there MLX's
//!   regular loop and 2048-k partitions must move words (red). CPU: each arm's pg2 text is pg2's pieces around its store.
//!   Lines: TURBOEXL3_HY4F_GATEMUL_PG2(_RED), TURBOEXL3_HY4F_SWIGLUMUL_PG2(_RED), TURBOEXL3_HY4F_QROPE_PG2(_RED),
//!   TURBOEXL3_HY4F_LINEAR_PG2(_RED, _DONE).
//! - finprep (rms.zig, the hy4f model plugin's `addnorm_fused`, `qnorm_fused`, `kvnorm_fused`,
//!   `down_prep_fused`; every name holds finprep, a one-word filter): `linearRms` with the add at the shared down
//!   (1536 -> 3072), from x and from prepared rows; `linearRmsPrep` at q_a (3072 -> 1024) with q_b's two widths;
//!   `linearKv` at kv_a's stored width (3072 -> 640, latent 512, rope 64) at both thetas; `linearSwigluPrep` at the
//!   shared expert's and the dense MLP's widths. K6 MCG, rows 20 to 3584, x bf16 and f32: every word the chain's
//!   (`root.linear`, MLX's astype, add, rms_norm, fast::rope, q_b's and the down's own prepare, and those linears
//!   continued from the prepared rows). Red: at 1024 rows, the add left out, the prepare without its suh, the kv norm
//!   without its weight and k_pe's CPU fallback frequency must each move words (the kv faults only their own output).
//!   CPU: the texts, and the decode rows and wrong operands refuse before any engine. No f32 matmul runs on either side,
//!   so MLX's TF32 setting moves no word. The gate's prep finish (`linearGateMulPrep`) and pg2 from prepared rows
//!   (`root.linearOnPrepared`, `linearQRopePrepared`) likewise. Lines: TURBOEXL3_HY4F_FINPREP_*(_RED, _DONE).
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
const ks = sdk.kernel_launch;

const testing = std.testing;

const width: u32 = 1536;
const experts: u32 = 512;
const top_k: u32 = 16;
const limit: u32 = 10;
const routed_scale: f32 = 3.998;
const sigma: f32 = 8;
const dec: format.Decode = .mcg;
/// The decode and prefill chains' bar (clamped_test's): 5e-3 of the reference's rms; bf16 beyond its store's rounding.
const bar: f64 = 5e-3;

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

/// Every Hy4 trunk EXL3 linear shape (in -> out as the 2.25bpw pack stores it) and the head; all K6 there.
const DenseCase = struct { name: []const u8, in: u32, out: u32 };
const dense_cases = [_]DenseCase{
    .{ .name = "q_a", .in = 3072, .out = 1024 },
    .{ .name = "q_b.indexed", .in = 1024, .out = 8192 },
    .{ .name = "q_b.sliding", .in = 1024, .out = 16384 },
    .{ .name = "kv_a(576 stored as 640)", .in = 3072, .out = 640 },
    .{ .name = "linear_gate.indexed", .in = 3072, .out = 8192 },
    .{ .name = "linear_gate.sliding", .in = 3072, .out = 16384 },
    .{ .name = "o_proj.indexed", .in = 8192, .out = 3072 },
    .{ .name = "o_proj.sliding", .in = 16384, .out = 3072 },
    .{ .name = "routed_expert_down/shared.gate_up", .in = 3072, .out = 1536 },
    .{ .name = "routed_expert_up/shared.down", .in = 1536, .out = 3072 },
    .{ .name = "dense.gate_up", .in = 3072, .out = 13824 },
    .{ .name = "dense.down", .in = 13824, .out = 3072 },
    .{ .name = "lm_head", .in = 3072, .out = 120832 },
};

test "turboexl3 hy4f: the routed geometry (1536 x 1536, 512 experts, top-16, limit 10 silu_out) is a decode and prefill geometry at K1 to K4; every trunk shape is a dense width" {
    for (dense_cases) |c| {
        try testing.expect(c.in % 128 == 0 and c.out % 128 == 0 and (c.in / 16) % 4 == 0);
        const k = dense.Dense.ksFor(c.in, c.out);
        try testing.expect(k >= 1 and ((c.in / 16) / 4) % k == 0);
    }
    try testing.expect(format.limitAdmitted(limit, .silu_out));
    for ([_]u32{ 16, 32, 48, 64 }) |n| {
        const rate = format.Rate.fromN(n).?;
        try testing.expect((decode.Geometry{ .hidden = width, .inter = width, .topk = top_k, .rate = rate, .dec = dec, .limit = limit, .clamp = .silu_out }).admitted());
        try testing.expect((prefill.Geometry{ .hidden = width, .inter = width, .topk = top_k, .experts = experts, .rate = rate, .dec = dec, .limit = limit, .clamp = .silu_out }).admitted());
    }
}

fn rootBank(g: *G, b: *const ref.Bank) !root.Bank {
    const E = ci(b.experts);
    const ht = ci(b.hidden / 16);
    const it = ci(b.inter / 16);
    const n = ci(b.rate.n);
    return .{
        .gate = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(b.gate_t), &.{ E, ht, it, n }, .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(b.gate_suh), &.{ E, ci(b.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.gate_svh), &.{ E, ci(b.inter) }, .float16) },
        .up = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(b.up_t), &.{ E, ht, it, n }, .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(b.up_suh), &.{ E, ci(b.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.up_svh), &.{ E, ci(b.inter) }, .float16) },
        .down = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(b.down_t), &.{ E, it, ht, n }, .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(b.down_suh), &.{ E, ci(b.inter) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.down_svh), &.{ E, ci(b.hidden) }, .float16) },
    };
}

/// One call's operands: x [rows, width] bf16 values, each row's top-16 distinct experts (row 0 takes the last expert) and
/// positive bf16-exact weights summing to 3.998 before rounding, as Hy4's router hands them.
const Call = struct {
    x: []f32,
    slots: []u32,
    scores: []f32,

    fn make(a: std.mem.Allocator, rnd: std.Random, rows: u32) !Call {
        const c: Call = .{ .x = try a.alloc(f32, rows * width), .slots = try a.alloc(u32, rows * top_k), .scores = try a.alloc(f32, rows * top_k) };
        for (c.x) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
        for (0..rows) |r| {
            const sl = c.slots[r * top_k ..][0..top_k];
            for (sl, 0..) |*e, i| {
                while (true) {
                    e.* = if (r == 0 and i == 0) experts - 1 else rnd.uintLessThan(u32, experts);
                    if (std.mem.indexOfScalar(u32, sl[0..i], e.*) == null) break;
                }
            }
            const ws = c.scores[r * top_k ..][0..top_k];
            var sum: f32 = 0;
            for (ws) |*w| {
                w.* = 0.05 + rnd.float(f32);
                sum += w.*;
            }
            for (ws) |*w| w.* = bfExact(routed_scale * w.* / sum);
        }
        return c;
    }

    fn deinit(c: *Call, a: std.mem.Allocator) void {
        a.free(c.x);
        a.free(c.slots);
        a.free(c.scores);
    }
};

/// `root.moeSwiglu` at limit 10 `silu_out` over the call's first `rows` rows, x in `dt`, y as f32 into `out`.
fn moeRows(g: *G, s: mlx.mlx_stream, rb: root.Bank, c: *const Call, rows: u32, dt: mlx.mlx_dtype, out: []f32) !void {
    const m = g.mark();
    defer g.resetTo(m);
    const x3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(c.x[0 .. rows * width]), &.{ 1, ci(rows), ci(width) }, .float32), dt);
    const inds3 = try g.hostArray(std.mem.sliceAsBytes(c.slots[0 .. rows * top_k]), &.{ 1, ci(rows), ci(top_k) }, .uint32);
    const sc3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(c.scores[0 .. rows * top_k]), &.{ 1, ci(rows), ci(top_k) }, .float32), .bfloat16);
    const y = try g.adopt(try root.moeSwiglu(s, x3, rb, inds3, sc3, dec, limit, .silu_out));
    try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(width) }, mlx.getShape(y));
    try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
    _ = try g.hostF32(try g.astype(y, .float32), out[0 .. rows * width]);
}

/// The worst ratio of `got`'s rows `sample` against `want` (row i of `want` is row sample[i]); the nonfinite count.
fn worstOf(got: []const f32, want: []const f32, sample: []const u32, dt: mlx.mlx_dtype) struct { worst: f64, bad: usize } {
    var worst: f64 = 0;
    var bad: usize = 0;
    for (sample, 0..) |r, i| {
        const gr = got[r * width ..][0..width];
        for (gr) |v| bad += @intFromBool(!std.math.isFinite(v));
        const wr = want[i * width ..][0..width];
        worst = @max(worst, (if (dt == .float32) ref.compare(gr, wr) else ref.compareRounded(gr, wr, 0x1p-8)).max_ratio);
    }
    return .{ .worst = worst, .bad = bad };
}

test "turboexl3 gpu hy4f moe: root.moeSwiglu at Hy4's routed geometry (1536 x 1536, 512 experts, top-16, MCG K1 to K4, limit 10 silu_out) equals the reference at decode rows 1 to 8 and a 1328-row prefill; the unclamped reference misses (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_HY4F_N", &.{ 16, 32, 48, 64 }, &nbuf);
    var failed = false;
    for (rates) |n| {
        defer root.releaseEngines();
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x4F4F ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, experts, width, width, rate);
        defer bank.deinit(a);
        const rb = try rootBank(&g, &bank);
        // decode: one call of 8 rows whose first `rows` rows each decode call takes; prefill: 1328 rows, both ends sampled
        var dc = try Call.make(a, rnd, 8);
        defer dc.deinit(a);
        var pc = try Call.make(a, rnd, 1328);
        defer pc.deinit(a);
        const dec_rows = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
        const pf_rows = [_]u32{ 0, 1327 };
        const want_d = try a.alloc(f32, 8 * width);
        defer a.free(want_d);
        const want_p = try a.alloc(f32, 2 * width);
        defer a.free(want_p);
        const plain0 = try a.alloc(f32, width);
        defer a.free(plain0);
        var hits: ref.Hits = .{};
        for (dec_rows) |r| hits = add(hits, try ref.moeRowClampedAt(a, &bank, dec, @floatFromInt(limit), .silu_out, dc.x[r * width ..][0..width], dc.slots[r * top_k ..][0..top_k], dc.scores[r * top_k ..][0..top_k], want_d[r * width ..][0..width]));
        for (pf_rows, 0..) |r, i| hits = add(hits, try ref.moeRowClampedAt(a, &bank, dec, @floatFromInt(limit), .silu_out, pc.x[r * width ..][0..width], pc.slots[r * top_k ..][0..top_k], pc.scores[r * top_k ..][0..top_k], want_p[i * width ..][0..width]));
        _ = try ref.moeRowClampedAt(a, &bank, dec, 0, .silu_out, dc.x[0..width], dc.slots[0..top_k], dc.scores[0..top_k], plain0);
        const hit = hits.gate * 100 >= hits.total and hits.up * 100 >= hits.total;
        const got = try a.alloc(f32, 1328 * width);
        defer a.free(got);
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
            for (1..9) |rows_u| {
                const rows: u32 = @intCast(rows_u);
                try moeRows(&g, s, rb, &dc, rows, dt, got);
                const e = worstOf(got, want_d, dec_rows[0..rows], dt);
                const ok = hit and e.bad == 0 and e.worst <= bar;
                std.debug.print("TURBOEXL3_HY4F_MOE n={d} rows={d} path=decode out={s} worst_vs_ref={e:.3} bar={e:.1} nonfinite={d} silu_clamped={d}/{d} ups_clamped={d}/{d} {s}\n", .{ n, rows, @tagName(dt), e.worst, bar, e.bad, hits.gate, hits.total, hits.up, hits.total, if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
                if (rows == 8 and dt == .float32) {
                    // the red check: decode row 0 against the unclamped reference must miss the bar
                    const red = worstOf(got, plain0, dec_rows[0..1], dt);
                    const red_ok = red.worst > bar;
                    std.debug.print("TURBOEXL3_HY4F_MOE_RED n={d} rows=8 row=0 vs_unclamped={e:.3} bar={e:.1} {s}\n", .{ n, red.worst, bar, if (red_ok) "PASS" else "FAIL" });
                    if (!red_ok) failed = true;
                }
            }
            try moeRows(&g, s, rb, &pc, 1328, dt, got);
            const e = worstOf(got, want_p, &pf_rows, dt);
            const ok = hit and e.bad == 0 and e.worst <= bar;
            std.debug.print("TURBOEXL3_HY4F_MOE n={d} rows=1328 path=prefill out={s} worst_vs_ref={e:.3} bar={e:.1} nonfinite={d} {s}\n", .{ n, @tagName(dt), e.worst, bar, e.bad, if (ok) "PASS" else "FAIL" });
            if (!ok) failed = true;
        }
    }
    try testing.expect(!failed);
}

fn add(h: ref.Hits, o: ref.Hits) ref.Hits {
    return .{ .gate = h.gate + o.gate, .up = h.up + o.up, .total = h.total + o.total };
}

/// DEC-ONE (1, 4), the GEMV chain (8), the register path (32), dequantize + matmul (512); the head at decode rows only.
const dense_rows = [_]u32{ 1, 4, 8, 32, 512 };
const head_rows = [_]u32{ 1, 4 };

/// The rows a case compares: every row up to 8; past that both ends and either side of the middle.
fn pickRows(rows: u32, buf: *[8]u32) []const u32 {
    if (rows <= 8) {
        for (buf[0..rows], 0..) |*r, i| r.* = @intCast(i);
        return buf[0..rows];
    }
    buf[0..6].* = .{ 0, 1, rows / 2 - 1, rows / 2, rows - 2, rows - 1 };
    return buf[0..6];
}

/// The worst ratio of `got`'s rows `pick` (rows of `out` values) against the reference rows `want`.
fn denseWorst(got: []const f32, want: []const f32, pick: []const u32, out: u32) f64 {
    var worst: f64 = 0;
    for (pick, 0..) |r, i| {
        const e = ref.compare(got[r * out ..][0..out], want[i * out ..][0..out]);
        worst = @max(worst, if (std.math.isFinite(e.max_abs)) e.max_ratio else std.math.inf(f64));
    }
    return worst;
}

fn denseCase(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, c: DenseCase) !usize {
    var prng = std.Random.DefaultPrng.init(0x4F4FDE ^ (@as(u64, c.in) << 24) ^ c.out);
    const rnd = prng.random();
    const rate = format.Rate.fromK(6);
    var lin = try ref.synthLinear(a, rnd, 1, c.in, c.out, rate, dec);
    defer lin.deinit(a);
    const mark = g.mark();
    defer g.resetTo(mark);
    const trellis = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(c.in / 16), ci(c.out / 16), ci(rate.n) }, .int16);
    const suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(c.in)}, .float16);
    const svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(c.out)}, .float16);
    const transformed = try a.alloc(f32, c.in);
    defer a.free(transformed);
    const inner = try a.alloc(f32, c.out);
    defer a.free(inner);
    const head = c.out == 120832;
    var failed: usize = 0;
    for (if (head) head_rows[0..] else dense_rows[0..]) |rows| {
        const rmark = g.mark();
        defer g.resetTo(rmark);
        const x = try a.alloc(f32, rows * c.in);
        defer a.free(x);
        for (x) |*v| v.* = bfExact(rnd.floatNorm(f32));
        var pick_buf: [8]u32 = undefined;
        const pick = pickRows(rows, &pick_buf);
        const want = try a.alloc(f32, pick.len * c.out);
        defer a.free(want);
        for (pick, 0..) |r, i| lin.row(0, x[r * c.in ..][0..c.in], transformed, inner, want[i * c.out ..][0..c.out]);
        const got = try a.alloc(f32, rows * c.out);
        defer a.free(got);
        const x3 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(c.in) }, .float32);
        for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dt| {
            if (dt == .bfloat16 and rows != 8 and rows != 512) continue;
            const y = try g.adopt(try root.linear(s, try g.astype(x3, dt), trellis, suh, svh, dec));
            try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(c.out) }, mlx.getShape(y));
            try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
            _ = try g.hostF32(try g.astype(y, .float32), got);
            const b: f64 = if (dt == .float32) 5e-3 else 2.5e-2;
            const worst = denseWorst(got, want, pick, c.out);
            const ok = worst <= b;
            std.debug.print("TURBOEXL3_HY4F_DENSE {s} {d}->{d} K6 rows={d} x={s} compared_rows={d} ratio={e:.3} bar={e:.1} {s}\n", .{ c.name, c.in, c.out, rows, @tagName(dt), pick.len, worst, b, if (ok) "PASS" else "FAIL" });
            failed += @intFromBool(!ok);
            if (c.out == 640 and rows == 4 and dt == .float32) {
                // the red check: the same trellis decoded as MUL1 must miss the bar
                var mul1 = lin;
                mul1.w = try a.alloc(u16, c.in * c.out);
                defer a.free(mul1.w);
                format.reconstructInner(lin.trellis, c.in, c.out, rate, .{ .codebook = .mul1 }, mul1.w);
                for (pick, 0..) |r, i| mul1.row(0, x[r * c.in ..][0..c.in], transformed, inner, want[i * c.out ..][0..c.out]);
                const red = denseWorst(got, want, pick, c.out);
                std.debug.print("TURBOEXL3_HY4F_DENSE_RED {s} rows=4 vs_mul1_reference={e:.3} bar={e:.1} {s}\n", .{ c.name, red, b, if (red > b) "PASS" else "FAIL" });
                failed += @intFromBool(red <= b);
            }
        }
    }
    return failed;
}

test "turboexl3 gpu hy4f dense: root.linear equals the host reference at every Hy4-flash trunk shape and the head, MCG K6, the pack's I16 trellis words; a MUL1 reference misses (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var failed: usize = 0;
    for (dense_cases) |c| {
        defer _ = mlx.mlx_clear_cache();
        failed += try denseCase(a, &g, s, c);
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

const config = @import("../../core/config.zig");

test "turboexl3 gpu hy4f routed tiles: form 9 at 4 output tiles per GEMV threadgroup from two rows (Config.decode_bt9 4) and the x texts (Config.decode_iop) give the served engine's bytes at decode rows 1 to 8, K1 to K4, on spread and on shared routing; a changed input does not, nor does the x texts' planted defect (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = config.get().*;
    defer config.configure(saved) catch unreachable;
    // bt9 4 alone; the x texts at bt9 4 (the served pair) and at the engine's tiles
    const Arm = struct { bt9: u8, iop: bool = false };
    const arms = [_]Arm{ .{ .bt9 = 4 }, .{ .bt9 = 4, .iop = true }, .{ .bt9 = 0, .iop = true } };
    var failed: usize = 0;
    for ([_]u32{ 16, 32, 48, 64 }) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n).?;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x4F50 ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, experts, width, width, rate);
        defer bank.deinit(a);
        const rb = try rootBank(&g, &bank);
        // spread routing (Call.make: each row 16 distinct experts of 512) and shared routing (each row 16 distinct of a
        // pool of 20, so most experts serve 5 to 8 rows and the leads take two members)
        var spread = try Call.make(a, rnd, 8);
        defer spread.deinit(a);
        var shared = try Call.make(a, rnd, 8);
        defer shared.deinit(a);
        for (0..8) |r| {
            const sl = shared.slots[r * top_k ..][0..top_k];
            for (sl, 0..) |*e, i| {
                while (true) {
                    e.* = 100 + rnd.uintLessThan(u32, 20);
                    if (std.mem.indexOfScalar(u32, sl[0..i], e.*) == null) break;
                }
            }
        }
        const base = try a.alloc(f32, 8 * width);
        defer a.free(base);
        const got = try a.alloc(f32, 8 * width);
        defer a.free(got);
        for ([_]*const Call{ &spread, &shared }, 0..) |call, ci_| {
            // the red input: row 0's first value moved
            var dr = try Call.make(a, rnd, 8);
            defer dr.deinit(a);
            @memcpy(dr.x, call.x);
            @memcpy(dr.slots, call.slots);
            @memcpy(dr.scores, call.scores);
            dr.x[0] = bfExact(dr.x[0] * 1.5 + 1.0);
            for (arms) |arm| {
                for (1..9) |rows_u| {
                    const rows: u32 = @intCast(rows_u);
                    root.releaseEngines();
                    try config.configure(saved);
                    try moeRows(&g, s, rb, call, rows, .bfloat16, base);
                    var cfg = saved;
                    cfg.decode_bt9 = arm.bt9;
                    cfg.decode_iop = arm.iop;
                    root.releaseEngines();
                    try config.configure(cfg);
                    try moeRows(&g, s, rb, call, rows, .bfloat16, got);
                    var diff: usize = 0;
                    for (base[0 .. rows * width], got[0 .. rows * width]) |x, y| diff += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
                    try moeRows(&g, s, rb, &dr, rows, .bfloat16, got);
                    var red: usize = 0;
                    for (base[0..width], got[0..width]) |x, y| red += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
                    const ok = diff == 0 and red > 0;
                    if (!ok or rows == 1 or rows == 6 or rows == 8) std.debug.print("TURBOEXL3_HY4F_ROUTED_TILES n={d} routing={s} bt9={d} iop={d} rows={d} vs_served={d}/{d} red_changed_input={d}/{d} {s}\n", .{ n, if (ci_ == 0) "spread" else "shared", arm.bt9, @intFromBool(arm.iop), rows, diff, rows * width, red, width, if (ok) "PASS" else "FAIL" });
                    failed += @intFromBool(!ok);
                }
            }
        }
        // the x texts' planted defect (decode.xHeader's red 1: the iop window one weight off) in the down GEMV, injected into
        // the cached engine, must move the served bytes at 6 rows on the spread routing (shared routing's two-member leads
        // too)
        const geo: decode.Geometry = .{ .hidden = width, .inter = width, .topk = top_k, .rate = rate, .dec = dec, .limit = limit, .clamp = .silu_out };
        for ([_]*const Call{ &spread, &shared }, 0..) |call, ci_| {
            const rows: u32 = 6;
            root.releaseEngines();
            try config.configure(saved);
            try moeRows(&g, s, rb, call, rows, .bfloat16, base);
            var cfg = saved;
            cfg.decode_bt9 = 4;
            cfg.decode_iop = true;
            root.releaseEngines();
            try config.configure(cfg);
            const k3 = try root.kernelsFor(geo, .bfloat16);
            const bad = ks.newKernel(decode.xName("down3", 1), &decode.down3_inputs, &decode.down3_outputs, decode.text_down3x, decode.xHeader(1), true) orelse return error.KernelCreateFailed;
            decode.freeKernel(k3.down);
            k3.down = bad;
            try moeRows(&g, s, rb, call, rows, .bfloat16, got);
            var diff: usize = 0;
            for (base[0 .. rows * width], got[0 .. rows * width]) |x, y| diff += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
            const ok = diff > 0;
            std.debug.print("TURBOEXL3_HY4F_ROUTED_TILES n={d} routing={s} red_iop rows={d} planted_vs_served={d}/{d} {s}\n", .{ n, if (ci_ == 0) "spread" else "shared", rows, diff, rows * width, if (ok) "PASS" else "FAIL" });
            failed += @intFromBool(!ok);
        }
        root.releaseEngines();
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

test "turboexl3 gpu hy4f dense verify rows: the chain at 2 output tiles (Config.chain_bt 2) gives the three-launch chain's bytes at every Hy4 trunk shape, K6 at rows 5 to 8 and K4 (the 2.08bpw pack's trunk) at the verify's 6 rows, bf16 and f32 x; a changed input does not; the lean chunks on the matrix units (Config.dense_mma) stay within the f32 summation bound against float64 beside today's chain, a planted lane map, dropped row and doubled K part miss it (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = config.get().*;
    defer config.configure(saved) catch unreachable;
    var tiles2 = saved;
    tiles2.chain_bt = 2;
    var mma = saved;
    mma.dense_mma = true;
    // the MMA GEMV's planted faults the bar must catch (`mma_red`)
    var faults: [mma_red.len]mlx.mlx_fast_metal_kernel = undefined;
    var made: usize = 0;
    defer for (faults[0..made]) |k| decode.freeKernel(k);
    inline for (mma_red, 0..) |f, i| {
        faults[i] = ks.newKernel(f.kernel, &dense.dgemv_in, &dense.gemv_out, comptime dense.dgemvMmaText(f.lane, f.live, f.store), dense.header_mma, true) orelse return error.KernelCreateFailed;
        made = i + 1;
    }
    // the rates and row counts: K6 (the 2.25bpw pack's trunk) at rows 5 to 8, K4 (the 2.08bpw pack's) at the verify's 6
    const Rates = struct { k: u32, rows: []const u32 };
    const rates = [_]Rates{ .{ .k = 6, .rows = &.{ 5, 6, 7, 8 } }, .{ .k = 4, .rows = &.{6} } };
    var failed: usize = 0;
    for (rates) |rt| for (dense_cases) |c| {
        if (c.out == 120832) continue;
        defer _ = mlx.mlx_clear_cache();
        // K6's draws as before; K4's its own
        var prng = std.Random.DefaultPrng.init(0x4F51 ^ (@as(u64, c.in) << 24) ^ c.out ^ (if (rt.k == 6) 0 else @as(u64, rt.k) << 56));
        const rnd = prng.random();
        const rate = format.Rate.fromK(rt.k);
        var lin = try ref.synthLinear(a, rnd, 1, c.in, c.out, rate, dec);
        defer lin.deinit(a);
        const mark = g.mark();
        defer g.resetTo(mark);
        const trellis = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(c.in / 16), ci(c.out / 16), ci(rate.n) }, .int16);
        const suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(c.in)}, .float16);
        const svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(c.out)}, .float16);
        for (rt.rows) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const x = try a.alloc(f32, rows * c.in);
            defer a.free(x);
            for (x) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const xr = try a.dupe(f32, x);
            defer a.free(xr);
            xr[c.in / 2] = bfExact(xr[c.in / 2] * 1.5 + 0.25);
            const got = try a.alloc(f32, 4 * rows * c.out);
            defer a.free(got);
            const n = rows * c.out;
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const x0 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ ci(rows), ci(c.in) }, .float32), dt);
                const x1 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xr), &.{ ci(rows), ci(c.in) }, .float32), dt);
                const runs = [_]struct { cfg: config.Config, x: mlx.mlx_array }{ .{ .cfg = saved, .x = x0 }, .{ .cfg = tiles2, .x = x0 }, .{ .cfg = tiles2, .x = x1 }, .{ .cfg = mma, .x = x0 } };
                for (runs, 0..) |r, k| {
                    root.releaseEngines();
                    try config.configure(r.cfg);
                    const y = try g.adopt(try root.linear(s, r.x, trellis, suh, svh, dec));
                    _ = try g.hostF32(try g.astype(y, .float32), got[k * n ..][0..n]);
                }
                var d_t2: usize = 0;
                var d_red: usize = 0;
                var d_mma: usize = 0;
                for (got[0..n], got[n..][0..n], got[2 * n ..][0..n], got[3 * n ..][0..n]) |o, t2, rr, m2| {
                    d_t2 += @intFromBool(@as(u32, @bitCast(t2)) != @as(u32, @bitCast(o)));
                    d_red += @intFromBool(@as(u32, @bitCast(rr)) != @as(u32, @bitCast(o)));
                    d_mma += @intFromBool(@as(u32, @bitCast(m2)) != @as(u32, @bitCast(o)));
                }
                // the MMA GEMV is reorder-class: its words against the chain's are counted, not held
                const ok = d_t2 == 0 and d_red > 0;
                std.debug.print("TURBOEXL3_HY4F_VERIFY_ROWS {s} in={d} out={d} K={d} rows={d} x={s} chainbt2_vs_chain={d}/{d} red_changed_input={d}/{d} mma_vs_chain={d}/{d} {s}\n", .{ c.name, c.in, c.out, rt.k, rows, @tagName(dt), d_t2, n, d_red, n, d_mma, n, if (ok) "PASS" else "FAIL" });
                failed += @intFromBool(!ok);
                // the float64 bar on the K-slice planes (the verify's 6 rows in f32 also run the planted faults)
                const red = rows == 6 and dt == .float32;
                failed += try mmaBar(a, &g, s, c, rt.k, rows, dt, x0, trellis, suh, saved, mma, if (red) faults[0..] else null);
            }
        }
    };
    root.releaseEngines();
    try testing.expectEqual(@as(usize, 0), failed);
}

/// The planted faults of text_dgemv_mma (`dense.dgemvMmaText` with one served piece replaced): each must miss the bar
const MmaFault = struct { name: []const u8, kernel: [*:0]const u8, lane: []const u8, live: []const u8, store: []const u8 };
const mma_red = [_]MmaFault{
    // the B^T fragment filled from trellis lane L instead of the lane plan's L'
    .{ .name = "lane_map", .kernel = "turboexl3_test_dgemv_mma_lane_fault", .lane = @embedFile("kernels/test/fault/mma_lane_natural.metal"), .live = dense.mma_live, .store = dense.dgemv_store },
    // fragment row 2 multiplying zeros: row 2 of each chunk dropped
    .{ .name = "row_dropped", .kernel = "turboexl3_test_dgemv_mma_row_fault", .lane = dense.mma_lane, .live = @embedFile("kernels/test/fault/mma_live_drop_row.metal"), .store = dense.dgemv_store },
    // simdgroup 1's part of each K slice summed twice into the plane
    .{ .name = "k_part_twice", .kernel = "turboexl3_test_dgemv_mma_twice_fault", .lane = dense.mma_lane, .live = dense.mma_live, .store = @embedFile("kernels/test/fault/mma_store_twice.metal") },
};

/// The worst ratio of |z - ref| to its bound over the sampled outputs, and the error's rms over the rms of the bound's
/// sums. z f32 [KS rows, out] (the K-slice planes, summed here in slice order in f32 as text_dfinish sums them); want and
/// mag f64 [rows, cols.len] (the float64 inner product and the sum of |products| of each sampled output).
const Bar = struct { ratio: f64, rel_rms: f64 };
fn barOf(z: []const f32, ksn: u32, rows: u32, out: u32, cols: []const u32, want: []const f64, mag: []const f64, gamma: f64) Bar {
    var worst: f64 = 0;
    var se: f64 = 0;
    var sm: f64 = 0;
    for (0..rows) |r| for (cols, 0..) |col, j| {
        var acc: f32 = z[r * out + col];
        for (1..ksn) |kk| acc += z[(kk * rows + r) * out + col];
        const i = r * cols.len + j;
        if (!std.math.isFinite(acc)) {
            worst = std.math.inf(f64);
            continue;
        }
        const e = @abs(@as(f64, acc) - want[i]);
        const b = gamma * mag[i];
        worst = @max(worst, if (b > 0) e / b else if (e > 0) std.math.inf(f64) else 0);
        se += e * e;
        sm += mag[i] * mag[i];
    };
    return .{ .ratio = worst, .rel_rms = if (sm > 0) @sqrt(se / sm) else 0 };
}

/// The lean chunks on the matrix units (`mma`'s configuration, `Dense.prepareGemv`) and today's chain (`today`) at one
/// shape, rate, row count and x dtype: each K-slice plane set z, summed in slice order in f32, against the float64 inner
/// product of the very operands the GEMV reads (prep_s's lane-order rows read back and put in natural order, the dequantize
/// launch's W), at three 128-column blocks (the first, middle and last: every lane, tile pair and fragment column). The bar
/// of an output is the summation bound of n = in exact products (half x half is exact in f32), any order, f32 round to
/// nearest: gamma_{n-1} sum_k |x_k w_k|, gamma_m = m u / (1 - m u), u = 2^-24. Both must hold it; each planted fault in
/// `faults` (swapped into the Dense's MMA kernel) must miss it. The MMA's rms gap against today's and the planes' words
/// against today's are printed. Returns the failures.
fn mmaBar(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, c: DenseCase, k: u32, rows: u32, dt: mlx.mlx_dtype, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, today: config.Config, mma: config.Config, faults: ?[]const mlx.mlx_fast_metal_kernel) !usize {
    const shape: dense.Shape = .{ .in = c.in, .out = c.out, .rate = format.Rate.fromK(k), .dec = dec, .out_dtype = dt };
    const m = g.mark();
    defer g.resetTo(m);
    const nob = c.out / 128;
    const blocks = [3]u32{ 0, nob / 2, nob - 1 };
    var cols: [3 * 128]u32 = undefined;
    for (blocks, 0..) |b, i| for (0..128) |j| {
        cols[i * 128 + j] = b * 128 + @as(u32, @intCast(j));
    };
    const nc = cols.len;
    root.releaseEngines();
    try config.configure(mma);
    const d = try root.denseCached(shape);
    const cm = try d.cfgs(rows);
    const ksn = d.ks_split;
    const nz = ksn * rows * c.out;
    // the GEMV's operands: prep_s's lane-order rows at the chain's config (the launch prepareGemv makes) and W
    var po: [2]mlx.mlx_array = undefined;
    try ks.applyConfig(d.prep_s, cm.prep, &.{ x, suh }, &po, s);
    const xp = try g.adopt(po[0]);
    _ = try g.adopt(po[1]);
    const xl = try a.alloc(f32, rows * c.in);
    defer a.free(xl);
    _ = try g.hostF32(try g.astype(xp, .float32), xl);
    const W = try g.adopt(try d.decodeTrellis(s, trellis));
    const wc = try a.alloc(f32, c.in * nc);
    defer a.free(wc);
    const wb = try a.alloc(f32, c.in * 128);
    defer a.free(wb);
    for (blocks, 0..) |b, i| {
        const v = try g.slice(W, &.{ 0, ci(b * 128) }, &.{ ci(c.in), ci(b * 128 + 128) }, &.{ 1, 1 });
        _ = try g.hostF32(try g.astype(v, .float32), wb);
        for (0..c.in) |kk| @memcpy(wc[kk * nc + i * 128 ..][0..128], wb[kk * 128 ..][0..128]);
    }
    // the float64 reference and the bound's sums: element kk of a 16-block of a lane-order row sits at slot alayPos16(kk)
    const want = try a.alloc(f64, rows * nc);
    defer a.free(want);
    const mag = try a.alloc(f64, rows * nc);
    defer a.free(mag);
    @memset(want, 0);
    @memset(mag, 0);
    for (0..rows) |r| for (0..c.in) |kk| {
        const xv: f64 = xl[r * c.in + (kk & ~@as(usize, 15)) + prefill.alayPos16(@intCast(kk & 15))];
        if (xv == 0) continue;
        const wr = wc[kk * nc ..][0..nc];
        const wo = want[r * nc ..][0..nc];
        const mo = mag[r * nc ..][0..nc];
        for (wr, wo, mo) |wv, *o, *mm| {
            const p = xv * @as(f64, wv);
            o.* += p;
            mm.* += @abs(p);
        }
    };
    const u = std.math.pow(f64, 2, -24);
    const nt: f64 = @floatFromInt(c.in - 1);
    const gamma = nt * u / (1 - nt * u);
    const zm = try a.alloc(f32, nz);
    defer a.free(zm);
    const zf = try a.alloc(f32, nz);
    defer a.free(zf);
    {
        const zz = try d.prepareGemv(s, x, trellis, suh, rows);
        _ = try g.adopt(zz[1]);
        _ = try g.hostF32(try g.adopt(zz[0]), zm);
    }
    const bm = barOf(zm, ksn, rows, c.out, &cols, want, mag, gamma);
    var failed: usize = 0;
    if (faults) |fs| {
        var red: [mma_red.len]f64 = undefined;
        for (fs, 0..) |fk, i| {
            const kept = d.dgemv_mma;
            d.dgemv_mma = fk;
            defer d.dgemv_mma = kept;
            const zz = try d.prepareGemv(s, x, trellis, suh, rows);
            _ = try g.adopt(zz[1]);
            _ = try g.hostF32(try g.adopt(zz[0]), zf);
            red[i] = barOf(zf, ksn, rows, c.out, &cols, want, mag, gamma).ratio;
        }
        var ok = cm.mma;
        for (red) |r| ok = ok and r > 1;
        std.debug.print("TURBOEXL3_HY4F_MMA_RED {s} in={d} out={d} K={d} rows={d} x={s} on_mma={} {s}={e:.3} {s}={e:.3} {s}={e:.3} (each must exceed 1) {s}\n", .{ c.name, c.in, c.out, k, rows, @tagName(dt), cm.mma, mma_red[0].name, red[0], mma_red[1].name, red[1], mma_red[2].name, red[2], if (ok) "PASS" else "FAIL" });
        failed += @intFromBool(!ok);
    }
    // today's chain (the engine `d` belonged to is freed here)
    root.releaseEngines();
    try config.configure(today);
    const d0 = try root.denseCached(shape);
    const c0 = try d0.cfgs(rows);
    {
        const zz = try d0.prepareGemv(s, x, trellis, suh, rows);
        _ = try g.adopt(zz[1]);
        _ = try g.hostF32(try g.adopt(zz[0]), zf);
    }
    const bt = barOf(zf, ksn, rows, c.out, &cols, want, mag, gamma);
    var zdiff: usize = 0;
    for (zm, zf) |p, q| zdiff += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
    // both within the f32 summation bound (exact products, f32 accumulation; the gaps
    // against today's are printed, rms_vs_today, not held)
    const ok = bm.ratio <= 1 and bt.ratio <= 1;
    std.debug.print("TURBOEXL3_HY4F_MMA_BAR {s} in={d} out={d} K={d} rows={d} x={s} on_mma={} today_lean={} ks={d} bar=gamma_{d}*sum|p| mma_worst={e:.3} today_worst={e:.3} mma_rel_rms={e:.3} today_rel_rms={e:.3} rms_vs_today={d:.3} z_words_vs_today={d}/{d} {s}\n", .{ c.name, c.in, c.out, k, rows, @tagName(dt), cm.mma, c0.lean, ksn, c.in - 1, bm.ratio, bt.ratio, bm.rel_rms, bt.rel_rms, if (bt.rel_rms > 0) bm.rel_rms / bt.rel_rms else 0, zdiff, nz, if (ok) "PASS" else "FAIL" });
    failed += @intFromBool(!ok);
    return failed;
}

// ── the gate's finish (`linearGateMul`, gate.zig): `gate_fused`'s call ──

// ── pg2arms: pg2 (core/pgemm.zig, the host's switch) under the arms and at the plain linears ──

const pgemm = @import("../../core/pgemm.zig");

/// The pg2 settings a pg2arms case runs under beside today's (pg2 off): on with the host's table naming another shape
/// only (the call's launches without pg2: the arm stays on), then on at each of `pgemm.candidates` for the case's shape
const Pg2Set = union(enum) { other, tile: usize };
const pg2_sets = blk: {
    var r: [1 + pgemm.candidates.len]Pg2Set = undefined;
    r[0] = .other;
    for (0..pgemm.candidates.len) |i| r[1 + i] = .{ .tile = i };
    break :blk r;
};

const Pg2Saved = struct { mode: pgemm.Mode, table: ?[]const pgemm.Entry, split: bool };

fn pg2Save() Pg2Saved {
    return .{ .mode = pgemm.mode.get().*, .table = pgemm.host_table.get().*, .split = pgemm.split_k.get().* };
}

fn pg2Restore(v: Pg2Saved) void {
    pgemm.mode.get().* = v.mode;
    pgemm.host_table.get().* = v.table;
    pgemm.split_k.get().* = v.split;
}

/// pg2 on (split K too) with the host's table `entry` for `set` at (k, n)
fn pg2Apply(set: Pg2Set, k: u32, n: u32, entry: *[1]pgemm.Entry) void {
    entry.* = switch (set) {
        .other => .{.{ .k = k, .n = n + 128, .tile = pgemm.candidates[0] }},
        .tile => |i| .{.{ .k = k, .n = n, .tile = pgemm.candidates[i] }},
    };
    pgemm.mode.get().* = .on;
    pgemm.split_k.get().* = true;
    pgemm.host_table.get().* = entry;
}

/// Whether a call at `rows` on `path` runs pg2's launch under `set` at (k, n): a tile whose N divides n, on `linear`'s
/// matmul path, where MLX runs its regular kernel (`regular_only`: an arm) or splits K at a device-independent size
fn pg2Fuses(set: Pg2Set, path: dense.Dense.Path, rows: u32, k: u32, n: u32, regular_only: bool) bool {
    const i = switch (set) {
        .other => return false,
        .tile => |ti| ti,
    };
    if (path != .matmul or n % pgemm.candidates[i].bn != 0) return false;
    if (pgemm.mlxRegular(rows, n, k)) return true;
    return !regular_only and pgemm.splitKAt(rows, n, k) != null;
}

/// A tile's name in a line: BMxBNxBK_WMxWN_sSWZ
fn tileName(buf: []u8, set: Pg2Set) []const u8 {
    return switch (set) {
        .other => "other",
        .tile => |i| blk: {
            const t = pgemm.candidates[i];
            break :blk std.fmt.bufPrint(buf, "{d}x{d}x{d}_{d}x{d}_s{d}", .{ t.bm, t.bn, t.bk, t.wm, t.wn, t.swz }) catch "?";
        },
    };
}

/// The red checks' pg2 tile (`pgemm.table`'s most common: 32 x 256 x 512, 1 x 4 simdgroups, swizzle 2)
const pg2_red_set: Pg2Set = .{ .tile = 1 };

const gate = @import("gate.zig");

test "turboexl3 hy4f gatemul texts: the gate's finish is finish_v, text_finish's store opening with the value kept, MLX's Sigmoid at f32, o read by index, the product stored" {
    try testing.expectEqualStrings(dense.finish_v ++ gate.gate_open ++ gate.gate_sig ++ gate.gate_src ++ gate.gate_close, gate.text_finish_gate);
    const pin = comptime support.pins(@embedFile("kernels/test/pins/gate_texts.metal"));
    // the gate's value and element: text_finish's store expression and index, once each in its store and the opening
    for (pin[0..2]) |line| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, dense.finish_store, line));
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, gate.gate_open, line));
    }
    // MLX's Sigmoid (unary_ops.h at T = float), o's element in the layout O_FLAT names (its rows' [1, S, H, VH] at 1,
    // its [1, H, S, VH] at 0), the product in G_T
    for (pin[2..4]) |line| try testing.expectEqual(@as(usize, 1), std.mem.count(u8, gate.gate_sig, line));
    for ([_][]const u8{ pin[4], pin[6] }) |line| try testing.expectEqual(@as(usize, 1), std.mem.count(u8, gate.gate_src, line));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, gate.gate_close, pin[5]));
    // the operands by name, no shape buffers
    for ([_][]const u8{ "o_src_shape", "o_src_strides", "z_shape", "svh_shape" }) |n| try testing.expectEqual(@as(usize, 0), std.mem.count(u8, gate.text_finish_gate, n));
}

test "turboexl3 hy4f gatemul refusals: decode rows (1 to 16) refuse with Exl3EngineNotImplemented, and wrong dtypes and shapes by name, before any engine is made" {
    // no launch: every refusal comes before the stream is used
    const s: mlx.mlx_stream = .{ .ctx = null };
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const in: c_int = 128;
    const out: c_int = 256;
    const zeros = try a.alloc(u32, 33 * 256 * 2);
    defer a.free(zeros);
    @memset(zeros, 0);
    const bytes = std.mem.sliceAsBytes(zeros);
    const p: root.Proj = .{
        .trellis = try g.hostArray(bytes[0 .. (128 / 16) * (256 / 16) * 96 * 2], &.{ 8, 16, 96 }, .int16),
        .suh = try g.hostArray(bytes[0 .. 128 * 2], &.{in}, .float16),
        .svh = try g.hostArray(bytes[0 .. 256 * 2], &.{out}, .float16),
    };
    const held = root.enginesHeld().dense;
    const Case = struct { rows: c_int, xd: mlx.mlx_dtype = .float32, od: mlx.mlx_dtype = .float32, gd: mlx.mlx_dtype = .bfloat16, oh: c_int = 2, os: c_int = 0, ov: c_int = 128, layout: gate.OLayout = .heads, want: anyerror };
    const cases = [_]Case{
        .{ .rows = 1, .want = error.Exl3EngineNotImplemented },
        .{ .rows = 4, .want = error.Exl3EngineNotImplemented },
        .{ .rows = 16, .want = error.Exl3EngineNotImplemented },
        .{ .rows = 33, .gd = .int32, .want = error.BadExl3Dtype },
        .{ .rows = 33, .xd = .int32, .want = error.BadExl3Dtype },
        .{ .rows = 33, .od = .bfloat16, .want = error.BadExl3Dtype },
        .{ .rows = 33, .os = 32, .want = error.BadExl3Shape },
        .{ .rows = 33, .ov = 64, .want = error.BadExl3Shape },
        // the flat layout [1, rows, H, VH]: its rows on axis 1, its heads on axis 2
        .{ .rows = 4, .layout = .flat, .want = error.Exl3EngineNotImplemented },
        .{ .rows = 33, .os = 32, .layout = .flat, .want = error.BadExl3Shape },
        .{ .rows = 33, .ov = 64, .layout = .flat, .want = error.BadExl3Shape },
        .{ .rows = 33, .od = .bfloat16, .layout = .flat, .want = error.BadExl3Dtype },
    };
    for (cases) |c| {
        const m = g.mark();
        defer g.resetTo(m);
        const x = try g.hostArray(bytes[0..@intCast(c.rows * in * 4)], &.{ 1, c.rows, in }, c.xd);
        const os = if (c.os != 0) c.os else c.rows;
        const osh: [4]c_int = if (c.layout == .flat) .{ 1, os, c.oh, c.ov } else .{ 1, c.oh, os, c.ov };
        const o_src = try g.hostArray(bytes[0..@intCast(c.oh * os * c.ov * 4)], &osh, c.od);
        try testing.expectError(c.want, gate.linearGateMulAt(s, x, p, o_src, c.layout, dec, c.gd));
    }
    try testing.expectEqual(held, root.enginesHeld().dense);
}

/// linear_gate at the indexed layers (3072 -> 32 heads of 256) and the sliding ones (64 heads), K6 MCG
const GateCase = struct { name: []const u8, heads: u32 };
const gate_cases = [_]GateCase{ .{ .name = "linear_gate.indexed", .heads = 32 }, .{ .name = "linear_gate.sliding", .heads = 64 } };
const gate_vh: u32 = 256;
const gate_in: u32 = 3072;
/// the register path (17 to 32 rows, with `Config.nax`), the matmul path's first row count, a prompt's ragged tail
/// chunk and two prompt chunks
const gate_rows = [_]u32{ 20, 33, 1024, 3125, 3584 };
/// the red checks' row count (16.7M words at the sliding width)
const gate_red_rows: u32 = 1024;

/// A projection [in -> out] of random K6 trellis words as the loader hands them (I16), its svh scaled so the outputs
/// reach about +-8 (both of the sigmoid's sides, deep into each)
fn gateProj(a: std.mem.Allocator, g: *G, rnd: std.Random, in: u32, out: u32) !root.Proj {
    const n: u32 = 96;
    const t = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(t);
    rnd.bytes(std.mem.sliceAsBytes(t));
    const su = try a.alloc(u16, in);
    defer a.free(su);
    for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, out);
    defer a.free(sv);
    for (sv) |*v| v.* = format.f32ToF16Bits(4 * (0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    return .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(t), &.{ ci(in / 16), ci(out / 16), ci(n) }, .int16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(out)}, .float16),
    };
}

/// The chain `gate_fused` replaces, op for op (hy4f src/model.zig `attention`): the gate's `linear` in x's dtype,
/// its f32 copy (none for an f32 x), MLX's sigmoid, o's transpose to [1, S, H VH], MLX's multiply, the cast to `gdt`
fn gateChain(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, o_src: mlx.mlx_array, rows: u32, out: u32, gdt: mlx.mlx_dtype) ![2]mlx.mlx_array {
    const y = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, dec));
    const yf = if (mlx.mlx_array_dtype(y) == .float32) y else try g.astype(y, .float32);
    const flat = try g.reshape(try g.transposeAxes(o_src, &.{ 0, 2, 1, 3 }), &.{ 1, ci(rows), ci(out) });
    return .{ try g.astype(try g.mul(flat, try g.sigmoid(yf)), gdt), yf };
}

/// o's heads layout `oh` [1, H, rows, VH] rewritten in its rows' layout [1, rows, H, VH] (the host's kv_b kernel's): the
/// same f32 values, element h VH + d of a row at h VH + d; the caller frees it
fn gateRows(a: std.mem.Allocator, oh: []const f32, heads: u32, rows: u32) ![]f32 {
    const out = try a.alloc(f32, oh.len);
    for (0..heads) |h| for (0..rows) |r| {
        @memcpy(out[(r * heads + h) * gate_vh ..][0..gate_vh], oh[(h * rows + r) * gate_vh ..][0..gate_vh]);
    };
    return out;
}

/// How a gate test hands o to `linearGateMulAt`: in its heads layout (contiguous), as the transposed view of its rows
/// (the heads layout the host's kv_b kernel gives today: not row-contiguous, so the launch copies it), or as the rows in
/// the flat layout (read where they lie)
const GateArm = enum { heads, view, flat };
const gate_arms = [_]GateArm{ .heads, .view, .flat };

/// o for `arm` from its heads layout `o_heads` and its rows `o_rows`, and the layout the call names
fn gateArm(g: *G, arm: GateArm, o_heads: mlx.mlx_array, o_rows: mlx.mlx_array) !struct { o: mlx.mlx_array, layout: gate.OLayout } {
    return switch (arm) {
        .heads => .{ .o = o_heads, .layout = .heads },
        .view => .{ .o = try g.transposeAxes(o_rows, &.{ 0, 2, 1, 3 }), .layout = .heads },
        .flat => .{ .o = o_rows, .layout = .flat },
    };
}

/// The words of `got` and `want` (n values, read as f32: a bf16's f32 is exact) that differ, and `want`'s nonfinite
fn gateWords(got: []const f32, want: []const f32) struct { differ: usize, bad: usize } {
    var d: usize = 0;
    var bad: usize = 0;
    for (got, want) |u, w| {
        d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
        bad += @intFromBool(!std.math.isFinite(w));
    }
    return .{ .differ = d, .bad = bad };
}

test "turboexl3 gpu hy4f gatemul pg2arms: linearGateMul gives the chain's words (linear, f32 copy, MLX's sigmoid, o's transpose, multiply, cast) at linear_gate's two widths, MCG K6, rows 20 to 3584, bf16 and f32, with o in its heads layout, as the transposed view of its rows and as the rows (flat), pg2 off and on (another shape tabled; each candidate tile: pg2's GEMM with the gate's finish); planted faults move them in both layouts and in pg2's launch (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer gate.unplant();
    const pg2_saved = pg2Save();
    defer pg2Restore(pg2_saved);
    var entry: [1]pgemm.Entry = undefined;
    var tbuf: [32]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(0x4F4F6A7E);
    const rnd = prng.random();
    var lines: usize = 0;
    var failed: usize = 0;
    var pg2_lines: usize = 0;
    var pg2_fused: usize = 0;
    for (gate_cases) |c| {
        defer _ = mlx.mlx_clear_cache();
        const cm = g.mark();
        defer g.resetTo(cm);
        const out = c.heads * gate_vh;
        const p = try gateProj(a, &g, rnd, gate_in, out);
        for (gate_rows) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const n: usize = @as(usize, rows) * out;
            const xh = try a.alloc(f32, @as(usize, rows) * gate_in);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const oh = try a.alloc(f32, n);
            defer a.free(oh);
            for (oh) |*v| v.* = rnd.floatNorm(f32);
            const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ 1, ci(rows), ci(gate_in) }, .float32);
            const o_src = try g.hostArray(std.mem.sliceAsBytes(oh), &.{ 1, ci(c.heads), ci(rows), ci(gate_vh) }, .float32);
            const orh = try gateRows(a, oh, c.heads, rows);
            defer a.free(orh);
            const o_rows = try g.hostArray(std.mem.sliceAsBytes(orh), &.{ 1, ci(rows), ci(c.heads), ci(gate_vh) }, .float32);
            const got = try a.alloc(f32, n);
            defer a.free(got);
            const want = try a.alloc(f32, n);
            defer a.free(want);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const dm = g.mark();
                defer g.resetTo(dm);
                const x = try g.astype(x32, dt);
                const path = (try root.denseCached(.{ .in = gate_in, .out = out, .rate = format.Rate.fromK(6), .dec = dec, .out_dtype = dt })).pathFor(rows);
                const ch = try gateChain(&g, s, x, p, o_src, rows, out, dt);
                // the gate's values on each side of the sigmoid
                _ = try g.hostF32(ch[1], want);
                var neg: usize = 0;
                for (want) |v| neg += @intFromBool(v < 0);
                _ = try g.hostF32(try g.astype(ch[0], .float32), want);
                for (gate_arms) |arm| {
                    const am = g.mark();
                    defer g.resetTo(am);
                    const oa = try gateArm(&g, arm, o_src, o_rows);
                    const y = try g.adopt(try gate.linearGateMulAt(s, x, p, oa.o, oa.layout, dec, dt));
                    try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(out) }, mlx.getShape(y));
                    try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
                    _ = try g.hostF32(try g.astype(y, .float32), got);
                    const w = gateWords(got, want);
                    const ok = w.differ == 0 and w.bad == 0 and neg > 0 and neg < n;
                    lines += 1;
                    failed += @intFromBool(!ok);
                    std.debug.print("TURBOEXL3_HY4F_GATEMUL {s} {d}->{d} heads={d} vh={d} K6 rows={d} path={s} o={s} x={s} gated={s} words={d} differing={d} nonfinite={d} gate_negative={d} {s}\n", .{ c.name, gate_in, out, c.heads, gate_vh, rows, @tagName(path), @tagName(arm), @tagName(dt), @tagName(dt), n, w.differ, w.bad, neg, if (ok) "PASS" else "FAIL" });
                }
                // pg2 on: another shape tabled (the gate's finish stays on), then each candidate tile at this shape
                // (o flat, the served layout; o in its heads layout at the red checks' tile too)
                for (pg2_sets) |set| for ([_]GateArm{ .flat, .heads }) |arm| {
                    if (arm == .heads and !std.meta.eql(set, pg2_red_set)) continue;
                    const am = g.mark();
                    defer g.resetTo(am);
                    const oa = try gateArm(&g, arm, o_src, o_rows);
                    pg2Apply(set, gate_in, out, &entry);
                    const before = pgemm.calls.get().*.load(.monotonic);
                    const r = gate.linearGateMulAt(s, x, p, oa.o, oa.layout, dec, dt);
                    const fused = pgemm.calls.get().*.load(.monotonic) > before;
                    pg2Restore(pg2_saved);
                    const y = try g.adopt(try r);
                    _ = try g.hostF32(try g.astype(y, .float32), got);
                    const w = gateWords(got, want);
                    const want_fused = pg2Fuses(set, path, rows, gate_in, out, true);
                    const ok = w.differ == 0 and w.bad == 0 and fused == want_fused;
                    pg2_lines += 1;
                    pg2_fused += @intFromBool(fused);
                    failed += @intFromBool(!ok);
                    std.debug.print("TURBOEXL3_HY4F_GATEMUL_PG2 {s} {d}->{d} rows={d} path={s} o={s} x={s} tile={s} fused={} (want {}) words={d} differing={d} nonfinite={d} {s}\n", .{ c.name, gate_in, out, rows, @tagName(path), @tagName(arm), @tagName(dt), tileName(&tbuf, set), fused, want_fused, n, w.differ, w.bad, if (ok) "PASS" else "FAIL" });
                };
                if (rows != gate_red_rows or c.heads != 64) continue;
                // the red checks in each layout: each planted fault's words against the chain's must differ
                for ([_]GateArm{ .heads, .flat }) |arm| for ([_]gate.Fault{ .o_layout, .sig_fast_exp, .sig_one_sided }) |f| {
                    const am = g.mark();
                    defer g.resetTo(am);
                    const oa = try gateArm(&g, arm, o_src, o_rows);
                    try gate.plant(f);
                    const red = gate.linearGateMulAt(s, x, p, oa.o, oa.layout, dec, dt);
                    gate.unplant();
                    const ry = try g.adopt(try red);
                    _ = try g.hostF32(try g.astype(ry, .float32), got);
                    const rw = gateWords(got, want);
                    const caught = rw.differ > 0;
                    lines += 1;
                    failed += @intFromBool(!caught);
                    std.debug.print("TURBOEXL3_HY4F_GATEMUL_RED {s} rows={d} o={s} x={s} fault={s} differing={d}/{d} (want > 0) {s}\n", .{ c.name, rows, @tagName(arm), @tagName(dt), @tagName(f), rw.differ, n, if (caught) "PASS" else "FAIL" });
                };
                // the red checks in pg2's launch: each planted fault in its store loop must move the words too
                for ([_]GateArm{ .heads, .flat }) |arm| for ([_]gate.Fault{ .o_layout, .sig_fast_exp, .sig_one_sided }) |f| {
                    const am = g.mark();
                    defer g.resetTo(am);
                    const oa = try gateArm(&g, arm, o_src, o_rows);
                    pg2Apply(pg2_red_set, gate_in, out, &entry);
                    try gate.plant(f);
                    const before = pgemm.calls.get().*.load(.monotonic);
                    const red = gate.linearGateMulAt(s, x, p, oa.o, oa.layout, dec, dt);
                    const fused = pgemm.calls.get().*.load(.monotonic) > before;
                    gate.unplant();
                    pg2Restore(pg2_saved);
                    const ry = try g.adopt(try red);
                    _ = try g.hostF32(try g.astype(ry, .float32), got);
                    const rw = gateWords(got, want);
                    const caught = rw.differ > 0 and fused;
                    pg2_lines += 1;
                    failed += @intFromBool(!caught);
                    std.debug.print("TURBOEXL3_HY4F_GATEMUL_PG2_RED {s} rows={d} o={s} x={s} tile={s} fault={s} fused={} differing={d}/{d} (want > 0) {s}\n", .{ c.name, rows, @tagName(arm), @tagName(dt), tileName(&tbuf, pg2_red_set), @tagName(f), fused, rw.differ, n, if (caught) "PASS" else "FAIL" });
                };
            }
        }
    }
    std.debug.print("TURBOEXL3_HY4F_GATEMUL_DONE lines={d} pg2_lines={d} pg2_fused={d} failed={d} {s}\n", .{ lines, pg2_lines, pg2_fused, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, gate_cases.len * gate_rows.len * 2 * gate_arms.len + 2 * 3 * 2), lines);
    try testing.expectEqual(@as(usize, gate_cases.len * gate_rows.len * 2 * (pg2_sets.len + 1) + 2 * 3 * 2), pg2_lines);
    try testing.expect(pg2_fused > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── the SwiGLU in up's finish (`linearSwigluMul`, swiglu.zig): `swiglu_fused`'s call ──

const swiglu = @import("swiglu.zig");

test "turboexl3 hy4f swiglumul texts: up's finish is finish_v, text_finish's store opening with up's value kept and the gate's read, gate.zig's MLX Sigmoid, swigluOut's two products stored" {
    try testing.expectEqualStrings(dense.finish_v ++ swiglu.swiglu_open ++ gate.gate_sig ++ swiglu.swiglu_close, swiglu.text_finish_swiglu);
    const pin = comptime support.pins(@embedFile("kernels/test/pins/swiglu_texts.metal"));
    // up's value and element: text_finish's store expression and index, once each in its store and the opening
    for (pin[0..2]) |line| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, dense.finish_store, line));
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, swiglu.swiglu_open, line));
    }
    // the gate's y at that element, and swigluOut's products in its order stored in H_T
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, swiglu.swiglu_open, pin[2]));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, swiglu.swiglu_close, pin[3]));
    // MLX's Sigmoid (unary_ops.h at T = float: gate.zig's pins), once in the text
    const gpin = comptime support.pins(@embedFile("kernels/test/pins/gate_texts.metal"));
    for (gpin[2..4]) |line| try testing.expectEqual(@as(usize, 1), std.mem.count(u8, swiglu.text_finish_swiglu, line));
    // the operands by name, no shape buffers
    for ([_][]const u8{ "gate_shape", "gate_strides", "z_shape", "svh_shape" }) |n| try testing.expectEqual(@as(usize, 0), std.mem.count(u8, swiglu.text_finish_swiglu, n));
}

test "turboexl3 hy4f swiglumul refusals: decode rows (1 to 16) refuse with Exl3EngineNotImplemented, and wrong dtypes and shapes by name, before any engine is made" {
    // no launch: every refusal comes before the stream is used
    const s: mlx.mlx_stream = .{ .ctx = null };
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const in: c_int = 128;
    const out: c_int = 256;
    const zeros = try a.alloc(u32, 33 * 256 * 2);
    defer a.free(zeros);
    @memset(zeros, 0);
    const bytes = std.mem.sliceAsBytes(zeros);
    const p: root.Proj = .{
        .trellis = try g.hostArray(bytes[0 .. (128 / 16) * (256 / 16) * 96 * 2], &.{ 8, 16, 96 }, .int16),
        .suh = try g.hostArray(bytes[0 .. 128 * 2], &.{in}, .float16),
        .svh = try g.hostArray(bytes[0 .. 256 * 2], &.{out}, .float16),
    };
    const held = root.enginesHeld().dense;
    const Case = struct { rows: c_int, xd: mlx.mlx_dtype = .float32, gd: mlx.mlx_dtype = .float32, hd: mlx.mlx_dtype = .float32, grows: c_int = 0, gout: c_int = 256, want: anyerror };
    const cases = [_]Case{
        .{ .rows = 1, .want = error.Exl3EngineNotImplemented },
        .{ .rows = 4, .want = error.Exl3EngineNotImplemented },
        .{ .rows = 16, .want = error.Exl3EngineNotImplemented },
        .{ .rows = 33, .hd = .int32, .want = error.BadExl3Dtype },
        .{ .rows = 33, .xd = .int32, .gd = .int32, .want = error.BadExl3Dtype },
        .{ .rows = 33, .gd = .bfloat16, .want = error.BadExl3Dtype },
        .{ .rows = 33, .grows = 32, .want = error.BadExl3Shape },
        .{ .rows = 33, .gout = 128, .want = error.BadExl3Shape },
    };
    for (cases) |c| {
        const m = g.mark();
        defer g.resetTo(m);
        const x = try g.hostArray(bytes[0..@intCast(c.rows * in * 4)], &.{ 1, c.rows, in }, c.xd);
        const gr = if (c.grows != 0) c.grows else c.rows;
        const gy = try g.hostArray(bytes[0..@intCast(gr * c.gout * 4)], &.{ 1, gr, c.gout }, c.gd);
        try testing.expectError(c.want, swiglu.linearSwigluMul(s, x, p, gy, dec, c.hd));
    }
    try testing.expectEqual(held, root.enginesHeld().dense);
}

/// The shared expert's gate and up (3072 -> 1536, 41 layers) and layer 0's dense MLP's (3072 -> 13824), K6 MCG
const SwigluCase = struct { name: []const u8, out: u32 };
const swiglu_cases = [_]SwigluCase{ .{ .name = "shared.gate_up", .out = 1536 }, .{ .name = "dense.gate_up", .out = 13824 } };
const swiglu_in: u32 = 3072;
/// the register path (17 to 32 rows, with `Config.nax`), the matmul path's first row count, a prompt's ragged tail
/// chunk and two prompt chunks
const swiglu_rows = [_]u32{ 20, 33, 1024, 3125, 3584 };
/// the red checks' row count and width (14.2M words)
const swiglu_red_rows: u32 = 1024;
const swiglu_red_out: u32 = 13824;

/// The chain `swiglu_fused` replaces, op for op (hy4f src/model.zig `mlp`, `swigluOut` at limit 0): the gate's and
/// up's `linear` in x's dtype, their f32 copies (none for an f32 x), MLX's sigmoid, MLX's multiply twice in swigluOut's
/// order, the cast to `hdt`; [h, the gate's y, the gate's f32 values]
fn swigluChain(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, gp: root.Proj, up: root.Proj, hdt: mlx.mlx_dtype) ![3]mlx.mlx_array {
    const gy = try g.adopt(try root.linear(s, x, gp.trellis, gp.suh, gp.svh, dec));
    const uy = try g.adopt(try root.linear(s, x, up.trellis, up.suh, up.svh, dec));
    const gf = if (mlx.mlx_array_dtype(gy) == .float32) gy else try g.astype(gy, .float32);
    const uf = if (mlx.mlx_array_dtype(uy) == .float32) uy else try g.astype(uy, .float32);
    const sg = try g.mul(gf, try g.sigmoid(gf));
    return .{ try g.astype(try g.mul(sg, uf), hdt), gy, gf };
}

test "turboexl3 gpu hy4f swiglumul pg2arms: linearSwigluMul gives the chain's words (the gate's and up's linear, f32 copies, MLX's sigmoid, two multiplies, cast) at the shared expert's and the dense MLP's widths, MCG K6, rows 20 to 3584, bf16 and f32, pg2 off and on (another shape tabled; each candidate tile: pg2's GEMM with up's SwiGLU finish); planted faults move them, in pg2's launch too (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer swiglu.unplant();
    const pg2_saved = pg2Save();
    defer pg2Restore(pg2_saved);
    var entry: [1]pgemm.Entry = undefined;
    var tbuf: [32]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(0x4F4F5761);
    const rnd = prng.random();
    var lines: usize = 0;
    var failed: usize = 0;
    var pg2_lines: usize = 0;
    var pg2_fused: usize = 0;
    for (swiglu_cases) |c| {
        defer _ = mlx.mlx_clear_cache();
        const cm = g.mark();
        defer g.resetTo(cm);
        const gp = try gateProj(a, &g, rnd, swiglu_in, c.out);
        const up = try gateProj(a, &g, rnd, swiglu_in, c.out);
        for (swiglu_rows) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const n: usize = @as(usize, rows) * c.out;
            const xh = try a.alloc(f32, @as(usize, rows) * swiglu_in);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ 1, ci(rows), ci(swiglu_in) }, .float32);
            const got = try a.alloc(f32, n);
            defer a.free(got);
            const want = try a.alloc(f32, n);
            defer a.free(want);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const dm = g.mark();
                defer g.resetTo(dm);
                const x = try g.astype(x32, dt);
                const path = (try root.denseCached(.{ .in = swiglu_in, .out = c.out, .rate = format.Rate.fromK(6), .dec = dec, .out_dtype = dt })).pathFor(rows);
                const ch = try swigluChain(&g, s, x, gp, up, dt);
                const y = try g.adopt(try swiglu.linearSwigluMul(s, x, up, ch[1], dec, dt));
                try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(c.out) }, mlx.getShape(y));
                try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
                _ = try g.hostF32(try g.astype(y, .float32), got);
                _ = try g.hostF32(try g.astype(ch[0], .float32), want);
                const w = gateWords(got, want);
                // the gate's values on each side of the sigmoid (`want` reuses the buffer after the comparison)
                _ = try g.hostF32(ch[2], want);
                var neg: usize = 0;
                for (want) |v| neg += @intFromBool(v < 0);
                const ok = w.differ == 0 and w.bad == 0 and neg > 0 and neg < n;
                lines += 1;
                failed += @intFromBool(!ok);
                std.debug.print("TURBOEXL3_HY4F_SWIGLUMUL {s} {d}->{d} K6 rows={d} path={s} x={s} h={s} words={d} differing={d} nonfinite={d} gate_negative={d} {s}\n", .{ c.name, swiglu_in, c.out, rows, @tagName(path), @tagName(dt), @tagName(dt), n, w.differ, w.bad, neg, if (ok) "PASS" else "FAIL" });
                // pg2 on: another shape tabled (up's SwiGLU finish stays on), then each candidate tile at this shape
                _ = try g.hostF32(try g.astype(ch[0], .float32), want);
                for (pg2_sets) |set| {
                    const pm = g.mark();
                    defer g.resetTo(pm);
                    pg2Apply(set, swiglu_in, c.out, &entry);
                    const before = pgemm.calls.get().*.load(.monotonic);
                    const r = swiglu.linearSwigluMul(s, x, up, ch[1], dec, dt);
                    const fused = pgemm.calls.get().*.load(.monotonic) > before;
                    pg2Restore(pg2_saved);
                    const py = try g.adopt(try r);
                    _ = try g.hostF32(try g.astype(py, .float32), got);
                    const pw = gateWords(got, want);
                    const want_fused = pg2Fuses(set, path, rows, swiglu_in, c.out, true);
                    const pok = pw.differ == 0 and pw.bad == 0 and fused == want_fused;
                    pg2_lines += 1;
                    pg2_fused += @intFromBool(fused);
                    failed += @intFromBool(!pok);
                    std.debug.print("TURBOEXL3_HY4F_SWIGLUMUL_PG2 {s} {d}->{d} rows={d} path={s} x={s} tile={s} fused={} (want {}) words={d} differing={d} nonfinite={d} {s}\n", .{ c.name, swiglu_in, c.out, rows, @tagName(path), @tagName(dt), tileName(&tbuf, set), fused, want_fused, n, pw.differ, pw.bad, if (pok) "PASS" else "FAIL" });
                }
                if (rows != swiglu_red_rows or c.out != swiglu_red_out) continue;
                // the red checks: each planted fault's words against the chain's must differ
                for ([_]swiglu.Fault{ .swapped, .sig_fast_exp }) |f| {
                    try swiglu.plant(f);
                    const red = swiglu.linearSwigluMul(s, x, up, ch[1], dec, dt);
                    swiglu.unplant();
                    const ry = try g.adopt(try red);
                    _ = try g.hostF32(try g.astype(ry, .float32), got);
                    const rw = gateWords(got, want);
                    const caught = rw.differ > 0;
                    lines += 1;
                    failed += @intFromBool(!caught);
                    std.debug.print("TURBOEXL3_HY4F_SWIGLUMUL_RED {s} rows={d} x={s} fault={s} differing={d}/{d} (want > 0) {s}\n", .{ c.name, rows, @tagName(dt), @tagName(f), rw.differ, n, if (caught) "PASS" else "FAIL" });
                }
                // the red checks in pg2's launch
                for ([_]swiglu.Fault{ .swapped, .sig_fast_exp }) |f| {
                    pg2Apply(pg2_red_set, swiglu_in, c.out, &entry);
                    try swiglu.plant(f);
                    const before = pgemm.calls.get().*.load(.monotonic);
                    const red = swiglu.linearSwigluMul(s, x, up, ch[1], dec, dt);
                    const fused = pgemm.calls.get().*.load(.monotonic) > before;
                    swiglu.unplant();
                    pg2Restore(pg2_saved);
                    const ry = try g.adopt(try red);
                    _ = try g.hostF32(try g.astype(ry, .float32), got);
                    const rw = gateWords(got, want);
                    const caught = rw.differ > 0 and fused;
                    pg2_lines += 1;
                    failed += @intFromBool(!caught);
                    std.debug.print("TURBOEXL3_HY4F_SWIGLUMUL_PG2_RED {s} rows={d} x={s} tile={s} fault={s} fused={} differing={d}/{d} (want > 0) {s}\n", .{ c.name, rows, @tagName(dt), tileName(&tbuf, pg2_red_set), @tagName(f), fused, rw.differ, n, if (caught) "PASS" else "FAIL" });
                }
            }
        }
    }
    std.debug.print("TURBOEXL3_HY4F_SWIGLUMUL_DONE lines={d} pg2_lines={d} pg2_fused={d} failed={d} {s}\n", .{ lines, pg2_lines, pg2_fused, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, swiglu_cases.len * swiglu_rows.len * 2 + 2 * 2), lines);
    try testing.expectEqual(@as(usize, swiglu_cases.len * swiglu_rows.len * 2 * pg2_sets.len + 2 * 2), pg2_lines);
    try testing.expect(pg2_fused > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── q_b with its prompt rope in the finish (`linearQRope`, qrope.zig): `qb_rope_fused`'s call ──

const qrope = @import("qrope.zig");

test "turboexl3 hy4f qrope texts: q_b's finish is finish_v, text_finish's store opening with q's value kept in f32, MLX's traditional pairs and its rope statements (precise exp2, fast cosine and sine), the f32 store" {
    try testing.expectEqualStrings(dense.finish_v ++ qrope.qrope_open ++ qrope.qrope_pair ++ qrope.qrope_rot ++ qrope.qrope_close, qrope.text_finish_qrope);
    const pin = comptime support.pins(@embedFile("kernels/test/pins/qrope_texts.metal"));
    // q's value and element: text_finish's store expression and index, once each in its store and the opening
    for (pin[0..2]) |line| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, dense.finish_store, line));
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, qrope.qrope_open, line));
    }
    // MLX's rope statements (rope.metal's rope_impl at T = float, forward): the angle, the fast cosine and sine, the
    // rotation's two values; the frequency's precise exp2 and the pair index over dims / 2
    for (pin[2..9]) |line| try testing.expectEqual(@as(usize, 1), std.mem.count(u8, qrope.qrope_rot, line));
    // the traditional pair: the neighbouring lane's value
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, qrope.qrope_pair, pin[9]));
    // the operands by name, no shape buffers
    for ([_][]const u8{ "off_shape", "off_strides", "z_shape", "svh_shape" }) |n| try testing.expectEqual(@as(usize, 0), std.mem.count(u8, qrope.text_finish_qrope, n));
    // MLX's base: log2 of the f32 theta to f32 (exact at a power of two)
    try testing.expectEqual(@as(f32, 10), qrope.log2Base(1024));
    try testing.expectEqual(@as(f32, 0), qrope.log2Base(1));
}

test "turboexl3 hy4f qrope refusals: decode rows (1 to 16) refuse with Exl3EngineNotImplemented, and wrong dtypes, thetas, positions and head geometries by name, before any engine is made" {
    // no launch: every refusal comes before the stream is used
    const s: mlx.mlx_stream = .{ .ctx = null };
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const in: c_int = 128;
    const out: c_int = 256;
    const zeros = try a.alloc(u32, 33 * 256 * 2);
    defer a.free(zeros);
    @memset(zeros, 0);
    const bytes = std.mem.sliceAsBytes(zeros);
    const p: root.Proj = .{
        .trellis = try g.hostArray(bytes[0 .. (128 / 16) * (256 / 16) * 96 * 2], &.{ 8, 16, 96 }, .int16),
        .suh = try g.hostArray(bytes[0 .. 128 * 2], &.{in}, .float16),
        .svh = try g.hostArray(bytes[0 .. 256 * 2], &.{out}, .float16),
    };
    const held = root.enginesHeld().dense;
    const Case = struct { rows: c_int, xd: mlx.mlx_dtype = .float32, theta: f32 = 1e4, pos: i32 = 0, hd: u32 = 128, rope: u32 = 64, want: anyerror };
    const cases = [_]Case{
        .{ .rows = 1, .want = error.Exl3EngineNotImplemented },
        .{ .rows = 6, .want = error.Exl3EngineNotImplemented },
        .{ .rows = 16, .want = error.Exl3EngineNotImplemented },
        .{ .rows = 33, .xd = .int32, .want = error.BadExl3Dtype },
        .{ .rows = 33, .theta = 0, .want = error.BadExl3Shape },
        .{ .rows = 33, .theta = std.math.nan(f32), .want = error.BadExl3Shape },
        .{ .rows = 33, .theta = std.math.inf(f32), .want = error.BadExl3Shape },
        .{ .rows = 33, .pos = -1, .want = error.BadExl3Shape },
        .{ .rows = 33, .hd = 96, .want = error.BadExl3Shape },
        .{ .rows = 33, .hd = 0, .want = error.BadExl3Shape },
        .{ .rows = 33, .rope = 0, .want = error.BadExl3Shape },
        .{ .rows = 33, .rope = 63, .want = error.BadExl3Shape },
        .{ .rows = 33, .rope = 192, .want = error.BadExl3Shape },
    };
    for (cases) |c| {
        const m = g.mark();
        defer g.resetTo(m);
        const x = try g.hostArray(bytes[0..@intCast(c.rows * in * 4)], &.{ 1, c.rows, in }, c.xd);
        try testing.expectError(c.want, qrope.linearQRope(s, x, p, c.theta, c.pos, c.hd, c.rope, dec));
    }
    try testing.expectEqual(held, root.enginesHeld().dense);
}

/// q_b at the indexed layers (1024 -> 32 heads of 256, theta 1e7) and the sliding ones (64 heads, theta 1e4), K6 MCG,
/// the last 64 dims of each head roped
const QropeCase = struct { name: []const u8, heads: u32, theta: f32 };
const qrope_cases = [_]QropeCase{ .{ .name = "q_b.indexed", .heads = 32, .theta = 1e7 }, .{ .name = "q_b.sliding", .heads = 64, .theta = 1e4 } };
const qrope_in: u32 = 1024;
const qrope_hd: u32 = 256;
const qrope_rope: u32 = 64;
/// the matmul path's first row count at position 0 and at the end of a 17,492-token prompt, two prompt chunks past
/// position 0, and that prompt's ragged tail chunk (its last 3125 rows)
const QropeRows = struct { rows: u32, pos: i32 };
const qrope_rows = [_]QropeRows{ .{ .rows = 33, .pos = 0 }, .{ .rows = 33, .pos = 17459 }, .{ .rows = 1024, .pos = 2048 }, .{ .rows = 3125, .pos = 14367 }, .{ .rows = 3584, .pos = 13905 } };
/// the red checks' call (the sliding width, 16.7M words)
const qrope_red_rows: u32 = 1024;

/// The chain `qb_rope_fused` replaces (hy4f src/model.zig `attention`): q_b's `linear` in x's dtype, its f32 copy
/// (none for an f32 x), the reshape and transpose to [1, H, rows, HD], q_nope the slice of each head's first HD - ROPE
/// dims and q_pe MLX's fast::rope (traditional, base theta, scale 1, offset pos) of the slice of its last ROPE; returned
/// as q_nope | q_pe along the head dim, back in [1, rows, H, HD]
fn qropeChain(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, rows: u32, heads: u32, theta: f32, pos: i32) !mlx.mlx_array {
    const y = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, dec));
    const yf = if (mlx.mlx_array_dtype(y) == .float32) y else try g.astype(y, .float32);
    const q = try g.transposeAxes(try g.reshape(yf, &.{ 1, ci(rows), ci(heads), ci(qrope_hd) }), &.{ 0, 2, 1, 3 });
    const nope: c_int = ci(qrope_hd - qrope_rope);
    const q_nope = try g.slice(q, &.{ 0, 0, 0, 0 }, &.{ 1, ci(heads), ci(rows), nope }, &.{ 1, 1, 1, 1 });
    const q_slice = try g.slice(q, &.{ 0, 0, 0, nope }, &.{ 1, ci(heads), ci(rows), ci(qrope_hd) }, &.{ 1, 1, 1, 1 });
    var pe = mlx.mlx_array_new();
    mlx.check(mlx.mlx_fast_rope(&pe, q_slice, ci(qrope_rope), true, .{ .value = theta, .has_value = true }, 1.0, pos, .{ .ctx = null }, s)) catch |e| {
        _ = mlx.mlx_array_free(pe);
        return e;
    };
    const q_pe = try g.adopt(pe);
    return g.transposeAxes(try g.concat(&.{ q_nope, q_pe }, 3), &.{ 0, 2, 1, 3 });
}

/// The words of `got` and `want` ([rows, H, HD] f32) that differ, in each head's first HD - ROPE dims and in its last
/// ROPE, and `want`'s nonfinite
const QropeWords = struct { nope: usize = 0, rope: usize = 0, bad: usize = 0 };
fn qropeWords(got: []const f32, want: []const f32) QropeWords {
    var r: QropeWords = .{};
    for (got, want, 0..) |u, w, i| {
        const differ = @as(u32, @bitCast(u)) != @as(u32, @bitCast(w));
        if (i % qrope_hd < qrope_hd - qrope_rope) r.nope += @intFromBool(differ) else r.rope += @intFromBool(differ);
        r.bad += @intFromBool(!std.math.isFinite(w));
    }
    return r;
}

test "turboexl3 gpu hy4f qrope pg2arms: linearQRope gives the chain's words (linear, f32 copy, transpose, MLX's fast::rope of the strided slice) at q_b's two widths and thetas, MCG K6, rows 33 to 3584 at positions 0 to 17459, x bf16 and f32, pg2 off and on (another shape tabled; each candidate tile: pg2's GEMM with q_b's rope finish); planted faults move them, in pg2's launch too (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer qrope.unplant();
    const pg2_saved = pg2Save();
    defer pg2Restore(pg2_saved);
    var entry: [1]pgemm.Entry = undefined;
    var tbuf: [32]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(0x4F4F7172);
    const rnd = prng.random();
    var lines: usize = 0;
    var failed: usize = 0;
    var pg2_lines: usize = 0;
    var pg2_fused: usize = 0;
    for (qrope_cases) |c| {
        defer _ = mlx.mlx_clear_cache();
        const cm = g.mark();
        defer g.resetTo(cm);
        const out = c.heads * qrope_hd;
        const p = try gateProj(a, &g, rnd, qrope_in, out);
        for (qrope_rows) |rc| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const n: usize = @as(usize, rc.rows) * out;
            const xh = try a.alloc(f32, @as(usize, rc.rows) * qrope_in);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ 1, ci(rc.rows), ci(qrope_in) }, .float32);
            const got = try a.alloc(f32, n);
            defer a.free(got);
            const want = try a.alloc(f32, n);
            defer a.free(want);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const dm = g.mark();
                defer g.resetTo(dm);
                const x = try g.astype(x32, dt);
                const path = (try root.denseCached(.{ .in = qrope_in, .out = out, .rate = format.Rate.fromK(6), .dec = dec, .out_dtype = dt })).pathFor(rc.rows);
                const ch = try qropeChain(&g, s, x, p, rc.rows, c.heads, c.theta, rc.pos);
                const y = try g.adopt(try qrope.linearQRope(s, x, p, c.theta, rc.pos, qrope_hd, qrope_rope, dec));
                try testing.expectEqualSlices(c_int, &.{ 1, ci(rc.rows), ci(out) }, mlx.getShape(y));
                try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(y));
                _ = try g.hostF32(y, got);
                _ = try g.hostF32(ch, want);
                const w = qropeWords(got, want);
                const ok = w.nope == 0 and w.rope == 0 and w.bad == 0;
                lines += 1;
                failed += @intFromBool(!ok);
                std.debug.print("TURBOEXL3_HY4F_QROPE {s} {d}->{d} heads={d} hd={d} rope={d} theta={d} K6 rows={d} pos={d} path={s} x={s} words={d} differing_nope={d} differing_rope={d} nonfinite={d} {s}\n", .{ c.name, qrope_in, out, c.heads, qrope_hd, qrope_rope, c.theta, rc.rows, rc.pos, @tagName(path), @tagName(dt), n, w.nope, w.rope, w.bad, if (ok) "PASS" else "FAIL" });
                // pg2 on: another shape tabled (q_b's rope finish stays on), then each candidate tile at this shape
                for (pg2_sets) |set| {
                    const pm = g.mark();
                    defer g.resetTo(pm);
                    pg2Apply(set, qrope_in, out, &entry);
                    const before = pgemm.calls.get().*.load(.monotonic);
                    const r = qrope.linearQRope(s, x, p, c.theta, rc.pos, qrope_hd, qrope_rope, dec);
                    const fused = pgemm.calls.get().*.load(.monotonic) > before;
                    pg2Restore(pg2_saved);
                    const py = try g.adopt(try r);
                    _ = try g.hostF32(py, got);
                    const pw = qropeWords(got, want);
                    const want_fused = pg2Fuses(set, path, rc.rows, qrope_in, out, true);
                    const pok = pw.nope == 0 and pw.rope == 0 and pw.bad == 0 and fused == want_fused;
                    pg2_lines += 1;
                    pg2_fused += @intFromBool(fused);
                    failed += @intFromBool(!pok);
                    std.debug.print("TURBOEXL3_HY4F_QROPE_PG2 {s} {d}->{d} rows={d} pos={d} path={s} x={s} tile={s} fused={} (want {}) words={d} differing_nope={d} differing_rope={d} nonfinite={d} {s}\n", .{ c.name, qrope_in, out, rc.rows, rc.pos, @tagName(path), @tagName(dt), tileName(&tbuf, set), fused, want_fused, n, pw.nope, pw.rope, pw.bad, if (pok) "PASS" else "FAIL" });
                }
                if (c.heads != 64) continue;
                if (rc.rows != qrope_red_rows) continue;
                // the red checks: each planted fault's words against the chain's must differ, on the rope dims
                for ([_]qrope.Fault{ .pair_neox, .freq_ln }) |f| {
                    try qrope.plant(f);
                    const red = qrope.linearQRope(s, x, p, c.theta, rc.pos, qrope_hd, qrope_rope, dec);
                    qrope.unplant();
                    const ry = try g.adopt(try red);
                    _ = try g.hostF32(ry, got);
                    const rw = qropeWords(got, want);
                    const caught = rw.rope > 0 and rw.nope == 0;
                    lines += 1;
                    failed += @intFromBool(!caught);
                    std.debug.print("TURBOEXL3_HY4F_QROPE_RED {s} rows={d} pos={d} x={s} fault={s} differing_rope={d} differing_nope={d}/{d} (want rope > 0, nope 0) {s}\n", .{ c.name, rc.rows, rc.pos, @tagName(dt), @tagName(f), rw.rope, rw.nope, n, if (caught) "PASS" else "FAIL" });
                }
                // the red checks in pg2's launch: the rope dims move, no other
                for ([_]qrope.Fault{ .pair_neox, .freq_ln }) |f| {
                    pg2Apply(pg2_red_set, qrope_in, out, &entry);
                    try qrope.plant(f);
                    const before = pgemm.calls.get().*.load(.monotonic);
                    const red = qrope.linearQRope(s, x, p, c.theta, rc.pos, qrope_hd, qrope_rope, dec);
                    const fused = pgemm.calls.get().*.load(.monotonic) > before;
                    qrope.unplant();
                    pg2Restore(pg2_saved);
                    const ry = try g.adopt(try red);
                    _ = try g.hostF32(ry, got);
                    const rw = qropeWords(got, want);
                    const caught = rw.rope > 0 and rw.nope == 0 and fused;
                    pg2_lines += 1;
                    failed += @intFromBool(!caught);
                    std.debug.print("TURBOEXL3_HY4F_QROPE_PG2_RED {s} rows={d} pos={d} x={s} tile={s} fault={s} fused={} differing_rope={d} differing_nope={d}/{d} (want rope > 0, nope 0) {s}\n", .{ c.name, rc.rows, rc.pos, @tagName(dt), tileName(&tbuf, pg2_red_set), @tagName(f), fused, rw.rope, rw.nope, n, if (caught) "PASS" else "FAIL" });
                }
            }
        }
    }
    std.debug.print("TURBOEXL3_HY4F_QROPE_DONE lines={d} pg2_lines={d} pg2_fused={d} failed={d} {s}\n", .{ lines, pg2_lines, pg2_fused, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, qrope_cases.len * qrope_rows.len * 2 + 2 * 2), lines);
    try testing.expectEqual(@as(usize, qrope_cases.len * qrope_rows.len * 2 * pg2_sets.len + 2 * 2), pg2_lines);
    try testing.expect(pg2_fused > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── pg2arms: the arms' pg2 texts, and pg2 at every trunk linear (`root.linear`), split K in registers ──

const hy4f_pg2 = @import("pg2.zig");

test "turboexl3 hy4f pg2arms texts: each arm's pg2 launch is pg2's GEMM and finish opening, the arm's own store loop (the gate's rows from the GEMM's M), the row loop's close; the arms' finishes keep their texts" {
    const tail = pgemm.fused_2_2_2 ++ pgemm.undef;
    try testing.expectEqualStrings(pgemm.fused_1 ++ pgemm.fused_2_1 ++ gate.gate_open_1 ++ gate.gate_open_2_pg2 ++ gate.gate_open_3 ++ gate.gate_sig ++ gate.gate_src ++ gate.gate_close ++ tail, gate.text_pg2_gate);
    try testing.expectEqualStrings(pgemm.fused_1 ++ pgemm.fused_2_1 ++ swiglu.swiglu_open ++ gate.gate_sig ++ swiglu.swiglu_close ++ tail, swiglu.text_pg2_swiglu);
    try testing.expectEqualStrings(pgemm.fused_1 ++ pgemm.fused_2_1 ++ qrope.qrope_open ++ qrope.qrope_pair ++ qrope.qrope_rot ++ qrope.qrope_close ++ tail, qrope.text_pg2_qrope);
    // the finishes' texts as they were: finish_v, then the store loop (the gate's opening in its three pieces)
    try testing.expectEqualStrings(dense.finish_v ++ gate.gate_open_1 ++ gate.gate_open_2 ++ gate.gate_open_3 ++ gate.gate_sig ++ gate.gate_src ++ gate.gate_close, gate.text_finish_gate);
    const pin = comptime support.pins(@embedFile("kernels/test/pins/pg2_texts.metal"));
    // the gate's rows: the finish grid's y there, the GEMM's M here; pg2's own store loop in none of the arms' launches
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, gate.gate_open_2, pin[0]));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, gate.gate_open_2_pg2, pin[1]));
    for ([_][]const u8{ gate.text_pg2_gate, swiglu.text_pg2_swiglu, qrope.text_pg2_qrope }) |t| {
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, t, pin[2]));
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, t, pgemm.fused_2_2_1));
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, t, pin[3]));
    }
}

/// A plain linear's pg2 settings beside the chain: each of `pg2_sets`, then the served table (`hy4f_pg2.serve`)
const LinSet = union(enum) { set: Pg2Set, served };

test "turboexl3 gpu hy4f linear pg2arms: root.linear with pg2 on gives the chain's words at every Hy4-flash trunk shape (each candidate tile at bf16, the served table at bf16 and f32), split K in registers at o_proj.sliding and dense.down, MCG K6, rows 33 to 3584; MLX's regular loop and a wrong partition move the split shapes' words (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const pg2_saved = pg2Save();
    defer pg2Restore(pg2_saved);
    var entry: [1]pgemm.Entry = undefined;
    var tbuf: [32]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(0x4F4F7032);
    const rnd = prng.random();
    const rows_list = [_]u32{ 33, 1024, 3125, 3584 };
    var lines: usize = 0;
    var fused_lines: usize = 0;
    var split_lines: usize = 0;
    var reds: usize = 0;
    var failed: usize = 0;
    for (hy4f_pg2.trunk) |l| {
        defer _ = mlx.mlx_clear_cache();
        const cm = g.mark();
        defer g.resetTo(cm);
        const p = try gateProj(a, &g, rnd, l.k, l.n);
        for (rows_list) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const n: usize = @as(usize, rows) * l.n;
            const xh = try a.alloc(f32, @as(usize, rows) * l.k);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(l.k) }, .float32);
            const got = try a.alloc(f32, n);
            defer a.free(got);
            const want = try a.alloc(f32, n);
            defer a.free(want);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const dm = g.mark();
                defer g.resetTo(dm);
                const x = try g.astype(x32, dt);
                const d = try root.denseCached(.{ .in = l.k, .out = l.n, .rate = format.Rate.fromK(6), .dec = dec, .out_dtype = dt });
                const path = d.pathFor(rows);
                pg2Restore(pg2_saved);
                const wy = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, dec));
                _ = try g.hostF32(try g.astype(wy, .float32), want);
                var sets_buf: [pg2_sets.len + 1]LinSet = undefined;
                var ns: usize = 0;
                if (dt == .bfloat16) for (pg2_sets) |set| {
                    sets_buf[ns] = .{ .set = set };
                    ns += 1;
                };
                sets_buf[ns] = .served;
                ns += 1;
                for (sets_buf[0..ns]) |ls| {
                    const pm = g.mark();
                    defer g.resetTo(pm);
                    var want_fused: bool = undefined;
                    switch (ls) {
                        .set => |set| {
                            pg2Apply(set, l.k, l.n, &entry);
                            want_fused = pg2Fuses(set, path, rows, l.k, l.n, false);
                        },
                        .served => {
                            hy4f_pg2.serve(true);
                            want_fused = path == .matmul and hy4f_pg2.has(l.k, l.n) and (pgemm.mlxRegular(rows, l.n, l.k) or pgemm.splitKAt(rows, l.n, l.k) != null);
                        },
                    }
                    const before = pgemm.calls.get().*.load(.monotonic);
                    const r = root.linear(s, x, p.trellis, p.suh, p.svh, dec);
                    const fused = pgemm.calls.get().*.load(.monotonic) > before;
                    pg2Restore(pg2_saved);
                    const y = try g.adopt(try r);
                    _ = try g.hostF32(try g.astype(y, .float32), got);
                    const w = gateWords(got, want);
                    const split = fused and !pgemm.mlxRegular(rows, l.n, l.k);
                    const ok = w.differ == 0 and w.bad == 0 and fused == want_fused;
                    lines += 1;
                    fused_lines += @intFromBool(fused);
                    split_lines += @intFromBool(split);
                    failed += @intFromBool(!ok);
                    std.debug.print("TURBOEXL3_HY4F_LINEAR_PG2 {s} {d}->{d} K6 rows={d} path={s} x={s} tile={s} split_k={} fused={} (want {}) words={d} differing={d} nonfinite={d} {s}\n", .{ l.name, l.k, l.n, rows, @tagName(path), @tagName(dt), switch (ls) {
                        .set => |set| tileName(&tbuf, set),
                        .served => "served",
                    }, split, fused, want_fused, n, w.differ, w.bad, if (ok) "PASS" else "FAIL" });
                }
                // the red checks at the split shapes: pg2's launch on `linear`'s own operands with MLX's regular loop (one
                // accumulator over all of K) and with 2048-k partitions in place of MLX's 4096 must each move words
                if (dt != .bfloat16 or rows != 1024 or pgemm.splitKAt(rows, l.n, l.k) == null) continue;
                const ops = (try d.prefillOperands(s, try g.reshape(x, &.{ ci(rows), ci(l.k) }), p.trellis, p.suh, rows)) orelse return error.TestUnexpectedResult;
                const xp = try g.adopt(ops[0]);
                const wt = try g.adopt(ops[1]);
                const t = pgemm.candidates[1];
                const sk = pgemm.splitKAt(rows, l.n, l.k).?;
                const RedPlan = struct { name: []const u8, plan: pgemm.Plan };
                const plans = [_]RedPlan{
                    .{ .name = "served_split", .plan = .{ .tile = t, .sk = sk } },
                    .{ .name = "regular_loop", .plan = .{ .tile = t, .sk = null } },
                    .{ .name = "partitions_2048", .plan = .{ .tile = t, .sk = .{ .parts = (l.k + 2047) / 2048, .size = 2048 } } },
                };
                for (plans) |rp| {
                    const pm = g.mark();
                    defer g.resetTo(pm);
                    const ry = try g.adopt(try pgemm.launch(s, xp, wt, p.svh, dt, rp.plan));
                    _ = try g.hostF32(try g.astype(ry, .float32), got);
                    const rw = gateWords(got, want);
                    const is_served = rp.plan.sk != null and rp.plan.sk.?.size == sk.size;
                    const ok = if (is_served) rw.differ == 0 else rw.differ > 0;
                    reds += 1;
                    failed += @intFromBool(!ok);
                    std.debug.print("TURBOEXL3_HY4F_LINEAR_PG2_RED {s} {d}->{d} rows={d} x={s} plan={s} differing={d}/{d} (want {s}) {s}\n", .{ l.name, l.k, l.n, rows, @tagName(dt), rp.name, rw.differ, n, if (is_served) "none" else "some", if (ok) "PASS" else "FAIL" });
                }
            }
        }
    }
    std.debug.print("TURBOEXL3_HY4F_LINEAR_PG2_DONE lines={d} fused={d} split={d} reds={d} failed={d} {s}\n", .{ lines, fused_lines, split_lines, reds, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, hy4f_pg2.trunk.len * rows_list.len * (pg2_sets.len + 2)), lines);
    try testing.expectEqual(@as(usize, 2 * 3), reds);
    try testing.expect(fused_lines > 0 and split_lines > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── the norms and the next linear's prepare in a linear's finish (rms.zig): the hy4f model plugin's
// `addnorm_fused`, `qnorm_fused`, `kvnorm_fused` and `down_prep_fused` calls. Every test
// name holds finprep (a one-word T3_FILTER). No f32 matmul runs on either side (the inner products are `linear`'s f16
// ones, the norms and the rope MLX's own kernels), so MLX's TF32 setting moves no word here.

const rms = @import("rms.zig");

test "turboexl3 hy4f finprep rms texts: the row finish is finish_v with the simdgroup's block, the row staged as text_finish's store rounds it, MLX's rms_single_row statements, its output loop or the next linear's prepare, k_pe's rope by qrope's statements" {
    // finish_v_row: finish_v with its block line the simdgroup's, every other byte the same
    const a = testing.allocator;
    const grid_b = "const uint b = threadgroup_position_in_grid.x;";
    const simd_b = "const uint b = simdgroup_index_in_threadgroup;";
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, dense.finish_v, grid_b));
    const swapped = try std.mem.replaceOwned(u8, a, dense.finish_v, grid_b, simd_b);
    defer a.free(swapped);
    try testing.expectEqualStrings(swapped, rms.finish_v_row);
    // the compositions
    try testing.expectEqualStrings(rms.finish_v_row ++ rms.rms_stage_add ++ rms.rms_sum ++ rms.rms_store, rms.text_add_rms);
    try testing.expectEqualStrings(rms.finish_v_row ++ rms.rms_stage ++ rms.rms_sum ++ rms.rms_qx ++ rms.rms_prep_load ++ dense.prep_2_2 ++ dense.prep_3, rms.text_rms_prep);
    try testing.expectEqualStrings(rms.finish_v_row ++ rms.rms_stage ++ rms.rms_sum ++ rms.rms_store ++ rms.kv_rope_open ++ qrope.qrope_pair ++ qrope.qrope_rot ++ rms.kv_rope_close, rms.text_kv);
    try testing.expectEqualStrings(dense.finish_v ++ rms.swp_open ++ swiglu.swiglu_open ++ gate.gate_sig ++ rms.swp_close ++ dense.prep_2_2 ++ dense.prep_3, rms.text_swiglu_prep);
    const pin = comptime support.pins(@embedFile("kernels/test/pins/rms_texts.metal"));
    // the linear's value and element: text_finish's store expression and index, once each in its store and each staging
    for (pin[0..2]) |line| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, dense.finish_store, line));
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rms.rms_stage, line));
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rms.rms_stage_add, line));
    }
    // MLX's square sum in both its branches, its two simd sums and its inverse mean
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rms.rms_sum, pin[2]));
    for (pin[3..6]) |line| try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rms.rms_sum, line));
    // MLX's output loop in both its branches; the same product in the prepare's staging
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rms.rms_store, pin[6]));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rms.rms_qx, pin[7]));
    // the next linear's prepare: prep_2_1's load statement on the staged Q_T row (x's f16 rounding times suh)
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rms.rms_prep_load, pin[8]));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, dense.prep_2_1, pin[9]));
    // the SwiGLU's h as swiglu_close stores it, then the down linear's prepare of it
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, swiglu.swiglu_close, pin[10]));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rms.swp_close, pin[11]));
    // k_pe's store, and qrope's pair and frequency in the kv text
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rms.kv_rope_close, pin[12]));
    for (pin[13..15]) |line| try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rms.text_kv, line));
    // the operands by name, no shape buffers
    for ([_][:0]const u8{ rms.text_add_rms, rms.text_rms_prep, rms.text_kv, rms.text_swiglu_prep }) |t|
        for ([_][]const u8{ "z_shape", "svh_shape", "w_shape", "add_shape", "gate_shape", "suh_shape" }) |n| try testing.expectEqual(@as(usize, 0), std.mem.count(u8, t, n));
}

/// A refusal test's projection [in -> out] of zero words (never launched)
fn zeroProj(g: *G, bytes: []const u8, in: c_int, out: c_int) !root.Proj {
    return .{
        .trellis = try g.hostArray(bytes[0..@intCast(@divExact(in, 16) * @divExact(out, 16) * 96 * 2)], &.{ @divExact(in, 16), @divExact(out, 16), 96 }, .int16),
        .suh = try g.hostArray(bytes[0..@intCast(in * 2)], &.{in}, .float16),
        .svh = try g.hostArray(bytes[0..@intCast(out * 2)], &.{out}, .float16),
    };
}

test "turboexl3 hy4f finprep rms refusals: decode rows (1 to 16), outputs past one threadgroup and wrong dtypes, shapes, weights, adds, kv geometries, thetas, positions and next linears refuse by name, before any engine is made" {
    // no launch: every refusal comes before the stream is used
    const s: mlx.mlx_stream = .{ .ctx = null };
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const zeros = try a.alloc(u32, 4 * 1024 * 1024);
    defer a.free(zeros);
    @memset(zeros, 0);
    const bytes = std.mem.sliceAsBytes(zeros);
    const in: c_int = 128;
    const p = try zeroProj(&g, bytes, in, 256);
    const wide = try zeroProj(&g, bytes, in, 4224);
    const odd = try zeroProj(&g, bytes, in, 144);
    const next = try zeroProj(&g, bytes, 256, 128);
    const next_bad = try zeroProj(&g, bytes, 128, 128);
    const held = root.enginesHeld().dense;
    const X = struct { rows: c_int = 33, xd: mlx.mlx_dtype = .float32, xp: bool = false, fd: mlx.mlx_dtype = .bfloat16 };
    const Fx = struct {
        fn x(gr: *G, b: []const u8, c: X) !rms.Src {
            const arr = try gr.hostArray(b[0..@intCast(c.rows * 128 * 4)], &.{ 1, c.rows, 128 }, c.xd);
            return if (c.xp) .{ .xp = arr } else .{ .x = arr };
        }
        fn f32s(gr: *G, b: []const u8, shape: []const c_int, dt: mlx.mlx_dtype) !mlx.mlx_array {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            return gr.hostArray(b[0 .. n * 4], shape, dt);
        }
    };
    const w256 = try Fx.f32s(&g, bytes, &.{256}, .float32);
    // linearRms (its add [rows, out] where the case names none)
    {
        const Case = struct { x: X = .{}, p: enum { p, wide, odd } = .p, w: c_int = 256, wd: mlx.mlx_dtype = .float32, add: ?[2]c_int = null, ad: mlx.mlx_dtype = .bfloat16, want: anyerror };
        const cases = [_]Case{
            .{ .x = .{ .rows = 1 }, .want = error.Exl3EngineNotImplemented },
            .{ .x = .{ .rows = 16 }, .want = error.Exl3EngineNotImplemented },
            .{ .x = .{ .rows = 16, .xp = true, .xd = .float16 }, .want = error.Exl3EngineNotImplemented },
            .{ .p = .wide, .want = error.Exl3EngineNotImplemented },
            .{ .p = .odd, .want = error.BadExl3Shape },
            .{ .x = .{ .xd = .int32 }, .want = error.BadExl3Dtype },
            .{ .x = .{ .xp = true, .xd = .bfloat16 }, .want = error.BadExl3Dtype },
            .{ .x = .{ .xp = true, .xd = .float16, .fd = .int32 }, .want = error.BadExl3Dtype },
            .{ .wd = .bfloat16, .want = error.BadExl3Dtype },
            .{ .w = 128, .want = error.BadExl3Shape },
            .{ .add = .{ 33, 256 }, .ad = .int32, .want = error.BadExl3Dtype },
            .{ .add = .{ 33, 128 }, .want = error.BadExl3Shape },
            .{ .add = .{ 32, 256 }, .want = error.BadExl3Shape },
        };
        for (cases) |c| {
            const m = g.mark();
            defer g.resetTo(m);
            const src = try Fx.x(&g, bytes, c.x);
            const pp = switch (c.p) {
                .p => p,
                .wide => wide,
                .odd => odd,
            };
            const w = try Fx.f32s(&g, bytes, &.{c.w}, c.wd);
            const out: c_int = switch (c.p) {
                .p => 256,
                .wide => 4224,
                .odd => 144,
            };
            const addend = if (c.add) |sh| try Fx.f32s(&g, bytes, &sh, c.ad) else try Fx.f32s(&g, bytes, &.{ c.x.rows, out }, .bfloat16);
            try testing.expectError(c.want, rms.linearRms(s, src, c.x.fd, pp, addend, w, 1e-5, dec));
        }
    }
    // linearRmsPrep
    {
        const Case = struct { x: X = .{}, qt: mlx.mlx_dtype = .bfloat16, next: enum { ok, bad, f32suh } = .ok, want: anyerror };
        const cases = [_]Case{
            .{ .x = .{ .rows = 8 }, .want = error.Exl3EngineNotImplemented },
            .{ .qt = .int32, .want = error.BadExl3Dtype },
            .{ .next = .bad, .want = error.BadExl3Shape },
            .{ .next = .f32suh, .want = error.BadExl3Dtype },
        };
        for (cases) |c| {
            const m = g.mark();
            defer g.resetTo(m);
            const src = try Fx.x(&g, bytes, c.x);
            const nx: root.Proj = switch (c.next) {
                .ok => next,
                .bad => next_bad,
                .f32suh => .{ .trellis = next.trellis, .suh = try Fx.f32s(&g, bytes, &.{256}, .float32), .svh = next.svh },
            };
            try testing.expectError(c.want, rms.linearRmsPrep(s, src, c.x.fd, p, w256, 1e-5, nx, c.qt, true, dec));
        }
    }
    // linearKv
    {
        const Case = struct { x: X = .{}, lat: u32 = 128, rope: u32 = 16, theta: f32 = 1e4, pos: i32 = 0, w: c_int = 128, want: anyerror };
        const cases = [_]Case{
            .{ .x = .{ .rows = 4 }, .want = error.Exl3EngineNotImplemented },
            .{ .lat = 0, .want = error.BadExl3Shape },
            .{ .lat = 96, .want = error.BadExl3Shape },
            .{ .rope = 0, .want = error.BadExl3Shape },
            .{ .rope = 15, .want = error.BadExl3Shape },
            .{ .rope = 130, .want = error.BadExl3Shape },
            .{ .lat = 256, .rope = 2, .want = error.BadExl3Shape },
            .{ .theta = 0, .want = error.BadExl3Shape },
            .{ .theta = std.math.nan(f32), .want = error.BadExl3Shape },
            .{ .theta = std.math.inf(f32), .want = error.BadExl3Shape },
            .{ .pos = -1, .want = error.BadExl3Shape },
            .{ .w = 256, .want = error.BadExl3Shape },
        };
        for (cases) |c| {
            const m = g.mark();
            defer g.resetTo(m);
            const src = try Fx.x(&g, bytes, c.x);
            const w = try Fx.f32s(&g, bytes, &.{c.w}, .float32);
            try testing.expectError(c.want, rms.linearKv(s, src, c.x.fd, p, w, 1e-5, c.lat, c.rope, c.theta, c.pos, dec));
        }
    }
    // linearSwigluPrep
    {
        const Case = struct { rows: c_int = 33, xd: mlx.mlx_dtype = .float32, gd: mlx.mlx_dtype = .float32, gw: c_int = 256, hd: mlx.mlx_dtype = .bfloat16, up: enum { p, odd } = .p, next: enum { ok, bad } = .ok, want: anyerror };
        const cases = [_]Case{
            .{ .rows = 1, .want = error.Exl3EngineNotImplemented },
            .{ .rows = 16, .want = error.Exl3EngineNotImplemented },
            .{ .xd = .int32, .gd = .int32, .want = error.BadExl3Dtype },
            .{ .hd = .int32, .want = error.BadExl3Dtype },
            .{ .gd = .bfloat16, .want = error.BadExl3Dtype },
            .{ .gw = 128, .want = error.BadExl3Shape },
            .{ .up = .odd, .gw = 144, .want = error.BadExl3Shape },
            .{ .next = .bad, .want = error.BadExl3Shape },
        };
        for (cases) |c| {
            const m = g.mark();
            defer g.resetTo(m);
            const x = try g.hostArray(bytes[0..@intCast(c.rows * 128 * 4)], &.{ 1, c.rows, 128 }, c.xd);
            const gy = try Fx.f32s(&g, bytes, &.{ 1, c.rows, c.gw }, c.gd);
            try testing.expectError(c.want, rms.linearSwigluPrep(s, x, if (c.up == .odd) odd else p, gy, if (c.next == .bad) next_bad else next, dec, c.hd));
        }
    }
    // qrope from prepared rows: an xp that is not f16, and a finish dtype that is not a float
    {
        const m = g.mark();
        defer g.resetTo(m);
        const xb = try g.hostArray(bytes[0 .. 33 * 128 * 4], &.{ 1, 33, 128 }, .bfloat16);
        try testing.expectError(error.BadExl3Dtype, qrope.linearQRopePrepared(s, xb, .bfloat16, p, 1e4, 0, 128, 64, dec));
        const xh = try g.hostArray(bytes[0 .. 33 * 128 * 4], &.{ 1, 33, 128 }, .float16);
        try testing.expectError(error.BadExl3Dtype, qrope.linearQRopePrepared(s, xh, .int32, p, 1e4, 0, 128, 64, dec));
        const xd = try g.hostArray(bytes[0 .. 16 * 128 * 4], &.{ 1, 16, 128 }, .float16);
        try testing.expectError(error.Exl3EngineNotImplemented, qrope.linearQRopePrepared(s, xd, .bfloat16, p, 1e4, 0, 128, 64, dec));
    }
    try testing.expectEqual(held, root.enginesHeld().dense);
}

// ── the GPU tests: every word the chain's ──

/// A norm's f32 weight [n] about 1 (0.5 to 1.5)
fn rmsWeight(a: std.mem.Allocator, g: *G, rnd: std.Random, n: u32) !mlx.mlx_array {
    const w = try a.alloc(f32, n);
    defer a.free(w);
    for (w) |*v| v.* = 0.5 + rnd.float(f32);
    return g.hostArray(std.mem.sliceAsBytes(w), &.{ci(n)}, .float32);
}

const rms_eps: f32 = 1e-5;
/// the register path, the matmul path's first row count, two prompt chunks
const rms_rows = [_]u32{ 20, 33, 1024, 3584 };
/// the red checks' row count
const rms_red_rows: u32 = 1024;

/// MLX's rms_norm of x (f32) by w at `rms_eps`
fn mlxRms(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
    var n = mlx.mlx_array_new();
    mlx.check(mlx.mlx_fast_rms_norm(&n, x, w, rms_eps, s)) catch |e| {
        _ = mlx.mlx_array_free(n);
        return e;
    };
    return g.adopt(n);
}

/// MLX's fast::rope of x (traditional, base theta, scale 1, offset pos) over its last `dims`
fn mlxRope(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, dims: u32, theta: f32, pos: i32) !mlx.mlx_array {
    var pe = mlx.mlx_array_new();
    mlx.check(mlx.mlx_fast_rope(&pe, x, ci(dims), true, .{ .value = theta, .has_value = true }, 1.0, pos, .{ .ctx = null }, s)) catch |e| {
        _ = mlx.mlx_array_free(pe);
        return e;
    };
    return g.adopt(pe);
}

/// f32(`linear(x)`) as the host's `proj` takes it: its y in x's dtype, then MLX's astype (none for an f32 x)
fn projF32(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj) !mlx.mlx_array {
    const y = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, dec));
    return if (mlx.mlx_array_dtype(y) == .float32) y else g.astype(y, .float32);
}

const Words = struct { differ: usize, bad: usize };

/// The words of `y` (any float dtype, read as f32: exact) against `want`, into `got`
fn rmsWords(g: *G, y: mlx.mlx_array, got: []f32, want: []const f32) !Words {
    _ = try g.hostF32(if (mlx.mlx_array_dtype(y) == .float32) y else try g.astype(y, .float32), got);
    const w = gateWords(got, want);
    return .{ .differ = w.differ, .bad = w.bad };
}

/// Runs `call`: the differing and non-finite words of its output against `want`
fn servedWords(comptime call: anytype, args: anytype, g: *G, got: []f32, want: []const f32) !usize {
    const m = g.mark();
    defer g.resetTo(m);
    const y = try @call(.auto, call, args);
    const w = try rmsWords(g, y, got, want);
    return w.differ + w.bad;
}

const RmsCase = struct { name: []const u8, in: u32, out: u32 };
/// the shared expert's down (1536 -> 3072) with the routed experts' up added: `addnorm_fused`'s
const addnorm_cases = [_]RmsCase{.{ .name = "shared.down", .in = 1536, .out = 3072 }};

/// linearRms' calls: from x and from x's own prepared rows (`root.linearPrep`: the source a prepared epilogue hands it)
const RmsSrcArm = enum { x, xp };

fn rmsCall(g: *G, s: mlx.mlx_stream, arm: RmsSrcArm, x: mlx.mlx_array, p: root.Proj, addend: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(x);
    const src: rms.Src = switch (arm) {
        .x => .{ .x = x },
        .xp => .{ .xp = try g.adopt(try root.linearPrep(s, x, p, dec)) },
    };
    return g.adopt(try rms.linearRms(s, src, dt, p, addend, w, rms_eps, dec));
}

/// The ADDNORM check: linearRms against rms_norm(f32(add) + f32(linear(x)))
fn rmsGpu(comptime cases: []const RmsCase, comptime tag: []const u8, seed: u64) !void {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer rms.unplant();
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var lines: usize = 0;
    var failed: usize = 0;
    for (cases) |c| {
        defer _ = mlx.mlx_clear_cache();
        const cm = g.mark();
        defer g.resetTo(cm);
        const p = try gateProj(a, &g, rnd, c.in, c.out);
        const w = try rmsWeight(a, &g, rnd, c.out);
        for (rms_rows) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const n: usize = @as(usize, rows) * c.out;
            const xh = try a.alloc(f32, @as(usize, rows) * c.in);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const ah = try a.alloc(f32, n);
            defer a.free(ah);
            for (ah) |*v| v.* = bfExact(8 * rnd.floatNorm(f32));
            const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ 1, ci(rows), ci(c.in) }, .float32);
            const a32 = try g.hostArray(std.mem.sliceAsBytes(ah), &.{ 1, ci(rows), ci(c.out) }, .float32);
            const got = try a.alloc(f32, n);
            defer a.free(got);
            const want = try a.alloc(f32, n);
            defer a.free(want);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const dm = g.mark();
                defer g.resetTo(dm);
                const x = try g.astype(x32, dt);
                // the other branch's y as the host holds it: the routed experts' up in the activation dtype
                const addend = try g.astype(a32, dt);
                const path = (try root.denseCached(.{ .in = c.in, .out = c.out, .rate = format.Rate.fromK(6), .dec = dec, .out_dtype = dt })).pathFor(rows);
                const yf = try projF32(&g, s, x, p);
                const pre = try g.add(try g.astype(addend, .float32), yf);
                _ = try g.hostF32(try mlxRms(&g, s, pre, w), want);
                for ([_]RmsSrcArm{ .x, .xp }) |arm| {
                    const d = try servedWords(rmsCall, .{ &g, s, arm, x, p, addend, w }, &g, got, want);
                    const ok = d == 0;
                    lines += 1;
                    failed += @intFromBool(!ok);
                    std.debug.print("TURBOEXL3_HY4F_FINPREP_" ++ tag ++ " {s} {d}->{d} K6 rows={d} path={s} x={s} src={s} words={d} differing={d} {s}\n", .{ c.name, c.in, c.out, rows, @tagName(path), @tagName(dt), @tagName(arm), n, d, if (ok) "PASS" else "FAIL" });
                }
                if (rows != rms_red_rows or dt != .bfloat16) continue;
                // the red check: the planted fault's words against the chain's must differ
                const f: rms.Fault = .no_add;
                try rms.plant(f);
                const red = rms.linearRms(s, .{ .x = x }, dt, p, addend, w, rms_eps, dec);
                rms.unplant();
                const rw = try rmsWords(&g, try g.adopt(try red), got, want);
                const caught = rw.differ > 0;
                lines += 1;
                failed += @intFromBool(!caught);
                std.debug.print("TURBOEXL3_HY4F_FINPREP_" ++ tag ++ "_RED {s} rows={d} x={s} fault={s} differing={d}/{d} (want > 0) {s}\n", .{ c.name, rows, @tagName(dt), @tagName(f), rw.differ, n, if (caught) "PASS" else "FAIL" });
            }
        }
    }
    std.debug.print("TURBOEXL3_HY4F_FINPREP_" ++ tag ++ "_DONE lines={d} failed={d} {s}\n", .{ lines, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, cases.len * (rms_rows.len * 2 * 2 + 1)), lines);
    try testing.expectEqual(@as(usize, 0), failed);
}

test "turboexl3 gpu hy4f finprep addnorm: linearRms with the add gives the chain's words (linear, f32 copies, MLX's add, rms_norm) at the shared down, MCG K6, rows 20 to 3584, bf16 and f32, from x and from prepared rows; the add left out moves them (TURBOEXL3_GPU=1)" {
    try rmsGpu(&addnorm_cases, "ADDNORM", 0x4F4F6164);
}

/// q_a (3072 -> 1024) with q_b at the indexed layers (32 heads of 256, theta 1e7, q_resid kept) and the sliding ones (64
/// heads, theta 1e4, no q_resid): `qnorm_fused`'s
const QnormCase = struct { name: []const u8, heads: u32, theta: f32, want_q: bool };
const qnorm_cases = [_]QnormCase{ .{ .name = "q_a.indexed", .heads = 32, .theta = 1e7, .want_q = true }, .{ .name = "q_a.sliding", .heads = 64, .theta = 1e4, .want_q = false } };
const qnorm_in: u32 = 3072;
const qnorm_lora: u32 = 1024;

fn qnormXp(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, w: mlx.mlx_array, q_b: root.Proj, want_q: bool) !mlx.mlx_array {
    const r = try rms.linearRmsPrep(s, .{ .x = x }, mlx.mlx_array_dtype(x), p, w, rms_eps, q_b, mlx.mlx_array_dtype(x), want_q, dec);
    if (r.q) |q| _ = try g.adopt(q);
    return g.adopt(r.xp);
}

fn qnormQ(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, w: mlx.mlx_array, q_b: root.Proj) !mlx.mlx_array {
    const r = try rms.linearRmsPrep(s, .{ .x = x }, mlx.mlx_array_dtype(x), p, w, rms_eps, q_b, mlx.mlx_array_dtype(x), true, dec);
    _ = try g.adopt(r.xp);
    return g.adopt(r.q.?);
}

test "turboexl3 gpu hy4f finprep qnorm: linearRmsPrep gives q_b's own prepared rows of the chain's normed query (linear, f32 copy, MLX's rms_norm, astype, q_b's prepare) and q_resid's words, and q_b from those rows gives its words (linear and the fused rope), at q_a and q_b's two widths, MCG K6, rows 20 to 3584, bf16 and f32; q_b's prepare without its suh moves them (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer rms.unplant();
    var prng = std.Random.DefaultPrng.init(0x4F4F716E);
    const rnd = prng.random();
    var lines: usize = 0;
    var failed: usize = 0;
    for (qnorm_cases) |c| {
        defer _ = mlx.mlx_clear_cache();
        const cm = g.mark();
        defer g.resetTo(cm);
        const out = c.heads * qrope_hd;
        const p = try gateProj(a, &g, rnd, qnorm_in, qnorm_lora);
        const q_b = try gateProj(a, &g, rnd, qnorm_lora, out);
        const w = try rmsWeight(a, &g, rnd, qnorm_lora);
        for (rms_rows) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const n: usize = @as(usize, rows) * qnorm_lora;
            const nb: usize = @as(usize, rows) * out;
            const xh = try a.alloc(f32, @as(usize, rows) * qnorm_in);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ 1, ci(rows), ci(qnorm_in) }, .float32);
            const got = try a.alloc(f32, nb);
            defer a.free(got);
            const want = try a.alloc(f32, nb);
            defer a.free(want);
            const pos: i32 = @intCast(rows * 3);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const dm = g.mark();
                defer g.resetTo(dm);
                const x = try g.astype(x32, dt);
                const path = (try root.denseCached(.{ .in = qnorm_in, .out = qnorm_lora, .rate = format.Rate.fromK(6), .dec = dec, .out_dtype = dt })).pathFor(rows);
                // the chain: q_resid, its cast, q_b's own prepare of it
                const q_resid = try mlxRms(&g, s, try projF32(&g, s, x, p), w);
                const qx = try g.astype(q_resid, dt);
                const xp_want = try g.adopt(try root.linearPrep(s, qx, q_b, dec));
                _ = try g.hostF32(try g.astype(xp_want, .float32), want[0..n]);
                const dxp = try servedWords(qnormXp, .{ &g, s, x, p, w, q_b, c.want_q }, &g, got[0..n], want[0..n]);
                var dq: usize = 0;
                if (c.want_q) {
                    _ = try g.hostF32(q_resid, want[0..n]);
                    dq = try servedWords(qnormQ, .{ &g, s, x, p, w, q_b }, &g, got[0..n], want[0..n]);
                }
                // q_b from the fused rows: `linear`'s words and the fused rope's
                const r = try rms.linearRmsPrep(s, .{ .x = x }, dt, p, w, rms_eps, q_b, dt, false, dec);
                const xp = try g.adopt(r.xp);
                _ = try g.hostF32(try g.astype(try g.adopt(try root.linear(s, qx, q_b.trellis, q_b.suh, q_b.svh, dec)), .float32), want);
                const wl = try rmsWords(&g, try g.adopt(try root.linearPrepared(s, xp, q_b, dec, dt)), got, want);
                _ = try g.hostF32(try g.adopt(try qrope.linearQRope(s, qx, q_b, c.theta, pos, qrope_hd, qrope_rope, dec)), want);
                const wr = try rmsWords(&g, try g.adopt(try qrope.linearQRopePrepared(s, xp, dt, q_b, c.theta, pos, qrope_hd, qrope_rope, dec)), got, want);
                const ok = dxp == 0 and dq == 0 and wl.differ + wl.bad == 0 and wr.differ + wr.bad == 0;
                lines += 1;
                failed += @intFromBool(!ok);
                std.debug.print("TURBOEXL3_HY4F_FINPREP_QNORM {s} {d}->{d}->{d} K6 rows={d} path={s} x={s} q_resid={s} xp_words={d} xp_differing={d} q_differing={d} q_b_linear_differing={d}/{d} q_b_rope_differing={d}/{d} {s}\n", .{ c.name, qnorm_in, qnorm_lora, out, rows, @tagName(path), @tagName(dt), if (c.want_q) "kept" else "none", n, dxp, dq, wl.differ + wl.bad, nb, wr.differ + wr.bad, nb, if (ok) "PASS" else "FAIL" });
                if (rows != rms_red_rows or dt != .bfloat16) continue;
                _ = try g.hostF32(try g.astype(xp_want, .float32), want[0..n]);
                try rms.plant(.prep_no_suh);
                const red = rms.linearRmsPrep(s, .{ .x = x }, dt, p, w, rms_eps, q_b, dt, false, dec);
                rms.unplant();
                const rr = try red;
                const rw = try rmsWords(&g, try g.adopt(rr.xp), got[0..n], want[0..n]);
                const caught = rw.differ > 0;
                lines += 1;
                failed += @intFromBool(!caught);
                std.debug.print("TURBOEXL3_HY4F_FINPREP_QNORM_RED {s} rows={d} x={s} fault=prep_no_suh differing={d}/{d} (want > 0) {s}\n", .{ c.name, rows, @tagName(dt), rw.differ, n, if (caught) "PASS" else "FAIL" });
            }
        }
    }
    std.debug.print("TURBOEXL3_HY4F_FINPREP_QNORM_DONE lines={d} failed={d} {s}\n", .{ lines, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, qnorm_cases.len * (rms_rows.len * 2 + 1)), lines);
    try testing.expectEqual(@as(usize, 0), failed);
}

/// kv_a as the pack stores it (3072 -> 640: the latent 512 and the rope part 64, then padding), at the indexed layers'
/// theta and the sliding ones': `kvnorm_fused`'s
const KvCase = struct { name: []const u8, theta: f32 };
const kv_cases = [_]KvCase{ .{ .name = "kv_a.indexed", .theta = 1e7 }, .{ .name = "kv_a.sliding", .theta = 1e4 } };
const kv_in: u32 = 3072;
const kv_out: u32 = 640;
const kv_lat: u32 = 512;
const kv_rope: u32 = 64;

fn kvC(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, w: mlx.mlx_array, theta: f32, pos: i32) !mlx.mlx_array {
    const r = try rms.linearKv(s, .{ .x = x }, mlx.mlx_array_dtype(x), p, w, rms_eps, kv_lat, kv_rope, theta, pos, dec);
    _ = try g.adopt(r.k_pe);
    return g.adopt(r.c_kv);
}

test "turboexl3 gpu hy4f finprep kvnorm: linearKv gives the chain's c_kv (linear, f32 copy, the latent's slice, MLX's rms_norm) and k_pe (the rope part's slice, MLX's fast::rope) at kv_a's stored width and both thetas, MCG K6, rows 33 to 3584 at positions 0 to 17459, bf16 and f32; the norm without its weight and MLX's CPU fallback frequency move them (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer rms.unplant();
    var prng = std.Random.DefaultPrng.init(0x4F4F6B76);
    const rnd = prng.random();
    var lines: usize = 0;
    var failed: usize = 0;
    for (kv_cases) |c| {
        defer _ = mlx.mlx_clear_cache();
        const cm = g.mark();
        defer g.resetTo(cm);
        const p = try gateProj(a, &g, rnd, kv_in, kv_out);
        const w = try rmsWeight(a, &g, rnd, kv_lat);
        for (qrope_rows) |rc| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const rows = rc.rows;
            const n: usize = @as(usize, rows) * kv_lat;
            const np: usize = @as(usize, rows) * kv_rope;
            const xh = try a.alloc(f32, @as(usize, rows) * kv_in);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ 1, ci(rows), ci(kv_in) }, .float32);
            const got = try a.alloc(f32, n);
            defer a.free(got);
            const want = try a.alloc(f32, n);
            defer a.free(want);
            const pe_want = try a.alloc(f32, np);
            defer a.free(pe_want);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const dm = g.mark();
                defer g.resetTo(dm);
                const x = try g.astype(x32, dt);
                const path = (try root.denseCached(.{ .in = kv_in, .out = kv_out, .rate = format.Rate.fromK(6), .dec = dec, .out_dtype = dt })).pathFor(rows);
                // the chain (hy4f's attention): the projection's first lat + rope values, the latent's norm, the rope
                // part's rope as [1, 1, S, rope]
                const yf = try projF32(&g, s, x, p);
                const lat = try g.slice(yf, &.{ 0, 0, 0 }, &.{ 1, ci(rows), ci(kv_lat) }, &.{ 1, 1, 1 });
                _ = try g.hostF32(try mlxRms(&g, s, lat, w), want);
                const pe = try g.reshape(try g.slice(yf, &.{ 0, 0, ci(kv_lat) }, &.{ 1, ci(rows), ci(kv_lat + kv_rope) }, &.{ 1, 1, 1 }), &.{ 1, 1, ci(rows), ci(kv_rope) });
                _ = try g.hostF32(try mlxRope(&g, s, pe, kv_rope, c.theta, rc.pos), pe_want);
                const d = try servedWords(kvC, .{ &g, s, x, p, w, c.theta, rc.pos }, &g, got, want);
                const r = try rms.linearKv(s, .{ .x = x }, dt, p, w, rms_eps, kv_lat, kv_rope, c.theta, rc.pos, dec);
                _ = try g.adopt(r.c_kv);
                try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(kv_rope) }, mlx.getShape(r.k_pe));
                const wp = try rmsWords(&g, try g.adopt(r.k_pe), got[0..np], pe_want);
                const ok = d == 0 and wp.differ + wp.bad == 0;
                lines += 1;
                failed += @intFromBool(!ok);
                std.debug.print("TURBOEXL3_HY4F_FINPREP_KVNORM {s} {d}->{d} lat={d} rope={d} theta={d} K6 rows={d} pos={d} path={s} x={s} c_kv_words={d} c_kv_differing={d} k_pe_differing={d}/{d} {s}\n", .{ c.name, kv_in, kv_out, kv_lat, kv_rope, c.theta, rows, rc.pos, @tagName(path), @tagName(dt), n, d, wp.differ + wp.bad, np, if (ok) "PASS" else "FAIL" });
                if (rows != qrope_red_rows or dt != .bfloat16) continue;
                for ([_]rms.Fault{ .kv_no_weight, .kv_freq_ln }) |f| {
                    try rms.plant(f);
                    const red = rms.linearKv(s, .{ .x = x }, dt, p, w, rms_eps, kv_lat, kv_rope, c.theta, rc.pos, dec);
                    rms.unplant();
                    const rr = try red;
                    const rc_w = try rmsWords(&g, try g.adopt(rr.c_kv), got, want);
                    const rp_w = try rmsWords(&g, try g.adopt(rr.k_pe), got[0..np], pe_want);
                    // each fault moves its own output and no other
                    const caught = if (f == .kv_no_weight) rc_w.differ > 0 and rp_w.differ == 0 else rp_w.differ > 0 and rc_w.differ == 0;
                    lines += 1;
                    failed += @intFromBool(!caught);
                    std.debug.print("TURBOEXL3_HY4F_FINPREP_KVNORM_RED {s} rows={d} pos={d} x={s} fault={s} c_kv_differing={d}/{d} k_pe_differing={d}/{d} (want the fault's own output > 0, the other 0) {s}\n", .{ c.name, rows, rc.pos, @tagName(dt), @tagName(f), rc_w.differ, n, rp_w.differ, np, if (caught) "PASS" else "FAIL" });
                }
            }
        }
    }
    std.debug.print("TURBOEXL3_HY4F_FINPREP_KVNORM_DONE lines={d} failed={d} {s}\n", .{ lines, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, kv_cases.len * (qrope_rows.len * 2 + 2)), lines);
    try testing.expectEqual(@as(usize, 0), failed);
}

/// The shared expert's up and down (3072 -> 1536 -> 3072) and the dense MLP's (3072 -> 13824 -> 3072):
/// `down_prep_fused`'s
const downprep_cases = [_]SwigluCase{ .{ .name = "shared.up_down", .out = 1536 }, .{ .name = "dense.up_down", .out = 13824 } };

test "turboexl3 gpu hy4f finprep downprep: linearSwigluPrep gives the down linear's own prepared rows of linearSwigluMul's h, and the down linear from those rows gives its words, at the shared expert's and the dense MLP's widths, MCG K6, rows 20 to 3584, bf16 and f32; the prepare without its suh moves them (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer rms.unplant();
    var prng = std.Random.DefaultPrng.init(0x4F4F6470);
    const rnd = prng.random();
    var lines: usize = 0;
    var failed: usize = 0;
    for (downprep_cases) |c| {
        defer _ = mlx.mlx_clear_cache();
        const cm = g.mark();
        defer g.resetTo(cm);
        const gp = try gateProj(a, &g, rnd, swiglu_in, c.out);
        const up = try gateProj(a, &g, rnd, swiglu_in, c.out);
        const down = try gateProj(a, &g, rnd, c.out, swiglu_in);
        for (swiglu_rows) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const n: usize = @as(usize, rows) * c.out;
            const nd: usize = @as(usize, rows) * swiglu_in;
            const xh = try a.alloc(f32, @as(usize, rows) * swiglu_in);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ 1, ci(rows), ci(swiglu_in) }, .float32);
            const got = try a.alloc(f32, @max(n, nd));
            defer a.free(got);
            const want = try a.alloc(f32, @max(n, nd));
            defer a.free(want);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const dm = g.mark();
                defer g.resetTo(dm);
                const x = try g.astype(x32, dt);
                const path = (try root.denseCached(.{ .in = swiglu_in, .out = c.out, .rate = format.Rate.fromK(6), .dec = dec, .out_dtype = dt })).pathFor(rows);
                const gy = try g.adopt(try root.linear(s, x, gp.trellis, gp.suh, gp.svh, dec));
                const h = try g.adopt(try swiglu.linearSwigluMul(s, x, up, gy, dec, dt));
                _ = try g.hostF32(try g.astype(try g.adopt(try root.linearPrep(s, h, down, dec)), .float32), want[0..n]);
                const xp = try g.adopt(try rms.linearSwigluPrep(s, x, up, gy, down, dec, dt));
                const wx = try rmsWords(&g, xp, got[0..n], want[0..n]);
                _ = try g.hostF32(try g.astype(try g.adopt(try root.linear(s, h, down.trellis, down.suh, down.svh, dec)), .float32), want[0..nd]);
                const wd = try rmsWords(&g, try g.adopt(try root.linearPrepared(s, xp, down, dec, dt)), got[0..nd], want[0..nd]);
                const ok = wx.differ + wx.bad == 0 and wd.differ + wd.bad == 0;
                lines += 1;
                failed += @intFromBool(!ok);
                std.debug.print("TURBOEXL3_HY4F_FINPREP_DOWNPREP {s} {d}->{d}->{d} K6 rows={d} path={s} x={s} h={s} xp_words={d} xp_differing={d} down_differing={d}/{d} {s}\n", .{ c.name, swiglu_in, c.out, swiglu_in, rows, @tagName(path), @tagName(dt), @tagName(dt), n, wx.differ + wx.bad, wd.differ + wd.bad, nd, if (ok) "PASS" else "FAIL" });
                if (rows != swiglu_red_rows or dt != .bfloat16) continue;
                _ = try g.hostF32(try g.astype(try g.adopt(try root.linearPrep(s, h, down, dec)), .float32), want[0..n]);
                try rms.plant(.swiglu_no_suh);
                const red = rms.linearSwigluPrep(s, x, up, gy, down, dec, dt);
                rms.unplant();
                const rw = try rmsWords(&g, try g.adopt(try red), got[0..n], want[0..n]);
                const caught = rw.differ > 0;
                lines += 1;
                failed += @intFromBool(!caught);
                std.debug.print("TURBOEXL3_HY4F_FINPREP_DOWNPREP_RED {s} rows={d} x={s} fault=swiglu_no_suh differing={d}/{d} (want > 0) {s}\n", .{ c.name, rows, @tagName(dt), rw.differ, n, if (caught) "PASS" else "FAIL" });
            }
        }
    }
    std.debug.print("TURBOEXL3_HY4F_FINPREP_DOWNPREP_DONE lines={d} failed={d} {s}\n", .{ lines, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, downprep_cases.len * (swiglu_rows.len * 2 + 1)), lines);
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── pg2 from prepared rows (`root.linearOnPrepared`, `gate.pg2OfPrepared`) and the gate's prep finish (the gate's
// finish writing o_proj's prepared rows, `linearGateMulPrep`). Names hold finprep. ──

test "turboexl3 hy4f finprep gateprep texts and refusals: the prep store loop is the gate's with o_proj's prepare after its value (finish and pg2 launches); wrong next linears and decode rows refuse before any engine" {
    try testing.expectEqualStrings(dense.finish_v ++ gate.gate_open_1 ++ gate.gate_open_2 ++ gate.gate_prep_open ++ gate.gate_open_3 ++ gate.gate_sig ++ gate.gate_src ++ gate.gate_close_prep ++ dense.prep_2_2 ++ dense.prep_3, gate.text_finish_gate_prep);
    try testing.expectEqualStrings(comptime gate.pg2Text(gate.gate_open_1 ++ gate.gate_open_2_pg2 ++ gate.gate_prep_open ++ gate.gate_open_3 ++ gate.gate_sig ++ gate.gate_src ++ gate.gate_close_prep ++ dense.prep_2_2 ++ dense.prep_3), gate.text_pg2_gate_prep);
    // the same product as gate_close stores, then prep_2_1's load expression on it with o_proj's suh
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, gate.gate_close, "G_T(ov * sg)"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, gate.gate_close_prep, "hp[j] = float(half(float(G_T(ov * sg)))) * float(suh_o[o]);"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, dense.prep_2_1, "float(half(float(x[row * DIMC + i]))) * float(suh[i])"));
    // no launch: every refusal comes before the stream is used
    const s: mlx.mlx_stream = .{ .ctx = null };
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const zeros = try a.alloc(u32, 1 << 20);
    defer a.free(zeros);
    @memset(zeros, 0);
    const bytes = std.mem.sliceAsBytes(zeros);
    const gp = try zeroProj(&g, bytes, 128, 256);
    const o_ok = try zeroProj(&g, bytes, 256, 128);
    const o_bad = try zeroProj(&g, bytes, 128, 128);
    const o_f32: root.Proj = .{ .trellis = o_ok.trellis, .suh = try g.hostArray(bytes[0 .. 256 * 4], &.{256}, .float32), .svh = o_ok.svh };
    const held = root.enginesHeld().dense;
    const Case = struct { rows: c_int = 33, o: root.Proj, want: anyerror };
    for ([_]Case{
        .{ .rows = 16, .o = o_ok, .want = error.Exl3EngineNotImplemented },
        .{ .o = o_bad, .want = error.BadExl3Shape },
        .{ .o = o_f32, .want = error.BadExl3Dtype },
    }) |c| {
        const m = g.mark();
        defer g.resetTo(m);
        const x = try g.hostArray(bytes[0..@intCast(c.rows * 128 * 4)], &.{ 1, c.rows, 128 }, .float32);
        const o_src = try g.hostArray(bytes[0..@intCast(c.rows * 256 * 4)], &.{ 1, c.rows, 2, 128 }, .float32);
        try testing.expectError(c.want, gate.linearGateMulPrep(s, x, gp, o_src, .flat, c.o, dec, .bfloat16));
    }
    // root.linearOnPrepared: the decode rows, an xp that is not f16, a dtype that is not a float
    {
        const m = g.mark();
        defer g.resetTo(m);
        const xp16 = try g.hostArray(bytes[0 .. 16 * 256 * 2], &.{ 16, 256 }, .float16);
        try testing.expectError(error.Exl3EngineNotImplemented, root.linearOnPrepared(s, xp16, o_ok, dec, .bfloat16));
        const xpb = try g.hostArray(bytes[0 .. 33 * 256 * 2], &.{ 33, 256 }, .bfloat16);
        try testing.expectError(error.BadExl3Shape, root.linearOnPrepared(s, xpb, o_ok, dec, .bfloat16));
        const xp33 = try g.hostArray(bytes[0 .. 33 * 256 * 2], &.{ 33, 256 }, .float16);
        try testing.expectError(error.BadExl3Dtype, root.linearOnPrepared(s, xp33, o_ok, dec, .int32));
    }
    try testing.expectEqual(held, root.enginesHeld().dense);
}

/// The served chunk rows (3584, the 17,492-token prompt's 3125-row tail) and its last 31 rows (the register path)
const prep_rows = [_]u32{ 3584, 3125, 31 };

/// pg2 off, or on at the served table (split K too)
const PrepPg2 = enum { off, served };

fn prepPg2Set(st: PrepPg2) void {
    switch (st) {
        .off => hy4f_pg2.unserve(),
        .served => hy4f_pg2.serve(true),
    }
}

fn pg2Calls() u64 {
    return pgemm.calls.get().*.load(.monotonic);
}

test "turboexl3 gpu hy4f finprep prepared pg2: root.linearOnPrepared on linear's own prepare gives linear's words and the same pg2 launches (the served table, split K at o_proj.sliding), and linearQRopePrepared runs pg2's launch at q_b's shapes with linearQRope's words, rows 3584, 3125 and 31, bf16 and f32; red: qrope's NeoX pairs in the prepared pg2 launch move the rope words (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = pg2Save();
    defer pg2Restore(saved);
    defer qrope.unplant();
    var prng = std.Random.DefaultPrng.init(0x4F4F7067);
    const rnd = prng.random();
    var lines: usize = 0;
    var failed: usize = 0;
    var pg2_lines: usize = 0;
    // the plain linears the prepared finishes reach: o_proj (the gate's prep finish), the shared down (the SwiGLU's); q_b's
    // (the rope finish) below
    const Lin = struct { name: []const u8, k: u32, n: u32 };
    for ([_]Lin{ .{ .name = "o_proj.indexed", .k = 8192, .n = 3072 }, .{ .name = "o_proj.sliding", .k = 16384, .n = 3072 }, .{ .name = "shared.down", .k = 1536, .n = 3072 } }) |l| {
        defer _ = mlx.mlx_clear_cache();
        const cm = g.mark();
        defer g.resetTo(cm);
        const p = try gateProj(a, &g, rnd, l.k, l.n);
        for (prep_rows) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const n: usize = @as(usize, rows) * l.n;
            const xh = try a.alloc(f32, @as(usize, rows) * l.k);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(l.k) }, .float32);
            const got = try a.alloc(f32, n);
            defer a.free(got);
            const want = try a.alloc(f32, n);
            defer a.free(want);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| for ([_]PrepPg2{ .off, .served }) |st| {
                const dm = g.mark();
                defer g.resetTo(dm);
                const x = try g.astype(x32, dt);
                prepPg2Set(st);
                const c0 = pg2Calls();
                _ = try g.hostF32(try g.astype(try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, dec)), .float32), want);
                const c1 = pg2Calls();
                const xp = try g.adopt(try root.linearPrep(s, x, p, dec));
                const c2 = pg2Calls();
                const w = try rmsWords(&g, try g.adopt(try root.linearOnPrepared(s, xp, p, dec, dt)), got, want);
                const c3 = pg2Calls();
                hy4f_pg2.unserve();
                const fused_linear = c1 - c0;
                const fused_prep = c3 - c2;
                const ok = w.differ + w.bad == 0 and fused_linear == fused_prep;
                lines += 1;
                pg2_lines += @intFromBool(fused_prep > 0);
                failed += @intFromBool(!ok);
                std.debug.print("TURBOEXL3_HY4F_FINPREP_PREPPG2 {s} {d}->{d} K6 rows={d} x={s} pg2={s} pg2_launches_linear={d} pg2_launches_prepared={d} words={d} differing={d} {s}\n", .{ l.name, l.k, l.n, rows, @tagName(dt), @tagName(st), fused_linear, fused_prep, n, w.differ + w.bad, if (ok) "PASS" else "FAIL" });
            };
        }
    }
    // q_b's rope finish from prepared rows, pg2's launch at its served shapes
    for (qrope_cases) |c| {
        defer _ = mlx.mlx_clear_cache();
        const cm = g.mark();
        defer g.resetTo(cm);
        const out = c.heads * qrope_hd;
        const p = try gateProj(a, &g, rnd, qrope_in, out);
        for (prep_rows) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const n: usize = @as(usize, rows) * out;
            const xh = try a.alloc(f32, @as(usize, rows) * qrope_in);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(qrope_in) }, .float32), .bfloat16);
            const got = try a.alloc(f32, n);
            defer a.free(got);
            const want = try a.alloc(f32, n);
            defer a.free(want);
            for ([_]PrepPg2{ .off, .served }) |st| {
                const dm = g.mark();
                defer g.resetTo(dm);
                prepPg2Set(st);
                _ = try g.hostF32(try g.adopt(try qrope.linearQRope(s, x, p, c.theta, 13905, qrope_hd, qrope_rope, dec)), want);
                const xp = try g.adopt(try root.linearPrep(s, x, p, dec));
                const pred = try qrope.preparedPg2(p, rows, .bfloat16, dec);
                const c0 = pg2Calls();
                _ = try g.hostF32(try g.adopt(try qrope.linearQRopePrepared(s, xp, .bfloat16, p, c.theta, 13905, qrope_hd, qrope_rope, dec)), got);
                const fused = pg2Calls() - c0;
                const w = qropeWords(got, want);
                // the predicate the host's switch reads says pg2's launch exactly where the prepared call runs it
                const ok = w.nope + w.rope + w.bad == 0 and pred == (fused > 0) and (st == .off or rows == 31 or fused > 0);
                lines += 1;
                pg2_lines += @intFromBool(fused > 0);
                failed += @intFromBool(!ok);
                std.debug.print("TURBOEXL3_HY4F_FINPREP_PREPPG2 {s} 1024->{d} K6 rows={d} x=bfloat16 pg2={s} prepared_pg2_predicate={} pg2_launches_prepared={d} differing_nope={d} differing_rope={d}/{d} {s}\n", .{ c.name, out, rows, @tagName(st), pred, fused, w.nope, w.rope, n, if (ok) "PASS" else "FAIL" });
                if (st == .served and rows == 3584 and c.heads == 64) {
                    try qrope.plant(.pair_neox);
                    const r0 = pg2Calls();
                    const red = qrope.linearQRopePrepared(s, xp, .bfloat16, p, c.theta, 13905, qrope_hd, qrope_rope, dec);
                    const rf = pg2Calls() - r0;
                    qrope.unplant();
                    _ = try g.hostF32(try g.adopt(try red), got);
                    const rw = qropeWords(got, want);
                    const caught = rw.rope > 0 and rw.nope == 0 and rf > 0;
                    lines += 1;
                    failed += @intFromBool(!caught);
                    std.debug.print("TURBOEXL3_HY4F_FINPREP_PREPPG2_RED {s} rows={d} fault=pair_neox pg2_launches={d} differing_rope={d} differing_nope={d}/{d} (want rope > 0, nope 0, pg2 > 0) {s}\n", .{ c.name, rows, rf, rw.rope, rw.nope, n, if (caught) "PASS" else "FAIL" });
                }
                hy4f_pg2.unserve();
            }
        }
    }
    std.debug.print("TURBOEXL3_HY4F_FINPREP_PREPPG2_DONE lines={d} pg2_lines={d} failed={d} {s}\n", .{ lines, pg2_lines, failed, if (failed == 0 and pg2_lines > 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 3 * 3 * 2 * 2 + qrope_cases.len * 3 * 2 + 1), lines);
    try testing.expect(pg2_lines > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}

test "turboexl3 gpu hy4f finprep gateprep: linearGateMulPrep gives o_proj's own prepare of linearGateMulAt's product, and o_proj from it (root.linearOnPrepared) gives o_proj's words on that product, at linear_gate's two widths, o in its rows' layout, rows 3584, 3125 and 31, bf16 and f32, pg2 off and on the served table; red: o_proj's prepare without its suh moves them in the finish's launch and pg2's (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = pg2Save();
    defer pg2Restore(saved);
    defer gate.unplant();
    var prng = std.Random.DefaultPrng.init(0x4F4F6770);
    const rnd = prng.random();
    var lines: usize = 0;
    var failed: usize = 0;
    var pg2_lines: usize = 0;
    for (gate_cases) |c| {
        defer _ = mlx.mlx_clear_cache();
        const cm = g.mark();
        defer g.resetTo(cm);
        const dim = c.heads * gate_vh;
        const gp = try gateProj(a, &g, rnd, gate_in, dim);
        const op = try gateProj(a, &g, rnd, dim, gate_in);
        for (prep_rows) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const n: usize = @as(usize, rows) * dim;
            const no: usize = @as(usize, rows) * gate_in;
            const xh = try a.alloc(f32, @as(usize, rows) * gate_in);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(rnd.floatNorm(f32));
            const oh = try a.alloc(f32, n);
            defer a.free(oh);
            for (oh) |*v| v.* = rnd.floatNorm(f32);
            const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ 1, ci(rows), ci(gate_in) }, .float32);
            const o_rows = try g.hostArray(std.mem.sliceAsBytes(oh), &.{ 1, ci(rows), ci(c.heads), ci(gate_vh) }, .float32);
            const got = try a.alloc(f32, @max(n, no));
            defer a.free(got);
            const want = try a.alloc(f32, @max(n, no));
            defer a.free(want);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| for ([_]PrepPg2{ .off, .served }) |st| {
                const dm = g.mark();
                defer g.resetTo(dm);
                const x = try g.astype(x32, dt);
                prepPg2Set(st);
                // the chain: the gate's product, o_proj's own prepare of it, o_proj on it
                const gated = try g.adopt(try gate.linearGateMulAt(s, x, gp, o_rows, .flat, dec, dt));
                _ = try g.hostF32(try g.astype(try g.adopt(try root.linearPrep(s, gated, op, dec)), .float32), want[0..n]);
                const c0 = pg2Calls();
                const xp = try g.adopt(try gate.linearGateMulPrep(s, x, gp, o_rows, .flat, op, dec, dt));
                const c1 = pg2Calls();
                const wx = try rmsWords(&g, xp, got[0..n], want[0..n]);
                _ = try g.hostF32(try g.astype(try g.adopt(try root.linear(s, gated, op.trellis, op.suh, op.svh, dec)), .float32), want[0..no]);
                const c2 = pg2Calls();
                const wy = try rmsWords(&g, try g.adopt(try root.linearOnPrepared(s, xp, op, dec, dt)), got[0..no], want[0..no]);
                const c3 = pg2Calls();
                hy4f_pg2.unserve();
                const ok = wx.differ + wx.bad == 0 and wy.differ + wy.bad == 0;
                lines += 1;
                pg2_lines += @intFromBool(c1 - c0 + c3 - c2 > 0);
                failed += @intFromBool(!ok);
                std.debug.print("TURBOEXL3_HY4F_FINPREP_GATEPREP {s} {d}->{d}->{d} K6 rows={d} x={s} pg2={s} gate_pg2_launches={d} o_proj_pg2_launches={d} xp_words={d} xp_differing={d} o_proj_differing={d}/{d} {s}\n", .{ c.name, gate_in, dim, gate_in, rows, @tagName(dt), @tagName(st), c1 - c0, c3 - c2, n, wx.differ + wx.bad, wy.differ + wy.bad, no, if (ok) "PASS" else "FAIL" });
                if (rows != 3584 or dt != .bfloat16 or c.heads != 64) continue;
                // the red check in this pg2 state's launch: o_proj's prepare without its suh
                _ = try g.hostF32(try g.astype(try g.adopt(try root.linearPrep(s, gated, op, dec)), .float32), want[0..n]);
                prepPg2Set(st);
                try gate.plant(.prep_no_suh);
                const r0 = pg2Calls();
                const red = gate.linearGateMulPrep(s, x, gp, o_rows, .flat, op, dec, dt);
                const rf = pg2Calls() - r0;
                gate.unplant();
                hy4f_pg2.unserve();
                const rw = try rmsWords(&g, try g.adopt(try red), got[0..n], want[0..n]);
                const caught = rw.differ > 0 and (st == .off or rf > 0);
                lines += 1;
                failed += @intFromBool(!caught);
                std.debug.print("TURBOEXL3_HY4F_FINPREP_GATEPREP_RED {s} rows={d} pg2={s} fault=prep_no_suh pg2_launches={d} differing={d}/{d} (want > 0) {s}\n", .{ c.name, rows, @tagName(st), rf, rw.differ, n, if (caught) "PASS" else "FAIL" });
            };
        }
    }
    std.debug.print("TURBOEXL3_HY4F_FINPREP_GATEPREP_DONE lines={d} pg2_lines={d} failed={d} {s}\n", .{ lines, pg2_lines, failed, if (failed == 0 and pg2_lines > 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, gate_cases.len * 3 * 2 * 2 + 2), lines);
    try testing.expect(pg2_lines > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}
