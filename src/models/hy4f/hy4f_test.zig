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
