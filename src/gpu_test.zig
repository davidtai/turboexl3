//! The decode chain's and the dense linear's GPU parity on whatever GPU runs the tests (TURBOEXL3_GPU=1, else skipped):
//! synthetic banks (`ref.synthBank`) at several rates, codebooks and windows, the two-launch chain against the host
//! reference (`ref.moeRowDec`), then the chain's time per layer. Lines: TURBOEXL3_PARITY per case, TURBOEXL3_TIME per timed
//! case. Parity is a tolerance (max |err| over the reference's rms): the kernels reduce in a different order than the
//! reference and the f16 rounding points can flip on a last bit. A path that promises another path's bytes (the later
//! decode forms, the fused shared expert, the router in form 9's prepare launch, the one-launch decode linear, the
//! two-stream split, the resident trunk) is held to them bit for bit; the un-rotated trunk to its stated bars. Besides the
//! chain's timed cases, the dense linear's timing (TURBOEXL3_DENSE_TIME=1) is the only benchmark.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("test_support.zig");
const G = support.Graph;
const format = @import("format.zig");
const decode = @import("decode.zig");
const ref = @import("ref.zig");
const root = @import("root.zig");

const testing = std.testing;

const Case = struct {
    hidden: u32,
    inter: u32,
    topk: u32,
    rows: u32,
    experts: u32,
    n: u32,
    dec: format.Decode,
    timed: bool = false,
};

const cases = [_]Case{
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 32, .dec = .mul1 },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 40, .dec = .mul1 },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 48, .dec = .{ .codebook = .mcg, .window = .w12 } },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 64, .dec = .mul1 },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 80, .dec = .mul1 },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 128, .dec = .mcg },
    // Qwen3.8-Flash-Next's geometry at the served rate: 1 row (decode), 4 and 8 rows (MTP verify widths)
    .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 1, .experts = 12, .n = 80, .dec = .mul1, .timed = true },
    .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 4, .experts = 12, .n = 80, .dec = .mul1, .timed = true },
    .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 8, .experts = 12, .n = 80, .dec = .mul1, .timed = true },
};

const tolerance: f64 = 5e-3;

fn ci(x: u32) c_int {
    return @intCast(x);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn runCase(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, c: Case) !void {
    var prng = std.Random.DefaultPrng.init(0xE7135EED ^ c.n);
    const rnd = prng.random();
    const rate = format.Rate.fromN(c.n).?;
    var bank = try ref.synthBank(a, rnd, c.experts, c.hidden, c.inter, rate);
    defer bank.deinit(a);
    const nslots = c.rows * c.topk;
    const x = try a.alloc(f32, c.rows * c.hidden);
    defer a.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const slots = try a.alloc(u32, nslots);
    defer a.free(slots);
    const scores = try a.alloc(f32, nslots);
    defer a.free(scores);
    for (0..c.rows) |r| {
        var sum: f32 = 0;
        for (0..c.topk) |k| {
            // distinct experts per row, two rows sharing some
            slots[r * c.topk + k] = @intCast((k * 2 + r) % c.experts);
            scores[r * c.topk + k] = 0.2 + rnd.float(f32);
            sum += scores[r * c.topk + k];
        }
        for (0..c.topk) |k| scores[r * c.topk + k] /= sum;
    }
    // host reference
    const y_ref = try a.alloc(f32, c.rows * c.hidden);
    defer a.free(y_ref);
    for (0..c.rows) |r| try ref.moeRowDec(a, &bank, c.dec, x[r * c.hidden ..][0..c.hidden], slots[r * c.topk ..][0..c.topk], scores[r * c.topk ..][0..c.topk], y_ref[r * c.hidden ..][0..c.hidden]);
    // device
    const mark = g.mark();
    defer g.resetTo(mark);
    const ht = ci(c.hidden / 16);
    const it = ci(c.inter / 16);
    const arrays: decode.Kernels.Bank = .{
        .gate_t = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(c.experts), ht, it, ci(c.n) }, .uint16),
        .gate_suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ ci(c.experts), ci(c.hidden) }, .float16),
        .gate_svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ ci(c.experts), ci(c.inter) }, .float16),
        .up_t = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ ci(c.experts), ht, it, ci(c.n) }, .uint16),
        .up_suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ ci(c.experts), ci(c.hidden) }, .float16),
        .up_svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ ci(c.experts), ci(c.inter) }, .float16),
        .down_t = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ ci(c.experts), it, ht, ci(c.n) }, .uint16),
        .down_suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ ci(c.experts), ci(c.inter) }, .float16),
        .down_svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ ci(c.experts), ci(c.hidden) }, .float16),
    };
    const xa = try g.hostArray(std.mem.sliceAsBytes(x), &.{ ci(c.rows), ci(c.hidden) }, .float32);
    const sa = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(nslots)}, .uint32);
    const wa = try g.hostArray(std.mem.sliceAsBytes(scores), &.{ci(nslots)}, .float32);
    var k = try decode.Kernels.init(.{ .hidden = c.hidden, .inter = c.inter, .topk = c.topk, .rate = rate, .dec = c.dec });
    defer k.deinit();
    const out = try k.moe(s, xa, sa, wa, arrays, c.rows);
    _ = try g.adopt(out.h);
    const y = try g.adopt(out.y);
    const got = try a.alloc(f32, c.rows * c.hidden);
    defer a.free(got);
    _ = try g.hostF32(y, got);
    const e = ref.compare(got, y_ref);
    var wbuf: [8]u8 = undefined;
    std.debug.print("TURBOEXL3_PARITY n={d} cb={s} w={d} geo={d}x{d} rows={d} topk={d} max_abs={e:.3} rms={e:.3} ratio={e:.3} {s}\n", .{
        c.n,                                              @tagName(c.dec.codebook), c.dec.window.bits(), c.hidden, c.inter, c.rows, c.topk, e.max_abs, e.rms_ref, e.max_ratio,
        if (e.max_ratio <= tolerance) "PASS" else "FAIL",
    });
    _ = &wbuf;
    try testing.expect(std.math.isFinite(e.max_abs));
    try testing.expect(e.max_ratio <= tolerance);
    if (!c.timed) return;
    // timing: the two launches, evaluated, median of 30 after 5 warm
    var times: [30]u64 = undefined;
    var i: usize = 0;
    while (i < 35) : (i += 1) {
        const t0 = nowNs();
        const o = try k.moe(s, xa, sa, wa, arrays, c.rows);
        try mlx.check(mlx.mlx_array_eval(o.y));
        const dt = nowNs() - t0;
        _ = mlx.mlx_array_free(o.h);
        _ = mlx.mlx_array_free(o.y);
        if (i >= 5) times[i - 5] = dt;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    const bytes = @as(f64, @floatFromInt(c.topk)) * 3.0 * @as(f64, @floatFromInt(c.hidden)) * @as(f64, @floatFromInt(c.inter)) * @as(f64, @floatFromInt(c.n)) / 128.0;
    const med_us = @as(f64, @floatFromInt(times[15])) / 1000.0;
    // the launches apart: gate/up alone, then down alone on a fixed h
    var t_gu: [30]u64 = undefined;
    var t_dn: [30]u64 = undefined;
    const h_fixed = try k.gateUp(s, xa, sa, arrays, c.rows);
    defer _ = mlx.mlx_array_free(h_fixed);
    try mlx.check(mlx.mlx_array_eval(h_fixed));
    i = 0;
    while (i < 35) : (i += 1) {
        var t0 = nowNs();
        const hh = try k.gateUp(s, xa, sa, arrays, c.rows);
        try mlx.check(mlx.mlx_array_eval(hh));
        const d1 = nowNs() - t0;
        _ = mlx.mlx_array_free(hh);
        t0 = nowNs();
        const yy = try k.downReduce(s, h_fixed, sa, wa, arrays, c.rows);
        try mlx.check(mlx.mlx_array_eval(yy));
        const d2 = nowNs() - t0;
        _ = mlx.mlx_array_free(yy);
        if (i >= 5) {
            t_gu[i - 5] = d1;
            t_dn[i - 5] = d2;
        }
    }
    std.mem.sort(u64, &t_gu, {}, std.sort.asc(u64));
    std.mem.sort(u64, &t_dn, {}, std.sort.asc(u64));
    const dist = @as(f64, @floatFromInt(c.rows)) * bytes; // every row its own experts here: bytes scale with rows
    std.debug.print("TURBOEXL3_TIME n={d} geo={d}x{d} rows={d} topk={d} median_us={d:.1} p10_us={d:.1} p90_us={d:.1} gu_us={d:.1} down_us={d:.1} expert_gbps={d:.1}\n", .{
        c.n,                                        c.hidden,                                   c.inter,                      c.rows, c.topk, med_us, @as(f64, @floatFromInt(times[3])) / 1000.0, @as(f64, @floatFromInt(times[27])) / 1000.0,
        @as(f64, @floatFromInt(t_gu[15])) / 1000.0, @as(f64, @floatFromInt(t_dn[15])) / 1000.0, dist / (med_us * 1e-6) / 1e9,
    });
    // the ALU shares: the same chain without the codebook hash (probe 1) and without the funnel either (probe 2)
    for ([_]u8{ 1, 2 }) |probe| {
        var kp = try decode.Kernels.init(.{ .hidden = c.hidden, .inter = c.inter, .topk = c.topk, .rate = rate, .dec = c.dec, .probe = probe });
        defer kp.deinit();
        var tp: [30]u64 = undefined;
        i = 0;
        while (i < 35) : (i += 1) {
            const t0 = nowNs();
            const o = try kp.moe(s, xa, sa, wa, arrays, c.rows);
            try mlx.check(mlx.mlx_array_eval(o.y));
            const dt = nowNs() - t0;
            _ = mlx.mlx_array_free(o.h);
            _ = mlx.mlx_array_free(o.y);
            if (i >= 5) tp[i - 5] = dt;
        }
        std.mem.sort(u64, &tp, {}, std.sort.asc(u64));
        std.debug.print("TURBOEXL3_PROBE n={d} rows={d} probe={d} median_us={d:.1} p10_us={d:.1}\n", .{ c.n, c.rows, probe, @as(f64, @floatFromInt(tp[15])) / 1000.0, @as(f64, @floatFromInt(tp[3])) / 1000.0 });
    }
}

test "turboexl3 gpu: the two-launch decode chain equals the host reference at every rate and codebook (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    for (cases) |c| try runCase(a, &g, s, c);
}

// The host's path: `root.moe` on a bf16 x [1, rows, D], i32 inds [1, rows, K] and bf16 scores, output in x's dtype.
// The routed bank (10 of 12 experts at K5) and the host's shared expert as a one-expert bank (top-1 at K7, every row on expert
// 0).
test "turboexl3 gpu: root.moe (the sushi_exl3-shaped entry) equals the reference through the bf16 host operands (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    try rootMoeParity(&g, s, a, .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 3, .experts = 12, .n = 80, .dec = .mul1 }, 0xB16, true);
    try rootMoeParity(&g, s, a, .{ .hidden = 2560, .inter = 640, .topk = 1, .rows = 4, .experts = 1, .n = 112, .dec = .mul1 }, 0x5A4ED, false);
}

fn rootMoeParity(g: *G, s: mlx.mlx_stream, a: std.mem.Allocator, c: Case, seed: u64, prefill_width: bool) !void {
    const mark = g.mark();
    defer g.resetTo(mark);
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const rate = format.Rate.fromN(c.n).?;
    var bank = try ref.synthBank(a, rnd, c.experts, c.hidden, c.inter, rate);
    defer bank.deinit(a);
    const nslots = c.rows * c.topk;
    const x = try a.alloc(f32, c.rows * c.hidden);
    defer a.free(x);
    const bf = support.bf16Bits;
    for (x) |*v| v.* = @bitCast(@as(u32, bf(rnd.floatNorm(f32))) << 16); // bf16-exact inputs
    const slots = try a.alloc(u32, nslots);
    defer a.free(slots);
    const scores = try a.alloc(f32, nslots);
    defer a.free(scores);
    for (0..c.rows) |r| for (0..c.topk) |k| {
        slots[r * c.topk + k] = @intCast((k * 2 + r) % c.experts);
        scores[r * c.topk + k] = @bitCast(@as(u32, bf(0.05 + 0.1 * rnd.float(f32))) << 16); // bf16-exact scores
    };
    const y_ref = try a.alloc(f32, c.rows * c.hidden);
    defer a.free(y_ref);
    for (0..c.rows) |r| try ref.moeRowDec(a, &bank, c.dec, x[r * c.hidden ..][0..c.hidden], slots[r * c.topk ..][0..c.topk], scores[r * c.topk ..][0..c.topk], y_ref[r * c.hidden ..][0..c.hidden]);
    const ht = ci(c.hidden / 16);
    const it = ci(c.inter / 16);
    const bank_arrays: root.Bank = .{
        .gate = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(c.experts), ht, it, ci(c.n) }, .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ ci(c.experts), ci(c.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ ci(c.experts), ci(c.inter) }, .float16) },
        .up = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ ci(c.experts), ht, it, ci(c.n) }, .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ ci(c.experts), ci(c.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ ci(c.experts), ci(c.inter) }, .float16) },
        .down = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ ci(c.experts), it, ht, ci(c.n) }, .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ ci(c.experts), ci(c.inter) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ ci(c.experts), ci(c.hidden) }, .float16) },
    };
    const x3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(c.rows), ci(c.hidden) }, .float32), .bfloat16);
    const inds_i32 = try a.alloc(i32, nslots);
    defer a.free(inds_i32);
    for (slots, inds_i32) |sl, *d| d.* = @intCast(sl);
    const inds3 = try g.hostArray(std.mem.sliceAsBytes(inds_i32), &.{ 1, ci(c.rows), ci(c.topk) }, .int32);
    const sc3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, ci(c.rows), ci(c.topk) }, .float32), .bfloat16);
    const y = try g.adopt(try root.moe(s, x3, bank_arrays, inds3, sc3, c.dec, false));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y));
    const got = try a.alloc(f32, c.rows * c.hidden);
    defer a.free(got);
    _ = try g.hostF32(try g.astype(y, .float32), got);
    const e = ref.compare(got, y_ref);
    std.debug.print("TURBOEXL3_PARITY root.moe bf16 geo={d}x{d} rows={d} topk={d} max_abs={e:.3} rms={e:.3} ratio={e:.3} {s}\n", .{ c.hidden, c.inter, c.rows, c.topk, e.max_abs, e.rms_ref, e.max_ratio, if (e.max_ratio <= 1.5e-2) "PASS" else "FAIL" });
    try testing.expect(e.max_ratio <= 1.5e-2); // the bf16 output rounding (2^-8) on top of the chain's tolerance
    // a prefill-width call is served by the fused chain (M3): x's shape and dtype back (prefill_test.zig holds its values)
    if (!prefill_width) return;
    const xw = try g.zeros(&.{ 1, 32, ci(c.hidden) }, .bfloat16);
    const iw = try g.zeros(&.{ 1, 32, ci(c.topk) }, .int32);
    const sw = try g.zeros(&.{ 1, 32, ci(c.topk) }, .bfloat16);
    const yw = try g.adopt(try root.moe(s, xw, bank_arrays, iw, sw, c.dec, false));
    try testing.expectEqualSlices(c_int, &.{ 1, 32, ci(c.hidden) }, mlx.getShape(yw));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(yw));
}

// The dense linear (the full-EXL3 trunk's path): rows 1, 4 and 16 through the GEMV kernels, 40 through dequantize-and-matmul,
// at K7 (the attention and GDN projections' rate), K6 (lm_head, MTP fc) and K5, both directions of a 2560 x 640 shape.
test "turboexl3 gpu: dense.linear equals format.project per row at the trunk's rates, GEMV and prefill paths (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const Shape = struct { in: u32, out: u32, n: u32, dec: format.Decode };
    const shapes = [_]Shape{
        .{ .in = 2560, .out = 640, .n = 112, .dec = .mul1 },
        .{ .in = 640, .out = 2560, .n = 112, .dec = .mul1 },
        .{ .in = 2560, .out = 512, .n = 96, .dec = .mul1 },
        .{ .in = 2560, .out = 640, .n = 80, .dec = .mul1 },
        .{ .in = 512, .out = 256, .n = 48, .dec = .{ .codebook = .mcg, .window = .w12 } },
    };
    // 1..5, 8: one chunk of the multi-row GEMV at every RB it serves; 9 and 16: two chunks (a partial one at 9); 17, 31, 32:
    // the register-path GEMM (one partial or full 32-row tile); 40, 129, 200: prefill (dequantize + matmul at the default
    // small_rows; `fused_prefill`: the fused GEMM, one partial 64-row tile or several plus a partial)
    const row_counts = [_]u32{ 1, 2, 3, 4, 5, 8, 9, 16, 17, 31, 32, 40, 129, 200 };
    // the trunk's widest GEMV shapes (a GDN qkv in-projection, an out-projection: the deepest K splits) at decode and verify
    // widths only, so the host reference stays seconds
    const big = [_]Shape{ .{ .in = 2560, .out = 10240, .n = 112, .dec = .mul1 }, .{ .in = 6144, .out = 2560, .n = 112, .dec = .mul1 } };
    const big_rows = [_]u32{ 1, 3, 4, 9 };
    for (shapes ++ big, 0..) |sh, si| {
        var prng = std.Random.DefaultPrng.init(0xD3A5E ^ sh.n ^ sh.out);
        const rnd = prng.random();
        const rate = format.Rate.fromN(sh.n).?;
        // one expert of a synthetic bank: its gate projection is a dense in -> out linear
        var bank = try ref.synthBank(a, rnd, 1, sh.in, sh.out, rate);
        defer bank.deinit(a);
        const ht = ci(sh.in / 16);
        const ot = ci(sh.out / 16);
        const mark = g.mark();
        defer g.resetTo(mark);
        const trellis = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ht, ot, ci(sh.n) }, .uint16);
        const suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ci(sh.in)}, .float16);
        const svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ci(sh.out)}, .float16);
        for (if (si < shapes.len) row_counts[0..] else big_rows[0..]) |rows| {
            const x = try a.alloc(f32, rows * sh.in);
            defer a.free(x);
            const bf = support.bf16Bits;
            for (x) |*v| v.* = @bitCast(@as(u32, bf(rnd.floatNorm(f32))) << 16);
            const y_ref = try a.alloc(f32, rows * sh.out);
            defer a.free(y_ref);
            const transformed = try a.alloc(f32, sh.in);
            defer a.free(transformed);
            const inner = try a.alloc(f32, sh.out);
            defer a.free(inner);
            for (0..rows) |r| format.project(x[r * sh.in ..][0..sh.in], bank.gate_t, bank.gate_suh, bank.gate_svh, sh.in, sh.out, rate, sh.dec, transformed, inner, y_ref[r * sh.out ..][0..sh.out]);
            // f32 x: the numeric bar is the decode chain's (5e-3 of the reference rms); the output is f32 too
            const x3 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(sh.in) }, .float32);
            const y = try g.adopt(try root.linear(s, x3, trellis, suh, svh, sh.dec));
            try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(y));
            const ysh = mlx.getShape(y);
            try testing.expect(ysh.len == 3 and ysh[1] == ci(rows) and ysh[2] == ci(sh.out));
            const got = try a.alloc(f32, rows * sh.out);
            defer a.free(got);
            _ = try g.hostF32(y, got);
            const e = ref.compare(got, y_ref);
            std.debug.print("TURBOEXL3_PARITY dense in={d} out={d} n={d} cb={s} rows={d} path={s} max_abs={e:.3} rms={e:.3} ratio={e:.3} {s}\n", .{
                sh.in, sh.out, sh.n, @tagName(sh.dec.codebook), rows, if (rows <= 16) "gemv" else if (rows <= 32) "regpath" else "prefill", e.max_abs, e.rms_ref, e.max_ratio, if (e.max_ratio <= 5e-3) "PASS" else "FAIL",
            });
            try testing.expect(e.max_ratio <= 5e-3);
            // bf16 x: the served dtype in and out (one bf16 ulp of rounding on top)
            if (rows == 4 or rows == 40) {
                const xb = try g.astype(x3, .bfloat16);
                const yb = try g.adopt(try root.linear(s, xb, trellis, suh, svh, sh.dec));
                try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(yb));
                _ = try g.hostF32(try g.astype(yb, .float32), got);
                const eb = ref.compare(got, y_ref);
                try testing.expect(eb.max_ratio <= 2.5e-2);
            }
        }
    }
}

// Real tensors: TURBOEXL3_REAL_SHARD=<trunk-exl3-NN.safetensors of the full Qwen3.8 pack> runs root.linear on two of its trellis
// triples (a GDN in-projection and a shared-expert down-projection) against format.project on the same halfwords, bf16 x,
// rows 3 (GEMV path) and 40 (matmul path). Pins the pack-side interpretation (orientation, dtypes, suh/svh) on turboderp's data.
test "turboexl3 gpu: root.linear on real trunk trellises from the full pack equals format.project (TURBOEXL3_REAL_SHARD)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const shard_c = std.c.getenv("TURBOEXL3_REAL_SHARD") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const path = std.mem.span(shard_c);
    // the safetensors header
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    var hl_buf: [8]u8 = undefined;
    try testing.expectEqual(@as(isize, 8), std.c.pread(fd, &hl_buf, 8, 0));
    const hlen: usize = @intCast(std.mem.readInt(u64, &hl_buf, .little));
    const hdr = try a.alloc(u8, hlen);
    defer a.free(hdr);
    try testing.expectEqual(@as(isize, @intCast(hlen)), std.c.pread(fd, hdr.ptr, hlen, 8));
    const parsed = try std.json.parseFromSlice(std.json.Value, a, hdr, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    const data_off: usize = 8 + hlen;
    const Tensor = struct { bytes: []u8, shape: [3]c_int, ndim: usize };
    const readT = struct {
        fn run(alloc: std.mem.Allocator, o: std.json.ObjectMap, f: c_int, base: usize, name: []const u8) !Tensor {
            const t = (o.get(name) orelse return error.MissingWeight).object;
            const offs = t.get("data_offsets").?.array.items;
            const o0: usize = @intCast(offs[0].integer);
            const o1: usize = @intCast(offs[1].integer);
            const bytes = try alloc.alloc(u8, o1 - o0);
            if (std.c.pread(f, bytes.ptr, bytes.len, @intCast(base + o0)) != @as(isize, @intCast(bytes.len))) return error.ShortRead;
            const dims = t.get("shape").?.array.items;
            var sh: [3]c_int = .{ 1, 1, 1 };
            for (dims, 0..) |d, i| sh[i] = @intCast(d.integer);
            return .{ .bytes = bytes, .shape = sh, .ndim = dims.len };
        }
    }.run;
    const bases = [_][]const u8{ "language_model.model.layers.4.linear_attn.in_proj_qkv", "language_model.model.layers.4.mlp.shared_expert.down_proj" };
    for (bases) |base| {
        var nb: [200]u8 = undefined;
        const w = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb, "{s}.weight", .{base}));
        defer a.free(w.bytes);
        var nb2: [200]u8 = undefined;
        const suh = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb2, "{s}.scales", .{base}));
        defer a.free(suh.bytes);
        var nb3: [200]u8 = undefined;
        const svh = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb3, "{s}.biases", .{base}));
        defer a.free(svh.bytes);
        try testing.expectEqual(@as(usize, 3), w.ndim);
        const in_dim: u32 = @intCast(w.shape[0] * 16);
        const out_dim: u32 = @intCast(w.shape[1] * 16);
        const n: u32 = @intCast(w.shape[2]);
        const rate = format.Rate.fromN(n).?;
        try testing.expectEqual(in_dim, @as(u32, @intCast(suh.shape[0])));
        try testing.expectEqual(out_dim, @as(u32, @intCast(svh.shape[0])));
        const trellis = std.mem.bytesAsSlice(u16, w.bytes);
        const suh16 = std.mem.bytesAsSlice(u16, suh.bytes);
        const svh16 = std.mem.bytesAsSlice(u16, svh.bytes);
        // suh/svh sanity: finite, O(1) scales
        var smin: f32 = 1e9;
        var smax: f32 = -1e9;
        for (suh16) |v| {
            const f: f32 = @floatCast(@as(f16, @bitCast(v)));
            smin = @min(smin, f);
            smax = @max(smax, f);
        }
        std.debug.print("TURBOEXL3_REAL {s}: in={d} out={d} n={d} suh[min,max]=[{d:.4}, {d:.4}]\n", .{ base, in_dim, out_dim, n, smin, smax });
        const mark = g.mark();
        defer g.resetTo(mark);
        const ta = try g.hostArray(w.bytes, &.{ w.shape[0], w.shape[1], w.shape[2] }, .uint16);
        const sa = try g.hostArray(suh.bytes, &.{ci(in_dim)}, .float16);
        const va = try g.hostArray(svh.bytes, &.{ci(out_dim)}, .float16);
        var prng = std.Random.DefaultPrng.init(0xBEEF ^ n);
        const rnd = prng.random();
        for ([_]u32{ 3, 40 }) |rows| {
            const x = try a.alloc(f32, rows * in_dim);
            defer a.free(x);
            const bf = support.bf16Bits;
            for (x) |*v| v.* = @bitCast(@as(u32, bf(rnd.floatNorm(f32))) << 16);
            const y_ref = try a.alloc(f32, rows * out_dim);
            defer a.free(y_ref);
            const transformed = try a.alloc(f32, in_dim);
            defer a.free(transformed);
            const inner = try a.alloc(f32, out_dim);
            defer a.free(inner);
            for (0..rows) |r| format.project(x[r * in_dim ..][0..in_dim], @alignCast(trellis), @alignCast(suh16), @alignCast(svh16), in_dim, out_dim, rate, .mul1, transformed, inner, y_ref[r * out_dim ..][0..out_dim]);
            const x3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(in_dim) }, .float32), .bfloat16);
            const y = try g.adopt(try root.linear(s, x3, ta, sa, va, .mul1));
            const got = try a.alloc(f32, rows * out_dim);
            defer a.free(got);
            _ = try g.hostF32(try g.astype(y, .float32), got);
            const e = ref.compare(got, y_ref);
            std.debug.print("TURBOEXL3_REAL_PARITY {s} rows={d} path={s} max_abs={e:.3} rms={e:.3} ratio={e:.3} {s}\n", .{ base, rows, if (rows <= 16) "gemv" else "matmul", e.max_abs, e.rms_ref, e.max_ratio, if (e.max_ratio <= 2.5e-2) "PASS" else "FAIL" });
            try testing.expect(e.max_ratio <= 2.5e-2);
        }
    }
}

// CPU-only tool. TURBOEXL3_PERM_FILE names a small safetensors file holding, per trunk module <base>, the EXL3 triple as
// "exl3:<base>.{weight,scales,biases}" and the mixed pack's affine 8-bit triple as "aff:<base>.{weight,scales,biases}"
// (weight U32 [out, in/4], scales/biases BF16 [out, in/group]). It reconstructs the EXL3 public weight [in, out]
// (format.reconstructPublic) and the affine weight [out, in], then reports how the EXL3 output columns map onto the
// affine output rows: per 128-block identity agreement and, for a sample of columns, the best-matching affine row.
// A permutation here means turboderp's export orders a projection's outputs differently from mlx-serve's converter.
test "turboexl3 tool: trunk column order, EXL3 vs the affine pack (TURBOEXL3_PERM_FILE)" {
    const path_c = std.c.getenv("TURBOEXL3_PERM_FILE") orelse return error.SkipZigTest;
    const a = std.heap.page_allocator;
    const path = std.mem.span(path_c);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    var hl_buf: [8]u8 = undefined;
    try testing.expectEqual(@as(isize, 8), std.c.pread(fd, &hl_buf, 8, 0));
    const hlen: usize = @intCast(std.mem.readInt(u64, &hl_buf, .little));
    const hdr = try a.alloc(u8, hlen);
    defer a.free(hdr);
    try testing.expectEqual(@as(isize, @intCast(hlen)), std.c.pread(fd, hdr.ptr, hlen, 8));
    const parsed = try std.json.parseFromSlice(std.json.Value, a, hdr, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    const data_off: usize = 8 + hlen;
    const Tensor = struct { bytes: []u8, shape: [3]usize, ndim: usize };
    const readT = struct {
        fn run(alloc: std.mem.Allocator, o: std.json.ObjectMap, f: c_int, base: usize, name: []const u8) !Tensor {
            const t = (o.get(name) orelse return error.MissingWeight).object;
            const offs = t.get("data_offsets").?.array.items;
            const o0: usize = @intCast(offs[0].integer);
            const o1: usize = @intCast(offs[1].integer);
            const bytes = try alloc.alloc(u8, o1 - o0);
            if (std.c.pread(f, bytes.ptr, bytes.len, @intCast(base + o0)) != @as(isize, @intCast(bytes.len))) return error.ShortRead;
            const dims = t.get("shape").?.array.items;
            var sh: [3]usize = .{ 1, 1, 1 };
            for (dims, 0..) |d, i| sh[i] = @intCast(d.integer);
            return .{ .bytes = bytes, .shape = sh, .ndim = dims.len };
        }
    }.run;
    var it = obj.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.startsWith(u8, k, "exl3:") or !std.mem.endsWith(u8, k, ".weight")) continue;
        const base = k[5 .. k.len - 7];
        var nb: [6][256]u8 = undefined;
        const w = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb[0], "exl3:{s}.weight", .{base}));
        defer a.free(w.bytes);
        const suh = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb[1], "exl3:{s}.scales", .{base}));
        defer a.free(suh.bytes);
        const svh = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb[2], "exl3:{s}.biases", .{base}));
        defer a.free(svh.bytes);
        const aw = readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb[3], "aff:{s}.weight", .{base})) catch {
            std.debug.print("TURBOEXL3_PERM {s}: no affine counterpart\n", .{base});
            continue;
        };
        defer a.free(aw.bytes);
        const asc = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb[4], "aff:{s}.scales", .{base}));
        defer a.free(asc.bytes);
        const abi = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb[5], "aff:{s}.biases", .{base}));
        defer a.free(abi.bytes);
        const in_dim = w.shape[0] * 16;
        const out_dim = w.shape[1] * 16;
        const n: u32 = @intCast(w.shape[2]);
        const rate = format.Rate.fromN(n) orelse return error.BadExl3Shape;
        try testing.expectEqual(out_dim, aw.shape[0]);
        try testing.expectEqual(in_dim / 4, aw.shape[1]);
        const gs = in_dim / asc.shape[1];
        // EXL3 public weight [in, out]
        const we16 = try a.alloc(u16, in_dim * out_dim);
        defer a.free(we16);
        const trellis: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, w.bytes));
        const suh16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, suh.bytes));
        const svh16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, svh.bytes));
        try format.reconstructPublic(a, trellis, suh16, svh16, in_dim, out_dim, rate, .mul1, we16);
        // normalized EXL3 columns ce[c][i], with their norms
        const ce = try a.alloc(f32, in_dim * out_dim);
        defer a.free(ce);
        const ne = try a.alloc(f32, out_dim);
        defer a.free(ne);
        for (0..out_dim) |c| {
            var ss: f64 = 0;
            for (0..in_dim) |i| {
                const v = format.f16BitsToF32(we16[i * out_dim + c]);
                ce[c * in_dim + i] = v;
                ss += @as(f64, v) * v;
            }
            const nrm: f32 = @floatCast(@sqrt(ss) + 1e-20);
            ne[c] = nrm;
            for (0..in_dim) |i| ce[c * in_dim + i] /= nrm;
        }
        // normalized affine rows ra[o][i] = (scale * q + bias) / norm
        const ra = try a.alloc(f32, in_dim * out_dim);
        defer a.free(ra);
        const na = try a.alloc(f32, out_dim);
        defer a.free(na);
        const aw32: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, aw.bytes));
        const asc16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, asc.bytes));
        const abi16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, abi.bytes));
        const groups = in_dim / gs;
        for (0..out_dim) |o| {
            var ss: f64 = 0;
            for (0..in_dim) |i| {
                const q: f32 = @floatFromInt((aw32[o * (in_dim / 4) + i / 4] >> @intCast(8 * (i % 4))) & 0xff);
                const s: f32 = @bitCast(@as(u32, asc16[o * groups + i / gs]) << 16);
                const b: f32 = @bitCast(@as(u32, abi16[o * groups + i / gs]) << 16);
                const v = s * q + b;
                ra[o * in_dim + i] = v;
                ss += @as(f64, v) * v;
            }
            const nrm: f32 = @floatCast(@sqrt(ss) + 1e-20);
            na[o] = nrm;
            for (0..in_dim) |i| ra[o * in_dim + i] /= nrm;
        }
        const dot = struct {
            fn run(x: []const f32, y: []const f32) f32 {
                var s: f32 = 0;
                for (x, y) |p, q| s += p * q;
                return s;
            }
        }.run;
        // identity agreement per 128-block
        const nblk = out_dim / 128;
        var blocks: [512]u8 = undefined;
        var ident: usize = 0;
        for (0..@min(nblk, blocks.len)) |bk| {
            var m: usize = 0;
            for (0..128) |j| {
                const c = bk * 128 + j;
                if (dot(ce[c * in_dim ..][0..in_dim], ra[c * in_dim ..][0..in_dim]) > 0.9) m += 1;
            }
            ident += m;
            blocks[bk] = if (m == 128) 'X' else if (m > 0) 'x' else '.';
        }
        std.debug.print("TURBOEXL3_PERM {s} in={d} out={d} n={d} gs={d} identity_cols={d}/{d} blocks(128)={s}\n", .{ base, in_dim, out_dim, n, gs, ident, out_dim, blocks[0..@min(nblk, blocks.len)] });
        // a sample of columns: where does each one's best match sit?
        const samples: usize = 24;
        for (0..samples) |j| {
            const c = (j * out_dim) / samples + 5;
            var best: usize = 0;
            var best_v: f32 = -2;
            for (0..out_dim) |o| {
                const v = dot(ce[c * in_dim ..][0..in_dim], ra[o * in_dim ..][0..in_dim]);
                if (v > best_v) {
                    best_v = v;
                    best = o;
                }
            }
            const id_v = dot(ce[c * in_dim ..][0..in_dim], ra[c * in_dim ..][0..in_dim]);
            std.debug.print("  col {d:>6} -> row {d:>6} corr {d:.3} (identity corr {d:.3}) norm_ratio {d:.3}\n", .{ c, best, best_v, id_v, ne[c] / na[best] });
        }
    }
}

// The multi-row GEMV's promise: a row's output bytes do not depend on how many rows share the launch (the summation order is
// fixed by the shape's K split), so a verify row equals the same row decoded alone.
test "turboexl3 gpu: dense multi-row GEMV, a row's bytes are the same alone and among 2..16 rows (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const dense = @import("dense.zig");
    const Shape = struct { in: u32, out: u32, n: u32 };
    var checks: usize = 0;
    for ([_]Shape{ .{ .in = 2560, .out = 640, .n = 112 }, .{ .in = 640, .out = 2560, .n = 96 }, .{ .in = 2560, .out = 10240, .n = 112 }, .{ .in = 6144, .out = 2560, .n = 112 } }) |sh| {
        var prng = std.Random.DefaultPrng.init(0xB17E ^ sh.out);
        const rnd = prng.random();
        const rate = format.Rate.fromN(sh.n).?;
        var bank = try ref.synthBank(a, rnd, 1, sh.in, sh.out, rate);
        defer bank.deinit(a);
        const mark = g.mark();
        defer g.resetTo(mark);
        const trellis = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(sh.in / 16), ci(sh.out / 16), ci(sh.n) }, .uint16);
        const suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ci(sh.in)}, .float16);
        const svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ci(sh.out)}, .float16);
        var d = try dense.Dense.init(.{ .in = sh.in, .out = sh.out, .rate = rate, .dec = .mul1, .out_dtype = .float32 });
        defer d.deinit();
        d.rowslot = false;
        d.dec_on = false; // the multi-row GEMV (the one-launch decode linear's rows have their own test)
        const max_rows = 16;
        const x = try a.alloc(f32, max_rows * sh.in);
        defer a.free(x);
        for (x) |*v| v.* = rnd.floatNorm(f32);
        const alone = try a.alloc(f32, max_rows * sh.out);
        defer a.free(alone);
        // each row alone
        for (0..max_rows) |r| {
            const xr = try g.hostArray(std.mem.sliceAsBytes(x[r * sh.in ..][0..sh.in]), &.{ 1, ci(sh.in) }, .float32);
            const y = try g.adopt(try d.linear(s, xr, trellis, suh, svh, 1));
            _ = try g.hostF32(y, alone[r * sh.out ..][0..sh.out]);
        }
        // the chain on text_dgemv (cap 5) and on the lean text past 5 rows (cap 8), a Dense each (the configs cache per Dense)
        var dl = try dense.Dense.init(.{ .in = sh.in, .out = sh.out, .rate = rate, .dec = .mul1, .out_dtype = .float32 });
        defer dl.deinit();
        dl.rowslot = false;
        dl.dec_on = false;
        dl.lean = true;
        d.lean = false;
        for ([_]*dense.Dense{ &d, &dl }) |dd| for ([_]u32{ 2, 3, 4, 5, 6, 7, 8, 9, 11, 16 }) |rows| {
            const xr = try g.hostArray(std.mem.sliceAsBytes(x[0 .. rows * sh.in]), &.{ ci(rows), ci(sh.in) }, .float32);
            const y = try g.adopt(try dd.linear(s, xr, trellis, suh, svh, rows));
            const got = try a.alloc(f32, rows * sh.out);
            defer a.free(got);
            _ = try g.hostF32(y, got);
            var diff: usize = 0;
            for (got, alone[0 .. rows * sh.out]) |p, q| {
                if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) diff += 1;
            }
            const rb = dd.chainRb(rows);
            std.debug.print("TURBOEXL3_ROWBYTES dense in={d} out={d} n={d} ks={d} rows={d} rb={d} lean={} differing={d}/{d} {s}\n", .{ sh.in, sh.out, sh.n, d.ks_split, rows, rb, dd.leanFor(rb), diff, got.len, if (diff == 0) "PASS" else "FAIL" });
            try testing.expectEqual(@as(usize, 0), diff);
            checks += 1;
        };
    }
    // reached only when every check passed (each failure returns above)
    std.debug.print("TURBOEXL3_ROWBYTES_DONE checks={d} PASS\n", .{checks});
}

// Two streams (a host's side stream): the dense linear split across two MLX streams, the prepare on the first
// and the prefill rows' inner product and finish on the second, records the same launches on the same operands, so its bytes
// are the one-stream bytes. Both results are built before one eval, so the second queue runs beside the first. Prefill rows
// (8192, 1328, 40) cross the streams on both inner products (dequantize + matmul, the default; the fused GEMM); register-path
// (31) and GEMV (4) rows stay on the first stream. Then the root entry (x and y reshaped on the first stream).
test "turboexl3 gpu: dense.linearOn across two streams gives linear's bytes on both prefill paths (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var dev = mlx.mlx_device{ .ctx = null };
    try testing.expectEqual(@as(c_int, 0), mlx.mlx_stream_get_device(&dev, s));
    defer _ = mlx.mlx_device_free(dev);
    const side = mlx.mlx_stream_new_device(dev);
    try testing.expect(side.ctx != null);
    defer _ = mlx.mlx_stream_free(side);
    try testing.expect(!mlx.mlx_stream_equal(side, s));
    const dense = @import("dense.zig");
    const Shape = struct { in: u32, out: u32, n: u32 };
    // the GDN z projection at K7 (the trunk's attention / GDN rate), the shared expert's gate (and up) and down at K5
    for ([_]Shape{ .{ .in = 2560, .out = 6144, .n = 112 }, .{ .in = 2560, .out = 640, .n = 80 }, .{ .in = 640, .out = 2560, .n = 80 } }) |sh| {
        var prng = std.Random.DefaultPrng.init(0x51DE ^ sh.out ^ (sh.in << 16));
        const rnd = prng.random();
        const rate = format.Rate.fromN(sh.n).?;
        var bank = try ref.synthBank(a, rnd, 1, sh.in, sh.out, rate);
        defer bank.deinit(a);
        const mark = g.mark();
        defer g.resetTo(mark);
        const trellis = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(sh.in / 16), ci(sh.out / 16), ci(sh.n) }, .uint16);
        const suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ci(sh.in)}, .float16);
        const svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ci(sh.out)}, .float16);
        var d = try dense.Dense.init(.{ .in = sh.in, .out = sh.out, .rate = rate, .dec = .mul1, .out_dtype = .bfloat16 });
        defer d.deinit();
        for ([_]bool{ false, true }) |fused| {
            d.fused_prefill = fused;
            for ([_]u32{ 8192, 1328, 40, 31, 4 }) |rows| {
                const m2 = g.mark();
                defer g.resetTo(m2);
                const x = try a.alloc(f32, rows * sh.in);
                defer a.free(x);
                for (x) |*v| v.* = rnd.floatNorm(f32);
                const xf = try g.hostArray(std.mem.sliceAsBytes(x), &.{ ci(rows), ci(sh.in) }, .float32);
                const xb = try g.astype(xf, .bfloat16);
                const y1 = try g.adopt(try d.linear(s, xb, trellis, suh, svh, rows));
                const y2 = try g.adopt(try d.linearOn(s, side, xb, trellis, suh, svh, rows));
                try g.evalAll(&.{ y1, y2 });
                const got1 = try a.alloc(f32, rows * sh.out);
                defer a.free(got1);
                const got2 = try a.alloc(f32, rows * sh.out);
                defer a.free(got2);
                _ = try g.hostF32(try g.astype(y1, .float32), got1);
                _ = try g.hostF32(try g.astype(y2, .float32), got2);
                var diff: usize = 0;
                var bad: usize = 0;
                for (got1, got2) |p, q| {
                    if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) diff += 1;
                    if (!std.math.isFinite(p)) bad += 1;
                }
                std.debug.print("TURBOEXL3_SIDEBYTES dense in={d} out={d} rows={d} prefill={s} differing={d}/{d} nonfinite={d} {s}\n", .{ sh.in, sh.out, rows, if (fused) "fused" else "dequant+matmul", diff, got1.len, bad, if (diff == 0 and bad == 0) "PASS" else "FAIL" });
                try testing.expectEqual(@as(usize, 0), diff);
                try testing.expectEqual(@as(usize, 0), bad);
            }
        }
        const rows: u32 = 1328;
        const x = try a.alloc(f32, rows * sh.in);
        defer a.free(x);
        for (x) |*v| v.* = rnd.floatNorm(f32);
        const x3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(sh.in) }, .float32), .bfloat16);
        const r1 = try g.adopt(try root.linear(s, x3, trellis, suh, svh, .mul1));
        const r2 = try g.adopt(try root.linearSplit(s, side, x3, trellis, suh, svh, .mul1));
        try testing.expectEqualSlices(c_int, mlx.getShape(r1), mlx.getShape(r2));
        try g.evalAll(&.{ r1, r2 });
        const got1 = try a.alloc(f32, rows * sh.out);
        defer a.free(got1);
        const got2 = try a.alloc(f32, rows * sh.out);
        defer a.free(got2);
        _ = try g.hostF32(try g.astype(r1, .float32), got1);
        _ = try g.hostF32(try g.astype(r2, .float32), got2);
        try testing.expect(std.mem.eql(u8, std.mem.sliceAsBytes(got1), std.mem.sliceAsBytes(got2)));
    }
}

// Timing of the dense linear on the trunk's shapes (TURBOEXL3_GPU=1 TURBOEXL3_DENSE_TIME=1): the multi-row GEMV, the per-row-slot
// GEMV it replaces, and MLX's 8-bit affine quantized_matmul at the same shape (what the mixed pack's trunk runs). Median us
// per call (the minimum over 7 timed rounds of 16 calls evaluated together, after 2 warm rounds).
test "turboexl3 gpu: dense timing, multi-row vs row-slot vs affine q8 (TURBOEXL3_DENSE_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_DENSE_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const dense = @import("dense.zig");
    const Shape = struct { name: []const u8, in: u32, out: u32, n: u32 };
    const shapes = [_]Shape{
        .{ .name = "gdn.in_proj_qkv", .in = 2560, .out = 10240, .n = 112 },
        .{ .name = "gdn.in_proj_z", .in = 2560, .out = 6144, .n = 112 },
        .{ .name = "attn.q_proj", .in = 2560, .out = 12288, .n = 112 },
        .{ .name = "gdn.out_proj", .in = 6144, .out = 2560, .n = 112 },
        .{ .name = "attn.k_proj", .in = 2560, .out = 512, .n = 112 },
        .{ .name = "shared.gate", .in = 2560, .out = 640, .n = 112 },
        .{ .name = "shared.down", .in = 640, .out = 2560, .n = 112 },
        .{ .name = "lm_head/10", .in = 2560, .out = 24832, .n = 96 },
    };
    // TURBOEXL3_DENSE_TIME_PREFILL=1: prompt-chunk row counts (the dequantize-and-matmul path) instead of decode rows;
    // TURBOEXL3_DENSE_TIME_TAIL=1: the snapshot tail's row counts, arm 0 = the register-path GEMM, arm 1 = dequantize + matmul
    const tail = std.c.getenv("TURBOEXL3_DENSE_TIME_TAIL") != null;
    const prefill = tail or std.c.getenv("TURBOEXL3_DENSE_TIME_PREFILL") != null;
    const row_counts: []const u32 = if (tail) &.{ 17, 24, 31, 32, 48, 64, 96, 128 } else if (prefill) &.{ 1359, 4096, 8192 } else &.{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const calls = 16;
    for (shapes) |sh| {
        var prng = std.Random.DefaultPrng.init(0x7173 ^ sh.out);
        const rnd = prng.random();
        const rate = format.Rate.fromN(sh.n).?;
        const mark = g.mark();
        defer g.resetTo(mark);
        const tw = try a.alloc(u16, (sh.in / 16) * (sh.out / 16) * sh.n);
        defer a.free(tw);
        for (tw) |*v| v.* = rnd.int(u16);
        const su = try a.alloc(u16, sh.in);
        defer a.free(su);
        for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
        const sv = try a.alloc(u16, sh.out);
        defer a.free(sv);
        for (sv) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
        const trellis = try g.hostArray(std.mem.sliceAsBytes(tw), &.{ ci(sh.in / 16), ci(sh.out / 16), ci(sh.n) }, .uint16);
        const suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(sh.in)}, .float16);
        const svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(sh.out)}, .float16);
        // the affine 8-bit operand at the same shape: a random bf16 [out, in] quantized by MLX (group 64)
        const wf = try a.alloc(f32, sh.out * sh.in);
        defer a.free(wf);
        for (wf) |*v| v.* = 0.02 * rnd.floatNorm(f32);
        const wb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(wf), &.{ ci(sh.out), ci(sh.in) }, .float32), .bfloat16);
        var qv = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(qv);
        try mlx.check(mlx.mlx_quantize(&qv, wb, .some(64), .some(8), "affine", .{}, s));
        var wq = mlx.mlx_array_new();
        var sc = mlx.mlx_array_new();
        var bi = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&wq, qv, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, qv, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, qv, 2));
        wq = try g.adopt(wq);
        sc = try g.adopt(sc);
        bi = try g.adopt(bi);
        try g.evalAll(&.{ trellis, suh, svh, wq, sc, bi });
        var multi = try dense.Dense.init(.{ .in = sh.in, .out = sh.out, .rate = rate, .dec = .mul1, .out_dtype = .bfloat16 });
        defer multi.deinit();
        multi.rowslot = false;
        multi.dec_on = false; // the multi-row GEMV, not the one-launch decode linear
        var slot = try dense.Dense.init(.{ .in = sh.in, .out = sh.out, .rate = rate, .dec = .mul1, .out_dtype = .bfloat16 });
        defer slot.deinit();
        slot.rowslot = true;
        // prefill rows: arm 0 = the fused GEMM, arm 1 = dequantize + MLX matmul; tail rows: arm 0 = the register path
        if (prefill) {
            multi.fused_prefill = !tail;
            slot.fused_prefill = false;
            multi.small_rows = if (tail) dense.Dense.small_rows_cap else 0;
            slot.small_rows = 0;
        }
        for (row_counts) |rows| {
            const xf = try a.alloc(f32, rows * sh.in);
            defer a.free(xf);
            for (xf) |*v| v.* = rnd.floatNorm(f32);
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xf), &.{ ci(rows), ci(sh.in) }, .float32), .bfloat16);
            try g.evalAll(&.{x});
            if (prefill and !tail) {
                // the fused GEMM against dequantize + matmul on the timed inputs: same weights, different summation order
                const yf = try g.adopt(try multi.linear(s, x, trellis, suh, svh, rows));
                const yd = try g.adopt(try slot.linear(s, x, trellis, suh, svh, rows));
                const nf = rows * sh.out;
                const gf = try a.alloc(f32, nf);
                defer a.free(gf);
                const gd = try a.alloc(f32, nf);
                defer a.free(gd);
                _ = try g.hostF32(try g.astype(yf, .float32), gf);
                _ = try g.hostF32(try g.astype(yd, .float32), gd);
                // both outputs are bf16, and the deferred MUL1 map rounds differently from per-weight f16 values, so the bar is
                // rounding-sized: the rms difference within a bf16 ulp's share (2^-8) of the rms, the worst within 2 ulps of a
                // ~10-sigma value (the reference test, f32 outputs against format.project, is the exact one)
                const e = ref.compare(gf, gd);
                var dsq: f64 = 0;
                for (gf, gd) |u, v| dsq += @as(f64, u - v) * @as(f64, u - v);
                const rms_diff = @sqrt(dsq / @as(f64, @floatFromInt(nf))) / e.rms_ref;
                const ok = rms_diff <= 3.9e-3 and e.max_ratio <= 8e-2;
                std.debug.print("TURBOEXL3_DENSE_PF_PARITY {s} rows={d} bm={d} dbuf={} unroll={} fragb={} intr={} defer={} max_abs={e:.3} rms={e:.3} ratio={e:.3} rms_diff={e:.3} {s}\n", .{
                    sh.name, rows, multi.pf.bm, multi.pf.dbuf, multi.pf.unroll, multi.pf.fragb, multi.pf.intr, multi.pf.defer_affine, e.max_abs, e.rms_ref, e.max_ratio, rms_diff, if (ok) "PASS" else "FAIL",
                });
                try testing.expect(ok);
            }
            var med: [3]f64 = .{ 0, 0, 0 };
            const arms: usize = if (std.c.getenv("TURBOEXL3_DENSE_TIME_MULTI_ONLY") != null) 1 else 3;
            for (0..arms) |arm| {
                var samples: [7]f64 = undefined;
                for (0..9) |round| {
                    var ys: [calls]mlx.mlx_array = undefined;
                    for (&ys) |*y| {
                        y.* = switch (arm) {
                            0 => try multi.linear(s, x, trellis, suh, svh, rows),
                            1 => try slot.linear(s, x, trellis, suh, svh, rows),
                            else => blk: {
                                var o = mlx.mlx_array_new();
                                try mlx.check(mlx.mlx_quantized_matmul(&o, x, wq, sc, bi, true, .some(64), .some(8), "affine", s));
                                break :blk o;
                            },
                        };
                    }
                    const t0 = nowNs();
                    try g.evalAll(&ys);
                    const dt = nowNs() - t0;
                    for (ys) |y| _ = mlx.mlx_array_free(y);
                    if (round >= 2) samples[round - 2] = @as(f64, @floatFromInt(dt)) / 1000.0 / calls;
                }
                // the minimum over rounds: the kernel's time with the least interference from whatever else shares the GPU
                std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
                med[arm] = samples[0];
            }
            if (tail) {
                std.debug.print("TURBOEXL3_DENSE_TAIL_TIME {s} in={d} out={d} n={d} rows={d} regpath_us={d:.1} dequant_matmul_us={d:.1} q8_us={d:.1} regpath_vs_dequant={d:.2}x regpath_vs_q8={d:.2}x\n", .{
                    sh.name, sh.in, sh.out, sh.n, rows, med[0], med[1], med[2], med[1] / med[0], med[2] / med[0],
                });
            } else if (prefill) {
                const tflops = 2.0 * @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(sh.in)) * @as(f64, @floatFromInt(sh.out)) / 1e6;
                std.debug.print("TURBOEXL3_DENSE_PF_TIME {s} in={d} out={d} n={d} rows={d} bm={d} defer={} fused_us={d:.1} dequant_matmul_us={d:.1} q8_us={d:.1} fused_vs_dequant={d:.2}x fused_vs_q8={d:.2}x fused_tflops={d:.1} dequant_tflops={d:.1}\n", .{
                    sh.name, sh.in, sh.out, sh.n, rows, multi.pf.bm, multi.pf.defer_affine, med[0], med[1], med[2], med[1] / med[0], med[2] / med[0], tflops / med[0], tflops / med[1],
                });
            } else std.debug.print("TURBOEXL3_DENSE_TIME {s} in={d} out={d} n={d} ks={d} rows={d} multi_us={d:.1} rowslot_us={d:.1} q8_us={d:.1} multi_vs_rowslot={d:.2}x multi_vs_q8={d:.2}x rb={d} lean={}\n", .{
                sh.name, sh.in, sh.out, sh.n, multi.ks_split, rows, med[0], med[1], med[2], med[1] / med[0], med[2] / med[0], multi.chainRb(rows), multi.leanFor(multi.chainRb(rows)),
            });
        }
    }
}

// DEC-ONE (`Config.dec_one`, on by default): the dense linear at decode rows in one GEMV launch (the finish in each output
// block's last-arriving threadgroup, the prepare inside the launch or in its own) must give the three-launch chain's bytes.
// The dense timing list's trunk shapes at their rates, rows 1-9, 11 and 16, both prepare placements, the deferred MUL1 map
// on and off, bf16 operands (served) and f32, and one MCG case; every check against the chain (DEC-ONE off) on the same
// inputs: the chain on the lean text past 5 rows (cap 8: 6-8 rows one chunk, 9 = 5 + 4, 11 = 6 + 5, 16 = 8 + 8) and on
// text_dgemv (cap 5: 6 = 3 + 3, 7 = 4 + 3, 11 = 4 + 4 + 3, 16 = 4 x 4), DEC-ONE at text_dgemv's cap. And every row of every
// call against the same row alone (a 1-row DEC-ONE call on it, once per variant): a row's bytes depend neither on the rows
// beside it, nor on how the call's rows are cut into chunks, nor on which GEMV text ran them.
test "turboexl3 gpu: the one-launch decode linear (DEC-ONE) equals the three-launch chain byte for byte (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const dense = @import("dense.zig");
    const Shape = struct { name: []const u8, in: u32, out: u32, n: u32, dec: format.Decode = .mul1 };
    const shapes = [_]Shape{
        .{ .name = "gdn.in_proj_qkv", .in = 2560, .out = 10240, .n = 112 },
        .{ .name = "gdn.in_proj_z", .in = 2560, .out = 6144, .n = 112 },
        .{ .name = "attn.q_proj", .in = 2560, .out = 12288, .n = 112 },
        .{ .name = "gdn.out_proj", .in = 6144, .out = 2560, .n = 112 },
        .{ .name = "attn.k_proj", .in = 2560, .out = 512, .n = 112 },
        .{ .name = "shared.gate", .in = 2560, .out = 640, .n = 112 },
        .{ .name = "shared.down", .in = 640, .out = 2560, .n = 112 },
        .{ .name = "lm_head/10", .in = 2560, .out = 24832, .n = 96 },
        .{ .name = "attn.idx_qk", .in = 2560, .out = 640, .n = 80 },
        .{ .name = "mtp.fc", .in = 2560, .out = 2560, .n = 96 },
        .{ .name = "lm_head", .in = 2560, .out = 248320, .n = 96 },
        .{ .name = "mcg.k_proj", .in = 2560, .out = 512, .n = 80, .dec = .mcg },
    };
    const Variant = struct { defer_affine: bool, bf16: bool };
    const variants = [_]Variant{ .{ .defer_affine = true, .bf16 = true }, .{ .defer_affine = true, .bf16 = false }, .{ .defer_affine = false, .bf16 = true } };
    const row_set = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16 };
    var failed = false;
    var checks: usize = 0;
    for (shapes) |sh| {
        var prng = std.Random.DefaultPrng.init(0xDEC1 ^ sh.out ^ (sh.in << 16));
        const rnd = prng.random();
        const rate = format.Rate.fromN(sh.n).?;
        const mark = g.mark();
        defer g.resetTo(mark);
        const tw = try a.alloc(u16, (sh.in / 16) * (sh.out / 16) * sh.n);
        defer a.free(tw);
        for (tw) |*v| v.* = rnd.int(u16);
        const su = try a.alloc(u16, sh.in);
        defer a.free(su);
        for (su) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * (if (rnd.boolean()) @as(f32, 1) else -1) / 16.0);
        const sv = try a.alloc(u16, sh.out);
        defer a.free(sv);
        for (sv) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * (if (rnd.boolean()) @as(f32, 1) else -1));
        const trellis = try g.hostArray(std.mem.sliceAsBytes(tw), &.{ ci(sh.in / 16), ci(sh.out / 16), ci(sh.n) }, .uint16);
        const suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(sh.in)}, .float16);
        const svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(sh.out)}, .float16);
        const max_rows: u32 = 16;
        const xf = try a.alloc(f32, max_rows * sh.in);
        defer a.free(xf);
        for (xf) |*v| v.* = rnd.floatNorm(f32);
        for (variants) |v| {
            if (sh.dec.codebook != .mul1 and v.defer_affine) continue; // MCG decodes per weight only
            const dt: mlx.mlx_dtype = if (v.bf16) .bfloat16 else .float32;
            const geo: dense.Shape = .{ .in = sh.in, .out = sh.out, .rate = rate, .dec = sh.dec, .out_dtype = dt };
            var dref = try dense.Dense.init(geo);
            defer dref.deinit();
            var dplain = try dense.Dense.init(geo);
            defer dplain.deinit();
            var din = try dense.Dense.init(geo);
            defer din.deinit();
            var dout = try dense.Dense.init(geo);
            defer dout.deinit();
            for ([_]*dense.Dense{ &dref, &dplain, &din, &dout }) |d| {
                d.rowslot = false;
                d.defer_affine = v.defer_affine;
                d.dec_rows = @intCast(decode.Kernels.max_rows);
            }
            dref.dec_on = false;
            dref.lean = true;
            dplain.dec_on = false;
            dplain.lean = false;
            din.dec_on = true;
            din.dec_prep_rows = @intCast(decode.Kernels.max_rows);
            dout.dec_on = true;
            dout.dec_prep_rows = 0;
            dout.dec_prep_out = 0;
            // each row alone: a 1-row DEC-ONE call on x's row r (one chunk of one row, whatever the row cap)
            const alone = try a.alloc(f32, max_rows * sh.out);
            defer a.free(alone);
            for (0..max_rows) |r| {
                const xr = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xf[r * sh.in ..][0..sh.in]), &.{ 1, ci(sh.in) }, .float32), dt);
                const y = try g.adopt(try din.linear(s, xr, trellis, suh, svh, 1));
                _ = try g.hostF32(try g.astype(y, .float32), alone[r * sh.out ..][0..sh.out]);
            }
            for (row_set) |rows| {
                const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xf[0 .. rows * sh.in]), &.{ ci(rows), ci(sh.in) }, .float32), dt);
                const n_out = rows * sh.out;
                const got = try a.alloc(f32, 4 * n_out);
                defer a.free(got);
                for ([_]*dense.Dense{ &dref, &din, &dout, &dplain }, 0..) |d, k| {
                    const y = try g.adopt(try d.linear(s, x, trellis, suh, svh, rows));
                    _ = try g.hostF32(try g.astype(y, .float32), got[k * n_out ..][0..n_out]);
                }
                const prep_in = try din.decPrepInside(rows);
                // each arm against the chain (on the chain's own line, 0 by construction) and every row against that row alone
                for ([_][]const u8{ "chain", "inside", "own_launch", "chain_plain" }, 0..) |arm, k| {
                    var diff: usize = 0;
                    for (got[k * n_out ..][0..n_out], got[0..n_out]) |p, q| {
                        if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) diff += 1;
                    }
                    var diff_alone: usize = 0;
                    for (got[k * n_out ..][0..n_out], alone[0..n_out]) |p, q| {
                        if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) diff_alone += 1;
                    }
                    const ok = diff == 0 and diff_alone == 0;
                    checks += 1;
                    const crb = if (k == 0) dref.chainRb(rows) else if (k == 3) dplain.chainRb(rows) else dense.Dense.rbFor(rows);
                    std.debug.print("TURBOEXL3_DEC_BYTES {s} in={d} out={d} n={d} cb={s} ks={d} rows={d} rb={d} lean={} defer={} bf16={} prepare={s}{s} differing={d}/{d} rows_vs_alone={d}/{d} {s}\n", .{
                        sh.name, sh.in, sh.out, sh.n, @tagName(sh.dec.codebook), dref.ks_split, rows, crb, k == 0 and dref.leanFor(crb), v.defer_affine, v.bf16, arm, if (k == 1 and !prep_in) "(did_not_fit:own_launch)" else "", diff, n_out, diff_alone, n_out, if (ok) "PASS" else "FAIL",
                    });
                    if (!ok) failed = true;
                }
            }
        }
    }
    std.debug.print("TURBOEXL3_DEC_BYTES_DONE checks={d} {s}\n", .{ checks, if (failed) "FAIL" else "PASS" });
    try testing.expect(!failed);
}

// Form 3, whose bytes the served form 9 keeps: an MTP verify round's width follows the draft schedule, so a greedy text is stable
// only if every row's output bytes are the same whatever rows share its call. Each row alone, then rows 0..R-1 together
// for R = 2..16; 4 of a row's 10 slots come from a hot pool of 6 experts, so slots of different rows choose the same expert
// and the expert-grouped (GROUP 2) member path runs. Both MUL1 decodes: the deferred affine map and the per-weight f16.
test "turboexl3 gpu: decode chain (form 3), a row's bytes are the same alone and among 2..16 rows (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const hidden: u32 = 2560;
    const inter: u32 = 640;
    const topk: u32 = 10;
    const experts: u32 = 48;
    const hot: u32 = 6;
    const n: u32 = 80;
    const max_rows: u32 = 16;
    var prng = std.Random.DefaultPrng.init(0xBA7C4);
    const rnd = prng.random();
    const rate = format.Rate.fromN(n).?;
    var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
    defer bank.deinit(a);
    const mark = g.mark();
    defer g.resetTo(mark);
    const ht = ci(hidden / 16);
    const it = ci(inter / 16);
    const arrays: decode.Kernels.Bank = .{
        .gate_t = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(experts), ht, it, ci(n) }, .uint16),
        .gate_suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ ci(experts), ci(hidden) }, .float16),
        .gate_svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ ci(experts), ci(inter) }, .float16),
        .up_t = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ ci(experts), ht, it, ci(n) }, .uint16),
        .up_suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ ci(experts), ci(hidden) }, .float16),
        .up_svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ ci(experts), ci(inter) }, .float16),
        .down_t = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ ci(experts), it, ht, ci(n) }, .uint16),
        .down_suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ ci(experts), ci(inter) }, .float16),
        .down_svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ ci(experts), ci(hidden) }, .float16),
    };
    const x = try a.alloc(f32, max_rows * hidden);
    defer a.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const slots = try a.alloc(u32, max_rows * topk);
    defer a.free(slots);
    const scores = try a.alloc(f32, max_rows * topk);
    defer a.free(scores);
    for (0..max_rows) |r| {
        var k: usize = 0;
        while (k < topk) {
            const e: u32 = if (k < 4) rnd.uintLessThan(u32, hot) else hot + rnd.uintLessThan(u32, experts - hot);
            var dup = false;
            for (slots[r * topk ..][0..k]) |o| dup = dup or o == e;
            if (dup) continue;
            slots[r * topk + k] = e;
            k += 1;
        }
        var sum: f32 = 0;
        for (scores[r * topk ..][0..topk]) |*w| {
            w.* = 0.2 + rnd.float(f32);
            sum += w.*;
        }
        for (scores[r * topk ..][0..topk]) |*w| w.* /= sum;
    }
    const geo: decode.Geometry = .{ .hidden = hidden, .inter = inter, .topk = topk, .rate = rate, .dec = .mul1 };
    var failed = false;
    for ([_]bool{ true, false }) |defer_affine| {
        // root.zig's form-3 launch shape: 1 output tile per threadgroup, up to 4 K slices, expert-grouped row-slots
        var k3 = try decode.Kernels3.initWith(geo, 1, 4, 2);
        defer k3.deinit();
        k3.out_dtype = .float32;
        k3.defer_affine = defer_affine;
        const alone = try a.alloc(f32, max_rows * hidden);
        defer a.free(alone);
        for (0..max_rows) |r| {
            const xr = try g.hostArray(std.mem.sliceAsBytes(x[r * hidden ..][0..hidden]), &.{ 1, ci(hidden) }, .float32);
            const sr = try g.hostArray(std.mem.sliceAsBytes(slots[r * topk ..][0..topk]), &.{ci(topk)}, .uint32);
            const wr = try g.hostArray(std.mem.sliceAsBytes(scores[r * topk ..][0..topk]), &.{ci(topk)}, .float32);
            const y = try g.adopt(try k3.moe(s, xr, sr, wr, arrays, 1));
            _ = try g.hostF32(y, alone[r * hidden ..][0..hidden]);
        }
        var rows: u32 = 2;
        while (rows <= max_rows) : (rows += 1) {
            const xb = try g.hostArray(std.mem.sliceAsBytes(x[0 .. rows * hidden]), &.{ ci(rows), ci(hidden) }, .float32);
            const sb = try g.hostArray(std.mem.sliceAsBytes(slots[0 .. rows * topk]), &.{ci(rows * topk)}, .uint32);
            const wb = try g.hostArray(std.mem.sliceAsBytes(scores[0 .. rows * topk]), &.{ci(rows * topk)}, .float32);
            const y = try g.adopt(try k3.moe(s, xb, sb, wb, arrays, rows));
            const got = try a.alloc(f32, rows * hidden);
            defer a.free(got);
            _ = try g.hostF32(y, got);
            var diff: usize = 0;
            for (got, alone[0 .. rows * hidden]) |p, q| {
                if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) diff += 1;
            }
            // the size of any difference on the parity tests' scale: max |batched - alone| over the rows' rms
            const e = ref.compare(got, alone[0 .. rows * hidden]);
            std.debug.print("TURBOEXL3_ROWBYTES form3 defer={} rows={d} differing={d}/{d} max_abs_over_rms={e:.2} {s}\n", .{ defer_affine, rows, diff, got.len, e.max_ratio, if (diff == 0) "PASS" else "FAIL" });
            if (diff != 0) failed = true;
        }
    }
    try testing.expect(!failed);
}

// The host's SwiGLU kernel (transformer.zig SWIGLU_SOURCE, verbatim): the reference for the fused shared expert's SwiGLU.
const host_swiglu_source = @embedFile("kernels/test/host_swiglu.metal");

// The gated shared expert inside form 9's launches (`moeShared`, with the fold of fin7s into gu3s past gu8_rows and
// without), in launches of its own beside them (`moeSharedSide`) and through the v2 heads, against the host's chain: form
// 3, the shared gate / up / down linears (the three-launch chain), the table SwiGLU and MLX's Sigmoid, Multiply and Add;
// rows 1..4, both routed MUL1 decodes, form 9's tiles 1 and 2. Ends with TURBOEXL3_SHARED_BYTES_DONE.
test "turboexl3 gpu: the shared expert fused into form 9's launches equals the host's chain byte for byte (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const dense = @import("dense.zig");
    const ks = @import("sdk").kernel_launch;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const hidden: u32 = 2560;
    const inter: u32 = 640;
    const topk: u32 = 10;
    const experts: u32 = 48;
    const hot: u32 = 6;
    const n: u32 = 80;
    const n_gu: u32 = 112; // the full pack's shared gate / up and down rates
    const n_dn: u32 = 96;
    const max_rows: u32 = 4;
    var prng = std.Random.DefaultPrng.init(0x5A4ED);
    const rnd = prng.random();
    var bank = try ref.synthBank(a, rnd, experts, hidden, inter, format.Rate.fromN(n).?);
    defer bank.deinit(a);
    var sgu = try ref.synthBank(a, rnd, 1, hidden, inter, format.Rate.fromN(n_gu).?);
    defer sgu.deinit(a);
    var sdn = try ref.synthBank(a, rnd, 1, hidden, inter, format.Rate.fromN(n_dn).?);
    defer sdn.deinit(a);
    const mark = g.mark();
    defer g.resetTo(mark);
    const ht = ci(hidden / 16);
    const it = ci(inter / 16);
    const arrays: decode.Kernels.Bank = .{
        .gate_t = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(experts), ht, it, ci(n) }, .uint16),
        .gate_suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ ci(experts), ci(hidden) }, .float16),
        .gate_svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ ci(experts), ci(inter) }, .float16),
        .up_t = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ ci(experts), ht, it, ci(n) }, .uint16),
        .up_suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ ci(experts), ci(hidden) }, .float16),
        .up_svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ ci(experts), ci(inter) }, .float16),
        .down_t = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ ci(experts), it, ht, ci(n) }, .uint16),
        .down_suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ ci(experts), ci(inter) }, .float16),
        .down_svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ ci(experts), ci(hidden) }, .float16),
    };
    // the shared expert as the host holds it: dense trellises [in/16, out/16, n]
    const sh = .{
        .gate_t = try g.hostArray(std.mem.sliceAsBytes(sgu.gate_t), &.{ ht, it, ci(n_gu) }, .uint16),
        .gate_suh = try g.hostArray(std.mem.sliceAsBytes(sgu.gate_suh), &.{ci(hidden)}, .float16),
        .gate_svh = try g.hostArray(std.mem.sliceAsBytes(sgu.gate_svh), &.{ci(inter)}, .float16),
        .up_t = try g.hostArray(std.mem.sliceAsBytes(sgu.up_t), &.{ ht, it, ci(n_gu) }, .uint16),
        .up_suh = try g.hostArray(std.mem.sliceAsBytes(sgu.up_suh), &.{ci(hidden)}, .float16),
        .up_svh = try g.hostArray(std.mem.sliceAsBytes(sgu.up_svh), &.{ci(inter)}, .float16),
        .down_t = try g.hostArray(std.mem.sliceAsBytes(sdn.down_t), &.{ it, ht, ci(n_dn) }, .uint16),
        .down_suh = try g.hostArray(std.mem.sliceAsBytes(sdn.down_suh), &.{ci(inter)}, .float16),
        .down_svh = try g.hostArray(std.mem.sliceAsBytes(sdn.down_svh), &.{ci(hidden)}, .float16),
    };
    // the SwiGLU table as the host builds it: MLX's sigmoid of every bf16 pattern
    const pats = try a.alloc(f32, 1 << 16);
    defer a.free(pats);
    for (pats, 0..) |*v, i| v.* = @bitCast(@as(u32, @intCast(i)) << 16);
    const vals = try g.astype(try g.hostArray(std.mem.sliceAsBytes(pats), &.{1 << 16}, .float32), .bfloat16);
    var sigtab_raw = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&sigtab_raw, vals, s));
    const sigtab = try g.adopt(sigtab_raw);
    try g.evalAll(&.{sigtab});
    const x = try a.alloc(f32, max_rows * hidden);
    defer a.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const slots = try a.alloc(u32, max_rows * topk);
    defer a.free(slots);
    const scores = try a.alloc(f32, max_rows * topk);
    defer a.free(scores);
    const logits = try a.alloc(f32, max_rows);
    defer a.free(logits);
    for (logits) |*v| v.* = 2.0 * rnd.floatNorm(f32);
    for (0..max_rows) |r| {
        var k: usize = 0;
        while (k < topk) {
            const e: u32 = if (k < 4) rnd.uintLessThan(u32, hot) else hot + rnd.uintLessThan(u32, experts - hot);
            var dup = false;
            for (slots[r * topk ..][0..k]) |o| dup = dup or o == e;
            if (dup) continue;
            slots[r * topk + k] = e;
            k += 1;
        }
        var sum: f32 = 0;
        for (scores[r * topk ..][0..topk]) |*w| {
            w.* = 0.2 + rnd.float(f32);
            sum += w.*;
        }
        for (scores[r * topk ..][0..topk]) |*w| w.* /= sum;
    }
    const dec: format.Decode = .mul1;
    const geo: decode.Geometry = .{ .hidden = hidden, .inter = inter, .topk = topk, .rate = format.Rate.fromN(n).?, .dec = dec };
    var dg = try dense.Dense.init(.{ .in = hidden, .out = inter, .rate = format.Rate.fromN(n_gu).?, .dec = dec, .out_dtype = .bfloat16 });
    defer dg.deinit();
    var dd = try dense.Dense.init(.{ .in = inter, .out = hidden, .rate = format.Rate.fromN(n_dn).?, .dec = dec, .out_dtype = .bfloat16 });
    defer dd.deinit();
    // the reference linears are the three-launch chain (DEC-ONE, its bytes, has its own test)
    dg.dec_on = false;
    dd.dec_on = false;
    const swiglu = ks.newKernel("turboexl3_test_host_swiglu", &.{ "gate", "up", "sigtab", "N_size" }, &.{"y"}, host_swiglu_source, "", true) orelse return error.KernelCreateFailed;
    defer decode.freeKernel(swiglu);
    var failed = false;
    var checks: usize = 0;
    for ([_]bool{ true, false }) |defer_affine| for ([_]u32{ 1, 2 }) |bt9| {
        var kref = try decode.Kernels3.initWith(geo, 1, 4, 2);
        defer kref.deinit();
        kref.out_dtype = .bfloat16;
        kref.defer_affine = defer_affine;
        var k3 = try decode.Kernels3.initWith(geo, 1, 4, 2);
        defer k3.deinit();
        k3.out_dtype = .bfloat16;
        k3.defer_affine = defer_affine;
        k3.bt9 = bt9;
        for (1..max_rows + 1) |rows| {
            const xb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x[0 .. rows * hidden]), &.{ ci(@intCast(rows)), ci(hidden) }, .float32), .bfloat16);
            const sb_ = try g.hostArray(std.mem.sliceAsBytes(slots[0 .. rows * topk]), &.{ci(@intCast(rows * topk))}, .uint32);
            const wb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(scores[0 .. rows * topk]), &.{ci(@intCast(rows * topk))}, .float32), .bfloat16);
            const lg = try g.astype(try g.hostArray(std.mem.sliceAsBytes(logits[0..rows]), &.{ ci(@intCast(rows)), 1 }, .float32), .bfloat16);
            // the host's chain
            const yr = try g.adopt(try kref.moeForm(s, xb, sb_, wb, arrays, rows, 3));
            const gate_o = try g.adopt(try dg.linear(s, xb, sh.gate_t, sh.gate_suh, sh.gate_svh, rows));
            const up_o = try g.adopt(try dg.linear(s, xb, sh.up_t, sh.up_suh, sh.up_svh, rows));
            const cnt: c_int = ci(@intCast(rows * inter));
            const tgs: u32 = @min(256, @max(32, ((@as(u32, @intCast(cnt)) + 31) / 32) * 32));
            const scfg = try ks.buildConfig(.{ .grid = .{ ((@as(u32, @intCast(cnt)) + tgs - 1) / tgs) * tgs, 1, 1 }, .threadgroup = .{ tgs, 1, 1 }, .out_shapes = &.{&[_]c_int{ ci(@intCast(rows)), ci(inter) }}, .out_dtypes = &.{.bfloat16}, .template = &.{.{ .name = "T", .value = .{ .dtype = .bfloat16 } }} });
            defer _ = mlx.mlx_fast_metal_kernel_config_free(scfg);
            const nsz = mlx.mlx_array_new_int(cnt);
            defer _ = mlx.mlx_array_free(nsz);
            var act_o: [1]mlx.mlx_array = undefined;
            try ks.applyConfig(swiglu, scfg, &.{ gate_o, up_o, sigtab, nsz }, &act_o, s);
            const act = try g.adopt(act_o[0]);
            const down_o = try g.adopt(try dd.linear(s, act, sh.down_t, sh.down_suh, sh.down_svh, rows));
            var sig_raw = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_sigmoid(&sig_raw, lg, s));
            const sig = try g.adopt(sig_raw);
            var gated_raw = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&gated_raw, sig, down_o, s));
            const gated = try g.adopt(gated_raw);
            var want_raw = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&want_raw, yr, gated, s));
            const want_a = try g.adopt(want_raw);
            // the fused launches
            const sbank: decode.Kernels3.SharedBank = .{
                .gate_t = sh.gate_t,
                .gate_suh = sh.gate_suh,
                .gate_svh = sh.gate_svh,
                .up_t = sh.up_t,
                .up_suh = sh.up_suh,
                .up_svh = sh.up_svh,
                .down_t = sh.down_t,
                .down_suh = sh.down_suh,
                .down_svh = sh.down_svh,
                .n_gu = n_gu,
                .n_dn = n_dn,
                .ks_gu = dense.Dense.ksFor(hidden, inter),
                .ks_dn = dense.Dense.ksFor(inter, hidden),
                .defer_affine = dense.Dense.deferFor(dec),
                .sigtab = sigtab,
                .logit = try g.slice(lg, &.{ 0, 0 }, &.{ ci(@intCast(rows)), 1 }, &.{ 1, 1 }),
            };
            if (!k3.sharedAdmitted(rows, sbank)) {
                std.debug.print("TURBOEXL3_SHARED_BYTES rows={d} NOT_ADMITTED FAIL\n", .{rows});
                failed = true;
                continue;
            }
            var l1 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&l1, lg, &[_]c_int{ci(@intCast(rows))}, 1, s));
            var sb2 = sbank;
            sb2.logit = try g.adopt(l1);
            const want = try a.alloc(f32, rows * hidden);
            defer a.free(want);
            _ = try g.hostF32(try g.astype(want_a, .float32), want);
            const got = try a.alloc(f32, rows * hidden);
            defer a.free(got);
            // inside the routed launches (`moeShared`: gu3sf past gu8_rows, "inside_nofold" its gu3s + fin7s) and in
            // launches of their own beside them (`moeSharedSide`), K slices per shared GEMV threadgroup (gate|up, down): one
            // each, root.zig's served 4 and 1, and 5 each (a launch shape: the planes, so the bytes, must not change; the
            // shared arrivals per block change with it: 160, 40, 32 at bt9 2)
            for ([_][2]u32{ .{ 1, 1 }, .{ 4, 1 }, .{ 5, 5 } }) |sp| for ([_]u8{ 0, 1, 2 }) |mode| {
                const side = mode == 2;
                k3.sspt = sp[0];
                k3.ssptd = sp[1];
                k3.fused_fin_fold = mode == 0;
                defer k3.fused_fin_fold = true;
                const got_a = try g.adopt(if (side) try k3.moeSharedSide(s, xb, sb_, wb, arrays, rows, sb2) else try k3.moeShared(s, xb, sb_, wb, arrays, rows, sb2));
                _ = try g.hostF32(try g.astype(got_a, .float32), got);
                var diff: usize = 0;
                for (got, want) |p, q| {
                    if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) diff += 1;
                }
                const e = ref.compare(got, want);
                checks += 1;
                std.debug.print("TURBOEXL3_SHARED_BYTES mode={s} folded={} spt={d} sptd={d} routed_defer={} bt9={d} rows={d} ks_gu={d} ks_dn={d} differing={d}/{d} max_abs_over_rms={e:.2} {s}\n", .{ if (side) "side" else if (mode == 0) "inside" else "inside_nofold", mode == 0 and rows > k3.gu8_rows, sp[0], sp[1], defer_affine, bt9, rows, sbank.ks_gu, sbank.ks_dn, diff, got.len, e.max_ratio, if (diff == 0) "PASS" else "FAIL" });
                if (diff != 0) failed = true;
            };
            // the v2 heads (`moeShared2` / `moeSharedSide2`) through every launch-shape knob's code path: output tiles
            // per simdgroup (1, 2; 4: threadgroups across both projections, rotating both; 5, 10: whole groups; down 1, 2, 4, 5,
            // 8), the gate|up input prepared in prep2t / sprep2 or rotated in-kernel (gu8t up to 4 rows, gu3t, sgu2), both plane
            // layouts, every preload depth, both grid orders; the planes, so the bytes, must not change
            const V2 = struct { ots: u32, otsd: u32, prep: bool, lm: bool, pre: u32, first: bool, gu8r: usize };
            const v2s = [_]V2{
                .{ .ots = 2, .otsd = 2, .prep = true, .lm = true, .pre = 4, .first = false, .gu8r = 1 },
                .{ .ots = 1, .otsd = 1, .prep = false, .lm = false, .pre = 0, .first = true, .gu8r = 1 },
                .{ .ots = 4, .otsd = 4, .prep = false, .lm = false, .pre = 8, .first = true, .gu8r = 1 },
                .{ .ots = 5, .otsd = 5, .prep = false, .lm = true, .pre = 2, .first = false, .gu8r = 4 },
                .{ .ots = 10, .otsd = 8, .prep = true, .lm = true, .pre = 8, .first = true, .gu8r = 1 },
            };
            // TURBOEXL3_SHARED_V2=<i,j,...> runs only these configurations; each run is announced before it builds, so a
            // back-end compiler abort (which ends the binary) names its configuration
            const v2_only = std.c.getenv("TURBOEXL3_SHARED_V2");
            for (v2s, 0..) |v, vi| for ([_]bool{ false, true }) |side| {
                if (v2_only) |only| {
                    var keep = false;
                    var cfgs_it = std.mem.splitScalar(u8, std.mem.span(only), ',');
                    while (cfgs_it.next()) |tok| {
                        if ((std.fmt.parseInt(usize, tok, 10) catch continue) == vi) keep = true;
                    }
                    if (!keep) continue;
                }
                std.debug.print("TURBOEXL3_SHARED_BYTES_RUN mode={s} cfg={d} rows={d} routed_defer={} bt9={d}\n", .{ if (side) "side2" else "inside2", vi, rows, defer_affine, bt9 });
                k3.sots = @splat(v.ots);
                k3.sotsd = @splat(v.otsd);
                k3.sprep = v.prep;
                k3.slm = v.lm;
                k3.spre = @splat(v.pre);
                k3.sfirst = v.first;
                k3.gu8_rows = v.gu8r;
                const got_a = try g.adopt(if (side) try k3.moeSharedSide2(s, xb, sb_, wb, arrays, rows, sb2) else try k3.moeShared2(s, xb, sb_, wb, arrays, rows, sb2));
                _ = try g.hostF32(try g.astype(got_a, .float32), got);
                var diff: usize = 0;
                for (got, want) |p, q| {
                    if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) diff += 1;
                }
                const e = ref.compare(got, want);
                const ran = k3.sharedTiles(rows);
                checks += 1;
                std.debug.print("TURBOEXL3_SHARED_BYTES mode={s} ots={d} otsd={d} ots_run={d} otsd_run={d} prep={} lm={} pre={d} first={} gu8_rows={d} routed_defer={} bt9={d} rows={d} ks_gu={d} ks_dn={d} differing={d}/{d} max_abs_over_rms={e:.2} {s}\n", .{ if (side) "side2" else "inside2", v.ots, v.otsd, ran[0], ran[1], v.prep, v.lm, v.pre, v.first, v.gu8r, defer_affine, bt9, rows, sbank.ks_gu, sbank.ks_dn, diff, got.len, e.max_ratio, if (diff == 0) "PASS" else "FAIL" });
                if (diff != 0) failed = true;
            };
            // the v1 runs of the next row count use the served gate|up prepare split (v1's configs do not key on it)
            k3.gu8_rows = 1;
        }
    };
    std.debug.print("TURBOEXL3_SHARED_BYTES_DONE checks={d} {s}\n", .{ checks, if (failed) "FAIL" else "PASS" });
    try testing.expect(!failed);
}

test "turboexl3 gpu: decode chain forms 7, 8 and 9 equal form 3 byte for byte, at any tiles per threadgroup (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const hidden: u32 = 2560;
    const inter: u32 = 640;
    const topk: u32 = 10;
    const experts: u32 = 48;
    const hot: u32 = 6;
    const n: u32 = 80;
    const max_rows: u32 = 16;
    var prng = std.Random.DefaultPrng.init(0xF0F7);
    const rnd = prng.random();
    const rate = format.Rate.fromN(n).?;
    var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
    defer bank.deinit(a);
    const mark = g.mark();
    defer g.resetTo(mark);
    const ht = ci(hidden / 16);
    const it = ci(inter / 16);
    const arrays: decode.Kernels.Bank = .{
        .gate_t = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(experts), ht, it, ci(n) }, .uint16),
        .gate_suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ ci(experts), ci(hidden) }, .float16),
        .gate_svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ ci(experts), ci(inter) }, .float16),
        .up_t = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ ci(experts), ht, it, ci(n) }, .uint16),
        .up_suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ ci(experts), ci(hidden) }, .float16),
        .up_svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ ci(experts), ci(inter) }, .float16),
        .down_t = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ ci(experts), it, ht, ci(n) }, .uint16),
        .down_suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ ci(experts), ci(inter) }, .float16),
        .down_svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ ci(experts), ci(hidden) }, .float16),
    };
    const x = try a.alloc(f32, max_rows * hidden);
    defer a.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const slots = try a.alloc(u32, max_rows * topk);
    defer a.free(slots);
    const scores = try a.alloc(f32, max_rows * topk);
    defer a.free(scores);
    for (0..max_rows) |r| {
        var k: usize = 0;
        while (k < topk) {
            const e: u32 = if (k < 4) rnd.uintLessThan(u32, hot) else hot + rnd.uintLessThan(u32, experts - hot);
            var dup = false;
            for (slots[r * topk ..][0..k]) |o| dup = dup or o == e;
            if (dup) continue;
            slots[r * topk + k] = e;
            k += 1;
        }
        var sum: f32 = 0;
        for (scores[r * topk ..][0..topk]) |*w| {
            w.* = 0.2 + rnd.float(f32);
            sum += w.*;
        }
        for (scores[r * topk ..][0..topk]) |*w| w.* /= sum;
    }
    const geo: decode.Geometry = .{ .hidden = hidden, .inter = inter, .topk = topk, .rate = rate, .dec = .mul1 };
    // bf16: the served operands (x, scores and y bf16); the others f32 as the form 5 test. Every form of a variant against
    // form 3 at one tile per threadgroup (the served launch shape), so the tiles (bt, form 9's bt1) are shown not to matter
    const Variant = struct { defer_affine: bool, bt: u32, bt1: u32, group: u32, bf16: bool, kdiv: [4]u32 = .{ 1, 2, 2, 4 }, bt9: u32 = 2, kdiv9: [4]u32 = .{ 1, 1, 1, 1 } };
    const variants = [_]Variant{
        .{ .defer_affine = true, .bt = 1, .bt1 = 2, .group = 2, .bf16 = false },
        .{ .defer_affine = true, .bt = 1, .bt1 = 2, .group = 2, .bf16 = true },
        .{ .defer_affine = true, .bt = 2, .bt1 = 4, .group = 2, .bf16 = false, .kdiv = .{ 1, 1, 1, 2 }, .bt9 = 4, .kdiv9 = .{ 2, 1, 2, 4 } },
        .{ .defer_affine = true, .bt = 1, .bt1 = 8, .group = 1, .bf16 = false, .kdiv = .{ 2, 4, 8, 1 }, .bt9 = 1 },
        .{ .defer_affine = false, .bt = 1, .bt1 = 2, .group = 2, .bf16 = false },
    };
    const row_set = [_]u32{ 1, 2, 3, 4, 5, 8, 16 };
    var failed = false;
    for (variants) |v| {
        var k3 = try decode.Kernels3.initWith(geo, v.bt, 4, v.group);
        defer k3.deinit();
        k3.out_dtype = if (v.bf16) .bfloat16 else .float32;
        k3.defer_affine = v.defer_affine;
        k3.bt1 = v.bt1;
        k3.kdiv = v.kdiv;
        k3.bt9 = v.bt9;
        k3.kdiv9 = v.kdiv9;
        var kref = try decode.Kernels3.initWith(geo, 1, 4, v.group);
        defer kref.deinit();
        kref.out_dtype = k3.out_dtype;
        kref.defer_affine = v.defer_affine;
        for (row_set) |rows| {
            const dt: mlx.mlx_dtype = if (v.bf16) .bfloat16 else .float32;
            const xb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x[0 .. rows * hidden]), &.{ ci(rows), ci(hidden) }, .float32), dt);
            const sb = try g.hostArray(std.mem.sliceAsBytes(slots[0 .. rows * topk]), &.{ci(rows * topk)}, .uint32);
            const wb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(scores[0 .. rows * topk]), &.{ci(rows * topk)}, .float32), dt);
            const y3 = try g.adopt(try kref.moeForm(s, xb, sb, wb, arrays, rows, 3));
            const got3 = try a.alloc(f32, rows * hidden);
            defer a.free(got3);
            _ = try g.hostF32(try g.astype(y3, .float32), got3);
            for ([_]u8{ 3, 7, 8, 9 }) |form| {
                const yf = try g.adopt(try k3.moeForm(s, xb, sb, wb, arrays, rows, form));
                const gotf = try a.alloc(f32, rows * hidden);
                defer a.free(gotf);
                _ = try g.hostF32(try g.astype(yf, .float32), gotf);
                var diff: usize = 0;
                for (gotf, got3) |p, q| {
                    if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) diff += 1;
                }
                const e = ref.compare(gotf, got3);
                std.debug.print("TURBOEXL3_FORM7_BYTES form={d} defer={} bt={d} bt1={d} group={d} bf16={} kdiv={d},{d},{d},{d} rows={d} gu8={} differing={d}/{d} max_abs_over_rms={e:.2} {s}\n", .{ form, v.defer_affine, v.bt, v.bt1, v.group, v.bf16, v.kdiv[0], v.kdiv[1], v.kdiv[2], v.kdiv[3], rows, (form == 8 and k3.gu8Admitted(rows, k3.kdiv)) or (form == 9 and rows == 1 and k3.gu8Admitted(rows, k3.kdiv9)), diff, gotf.len, e.max_ratio, if (diff == 0) "PASS" else "FAIL" });
                if (diff != 0) failed = true;
            }
        }
    }
    try testing.expect(!failed);
}

// Form 5 (three launches, expert groups of up to GMAX row-slots) against form 3 on the same inputs: every output's f32 bits
// equal at rows 1..5, 8, 16 for several (gmax, bt) and both MUL1 decodes. The routing is the form 3 test's: 4 of a row's 10
// slots from a hot pool of 6 experts, so groups of 2..4 members form from two rows up.
test "turboexl3 gpu: decode chain form 5 equals form 3 byte for byte (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const hidden: u32 = 2560;
    const inter: u32 = 640;
    const topk: u32 = 10;
    const experts: u32 = 48;
    const hot: u32 = 6;
    const n: u32 = 80;
    const max_rows: u32 = 16;
    var prng = std.Random.DefaultPrng.init(0xF0F5);
    const rnd = prng.random();
    const rate = format.Rate.fromN(n).?;
    var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
    defer bank.deinit(a);
    const mark = g.mark();
    defer g.resetTo(mark);
    const ht = ci(hidden / 16);
    const it = ci(inter / 16);
    const arrays: decode.Kernels.Bank = .{
        .gate_t = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(experts), ht, it, ci(n) }, .uint16),
        .gate_suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ ci(experts), ci(hidden) }, .float16),
        .gate_svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ ci(experts), ci(inter) }, .float16),
        .up_t = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ ci(experts), ht, it, ci(n) }, .uint16),
        .up_suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ ci(experts), ci(hidden) }, .float16),
        .up_svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ ci(experts), ci(inter) }, .float16),
        .down_t = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ ci(experts), it, ht, ci(n) }, .uint16),
        .down_suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ ci(experts), ci(inter) }, .float16),
        .down_svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ ci(experts), ci(hidden) }, .float16),
    };
    const x = try a.alloc(f32, max_rows * hidden);
    defer a.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const slots = try a.alloc(u32, max_rows * topk);
    defer a.free(slots);
    const scores = try a.alloc(f32, max_rows * topk);
    defer a.free(scores);
    for (0..max_rows) |r| {
        var k: usize = 0;
        while (k < topk) {
            const e: u32 = if (k < 4) rnd.uintLessThan(u32, hot) else hot + rnd.uintLessThan(u32, experts - hot);
            var dup = false;
            for (slots[r * topk ..][0..k]) |o| dup = dup or o == e;
            if (dup) continue;
            slots[r * topk + k] = e;
            k += 1;
        }
        var sum: f32 = 0;
        for (scores[r * topk ..][0..topk]) |*w| {
            w.* = 0.2 + rnd.float(f32);
            sum += w.*;
        }
        for (scores[r * topk ..][0..topk]) |*w| w.* /= sum;
    }
    const geo: decode.Geometry = .{ .hidden = hidden, .inter = inter, .topk = topk, .rate = rate, .dec = .mul1 };
    const Variant = struct { defer_affine: bool, gmax: u32, bt: u32 };
    const variants = [_]Variant{
        .{ .defer_affine = true, .gmax = 4, .bt = 4 },
        .{ .defer_affine = true, .gmax = 2, .bt = 1 },
        .{ .defer_affine = true, .gmax = 1, .bt = 2 },
        .{ .defer_affine = true, .gmax = 3, .bt = 8 },
        .{ .defer_affine = false, .gmax = 4, .bt = 4 },
    };
    const row_set = [_]u32{ 1, 2, 3, 4, 5, 8, 16 };
    var failed = false;
    for (variants) |v| {
        var k3 = try decode.Kernels3.initWith(geo, 1, 4, 2);
        defer k3.deinit();
        k3.out_dtype = .float32;
        k3.defer_affine = v.defer_affine;
        var k5 = try decode.Kernels5.init(geo, v.bt, 4, v.gmax);
        defer k5.deinit();
        k5.out_dtype = .float32;
        k5.defer_affine = v.defer_affine;
        for (row_set) |rows| {
            const xb = try g.hostArray(std.mem.sliceAsBytes(x[0 .. rows * hidden]), &.{ ci(rows), ci(hidden) }, .float32);
            const sb = try g.hostArray(std.mem.sliceAsBytes(slots[0 .. rows * topk]), &.{ci(rows * topk)}, .uint32);
            const wb = try g.hostArray(std.mem.sliceAsBytes(scores[0 .. rows * topk]), &.{ci(rows * topk)}, .float32);
            const y3 = try g.adopt(try k3.moe(s, xb, sb, wb, arrays, rows));
            const got3 = try a.alloc(f32, rows * hidden);
            defer a.free(got3);
            _ = try g.hostF32(y3, got3);
            // form 5 (three launches) and form 6 (the slot reduce by arrival in the down launch)
            for ([_]bool{ false, true }) |fused| {
                const y5 = try g.adopt(try k5.moe(s, xb, sb, wb, arrays, rows, fused));
                const got5 = try a.alloc(f32, rows * hidden);
                defer a.free(got5);
                _ = try g.hostF32(y5, got5);
                var diff: usize = 0;
                for (got5, got3) |p, q| {
                    if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) diff += 1;
                }
                const e = ref.compare(got5, got3);
                std.debug.print("TURBOEXL3_FORM5_BYTES form={d} defer={} gmax={d} bt={d} rows={d} differing={d}/{d} max_abs_over_rms={e:.2} {s}\n", .{ @as(u32, if (fused) 6 else 5), v.defer_affine, v.gmax, v.bt, rows, diff, got5.len, e.max_ratio, if (diff == 0) "PASS" else "FAIL" });
                if (diff != 0) failed = true;
            }
        }
    }
    try testing.expect(!failed);
}

// ── The router folded into form 9's prepare launch ──

/// Router test data: `n` f32 draws (normal x `scale`; `steps` > 0 quantizes to multiples of 1 / steps, so a bf16 row holds
/// exact ties) as an array of `dt` and `shape`.
fn routerRand(g: *G, rnd: std.Random, shape: []const c_int, scale: f32, steps: f32, dt: mlx.mlx_dtype) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try std.heap.page_allocator.alloc(f32, n);
    defer std.heap.page_allocator.free(buf);
    for (buf) |*v| {
        const r = rnd.floatNorm(f32) * scale;
        v.* = if (steps > 0) @round(r * steps) / steps else r;
    }
    const a = try g.hostArray(std.mem.sliceAsBytes(buf), shape, .float32);
    if (dt == .float32) return a;
    return g.astype(a, dt);
}

/// The elements whose bits differ between two arrays of one size (each widened to f32 first, exact for bf16 / f16; u32 read
/// as is); `n` the element count.
fn routerDiff(g: *G, x: mlx.mlx_array, y: mlx.mlx_array) !struct { diff: usize, n: usize } {
    const a = std.heap.page_allocator;
    const n = mlx.mlx_array_size(x);
    if (n != mlx.mlx_array_size(y)) return error.SizeMismatch;
    if (mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(y)) return error.DtypeMismatch;
    var d: usize = 0;
    if (mlx.mlx_array_dtype(x) == .uint32) {
        const p = try a.alloc(u32, n);
        defer a.free(p);
        const q = try a.alloc(u32, n);
        defer a.free(q);
        _ = try g.hostU32(x, p);
        _ = try g.hostU32(y, q);
        for (p, q) |u, v| d += @intFromBool(u != v);
    } else {
        const p = try a.alloc(f32, n);
        defer a.free(p);
        const q = try a.alloc(f32, n);
        defer a.free(q);
        _ = try g.hostF32(if (mlx.mlx_array_dtype(x) == .float32) x else try g.astype(x, .float32), p);
        _ = try g.hostF32(if (mlx.mlx_array_dtype(y) == .float32) y else try g.astype(y, .float32), q);
        for (p, q) |u, v| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
    }
    return .{ .diff = d, .n = n };
}

/// The chain's router: the host's softmax kernel (`decode.text_router_softmax`) at the host's config
/// (`moeRouterTopK`: TG = moeRouterThreadGroup(E), template TOUT NE NK TG NORM); inds u32 and scores in `tout`.
fn routerTopK(k: mlx.mlx_fast_metal_kernel, s: mlx.mlx_stream, logits: mlx.mlx_array, rows: u32, ne: u32, topk: u32, tout: mlx.mlx_dtype) ![2]mlx.mlx_array {
    const ks = @import("sdk").kernel_launch;
    const tg: u32 = @max(32, @min(256, ((ne + 31) / 32) * 32));
    const cfg = try ks.buildConfig(.{
        .grid = .{ tg, rows, 1 },
        .threadgroup = .{ tg, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ ci(rows), ci(topk) }, &[_]c_int{ ci(rows), ci(topk) } },
        .out_dtypes = &.{ .uint32, tout },
        .template = &.{ .{ .name = "TOUT", .value = .{ .dtype = tout } }, .{ .name = "NE", .value = .{ .int = ci(ne) } }, .{ .name = "NK", .value = .{ .int = ci(topk) } }, .{ .name = "TG", .value = .{ .int = ci(tg) } }, .{ .name = "NORM", .value = .{ .int = 1 } } },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [2]mlx.mlx_array = undefined;
    try ks.applyConfig(k, cfg, &.{logits}, &o, s);
    return o;
}

/// MLX's logit of the shared gate as the host computes it: x [1, rows, H] @ w^T, w [1, H] transposed as a view
/// (`maybeTransposeForBf16`), one `mlx_matmul` (the gemv the merged launch's gate logit replicates); bf16 [1, rows, 1].
fn sharedGateLogitMlx(g: *G, s: mlx.mlx_stream, x2: mlx.mlx_array, w1: mlx.mlx_array, rows: u32, hidden: u32) !mlx.mlx_array {
    var x3 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&x3, x2, &[_]c_int{ 1, ci(rows), ci(hidden) }, 3, s));
    _ = try g.adopt(x3);
    var wt = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose(&wt, w1, s));
    _ = try g.adopt(wt);
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&y, x3, wt, s));
    return g.adopt(y);
}

test "turboexl3 gpu: the router folded into form 9's prepare launch (and the shared gate's logit with it) equals the router kernel + prep2 and the host's chain byte for byte (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const dense = @import("dense.zig");
    const ks = @import("sdk").kernel_launch;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const hidden: u32 = 2560;
    const inter: u32 = 640;
    const topk: u32 = 10;
    var prng = std.Random.DefaultPrng.init(0x4DECC5);
    const rnd = prng.random();
    const router = ks.newKernel("turboexl3_test_router", &decode.router_inputs, &decode.router_outputs, decode.text_router_softmax, "", true) orelse return error.KernelCreateFailed;
    defer decode.freeKernel(router);
    var failed = false;
    var checks: usize = 0;
    // the shared gate's weight [1, H] and its contiguous vector (the host's mw.shared_gate_vec)
    const gw1 = try routerRand(&g, rnd, &.{ 1, ci(hidden) }, 0.05, 0, .bfloat16);
    var gvec_raw = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&gvec_raw, gw1, &[_]c_int{ci(hidden)}, 1, s));
    const gvec = try g.adopt(gvec_raw);
    try g.evalAll(&.{ gw1, gvec });

    // 1. the merged launch alone against the router kernel + prep2 (form 9's prepare config): every output, at the served
    // 512 experts (only the gate / up suh are read: no trellis), rows 1..4, 8, 16, bf16 and f16 logits, with and without the
    // gate's logit
    {
        const ne: u32 = 512;
        const geo: decode.Geometry = .{ .hidden = hidden, .inter = inter, .topk = topk, .rate = format.Rate.fromN(80).?, .dec = .mul1 };
        var k3 = try decode.Kernels3.initWith(geo, 1, 4, 2);
        defer k3.deinit();
        k3.out_dtype = .bfloat16;
        const gsuh = try routerRand(&g, rnd, &.{ ci(ne), ci(hidden) }, 0.3, 0, .float16);
        const usuh = try routerRand(&g, rnd, &.{ ci(ne), ci(hidden) }, 0.3, 0, .float16);
        const kb: decode.Kernels.Bank = .{ .gate_t = gsuh, .gate_suh = gsuh, .gate_svh = gsuh, .up_t = usuh, .up_suh = usuh, .up_svh = usuh, .down_t = gsuh, .down_suh = gsuh, .down_svh = gsuh };
        for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |tout| for ([_]u32{ 1, 2, 3, 4, 8, 16 }) |rows| for ([_]bool{ false, true }) |gate| {
            if (gate and (tout != .bfloat16 or rows < 2 or rows > 4)) continue;
            const mark = g.mark();
            defer g.resetTo(mark);
            const x = try routerRand(&g, rnd, &.{ ci(rows), ci(hidden) }, 1.0, 0, .bfloat16);
            // ties: every other call's logits on a 1/8 grid (many exact ties in a row), the rest continuous
            const logits = try routerRand(&g, rnd, &.{ ci(rows), ci(ne) }, 3.0, if (rows % 2 == 0) @as(f32, 8) else 0, tout);
            const ro = try routerTopK(router, s, logits, rows, ne, topk, tout);
            const inds = try g.adopt(ro[0]);
            const scores = try g.adopt(ro[1]);
            var inds1 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&inds1, inds, &[_]c_int{ci(rows * topk)}, 1, s));
            const slots = try g.adopt(inds1);
            const xp = try k3.routeChainPrep(s, x, slots, kb, rows);
            for (xp) |o| _ = try g.adopt(o);
            const r = try k3.routeLaunch(s, logits, x, kb, rows, if (gate) gvec else null);
            inline for (.{ r.xg, r.xu, r.xsg, r.xsu, r.slots, r.scores }) |o| _ = try g.adopt(o);
            if (r.glogit.ctx != null) _ = try g.adopt(r.glogit);
            var sc1 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&sc1, scores, &[_]c_int{ci(rows * topk)}, 1, s));
            const scores1 = try g.adopt(sc1);
            const pairs = [_]struct { name: []const u8, got: mlx.mlx_array, want: mlx.mlx_array }{
                .{ .name = "slots", .got = r.slots, .want = slots },
                .{ .name = "scores", .got = r.scores, .want = scores1 },
                .{ .name = "xg", .got = r.xg, .want = xp[0] },
                .{ .name = "xu", .got = r.xu, .want = xp[1] },
                .{ .name = "xsg", .got = r.xsg, .want = xp[2] },
                .{ .name = "xsu", .got = r.xsu, .want = xp[3] },
            };
            for (pairs) |pr| {
                const d = try routerDiff(&g, pr.got, pr.want);
                checks += 1;
                std.debug.print("TURBOEXL3_ROUTER_BYTES part=launch out={s} experts={d} rows={d} tout={s} gate={} differing={d}/{d} {s}\n", .{ pr.name, ne, rows, @tagName(tout), gate, d.diff, d.n, if (d.diff == 0) "PASS" else "FAIL" });
                if (d.diff != 0) failed = true;
            }
            if (gate) {
                // the merged launch's gate logit against MLX's matmul (the host's qmatmulBits for the bf16 gate)
                const want = try sharedGateLogitMlx(&g, s, x, gw1, rows, hidden);
                const d = try routerDiff(&g, r.glogit, want);
                checks += 1;
                std.debug.print("TURBOEXL3_ROUTER_BYTES part=gate_logit rows={d} differing={d}/{d} {s}\n", .{ rows, d.diff, d.n, if (d.diff == 0) "PASS" else "FAIL" });
                if (d.diff != 0) failed = true;
            }
        };
    }

    // 2. end to end: moeSharedRouted (with the fold of fin7s into gu3s and without) against the router kernel + moeShared
    // without it (the host's chain at decode rows), rows 2..4, both routed deferrals, the gate's logit from MLX's matmul or
    // from the merged launch; 64 experts at the served rate and the served 512 at a light one (a 512-expert bank at 80
    // would be 1.6 GB)
    {
        const n_gu: u32 = 112;
        const n_dn: u32 = 96;
        var sgu = try ref.synthBank(a, rnd, 1, hidden, inter, format.Rate.fromN(n_gu).?);
        defer sgu.deinit(a);
        var sdn = try ref.synthBank(a, rnd, 1, hidden, inter, format.Rate.fromN(n_dn).?);
        defer sdn.deinit(a);
        const ht = ci(hidden / 16);
        const it = ci(inter / 16);
        const shm = g.mark();
        defer g.resetTo(shm);
        const sh = .{
            .gate_t = try g.hostArray(std.mem.sliceAsBytes(sgu.gate_t), &.{ ht, it, ci(n_gu) }, .uint16),
            .gate_suh = try g.hostArray(std.mem.sliceAsBytes(sgu.gate_suh), &.{ci(hidden)}, .float16),
            .gate_svh = try g.hostArray(std.mem.sliceAsBytes(sgu.gate_svh), &.{ci(inter)}, .float16),
            .up_t = try g.hostArray(std.mem.sliceAsBytes(sgu.up_t), &.{ ht, it, ci(n_gu) }, .uint16),
            .up_suh = try g.hostArray(std.mem.sliceAsBytes(sgu.up_suh), &.{ci(hidden)}, .float16),
            .up_svh = try g.hostArray(std.mem.sliceAsBytes(sgu.up_svh), &.{ci(inter)}, .float16),
            .down_t = try g.hostArray(std.mem.sliceAsBytes(sdn.down_t), &.{ it, ht, ci(n_dn) }, .uint16),
            .down_suh = try g.hostArray(std.mem.sliceAsBytes(sdn.down_suh), &.{ci(inter)}, .float16),
            .down_svh = try g.hostArray(std.mem.sliceAsBytes(sdn.down_svh), &.{ci(hidden)}, .float16),
        };
        const pats = try a.alloc(f32, 1 << 16);
        defer a.free(pats);
        for (pats, 0..) |*v, i| v.* = @bitCast(@as(u32, @intCast(i)) << 16);
        const vals = try g.astype(try g.hostArray(std.mem.sliceAsBytes(pats), &.{1 << 16}, .float32), .bfloat16);
        var sigtab_raw = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sigmoid(&sigtab_raw, vals, s));
        const sigtab = try g.adopt(sigtab_raw);
        try g.evalAll(&.{sigtab});
        for ([_][2]u32{ .{ 64, 80 }, .{ 512, 16 } }) |cfg| {
            const ne = cfg[0];
            const n = cfg[1];
            var bank = try ref.synthBank(a, rnd, ne, hidden, inter, format.Rate.fromN(n).?);
            defer bank.deinit(a);
            const bm = g.mark();
            defer g.resetTo(bm);
            const arrays: decode.Kernels.Bank = .{
                .gate_t = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(ne), ht, it, ci(n) }, .uint16),
                .gate_suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ ci(ne), ci(hidden) }, .float16),
                .gate_svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ ci(ne), ci(inter) }, .float16),
                .up_t = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ ci(ne), ht, it, ci(n) }, .uint16),
                .up_suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ ci(ne), ci(hidden) }, .float16),
                .up_svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ ci(ne), ci(inter) }, .float16),
                .down_t = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ ci(ne), it, ht, ci(n) }, .uint16),
                .down_suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ ci(ne), ci(inter) }, .float16),
                .down_svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ ci(ne), ci(hidden) }, .float16),
            };
            const geo: decode.Geometry = .{ .hidden = hidden, .inter = inter, .topk = topk, .rate = format.Rate.fromN(n).?, .dec = .mul1 };
            for ([_]bool{ true, false }) |defer_affine| {
                var k3 = try decode.Kernels3.initWith(geo, 1, 4, 2);
                defer k3.deinit();
                k3.out_dtype = .bfloat16;
                k3.defer_affine = defer_affine;
                for ([_]u32{ 1, 2, 3, 4 }) |rows| for ([_]bool{ false, true }) |gate| {
                    const mark = g.mark();
                    defer g.resetTo(mark);
                    const x = try routerRand(&g, rnd, &.{ ci(rows), ci(hidden) }, 1.0, 0, .bfloat16);
                    const logits = try routerRand(&g, rnd, &.{ ci(rows), ci(ne) }, 3.0, if (rows == 3) @as(f32, 8) else 0, .bfloat16);
                    const lg_rand = try routerRand(&g, rnd, &.{ci(rows)}, 2.0, 0, .bfloat16);
                    var lg = lg_rand;
                    if (gate and rows >= 2) {
                        var l1 = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_reshape(&l1, try sharedGateLogitMlx(&g, s, x, gw1, rows, hidden), &[_]c_int{ci(rows)}, 1, s));
                        lg = try g.adopt(l1);
                    }
                    const sb: decode.Kernels3.SharedBank = .{
                        .gate_t = sh.gate_t,
                        .gate_suh = sh.gate_suh,
                        .gate_svh = sh.gate_svh,
                        .up_t = sh.up_t,
                        .up_suh = sh.up_suh,
                        .up_svh = sh.up_svh,
                        .down_t = sh.down_t,
                        .down_suh = sh.down_suh,
                        .down_svh = sh.down_svh,
                        .n_gu = n_gu,
                        .n_dn = n_dn,
                        .ks_gu = dense.Dense.ksFor(hidden, inter),
                        .ks_dn = dense.Dense.ksFor(inter, hidden),
                        .defer_affine = dense.Dense.deferFor(.mul1),
                        .sigtab = sigtab,
                        .logit = lg,
                    };
                    // the chain: the router kernel, then moeShared on its slots and scores
                    const ro = try routerTopK(router, s, logits, rows, ne, topk, .bfloat16);
                    const inds = try g.adopt(ro[0]);
                    const scores = try g.adopt(ro[1]);
                    var ind1 = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_reshape(&ind1, inds, &[_]c_int{ci(rows * topk)}, 1, s));
                    const slots = try g.adopt(ind1);
                    var s1 = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_reshape(&s1, scores, &[_]c_int{ci(rows * topk)}, 1, s));
                    const scores1 = try g.adopt(s1);
                    // (moeShared without the fold: gu3s + fin7s, so the folded launch below is held to the unfolded chain)
                    k3.fused_fin_fold = false;
                    const want = try g.adopt(try k3.moeShared(s, x, slots, scores1, arrays, rows, sb));
                    k3.fused_fin_fold = true;
                    // the merged launch in prep2's place; one row: gu8s prepares inside the gate|up launch, so it refuses
                    const got_e = k3.moeSharedRouted(s, x, logits, arrays, rows, sb, if (gate and rows >= 2) gvec else null);
                    if (rows == 1) {
                        const refused = if (got_e) |y| blk: {
                            _ = try g.adopt(y);
                            break :blk false;
                        } else |e| e == error.Exl3EngineNotImplemented;
                        checks += 1;
                        std.debug.print("TURBOEXL3_ROUTER_BYTES part=refuse_gu8 experts={d} rows=1 refused={} {s}\n", .{ ne, refused, if (refused) "PASS" else "FAIL" });
                        if (!refused) failed = true;
                        continue;
                    }
                    const got = try g.adopt(try got_e);
                    // and the merged launch with the fold off (gu3s + fin7s)
                    k3.fused_fin_fold = false;
                    const got_nf = try g.adopt(try k3.moeSharedRouted(s, x, logits, arrays, rows, sb, if (gate and rows >= 2) gvec else null));
                    k3.fused_fin_fold = true;
                    for ([_]mlx.mlx_array{ got, got_nf }, [_][]const u8{ "folded", "unfolded" }) |gy, fname| {
                        const d = try routerDiff(&g, gy, want);
                        checks += 1;
                        std.debug.print("TURBOEXL3_ROUTER_BYTES part=moe experts={d} n={d} routed_defer={} rows={d} gate={s} fold={s} differing={d}/{d} {s}\n", .{ ne, n, defer_affine, rows, if (gate) "launch" else "given", fname, d.diff, d.n, if (d.diff == 0) "PASS" else "FAIL" });
                        if (d.diff != 0) failed = true;
                    }
                };
            }
        }
    }
    std.debug.print("TURBOEXL3_ROUTER_BYTES_DONE checks={d} {s}\n", .{ checks, if (failed) "FAIL" else "PASS" });
    try testing.expect(!failed);
}

// ---- The resident trunk: each trunk linear's W_hat kept (dense.zig `residentArm`) ----
//
// Every trunk linear kind of Qwen3.8-Flash-Next's 5.05bpw pack (shape and rate) through `root.linear` at the prefill row
// counts (33, the first that dequantizes, up to 8703), each against the per-call path (`dense.resident_bypass`, the table
// still armed) BIT FOR BIT, on the call that keeps the W_hat or reuses it and on a second call that reuses it; the row
// counts run in order on one kept W_hat, so every one after the first reads what another row count kept. Also
// `root.linearSplit` across two streams at 1359 and 8192 rows; ATTN-IN and ATTN-OUT (attn_io) at 33, 1359 and 8192
// rows; and decode / register-path rows (1, 4, 16; 17, 32) untouched: equal bytes, nothing kept or reused.
// Lines: TURBOEXL3_TRUNKRES (differing elements / total, PASS when 0).

const trunkres_bf16 = support.bf16Bits;

/// A trunk linear kind of the pack: its shape and rate (kinds that share both are one entry).
const TrunkresKind = struct { name: []const u8, in: u32, out: u32, n: u32 };
const trunkres_kinds = [_]TrunkresKind{
    .{ .name = "gdn.in_proj_qkv", .in = 2560, .out = 10240, .n = 112 },
    .{ .name = "gdn.in_proj_z", .in = 2560, .out = 6144, .n = 112 },
    .{ .name = "gdn.out_proj+attn.o_proj", .in = 6144, .out = 2560, .n = 112 },
    .{ .name = "attn.q_proj", .in = 2560, .out = 12288, .n = 112 },
    .{ .name = "attn.k_proj+v_proj", .in = 2560, .out = 512, .n = 112 },
    .{ .name = "attn.indexer", .in = 2560, .out = 640, .n = 80 },
    .{ .name = "shared.gate+up", .in = 2560, .out = 640, .n = 112 },
    .{ .name = "shared.down", .in = 640, .out = 2560, .n = 112 },
    .{ .name = "mtp.fc_embedding+fc_hidden", .in = 2560, .out = 2560, .n = 96 },
    .{ .name = "mtp.q_proj", .in = 2560, .out = 12288, .n = 96 },
    .{ .name = "mtp.k_proj+v_proj", .in = 2560, .out = 512, .n = 96 },
    .{ .name = "mtp.o_proj", .in = 6144, .out = 2560, .n = 96 },
    .{ .name = "mtp.indexer+shared.gate+up", .in = 2560, .out = 640, .n = 96 },
    .{ .name = "mtp.shared.down", .in = 640, .out = 2560, .n = 96 },
};
const trunkres_rows = [_]u32{ 33, 100, 1000, 1359, 4096, 8192, 8703 };

/// A dense EXL3 projection in -> out at rate n as the host holds it: trellis U16 [in/16, out/16, n] (every pattern a valid
/// codeword), suh F16 [in] in [0.5, 1.5), svh F16 [out] in [0.5, 1.5) / sqrt(in) (outputs O(1)); evaluated.
fn trunkresProj(a: std.mem.Allocator, g: *G, rnd: std.Random, in: u32, out: u32, n: u32) !root.Proj {
    const words = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(words);
    for (words) |*w| w.* = rnd.int(u16);
    const suh = try a.alloc(u16, in);
    defer a.free(suh);
    for (suh) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const svh = try a.alloc(u16, out);
    defer a.free(svh);
    const sc: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(in)));
    for (svh) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * sc);
    const p: root.Proj = .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(words), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(suh), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(svh), &.{ci(out)}, .float16),
    };
    try g.evalAll(&.{ p.trellis, p.suh, p.svh });
    return p;
}

fn trunkresBytes(p: root.Proj) u64 {
    const sh = mlx.getShape(p.trellis);
    return @as(u64, @intCast(sh[0])) * 16 * @as(u64, @intCast(sh[1])) * 16 * 2;
}

/// A bf16 array of `shape` with bf16-exact values lo + span * N(0,1) (lo + span * U[0,1) when `uniform`).
fn trunkresRand(a: std.mem.Allocator, g: *G, rnd: std.Random, shape: []const c_int, lo: f32, span: f32, uniform: bool) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try a.alloc(f32, n);
    defer a.free(buf);
    for (buf) |*v| {
        const r = if (uniform) rnd.float(f32) else rnd.floatNorm(f32);
        v.* = @bitCast(@as(u32, trunkres_bf16(lo + span * r)) << 16);
    }
    return g.astype(try g.hostArray(std.mem.sliceAsBytes(buf), shape, .float32), .bfloat16);
}

/// x bf16 [1, rows, in]: the shared host buffer's first rows * in values (bf16-exact).
fn trunkresX(g: *G, xbuf: []const f32, rows: u32, in: u32) !mlx.mlx_array {
    const n = @as(usize, rows) * in;
    return g.astype(try g.hostArray(std.mem.sliceAsBytes(xbuf[0..n]), &.{ 1, ci(rows), ci(in) }, .float32), .bfloat16);
}

/// The elements of `got` whose bits differ from `want`'s (maxInt on a shape or dtype mismatch): both made contiguous,
/// evaluated and compared in place in unified memory.
fn trunkresDiff(g: *G, s: mlx.mlx_stream, want: mlx.mlx_array, got: mlx.mlx_array) !usize {
    if (!std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got)) or mlx.mlx_array_dtype(want) != mlx.mlx_array_dtype(got)) return std.math.maxInt(usize);
    var cw = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&cw, want, false, s));
    const w = try g.adopt(cw);
    var cg = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&cg, got, false, s));
    const o = try g.adopt(cg);
    try g.evalAll(&.{ w, o });
    const item = mlx.mlx_array_itemsize(w);
    const n = mlx.mlx_array_size(w);
    const pw = (mlx.mlx_array_data_uint8(w) orelse return error.MlxNoData)[0 .. n * item];
    const po = (mlx.mlx_array_data_uint8(o) orelse return error.MlxNoData)[0 .. n * item];
    if (std.mem.eql(u8, pw, po)) return 0;
    var diff: usize = 0;
    for (0..n) |i| {
        if (!std.mem.eql(u8, pw[i * item ..][0..item], po[i * item ..][0..item])) diff += 1;
    }
    return diff;
}

/// MLX's sigmoid of every bf16 pattern (the host's `swigluSigTable`): the gate table ATTN-OUT reads.
fn trunkresSigtab(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream) !mlx.mlx_array {
    const pats = try a.alloc(f32, 1 << 16);
    defer a.free(pats);
    for (pats, 0..) |*v, i| v.* = @bitCast(@as(u32, @intCast(i)) << 16);
    const vals = try g.astype(try g.hostArray(std.mem.sliceAsBytes(pats), &.{1 << 16}, .float32), .bfloat16);
    var t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&t, vals, s));
    const tab = try g.adopt(t);
    try g.evalAll(&.{tab});
    return tab;
}

fn trunkresPath(rows: u32) []const u8 {
    return if (rows <= 16) "gemv" else if (rows <= 32) "regpath" else "dequant+matmul";
}

test "turboexl3 gpu resident trunk: a resident W_hat gives the per-call dequantize's bytes on every trunk linear kind, both streams, decode rows untouched (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const dense = @import("dense.zig");
    defer dense.residentRelease();
    defer dense.resident_bypass = false;
    var dev = mlx.mlx_device{ .ctx = null };
    try testing.expectEqual(@as(c_int, 0), mlx.mlx_stream_get_device(&dev, s));
    defer _ = mlx.mlx_device_free(dev);
    const side = mlx.mlx_stream_new_device(dev);
    try testing.expect(side.ctx != null);
    defer _ = mlx.mlx_stream_free(side);
    var prng = std.Random.DefaultPrng.init(0x7E5_1DE7);
    const rnd = prng.random();
    // one host buffer of bf16-exact values for every x (the widest call of the widest input)
    const xbuf = try a.alloc(f32, @as(usize, trunkres_rows[trunkres_rows.len - 1]) * 6144);
    defer a.free(xbuf);
    for (xbuf) |*v| v.* = @bitCast(@as(u32, trunkres_bf16(rnd.floatNorm(f32))) << 16);
    var failed = false;
    for (trunkres_kinds) |k| {
        const top = g.mark();
        defer g.resetTo(top);
        const p = try trunkresProj(a, &g, rnd, k.in, k.out, k.n);
        const bytes = trunkresBytes(p);
        try dense.residentArm(&.{p.trellis}, &.{bytes});
        // decode and register-path rows: the table is never asked (nothing kept or reused), the bytes the per-call path's
        for ([_]u32{ 1, 4, 16, 17, 32 }) |rows| {
            const m = g.mark();
            defer g.resetTo(m);
            const x = try trunkresX(&g, xbuf, rows, k.in);
            dense.resident_bypass = true;
            const want = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
            try g.evalAll(&.{want});
            dense.resident_bypass = false;
            const got = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
            const d = try trunkresDiff(&g, s, want, got);
            const st = dense.residentStats();
            const ok = d == 0 and st.kept == 0 and st.reuses == 0 and dense.residentPending() == bytes;
            std.debug.print("TURBOEXL3_TRUNKRES what=linear kind={s} n={d} rows={d} path={s} differing={d}/{d} kept={d} reuses={d} {s}\n", .{ k.name, k.n, rows, trunkresPath(rows), d, rows * k.out, st.kept, st.reuses, if (ok) "PASS" else "FAIL" });
            if (!ok) failed = true;
        }
        // prefill rows, in order on one kept W_hat: the per-call reference, then two resident calls (the first row count's
        // first call keeps it, every other call reuses it), and at 1359 / 8192 rows the two-stream split
        var prev_reuses: u64 = 0;
        for (trunkres_rows, 0..) |rows, ri| {
            const m = g.mark();
            defer g.resetTo(m);
            const x = try trunkresX(&g, xbuf, rows, k.in);
            dense.resident_bypass = true;
            const want = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
            try g.evalAll(&.{want});
            dense.resident_bypass = false;
            const got1 = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
            const got2 = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
            const d1 = try trunkresDiff(&g, s, want, got1);
            const d2 = try trunkresDiff(&g, s, want, got2);
            const split = rows == 1359 or rows == 8192;
            var d3: usize = 0;
            if (split) d3 = try trunkresDiff(&g, s, want, try g.adopt(try root.linearSplit(s, side, x, p.trellis, p.suh, p.svh, .mul1)));
            const st = dense.residentStats();
            const reused: u64 = (if (ri == 0) @as(u64, 1) else 2) + @intFromBool(split);
            const ok = d1 == 0 and d2 == 0 and d3 == 0 and st.kept == 1 and st.held == bytes and st.reuses - prev_reuses == reused and
                dense.residentPending() == 0 and dense.residentKeeps(p.trellis);
            prev_reuses = st.reuses;
            std.debug.print("TURBOEXL3_TRUNKRES what=linear kind={s} n={d} rows={d} path={s} {s}={d}/{d} reuse={d}/{d} split={s}{d} kept={d} held={d} reuses={d} {s}\n", .{
                k.name, k.n, rows, trunkresPath(rows), if (ri == 0) "keep" else "reuse", d1, rows * k.out, d2, rows * k.out, if (split) "" else "n/a:", d3, st.kept, st.held, st.reuses, if (ok) "PASS" else "FAIL",
            });
            if (!ok) failed = true;
        }
        dense.residentRelease();
        const st = dense.residentStats();
        if (st.modules != 0 or st.kept != 0 or st.held != 0 or dense.residentPending() != 0 or dense.residentKeeps(p.trellis)) {
            std.debug.print("TURBOEXL3_TRUNKRES what=release kind={s} modules={d} kept={d} held={d} FAIL\n", .{ k.name, st.modules, st.kept, st.held });
            failed = true;
        }
    }
    try testing.expect(!failed);
}

test "turboexl3 gpu resident trunk: ATTN-IN and ATTN-OUT over resident W_hat give the per-call path's bytes (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const dense = @import("dense.zig");
    defer dense.residentRelease();
    defer dense.resident_bypass = false;
    const top = g.mark();
    defer g.resetTo(top);
    var prng = std.Random.DefaultPrng.init(0x7E5_A77E);
    const rnd = prng.random();
    // Qwen3.8-Flash-Next's attention (24 query heads [query | gate], 2 kv heads, head 256, RoPE on 64), K7
    const hidden: u32 = 2560;
    const hq: u32 = 24;
    const hk: u32 = 2;
    const hd: u32 = 256;
    const q = try trunkresProj(a, &g, rnd, hidden, 2 * hq * hd, 112);
    const k = try trunkresProj(a, &g, rnd, hidden, hk * hd, 112);
    const v = try trunkresProj(a, &g, rnd, hidden, hk * hd, 112);
    const o = try trunkresProj(a, &g, rnd, hq * hd, hidden, 112);
    const projs = [_]root.Proj{ q, k, v, o };
    var trellises: [projs.len]mlx.mlx_array = undefined;
    var bytes: [projs.len]u64 = undefined;
    var total: u64 = 0;
    for (projs, 0..) |p, i| {
        trellises[i] = p.trellis;
        bytes[i] = trunkresBytes(p);
        total += bytes[i];
    }
    try dense.residentArm(&trellises, &bytes);
    const sigtab = try trunkresSigtab(a, &g, s);
    const qw = try trunkresRand(a, &g, rnd, &.{ci(hd)}, 0.5, 1.0, true);
    const kw = try trunkresRand(a, &g, rnd, &.{ci(hd)}, 0.5, 1.0, true);
    const eps = mlx.mlx_array_new_float(1.0e-6);
    defer _ = mlx.mlx_array_free(eps);
    var failed = false;
    for ([_]u32{ 33, 1359, 8192 }) |rows| {
        const m = g.mark();
        defer g.resetTo(m);
        const x = try trunkresRand(a, &g, rnd, &.{ 1, ci(rows), ci(hidden) }, 0.0, 1.0, false);
        // any f32 cos | sin rows: both arms read the same table
        const ang = try a.alloc(f32, @as(usize, rows) * 64);
        defer a.free(ang);
        for (ang) |*e| e.* = 2.0 * rnd.float(f32) - 1.0;
        const angles = try g.hostArray(std.mem.sliceAsBytes(ang), &.{ci(rows * 64)}, .float32);
        const ep: root.attn_io.Epilogue = .{ .heads = hq, .kv_heads = hk, .rope_dims = 64, .q_norm = qw, .k_norm = kw, .eps = eps, .angles = angles };
        // ATTN-IN: q|gate, k, v (one prepare, the three inner products, one finish)
        dense.resident_bypass = true;
        var want = try root.attnIn(s, x, q, k, v, ep, .mul1);
        defer want.deinit();
        try g.evalAll(&.{ want.q, want.gate, want.k, want.v });
        dense.resident_bypass = false;
        for (0..2) |rep| {
            var got = try root.attnIn(s, x, q, k, v, ep, .mul1);
            defer got.deinit();
            const dq = try trunkresDiff(&g, s, want.q, got.q);
            const dg = try trunkresDiff(&g, s, want.gate, got.gate);
            const dk = try trunkresDiff(&g, s, want.k, got.k);
            const dv = try trunkresDiff(&g, s, want.v, got.v);
            const ok = dq == 0 and dg == 0 and dk == 0 and dv == 0;
            std.debug.print("TURBOEXL3_TRUNKRES what=attn_in rows={d} call={d} q={d} gate={d} k={d} v={d} {s}\n", .{ rows, rep, dq, dg, dk, dv, if (ok) "PASS" else "FAIL" });
            if (!ok) failed = true;
        }
        // ATTN-OUT: o_proj over the gated prepare, the gate read in place from ATTN-IN's
        const attn = try trunkresRand(a, &g, rnd, &.{ 1, ci(hq), ci(rows), ci(hd) }, 0.0, 1.0, false);
        const gate: root.attn_io.Gate = .{ .base = want.gate, .row_stride = hq * hd, .head_stride = hd, .off = 0 };
        dense.resident_bypass = true;
        const yo_want = try g.adopt(try root.linearGated(s, attn, gate, sigtab, o, .mul1));
        try g.evalAll(&.{yo_want});
        dense.resident_bypass = false;
        for (0..2) |rep| {
            const d = try trunkresDiff(&g, s, yo_want, try g.adopt(try root.linearGated(s, attn, gate, sigtab, o, .mul1)));
            std.debug.print("TURBOEXL3_TRUNKRES what=attn_out rows={d} call={d} y={d}/{d} {s}\n", .{ rows, rep, d, rows * hidden, if (d == 0) "PASS" else "FAIL" });
            if (d != 0) failed = true;
        }
    }
    const st = dense.residentStats();
    const ok = st.modules == projs.len and st.kept == projs.len and st.held == total and dense.residentPending() == 0;
    std.debug.print("TURBOEXL3_TRUNKRES what=entries modules={d} kept={d} held={d}/{d} reuses={d} {s}\n", .{ st.modules, st.kept, st.held, total, st.reuses, if (ok) "PASS" else "FAIL" });
    try testing.expect(ok and !failed);
}

// ---- The un-rotated trunk: W_std out of the Hadamard basis (dense.zig's module comment) ----
//
// On `trunkres_kinds` (every trunk linear kind of the 5.05bpw pack: shape and rate):
//   unrot   `root.trunkUnrot` on an armed table, each kind's W_std against the f64 reference on three sampled 128 x 128 tiles
//           (the first, a middle and the last: every element within one bf16 step or, where its 128 x 128 terms cancel to
//           near zero, half a step of its own plus 16 f32 eps of its terms' absolute sum, `dense.unrotElemOk`; the count
//           past one step reported; at most 1 in 1000 off the reference's own bf16 rounding); the bill unchanged (held =
//           the W_hat bytes, nothing pending) and MLX's active bytes with the W_std resident against the W_hat resident
//   linear  `root.linear` at decode / register-path rows (1, 4, 16, 17, 32: the chain's bytes, the table never asked) and at
//           prefill rows (33, 64, 97, 1328, 2100, 4096, 8192: the plain matmul against the rotated chain within the bars),
//           `root.linearSplit` at 1328 / 8192 rows (the plain matmul on the side stream: `linear`'s bytes)
//   joins   GDN qkv | z, QSA q | k | v | indexer, the shared gate | up: one group, consecutive columns, every member a
//           zero-copy view of the joined W_std (its row stride, its data at the member's column); the joined product's
//           columns against each member's own plain matmul; ATTN-IN's standard-basis epilogue over the joined product
//           against `attnIn` (rotated) and over the members' own products; ATTN-OUT's against `linearGated`
// Bars (x and W_std in bf16, one rounding of each weight and of each output against the chain's f16 roundings of x's rotated
// copy, the inner product and the finish): rel RMS(plain - chain) <= 2^-7.5 and max |plain - chain| <= 2 bf16 steps of
// max|chain|'s binade (the expected values: ~2^-8.5 and under one step); the epilogues' outputs the same bars. Lines:
// TURBOEXL3_TRUNKUNROT ... PASS.

const unrot_bar_rms: f64 = 0x1p-7 / std.math.sqrt2;
const unrot_bar_steps: f64 = 2.0;

/// `x` (any float dtype) as host f32, contiguous.
fn unrotHost(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, x: mlx.mlx_array) ![]f32 {
    var c = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&c, x, false, s));
    const f = try g.astype(try g.adopt(c), .float32);
    try g.evalAll(&.{f});
    const n = mlx.mlx_array_size(f);
    const p = mlx.mlx_array_data_float32(f) orelse return error.MlxNoData;
    const out = try a.alloc(f32, n);
    @memcpy(out, p[0..n]);
    return out;
}

/// How `got` departs from `want`: rel RMS, max |diff|, max |want|, whether every value is equal, and the bars' verdict.
const UnrotCmp = struct {
    rel_rms: f64 = 0,
    max_abs: f64 = 0,
    max_want: f64 = 0,
    identical: bool = true,
    finite: bool = true,

    fn of(want: []const f32, got: []const f32) UnrotCmp {
        var c: UnrotCmp = .{};
        if (want.len != got.len) return .{ .rel_rms = std.math.inf(f64), .identical = false };
        var se: f64 = 0;
        var ss: f64 = 0;
        for (want, got) |w, o| {
            if (!std.math.isFinite(o)) c.finite = false;
            const d = @as(f64, o) - @as(f64, w);
            se += d * d;
            ss += @as(f64, w) * @as(f64, w);
            c.max_abs = @max(c.max_abs, @abs(d));
            c.max_want = @max(c.max_want, @abs(@as(f64, w)));
            if (@as(u32, @bitCast(w)) != @as(u32, @bitCast(o))) c.identical = false;
        }
        c.rel_rms = if (ss > 0) @sqrt(se / ss) else @sqrt(se);
        return c;
    }

    fn ok(c: UnrotCmp) bool {
        return c.finite and c.rel_rms <= unrot_bar_rms and c.max_abs <= unrot_bar_steps * @import("dense.zig").bf16Step(c.max_want);
    }
};

/// Arm the table with `projs` and un-rotate them as one group (`join`) or each alone.
fn unrotArm(s: mlx.mlx_stream, projs: []const root.Proj, join: bool) !root.UnrotStats {
    const dense = @import("dense.zig");
    var trellises: [8]mlx.mlx_array = undefined;
    var bytes: [8]u64 = undefined;
    for (projs, 0..) |p, i| {
        trellises[i] = p.trellis;
        bytes[i] = trunkresBytes(p);
    }
    try dense.residentArm(trellises[0..projs.len], bytes[0..projs.len]);
    var sizes: [8]u8 = @splat(1);
    const ng: usize = if (join) 1 else projs.len;
    if (join) sizes[0] = @intCast(projs.len);
    return root.trunkUnrot(s, projs, sizes[0..ng], .mul1, .bfloat16, null);
}

test "turboexl3 gpu un-rotated trunk: every trunk linear kind's un-rotation equals the f64 reference on sampled tiles, in the W_hat's place in the table (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const dense = @import("dense.zig");
    defer dense.residentRelease();
    var prng = std.Random.DefaultPrng.init(0x7_2B0A);
    const rnd = prng.random();
    var failed = false;
    for (trunkres_kinds) |k| {
        const top = g.mark();
        defer g.resetTo(top);
        const p = try trunkresProj(a, &g, rnd, k.in, k.out, k.n);
        const bytes = trunkresBytes(p);
        // the W_hat the chain multiplies, and its suh / svh, on the host
        var d = try dense.Dense.init(.{ .in = k.in, .out = k.out, .rate = format.Rate.fromN(k.n).?, .dec = .mul1, .out_dtype = .bfloat16 });
        defer d.deinit();
        const w_hat = try unrotHost(a, &g, s, try g.adopt(try d.decodeTrellis(s, p.trellis)));
        defer a.free(w_hat);
        const suh = try unrotHost(a, &g, s, p.suh);
        defer a.free(suh);
        const svh = try unrotHost(a, &g, s, p.svh);
        defer a.free(svh);
        // MLX's active bytes with the kind's W_hat resident (kept by its first dequantize), then with its W_std (the table
        // rearmed in between)
        try dense.residentArm(&.{p.trellis}, &.{bytes});
        _ = mlx.mlx_array_free(try d.dequantize(s, p.trellis));
        if (!dense.residentKeeps(p.trellis)) return error.TestUnexpectedResult;
        _ = mlx.mlx_clear_cache();
        var act_hat: usize = 0;
        _ = mlx.mlx_get_active_memory(&act_hat);
        dense.residentRelease();
        const st = try unrotArm(s, &.{p}, false);
        _ = mlx.mlx_clear_cache();
        var act_std: usize = 0;
        _ = mlx.mlx_get_active_memory(&act_std);
        const rs = dense.residentStats();
        const bill_ok = st.groups == 1 and st.modules == 1 and st.bytes == bytes and rs.unrot == 1 and rs.kept == 1 and rs.held == bytes and dense.residentPending() == 0 and dense.residentUnrot(p.trellis) and !dense.residentKeeps(p.trellis);
        const w = dense.residentStd(p.trellis) orelse return error.TestUnexpectedResult;
        const w_std = try unrotHost(a, &g, s, try g.adopt(w));
        defer a.free(w_std);
        const nrb = k.in / 128;
        const ncb = k.out / 128;
        const tiles = [3][2]usize{ .{ 0, 0 }, .{ nrb / 2, ncb / 2 }, .{ nrb - 1, ncb - 1 } };
        const ref_tile = try a.create([128 * 128]f64);
        defer a.destroy(ref_tile);
        // each element against `dense.unrotElemOk`: one bf16 step of the reference, or (an element whose 128 x 128 terms
        // cancel to near zero) half a step of its own plus 16 f32 eps of its terms' absolute sum; the elements past one step
        // and the worst of them in eps x terms are reported, not a verdict
        var worst_steps: f64 = 0;
        var worst_terms: f64 = 0;
        var over_step: usize = 0;
        var failing: usize = 0;
        var off_round: usize = 0;
        for (tiles) |tc| {
            dense.refUnrotTile(w_hat, suh, svh, k.out, tc[0], tc[1], ref_tile);
            // every element of the tile is a signed sum of the tile's W_hat values times its suh * svh / 128
            const tile_abs = dense.unrotTileAbsSum(w_hat, k.out, tc[0], tc[1]);
            for (0..128) |i| for (0..128) |o| {
                const gi = tc[0] * 128 + i;
                const go = tc[1] * 128 + o;
                const r = ref_tile[i * 128 + o];
                const got: f64 = w_std[gi * k.out + go];
                const dist = @abs(got - r);
                const terms = @abs(@as(f64, suh[gi]) * @as(f64, svh[go])) / 128.0 * tile_abs;
                worst_steps = @max(worst_steps, dist / dense.bf16Step(r));
                if (dist > dense.bf16Step(r)) {
                    over_step += 1;
                    worst_terms = @max(worst_terms, dist / (0x1p-23 * terms));
                }
                if (!dense.unrotElemOk(got, r, terms)) failing += 1;
                const rb: f32 = @floatCast(r);
                const rr: u32 = @bitCast(rb);
                const r16: f32 = @bitCast((rr +% (0x7FFF + ((rr >> 16) & 1))) & 0xFFFF_0000);
                if (got != r16) off_round += 1;
            };
        }
        const ok = bill_ok and failing == 0 and off_round * 1000 <= 3 * 128 * 128;
        std.debug.print("TURBOEXL3_TRUNKUNROT what=unrot kind={s} in={d} out={d} n={d} failing={d} (bar: 1 bf16 step, or 0.5 step + {d:.0} eps x terms) over_1_step={d}/{d} worst_bf16_steps={d:.3} worst_over_step_eps_x_terms={d:.2} (info) off_round={d}/{d} held={d} bytes={d} active_hat={d} active_std={d} delta={d} {s}\n", .{ k.name, k.in, k.out, k.n, failing, dense.unrot_terms_eps, over_step, 3 * 128 * 128, worst_steps, worst_terms, off_round, 3 * 128 * 128, rs.held, bytes, act_hat, act_std, @as(i64, @intCast(act_std)) - @as(i64, @intCast(act_hat)), if (ok) "PASS" else "FAIL" });
        if (!ok) failed = true;
        dense.residentRelease();
    }
    try testing.expect(!failed);
}

test "turboexl3 gpu un-rotated trunk: root.linear on an un-rotated linear is one plain matmul within the bars at prefill rows and the chain's bytes at decode and register-path rows (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const dense = @import("dense.zig");
    defer dense.residentRelease();
    var dev = mlx.mlx_device{ .ctx = null };
    try testing.expectEqual(@as(c_int, 0), mlx.mlx_stream_get_device(&dev, s));
    defer _ = mlx.mlx_device_free(dev);
    const side = mlx.mlx_stream_new_device(dev);
    try testing.expect(side.ctx != null);
    defer _ = mlx.mlx_stream_free(side);
    var prng = std.Random.DefaultPrng.init(0x7_2B0B);
    const rnd = prng.random();
    const row_set = [_]u32{ 1, 4, 16, 17, 32, 33, 64, 97, 1328, 2100, 4096, 8192 };
    const xbuf = try a.alloc(f32, @as(usize, row_set[row_set.len - 1]) * 6144);
    defer a.free(xbuf);
    for (xbuf) |*v| v.* = @bitCast(@as(u32, trunkres_bf16(rnd.floatNorm(f32))) << 16);
    var failed = false;
    for (trunkres_kinds) |k| {
        const top = g.mark();
        defer g.resetTo(top);
        const p = try trunkresProj(a, &g, rnd, k.in, k.out, k.n);
        for (row_set) |rows| {
            const m = g.mark();
            defer g.resetTo(m);
            const x = try trunkresX(&g, xbuf, rows, k.in);
            // the chain: the table disarmed (the per-call path)
            dense.residentRelease();
            const want = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
            try g.evalAll(&.{want});
            _ = try unrotArm(s, &.{p}, false);
            const plain0 = dense.residentStats().plain;
            const got = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
            try g.evalAll(&.{got});
            const plain = dense.residentStats().plain - plain0;
            const hw = try unrotHost(a, &g, s, want);
            defer a.free(hw);
            const hg = try unrotHost(a, &g, s, got);
            defer a.free(hg);
            const c = UnrotCmp.of(hw, hg);
            var ok: bool = undefined;
            var split_same = true;
            if (rows <= dense.unrot_min_rows) {
                // the trellis paths: the chain's bytes, the plain arm never taken
                ok = c.identical and plain == 0 and mlx.mlx_array_dtype(got) == .bfloat16;
            } else {
                ok = c.ok() and plain == 1 and mlx.mlx_array_dtype(got) == .bfloat16;
                if (rows == 1328 or rows == 8192) {
                    const gs = try g.adopt(try root.linearSplit(s, side, x, p.trellis, p.suh, p.svh, .mul1));
                    split_same = try trunkresDiff(&g, s, got, gs) == 0;
                    ok = ok and split_same;
                }
            }
            std.debug.print("TURBOEXL3_TRUNKUNROT what=linear kind={s} n={d} rows={d} path={s} rel_rms={e:.3} max_abs={e:.3} max_chain={e:.3} bf16_step={e:.3} identical={} plain_calls={d} split_same={} {s}\n", .{ k.name, k.n, rows, if (rows <= dense.unrot_min_rows) trunkresPath(rows) else "plain", c.rel_rms, c.max_abs, c.max_want, dense.bf16Step(c.max_want), c.identical, plain, split_same, if (ok) "PASS" else "FAIL" });
            if (!ok) failed = true;
        }
        dense.residentRelease();
    }
    try testing.expect(!failed);
}

test "turboexl3 gpu un-rotated trunk: a join is one buffer whose members are zero-copy views, its product's columns are the members' plain matmuls, and ATTN-IN / ATTN-OUT in the standard basis are the rotated launches' within the bars (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const dense = @import("dense.zig");
    defer dense.residentRelease();
    var prng = std.Random.DefaultPrng.init(0x7_2B0C);
    const rnd = prng.random();
    const hidden: u32 = 2560;
    const hq: u32 = 24;
    const hk: u32 = 2;
    const hd: u32 = 256;
    const top = g.mark();
    defer g.resetTo(top);
    // the served joins at the pack's rates (the indexer at K5)
    const gdn = [_]root.Proj{ try trunkresProj(a, &g, rnd, hidden, 10240, 112), try trunkresProj(a, &g, rnd, hidden, 6144, 112) };
    const qsa = [_]root.Proj{ try trunkresProj(a, &g, rnd, hidden, 2 * hq * hd, 112), try trunkresProj(a, &g, rnd, hidden, hk * hd, 112), try trunkresProj(a, &g, rnd, hidden, hk * hd, 112), try trunkresProj(a, &g, rnd, hidden, 640, 80) };
    const shared = [_]root.Proj{ try trunkresProj(a, &g, rnd, hidden, 640, 112), try trunkresProj(a, &g, rnd, hidden, 640, 112) };
    const o = try trunkresProj(a, &g, rnd, hq * hd, hidden, 112);
    const Join = struct { name: []const u8, projs: []const root.Proj };
    const joins = [_]Join{ .{ .name = "gdn_qkv_z", .projs = &gdn }, .{ .name = "qsa_q_k_v_idx", .projs = &qsa }, .{ .name = "shared_gate_up", .projs = &shared } };
    const sigtab = try trunkresSigtab(a, &g, s);
    const qw = try trunkresRand(a, &g, rnd, &.{ci(hd)}, 0.5, 1.0, true);
    const kw = try trunkresRand(a, &g, rnd, &.{ci(hd)}, 0.5, 1.0, true);
    const eps = mlx.mlx_array_new_float(1.0e-6);
    defer _ = mlx.mlx_array_free(eps);
    var failed = false;
    for (joins) |jn| {
        // the rotated chains first (the table disarmed), then the join
        dense.residentRelease();
        var want_by_rows: [4][dense.max_join][]f32 = undefined;
        const rows_set = [_]u32{ 33, 1328, 4096, 8192 };
        const xs: [rows_set.len]mlx.mlx_array = .{
            try trunkresRand(a, &g, rnd, &.{ 1, 33, ci(hidden) }, 0.0, 1.0, false),
            try trunkresRand(a, &g, rnd, &.{ 1, 1328, ci(hidden) }, 0.0, 1.0, false),
            try trunkresRand(a, &g, rnd, &.{ 1, 4096, ci(hidden) }, 0.0, 1.0, false),
            try trunkresRand(a, &g, rnd, &.{ 1, 8192, ci(hidden) }, 0.0, 1.0, false),
        };
        for (xs, 0..) |x, ri| for (jn.projs, 0..) |p, mi| {
            want_by_rows[ri][mi] = try unrotHost(a, &g, s, try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1)));
        };
        defer for (0..rows_set.len) |ri| for (0..jn.projs.len) |mi| a.free(want_by_rows[ri][mi]);
        const st = try unrotArm(s, jn.projs, true);
        var layout_ok = st.groups == 1 and st.joined == 1 and st.modules == jn.projs.len;
        // one group, consecutive columns, each member a view of the joined buffer at its column
        var col: u32 = 0;
        var group: u32 = 0;
        var jw: mlx.mlx_array = .{ .ctx = null };
        for (jn.projs, 0..) |p, mi| {
            const j = dense.residentJoined(p.trellis) orelse {
                layout_ok = false;
                break;
            };
            if (mi == 0) {
                group = j.group;
                jw = try g.adopt(j.w);
            } else _ = mlx.mlx_array_free(j.w);
            const v = try g.adopt(dense.residentStd(p.trellis) orelse return error.TestUnexpectedResult);
            const vs = mlx.mlx_array_strides(v);
            const jbase = @intFromPtr(mlx.mlx_array_data_uint8(jw) orelse return error.MlxNoData);
            const vbase = @intFromPtr(mlx.mlx_array_data_uint8(v) orelse return error.MlxNoData);
            const width: u32 = @intCast(mlx.getShape(p.trellis)[1] * 16);
            layout_ok = layout_ok and j.group == group and j.col == col and j.width == width and vs[0] == @as(usize, @intCast(mlx.getShape(jw)[1])) and vs[1] == 1 and vbase - jbase == @as(usize, col) * 2;
            col += width;
        }
        layout_ok = layout_ok and jw.ctx != null and mlx.getShape(jw)[1] == @as(c_int, @intCast(col));
        std.debug.print("TURBOEXL3_TRUNKUNROT what=join_layout join={s} members={d} width={d} {s}\n", .{ jn.name, jn.projs.len, col, if (layout_ok) "PASS" else "FAIL" });
        if (!layout_ok) failed = true;
        if (jw.ctx == null) continue;
        for (xs, rows_set, 0..) |x, rows, ri| {
            const m = g.mark();
            defer g.resetTo(m);
            const yj = try g.matmul(x, jw);
            var c0: c_int = 0;
            for (jn.projs, 0..) |p, mi| {
                const width = mlx.getShape(p.trellis)[1] * 16;
                var sl = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&sl, yj, &[_]c_int{ 0, 0, c0 }, 3, &[_]c_int{ 1, ci(rows), c0 + width }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
                const part = try unrotHost(a, &g, s, try g.adopt(sl));
                defer a.free(part);
                const own = try unrotHost(a, &g, s, try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1)));
                defer a.free(own);
                const vs_own = UnrotCmp.of(own, part);
                const vs_chain = UnrotCmp.of(want_by_rows[ri][mi], part);
                const ok = vs_own.ok() and vs_chain.ok();
                std.debug.print("TURBOEXL3_TRUNKUNROT what=join join={s} member={d} rows={d} joined_vs_own identical={} rel_rms={e:.3} joined_vs_chain rel_rms={e:.3} max_abs={e:.3} (2 steps {e:.3}) {s}\n", .{ jn.name, mi, rows, vs_own.identical, vs_own.rel_rms, vs_chain.rel_rms, vs_chain.max_abs, 2.0 * dense.bf16Step(vs_chain.max_want), if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
                c0 += width;
            }
        }
        // ATTN-IN and ATTN-OUT over the QSA join
        if (jn.projs.len == 4) for (xs, rows_set) |x, rows| {
            const m = g.mark();
            defer g.resetTo(m);
            const ang = try a.alloc(f32, @as(usize, rows) * 64);
            defer a.free(ang);
            for (ang) |*e| e.* = 2.0 * rnd.float(f32) - 1.0;
            const angles = try g.hostArray(std.mem.sliceAsBytes(ang), &.{ci(rows * 64)}, .float32);
            const ep: root.attn_io.Epilogue = .{ .heads = hq, .kv_heads = hk, .rope_dims = 64, .q_norm = qw, .k_norm = kw, .eps = eps, .angles = angles };
            // the rotated launches on a disarmed table, then the join's product and the members' own products
            dense.residentRelease();
            var want = try root.attnIn(s, x, qsa[0], qsa[1], qsa[2], ep, .mul1);
            defer want.deinit();
            const gate: root.attn_io.Gate = .{ .base = want.gate, .row_stride = hq * hd, .head_stride = hd, .off = 0 };
            const attn = try trunkresRand(a, &g, rnd, &.{ 1, ci(hq), ci(rows), ci(hd) }, 0.0, 1.0, false);
            const yo_want = try g.adopt(try root.linearGated(s, attn, gate, sigtab, o, .mul1));
            try g.evalAll(&.{ want.q, want.gate, want.k, want.v, yo_want });
            const projs5 = [5]root.Proj{ qsa[0], qsa[1], qsa[2], qsa[3], o };
            var trs: [5]mlx.mlx_array = undefined;
            var bys: [5]u64 = undefined;
            for (projs5, 0..) |p, i| {
                trs[i] = p.trellis;
                bys[i] = trunkresBytes(p);
            }
            try dense.residentArm(&trs, &bys);
            _ = try root.trunkUnrot(s, &projs5, &.{ 4, 1 }, .mul1, .bfloat16, null);
            const jq = dense.residentJoined(qsa[0].trellis) orelse return error.TestUnexpectedResult;
            const jwq = try g.adopt(jq.w);
            const yj = try g.matmul(x, jwq);
            var got_j = try root.attnInStd(s, .{ yj, yj, yj }, .{ 0, 2 * hq * hd, 2 * hq * hd + hk * hd }, ep);
            defer got_j.deinit();
            const own = [3]mlx.mlx_array{
                try g.adopt(try root.linear(s, x, qsa[0].trellis, qsa[0].suh, qsa[0].svh, .mul1)),
                try g.adopt(try root.linear(s, x, qsa[1].trellis, qsa[1].suh, qsa[1].svh, .mul1)),
                try g.adopt(try root.linear(s, x, qsa[2].trellis, qsa[2].suh, qsa[2].svh, .mul1)),
            };
            var got_s = try root.attnInStd(s, own, .{ 0, 0, 0 }, ep);
            defer got_s.deinit();
            const yo_got = try g.adopt(try root.linearGatedStd(s, attn, gate, sigtab, o));
            const names = [_][]const u8{ "q", "gate", "k", "v" };
            const wants = [_]mlx.mlx_array{ want.q, want.gate, want.k, want.v };
            const gotj = [_]mlx.mlx_array{ got_j.q, got_j.gate, got_j.k, got_j.v };
            const gots = [_]mlx.mlx_array{ got_s.q, got_s.gate, got_s.k, got_s.v };
            for (names, wants, gotj, gots) |nm, w, gj, gsep| {
                const hw = try unrotHost(a, &g, s, w);
                defer a.free(hw);
                const hj = try unrotHost(a, &g, s, gj);
                defer a.free(hj);
                const hs = try unrotHost(a, &g, s, gsep);
                defer a.free(hs);
                const cj = UnrotCmp.of(hw, hj);
                const cs = UnrotCmp.of(hs, hj);
                const ok = cj.ok() and cs.ok() and std.mem.eql(c_int, mlx.getShape(w), mlx.getShape(gj));
                std.debug.print("TURBOEXL3_TRUNKUNROT what=attn_in out={s} rows={d} std_vs_rotated rel_rms={e:.3} max_abs={e:.3} joined_vs_separate identical={} rel_rms={e:.3} {s}\n", .{ nm, rows, cj.rel_rms, cj.max_abs, cs.identical, cs.rel_rms, if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            }
            const how = try unrotHost(a, &g, s, yo_want);
            defer a.free(how);
            const hog = try unrotHost(a, &g, s, yo_got);
            defer a.free(hog);
            const co = UnrotCmp.of(how, hog);
            std.debug.print("TURBOEXL3_TRUNKUNROT what=attn_out rows={d} rel_rms={e:.3} max_abs={e:.3} (2 steps {e:.3}) {s}\n", .{ rows, co.rel_rms, co.max_abs, 2.0 * dense.bf16Step(co.max_want), if (co.ok()) "PASS" else "FAIL" });
            if (!co.ok()) failed = true;
        };
        dense.residentRelease();
    }
    try testing.expect(!failed);
}
