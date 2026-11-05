//! Byte tests of the core's paths no other test pins (TURBOEXL3_GPU=1, else skipped):
//! - pg2 (`pgemm.zig`, `Mode.on`): at every tabled (K, N) and a prefill row count where MLX runs its regular fused kernel,
//!   `root.linear` with the fused matmul and finish gives the words of dequantize + MLX's matmul + the finish (`Mode.off`),
//!   which is pg2's stated relation. Lines: TURBOEXL3_PG2_BYTES.
//! - `turboexl3_dense_finish_pieces` (`root.linearPiecesFinish`): the finish over 2 to 4 row pieces' inner products gives
//!   the one-call finish's words over the pieces' inner products joined (`root.linearFinish`). Lines:
//!   TURBOEXL3_PIECES_BYTES (also: whether the pieces equal `linear` of the joined rows, which holds wherever MLX's matmul
//!   runs one kernel at every piece's row count; reported, not asserted).
//! - the unfused decode chain (`unfused.Steps`, 11 launches, not served): within the decode chain's tolerance (5e-3) of the
//!   host reference (`ref.moeRowDec`) at form 1's cases; its words against form 1's (`decode.Kernels`) reported, not
//!   asserted (the stated relation is the same math, not the same order). Lines: TURBOEXL3_UNFUSED.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("test_support.zig");
const G = support.Graph;
const A = mlx.mlx_array;
const format = @import("format.zig");
const ref = @import("ref.zig");
const decode = @import("decode.zig");
const unfused = @import("unfused.zig");
const pgemm = @import("pgemm.zig");
const root = @import("../root.zig");

const testing = std.testing;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn bf16Exact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
}

fn randProj(a: std.mem.Allocator, g: *G, rnd: std.Random, in: u32, out: u32, n: u32) !root.Proj {
    const t = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(t);
    rnd.bytes(std.mem.sliceAsBytes(t));
    const su = try a.alloc(u16, in);
    defer a.free(su);
    for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, out);
    defer a.free(sv);
    for (sv) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    return .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(t), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(out)}, .float16),
    };
}

fn randX(a: std.mem.Allocator, g: *G, rnd: std.Random, rows: u32, in: u32) !A {
    const xh = try a.alloc(f32, @as(usize, rows) * in);
    defer a.free(xh);
    for (xh) |*v| v.* = bf16Exact(rnd.floatNorm(f32));
    return g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(in) }, .float32), .bfloat16);
}

/// the words of `x` and `y` (any float dtype, read as f32: exact for bf16 and f16) that differ, of `n`
fn wordsDiffer(a: std.mem.Allocator, g: *G, x: A, y: A, n: usize) !usize {
    const p = try a.alloc(f32, n);
    defer a.free(p);
    const q = try a.alloc(f32, n);
    defer a.free(q);
    _ = try g.hostF32(try g.astype(x, .float32), p);
    _ = try g.hostF32(try g.astype(y, .float32), q);
    var d: usize = 0;
    for (p, q) |u, w| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return d;
}

test "turboexl3 gpu pg2: the fused matmul and finish give dequantize + matmul + finish's words at every tabled shape (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x9632);
    const rnd = prng.random();
    const saved = pgemm.mode.get().*;
    defer pgemm.mode.get().* = saved;
    var bad: usize = 0;
    var fused_lines: usize = 0;
    for (pgemm.table) |e| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const p = try randProj(a, &g, rnd, e.k, e.n, 48);
        for ([_]u32{ 64, 100, 2048 }) |rows| {
            if (!pgemm.mlxRegular(rows, e.n, e.k)) continue;
            const rm = g.mark();
            defer g.resetTo(rm);
            const x = try randX(a, &g, rnd, rows, e.k);
            pgemm.mode.get().* = .off;
            const want = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
            try g.evalAll(&.{want});
            pgemm.mode.get().* = .on;
            const before = pgemm.calls.get().*.load(.monotonic);
            const got = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
            try g.evalAll(&.{got});
            const fused = pgemm.calls.get().*.load(.monotonic) > before;
            pgemm.mode.get().* = .off;
            const d = try wordsDiffer(a, &g, got, want, @as(usize, rows) * e.n);
            fused_lines += @intFromBool(fused);
            bad += @intFromBool(d != 0 or !fused);
            std.debug.print("TURBOEXL3_PG2_BYTES k={d} n={d} rows={d} fused={} differing={d}/{d} {s}\n", .{ e.k, e.n, rows, fused, d, @as(usize, rows) * e.n, if (d == 0 and fused) "OK" else "DIFFER" });
        }
    }
    try testing.expect(fused_lines > 0);
    try testing.expectEqual(@as(usize, 0), bad);
}

test "turboexl3 gpu linear pieces: the pieces' one finish gives the one-call finish's words over their joined inner products (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x91EC);
    const rnd = prng.random();
    const in: u32 = 2560;
    const out: u32 = 6144;
    const p = try randProj(a, &g, rnd, in, out, 64);
    var bad: usize = 0;
    const sets = [_][]const usize{ &.{ 40, 33 }, &.{ 64, 64, 64 }, &.{ 100, 50, 33, 40 } };
    for (sets) |pieces| {
        const rm = g.mark();
        defer g.resetTo(rm);
        var total: u32 = 0;
        for (pieces) |r| total += @intCast(r);
        const x = try randX(a, &g, rnd, total, in);
        const W = try g.adopt(try root.linearPiecesBegin(s, p, pieces, .bfloat16, .mul1));
        var inners: [4]A = undefined;
        var r0: u32 = 0;
        for (pieces, 0..) |r, i| {
            const xi = try g.slice(x, &.{ ci(r0), 0 }, &.{ ci(r0 + r), ci(in) }, &.{ 1, 1 });
            inners[i] = try g.adopt(try root.linearPieceInner(s, xi, W, p, .mul1));
            r0 += @intCast(r);
        }
        const got = try g.adopt(try root.linearPiecesFinish(s, inners[0..pieces.len], p, .mul1, .bfloat16));
        const want = try g.adopt(try root.linearFinish(s, try g.concat(inners[0..pieces.len], 0), p, .mul1, .bfloat16));
        const d = try wordsDiffer(a, &g, got, want, @as(usize, total) * out);
        const whole = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
        const dl = try wordsDiffer(a, &g, got, whole, @as(usize, total) * out);
        bad += @intFromBool(d != 0);
        std.debug.print("TURBOEXL3_PIECES_BYTES pieces={d} rows={d} differing={d}/{d} {s} (against linear of the joined rows: {d} differing)\n", .{ pieces.len, total, d, @as(usize, total) * out, if (d == 0) "OK" else "DIFFER", dl });
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

const Case = struct { hidden: u32, inter: u32, topk: u32, rows: u32, experts: u32, n: u32, dec: format.Decode };

/// form 1's cases (gpu_test.zig)
const cases = [_]Case{
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 32, .dec = .mul1 },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 40, .dec = .mul1 },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 48, .dec = .{ .codebook = .mcg, .window = .w12 } },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 128, .dec = .mcg },
    .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 1, .experts = 12, .n = 80, .dec = .mul1 },
    .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 4, .experts = 12, .n = 80, .dec = .mul1 },
    .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 8, .experts = 12, .n = 80, .dec = .mul1 },
};

test "turboexl3 gpu unfused: the eleven-launch chain equals the host reference within the decode chain's tolerance, at form 1's cases (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var failed: usize = 0;
    for (cases) |c| {
        var prng = std.Random.DefaultPrng.init(0x0F05ED ^ c.n);
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
                slots[r * c.topk + k] = @intCast((k * 2 + r) % c.experts);
                scores[r * c.topk + k] = 0.2 + rnd.float(f32);
                sum += scores[r * c.topk + k];
            }
            for (0..c.topk) |k| scores[r * c.topk + k] /= sum;
        }
        const y_ref = try a.alloc(f32, c.rows * c.hidden);
        defer a.free(y_ref);
        for (0..c.rows) |r| try ref.moeRowDec(a, &bank, c.dec, x[r * c.hidden ..][0..c.hidden], slots[r * c.topk ..][0..c.topk], scores[r * c.topk ..][0..c.topk], y_ref[r * c.hidden ..][0..c.hidden]);
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
        const geo: decode.Geometry = .{ .hidden = c.hidden, .inter = c.inter, .topk = c.topk, .rate = rate, .dec = c.dec };
        var steps = try unfused.Steps.init(geo);
        defer steps.deinit();
        const y = try g.adopt(try steps.moe(s, xa, sa, wa, arrays, c.rows));
        const got = try a.alloc(f32, c.rows * c.hidden);
        defer a.free(got);
        _ = try g.hostF32(y, got);
        const e = ref.compare(got, y_ref);
        // against form 1's words (the same math in two launches): reported
        var k1 = try decode.Kernels.init(geo);
        defer k1.deinit();
        const o1 = try k1.moe(s, xa, sa, wa, arrays, c.rows);
        _ = try g.adopt(o1.h);
        const y1 = try g.adopt(o1.y);
        const d1 = try wordsDiffer(a, &g, y, y1, c.rows * c.hidden);
        const ok = std.math.isFinite(e.max_abs) and e.max_ratio <= 5e-3;
        failed += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_UNFUSED n={d} cb={s} w={d} geo={d}x{d} rows={d} topk={d} max_abs={e:.3} rms={e:.3} ratio={e:.3} bar=5.0e-3 {s} (against form 1: {d}/{d} words differ)\n", .{ c.n, @tagName(c.dec.codebook), c.dec.window.bits(), c.hidden, c.inter, c.rows, c.topk, e.max_abs, e.rms_ref, e.max_ratio, if (ok) "PASS" else "FAIL", d1, c.rows * c.hidden });
    }
    try testing.expectEqual(@as(usize, 0), failed);
}
