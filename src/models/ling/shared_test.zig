//! Ling-3.1-flash's shared expert at decode rows (`models/ling/shared.zig`, `root.lingSharedExpert`). CPU (always): the
//! two joint texts are dense's DEC-ONE with the prepare inside around the member choice, its own tile address and the
//! member's store, then the pair's arrival, the host's SwiGLU (unclamped, or clamped after the SiLU) and text_prep_s's
//! statements; the plan's refusals. GPU (TURBOEXL3_GPU=1, else skipped), at Ling's shared expert (4096 -> 1280 -> 4096,
//! K6, MUL1): the shared expert gives the composed chain's words (`linear(gate)`, `linear(up)`, the host's SwiGLU as
//! MLX's ops, `linear(down)`) at rows 1 to 4, limits 0 and 10, bf16 and f16, with the clamp reached; the joint's columns
//! give each member's `linear` words. Lines: TURBOEXL3_LING_SHARED (then TURBOEXL3_LING_SHARED_DONE),
//! TURBOEXL3_LING_SHARED_CLAMP.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const dense = @import("../../core/dense.zig");
const root = @import("../../root.zig");
const shared = @import("shared.zig");

const testing = std.testing;
const A = mlx.mlx_array;

const hidden: u32 = 4096;
const inter: u32 = 1280;
const n_k6: u32 = 96;
const limit: u32 = 10;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

test "turboexl3 ling shared texts: dense's DEC-ONE (prepare inside) around the member choice, its tile address and the member's store, then the pair's arrival, the host's SwiGLU and text_prep_s's statements" {
    const body = dense.dec_prep1 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive_1 ++ shared.mtile ++ dense.dec_gemv_arrive_2 ++ dense.dec_xsum1 ++ dense.dec_finish_v ++ shared.sgu_store ++ shared.sgu_pair_1;
    try testing.expect(std.mem.eql(u8, shared.text_sgu, dense.dec_head ++ shared.msel ++ body ++ shared.sgu_swiglu ++ shared.sgu_pair_2));
    try testing.expect(std.mem.eql(u8, shared.text_sguco, dense.dec_head ++ shared.msel ++ body ++ shared.sgu_swiglu_clamp_out ++ shared.sgu_pair_2));
    // dense's one-launch text has the same pieces where these have the member's (the tile address, the store)
    try testing.expect(std.mem.eql(u8, dense.text_dec1, dense.dec_head ++ dense.dec_prep1 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive_1 ++ dense.dec_tile ++ dense.dec_gemv_arrive_2 ++ dense.dec_xsum1 ++ dense.dec_finish_v ++ dense.dec_finish_store));
    const pin = comptime support.pins(@embedFile("kernels/test/pins/shared_texts.metal"));
    // the member's store is dense's with the member's svh; the tile address is dense's in the member's own trellis
    try testing.expectEqual(@as(usize, 1), count(shared.sgu_store, pin[0]));
    try testing.expectEqual(@as(usize, 1), count(dense.dec_finish_store, pin[1]));
    try testing.expectEqual(@as(usize, 1), count(shared.mtile, pin[2]));
    try testing.expectEqual(@as(usize, 1), count(dense.dec_tile, pin[3]));
    // the down's prepare: text_prep_s's H128, rounding, lane order and block sum, on the activation as prep_s reads x
    for (pin[4..9]) |line| {
        try testing.expectEqual(@as(usize, 1), count(shared.sgu_pair_2, line));
        try testing.expectEqual(@as(usize, 1), count(dense.text_prep_s, line));
    }
    try testing.expectEqual(@as(usize, 1), count(shared.sgu_pair_2, pin[9]));
    try testing.expectEqual(@as(usize, 1), count(dense.text_prep_s, pin[10]));
    // the host's SwiGLU: mlx-serve's fusedSwiGLU in the activation dtype, or the plugin's swigluClampOut in f32
    for (pin[11..14]) |line| try testing.expectEqual(@as(usize, 1), count(shared.sgu_swiglu, line));
    for (pin[14..21]) |line| try testing.expectEqual(@as(usize, 1), count(shared.sgu_swiglu_clamp_out, line));
    try testing.expectEqual(@as(usize, 0), count(shared.sgu_swiglu, "sig32"));
    try testing.expectEqual(@as(usize, 0), count(shared.sgu_swiglu_clamp_out, "sigtab"));
    // the pair's arrival: DEC-ONE's arrival idiom on its own counters, the second of two finishing
    try testing.expectEqual(@as(usize, 1), count(shared.sgu_pair_1, pin[21]));
    try testing.expectEqual(@as(usize, 1), count(shared.sgu_pair_1, pin[22]));
    try testing.expectEqual(@as(usize, 1), count(dense.dec_gemv_arrive_2, pin[23]));
}

// ── GPU ──

/// A projection [in -> out] of random trellis words, its svh scaled so outputs reach about `scale` in magnitude
fn randProj(a: std.mem.Allocator, g: *G, rnd: std.Random, in: u32, out: u32, n: u32, scale: f32) !root.Proj {
    const t = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(t);
    rnd.bytes(std.mem.sliceAsBytes(t));
    const su = try a.alloc(u16, in);
    defer a.free(su);
    for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, out);
    defer a.free(sv);
    for (sv) |*v| v.* = format.f32ToF16Bits(scale * (0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    return .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(t), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(out)}, .float16),
    };
}

fn randX(a: std.mem.Allocator, g: *G, rnd: std.Random, rows: u32, dt: mlx.mlx_dtype) !A {
    const xh = try a.alloc(f32, @as(usize, rows) * hidden);
    defer a.free(xh);
    for (xh) |*v| v.* = @bitCast(@as(u32, support.bf16Bits(rnd.floatNorm(f32))) << 16);
    return g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(hidden) }, .float32), dt);
}

fn op2(g: *G, comptime f: anytype, a: A, b: A) !A {
    var r = mlx.mlx_array_new();
    mlx.check(f(&r, a, b, g.s)) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
}

/// MLX's clip (mlx/ops.cpp): maximum with the lower bound, then minimum with the upper
fn clip(g: *G, a: A, lo: A, hi: A) !A {
    return op2(g, mlx.mlx_minimum, try op2(g, mlx.mlx_maximum, a, lo), hi);
}

/// The plugin's composed shared expert at decode rows, op for op: the gate and the up `linear`; at limit 0 the host's
/// SwiGLU as MLX's ops in x's dtype (sigmoid, multiply, multiply: the words of mlx-serve's fusedSwiGLU, which reads
/// MLX's sigmoid from its table); above, the plugin's swigluClampOut (astype f32, sigmoid, multiply, minimum, clip,
/// multiply, astype); then the down `linear`
fn hostShared(g: *G, s: mlx.mlx_stream, x: A, gate: root.Proj, up: root.Proj, down: root.Proj, lim: u32) !A {
    const dt = mlx.mlx_array_dtype(x);
    const gv = try g.adopt(try root.linear(s, x, gate.trellis, gate.suh, gate.svh, .mul1));
    const uv = try g.adopt(try root.linear(s, x, up.trellis, up.suh, up.svh, .mul1));
    const h = if (lim == 0) try g.mul(try g.mul(gv, try g.sigmoid(gv)), uv) else blk: {
        const g32 = try g.astype(gv, .float32);
        const u32_ = try g.astype(uv, .float32);
        const act = try g.mul(g32, try g.sigmoid(g32));
        const hi = try g.scalar(@floatFromInt(lim), .float32);
        const lo = try g.scalar(-@as(f64, @floatFromInt(lim)), .float32);
        const am = try op2(g, mlx.mlx_minimum, act, hi);
        const uc = try clip(g, u32_, lo, hi);
        break :blk try g.astype(try g.mul(am, uc), dt);
    };
    return g.adopt(try root.linear(s, h, down.trellis, down.suh, down.svh, .mul1));
}

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

test "turboexl3 gpu ling shared: the gate and the up in one launch with the host's SwiGLU and the down's prepare in their pairs' second finishes, then the down, give the composed shared expert's words at rows 1 to 4, limits 0 and 10 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    std.debug.print("TURBOEXL3_LING_SHARED_START\n", .{});
    var prng = std.Random.DefaultPrng.init(0x11A65E);
    const rnd = prng.random();
    // the gate and the up scaled so their outputs pass the limit (the SiLU above 10, the up beyond +-10) on some values
    const gate = try randProj(a, &g, rnd, hidden, inter, n_k6, 8);
    const up = try randProj(a, &g, rnd, hidden, inter, n_k6, 8);
    const down = try randProj(a, &g, rnd, inter, hidden, n_k6, 1);
    var lines: usize = 0;
    var bad: usize = 0;
    var clamped_seen = false;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
        const tabs = [2]A{ try g.adopt(try shared.sigmoidTable(s, dt, false)), try g.adopt(try shared.sigmoidTable(s, dt, true)) };
        for ([_]u32{ 0, limit }) |lim| for ([_]u32{ 1, 2, 3, 4 }) |rows| {
            if (dt == .float16 and rows != 1 and rows != 3) continue;
            const rm = g.mark();
            defer g.resetTo(rm);
            const x = try randX(a, &g, rnd, rows, dt);
            const tab = tabs[@intFromBool(lim != 0)];
            try testing.expect(try shared.sharedTakes(x, gate, up, down, .mul1));
            const want = try hostShared(&g, s, x, gate, up, down, lim);
            const got = try g.adopt(try shared.sharedExpert(s, x, gate, up, down, lim, tab, .mul1));
            try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(hidden) }, mlx.getShape(got));
            const d = try wordsDiffer(a, &g, got, want, rows * hidden);
            // the joint's columns: each member's `linear` words
            const parts = try shared.jointParts(s, x, gate, up, down, lim, tab, .mul1);
            const jy = try g.adopt(parts[0]);
            _ = try g.adopt(parts[1]);
            _ = try g.adopt(parts[2]);
            const gv = try g.adopt(try root.linear(s, x, gate.trellis, gate.suh, gate.svh, .mul1));
            const uv = try g.adopt(try root.linear(s, x, up.trellis, up.suh, up.svh, .mul1));
            const dj = try wordsDiffer(a, &g, jy, try g.concat(&.{ gv, uv }, 1), rows * 2 * inter);
            if (lim != 0 and rows == 4) {
                const nel: usize = 4 * inter;
                const hv = try a.alloc(f32, nel);
                defer a.free(hv);
                var over: [2]usize = .{ 0, 0 };
                _ = try g.hostF32(try g.astype(gv, .float32), hv);
                for (hv) |v| over[0] += @intFromBool(v * (1.0 / (1.0 + @exp(-v))) > @as(f32, @floatFromInt(limit)));
                _ = try g.hostF32(try g.astype(uv, .float32), hv);
                for (hv) |v| over[1] += @intFromBool(@abs(v) > @as(f32, @floatFromInt(limit)));
                std.debug.print("TURBOEXL3_LING_SHARED_CLAMP rows=4 x={s} silu_over={d}/{d} ups_over={d}/{d}\n", .{ @tagName(dt), over[0], nel, over[1], nel });
                if (over[0] > 0 and over[1] > 0) clamped_seen = true;
            }
            lines += 1;
            const ok = d == 0 and dj == 0;
            bad += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_LING_SHARED rows={d} x={s} limit={d} differing={d}/{d} joint_vs_linears={d}/{d} {s}\n", .{ rows, @tagName(dt), lim, d, rows * hidden, dj, rows * 2 * inter, if (ok) "OK" else "DIFFER" });
        };
    }
    // past DEC-ONE's rows the call is refused before any launch
    const x5 = try randX(a, &g, rnd, 5, .bfloat16);
    try testing.expect(!(try shared.sharedTakes(x5, gate, up, down, .mul1)));
    const tab5 = try g.adopt(try shared.sigmoidTable(s, .bfloat16, false));
    try testing.expectError(error.Exl3EngineNotImplemented, shared.sharedExpert(s, x5, gate, up, down, 0, tab5, .mul1));
    std.debug.print("TURBOEXL3_LING_SHARED_DONE lines={d} differing_lines={d} clamp_reached={} {s}\n", .{ lines, bad, clamped_seen, if (bad == 0 and clamped_seen) "BYTES_OK" else "BYTES_DIFFER" });
    try testing.expect(clamped_seen);
    try testing.expectEqual(@as(usize, 0), bad);
}
