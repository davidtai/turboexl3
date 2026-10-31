//! DeepSeek-V4.1's trunk on turboexl3's linears (TURBOEXL3_GPU=1, else skipped; the CPU test always runs):
//! - `root.linear` at every trunk and vision shape the pack quantizes, all K6 MUL1;
//! - `root.linearBias` where the vision tower and the aligner carry a bias;
//! - `root.linearGrouped` at the attention's `wo_a`, 8 slices of 4096 -> 1024.
//! Each is held to the host reference (`ref.Linear`: `format.project`'s values on synthetic weights from
//! `ref.synthLinear`, plus the bias) at rows 1, 4, 16, 32 and 512: DEC-ONE, the GEMV chain's two chunks, the register
//! path and dequantize + matmul. The head runs rows 1 and 4 only: its 662M-weight host reference is the cost. x is
//! bf16-exact. f32 x and y are held to 5e-3 of the reference rms, the dense chain's bar; bf16 x and y (rows 4 and 512) to
//! 2.5e-2. Up to 16 rows every row is compared, past that six (both ends, either side of the middle). Each slice of
//! `linearGrouped` must also equal `linear` on that slice, value for value (a zero's sign may differ at decode rows).
//! Lines: TURBOEXL3_DS_PARITY per case, row count and dtype; TURBOEXL3_DS_SLICES per grouped row count.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("test_support.zig");
const G = support.Graph;
const format = @import("format.zig");
const ref = @import("ref.zig");
const root = @import("root.zig");
const dense = @import("dense.zig");
const grouped = @import("grouped.zig");

const testing = std.testing;

/// K6: the rate of every DeepSeek-V4.1 trunk linear in the pack
const N: u32 = 96;

const Case = struct { name: []const u8, in: u32, out: u32, bias: bool = false, head: bool = false };

/// Every distinct trunk and vision shape; a name lists the linears that share it
const cases = [_]Case{
    .{ .name = "attn.wq_a", .in = 5120, .out = 1280 },
    .{ .name = "attn.wq_b", .in = 1280, .out = 32768 },
    .{ .name = "attn.wkv/compressor.wkv/compressor.wgate", .in = 5120, .out = 512 },
    .{ .name = "attn.wo_b", .in = 8192, .out = 5120 },
    .{ .name = "indexer.wq_b", .in = 1280, .out = 4096 },
    .{ .name = "shared.w1/w3", .in = 5120, .out = 2304 },
    .{ .name = "shared.w2", .in = 2304, .out = 5120 },
    .{ .name = "head", .in = 5120, .out = 129280, .head = true },
    .{ .name = "vision.q/k/v/wo", .in = 1024, .out = 1024, .bias = true },
    .{ .name = "vision.mlp.gate/up", .in = 1024, .out = 2816 },
    .{ .name = "vision.mlp.w2", .in = 2816, .out = 1024 },
    .{ .name = "aligner.w1", .in = 9216, .out = 5120, .bias = true },
    .{ .name = "aligner.w2", .in = 5120, .out = 5120, .bias = true },
};

/// The attention's wo_a: 8 slices of 4096 -> 1024
const wo_a = struct {
    const groups: u32 = 8;
    const in: u32 = 4096;
    const out: u32 = 1024;
};

/// DEC-ONE (1, 4), the GEMV chain's two chunks (16), the register path (32), dequantize + matmul (512)
const row_counts = [_]u32{ 1, 4, 16, 32, 512 };
const head_rows = [_]u32{ 1, 4 };

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn bf16Exact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
}

/// The rows a case compares: every row up to 16; past that both ends and either side of the middle
fn pickRows(rows: u32, buf: *[16]u32) []const u32 {
    if (rows <= 16) {
        for (buf[0..rows], 0..) |*r, i| r.* = @intCast(i);
        return buf[0..rows];
    }
    buf[0..6].* = .{ 0, 1, rows / 2 - 1, rows / 2, rows - 2, rows - 1 };
    return buf[0..6];
}

/// The picked rows of `got` (rows of `width` values) against `want` (the picked rows' reference, in order): one
/// TURBOEXL3_DS_PARITY line, and 1 on a failure
fn report(name: []const u8, rows: u32, xd: []const u8, width: usize, got: []const f32, pick: []const u32, want: []const f32, sel: []f32, bar: f64) usize {
    for (pick, 0..) |r, i| @memcpy(sel[i * width ..][0..width], got[r * width ..][0..width]);
    const e = ref.compare(sel[0 .. pick.len * width], want[0 .. pick.len * width]);
    const ok = std.math.isFinite(e.max_abs) and e.max_ratio <= bar;
    std.debug.print("TURBOEXL3_DS_PARITY {s} rows={d} x={s} compared_rows={d} max_abs={e:.3} rms={e:.3} ratio={e:.3} bar={e:.1} {s}\n", .{ name, rows, xd, pick.len, e.max_abs, e.rms_ref, e.max_ratio, bar, if (ok) "PASS" else "FAIL" });
    return @intFromBool(!ok);
}

test "turboexl3 deepseek: every DeepSeek-V4.1 trunk and vision shape is a width the dense kernels admit, wo_a a grouped shape" {
    for (cases) |c| {
        try testing.expect(c.in % 128 == 0 and c.out % 128 == 0 and (c.in / 16) % 4 == 0);
        // the GEMV's K split: whole four-tile parts in every slice
        const k = dense.Dense.ksFor(c.in, c.out);
        try testing.expect(k >= 1 and ((c.in / 16) / 4) % k == 0);
    }
    const sh: grouped.Shape = .{ .groups = wo_a.groups, .in = wo_a.in, .out = wo_a.out, .rate = format.Rate.fromN(N).?, .dec = .mul1, .out_dtype = .bfloat16 };
    try testing.expect(sh.admitted());
}

/// One shape through `root.linear` (`root.linearBias` where it carries a bias) at its row counts: the number of failed lines
fn linearCase(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, c: Case) !usize {
    var prng = std.Random.DefaultPrng.init(0xD5E41 ^ (@as(u64, c.in) << 24) ^ c.out);
    const rnd = prng.random();
    var lin = try ref.synthLinear(a, rnd, 1, c.in, c.out, format.Rate.fromN(N).?, .mul1);
    defer lin.deinit(a);
    const bias = try a.alloc(u16, c.out);
    defer a.free(bias);
    for (bias) |*v| v.* = format.f32ToF16Bits(2.0 * rnd.float(f32) - 1.0);
    const mark = g.mark();
    defer g.resetTo(mark);
    const trellis = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(c.in / 16), ci(c.out / 16), ci(N) }, .uint16);
    const suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(c.in)}, .float16);
    const svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(c.out)}, .float16);
    // the bias in each activation dtype, as the host converts it at load
    const bias16 = try g.hostArray(std.mem.sliceAsBytes(bias), &.{ci(c.out)}, .float16);
    const bias32 = try g.astype(bias16, .float32);
    const bias_bf = try g.astype(bias16, .bfloat16);
    const transformed = try a.alloc(f32, c.in);
    defer a.free(transformed);
    const inner = try a.alloc(f32, c.out);
    defer a.free(inner);
    var failed: usize = 0;
    for (if (c.head) head_rows[0..] else row_counts[0..]) |rows| {
        const rmark = g.mark();
        defer g.resetTo(rmark);
        const x = try a.alloc(f32, rows * c.in);
        defer a.free(x);
        for (x) |*v| v.* = bf16Exact(rnd.floatNorm(f32));
        var pick_buf: [16]u32 = undefined;
        const pick = pickRows(rows, &pick_buf);
        const want = try a.alloc(f32, pick.len * c.out);
        defer a.free(want);
        for (pick, 0..) |r, i| {
            const yr = want[i * c.out ..][0..c.out];
            lin.row(0, x[r * c.in ..][0..c.in], transformed, inner, yr);
            if (c.bias) for (yr, bias) |*v, b| {
                v.* += format.f16BitsToF32(b);
            };
        }
        const got = try a.alloc(f32, rows * c.out);
        defer a.free(got);
        const sel = try a.alloc(f32, pick.len * c.out);
        defer a.free(sel);
        // f32 x and y: the kernels' arithmetic against the reference's
        const x3 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(c.in) }, .float32);
        const y = try g.adopt(if (c.bias) try root.linearBias(s, x3, trellis, suh, svh, bias32, .mul1) else try root.linear(s, x3, trellis, suh, svh, .mul1));
        try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(y));
        try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(c.out) }, mlx.getShape(y));
        _ = try g.hostF32(y, got);
        failed += report(c.name, rows, "f32", c.out, got, pick, want, sel, 5e-3);
        // bf16 x and y, as served (one bf16 rounding of y, and of the sum, on top)
        if (rows == 4 or rows == 512) {
            const xb = try g.astype(x3, .bfloat16);
            const yb = try g.adopt(if (c.bias) try root.linearBias(s, xb, trellis, suh, svh, bias_bf, .mul1) else try root.linear(s, xb, trellis, suh, svh, .mul1));
            try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(yb));
            _ = try g.hostF32(try g.astype(yb, .float32), got);
            failed += report(c.name, rows, "bf16", c.out, got, pick, want, sel, 2.5e-2);
        }
    }
    return failed;
}

test "turboexl3 gpu deepseek: root.linear and root.linearBias equal the host reference at every DeepSeek-V4.1 trunk and vision shape, K6 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    // the head's W_hat alone is 1.3 GB of host reference
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var failed: usize = 0;
    for (cases) |c| failed += try linearCase(a, &g, s, c);
    try testing.expectEqual(@as(usize, 0), failed);
}

test "turboexl3 gpu deepseek: root.linearGrouped at wo_a (8 slices of 4096 -> 1024) equals the host reference, each slice linear's values on it (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const gs = wo_a.groups;
    const in = wo_a.in;
    const out = wo_a.out;
    var prng = std.Random.DefaultPrng.init(0xA0A6);
    const rnd = prng.random();
    var lin = try ref.synthLinear(a, rnd, gs, in, out, format.Rate.fromN(N).?, .mul1);
    defer lin.deinit(a);
    const mark = g.mark();
    defer g.resetTo(mark);
    const p: root.Proj = .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(gs), ci(in / 16), ci(out / 16), ci(N) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ ci(gs), ci(in) }, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ ci(gs), ci(out) }, .float16),
    };
    // each slice as a dense linear of its own
    var slices: [wo_a.groups]root.Proj = undefined;
    for (&slices, 0..) |*sl, gi| {
        const suh_g: []const u16 = lin.suh[gi * in ..][0..in];
        const svh_g: []const u16 = lin.svh[gi * out ..][0..out];
        sl.* = .{
            .trellis = try g.hostArray(std.mem.sliceAsBytes(lin.trellisOf(gi)), &.{ ci(in / 16), ci(out / 16), ci(N) }, .uint16),
            .suh = try g.hostArray(std.mem.sliceAsBytes(suh_g), &.{ci(in)}, .float16),
            .svh = try g.hostArray(std.mem.sliceAsBytes(svh_g), &.{ci(out)}, .float16),
        };
    }
    const transformed = try a.alloc(f32, in);
    defer a.free(transformed);
    const inner = try a.alloc(f32, out);
    defer a.free(inner);
    // a row of y: its G slices
    const width: usize = gs * out;
    var failed: usize = 0;
    for (row_counts) |rows| {
        const rmark = g.mark();
        defer g.resetTo(rmark);
        const x = try a.alloc(f32, rows * gs * in);
        defer a.free(x);
        for (x) |*v| v.* = bf16Exact(rnd.floatNorm(f32));
        var pick_buf: [16]u32 = undefined;
        const pick = pickRows(rows, &pick_buf);
        const want = try a.alloc(f32, pick.len * width);
        defer a.free(want);
        for (pick, 0..) |r, i| for (0..gs) |gi| {
            lin.row(gi, x[(r * gs + gi) * in ..][0..in], transformed, inner, want[i * width + gi * out ..][0..out]);
        };
        const got = try a.alloc(f32, rows * width);
        defer a.free(got);
        const sel = try a.alloc(f32, pick.len * width);
        defer a.free(sel);
        const x4 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(gs), ci(in) }, .float32);
        const y = try g.adopt(try root.linearGrouped(s, x4, p, .mul1));
        try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(y));
        try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(gs), ci(out) }, mlx.getShape(y));
        _ = try g.hostF32(y, got);
        failed += report("attn.wo_a/grouped", rows, "f32", width, got, pick, want, sel, 5e-3);
        // each slice against `linear` on it, every row: a wrong slot, slice or stride shows here whatever the tolerance
        {
            const xg = try a.alloc(f32, rows * in);
            defer a.free(xg);
            const yg = try a.alloc(f32, rows * out);
            defer a.free(yg);
            var diff: usize = 0;
            var max_d: f64 = 0;
            for (slices, 0..) |sl, gi| {
                for (0..rows) |r| @memcpy(xg[r * in ..][0..in], x[(r * gs + gi) * in ..][0..in]);
                const xa = try g.hostArray(std.mem.sliceAsBytes(xg), &.{ 1, ci(rows), ci(in) }, .float32);
                _ = try g.hostF32(try g.adopt(try root.linear(s, xa, sl.trellis, sl.suh, sl.svh, .mul1)), yg);
                for (0..rows) |r| for (got[(r * gs + gi) * out ..][0..out], yg[r * out ..][0..out]) |u, w| {
                    if (u != w) {
                        diff += 1;
                        max_d = @max(max_d, @abs(@as(f64, u) - @as(f64, w)));
                    }
                };
            }
            const ok = diff == 0;
            std.debug.print("TURBOEXL3_DS_SLICES attn.wo_a rows={d} differing={d}/{d} max_abs_diff={e:.3} {s}\n", .{ rows, diff, rows * width, max_d, if (ok) "PASS" else "FAIL" });
            failed += @intFromBool(!ok);
        }
        // bf16 x and y, as served
        if (rows == 4 or rows == 512) {
            const xb = try g.astype(x4, .bfloat16);
            const yb = try g.adopt(try root.linearGrouped(s, xb, p, .mul1));
            try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(yb));
            _ = try g.hostF32(try g.astype(yb, .float32), got);
            failed += report("attn.wo_a/grouped", rows, "bf16", width, got, pick, want, sel, 2.5e-2);
        }
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const Stat = struct { min: f64, med: f64 };

/// `ctx.call()` timed on the device: `calls` graphs per evaluation, 2 warm-up rounds, then the min and median us per call
/// over 8 rounds.
fn timeIt(g: *G, ctx: anytype, calls: usize) !Stat {
    var samples: [8]f64 = undefined;
    for (0..samples.len + 2) |round| {
        var ys: [8]mlx.mlx_array = undefined;
        for (ys[0..calls]) |*y| y.* = try ctx.call();
        const t0 = nowNs();
        try g.evalAll(ys[0..calls]);
        const dt = nowNs() - t0;
        for (ys[0..calls]) |y| _ = mlx.mlx_array_free(y);
        if (round >= 2) samples[round - 2] = @as(f64, @floatFromInt(dt)) / 1000.0 / @as(f64, @floatFromInt(calls));
    }
    std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
    return .{ .min = samples[0], .med = samples[samples.len / 2] };
}

/// A projection's random operands on the device: every trellis pattern is a codeword; suh, svh as `synthBank` scales them.
fn randProj(a: std.mem.Allocator, g: *G, rnd: std.Random, lead: ?u32, in: u32, out: u32, n: u32) !root.Proj {
    const L: usize = lead orelse 1;
    const t = try a.alloc(u16, L * (in / 16) * (out / 16) * n);
    defer a.free(t);
    rnd.bytes(std.mem.sliceAsBytes(t));
    const su = try a.alloc(u16, L * in);
    defer a.free(su);
    for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, L * out);
    defer a.free(sv);
    for (sv) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    if (lead) |e| return .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(t), &.{ ci(e), ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ ci(e), ci(in) }, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ ci(e), ci(out) }, .float16),
    };
    return .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(t), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(out)}, .float16),
    };
}

fn randBf16(a: std.mem.Allocator, g: *G, rnd: std.Random, shape: []const c_int) !mlx.mlx_array {
    var len: usize = 1;
    for (shape) |d| len *= @intCast(d);
    const x = try a.alloc(f32, len);
    defer a.free(x);
    for (x) |*v| v.* = bf16Exact(rnd.floatNorm(f32));
    return g.astype(try g.hostArray(std.mem.sliceAsBytes(x), shape, .float32), .bfloat16);
}

// DeepSeek-V4.1-Flash's projections timed on the device, for the serving plan (TURBOEXL3_DS_TIME=1 with TURBOEXL3_GPU=1;
// it asserts nothing): the clamped routed MoE (5120 x 2304, 384 experts, top-6, limit 10) at K3 and K2, at decode rows 1, 2,
// 4, 6 and a 4096-row prefill chunk (TURBOEXL3_DS_TIME_ROWS overrides), its routing drawn as the clamped test draws it
// (lognormal expert popularity, sigma 1.5); then every K6 trunk linear and the grouped wo_a at rows 1, 4 and 4096. Each
// line gives the min and median us per call and the GEMM work's TFLOPS at the min.
test "turboexl3 gpu deepseek timing: the routed MoE at K3 and K2, the K6 trunk linears and the grouped wo_a at DeepSeek-V4.1's shapes (TURBOEXL3_DS_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_DS_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xD5711E);
    const rnd = prng.random();
    const hidden: u32 = 5120;
    const inter: u32 = 2304;
    const experts: u32 = 384;
    const topk: u32 = 6;
    var rows_buf: [8]u32 = undefined;
    var row_list: []const u32 = &.{ 1, 2, 4, 6, 4096 };
    if (std.c.getenv("TURBOEXL3_DS_TIME_ROWS")) |v| {
        var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
        var nr: usize = 0;
        while (it.next()) |tok| : (nr += 1) rows_buf[nr] = std.fmt.parseInt(u32, tok, 10) catch return error.BadRows;
        row_list = rows_buf[0..nr];
    }
    for ([_]u32{ 48, 32 }) |n| {
        const mark = g.mark();
        defer g.resetTo(mark);
        defer _ = mlx.mlx_clear_cache();
        const bank: root.Bank = .{
            .gate = try randProj(a, &g, rnd, experts, hidden, inter, n),
            .up = try randProj(a, &g, rnd, experts, hidden, inter, n),
            .down = try randProj(a, &g, rnd, experts, inter, hidden, n),
        };
        for (row_list) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const logw = try a.alloc(f64, experts);
            defer a.free(logw);
            for (logw) |*w| w.* = 1.5 * rnd.floatNorm(f64);
            const keys = try a.alloc(f64, experts);
            defer a.free(keys);
            const inds = try a.alloc(i32, rows * topk);
            defer a.free(inds);
            const sc = try a.alloc(f32, rows * topk);
            defer a.free(sc);
            for (0..rows) |r| {
                for (keys, logw) |*kv, lw| kv.* = lw - @log(-@log(@max(rnd.float(f64), 1e-300)));
                for (0..topk) |k| {
                    var best: usize = 0;
                    for (keys, 0..) |kv, e| if (kv > keys[best]) {
                        best = e;
                    };
                    inds[r * topk + k] = @intCast(best);
                    keys[best] = -std.math.inf(f64);
                    sc[r * topk + k] = bf16Exact(1.0 / @as(f32, @floatFromInt(topk)));
                }
            }
            const Ctx = struct {
                s: mlx.mlx_stream,
                x: mlx.mlx_array,
                bank: root.Bank,
                inds: mlx.mlx_array,
                sc: mlx.mlx_array,
                fn call(c: @This()) !mlx.mlx_array {
                    return root.moeClamped(c.s, c.x, c.bank, c.inds, c.sc, .mul1, 10);
                }
            };
            const ctx: Ctx = .{
                .s = s,
                .x = try randBf16(a, &g, rnd, &.{ 1, ci(rows), ci(hidden) }),
                .bank = bank,
                .inds = try g.hostArray(std.mem.sliceAsBytes(inds), &.{ 1, ci(rows), ci(topk) }, .int32),
                .sc = try g.astype(try g.hostArray(std.mem.sliceAsBytes(sc), &.{ 1, ci(rows), ci(topk) }, .float32), .bfloat16),
            };
            const st = try timeIt(&g, ctx, if (rows <= 16) 8 else 2);
            const flops = 2.0 * @as(f64, @floatFromInt(rows * topk)) * 3.0 * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter));
            std.debug.print("TURBOEXL3_DS_TIME moe K{d} rows={d} min_us={d:.1} med_us={d:.1} tflops={d:.2}\n", .{ n / 16, rows, st.min, st.med, flops / (st.min * 1e6) });
        }
    }
    for (cases) |c| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const p = try randProj(a, &g, rnd, null, c.in, c.out, N);
        for ([_]u32{ 1, 4, 4096 }) |rows| {
            if (c.head and rows > 4) continue;
            const rm = g.mark();
            defer g.resetTo(rm);
            const Ctx = struct {
                s: mlx.mlx_stream,
                x: mlx.mlx_array,
                p: root.Proj,
                fn call(cx: @This()) !mlx.mlx_array {
                    return root.linear(cx.s, cx.x, cx.p.trellis, cx.p.suh, cx.p.svh, .mul1);
                }
            };
            const ctx: Ctx = .{ .s = s, .x = try randBf16(a, &g, rnd, &.{ ci(rows), ci(c.in) }), .p = p };
            const st = try timeIt(&g, ctx, if (rows <= 16) 8 else 2);
            const flops = 2.0 * @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(c.in)) * @as(f64, @floatFromInt(c.out));
            std.debug.print("TURBOEXL3_DS_TIME linear {s} {d}x{d} rows={d} min_us={d:.1} med_us={d:.1} tflops={d:.2}\n", .{ c.name, c.in, c.out, rows, st.min, st.med, flops / (st.min * 1e6) });
        }
    }
    {
        const mark = g.mark();
        defer g.resetTo(mark);
        const p = try randProj(a, &g, rnd, wo_a.groups, wo_a.in, wo_a.out, N);
        for ([_]u32{ 1, 4, 4096 }) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const Ctx = struct {
                s: mlx.mlx_stream,
                x: mlx.mlx_array,
                p: root.Proj,
                fn call(cx: @This()) !mlx.mlx_array {
                    return root.linearGrouped(cx.s, cx.x, cx.p, .mul1);
                }
            };
            const ctx: Ctx = .{ .s = s, .x = try randBf16(a, &g, rnd, &.{ ci(rows), ci(wo_a.groups), ci(wo_a.in) }), .p = p };
            const st = try timeIt(&g, ctx, if (rows <= 16) 8 else 2);
            const flops = 2.0 * @as(f64, @floatFromInt(rows * wo_a.groups)) * @as(f64, @floatFromInt(wo_a.in)) * @as(f64, @floatFromInt(wo_a.out));
            std.debug.print("TURBOEXL3_DS_TIME grouped wo_a 8x{d}x{d} rows={d} min_us={d:.1} med_us={d:.1} tflops={d:.2}\n", .{ wo_a.in, wo_a.out, rows, st.min, st.med, flops / (st.min * 1e6) });
        }
    }
}
