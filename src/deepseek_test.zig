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
pub const N: u32 = 96;

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
pub const wo_a = struct {
    pub const groups: u32 = 8;
    pub const in: u32 = 4096;
    pub const out: u32 = 1024;
};

/// DEC-ONE (1, 4), the GEMV chain's two chunks (16), the register path (32), dequantize + matmul (512)
const row_counts = [_]u32{ 1, 4, 16, 32, 512 };
const head_rows = [_]u32{ 1, 4 };

pub fn ci(x: anytype) c_int {
    return @intCast(x);
}

pub fn bf16Exact(v: f32) f32 {
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

pub fn nowNs() u64 {
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

/// One projection's host values (or `lead` stacked ones), drawn as `synthBank` scales them: every trellis pattern a
/// codeword, suh in [0.5, 1.5], svh in [0.5, 1.5] / sqrt(in). `upload` copies them into new device arrays, each call its
/// own buffers, so a chain of layers reads each layer's weights from its own memory as a forward does.
pub const HostProj = struct {
    lead: ?u32,
    in: u32,
    out: u32,
    n: u32,
    t: []u16,
    su: []u16,
    sv: []u16,

    pub fn draw(a: std.mem.Allocator, rnd: std.Random, lead: ?u32, in: u32, out: u32, n: u32) !HostProj {
        const L: usize = lead orelse 1;
        const t = try a.alloc(u16, L * (in / 16) * (out / 16) * n);
        errdefer a.free(t);
        rnd.bytes(std.mem.sliceAsBytes(t));
        const su = try a.alloc(u16, L * in);
        errdefer a.free(su);
        for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
        const sv = try a.alloc(u16, L * out);
        for (sv) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
        return .{ .lead = lead, .in = in, .out = out, .n = n, .t = t, .su = su, .sv = sv };
    }

    pub fn deinit(h: *HostProj, a: std.mem.Allocator) void {
        a.free(h.t);
        a.free(h.su);
        a.free(h.sv);
    }

    pub fn upload(h: *const HostProj, g: *G) !root.Proj {
        if (h.lead) |e| return .{
            .trellis = try g.hostArray(std.mem.sliceAsBytes(h.t), &.{ ci(e), ci(h.in / 16), ci(h.out / 16), ci(h.n) }, .uint16),
            .suh = try g.hostArray(std.mem.sliceAsBytes(h.su), &.{ ci(e), ci(h.in) }, .float16),
            .svh = try g.hostArray(std.mem.sliceAsBytes(h.sv), &.{ ci(e), ci(h.out) }, .float16),
        };
        return .{
            .trellis = try g.hostArray(std.mem.sliceAsBytes(h.t), &.{ ci(h.in / 16), ci(h.out / 16), ci(h.n) }, .uint16),
            .suh = try g.hostArray(std.mem.sliceAsBytes(h.su), &.{ci(h.in)}, .float16),
            .svh = try g.hostArray(std.mem.sliceAsBytes(h.sv), &.{ci(h.out)}, .float16),
        };
    }
};

/// A projection's random operands on the device (`HostProj`, drawn and uploaded once).
pub fn randProj(a: std.mem.Allocator, g: *G, rnd: std.Random, lead: ?u32, in: u32, out: u32, n: u32) !root.Proj {
    var h = try HostProj.draw(a, rnd, lead, in, out, n);
    defer h.deinit(a);
    return h.upload(g);
}

pub fn randBf16(a: std.mem.Allocator, g: *G, rnd: std.Random, shape: []const c_int) !mlx.mlx_array {
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

/// DeepSeek-V4.1's layers that run the compressor (its KV-source layers) and the indexer (its index-source layers), and the
/// SC 2.81 pack's K2 routed layers
pub const comp_layers = [_]u32{ 2, 8, 14, 20 };
pub const index_layers = [_]u32{ 2, 8, 14, 20, 24, 28, 32, 36 };
const k2_layers = [_]u32{ 23, 32, 33, 34, 35, 36, 37, 38, 39 };

/// One layer's weights of the decode chain, each in its own device buffers
pub const ChainLayer = struct {
    wq_a: root.Proj,
    wq_b: root.Proj,
    wkv: root.Proj,
    cwkv: ?root.Proj,
    cwgate: ?root.Proj,
    iwq_b: ?root.Proj,
    wo_a: root.Proj,
    wo_b: root.Proj,
    sw1: root.Proj,
    sw3: root.Proj,
    sw2: root.Proj,
    bank: root.Bank,
};

/// What a chained forward runs in each layer: the trunk alone, the routed MoE alone, or both (and then the head)
pub const ChainArm = enum { trunk, routed, layer };

fn chainLin(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj) !mlx.mlx_array {
    return g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
}

/// One decode forward of `rows` rows through `layers`, chained through the residual as the model's layers are: the
/// trunk in a layer's order (wq_a then wq_b, wkv, the compressor's wkv and wgate, the indexer's wq_b on wq_a's output,
/// the grouped wo_a on q reshaped as the attention output it stands in for, wo_b, then the shared expert: w1 and w3, the
/// SwiGLU, w2), then the routed MoE (moeClamped at limit 10 on fixed ids). The attention core, the hyper-connections, the
/// norms, the router and Engram are not in it.
pub fn chainForward(g: *G, s: mlx.mlx_stream, layers: []const ChainLayer, inds: []const mlx.mlx_array, sc: []const mlx.mlx_array, x: mlx.mlx_array, rows: u32, arm: ChainArm, head: ?root.Proj) !mlx.mlx_array {
    const R = ci(rows);
    var h = x;
    for (layers, 0..) |ly, L| {
        if (arm != .routed) {
            const qa = try chainLin(g, s, h, ly.wq_a);
            const q = try chainLin(g, s, qa, ly.wq_b);
            var extra = try g.mean(try chainLin(g, s, h, ly.wkv), 1, true);
            if (ly.cwkv) |p| extra = try g.add(extra, try g.mean(try chainLin(g, s, h, p), 1, true));
            if (ly.cwgate) |p| extra = try g.add(extra, try g.mean(try chainLin(g, s, h, p), 1, true));
            if (ly.iwq_b) |p| extra = try g.add(extra, try g.mean(try chainLin(g, s, qa, p), 1, true));
            const oa = try g.adopt(try root.linearGrouped(s, try g.reshape(q, &.{ R, 8, 4096 }), ly.wo_a, .mul1));
            const ob = try chainLin(g, s, try g.reshape(oa, &.{ R, 8192 }), ly.wo_b);
            h = try g.add(try g.add(h, ob), extra);
            const g1 = try chainLin(g, s, h, ly.sw1);
            const up = try chainLin(g, s, h, ly.sw3);
            h = try g.add(h, try chainLin(g, s, try g.mul(try g.mul(try g.sigmoid(g1), g1), up), ly.sw2));
        }
        if (arm != .trunk) {
            const r = try g.adopt(try root.moeClamped(s, try g.reshape(h, &.{ 1, R, 5120 }), ly.bank, inds[L], sc[L], .mul1, 10));
            h = try g.add(h, try g.reshape(r, &.{ R, 5120 }));
        }
    }
    if (head) |p| return chainLin(g, s, h, p);
    return h;
}

// DeepSeek-V4.1's decode forward as one evaluation, for the serving plan (TURBOEXL3_DS_TIME=1 with TURBOEXL3_GPU=1; it
// asserts nothing): `chainForward` over 40 layers, each with its own weights in its own buffers (one draw per shape, copied
// per layer), the routed bank of each layer 48 experts at K3 (K2 on the pack's K2 layers), its ids drawn without repeats.
// Arms: the trunk alone, the routed MoE alone, the whole layer plus the head; rows 1, 4 and 6 (a serial step and DSpark's
// verify widths). Each line: the token's min and median ms (host graph build plus evaluation, as a decode step pays it),
// the per-layer us at the min, and the median host graph-build ms.
test "turboexl3 gpu deepseek timing: the decode forward's 40 layers chained in one evaluation (TURBOEXL3_DS_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_DS_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xD5C4A1);
    const rnd = prng.random();
    const nl = 40;
    const bank_e: u32 = 48;
    const topk: u32 = 6;
    var hp = [_]HostProj{
        try HostProj.draw(a, rnd, null, 5120, 1280, N), // 0 wq_a
        try HostProj.draw(a, rnd, null, 1280, 32768, N), // 1 wq_b
        try HostProj.draw(a, rnd, null, 5120, 512, N), // 2 wkv, the compressor's wkv and wgate
        try HostProj.draw(a, rnd, null, 1280, 4096, N), // 3 the indexer's wq_b
        try HostProj.draw(a, rnd, wo_a.groups, wo_a.in, wo_a.out, N), // 4 wo_a
        try HostProj.draw(a, rnd, null, 8192, 5120, N), // 5 wo_b
        try HostProj.draw(a, rnd, null, 5120, 2304, N), // 6 the shared w1 and w3
        try HostProj.draw(a, rnd, null, 2304, 5120, N), // 7 the shared w2
        try HostProj.draw(a, rnd, bank_e, 5120, 2304, 48), // 8 K3 gate and up
        try HostProj.draw(a, rnd, bank_e, 2304, 5120, 48), // 9 K3 down
        try HostProj.draw(a, rnd, bank_e, 5120, 2304, 32), // 10 K2 gate and up
        try HostProj.draw(a, rnd, bank_e, 2304, 5120, 32), // 11 K2 down
        try HostProj.draw(a, rnd, null, 5120, 129280, N), // 12 the head
    };
    defer for (&hp) |*h| h.deinit(a);
    var layers: [nl]ChainLayer = undefined;
    for (&layers, 0..) |*ly, li| {
        const L: u32 = @intCast(li);
        const k2 = std.mem.indexOfScalar(u32, &k2_layers, L) != null;
        const comp = std.mem.indexOfScalar(u32, &comp_layers, L) != null;
        const idx = std.mem.indexOfScalar(u32, &index_layers, L) != null;
        ly.* = .{
            .wq_a = try hp[0].upload(&g),
            .wq_b = try hp[1].upload(&g),
            .wkv = try hp[2].upload(&g),
            .cwkv = if (comp) try hp[2].upload(&g) else null,
            .cwgate = if (comp) try hp[2].upload(&g) else null,
            .iwq_b = if (idx) try hp[3].upload(&g) else null,
            .wo_a = try hp[4].upload(&g),
            .wo_b = try hp[5].upload(&g),
            .sw1 = try hp[6].upload(&g),
            .sw3 = try hp[6].upload(&g),
            .sw2 = try hp[7].upload(&g),
            .bank = .{ .gate = try hp[if (k2) 10 else 8].upload(&g), .up = try hp[if (k2) 10 else 8].upload(&g), .down = try hp[if (k2) 11 else 9].upload(&g) },
        };
    }
    const head = try hp[12].upload(&g);
    for ([_]u32{ 1, 4, 6 }) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        var inds: [nl]mlx.mlx_array = undefined;
        var sc: [nl]mlx.mlx_array = undefined;
        const ib = try a.alloc(i32, rows * topk);
        defer a.free(ib);
        const sb = try a.alloc(f32, rows * topk);
        defer a.free(sb);
        for (0..nl) |L| {
            for (0..rows) |r| {
                var used: u64 = 0;
                for (0..topk) |k| {
                    var e: u32 = rnd.uintLessThan(u32, bank_e);
                    while ((used >> @as(u6, @intCast(e))) & 1 != 0) e = rnd.uintLessThan(u32, bank_e);
                    used |= @as(u64, 1) << @as(u6, @intCast(e));
                    ib[r * topk + k] = @intCast(e);
                    sb[r * topk + k] = bf16Exact(1.0 / @as(f32, @floatFromInt(topk)));
                }
            }
            inds[L] = try g.hostArray(std.mem.sliceAsBytes(ib), &.{ 1, ci(rows), ci(topk) }, .int32);
            sc[L] = try g.astype(try g.hostArray(std.mem.sliceAsBytes(sb), &.{ 1, ci(rows), ci(topk) }, .float32), .bfloat16);
        }
        const x = try randBf16(a, &g, rnd, &.{ ci(rows), 5120 });
        for ([_]ChainArm{ .trunk, .routed, .layer }) |arm| {
            var build: [5]f64 = undefined;
            var total: [5]f64 = undefined;
            for (0..build.len + 2) |round| {
                const fm = g.mark();
                defer g.resetTo(fm);
                const t0 = nowNs();
                const y = try chainForward(&g, s, &layers, &inds, &sc, x, rows, arm, if (arm == .layer) head else null);
                const t1 = nowNs();
                try g.evalAll(&.{y});
                const t2 = nowNs();
                if (round >= 2) {
                    build[round - 2] = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
                    total[round - 2] = @as(f64, @floatFromInt(t2 - t0)) / 1000.0;
                }
            }
            std.mem.sort(f64, &build, {}, std.sort.asc(f64));
            std.mem.sort(f64, &total, {}, std.sort.asc(f64));
            std.debug.print("TURBOEXL3_DS_CHAIN arm={s} rows={d} token_min_ms={d:.2} token_med_ms={d:.2} per_layer_us={d:.1} graph_build_med_ms={d:.2}{s}\n", .{ @tagName(arm), rows, total[0] / 1000.0, total[total.len / 2] / 1000.0, total[0] / @as(f64, nl), build[build.len / 2] / 1000.0, if (arm == .layer) " (the head included)" else "" });
        }
    }
    // TURBOEXL3_DS_SWEEP=1: the routed decode forms, then the dense options, on chains of 40 layers at rows 1, 4 and 6. The
    // routed chain runs over 384-expert banks (two K3 banks alternating, one K2 bank on the K2 layers), each row's top-6
    // by Gumbel top-k over a per-layer lognormal popularity (sigma 0: uniform; 1.5: the repeats captured routing shows),
    // through root.moe: forms 5 and 6 (one decode per expert among the row-slots that chose it) refuse a clamp, and the
    // clamp moves no bytes. Each line: the token's min ms and the experts per layer the rows chose.
    if (std.c.getenv("TURBOEXL3_DS_SWEEP") == null) return;
    const full_e: u32 = 384;
    var bh = [_]HostProj{
        try HostProj.draw(a, rnd, full_e, 5120, 2304, 48),
        try HostProj.draw(a, rnd, full_e, 2304, 5120, 48),
        try HostProj.draw(a, rnd, full_e, 5120, 2304, 32),
        try HostProj.draw(a, rnd, full_e, 2304, 5120, 32),
    };
    defer for (&bh) |*h| h.deinit(a);
    var k3b: [2]root.Bank = undefined;
    for (&k3b) |*b| b.* = .{ .gate = try bh[0].upload(&g), .up = try bh[0].upload(&g), .down = try bh[1].upload(&g) };
    const k2b: root.Bank = .{ .gate = try bh[2].upload(&g), .up = try bh[2].upload(&g), .down = try bh[3].upload(&g) };
    defer root.decode_form_override.get().* = null;
    const logw = try a.alloc(f64, full_e);
    defer a.free(logw);
    const keys = try a.alloc(f64, full_e);
    defer a.free(keys);
    for ([_]f64{ 0.0, 1.5 }) |skew| for ([_]u32{ 1, 4, 6 }) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        var inds: [nl]mlx.mlx_array = undefined;
        var sc: [nl]mlx.mlx_array = undefined;
        const ib = try a.alloc(i32, rows * topk);
        defer a.free(ib);
        const sb = try a.alloc(f32, rows * topk);
        defer a.free(sb);
        var distinct: usize = 0;
        for (0..nl) |L| {
            for (logw) |*w| w.* = skew * rnd.floatNorm(f64);
            var seen: [6]u64 = @splat(0); // which of the 384 experts the rows chose
            for (0..rows) |r| {
                for (keys, logw) |*kv, lw| kv.* = lw - @log(-@log(@max(rnd.float(f64), 1e-300)));
                for (0..topk) |k| {
                    var best: usize = 0;
                    for (keys, 0..) |kv, e| {
                        if (kv > keys[best]) best = e;
                    }
                    keys[best] = -std.math.inf(f64);
                    seen[best / 64] |= @as(u64, 1) << @as(u6, @intCast(best % 64));
                    ib[r * topk + k] = @intCast(best);
                    sb[r * topk + k] = bf16Exact(1.0 / @as(f32, @floatFromInt(topk)));
                }
            }
            for (seen) |w| distinct += @popCount(w);
            inds[L] = try g.hostArray(std.mem.sliceAsBytes(ib), &.{ 1, ci(rows), ci(topk) }, .int32);
            sc[L] = try g.astype(try g.hostArray(std.mem.sliceAsBytes(sb), &.{ 1, ci(rows), ci(topk) }, .float32), .bfloat16);
        }
        const x = try randBf16(a, &g, rnd, &.{ ci(rows), 5120 });
        for ([_]u8{ 3, 5, 6, 7, 8, 9 }) |form| {
            root.decode_form_override.get().* = form;
            var total: [5]f64 = undefined;
            for (0..total.len + 2) |round| {
                const fm = g.mark();
                defer g.resetTo(fm);
                const t0 = nowNs();
                var h = x;
                for (0..nl) |L| {
                    const k2 = std.mem.indexOfScalar(u32, &k2_layers, @as(u32, @intCast(L))) != null;
                    const bank = if (k2) k2b else k3b[L % 2];
                    const r = try g.adopt(try root.moe(s, try g.reshape(h, &.{ 1, ci(rows), 5120 }), bank, inds[L], sc[L], .mul1, false));
                    h = try g.add(h, try g.reshape(r, &.{ ci(rows), 5120 }));
                }
                try g.evalAll(&.{h});
                if (round >= 2) total[round - 2] = @as(f64, @floatFromInt(nowNs() - t0)) / 1000.0;
            }
            std.mem.sort(f64, &total, {}, std.sort.asc(f64));
            std.debug.print("TURBOEXL3_DS_SWEEP routed skew={d:.1} form={d} rows={d} experts_per_layer={d:.1} token_min_ms={d:.2} per_layer_us={d:.1}\n", .{ skew, form, rows, @as(f64, @floatFromInt(distinct)) / @as(f64, nl), total[0] / 1000.0, total[0] / @as(f64, nl) });
        }
        root.decode_form_override.get().* = null;
    };
    defer root.configure(.{}) catch {};
    for ([_][2]bool{ .{ true, true }, .{ false, true }, .{ true, false }, .{ false, false } }) |opt| {
        try root.configure(.{ .dec_one = opt[0], .dense_lean = opt[1] });
        for ([_]u32{ 1, 4, 6 }) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const x = try randBf16(a, &g, rnd, &.{ ci(rows), 5120 });
            var total: [5]f64 = undefined;
            for (0..total.len + 2) |round| {
                const fm = g.mark();
                defer g.resetTo(fm);
                const t0 = nowNs();
                const y = try chainForward(&g, s, &layers, &.{}, &.{}, x, rows, .trunk, null);
                try g.evalAll(&.{y});
                if (round >= 2) total[round - 2] = @as(f64, @floatFromInt(nowNs() - t0)) / 1000.0;
            }
            std.mem.sort(f64, &total, {}, std.sort.asc(f64));
            std.debug.print("TURBOEXL3_DS_SWEEP trunk dec_one={} dense_lean={} rows={d} token_min_ms={d:.2} per_layer_us={d:.1}\n", .{ opt[0], opt[1], rows, total[0] / 1000.0, total[0] / @as(f64, nl) });
        }
    }
    // the trunk alone across the dense paths' row ranges (DEC-ONE to 4, the multi-row GEMV to 16, the register-path NAX GEMM
    // to 32): whether a verify width's trunk is cheaper on the next path up
    try root.configure(.{});
    for ([_]u32{ 6, 8, 12, 16, 17, 20, 24, 32 }) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        const x = try randBf16(a, &g, rnd, &.{ ci(rows), 5120 });
        var total: [5]f64 = undefined;
        for (0..total.len + 2) |round| {
            const fm = g.mark();
            defer g.resetTo(fm);
            const t0 = nowNs();
            const y = try chainForward(&g, s, &layers, &.{}, &.{}, x, rows, .trunk, null);
            try g.evalAll(&.{y});
            if (round >= 2) total[round - 2] = @as(f64, @floatFromInt(nowNs() - t0)) / 1000.0;
        }
        std.mem.sort(f64, &total, {}, std.sort.asc(f64));
        std.debug.print("TURBOEXL3_DS_SWEEP trunk_rows rows={d} token_min_ms={d:.2} per_layer_us={d:.1}\n", .{ rows, total[0] / 1000.0, total[0] / @as(f64, nl) });
    }
}

// gu8 (the gate|up launch with the prepare inside: form 8, and form 9's one-row arm) at DeepSeek-V4.1's 5120-wide hidden:
// from two rows its threadgroup memory would pass Metal's 32 KB (43,008 bytes at GROUP 2, SPT 4) and form 8 failed to load
// its kernel; gu8Admitted sends those rows to prep2. Forms 7, 8 and 9 give form 3's bytes unclamped, and forms 8 and 9 give
// form 7's clamped (a clamped call refuses form 3), at rows 1 to 6.
test "turboexl3 gpu deepseek: decode forms 7, 8 and 9 give form 3's bytes at DeepSeek-V4.1's routed geometry, rows 1 to 6, clamped and not (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer root.decode_form_override.get().* = null;
    var prng = std.Random.DefaultPrng.init(0xD5F08);
    const rnd = prng.random();
    const experts: u32 = 48;
    const topk: u32 = 6;
    const bank: root.Bank = .{
        .gate = try randProj(a, &g, rnd, experts, 5120, 2304, 48),
        .up = try randProj(a, &g, rnd, experts, 5120, 2304, 48),
        .down = try randProj(a, &g, rnd, experts, 2304, 5120, 48),
    };
    var failed = false;
    for (1..7) |ri| {
        const rows: u32 = @intCast(ri);
        const rm = g.mark();
        defer g.resetTo(rm);
        const ib = try a.alloc(i32, rows * topk);
        defer a.free(ib);
        const sb = try a.alloc(f32, rows * topk);
        defer a.free(sb);
        for (0..rows) |r| {
            var used: u64 = 0;
            for (0..topk) |k| {
                var e: u32 = rnd.uintLessThan(u32, experts);
                while ((used >> @as(u6, @intCast(e))) & 1 != 0) e = rnd.uintLessThan(u32, experts);
                used |= @as(u64, 1) << @as(u6, @intCast(e));
                ib[r * topk + k] = @intCast(e);
                sb[r * topk + k] = bf16Exact(0.05 + 0.3 * rnd.float(f32));
            }
        }
        const x = try randBf16(a, &g, rnd, &.{ 1, ci(rows), 5120 });
        const inds = try g.hostArray(std.mem.sliceAsBytes(ib), &.{ 1, ci(rows), ci(topk) }, .int32);
        const sc = try g.astype(try g.hostArray(std.mem.sliceAsBytes(sb), &.{ 1, ci(rows), ci(topk) }, .float32), .bfloat16);
        const want = try a.alloc(f32, rows * 5120);
        defer a.free(want);
        const got = try a.alloc(f32, rows * 5120);
        defer a.free(got);
        for ([_]c_int{ 0, 10 }) |limit| {
            const forms: []const u8 = if (limit == 0) &.{ 3, 7, 8, 9 } else &.{ 7, 8, 9 };
            for (forms, 0..) |form, fi| {
                root.decode_form_override.get().* = form;
                const y = try g.adopt(if (limit == 0) try root.moe(s, x, bank, inds, sc, .mul1, false) else try root.moeClamped(s, x, bank, inds, sc, .mul1, limit));
                _ = try g.hostF32(try g.astype(y, .float32), if (fi == 0) want else got);
                if (fi == 0) continue;
                var differing: usize = 0;
                for (got, want) |p, q| differing += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
                std.debug.print("TURBOEXL3_DS_FORMS rows={d} limit={d} form={d} vs form={d} differing={d}/{d} {s}\n", .{ rows, limit, form, forms[0], differing, got.len, if (differing == 0) "PASS" else "FAIL" });
                if (differing != 0) failed = true;
            }
        }
    }
    try testing.expect(!failed);
}
