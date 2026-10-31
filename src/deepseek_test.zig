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
