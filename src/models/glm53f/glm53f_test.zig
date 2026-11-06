//! GLM-5.3-Flash's dense EXL3 linears on turboexl3 (turboderp/GLM-5.3-Flash-exl3's self-calibrated branches sc_2.10bpw,
//! sc_2.60bpw, sc_3.10bpw and sc_4.10bpw, exllamav3 1.6, MUL1). Every linear of those packs that is not a routed expert:
//! the KDA layers' joined `qkv_proj` and `o_proj`, the DSA layers' `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa` and
//! `o_proj`, the indexer's `wq_b`, the shared expert, the three dense layers' MLP, the MTP layer's `eh_proj` (and its own
//! DSA and shared expert), the 6-bit `lm_head`, and the vision tower's blocks (biased) and merger. Each shape runs at
//! every K the four packs store it at (the packs' tensor headers, `rates` below).
//! - CPU (always): each shape is a width the dense kernels admit at each of its rates (in and out whole 128-blocks, the
//!   GEMV's K split whole four-tile parts), and its trellis shape [in / 16, out / 16, 16 K] reads back as that rate.
//! - GPU (TURBOEXL3_GPU=1, else skipped): `root.linear` (`root.linearBias` for the tower's biased linears) at each shape
//!   and rate against the host reference (`ref.Linear`: `format.project`'s values on `ref.synthLinear`'s weights, plus
//!   the bias), at rows 1, 4, 16, 32 and 512 (DEC-ONE, the GEMV chain's two chunks, the register path, dequantize +
//!   matmul); the head at rows 1 and 4 only (its host reference holds 634M weights). The bars are DeepSeek-V4.1's trunk
//!   test's: f32 x and y within 5e-3 of the reference rms, bf16 x and y (rows 4 and 512) within 2.5e-2; up to 16 rows
//!   every row is compared, past that six. TURBOEXL3_GLM_K (a comma list of K, default every rate) narrows the rates.
//!   Lines: TURBOEXL3_GLM_PARITY per case, rate, row count and dtype.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const dense = @import("../../core/dense.zig");

const testing = std.testing;

/// The integer K a linear is stored at: K bits a weight, 16 K halfwords a 256-weight tile.
const K = u32;

const Case = struct { name: []const u8, in: u32, out: u32, rates: []const K, bias: bool = false, head: bool = false };

/// Every distinct shape a GLM-5.3-Flash SC pack stores as EXL3 outside the routed experts, with the K values the four
/// packs store it at: the trunk's attention at 5 (2.10), 6 (2.60, 3.10) and 8 (4.10); the MTP layer's attention and
/// shared expert at 3 (2.10), 4 (2.60, 3.10) and 6 (4.10), its `eh_proj` at 3, 4 and 5; the trunk's shared expert at 6
/// and 8; the dense MLP at 5 and 6; the head and the vision tower at 6.
pub const cases = [_]Case{
    .{ .name = "kda.qkv_proj", .in = 4096, .out = 24576, .rates = &.{ 5, 6, 8 } },
    .{ .name = "kda.o_proj", .in = 8192, .out = 4096, .rates = &.{ 5, 6, 8 } },
    .{ .name = "dsa.q_a_proj", .in = 4096, .out = 1536, .rates = &.{ 3, 4, 5, 6, 8 } },
    .{ .name = "dsa.q_b_proj", .in = 1536, .out = 16384, .rates = &.{ 3, 4, 5, 6, 8 } },
    .{ .name = "dsa.kv_a_proj_with_mqa", .in = 4096, .out = 512, .rates = &.{ 3, 4, 5, 6, 8 } },
    .{ .name = "dsa.o_proj", .in = 16384, .out = 4096, .rates = &.{ 3, 4, 5, 6, 8 } },
    .{ .name = "indexer.wq_b", .in = 1536, .out = 4096, .rates = &.{ 3, 4, 5, 6, 8 } },
    .{ .name = "shared_experts.gate_proj/up_proj", .in = 4096, .out = 2048, .rates = &.{ 3, 4, 6, 8 } },
    .{ .name = "shared_experts.down_proj", .in = 2048, .out = 4096, .rates = &.{ 3, 4, 6, 8 } },
    .{ .name = "mlp.gate_proj/up_proj (dense layers)", .in = 4096, .out = 12288, .rates = &.{ 5, 6 } },
    .{ .name = "mlp.down_proj (dense layers)", .in = 12288, .out = 4096, .rates = &.{ 5, 6 } },
    .{ .name = "mtp.eh_proj", .in = 8192, .out = 4096, .rates = &.{ 3, 4, 5 } },
    .{ .name = "lm_head", .in = 4096, .out = 154880, .rates = &.{6}, .head = true },
    .{ .name = "visual.attn.q/k/v_proj/proj", .in = 1024, .out = 1024, .rates = &.{6}, .bias = true },
    .{ .name = "visual.mlp.gate_proj/up_proj", .in = 1024, .out = 4096, .rates = &.{6}, .bias = true },
    .{ .name = "visual.mlp.down_proj", .in = 4096, .out = 1024, .rates = &.{6}, .bias = true },
    .{ .name = "visual.merger.proj", .in = 4096, .out = 4096, .rates = &.{6} },
    .{ .name = "visual.merger.gate_proj/up_proj", .in = 4096, .out = 10240, .rates = &.{6} },
    .{ .name = "visual.merger.down_proj", .in = 10240, .out = 4096, .rates = &.{6} },
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

test "turboexl3 glm53f: every GLM-5.3-Flash dense EXL3 shape is a width the dense kernels admit at each of its rates, its trellis reading back as that rate" {
    var seen: usize = 0;
    for (cases) |c| {
        try testing.expect(c.in % 128 == 0 and c.out % 128 == 0 and (c.in / 16) % 4 == 0);
        // the GEMV's K split: whole four-tile parts in every slice
        const ks = dense.Dense.ksFor(c.in, c.out);
        try testing.expect(ks >= 1 and ((c.in / 16) / 4) % ks == 0);
        try testing.expect(c.rates.len > 0);
        for (c.rates) |k| {
            try testing.expect(k >= 3 and k <= 8);
            const rate = format.Rate.fromK(k);
            try testing.expectEqual(16 * k, rate.n);
            // the stored trellis [in / 16, out / 16, n]: its last dim is the rate `root.linear` reads
            try testing.expectEqual(16 * k, (format.Rate.fromN(16 * k) orelse return error.TestUnexpectedResult).n);
            try testing.expectEqual(16 * k, (format.kFromPackedDim(16 * k) orelse return error.TestUnexpectedResult).n);
            seen += 1;
        }
    }
    // 19 shapes, 51 (shape, rate) cases
    try testing.expectEqual(@as(usize, 19), cases.len);
    try testing.expectEqual(@as(usize, 51), seen);
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

/// The picked rows of `got` (rows of `width` values) against `want` (the picked rows' reference, in order): one
/// TURBOEXL3_GLM_PARITY line, and 1 on a failure
fn report(name: []const u8, k: K, rows: u32, xd: []const u8, width: usize, got: []const f32, pick: []const u32, want: []const f32, sel: []f32, bar: f64) usize {
    for (pick, 0..) |r, i| @memcpy(sel[i * width ..][0..width], got[r * width ..][0..width]);
    const e = ref.compare(sel[0 .. pick.len * width], want[0 .. pick.len * width]);
    const ok = std.math.isFinite(e.max_abs) and e.max_ratio <= bar;
    std.debug.print("TURBOEXL3_GLM_PARITY {s} k={d} rows={d} x={s} compared_rows={d} max_abs={e:.3} rms={e:.3} ratio={e:.3} bar={e:.1} {s}\n", .{ name, k, rows, xd, pick.len, e.max_abs, e.rms_ref, e.max_ratio, bar, if (ok) "PASS" else "FAIL" });
    return @intFromBool(!ok);
}

/// One shape at one rate through `root.linear` (`root.linearBias` where it carries a bias) at its row counts: the number
/// of failed lines
fn linearCase(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, c: Case, k: K) !usize {
    const n = 16 * k;
    var prng = std.Random.DefaultPrng.init(0x61A53F ^ (@as(u64, c.in) << 24) ^ (@as(u64, c.out) << 4) ^ k);
    const rnd = prng.random();
    var lin = try ref.synthLinear(a, rnd, 1, c.in, c.out, format.Rate.fromN(n).?, .mul1);
    defer lin.deinit(a);
    const bias = try a.alloc(u16, c.out);
    defer a.free(bias);
    for (bias) |*v| v.* = format.f32ToF16Bits(2.0 * rnd.float(f32) - 1.0);
    const mark = g.mark();
    defer g.resetTo(mark);
    // the pack stores the trellis I16; the words are the same read as U16
    const trellis = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(c.in / 16), ci(c.out / 16), ci(n) }, .int16);
    const suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(c.in)}, .float16);
    const svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(c.out)}, .float16);
    // the tower's bias (the pack's f16) in each activation dtype, as the host converts it at load
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
        failed += report(c.name, k, rows, "f32", c.out, got, pick, want, sel, 5e-3);
        // bf16 x and y, as served (one bf16 rounding of y, and of the sum, on top)
        if (rows == 4 or rows == 512) {
            const xb = try g.astype(x3, .bfloat16);
            const yb = try g.adopt(if (c.bias) try root.linearBias(s, xb, trellis, suh, svh, bias_bf, .mul1) else try root.linear(s, xb, trellis, suh, svh, .mul1));
            try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(yb));
            _ = try g.hostF32(try g.astype(yb, .float32), got);
            failed += report(c.name, k, rows, "bf16", c.out, got, pick, want, sel, 2.5e-2);
        }
    }
    return failed;
}

test "turboexl3 gpu glm53f: root.linear and root.linearBias equal the host reference at every GLM-5.3-Flash dense EXL3 shape, at every K the SC packs store it at (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    // the head's W_hat alone is 1.3 GB of host reference
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var kbuf: [8]u32 = undefined;
    const only = envList("TURBOEXL3_GLM_K", &.{}, &kbuf);
    var failed: usize = 0;
    var ran: usize = 0;
    for (cases) |c| for (c.rates) |k| {
        if (only.len > 0 and std.mem.indexOfScalar(u32, only, k) == null) continue;
        defer _ = mlx.mlx_clear_cache();
        failed += try linearCase(a, &g, s, c, k);
        ran += 1;
    };
    std.debug.print("TURBOEXL3_GLM_PARITY cases={d} failed_lines={d}\n", .{ ran, failed });
    try testing.expect(ran > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}
