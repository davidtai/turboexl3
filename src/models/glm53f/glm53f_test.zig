//! GLM-5.3-Flash's dense EXL3 linears on turboexl3 (turboderp/GLM-5.3-Flash-exl3's self-calibrated branches sc_2.10bpw,
//! sc_2.60bpw, sc_3.10bpw and sc_4.10bpw, exllamav3 1.6, MUL1). Every linear of those packs that is not a routed expert:
//! the KDA layers' joined `qkv_proj` and `o_proj`, the DSA layers' `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa` and
//! `o_proj`, the indexer's `wq_b`, the shared expert, the three dense layers' MLP, the MTP layer's `eh_proj` (and its own
//! DSA and shared expert), the 6-bit `lm_head`, and the vision tower's blocks (biased) and merger. Each shape runs at
//! every K the four packs store it at (the packs' tensor headers, `rates` below).
//! - CPU (always): each shape is a width the dense kernels admit at each of its rates (in and out whole 128-blocks, the
//!   GEMV's K split whole four-tile parts), and its trellis shape [in / 16, out / 16, 16 K] reads back as that rate; the
//!   cases' rates are exactly the four packs' (`packs`, from their shard headers), and one served pack's engines fit
//!   root's cache slots (`root.dense_cache_slots`, `engine_cache_slots`, `prefill_cache_slots`).
//! - GPU (TURBOEXL3_GPU=1, else skipped): `root.linear` (`root.linearBias` for the tower's biased linears) at each shape
//!   and rate against the host reference (`ref.Linear`: `format.project`'s values on `ref.synthLinear`'s weights, plus
//!   the bias), at rows 1, 4, 16, 32 and 512 (DEC-ONE, the GEMV chain's two chunks, the register path, dequantize +
//!   matmul); the head at rows 1 and 4 only (its host reference holds 634M weights). The bars are DeepSeek-V4.1's trunk
//!   test's: f32 x and y within 5e-3 of the reference rms, bf16 x and y (rows 4 and 512) within 2.5e-2; up to 16 rows
//!   every row is compared, past that six. TURBOEXL3_GLM_K (a comma list of K, default every rate) narrows the rates.
//!   Each case's engines are freed after it (`root.releaseEngines`): the 52 configurations at two dtypes outgrow the
//!   dense cache, which one served pack's 20 fit.
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
const routed = @import("routed_test.zig");

const testing = std.testing;

/// The integer K a linear is stored at: K bits a weight, 16 K halfwords a 256-weight tile.
const K = u32;

/// Which of a pack's rates a shape is stored at (`Pack`): the trunk's attention (KDA and DSA, layers 0-44), the DSA
/// attention and indexer (the trunk's and the MTP layer's), the shared expert (the trunk's and the MTP layer's), the
/// dense layers' MLP, the MTP layer's `eh_proj`, the head, the vision tower (6 bits in every pack, not served).
const Role = enum { kda, dsa, shared, dense_mlp, eh, head, vision };

const Case = struct { name: []const u8, in: u32, out: u32, rates: []const K, role: Role, bias: bool = false, head: bool = false };

/// Every distinct shape a GLM-5.3-Flash SC pack stores as EXL3 outside the routed experts, with the K values the four
/// packs store it at (`packs`): the trunk's attention at 5 (2.10), 6 (2.60, 3.10) and 8 (4.10); the MTP layer's attention and
/// shared expert at 3 (2.10), 4 (2.60, 3.10) and 6 (4.10), its `eh_proj` at 3, 4 and 5; the trunk's shared expert at 6
/// and 8; the dense MLP at 5 and 6; the head and the vision tower at 6. 53 (shape, K) entries; `kda.o_proj` and
/// `mtp.eh_proj` are one shape (8192 x 4096) and both hold K5 (sc_2.10's attention, sc_4.10's `eh_proj`), so 52 distinct
/// (in, out, K) kernel configurations.
pub const cases = [_]Case{
    .{ .name = "kda.qkv_proj", .in = 4096, .out = 24576, .rates = &.{ 5, 6, 8 }, .role = .kda },
    .{ .name = "kda.o_proj", .in = 8192, .out = 4096, .rates = &.{ 5, 6, 8 }, .role = .kda },
    .{ .name = "dsa.q_a_proj", .in = 4096, .out = 1536, .rates = &.{ 3, 4, 5, 6, 8 }, .role = .dsa },
    .{ .name = "dsa.q_b_proj", .in = 1536, .out = 16384, .rates = &.{ 3, 4, 5, 6, 8 }, .role = .dsa },
    .{ .name = "dsa.kv_a_proj_with_mqa", .in = 4096, .out = 512, .rates = &.{ 3, 4, 5, 6, 8 }, .role = .dsa },
    .{ .name = "dsa.o_proj", .in = 16384, .out = 4096, .rates = &.{ 3, 4, 5, 6, 8 }, .role = .dsa },
    .{ .name = "indexer.wq_b", .in = 1536, .out = 4096, .rates = &.{ 3, 4, 5, 6, 8 }, .role = .dsa },
    .{ .name = "shared_experts.gate_proj/up_proj", .in = 4096, .out = 2048, .rates = &.{ 3, 4, 6, 8 }, .role = .shared },
    .{ .name = "shared_experts.down_proj", .in = 2048, .out = 4096, .rates = &.{ 3, 4, 6, 8 }, .role = .shared },
    .{ .name = "mlp.gate_proj/up_proj (dense layers)", .in = 4096, .out = 12288, .rates = &.{ 5, 6 }, .role = .dense_mlp },
    .{ .name = "mlp.down_proj (dense layers)", .in = 12288, .out = 4096, .rates = &.{ 5, 6 }, .role = .dense_mlp },
    .{ .name = "mtp.eh_proj", .in = 8192, .out = 4096, .rates = &.{ 3, 4, 5 }, .role = .eh },
    .{ .name = "lm_head", .in = 4096, .out = 154880, .rates = &.{6}, .role = .head, .head = true },
    .{ .name = "visual.attn.q/k/v_proj/proj", .in = 1024, .out = 1024, .rates = &.{6}, .role = .vision, .bias = true },
    .{ .name = "visual.mlp.gate_proj/up_proj", .in = 1024, .out = 4096, .rates = &.{6}, .role = .vision, .bias = true },
    .{ .name = "visual.mlp.down_proj", .in = 4096, .out = 1024, .rates = &.{6}, .role = .vision, .bias = true },
    .{ .name = "visual.merger.proj", .in = 4096, .out = 4096, .rates = &.{6}, .role = .vision },
    .{ .name = "visual.merger.gate_proj/up_proj", .in = 4096, .out = 10240, .rates = &.{6}, .role = .vision },
    .{ .name = "visual.merger.down_proj", .in = 10240, .out = 4096, .rates = &.{6}, .role = .vision },
};

/// One SC pack's EXL3 rates, read from its 22 or 23 shard headers (each repack's `receipts/source-headers.json.gz`,
/// 2026-11-06): the trunk's attention (KDA and DSA, the dense layers' too), the trunk's shared expert, the dense layers'
/// MLP, the MTP layer's attention and shared expert, its `eh_proj`; the routed experts in halfwords a tile: layers 4-44,
/// layer 3, the MTP layer's. The head is 6 bits in every pack.
const Pack = struct { name: []const u8, attn: K, shared: K, dense_mlp: K, mtp: K, eh: K, routed_n: u32, l3_n: u32, mtp_n: u32 };

pub const packs = [_]Pack{
    .{ .name = "sc_2.10bpw", .attn = 5, .shared = 6, .dense_mlp = 5, .mtp = 3, .eh = 3, .routed_n = 32, .l3_n = 48, .mtp_n = 48 },
    .{ .name = "sc_2.60bpw", .attn = 6, .shared = 6, .dense_mlp = 6, .mtp = 4, .eh = 4, .routed_n = 40, .l3_n = 56, .mtp_n = 64 },
    .{ .name = "sc_3.10bpw", .attn = 6, .shared = 6, .dense_mlp = 6, .mtp = 4, .eh = 4, .routed_n = 48, .l3_n = 64, .mtp_n = 64 },
    .{ .name = "sc_4.10bpw", .attn = 8, .shared = 8, .dense_mlp = 6, .mtp = 6, .eh = 5, .routed_n = 64, .l3_n = 80, .mtp_n = 64 },
};

/// The K values pack `p` stores a shape of role `r` at (two where the trunk and the MTP layer both hold it): `buf[0..n]`.
fn packRates(p: Pack, r: Role, buf: *[2]K) []const K {
    buf.* = switch (r) {
        .kda => .{ p.attn, p.attn },
        .dsa => .{ p.attn, p.mtp },
        .shared => .{ p.shared, p.mtp },
        .dense_mlp => .{ p.dense_mlp, p.dense_mlp },
        .eh => .{ p.eh, p.eh },
        .head, .vision => .{ 6, 6 },
    };
    return if (buf[0] == buf[1]) buf[0..1] else buf[0..2];
}

/// A served pack's distinct dense (in, out, K): every role but the vision tower (the plugin does not serve it).
fn packDense(p: Pack, out: *[cases.len * 2][3]u32) usize {
    var n: usize = 0;
    for (cases) |c| {
        if (c.role == .vision) continue;
        var rb: [2]K = undefined;
        for (packRates(p, c.role, &rb)) |k| {
            const key = [3]u32{ c.in, c.out, k };
            for (out[0..n]) |e| {
                if (std.mem.eql(u32, &e, &key)) break;
            } else {
                out[n] = key;
                n += 1;
            }
        }
    }
    return n;
}

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
    // 19 named shapes, 53 (shape, rate) entries (3 + 3 + 5 x 5 + 4 + 4 + 2 + 2 + 3 + 7 x 1), 52 distinct (in, out, K):
    // kda.o_proj's K5 and mtp.eh_proj's K5 are one configuration
    try testing.expectEqual(@as(usize, 19), cases.len);
    try testing.expectEqual(@as(usize, 53), seen);
    var keys: [cases.len * 8][3]u32 = undefined;
    var distinct: usize = 0;
    for (cases) |c| for (c.rates) |k| {
        const key = [3]u32{ c.in, c.out, k };
        for (keys[0..distinct]) |e| {
            if (std.mem.eql(u32, &e, &key)) break;
        } else {
            keys[distinct] = key;
            distinct += 1;
        }
    };
    try testing.expectEqual(@as(usize, 52), distinct);
}

test "turboexl3 glm53f: the dense shapes' rates are exactly the four SC packs' (their shard headers), and one served pack's engines fit turboexl3's caches" {
    // every rate a case holds is one a pack stores it at, and every rate a pack stores a shape at is in the case
    for (cases) |c| {
        for (c.rates) |k| {
            var stored = false;
            for (packs) |p| {
                var rb: [2]K = undefined;
                if (std.mem.indexOfScalar(K, packRates(p, c.role, &rb), k) != null) stored = true;
            }
            if (!stored) std.debug.print("glm53f: {s} holds K{d}, which no SC pack stores it at\n", .{ c.name, k });
            try testing.expect(stored);
        }
        for (packs) |p| {
            var rb: [2]K = undefined;
            for (packRates(p, c.role, &rb)) |k| {
                if (std.mem.indexOfScalar(K, c.rates, k) == null) std.debug.print("glm53f: {s} stores {s} at K{d}, not a case\n", .{ p.name, c.name, k });
                try testing.expect(std.mem.indexOfScalar(K, c.rates, k) != null);
            }
        }
    }
    for (packs) |p| {
        // the trunk, the head and the MTP layer: 20 distinct dense (in, out, K) in every pack (its shard headers' count,
        // the vision tower left out); the plugin's activations are bf16, so 20 of the dense cache's slots, and 40 at two
        // activation dtypes
        var keys: [cases.len * 2][3]u32 = undefined;
        const n = packDense(p, &keys);
        try testing.expectEqual(@as(usize, 20), n);
        try testing.expect(2 * n <= root.dense_cache_slots);
        // the routed rates are routed_test.zig's
        for ([_]u32{ p.routed_n, p.l3_n, p.mtp_n }) |rn| try testing.expect(std.mem.indexOfScalar(u32, &routed.rates_n, rn) != null);
        // the decode engines (`decode.Kernels3`, one per rate, top-k, clamp and output dtype): the banked decode's top-1 f32
        // at the two routed rates, the MTP layer's resident bank at top-8 at its rate in two activation dtypes
        try testing.expect(p.routed_n != p.l3_n);
        try testing.expect(2 + 2 <= root.engine_cache_slots);
        // the prefill engines (one per rate, top-k and expert count): the prompt calls' top-1 over each routed rate's base
        // and transient slot banks (the prompt phase's rows are one count across the layers), the MTP layer's top-8 over
        // its 288 experts
        try testing.expect(2 * 2 + 1 <= root.prefill_cache_slots);
    }
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
        // 52 configurations at two dtypes outgrow the dense cache's slots, which a served pack's 20 fit: each case's
        // engines are freed after it (its arrays were read back, so no call is in flight)
        defer root.releaseEngines();
        failed += try linearCase(a, &g, s, c, k);
        ran += 1;
    };
    std.debug.print("TURBOEXL3_GLM_PARITY cases={d} failed_lines={d}\n", .{ ran, failed });
    try testing.expect(ran > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}
