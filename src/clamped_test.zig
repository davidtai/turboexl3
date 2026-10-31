//! The clamped SwiGLU (`root.moeClamped`: DeepSeek-V4's and GLM's `swiglu_limit`). CPU (always): each clamped kernel text
//! is its base text with the SwiGLU line swapped for the clamp piece and nothing else; the reference clamp's rule; the
//! refusals that need no GPU. GPU (TURBOEXL3_GPU=1, else skipped): at a limit no value reaches, every clamped launch gives
//! its unclamped launch's bytes (`root` at decode forms 3 and 5 to 9 and at prefill rows, and the prefill mids `root` does
//! not serve); and at DeepSeek-V4.1-Flash's routed geometry (hidden 5120, expert width 2304, 384 experts, top-6, MUL1 at
//! K2, K3 and K6, swiglu_limit 10) `root.moeClamped` at decode rows 1, 4, 16 and prefill rows 32, 1328 against
//! `ref.moeRowClamped` on sampled rows, the inputs scaled so that a tenth of the gates and a quarter of the ups pass the
//! limit. TURBOEXL3_CLAMP_N (rates in halfwords per tile, a comma list; default 32,48,96) and TURBOEXL3_CLAMP_ROWS
//! (default 1,4,16,32,1328) narrow the second; one rate's bank is held twice (host and device): 2 x 3.4 GB at K2, 2 x 5.1 GB
//! at K3, 2 x 10.2 GB at K6, and the reference (it decodes every weight of a sampled row's experts on the CPU) takes most of
//! the time. Lines: TURBOEXL3_CLAMP_BYTES, TURBOEXL3_CLAMP_PARITY.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("test_support.zig");
const G = support.Graph;
const format = @import("format.zig");
const decode = @import("decode.zig");
const prefill = @import("prefill.zig");
const ref = @import("ref.zig");
const root = @import("root.zig");

const testing = std.testing;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// `v` rounded to bf16 (as MLX rounds), back in f32: a value both sides read alike.
fn bfExact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
}

test "turboexl3 clamped texts: each clamped kernel is its base text with the SwiGLU line swapped for the clamp piece, nothing else; only the clamp reads LIMIT" {
    const fin7_line = @embedFile("kernels/decode/fin7_swiglu.metal");
    const fin7_clamp = @embedFile("kernels/decode/fin7_swiglu_clamp.metal");
    const mid_line = @embedFile("kernels/prefill/mid_swiglu.metal");
    const mid_clamp = @embedFile("kernels/prefill/mid_swiglu_clamp.metal");
    const Pair = struct { base: []const u8, clamped: []const u8, line: []const u8, clamp: []const u8 };
    const pairs = [_]Pair{
        .{ .base = decode.text_fin7, .clamped = decode.text_fin7c, .line = fin7_line, .clamp = fin7_clamp },
        .{ .base = prefill.text_mid, .clamped = prefill.text_midc, .line = mid_line, .clamp = mid_clamp },
        .{ .base = prefill.text_mid_v, .clamped = prefill.text_mid_vc, .line = mid_line, .clamp = mid_clamp },
        .{ .base = prefill.text_mid_t128, .clamped = prefill.text_mid_t128c, .line = mid_line, .clamp = mid_clamp },
    };
    for (pairs) |p| {
        // the base holds the SwiGLU line once, the clamped text the clamp piece once in its place and the line nowhere
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, p.base, p.line));
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, p.clamped, p.clamp));
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, p.clamped, p.line));
        const i = std.mem.indexOf(u8, p.base, p.line).?;
        try testing.expectEqual(i, std.mem.indexOf(u8, p.clamped, p.clamp).?);
        try testing.expectEqualStrings(p.base[0..i], p.clamped[0..i]);
        try testing.expectEqualStrings(p.base[i + p.line.len ..], p.clamped[i + p.clamp.len ..]);
        // an unclamped kernel takes no LIMIT (its launch passes the constants it always did)
        try testing.expect(std.mem.indexOf(u8, p.base, "LIMIT") == null);
    }
    for ([_][]const u8{ fin7_clamp, mid_clamp }) |piece| try testing.expect(std.mem.indexOf(u8, piece, "LIMIT") != null);
}

test "turboexl3 clamped reference: the gate clamps from above only and up both ways, before the SiLU; a NaN passes; limit 0 and a limit above every value are moeRowDec" {
    const silu = struct {
        fn f(v: f32) f32 {
            return v / (1.0 + @exp(-v));
        }
    }.f;
    const tol: f32 = 1e-6;
    try testing.expectApproxEqRel(silu(10) * 10, ref.swigluClamped(20, 20, 10), tol);
    try testing.expectApproxEqRel(silu(10) * -10, ref.swigluClamped(30, -12, 10), tol);
    try testing.expectApproxEqRel(silu(-20) * 3, ref.swigluClamped(-20, 3, 10), tol); // a gate below -limit stays
    try testing.expectApproxEqRel(silu(4) * -2.5, ref.swigluClamped(4, -2.5, 10), tol);
    try testing.expectApproxEqRel(silu(20) * 20, ref.swigluClamped(20, 20, 0), tol); // no clamp
    try testing.expect(std.math.isNan(ref.swigluClamped(std.math.nan(f32), 1, 10)));
    try testing.expect(std.math.isNan(ref.swigluClamped(1, std.math.nan(f32), 10)));
    // a small K4 bank: the projections are O(1) for x ~ N(0, 1), so a limit of 1e30 never clamps, and one of 1 does
    var prng = std.Random.DefaultPrng.init(0xC1A7);
    const rnd = prng.random();
    const a = testing.allocator;
    var b = try ref.synthBank(a, rnd, 4, 256, 128, format.Rate.fromK(4));
    defer b.deinit(a);
    var x: [256]f32 = undefined;
    for (&x) |*v| v.* = rnd.floatNorm(f32);
    const slots = [_]u32{ 1, 3 };
    const scores = [_]f32{ 0.7, 0.3 };
    var y0: [256]f32 = undefined;
    try ref.moeRowDec(a, &b, .mul1, &x, &slots, &scores, &y0);
    var y: [256]f32 = undefined;
    const none = try ref.moeRowClamped(a, &b, .mul1, 1e30, &x, &slots, &scores, &y);
    try testing.expectEqualSlices(f32, &y0, &y);
    try testing.expect(none.gate == 0 and none.up == 0 and none.total == 2 * 128);
    const hits = try ref.moeRowClamped(a, &b, .mul1, 1, &x, &slots, &scores, &y);
    try testing.expect(hits.gate > 0 and hits.up > 0);
    try testing.expect(ref.compare(&y, &y0).max_ratio > 1e-3);
}

test "turboexl3 clamped refusals: moeClamped takes a limit in 1..128; form 1 (Kernels) and forms 5 and 6 (Kernels5) refuse a clamped geometry" {
    const none: mlx.mlx_array = .{ .ctx = null };
    const p: root.Proj = .{ .trellis = none, .suh = none, .svh = none };
    const b: root.Bank = .{ .gate = p, .up = p, .down = p };
    const s0: mlx.mlx_stream = .{ .ctx = null };
    // refused before any operand is read
    for ([_]c_int{ 0, -10, 129 }) |l| try testing.expectError(error.InvalidExl3Clamp, root.moeClamped(s0, none, b, none, none, .mul1, l));
    const geo: decode.Geometry = .{ .hidden = 5120, .inter = 2304, .topk = 6, .rate = format.Rate.fromK(2), .dec = .mul1, .limit = 10 };
    try testing.expectError(error.Exl3EngineNotImplemented, decode.Kernels.init(geo));
    try testing.expectError(error.Exl3EngineNotImplemented, decode.Kernels5.init(geo, 4, 4, 4));
}

/// The bank as `root` takes it: each projection's arrays from the host bank.
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

/// One call's routing: each row's top-k distinct experts by Gumbel top-k over lognormal popularity (sigma `skew`: a few
/// experts take many rows, as captured routing does, so prefill fills 64- and 128-row tiles and decode rows share
/// experts), scores positive, summing to one per row before their bf16 rounding; row 0 takes the bank's last expert.
fn route(a: std.mem.Allocator, rnd: std.Random, rows: u32, experts: u32, topk: u32, skew: f64, slots: []u32, scores: []f32) !void {
    const logw = try a.alloc(f64, experts);
    defer a.free(logw);
    for (logw) |*w| w.* = skew * rnd.floatNorm(f64);
    const keys = try a.alloc(f64, experts);
    defer a.free(keys);
    for (0..rows) |r| {
        for (keys, logw) |*kv, lw| kv.* = lw - @log(-@log(@max(rnd.float(f64), 1e-300)));
        for (slots[r * topk ..][0..topk]) |*sl| {
            var best: usize = 0;
            for (keys, 0..) |kv, e| {
                if (kv > keys[best]) best = e;
            }
            sl.* = @intCast(best);
            keys[best] = -std.math.inf(f64);
        }
        const ws = scores[r * topk ..][0..topk];
        var sum: f32 = 0;
        for (ws) |*w| {
            w.* = 0.05 + rnd.float(f32);
            sum += w.*;
        }
        for (ws) |*w| w.* = bfExact(w.* / sum);
    }
    if (std.mem.indexOfScalar(u32, slots[0..topk], experts - 1) == null) slots[topk - 1] = experts - 1;
}

/// One call's host values (x ~ N(0, sigma^2) and the scores bf16-exact, as both sides read them) and its operands as a host
/// passes them: x [1, rows, hidden] bf16, the slots i32 and the scores bf16 [1, rows, topk].
const Call = struct {
    x: []f32,
    slots: []u32,
    scores: []f32,
    x3: mlx.mlx_array,
    inds3: mlx.mlx_array,
    sc3: mlx.mlx_array,

    fn make(a: std.mem.Allocator, g: *G, rnd: std.Random, rows: u32, experts: u32, topk: u32, hidden: u32, sigma: f32, skew: f64) !Call {
        const x = try a.alloc(f32, rows * hidden);
        errdefer a.free(x);
        for (x) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
        const slots = try a.alloc(u32, rows * topk);
        errdefer a.free(slots);
        const scores = try a.alloc(f32, rows * topk);
        errdefer a.free(scores);
        try route(a, rnd, rows, experts, topk, skew, slots, scores);
        const inds = try a.alloc(i32, rows * topk);
        defer a.free(inds);
        for (slots, inds) |sl, *d| d.* = @intCast(sl);
        return .{
            .x = x,
            .slots = slots,
            .scores = scores,
            .x3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(hidden) }, .float32), .bfloat16),
            .inds3 = try g.hostArray(std.mem.sliceAsBytes(inds), &.{ 1, ci(rows), ci(topk) }, .int32),
            .sc3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, ci(rows), ci(topk) }, .float32), .bfloat16),
        };
    }

    fn deinit(c: *Call, a: std.mem.Allocator) void {
        a.free(c.x);
        a.free(c.slots);
        a.free(c.scores);
    }
};

/// The elements whose bits differ between two arrays of one size (both read as f32: bf16 and f32 widen exactly).
fn bitDiff(a: std.mem.Allocator, g: *G, x: mlx.mlx_array, y: mlx.mlx_array) !usize {
    const n = mlx.mlx_array_size(x);
    if (mlx.mlx_array_size(y) != n) return error.SizeMismatch;
    const p = try a.alloc(f32, n);
    defer a.free(p);
    const q = try a.alloc(f32, n);
    defer a.free(q);
    _ = try g.hostF32(try g.astype(x, .float32), p);
    _ = try g.hostF32(try g.astype(y, .float32), q);
    var d: usize = 0;
    for (p, q) |u, v| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
    return d;
}

// Qwen3.8-Flash-Next's routed geometry on 48 experts at K5, x ~ N(0, 1): every gate and up stays near 1, far below the limit
// 128, so the clamp never acts and each clamped launch must give its unclamped launch's bytes. `root` at decode rows under
// every form (a clamped call runs forms 7, 8 and 9 as configured, form 9 for forms 3, 5 and 6) and at prefill rows (the
// served 128-row tiles: text_mid_t128c), bf16 and f32; then the clamped mids `root` does not serve, each against its
// unclamped launch at the served option set otherwise: the mixed chain without the 128-row tiles (text_midc) and the vector
// layout (text_mid_vc). Then a clamped Kernels3's refusals.
test "turboexl3 gpu clamped: at a limit no value reaches, every clamped launch gives the unclamped one's bytes (root's decode forms and prefill, the other prefill mids); a clamped Kernels3 refuses form 3 and the shared heads (TURBOEXL3_GPU=1)" {
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
    const rate = format.Rate.fromN(80).?;
    const limit: c_int = 128;
    var prng = std.Random.DefaultPrng.init(0xC1A3B17E);
    const rnd = prng.random();
    var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
    defer bank.deinit(a);
    const rb = try rootBank(&g, &bank);
    const ops: prefill.Operands = .{
        .gate_t = rb.gate.trellis,
        .gate_suh = rb.gate.suh,
        .gate_svh = rb.gate.svh,
        .up_t = rb.up.trellis,
        .up_suh = rb.up.suh,
        .up_svh = rb.up.svh,
        .down_t = rb.down.trellis,
        .down_suh = rb.down.suh,
        .down_svh = rb.down.svh,
    };
    // root.pfOptions' set without the 128-row tiles (n 80: the integer-rate readers), and its vector-layout arm
    const mixed: prefill.Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .sintr = true, .defer_affine = true, .ncb = 4, .froute = true, .alay = 2 };
    var vec = mixed;
    vec.alay = 0;
    vec.aux = 8;
    const geo: prefill.Geometry = .{ .hidden = hidden, .inter = inter, .topk = topk, .experts = experts, .rate = rate, .dec = .mul1 };
    var geo_c = geo;
    geo_c.limit = @intCast(limit);
    var pu_m = try prefill.Prefill.init(geo, mixed);
    defer pu_m.deinit();
    var pc_m = try prefill.Prefill.init(geo_c, mixed);
    defer pc_m.deinit();
    var pu_v = try prefill.Prefill.init(geo, vec);
    defer pu_v.deinit();
    var pc_v = try prefill.Prefill.init(geo_c, vec);
    defer pc_v.deinit();
    defer root.decode_form_override.get().* = null;
    var failed = false;
    for ([_]u32{ 1, 4, 16, 40, 400 }) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        var c = try Call.make(a, &g, rnd, rows, experts, topk, hidden, 1, 1.5);
        defer c.deinit(a);
        const decode_rows = rows <= root.kernels.DECODE_ROWS_MAX;
        const forms: []const u8 = if (decode_rows) &.{ 3, 5, 6, 7, 8, 9 } else &.{9};
        for (forms) |form| {
            root.decode_form_override.get().* = form;
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const x = try g.astype(c.x3, dt);
                const yu = try g.adopt(try root.moe(s, x, rb, c.inds3, c.sc3, .mul1, false));
                const yc = try g.adopt(try root.moeClamped(s, x, rb, c.inds3, c.sc3, .mul1, limit));
                const d = try bitDiff(a, &g, yu, yc);
                std.debug.print("TURBOEXL3_CLAMP_BYTES root rows={d} form={d} out={s} differing={d}/{d} {s}\n", .{ rows, form, @tagName(dt), d, rows * hidden, if (d == 0) "PASS" else "FAIL" });
                if (d != 0) failed = true;
            }
        }
        root.decode_form_override.get().* = null;
        if (decode_rows) continue;
        const x2 = try g.reshape(c.x3, &.{ ci(rows), ci(hidden) });
        const inds1 = try g.reshape(c.inds3, &.{ci(rows * topk)});
        const sc1 = try g.reshape(c.sc3, &.{ci(rows * topk)});
        const Arm = struct { name: []const u8, u: *prefill.Prefill, c: *prefill.Prefill };
        for ([_]Arm{ .{ .name = "midc", .u = &pu_m, .c = &pc_m }, .{ .name = "mid_vc", .u = &pu_v, .c = &pc_v } }) |arm| {
            const yu = try g.adopt(try arm.u.moe(s, x2, inds1, sc1, ops, rows, .float32));
            const yc = try g.adopt(try arm.c.moe(s, x2, inds1, sc1, ops, rows, .float32));
            const d = try bitDiff(a, &g, yu, yc);
            std.debug.print("TURBOEXL3_CLAMP_BYTES {s} rows={d} differing={d}/{d} {s}\n", .{ arm.name, rows, d, rows * hidden, if (d == 0) "PASS" else "FAIL" });
            if (d != 0) failed = true;
        }
    }
    // a clamped Kernels3: form 3's fin3 has no clamp, and neither has the shared heads' table SwiGLU (both refused before
    // any launch)
    var k3 = try decode.Kernels3.initWith(.{ .hidden = hidden, .inter = inter, .topk = topk, .rate = rate, .dec = .mul1, .limit = @intCast(limit) }, 1, 4, 2);
    defer k3.deinit();
    const none: mlx.mlx_array = .{ .ctx = null };
    const kb: decode.Kernels.Bank = .{ .gate_t = none, .gate_suh = none, .gate_svh = none, .up_t = none, .up_suh = none, .up_svh = none, .down_t = none, .down_suh = none, .down_svh = none };
    try testing.expectError(error.Exl3EngineNotImplemented, k3.moeForm(s, none, none, none, kb, 1, 3));
    const sb: decode.Kernels3.SharedBank = .{ .gate_t = none, .gate_suh = none, .gate_svh = none, .up_t = none, .up_suh = none, .up_svh = none, .down_t = none, .down_suh = none, .down_svh = none, .n_gu = 80, .n_dn = 80, .ks_gu = 20, .ks_dn = 5, .defer_affine = true, .sigtab = none, .logit = none };
    try testing.expect(!k3.sharedAdmitted(1, sb));
    try testing.expect(!failed);
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

/// The rows a call's reference covers (it decodes every weight of their experts on the CPU): the first and the last, and
/// the middle one past 32 rows.
fn sampleRows(rows: u32, buf: *[3]u32) []const u32 {
    buf.* = .{ 0, rows - 1, rows / 2 };
    if (rows == 1) return buf[0..1];
    return if (rows <= 32) buf[0..2] else buf;
}

// DeepSeek-V4.1-Flash's routed experts: hidden 5120, expert width 2304, 384 experts, top-6, swiglu_limit 10, MUL1 banks at
// K2, K3 and K6. x ~ N(0, 8^2): synthBank's projections keep a gain near 1.08 (decoded weights ~ N(0, 1), suh and svh means
// 1 and 1 / sqrt(in)), so gate and up come out ~ N(0, 8.6^2) and about 12% of the gates pass 10 and 25% of the ups pass
// +-10; the test requires 1% of each on the sampled rows. Routing: Gumbel top-k at skew 1.5 (row 0 on expert 383), so the
// 1328-row call fills 128-row, 64-row and register-path tiles and decode rows share experts. Each call in bf16 (served:
// within 1.5e-2 of the reference, as root.moe) and f32 (within 5e-3).
test "turboexl3 gpu clamped: root.moeClamped at DeepSeek-V4.1-Flash's routed geometry (5120 x 2304, 384 experts, top-6, K2 K3 K6, limit 10) equals the clamped reference at decode rows 1, 4, 16 and prefill rows 32, 1328 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const hidden: u32 = 5120;
    const inter: u32 = 2304;
    const experts: u32 = 384;
    const topk: u32 = 6;
    const limit: c_int = 10;
    const sigma: f32 = 8;
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_CLAMP_N", &.{ 32, 48, 96 }, &nbuf);
    var rbuf: [8]u32 = undefined;
    const row_counts = envList("TURBOEXL3_CLAMP_ROWS", &.{ 1, 4, 16, 32, 1328 }, &rbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0xD5E4100 ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
        defer bank.deinit(a);
        const rb = try rootBank(&g, &bank);
        for (row_counts) |rows| {
            if (rows == 0) continue;
            const m2 = g.mark();
            defer g.resetTo(m2);
            var c = try Call.make(a, &g, rnd, rows, experts, topk, hidden, sigma, 1.5);
            defer c.deinit(a);
            var sbuf: [3]u32 = undefined;
            const sample = sampleRows(rows, &sbuf);
            const want = try a.alloc(f32, sample.len * hidden);
            defer a.free(want);
            var hits: ref.Hits = .{};
            for (sample, 0..) |r, i| {
                const h = try ref.moeRowClamped(a, &bank, .mul1, @floatFromInt(limit), c.x[r * hidden ..][0..hidden], c.slots[r * topk ..][0..topk], c.scores[r * topk ..][0..topk], want[i * hidden ..][0..hidden]);
                hits.gate += h.gate;
                hits.up += h.up;
                hits.total += h.total;
            }
            const hit = hits.gate * 100 >= hits.total and hits.up * 100 >= hits.total;
            const got = try a.alloc(f32, rows * hidden);
            defer a.free(got);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const y = try g.adopt(try root.moeClamped(s, try g.astype(c.x3, dt), rb, c.inds3, c.sc3, .mul1, limit));
                try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(hidden) }, mlx.getShape(y));
                try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
                _ = try g.hostF32(try g.astype(y, .float32), got);
                var bad: usize = 0;
                for (got) |v| bad += @intFromBool(!std.math.isFinite(v));
                // the chain's tolerance in both dtypes: a bf16 output is judged beyond its store's rounding (2^-8 of each
                // value), which a flat bar over the RMS cannot carry for a row's largest outputs (1328 rows at K3: 1.69e-2)
                var worst: f64 = 0;
                for (sample, 0..) |r, i| {
                    const gr = got[r * hidden ..][0..hidden];
                    const wr = want[i * hidden ..][0..hidden];
                    worst = @max(worst, (if (dt == .float32) ref.compare(gr, wr) else ref.compareRounded(gr, wr, 0x1p-8)).max_ratio);
                }
                const bar: f64 = 5e-3;
                const ok = hit and bad == 0 and worst <= bar;
                const path: []const u8 = if (rows <= root.kernels.DECODE_ROWS_MAX) "decode" else "prefill";
                std.debug.print("TURBOEXL3_CLAMP_PARITY n={d} rows={d} path={s} out={s} worst_vs_ref={e:.3} bar={e:.1} nonfinite={d} gates_clamped={d}/{d} ups_clamped={d}/{d} {s}\n", .{ n, rows, path, @tagName(dt), worst, bar, bad, hits.gate, hits.total, hits.up, hits.total, if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            }
        }
    }
    try testing.expect(!failed);
}
