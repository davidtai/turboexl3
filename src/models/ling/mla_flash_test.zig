//! Ling-3.1-flash's MLA prompt attention in one launch (`mla_flash.zig`, `root.ling.mlaFlashPrefill`). CPU (always): the
//! kernel's text is MLX's attention_nax statements (its pin file) with no mask, sinks, log-sum-exp or window arm, and
//! the header and source end in the attribute's define and undefine; the refusals, before any launch. GPU
//! (TURBOEXL3_GPU=1, else skipped):
//! - at the head dims MLX instantiates its fused NAX kernel for (qk = v = 128; qk 96, v 64) the kernel gives MLX's
//!   fused causal SDPA's words (`force_fused`), the keys and values read in place from views of a wider cache, GQA too;
//! - at Ling's heads (qk 192, v 128) it is within tolerance of an f64 host reference on sampled query rows of every
//!   head (bf16 and f16; prompt chunks from 64 to 4096 rows over up to 8192 keys, and Ling's 64 heads), and no worse
//!   than MLX's composed path against the same reference;
//! - at the served sizes (FL-1, main's operand audit): Ling's 64 heads in bf16 over views of the cache's
//!   [1, 64, 32768, *] buffers (--ctx-size 32768) at layer-major's calls of the standard request (4096 query rows over
//!   12,288 and 16,384 keys, 1965 over 18,349, 31 over 18,380) and the narrowest call the plugin hands it (17 over
//!   18,380), the reference on sampled rows of heads 0, 21, 42 and 63, MLX's composed path one head at a time.
//! TURBOEXL3_METAL_DUMP=1: the kernel as standalone translation units (Ling's bf16 heads aligned and not, f16 at 128)
//! for an offline `xcrun metal` compile. Lines: TURBOEXL3_LING_FLASH_WORDS, TURBOEXL3_LING_FLASH,
//! TURBOEXL3_LING_FLASH_SERVED.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const config = @import("../../core/config.zig");
const prefill = @import("../../core/prefill.zig");
const root = @import("../../root.zig");
const flash = @import("mla_flash.zig");

const testing = std.testing;

fn has(text: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, text, needle) != null;
}

fn ci(x: anytype) c_int {
    return @intCast(x);
}

test "turboexl3 ling mla flash texts: the body is MLX's attention_nax statements with no mask, sinks, log-sum-exp or window arm; the attribute defined after the header, undefined after the body" {
    const pin = comptime support.pins(@embedFile("kernels/test/pins/mla_flash.metal"));
    for (pin[0 .. pin.len - 2]) |line| try testing.expect(has(flash.text_body, line));
    try testing.expect(std.mem.endsWith(u8, flash.header, flash.text_head));
    try testing.expect(std.mem.endsWith(u8, std.mem.trimEnd(u8, flash.text_head, "\n"), pin[pin.len - 2]));
    try testing.expect(std.mem.endsWith(u8, flash.source, flash.text_close));
    try testing.expect(has(flash.text_close, pin[pin.len - 1]));
    try testing.expect(std.mem.startsWith(u8, flash.header, @import("../../core/steel.zig").nax));
    // the operands MLX binds by name: the shapes and strides the body reads (the scale: the pin of its read)
    for ([_][]const u8{ "q_shape", "k_shape", "q_strides", "k_strides", "v_strides" }) |n| try testing.expect(has(flash.text_body, n));
    // attention_nax's mask, sinks, log-sum-exp and window arms gone (their function constants and operands)
    for ([_][]const u8{ "v_shape", "has_mask", "mask_params", "has_sinks", "save_lse", "has_window", "window_size" }) |n| try testing.expect(!has(flash.text_body, n));
    for ([_][]const u8{ "BQ", "BK", "BD", "BV", "WM", "WN", "ALIGN_Q", "ALIGN_K", "CAUSAL" }) |n| try testing.expect(has(flash.text_body, n));
}

test "turboexl3 ling mla flash refusals: shapes and dtypes the kernel does not take, and the NAX gate, refused before any launch" {
    const s = mlx.mlx_default_gpu_stream_new();
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    // zeros enough for the largest operand (4 heads x 64 rows x 256)
    const zeros = try a.alloc(u16, 4 * 64 * 256);
    defer a.free(zeros);
    @memset(zeros, 0);
    const bytes = std.mem.sliceAsBytes(zeros);
    const q = try g.hostArray(bytes, &.{ 1, 4, 32, 192 }, .bfloat16);
    const k = try g.hostArray(bytes, &.{ 1, 4, 64, 192 }, .bfloat16);
    const v = try g.hostArray(bytes, &.{ 1, 4, 64, 128 }, .bfloat16);
    const d = try flash.dimsOf(q, k, v);
    try testing.expectEqual(flash.Dims{ .b = 1, .h = 4, .hk = 4, .ql = 32, .kl = 64, .qk = 192, .v = 128 }, d);
    // kL < qL; a key head count that does not divide the query heads; v's rows not k's; a 3-d operand
    try testing.expectError(error.BadExl3Shape, flash.dimsOf(k, q, v));
    const k3 = try g.hostArray(bytes, &.{ 1, 3, 64, 192 }, .bfloat16);
    const v3 = try g.hostArray(bytes, &.{ 1, 3, 64, 128 }, .bfloat16);
    try testing.expectError(error.BadExl3Shape, flash.dimsOf(q, k3, v3));
    const v_short = try g.hostArray(bytes, &.{ 1, 4, 32, 128 }, .bfloat16);
    try testing.expectError(error.BadExl3Shape, flash.dimsOf(q, k, v_short));
    const q3 = try g.hostArray(bytes, &.{ 4, 32, 192 }, .bfloat16);
    try testing.expectError(error.BadExl3Shape, flash.dimsOf(q3, k, v));
    // head dims the kernel does not take (v 256: past the f32 output a lane keeps)
    const v256 = try g.hostArray(bytes, &.{ 1, 4, 64, 256 }, .bfloat16);
    try testing.expectError(error.BadExl3Shape, flash.dimsOf(q, k, v256));
    // f32 and mixed dtypes
    const vf = try g.astype(v, .float32);
    try testing.expectError(error.BadExl3Dtype, flash.dimsOf(q, k, vf));
    const qf = try g.astype(q, .float32);
    const kf = try g.astype(k, .float32);
    try testing.expectError(error.BadExl3Dtype, flash.dimsOf(qf, kf, vf));
    // the NAX gate
    const saved = config.get().*;
    defer config.configure(saved) catch unreachable;
    var c = saved;
    c.nax = false;
    try config.configure(c);
    try testing.expectError(error.Exl3EngineNotImplemented, root.ling.mlaFlashPrefill(s, q, k, v, 0.125));
}

test "turboexl3 ling mla flash metal source: the kernel as standalone translation units (TURBOEXL3_METAL_DUMP=1)" {
    _ = std.c.getenv("TURBOEXL3_METAL_DUMP") orelse return error.SkipZigTest;
    const d = comptime support.pins(@embedFile("kernels/test/dump/mla_flash.metal"));
    prefill.printMetal("turboexl3_ling_mla_flash_bf16_192_128_aligned", flash.header, d[0], d[3], flash.source);
    prefill.printMetal("turboexl3_ling_mla_flash_bf16_192_128", flash.header, d[1], d[3], flash.source);
    prefill.printMetal("turboexl3_ling_mla_flash_f16_128_128", flash.header, d[2], d[4], flash.source);
}

// ── GPU ──

/// MLX's SDPA (`mlx_fast_scaled_dot_product_attention`, mlx-c's signature with `sinks` and `force_fused`): the
/// consumer's binding where its `sdk.mlx` declares it, else this declaration of the same symbol.
const sdpa = if (@hasDecl(mlx, "mlx_fast_scaled_dot_product_attention")) mlx.mlx_fast_scaled_dot_product_attention else struct {
    extern "c" fn mlx_fast_scaled_dot_product_attention(res: *mlx.mlx_array, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mask_mode: [*:0]const u8, mask_arr: mlx.mlx_array, sinks: mlx.mlx_array, force_fused: bool, s: mlx.mlx_stream) c_int;
}.mlx_fast_scaled_dot_product_attention;

/// The RMS of got - want over the reference's RMS
fn rmsRatio(got: []const f32, want: []const f32, rms_ref: f64) f64 {
    var ss: f64 = 0;
    for (got, want) |u, w| ss += (@as(f64, u) - w) * (@as(f64, u) - w);
    const r = @sqrt(ss / @as(f64, @floatFromInt(want.len)));
    return if (rms_ref > 0) r / rms_ref else r;
}

/// MLX's causal SDPA of q, k, v (`force_fused`: its fused kernel or an error)
fn mlxCausal(g: *G, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, force_fused: bool) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    try mlx.check(sdpa(&out, q, k, v, scale, "causal", none, .{ .ctx = null }, force_fused, g.s));
    return g.adopt(out);
}

fn bfExact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
}

fn halfExact(v: f32) f32 {
    return @floatCast(@as(f16, @floatCast(v)));
}

/// One case: B, query heads H, key heads Hk, qL query rows over kL keys, head dims (qk, v); the keys and values a view
/// of `cap` >= kL cached rows
const Case = struct { b: u32 = 1, h: u32, hk: u32, ql: u32, kl: u32, qk: u32, v: u32, cap: u32 };

/// The operands of a case in `dt`: q contiguous [B, H, qL, qk] (q ~ N(0, 4)), k and v [B, Hk, kL, *] views of the first
/// kL rows of [B, Hk, cap, *] (N(0, 1)), each value exact in `dt`; their host copies (f32) in the same layouts as q,
/// and k, v at their view's rows
const Operands = struct { q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, hq: []f32, hk: []f32, hv: []f32 };

fn operands(a: std.mem.Allocator, g: *G, rnd: std.Random, c: Case, dt: mlx.mlx_dtype) !Operands {
    const exact: *const fn (f32) f32 = if (dt == .bfloat16) &bfExact else &halfExact;
    const nq = c.b * c.h * c.ql * c.qk;
    const nk = c.b * c.hk * c.cap * c.qk;
    const nv = c.b * c.hk * c.cap * c.v;
    const hq = try a.alloc(f32, nq);
    const hk = try a.alloc(f32, nk);
    const hv = try a.alloc(f32, nv);
    for (hq) |*x| x.* = exact(2 * rnd.floatNorm(f32));
    for (hk) |*x| x.* = exact(rnd.floatNorm(f32));
    for (hv) |*x| x.* = exact(rnd.floatNorm(f32));
    const q = try g.astype(try g.hostArray(std.mem.sliceAsBytes(hq), &.{ ci(c.b), ci(c.h), ci(c.ql), ci(c.qk) }, .float32), dt);
    const kc = try g.astype(try g.hostArray(std.mem.sliceAsBytes(hk), &.{ ci(c.b), ci(c.hk), ci(c.cap), ci(c.qk) }, .float32), dt);
    const vc = try g.astype(try g.hostArray(std.mem.sliceAsBytes(hv), &.{ ci(c.b), ci(c.hk), ci(c.cap), ci(c.v) }, .float32), dt);
    try g.evalAll(&.{ q, kc, vc });
    const k = try g.slice(kc, &.{ 0, 0, 0, 0 }, &.{ ci(c.b), ci(c.hk), ci(c.kl), ci(c.qk) }, &.{ 1, 1, 1, 1 });
    const v = try g.slice(vc, &.{ 0, 0, 0, 0 }, &.{ ci(c.b), ci(c.hk), ci(c.kl), ci(c.v) }, &.{ 1, 1, 1, 1 });
    return .{ .q = q, .k = k, .v = v, .hq = hq, .hk = hk, .hv = hv };
}

fn freeOperands(a: std.mem.Allocator, o: Operands) void {
    a.free(o.hq);
    a.free(o.hk);
    a.free(o.hv);
}

/// The query rows a case compares: both ends, either side of each 64-row tile edge near the start, the middle
fn sampleRows(ql: u32, buf: *[12]u32) []const u32 {
    const cands = [_]u32{ 0, 1, 15, 16, 63, 64, 65, ql / 2, ql / 2 + 1, ql -| 65, ql - 2, ql - 1 };
    var n: usize = 0;
    for (cands) |r| {
        if (r >= ql) continue;
        if (std.mem.indexOfScalar(u32, buf[0..n], r) != null) continue;
        buf[n] = r;
        n += 1;
    }
    return buf[0..n];
}

/// The f64 host attention of query row `r` of head (b, h): keys 0 .. kL - qL + r (MLX's causal alignment), into `out`
/// [v]
fn hostRow(c: Case, o: Operands, scale: f32, b: u32, h: u32, r: u32, out: []f64, w: []f64) void {
    const kh = h / (c.h / c.hk);
    const qrow = o.hq[((b * c.h + h) * c.ql + r) * c.qk ..][0..c.qk];
    const last = c.kl - c.ql + r;
    var m: f64 = -std.math.inf(f64);
    for (0..last + 1) |j| {
        const krow = o.hk[((b * c.hk + kh) * c.cap + j) * c.qk ..][0..c.qk];
        var sum: f64 = 0;
        for (qrow, krow) |x, y| sum += @as(f64, x) * @as(f64, y);
        w[j] = sum * @as(f64, scale);
        m = @max(m, w[j]);
    }
    var den: f64 = 0;
    for (w[0 .. last + 1]) |*x| {
        x.* = @exp(x.* - m);
        den += x.*;
    }
    @memset(out, 0);
    for (0..last + 1) |j| {
        const vrow = o.hv[((b * c.hk + kh) * c.cap + j) * c.v ..][0..c.v];
        for (out, vrow) |*acc, y| acc.* += w[j] * @as(f64, y);
    }
    for (out) |*x| x.* /= den;
}

test "turboexl3 gpu ling mla flash: at the head dims MLX instantiates its fused NAX kernel for, the kernel gives MLX's fused causal SDPA's words (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    if (!config.get().nax) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const cases = [_]Case{
        .{ .h = 4, .hk = 4, .ql = 256, .kl = 256, .qk = 128, .v = 128, .cap = 300 },
        .{ .h = 4, .hk = 4, .ql = 100, .kl = 356, .qk = 128, .v = 128, .cap = 400 },
        .{ .h = 4, .hk = 2, .ql = 64, .kl = 64, .qk = 128, .v = 128, .cap = 64 },
        .{ .b = 2, .h = 2, .hk = 2, .ql = 333, .kl = 1024, .qk = 128, .v = 128, .cap = 1100 },
        .{ .h = 4, .hk = 4, .ql = 256, .kl = 512, .qk = 96, .v = 64, .cap = 512 },
        .{ .h = 4, .hk = 4, .ql = 77, .kl = 333, .qk = 96, .v = 64, .cap = 340 },
    };
    var failed: usize = 0;
    for (cases, 0..) |c, ci_| {
        for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
            var prng = std.Random.DefaultPrng.init(0xF1A5 ^ (@as(u64, ci_) << 8) ^ @as(u64, @intCast(@intFromEnum(dt))));
            const mark = g.mark();
            defer g.resetTo(mark);
            const o = try operands(a, &g, prng.random(), c, dt);
            defer freeOperands(a, o);
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.qk)));
            const want = try mlxCausal(&g, o.q, o.k, o.v, scale, true);
            const got = try g.adopt(try root.ling.mlaFlashPrefill(s, o.q, o.k, o.v, scale));
            const n = c.b * c.h * c.ql * c.v;
            const wf = try a.alloc(f32, n);
            defer a.free(wf);
            const gf = try a.alloc(f32, n);
            defer a.free(gf);
            _ = try g.hostF32(try g.astype(want, .float32), wf);
            _ = try g.hostF32(try g.astype(got, .float32), gf);
            var differing: usize = 0;
            for (gf, wf) |x, y| differing += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
            const ok = differing == 0;
            std.debug.print("TURBOEXL3_LING_FLASH_WORDS b={d} h={d} hk={d} ql={d} kl={d} qk={d} v={d} x={s} differing={d}/{d} {s}\n", .{ c.b, c.h, c.hk, c.ql, c.kl, c.qk, c.v, @tagName(dt), differing, n, if (ok) "PASS" else "FAIL" });
            failed += @intFromBool(!ok);
        }
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

test "turboexl3 gpu ling mla flash: at Ling's heads (qk 192, v 128) the causal prompt attention is within tolerance of an f64 host reference and no worse than MLX's composed path against it (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    if (!config.get().nax) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const cases = [_]Case{
        .{ .h = 4, .hk = 4, .ql = 64, .kl = 64, .qk = 192, .v = 128, .cap = 64 },
        .{ .h = 4, .hk = 4, .ql = 100, .kl = 100, .qk = 192, .v = 128, .cap = 128 },
        .{ .h = 4, .hk = 4, .ql = 77, .kl = 4173, .qk = 192, .v = 128, .cap = 4200 },
        .{ .h = 2, .hk = 2, .ql = 4096, .kl = 8192, .qk = 192, .v = 128, .cap = 8192 },
        .{ .h = 4, .hk = 4, .ql = 1996, .kl = 6092, .qk = 192, .v = 128, .cap = 6144 },
        .{ .h = 64, .hk = 64, .ql = 256, .kl = 4352, .qk = 192, .v = 128, .cap = 4400 },
    };
    // the bar beyond the output's own rounding, over the reference's RMS (the dense bf16 bar)
    const bar: f64 = 2.5e-2;
    var failed: usize = 0;
    for (cases, 0..) |c, ci_| {
        for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
            var prng = std.Random.DefaultPrng.init(0x19A1 ^ (@as(u64, ci_) << 8) ^ @as(u64, @intCast(@intFromEnum(dt))));
            const mark = g.mark();
            defer g.resetTo(mark);
            const o = try operands(a, &g, prng.random(), c, dt);
            defer freeOperands(a, o);
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.qk)));
            const got = try g.adopt(try root.ling.mlaFlashPrefill(s, o.q, o.k, o.v, scale));
            const composed = try mlxCausal(&g, o.q, o.k, o.v, scale, false);
            const n = c.b * c.h * c.ql * c.v;
            const gf = try a.alloc(f32, n);
            defer a.free(gf);
            const cf = try a.alloc(f32, n);
            defer a.free(cf);
            _ = try g.hostF32(try g.astype(got, .float32), gf);
            _ = try g.hostF32(try g.astype(composed, .float32), cf);
            var rows_buf: [12]u32 = undefined;
            const rows = sampleRows(c.ql, &rows_buf);
            const m = c.b * c.h * rows.len * c.v;
            const want = try a.alloc(f32, m);
            defer a.free(want);
            const sel = try a.alloc(f32, m);
            defer a.free(sel);
            const selc = try a.alloc(f32, m);
            defer a.free(selc);
            const row = try a.alloc(f64, c.v);
            defer a.free(row);
            const w = try a.alloc(f64, c.kl);
            defer a.free(w);
            var at: usize = 0;
            for (0..c.b) |b| for (0..c.h) |h| for (rows) |r| {
                hostRow(c, o, scale, @intCast(b), @intCast(h), r, row, w);
                const off = ((b * c.h + h) * c.ql + r) * c.v;
                for (0..c.v) |j| {
                    want[at + j] = @floatCast(row[j]);
                    sel[at + j] = gf[off + j];
                    selc[at + j] = cf[off + j];
                }
                at += c.v;
            };
            const half_ulp: f64 = if (dt == .bfloat16) 1.0 / 256.0 else 1.0 / 2048.0;
            const e = @import("../../core/ref.zig").compareRounded(sel, want, half_ulp);
            const ec = @import("../../core/ref.zig").compareRounded(selc, want, half_ulp);
            // no worse than the composed path the prompt is served on today (MLX's SDPA at qk 192, spans of it): the RMS
            // within 10% and the worst element within 25% of its, each with a floor at the output's own rounding scale
            const rk = rmsRatio(sel, want, e.rms_ref);
            const rc = rmsRatio(selc, want, ec.rms_ref);
            const no_worse = rk <= @max(1.10 * rc, rc + 1e-4) and e.max_ratio <= @max(1.25 * ec.max_ratio, ec.max_ratio + 1.0 / 512.0);
            const ok = std.math.isFinite(e.max_abs) and std.math.isFinite(rk) and e.max_ratio <= bar and no_worse;
            std.debug.print("TURBOEXL3_LING_FLASH b={d} h={d} ql={d} kl={d} qk={d} v={d} x={s} rows={d} max_abs={e:.3} ratio={e:.3} composed_max_abs={e:.3} composed_ratio={e:.3} rms={e:.3} composed_rms={e:.3} bar={e:.1} no_worse={} {s}\n", .{ c.b, c.h, c.ql, c.kl, c.qk, c.v, @tagName(dt), rows.len, e.max_abs, e.max_ratio, ec.max_abs, ec.max_ratio, rk, rc, bar, no_worse, if (ok) "PASS" else "FAIL" });
            failed += @intFromBool(!ok);
        }
    }
    // FL-1 (main's operand audit, 2026-11-08): the served sizes. Ling's 64 heads in bf16, k and v views of the first kL
    // rows of the cache's [1, 64, 32768, 192 | 128] buffers (--ctx-size 32768: the request's reservation), at
    // layer-major's (query rows, key rows) of the standard request and the narrowest call the plugin hands the kernel
    // (17 rows). The f64 reference on the sampled rows of heads 0, 21, 42 and 63 (the host's time); MLX's composed path
    // for those heads one at a time, each its own call (one head's scores, not 64 heads'), the same bars.
    {
        const served_calls = [_][2]u32{ .{ 4096, 12288 }, .{ 4096, 16384 }, .{ 1965, 18349 }, .{ 31, 18380 }, .{ 17, 18380 } };
        const ref_heads = [_]u32{ 0, 21, 42, 63 };
        const nh: u32 = 64;
        const cap_rows: u32 = 32768;
        const xdt: mlx.mlx_dtype = .bfloat16;
        const fl_scale: f32 = 1.0 / @sqrt(192.0);
        var fl_prng = std.Random.DefaultPrng.init(0xF11);
        const fl_rnd = fl_prng.random();
        const mark0 = g.mark();
        defer g.resetTo(mark0);
        // The cache's buffers, once for every call: host copies (f32, each value exact in bf16) and the device's bf16.
        const hk_all = try a.alloc(f32, nh * cap_rows * 192);
        defer a.free(hk_all);
        const hv_all = try a.alloc(f32, nh * cap_rows * 128);
        defer a.free(hv_all);
        for (hk_all) |*x| x.* = bfExact(fl_rnd.floatNorm(f32));
        for (hv_all) |*x| x.* = bfExact(fl_rnd.floatNorm(f32));
        const kbuf = try g.astype(try g.hostArray(std.mem.sliceAsBytes(hk_all), &.{ 1, ci(nh), ci(cap_rows), 192 }, .float32), xdt);
        const vbuf = try g.astype(try g.hostArray(std.mem.sliceAsBytes(hv_all), &.{ 1, ci(nh), ci(cap_rows), 128 }, .float32), xdt);
        try g.evalAll(&.{ kbuf, vbuf });
        for (served_calls) |qk_rows| {
            const fc: Case = .{ .h = nh, .hk = nh, .ql = qk_rows[0], .kl = qk_rows[1], .qk = 192, .v = 128, .cap = cap_rows };
            const mark = g.mark();
            defer g.resetTo(mark);
            const hq_all = try a.alloc(f32, nh * fc.ql * 192);
            defer a.free(hq_all);
            for (hq_all) |*x| x.* = bfExact(2 * fl_rnd.floatNorm(f32));
            const fq = try g.astype(try g.hostArray(std.mem.sliceAsBytes(hq_all), &.{ 1, ci(nh), ci(fc.ql), 192 }, .float32), xdt);
            const fk = try g.slice(kbuf, &.{ 0, 0, 0, 0 }, &.{ 1, ci(nh), ci(fc.kl), 192 }, &.{ 1, 1, 1, 1 });
            const fv = try g.slice(vbuf, &.{ 0, 0, 0, 0 }, &.{ 1, ci(nh), ci(fc.kl), 128 }, &.{ 1, 1, 1, 1 });
            const fo: Operands = .{ .q = fq, .k = fk, .v = fv, .hq = hq_all, .hk = hk_all, .hv = hv_all };
            const fgot = try g.adopt(try root.ling.mlaFlashPrefill(s, fq, fk, fv, fl_scale));
            const fgf = try a.alloc(f32, nh * fc.ql * 128);
            defer a.free(fgf);
            _ = try g.hostF32(try g.astype(fgot, .float32), fgf);
            var frows_buf: [12]u32 = undefined;
            const frows = sampleRows(fc.ql, &frows_buf);
            const fm = ref_heads.len * frows.len * 128;
            const fwant = try a.alloc(f32, fm);
            defer a.free(fwant);
            const fsel = try a.alloc(f32, fm);
            defer a.free(fsel);
            const fselc = try a.alloc(f32, fm);
            defer a.free(fselc);
            const fcf = try a.alloc(f32, fc.ql * 128);
            defer a.free(fcf);
            const frow = try a.alloc(f64, 128);
            defer a.free(frow);
            const fw = try a.alloc(f64, fc.kl);
            defer a.free(fw);
            var fat: usize = 0;
            for (ref_heads) |h| {
                const hi = ci(h);
                const qh = try g.slice(fq, &.{ 0, hi, 0, 0 }, &.{ 1, hi + 1, ci(fc.ql), 192 }, &.{ 1, 1, 1, 1 });
                const kh = try g.slice(fk, &.{ 0, hi, 0, 0 }, &.{ 1, hi + 1, ci(fc.kl), 192 }, &.{ 1, 1, 1, 1 });
                const vh = try g.slice(fv, &.{ 0, hi, 0, 0 }, &.{ 1, hi + 1, ci(fc.kl), 128 }, &.{ 1, 1, 1, 1 });
                const composed_h = try mlxCausal(&g, qh, kh, vh, fl_scale, false);
                _ = try g.hostF32(try g.astype(composed_h, .float32), fcf);
                for (frows) |r| {
                    hostRow(fc, fo, fl_scale, 0, h, r, frow, fw);
                    const off = (h * fc.ql + r) * 128;
                    for (0..128) |j| {
                        fwant[fat + j] = @floatCast(frow[j]);
                        fsel[fat + j] = fgf[off + j];
                        fselc[fat + j] = fcf[r * 128 + j];
                    }
                    fat += 128;
                }
            }
            const fe = @import("../../core/ref.zig").compareRounded(fsel, fwant, 1.0 / 256.0);
            const fec = @import("../../core/ref.zig").compareRounded(fselc, fwant, 1.0 / 256.0);
            const frk = rmsRatio(fsel, fwant, fe.rms_ref);
            const frc = rmsRatio(fselc, fwant, fec.rms_ref);
            const fno_worse = frk <= @max(1.10 * frc, frc + 1e-4) and fe.max_ratio <= @max(1.25 * fec.max_ratio, fec.max_ratio + 1.0 / 512.0);
            const fok = std.math.isFinite(fe.max_abs) and std.math.isFinite(frk) and fe.max_ratio <= bar and fno_worse;
            std.debug.print("TURBOEXL3_LING_FLASH_SERVED b=1 h={d} heads=0,21,42,63 ql={d} kl={d} cap={d} qk=192 v=128 x=bfloat16 rows={d} max_abs={e:.3} ratio={e:.3} composed_max_abs={e:.3} composed_ratio={e:.3} rms={e:.3} composed_rms={e:.3} bar={e:.1} no_worse={} {s}\n", .{ nh, fc.ql, fc.kl, cap_rows, frows.len, fe.max_abs, fe.max_ratio, fec.max_abs, fec.max_ratio, frk, frc, bar, fno_worse, if (fok) "PASS" else "FAIL" });
            failed += @intFromBool(!fok);
        }
    }
    try testing.expectEqual(@as(usize, 0), failed);
}
