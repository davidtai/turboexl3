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
//! - the rope-once cache (`mlaFlashPrefillRope1`, `Config.mla_rope_once`): over each head's nope keys and values in one
//!   row a token (the outside plugin's kv_b rows) and the token's one roped key, read in place, the launch gives
//!   `mlaFlashPrefill`'s words over the expanded key and the values built from the same words, every word, at bf16 and
//!   f16, GQA and two batches, Ling's 64 heads and the served calls; red checks that must differ: a planted fault that
//!   reads every key slice from the nope keys' row (`turboexl3_test_ling_mla_flash_r1_nope_fault`) and token 0's roped key
//!   negated.
//! TURBOEXL3_METAL_DUMP=1: the kernel as standalone translation units (Ling's bf16 heads aligned and not, f16 at 128)
//! for an offline `xcrun metal` compile. Lines: TURBOEXL3_LING_FLASH_WORDS, TURBOEXL3_LING_FLASH,
//! TURBOEXL3_LING_FLASH_SERVED, TURBOEXL3_LING_FLASH_ROPE1 (_RED, _DONE).
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ks = sdk.kernel_launch;
const decode = @import("../../core/decode.zig");
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

/// The planted fault: the rope-once text with the key tile's source pinned to the nope keys (`kernels/test/fault/`):
/// every slice of a key's dims read from k's row at k's stride, the rope dims past KN too
const fault_kofs_nope = @embedFile("kernels/test/fault/mla_flash_r1_kofs_nope.metal");
const text_r1_nope: [:0]const u8 = flash.body_1 ++ flash.r1_setup ++ flash.body_2 ++ fault_kofs_nope ++ flash.body_3 ++ flash.r1_kload ++ flash.body_4 ++ flash.body_kstep ++ flash.r1_kstep ++ flash.body_5 ++ flash.text_close;

test "turboexl3 ling mla flash rope once texts: the base body is its pieces in order; the rope-once body swaps the key tile's offset and load for reads from k or kr and adds kr's setup and step, every other statement the base's; the planted fault reads every slice from k" {
    const pin = comptime support.pins(@embedFile("kernels/test/pins/mla_flash_r1.metal"));
    // the rope-once statements: kr's setup, the slice's source, stride and offset, kr's step; once each, none in the base
    for (pin[0..6]) |line| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, flash.text_body_r1, line));
        try testing.expect(!has(flash.text_body, line));
    }
    // both key tile loads (the partial last tile's and the whole tile's) read from the slice's source, never from K
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, flash.text_body_r1, pin[6]));
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, flash.text_body_r1, pin[7]));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, flash.text_body, pin[7]));
    // the base's key offset gone; the nope keys' step kept, at k's stride
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, flash.text_body, pin[8]));
    try testing.expect(!has(flash.text_body_r1, pin[8]));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, flash.text_body, pin[9]));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, flash.text_body_r1, pin[9]));
    // every attention_nax statement the base's pin file holds is the rope-once body's too
    const base_pin = comptime support.pins(@embedFile("kernels/test/pins/mla_flash.metal"));
    for (base_pin[0 .. base_pin.len - 2]) |line| try testing.expect(has(flash.text_body_r1, line));
    // the rope-once body: the base's pieces, the offset and the load swapped, the setup and the step added
    try testing.expectEqual(flash.text_body.len - flash.body_kofs.len - flash.body_kload.len + flash.r1_setup.len + flash.r1_kofs.len + flash.r1_kload.len + flash.r1_kstep.len, flash.text_body_r1.len);
    try testing.expect(std.mem.endsWith(u8, flash.source_r1, flash.text_close));
    try testing.expect(std.mem.startsWith(u8, flash.source_r1, flash.body_1));
    // the operand MLX binds by name for kr's strides, and the template's nope width
    try testing.expect(has(flash.text_body_r1, "kr_strides") and !has(flash.text_body, "kr_strides"));
    try testing.expect(has(flash.r1_kofs, "KN"));
    // the fault: the rope-once text with the slice's source pinned to k
    try testing.expect(has(fault_kofs_nope, pin[10]) and !has(flash.r1_kofs, pin[10]));
    try testing.expectEqual(flash.text_body_r1.len - flash.r1_kofs.len + fault_kofs_nope.len + flash.text_close.len, text_r1_nope.len);
    const before = flash.body_1.len + flash.r1_setup.len + flash.body_2.len;
    try testing.expectEqualStrings(flash.source_r1[0..before], text_r1_nope[0..before]);
    try testing.expectEqualStrings(flash.source_r1[before + flash.r1_kofs.len ..], text_r1_nope[before + fault_kofs_nope.len ..]);
}

test "turboexl3 ling mla flash rope once refusals: shapes and dtypes the rope-once launch does not take, Config.mla_rope_once off and the NAX gate, refused before any launch" {
    const s = mlx.mlx_default_gpu_stream_new();
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const zeros = try a.alloc(u16, 4 * 96 * 256); // the largest operand here: q_long [1, 4, 96, 192]
    defer a.free(zeros);
    @memset(zeros, 0);
    const bytes = std.mem.sliceAsBytes(zeros);
    const q = try g.hostArray(bytes, &.{ 1, 4, 32, 192 }, .bfloat16);
    const k = try g.hostArray(bytes, &.{ 1, 4, 64, 128 }, .bfloat16);
    const kr = try g.hostArray(bytes, &.{ 1, 1, 64, 64 }, .bfloat16);
    const v = try g.hostArray(bytes, &.{ 1, 4, 64, 128 }, .bfloat16);
    const rd = try flash.dimsOfRope1(q, k, kr, v);
    try testing.expectEqual(flash.Rope1Dims{ .d = .{ .b = 1, .h = 4, .hk = 4, .ql = 32, .kl = 64, .qk = 192, .v = 128 }, .kn = 128 }, rd);
    // a roped key a head (not one a token); kr's rows not k's; nope + rope not qk; a nope width off the 16-wide slices
    const kr4 = try g.hostArray(bytes, &.{ 1, 4, 64, 64 }, .bfloat16);
    try testing.expectError(error.BadExl3Shape, flash.dimsOfRope1(q, k, kr4, v));
    const kr_short = try g.hostArray(bytes, &.{ 1, 1, 32, 64 }, .bfloat16);
    try testing.expectError(error.BadExl3Shape, flash.dimsOfRope1(q, k, kr_short, v));
    const kr48 = try g.hostArray(bytes, &.{ 1, 1, 64, 48 }, .bfloat16);
    try testing.expectError(error.BadExl3Shape, flash.dimsOfRope1(q, k, kr48, v));
    const k120 = try g.hostArray(bytes, &.{ 1, 4, 64, 120 }, .bfloat16);
    const kr72 = try g.hostArray(bytes, &.{ 1, 1, 64, 72 }, .bfloat16);
    try testing.expectError(error.BadExl3Shape, flash.dimsOfRope1(q, k120, kr72, v));
    // kL < qL
    const q_long = try g.hostArray(bytes, &.{ 1, 4, 96, 192 }, .bfloat16);
    try testing.expectError(error.BadExl3Shape, flash.dimsOfRope1(q_long, k, kr, v));
    // a mixed dtype
    const krh = try g.astype(kr, .float16);
    try testing.expectError(error.BadExl3Dtype, flash.dimsOfRope1(q, k, krh, v));
    // the gates: Config.mla_rope_once off (the default), then on with the NAX gate off
    const saved = config.get().*;
    defer config.configure(saved) catch unreachable;
    var c = saved;
    c.mla_rope_once = false;
    try config.configure(c);
    try testing.expectError(error.Exl3EngineNotImplemented, root.ling.mlaFlashPrefillRope1(s, q, k, kr, v, 0.125));
    c.mla_rope_once = true;
    c.nax = false;
    try config.configure(c);
    try testing.expectError(error.Exl3EngineNotImplemented, root.ling.mlaFlashPrefillRope1(s, q, k, kr, v, 0.125));
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

// ── the rope-once cache (`mlaFlashPrefillRope1`) ──

/// Ling's MLA heads: a key's nope and roped dims, a value's dims
const r1_kn: u32 = 128;
const r1_rope: u32 = 64;
const r1_v: u32 = 128;

/// A rope-once case: B, query heads H, key heads Hk, qL query rows over kL keys of a cache of `cap` >= kL tokens; `both`:
/// bf16 and f16, else bf16 only (the served calls)
const Rope1Case = struct { b: u32 = 1, h: u32, hk: u32, ql: u32, kl: u32, cap: u32, both: bool = true };

/// A case's operands from one set of words (`dt`'s bits): the rope-once cache's buffers, kv_b's rows [B, cap, Hk, KN + v]
/// (each head's nope keys then its values, one row a token, as the outside plugin's cache keeps them) and the roped keys
/// [B, 1, cap, rope] (one a token), with k, v and kr views of their first kL tokens; the served operands, the expanded key
/// [B, Hk, kL, KN + rope] (each head's nope keys ++ the token's roped key) and the values [B, Hk, kL, v], each built
/// contiguous on the host from the same words; q [B, H, qL, KN + rope]. `rope` keeps the roped keys' words (the red check
/// changes them).
const Rope1Ops = struct { q: mlx.mlx_array, k: mlx.mlx_array, kr: mlx.mlx_array, v: mlx.mlx_array, k_full: mlx.mlx_array, v_full: mlx.mlx_array, rope: []u16 };

/// One word of `dt` drawn from N(0, sigma^2), as its bits
fn wordOf(rnd: std.Random, dt: mlx.mlx_dtype, sigma: f32) u16 {
    const x = sigma * rnd.floatNorm(f32);
    return if (dt == .bfloat16) support.bf16Bits(x) else @bitCast(@as(f16, @floatCast(x)));
}

fn rope1Operands(a: std.mem.Allocator, g: *G, rnd: std.Random, c: Rope1Case, dt: mlx.mlx_dtype) !Rope1Ops {
    const row = r1_kn + r1_v;
    const qk = r1_kn + r1_rope;
    const hq = try a.alloc(u16, c.b * c.h * c.ql * qk);
    defer a.free(hq);
    const rows = try a.alloc(u16, c.b * c.cap * c.hk * row);
    defer a.free(rows);
    const rope = try a.alloc(u16, c.b * c.cap * r1_rope);
    errdefer a.free(rope);
    const kf = try a.alloc(u16, c.b * c.hk * c.kl * qk);
    defer a.free(kf);
    const vf = try a.alloc(u16, c.b * c.hk * c.kl * r1_v);
    defer a.free(vf);
    for (hq) |*x| x.* = wordOf(rnd, dt, 2);
    for (rows) |*x| x.* = wordOf(rnd, dt, 1);
    for (rope) |*x| x.* = wordOf(rnd, dt, 1);
    for (0..c.b) |b| for (0..c.hk) |h| for (0..c.kl) |t| {
        const src = rows[((b * c.cap + t) * c.hk + h) * row ..][0..row];
        const kd = kf[((b * c.hk + h) * c.kl + t) * qk ..][0..qk];
        @memcpy(kd[0..r1_kn], src[0..r1_kn]);
        @memcpy(kd[r1_kn..], rope[(b * c.cap + t) * r1_rope ..][0..r1_rope]);
        @memcpy(vf[((b * c.hk + h) * c.kl + t) * r1_v ..][0..r1_v], src[r1_kn..]);
    };
    const q = try g.hostArray(std.mem.sliceAsBytes(hq), &.{ ci(c.b), ci(c.h), ci(c.ql), ci(qk) }, dt);
    const rbuf = try g.hostArray(std.mem.sliceAsBytes(rows), &.{ ci(c.b), ci(c.cap), ci(c.hk), ci(row) }, dt);
    const kbuf = try g.hostArray(std.mem.sliceAsBytes(rope), &.{ ci(c.b), 1, ci(c.cap), ci(r1_rope) }, dt);
    const k_full = try g.hostArray(std.mem.sliceAsBytes(kf), &.{ ci(c.b), ci(c.hk), ci(c.kl), ci(qk) }, dt);
    const v_full = try g.hostArray(std.mem.sliceAsBytes(vf), &.{ ci(c.b), ci(c.hk), ci(c.kl), ci(r1_v) }, dt);
    try g.evalAll(&.{ q, rbuf, kbuf, k_full, v_full });
    // [B, Hk, cap, KN + v]: each head's row at its own strides (head 256 words, token Hk x 256), no copy
    const heads = try g.transposeAxes(rbuf, &.{ 0, 2, 1, 3 });
    const k = try g.slice(heads, &.{ 0, 0, 0, 0 }, &.{ ci(c.b), ci(c.hk), ci(c.kl), ci(r1_kn) }, &.{ 1, 1, 1, 1 });
    const v = try g.slice(heads, &.{ 0, 0, 0, ci(r1_kn) }, &.{ ci(c.b), ci(c.hk), ci(c.kl), ci(row) }, &.{ 1, 1, 1, 1 });
    const kr = try g.slice(kbuf, &.{ 0, 0, 0, 0 }, &.{ ci(c.b), 1, ci(c.kl), ci(r1_rope) }, &.{ 1, 1, 1, 1 });
    try g.evalAll(&.{ k, v, kr });
    return .{ .q = q, .k = k, .kr = kr, .v = v, .k_full = k_full, .v_full = v_full, .rope = rope };
}

/// The words of x and y (n each, both T) that differ
fn wordsDiffering(a: std.mem.Allocator, g: *G, x: mlx.mlx_array, y: mlx.mlx_array, n: usize) !usize {
    const xf = try a.alloc(f32, n);
    defer a.free(xf);
    const yf = try a.alloc(f32, n);
    defer a.free(yf);
    _ = try g.hostF32(try g.astype(x, .float32), xf);
    _ = try g.hostF32(try g.astype(y, .float32), yf);
    var d: usize = 0;
    for (xf, yf) |u, w| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return d;
}

test "turboexl3 gpu ling mla flash rope once: over the rope-once cache (each head's nope keys and values one row a token, the roped key once a token, read in place) the launch gives mlaFlashPrefill's words over the expanded key; a planted fault and a changed roped key differ (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    if (!config.get().nax) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = config.get().*;
    defer config.configure(saved) catch {};
    var cfg = saved;
    cfg.mla_rope_once = true;
    try config.configure(cfg);
    const nope_fault = ks.newKernel("turboexl3_test_ling_mla_flash_r1_nope_fault", &flash.flash_r1_in, &flash.flash_out, text_r1_nope, flash.header, false) orelse return error.KernelCreateFailed;
    defer decode.freeKernel(nope_fault);
    // one aligned tile; partial query and key tiles (the red checks' case); a prompt chunk after a long prefix; two
    // batches (kr's batch offset) with GQA; Ling's 64 heads; FL-1's served calls at a 18,432-token cache (bf16)
    const cases = [_]Rope1Case{
        .{ .h = 4, .hk = 4, .ql = 64, .kl = 64, .cap = 64 },
        .{ .h = 4, .hk = 4, .ql = 100, .kl = 100, .cap = 128 },
        .{ .h = 4, .hk = 4, .ql = 77, .kl = 4173, .cap = 4200 },
        .{ .b = 2, .h = 4, .hk = 2, .ql = 333, .kl = 1024, .cap = 1100 },
        .{ .h = 64, .hk = 64, .ql = 256, .kl = 4352, .cap = 4400 },
        .{ .h = 64, .hk = 64, .ql = 4096, .kl = 16384, .cap = 18432, .both = false },
        .{ .h = 64, .hk = 64, .ql = 17, .kl = 18380, .cap = 18432, .both = false },
    };
    const red_case: usize = 1;
    var lines: usize = 0;
    var failed: usize = 0;
    var red_failed: usize = 0;
    for (cases, 0..) |c, ci_| {
        for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
            if (dt == .float16 and !c.both) continue;
            var prng = std.Random.DefaultPrng.init(0x201E ^ (@as(u64, ci_) << 8) ^ @as(u64, @intCast(@intFromEnum(dt))));
            const mark = g.mark();
            defer g.resetTo(mark);
            const o = try rope1Operands(a, &g, prng.random(), c, dt);
            defer a.free(o.rope);
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(r1_kn + r1_rope)));
            const want = try g.adopt(try root.ling.mlaFlashPrefill(s, o.q, o.k_full, o.v_full, scale));
            const got = try g.adopt(try root.ling.mlaFlashPrefillRope1(s, o.q, o.k, o.kr, o.v, scale));
            const n = c.b * c.h * c.ql * r1_v;
            const differing = try wordsDiffering(a, &g, got, want, n);
            const ok = differing == 0;
            std.debug.print("TURBOEXL3_LING_FLASH_ROPE1 b={d} h={d} hk={d} ql={d} kl={d} cap={d} qk={d}+{d} v={d} x={s} differing={d}/{d} {s}\n", .{ c.b, c.h, c.hk, c.ql, c.kl, c.cap, r1_kn, r1_rope, r1_v, @tagName(dt), differing, n, if (ok) "PASS" else "FAIL" });
            lines += 1;
            failed += @intFromBool(!ok);
            if (ci_ != red_case or dt != .bfloat16) continue;
            // red 1: the planted fault (every key slice from the nope keys' row: the rope dims read the head's values)
            const rd = try flash.dimsOfRope1(o.q, o.k, o.kr, o.v);
            const faulty = try g.adopt(try flash.rope1On(s, nope_fault, rd, o.q, o.k, o.kr, o.v, scale));
            const d1 = try wordsDiffering(a, &g, faulty, want, n);
            std.debug.print("TURBOEXL3_LING_FLASH_ROPE1_RED check=nope_fault ql={d} kl={d} x={s} differing={d}/{d} (a fault must differ) {s}\n", .{ c.ql, c.kl, @tagName(dt), d1, n, if (d1 > 0) "PASS" else "FAIL" });
            red_failed += @intFromBool(d1 == 0);
            // red 2: token 0's roped key negated (every query row attends key 0), the served operands as they were
            const rope2 = try a.dupe(u16, o.rope);
            defer a.free(rope2);
            for (0..c.b) |b| for (rope2[b * c.cap * r1_rope ..][0..r1_rope]) |*w| {
                w.* ^= 0x8000;
            };
            const kbuf2 = try g.hostArray(std.mem.sliceAsBytes(rope2), &.{ ci(c.b), 1, ci(c.cap), ci(r1_rope) }, dt);
            const kr2 = try g.slice(kbuf2, &.{ 0, 0, 0, 0 }, &.{ ci(c.b), 1, ci(c.kl), ci(r1_rope) }, &.{ 1, 1, 1, 1 });
            try g.evalAll(&.{kr2});
            const changed = try g.adopt(try root.ling.mlaFlashPrefillRope1(s, o.q, o.k, kr2, o.v, scale));
            const d2 = try wordsDiffering(a, &g, changed, want, n);
            std.debug.print("TURBOEXL3_LING_FLASH_ROPE1_RED check=rope_token0_negated ql={d} kl={d} x={s} differing={d}/{d} (a changed roped key must differ) {s}\n", .{ c.ql, c.kl, @tagName(dt), d2, n, if (d2 > 0) "PASS" else "FAIL" });
            red_failed += @intFromBool(d2 == 0);
        }
    }
    std.debug.print("TURBOEXL3_LING_FLASH_ROPE1_DONE {d} lines, {d} failed; red checks failed {d}\n", .{ lines, failed, red_failed });
    try testing.expectEqual(@as(usize, 0), failed);
    try testing.expectEqual(@as(usize, 0), red_failed);
}
