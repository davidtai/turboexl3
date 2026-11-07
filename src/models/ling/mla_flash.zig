//! Ling-3.1-flash's MLA prompt attention in one launch (`mlaFlashPrefill`): causal flash attention over the expanded KV
//! at the model's head dims (qk 192 = 128 nope + 64 rope, v 128, 64 heads), which MLX 0.32.5 runs only on its composed
//! path at prompt widths (`ScaledDotProductAttention::use_fallback` for a head dim of 192): the scores and the
//! probabilities written to and read from device memory at the activation dtype, about 9 TFLOPS on Ling's prompt chunks.
//! The kernel is MLX's own NAX attention (steel/attn/kernels/steel_attention_nax.h `attention_nax`, MLX 0.32.5;
//! kernels/mla_flash/body.metal its statements less the mask, sinks, log-sum-exp and window arms) at a tile MLX does not
//! instantiate: bq 64, bk 32, bd 192, bv 128 and 4 simdgroups along the queries, MLX's own tile for head dims up to 128.
//! The template takes any `admits` pair, so at a pair MLX instantiates (qk = v = 128; qk 96, v 64) it is MLX's fused
//! kernel itself.
//! Numerics: MLX's fused kernel's, not the composed path's. The scores, the running row max and sum and the output
//! accumulate in f32 (exp2 of the log2(e)-scaled scores), P meets V in the NAX MMA, and the output rounds once to the
//! activation dtype; the composed path rounds the scaled query, the scores and the probabilities to the activation dtype
//! on the way. A reordering against the composed path: held to tolerance of an f32 reference (its GPU test), never to
//! the composed path's bytes.
//! The launch is MLX's: threadgroups (ceil(qL / BQ), H, B) of (32, WM, WN) threads, the operands at their own strides (a
//! cache's views read in place, no copy), the output contiguous.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const config = @import("../../core/config.zig");
const decode = @import("../../core/decode.zig");
const steel_nax = @import("../../core/steel.zig").nax;

/// After MLX steel's NAX tiles: attention_nax's row ops and finite minimum, the kernel attribute (MLX's threads per
/// threadgroup) on the generated definition
pub const text_head = @embedFile("kernels/mla_flash/head.metal");
/// attention_nax's statements for causal prompt attention, its parameters from the operands
pub const text_body = @embedFile("kernels/mla_flash/body.metal");
/// the attribute undefined after the definition, so MLX's explicit instantiation keeps a plain [[kernel]]
pub const text_close = @embedFile("kernels/mla_flash/close.metal");

fn sentinel(comptime s: []const u8) [:0]const u8 {
    comptime {
        var buf: [s.len:0]u8 = undefined;
        @memcpy(buf[0..s.len], s);
        const final = buf;
        return &final;
    }
}

pub const header: [:0]const u8 = sentinel(steel_nax ++ text_head);
pub const source: [:0]const u8 = sentinel(text_body ++ text_close);

pub const flash_in = [_][*:0]const u8{ "q", "k", "v", "scale" };
pub const flash_out = [_][*:0]const u8{"o"};

/// The launch tile: query rows a threadgroup (bq), key rows a step (bk), simdgroups along the queries (wm) and the head
/// dim (wn; MLX's split-D kernel only, so 1 here); bq = 16 wm wn, one 16-row fragment a simdgroup (attention_nax's TQ 1)
pub const Tile = struct { bq: u32, bk: u32, wm: u32, wn: u32 };
/// MLX's tile for head dims up to 128 (`sdpa_full_self_attention_nax`: bq 64, bk 32, wm 4, wn 1), taken at Ling's 192
pub const tile: Tile = .{ .bq = 64, .bk = 32, .wm = 4, .wn = 1 };

/// Whether the kernel serves head dims (qk, v): qk a multiple of 16 up to 256 (its query and key fragments, loaded a
/// 16-wide slice at a time), v a multiple of 32 up to 128 (the f32 output a lane keeps, v / 2 floats: MLX's own bound
/// for this kernel, past which it splits D)
pub fn admits(qk: u32, v: u32) bool {
    return qk >= 16 and qk <= 256 and qk % 16 == 0 and v >= 32 and v <= 128 and v % 32 == 0;
}

/// A call's launch: MLX's grid (threadgroups (ceil(qL / bq), H, B) of (32, wm, wn) threads, as threads) and whether its
/// query and key rows fill whole tiles (attention_nax's align_Q, align_K)
pub const Plan = struct { grid: [3]u32, threadgroup: [3]u32, align_q: bool, align_k: bool };

pub fn planFor(b: u32, h: u32, ql: u32, kl: u32, t: Tile) Plan {
    const nq = (ql + t.bq - 1) / t.bq;
    return .{
        .grid = .{ nq * 32, h * t.wm, b * t.wn },
        .threadgroup = .{ 32, t.wm, t.wn },
        .align_q = ql % t.bq == 0,
        .align_k = kl % t.bk == 0,
    };
}

const kernel_cache = Global(?mlx.mlx_fast_metal_kernel, "ling_mla_flash_kernel", null);
const kernel_lock = Global(std.atomic.Value(u32), "ling_mla_flash_lock", std.atomic.Value(u32).init(0));
const said = Global(bool, "ling_mla_flash_said", false);

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_cache, kernel_lock, said };

fn kernelGet() !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    if (kernel_cache.get().*) |k| return k;
    // operands at their own strides: a cache's views are read in place (the body reads q_strides, k_strides, v_strides)
    const k = ks.newKernel("turboexl3_ling_mla_flash", &flash_in, &flash_out, source, header, false) orelse return error.KernelCreateFailed;
    kernel_cache.get().* = k;
    return k;
}

/// Drops the kernel (`decode.freeKernel`, the consumer's drop); the next call makes it again
pub fn release() void {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    if (kernel_cache.get().*) |k| decode.freeKernel(k);
    kernel_cache.get().* = null;
}

/// The operands `mlaFlashPrefill` takes, checked before any launch: (B, H, Hk, qL, kL, qk, v)
pub const Dims = struct { b: u32, h: u32, hk: u32, ql: u32, kl: u32, qk: u32, v: u32 };

/// q [B, H, qL, qk], k [B, Hk, kL, qk], v [B, Hk, kL, v]: 4-d, the same B, H a multiple of Hk, qL >= 1, kL >= qL,
/// `admits` (qk, v); bf16 or f16, all three the same. Else error.BadExl3Shape / error.BadExl3Dtype. Each operand's last
/// axis must be contiguous (any other strides: a cache's views): the kernel reads the strides MLX binds at the launch,
/// which a lazy operand does not have yet, so the caller holds it (MLX's own SDPA copies such an operand first).
pub fn dimsOf(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array) !Dims {
    const qs = mlx.getShape(q);
    const kss = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (qs.len != 4 or kss.len != 4 or vs.len != 4) return error.BadExl3Shape;
    for ([_][]const c_int{ qs, kss, vs }) |sh| for (sh) |d| if (d < 1) return error.BadExl3Shape;
    if (kss[0] != qs[0] or vs[0] != qs[0] or vs[1] != kss[1] or vs[2] != kss[2] or kss[3] != qs[3]) return error.BadExl3Shape;
    if (@mod(qs[1], kss[1]) != 0 or kss[2] < qs[2]) return error.BadExl3Shape;
    const d: Dims = .{ .b = @intCast(qs[0]), .h = @intCast(qs[1]), .hk = @intCast(kss[1]), .ql = @intCast(qs[2]), .kl = @intCast(kss[2]), .qk = @intCast(qs[3]), .v = @intCast(vs[3]) };
    if (!admits(d.qk, d.v)) return error.BadExl3Shape;
    const dt = mlx.mlx_array_dtype(q);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(k) != dt or mlx.mlx_array_dtype(v) != dt) return error.BadExl3Dtype;
    return d;
}

/// Ling's MLA prompt attention (qk 192, v 128; any `admits` pair) as one causal flash launch: o [B, H, qL, v] in q's
/// dtype = softmax(scale q k^T) v per head (k and v head h / (H / Hk)), query row i attending keys 0 .. kL - qL + i
/// (MLX's causal alignment: the queries the last qL of the kL). `dimsOf`'s operands (each one's last axis contiguous,
/// the caller's to hold; a row stride under 2^31 elements, as MLX's kernel steps rows); refused with
/// error.Exl3EngineNotImplemented where the device fails the NAX gate (`Config.nax`). The caller owns o.
pub fn mlaFlashPrefill(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32) !mlx.mlx_array {
    if (!config.get().nax) return error.Exl3EngineNotImplemented;
    const d = try dimsOf(q, k, v);
    const kern = try kernelGet();
    const p = planFor(d.b, d.h, d.ql, d.kl, tile);
    const dt = mlx.mlx_array_dtype(q);
    if (!said.get().*) {
        said.get().* = true;
        std.debug.print("[turboexl3] Ling MLA prompt attention: one causal flash launch (MLX's attention_nax at qk {d}, v {d}; bq {d}, bk {d})\n", .{ d.qk, d.v, tile.bq, tile.bk });
    }
    const cfg = try ks.buildConfig(.{
        .grid = p.grid,
        .threadgroup = p.threadgroup,
        .out_shapes = &.{&[_]c_int{ @intCast(d.b), @intCast(d.h), @intCast(d.ql), @intCast(d.v) }},
        .out_dtypes = &.{dt},
        .template = &.{
            .{ .name = "T", .value = .{ .dtype = dt } },
            .{ .name = "BQ", .value = .{ .int = @intCast(tile.bq) } },
            .{ .name = "BK", .value = .{ .int = @intCast(tile.bk) } },
            .{ .name = "BD", .value = .{ .int = @intCast(d.qk) } },
            .{ .name = "BV", .value = .{ .int = @intCast(d.v) } },
            .{ .name = "WM", .value = .{ .int = @intCast(tile.wm) } },
            .{ .name = "WN", .value = .{ .int = @intCast(tile.wn) } },
            .{ .name = "ALIGN_Q", .value = .{ .int = @intFromBool(p.align_q) } },
            .{ .name = "ALIGN_K", .value = .{ .int = @intFromBool(p.align_k) } },
            .{ .name = "CAUSAL", .value = .{ .int = 1 } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    // the scale as one f32 (MLX binds it in the constant space), as MLX's AttnParams carries it
    const sv = [_]f32{scale};
    const sc = mlx.mlx_array_new_data(&sv, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(sc);
    var o: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(kern, cfg, &.{ q, k, v, sc }, &o, s);
    return o[0];
}

const testing = std.testing;

test "turboexl3 ling mla flash plan: MLX's grid and threadgroup, alignment by whole tiles, the admitted head dims" {
    const p = planFor(1, 64, 4096, 8192, tile);
    try testing.expectEqual([3]u32{ 64 * 32, 64 * 4, 1 }, p.grid);
    try testing.expectEqual([3]u32{ 32, 4, 1 }, p.threadgroup);
    try testing.expect(p.align_q and p.align_k);
    const r = planFor(2, 8, 1996, 18380, tile);
    try testing.expectEqual([3]u32{ 32 * 32, 8 * 4, 2 }, r.grid);
    try testing.expect(!r.align_q and !r.align_k);
    try testing.expect(planFor(1, 1, 100, 128, tile).align_k and !planFor(1, 1, 100, 128, tile).align_q);
    // one 16-row fragment a simdgroup (attention_nax's TQ 1)
    try testing.expectEqual(tile.bq, 16 * tile.wm * tile.wn);
    try testing.expect(admits(192, 128) and admits(128, 128) and admits(96, 64) and admits(64, 64) and admits(256, 128));
    for ([_][2]u32{ .{ 192, 256 }, .{ 200, 128 }, .{ 192, 48 }, .{ 0, 128 }, .{ 272, 128 }, .{ 192, 0 } }) |x| try testing.expect(!admits(x[0], x[1]));
}
