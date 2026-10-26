//! The decode-row merges' GPU byte tests (TURBOEXL3_GPU=1, else skipped): every merge (`decb.zig`) against the chain it
//! replaces, bit for bit, on synthetic trellises at Qwen3.8-Flash-Next's attention geometry (hidden 2560, 24 query heads
//! [query | gate] x 256, the indexer's 4 query heads + 1 key block x 128, rope 64 dims at theta 1e7, block ratio 4). Each
//! case runs with DEC-ONE off (the three-launch chain), on with the prepare inside its launch, and on with the prepare in
//! its own launch (the Dense's own switches set per case, as the DEC-ONE test sets them). The references are the host's
//! chains:
//! - o_proj over the gated output (`gatedDecode`, `mergeGatedDecode`): MLX's sigmoid and multiply (the host's strided gate
//!   kernel is pinned to them) on the attention output's transposed view and the gate's view, then `Dense.linear`; the
//!   qmerge form's merge is the host's `msv_attn_qsa256_qmerge` text (`decb.merge_body ++ decb.merge_store`, which the
//!   host's CPU test pins to its own) launched as a kernel here.
//! - the q_proj planes (`qPlanes`): `Dense.linear`, reshape [1, S, 24, 512], split, each half made contiguous.
//! - the indexer projection (`indexerQk`): `Dense.linear`, the q slice, reshape, MLX rms_norm, transpose, MLX rope (64
//!   dims, theta 1e7, the call's offset); the pooled key from the ring's leftover rows and the call's first rows: f32, mean
//!   over the block, bf16, rms_norm, the composed bf16 cos / sin rotation (the host's `qsaPooledRopeComposed`, which its
//!   fused kernel is pinned to).
//! Lines: TURBOEXL3_DECB <merge> ... differing=<n>/<total> PASS|FAIL; each test ends with TURBOEXL3_DECB_DONE <merge>
//! checks=<n> PASS|FAIL.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("test_support.zig");
const G = support.Graph;
const bf16Bits = support.bf16Bits;
const format = @import("format.zig");
const dense = @import("dense.zig");
const decode = @import("decode.zig");
const decb = @import("decb.zig");
const ks = @import("sdk").kernel_launch;

const testing = std.testing;

const HIDDEN: u32 = 2560;
const HQ: u32 = 24;
const HD: u32 = 256;
const RD: c_int = 64;
const THETA: f32 = 10_000_000.0;
const EPS: f32 = 1.0e-6;
const NIDX: u32 = 4;
const RATIO: u32 = 4;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// A dense EXL3 projection in -> out at rate n: trellis U16 [in/16, out/16, n] (every pattern is a codeword), suh F16 [in] in
/// +-[0.5, 1.5] / 16, svh F16 [out] in +-[0.5, 1.5] (the DEC-ONE test's operands: outputs O(1), both signs).
fn decbProj(a: std.mem.Allocator, g: *G, rnd: std.Random, in: u32, out: u32, n: u32) !struct { trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array } {
    const words = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(words);
    for (words) |*w| w.* = rnd.int(u16);
    const suh = try a.alloc(u16, in);
    defer a.free(suh);
    for (suh) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * (if (rnd.boolean()) @as(f32, 1) else -1) / 16.0);
    const svh = try a.alloc(u16, out);
    defer a.free(svh);
    for (svh) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * (if (rnd.boolean()) @as(f32, 1) else -1));
    return .{
        .trellis = g.keep(try g.hostArray(std.mem.sliceAsBytes(words), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16)),
        .suh = g.keep(try g.hostArray(std.mem.sliceAsBytes(suh), &.{ci(in)}, .float16)),
        .svh = g.keep(try g.hostArray(std.mem.sliceAsBytes(svh), &.{ci(out)}, .float16)),
    };
}

/// A bf16 array of `shape` with bf16-exact values lo + span * N(0,1) (or U[0,1) when `uniform`); a few exact zeros of both
/// signs when `zeros` (the finish and the norms must carry them as the chain does).
fn decbBf16(a: std.mem.Allocator, g: *G, rnd: std.Random, shape: []const c_int, lo: f32, span: f32, uniform: bool, zeros: bool) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try a.alloc(f32, n);
    defer a.free(buf);
    for (buf, 0..) |*v, i| {
        const r = if (uniform) rnd.float(f32) else rnd.floatNorm(f32);
        v.* = @bitCast(@as(u32, bf16Bits(lo + span * r)) << 16);
        if (zeros and i % 97 == 5) v.* = if (i % 2 == 0) 0.0 else -0.0;
    }
    return g.astype(try g.hostArray(std.mem.sliceAsBytes(buf), shape, .float32), .bfloat16);
}

/// The host's `swigluSigTable` for bf16: MLX's sigmoid of every 16-bit pattern (widened to the f32 that converts back
/// exactly, as the host's `all16BitPatterns`)
fn decbSigtab(a: std.mem.Allocator, g: *G) !mlx.mlx_array {
    const n: usize = 1 << 16;
    const buf = try a.alloc(f32, n);
    defer a.free(buf);
    for (buf, 0..) |*v, i| v.* = @bitCast(@as(u32, @intCast(i)) << 16);
    const vals = try g.astype(try g.hostArray(std.mem.sliceAsBytes(buf), &.{ci(n)}, .float32), .bfloat16);
    const t = try g.sigmoid(vals);
    try g.evalAll(&.{t});
    return t;
}

/// The host's `ropeAngleRows`: MLX's rope over a [1, 1, rows, RD] probe of ones in each pair's first half, flattened
fn decbAngles(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, rows: u32, offset: c_int) !mlx.mlx_array {
    const n: usize = @intCast(RD);
    const buf = try a.alloc(f32, n * rows);
    defer a.free(buf);
    for (buf, 0..) |*v, i| v.* = if ((i % n) < n / 2) 1.0 else 0.0;
    const probe = try g.hostArray(std.mem.sliceAsBytes(buf), &.{ 1, 1, ci(rows), RD }, .float32);
    var rot = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(&rot, probe, RD, false, .{ .value = THETA, .has_value = true }, 1.0, offset, .{ .ctx = null }, s));
    return g.reshape(try g.adopt(rot), &.{ci(n * rows)});
}

/// Elements whose bits differ (bf16 / f16 -> f32 is one-to-one on bit patterns), after a shape and dtype check
fn decbDiff(a: std.mem.Allocator, g: *G, want: mlx.mlx_array, got: mlx.mlx_array) !usize {
    if (!std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got)) or mlx.mlx_array_dtype(want) != mlx.mlx_array_dtype(got)) return std.math.maxInt(usize);
    const n = mlx.mlx_array_size(want);
    const wb = try a.alloc(f32, n);
    defer a.free(wb);
    const gb = try a.alloc(f32, n);
    defer a.free(gb);
    _ = try g.hostF32(try g.astype(want, .float32), wb);
    _ = try g.hostF32(try g.astype(got, .float32), gb);
    var diff: usize = 0;
    for (wb, gb) |p, q| {
        if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) diff += 1;
    }
    return diff;
}

/// The three arms every merge runs in: DEC-ONE off; on with the prepare inside wherever it fits; on with it in its own launch
const Arm = enum { chain, dec_inside, dec_own };
const arms = [_]Arm{ .chain, .dec_inside, .dec_own };

fn decbDense(geo: dense.Shape, arm: Arm) !dense.Dense {
    var d = try dense.Dense.init(geo);
    d.rowslot = false;
    d.dec_rows = @intCast(decode.Kernels.max_rows);
    switch (arm) {
        .chain => d.dec_on = false,
        .dec_inside => {
            d.dec_on = true;
            d.dec_prep_rows = @intCast(decode.Kernels.max_rows);
        },
        .dec_own => {
            d.dec_on = true;
            d.dec_prep_rows = 0;
            d.dec_prep_out = 0;
        },
    }
    return d;
}

/// Whether a merge's entry refused (error.Exl3EngineNotImplemented); a served call's result is freed
fn decbRefused(r: anytype) bool {
    if (r) |v| {
        switch (@TypeOf(v)) {
            mlx.mlx_array => _ = mlx.mlx_array_free(v),
            [2]mlx.mlx_array => for (v) |x| {
                _ = mlx.mlx_array_free(x);
            },
            else => {
                var o = v;
                o.deinit();
            },
        }
        return false;
    } else |e| return e == error.Exl3EngineNotImplemented;
}

const Tally = struct {
    checks: usize = 0,
    failed: bool = false,
    fn note(t: *Tally, ok: bool) void {
        t.checks += 1;
        if (!ok) t.failed = true;
    }
};

test "turboexl3 gpu decode merges: o_proj's prepare inside the gate kernel and inside the QSA merge equals the strided gate + linear chain byte for byte (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x4DEC3);
    const rnd = prng.random();
    const sigtab = g.keep(try decbSigtab(a, &g));
    defer g.release(sigtab);
    // the reference merge: the host's msv_attn_qsa256_qmerge text, launched as the host launches it
    const merge_src: [:0]const u8 = decb.merge_body ++ decb.merge_store;
    const merge_ins = [_][*:0]const u8{ "pacc", "pml" };
    const merge_outs = [_][*:0]const u8{"out"};
    const ref_merge = ks.newKernel("turboexl3_test_qmerge", &merge_ins, &merge_outs, merge_src, "", false) orelse return error.KernelCreateFailed;
    defer _ = mlx.mlx_fast_metal_kernel_free(ref_merge);
    var tally: Tally = .{};
    for ([_]u32{ 112, 80 }) |n| {
        const o = try decbProj(a, &g, rnd, HQ * HD, HIDDEN, n);
        defer for ([_]mlx.mlx_array{ o.trellis, o.suh, o.svh }) |x| g.release(x);
        const geo: dense.Shape = .{ .in = HQ * HD, .out = HIDDEN, .rate = format.Rate.fromN(n).?, .dec = .mul1, .out_dtype = .bfloat16 };
        var dref = try decbDense(geo, .chain);
        defer dref.deinit();
        for ([_]u32{ 1, 2, 3, 4, 8, 16 }) |rows| {
            const mark = g.mark();
            defer g.resetTo(mark);
            const r = ci(rows);
            // the attention output, head-major [1, H, S, D], and the two gate sources: q_proj [1, S, 2 H D] and a plane
            const attn = try decbBf16(a, &g, rnd, &.{ 1, ci(HQ), r, ci(HD) }, 0, 1, false, true);
            const qproj = try decbBf16(a, &g, rnd, &.{ 1, r, ci(2 * HQ * HD) }, 0, 2, false, true);
            const plane = try decbBf16(a, &g, rnd, &.{ 1, r, ci(HQ), ci(HD) }, 0, 2, false, true);
            const attn_t = try g.transposeAxes(attn, &.{ 0, 2, 1, 3 });
            const qg = try g.reshape(qproj, &.{ 1, r, ci(HQ), ci(2 * HD) });
            const gate_q = try g.slice(qg, &.{ 0, 0, 0, ci(HD) }, &.{ 1, r, ci(HQ), ci(2 * HD) }, &.{ 1, 1, 1, 1 });
            const Src = struct { name: []const u8, view: mlx.mlx_array, gate: decb.Gate };
            const srcs = [_]Src{
                .{ .name = "q_proj", .view = gate_q, .gate = .{ .base = qproj, .row_stride = 2 * HQ * HD, .head_stride = 2 * HD, .off = HD } },
                .{ .name = "plane", .view = plane, .gate = .{ .base = plane, .row_stride = HQ * HD, .head_stride = HD, .off = 0 } },
            };
            // the merge's partials: random accumulators and (m, l) pairs, a few splits empty (l = 0), never a whole row
            const NS: u32 = 16;
            const pacc = try g.astype(try decbBf16(a, &g, rnd, &.{ ci(HQ), r, ci(NS), 256 }, 0, 3, false, false), .float32);
            const ml = try a.alloc(f32, HQ * rows * NS * 2);
            defer a.free(ml);
            for (0..HQ * rows) |hr| for (0..NS) |j| {
                const at = (hr * NS + j) * 2;
                ml[at] = 10.0 * (rnd.float(f32) - 0.5);
                ml[at + 1] = if (j % 5 == 3) 0.0 else 0.25 + 4.0 * rnd.float(f32);
            };
            const pml = try g.hostArray(std.mem.sliceAsBytes(ml), &.{ ci(HQ), r, ci(NS), 2 }, .float32);
            // the reference merge output [1, H, S, 256] bf16
            const ref_attn = blk: {
                const o_shape = [_]c_int{ 1, ci(HQ), r, 256 };
                const shapes = [_][]const c_int{&o_shape};
                const dtypes = [_]mlx.mlx_dtype{.bfloat16};
                const tmpl = [_]ks.TemplateArg{.{ .name = "T", .value = .{ .dtype = .bfloat16 } }};
                const cfg = try ks.buildConfig(.{ .grid = .{ 256, rows, HQ }, .threadgroup = .{ 256, 1, 1 }, .out_shapes = &shapes, .out_dtypes = &dtypes, .template = &tmpl });
                defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
                const ins = [_]mlx.mlx_array{ pacc, pml };
                var out: [1]mlx.mlx_array = undefined;
                try ks.applyConfig(ref_merge, cfg, &ins, &out, s);
                break :blk try g.adopt(out[0]);
            };
            const ref_attn_t = try g.transposeAxes(ref_attn, &.{ 0, 2, 1, 3 });
            for (srcs) |src| {
                // the chain: sigmoid(gate) * attn in bf16 ([1, S, H, D]), reshape, linear
                const gated = try g.reshape(try g.mul(attn_t, try g.sigmoid(src.view)), &.{ r, ci(HQ * HD) });
                const want = try g.adopt(try dref.linear(s, gated, o.trellis, o.suh, o.svh, rows));
                const gated_m = try g.reshape(try g.mul(ref_attn_t, try g.sigmoid(src.view)), &.{ r, ci(HQ * HD) });
                const want_m = try g.adopt(try dref.linear(s, gated_m, o.trellis, o.suh, o.svh, rows));
                for (arms) |arm| {
                    var d = try decbDense(geo, arm);
                    defer d.deinit();
                    const mem: decb.Member = .{ .d = &d, .trellis = o.trellis, .suh = o.suh, .svh = o.svh };
                    const got = try g.adopt(try decb.gatedDecode(s, attn, src.gate, sigtab, mem, HQ, HD, rows));
                    const diff = try decbDiff(a, &g, want, got);
                    tally.note(diff == 0);
                    std.debug.print("TURBOEXL3_DECB gate n={d} rows={d} gate={s} arm={s} differing={d}/{d} {s}\n", .{ n, rows, src.name, @tagName(arm), diff, rows * HIDDEN, if (diff == 0) "PASS" else "FAIL" });
                    const got_m = try g.adopt(try decb.mergeGatedDecode(s, pacc, pml, src.gate, sigtab, mem, HQ, rows));
                    const diff_m = try decbDiff(a, &g, want_m, got_m);
                    tally.note(diff_m == 0);
                    std.debug.print("TURBOEXL3_DECB merge_gate n={d} rows={d} gate={s} arm={s} differing={d}/{d} {s}\n", .{ n, rows, src.name, @tagName(arm), diff_m, rows * HIDDEN, if (diff_m == 0) "PASS" else "FAIL" });
                }
            }
        }
        // rows past the GEMV path (17: the register-path GEMM) are refused before any launch, so the host keeps its chain
        // there (prefill widths keep their paths): no operand is read, so any arrays stand in
        for (arms) |arm| {
            var d = try decbDense(geo, arm);
            defer d.deinit();
            const mem: decb.Member = .{ .d = &d, .trellis = o.trellis, .suh = o.suh, .svh = o.svh };
            const g17: decb.Gate = .{ .base = sigtab, .row_stride = 2 * HQ * HD, .head_stride = 2 * HD, .off = HD };
            const ok = decbRefused(decb.gatedDecode(s, sigtab, g17, sigtab, mem, HQ, HD, 17)) and
                decbRefused(decb.mergeGatedDecode(s, sigtab, sigtab, g17, sigtab, mem, HQ, 17));
            tally.note(ok);
            std.debug.print("TURBOEXL3_DECB gate n={d} rows=17 arm={s} refused={} {s}\n", .{ n, @tagName(arm), ok, if (ok) "PASS" else "FAIL" });
        }
    }
    std.debug.print("TURBOEXL3_DECB_DONE gate checks={d} {s}\n", .{ tally.checks, if (tally.failed) "FAIL" else "PASS" });
    try testing.expect(!tally.failed);
}

test "turboexl3 gpu decode merges: q_proj's finish writing the query and gate planes equals linear + split, each half contiguous (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x4DEC12);
    const rnd = prng.random();
    var tally: Tally = .{};
    for ([_]u32{ 112, 96 }) |n| {
        const q = try decbProj(a, &g, rnd, HIDDEN, 2 * HQ * HD, n);
        defer for ([_]mlx.mlx_array{ q.trellis, q.suh, q.svh }) |x| g.release(x);
        const geo: dense.Shape = .{ .in = HIDDEN, .out = 2 * HQ * HD, .rate = format.Rate.fromN(n).?, .dec = .mul1, .out_dtype = .bfloat16 };
        var dref = try decbDense(geo, .chain);
        defer dref.deinit();
        for ([_]u32{ 1, 2, 3, 4, 8, 16 }) |rows| {
            const mark = g.mark();
            defer g.resetTo(mark);
            const r = ci(rows);
            const x = try decbBf16(a, &g, rnd, &.{ r, ci(HIDDEN) }, 0, 1, false, true);
            const y = try g.adopt(try dref.linear(s, x, q.trellis, q.suh, q.svh, rows));
            const yq = try g.reshape(y, &.{ 1, r, ci(HQ), ci(2 * HD) });
            var halves: [2]mlx.mlx_array = undefined;
            for (&halves, 0..) |*h, i| {
                const v = try g.slice(yq, &.{ 0, 0, 0, ci(i * HD) }, &.{ 1, r, ci(HQ), ci((i + 1) * HD) }, &.{ 1, 1, 1, 1 });
                var c = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_contiguous(&c, v, false, s));
                h.* = try g.reshape(try g.adopt(c), &.{ r, ci(HQ * HD) });
            }
            for (arms) |arm| {
                var d = try decbDense(geo, arm);
                defer d.deinit();
                const planes = try decb.qPlanes(s, x, rows, .{ .d = &d, .trellis = q.trellis, .suh = q.suh, .svh = q.svh }, HQ, HD);
                const pq = try g.adopt(planes[0]);
                const pg = try g.adopt(planes[1]);
                const dq = try decbDiff(a, &g, halves[0], pq);
                const dg = try decbDiff(a, &g, halves[1], pg);
                tally.note(dq == 0 and dg == 0);
                std.debug.print("TURBOEXL3_DECB q_planes n={d} rows={d} arm={s} queries_differing={d} gate_differing={d}/{d} {s}\n", .{ n, rows, @tagName(arm), dq, dg, rows * HQ * HD, if (dq + dg == 0) "PASS" else "FAIL" });
            }
        }
        // rows 17 refused before any launch (the host keeps linear + split there)
        for (arms) |arm| {
            var d = try decbDense(geo, arm);
            defer d.deinit();
            const ok = decbRefused(decb.qPlanes(s, q.svh, 17, .{ .d = &d, .trellis = q.trellis, .suh = q.suh, .svh = q.svh }, HQ, HD));
            tally.note(ok);
            std.debug.print("TURBOEXL3_DECB q_planes n={d} rows=17 arm={s} refused={} {s}\n", .{ n, @tagName(arm), ok, if (ok) "PASS" else "FAIL" });
        }
    }
    std.debug.print("TURBOEXL3_DECB_DONE q_planes checks={d} {s}\n", .{ tally.checks, if (tally.failed) "FAIL" else "PASS" });
    try testing.expect(!tally.failed);
}

test "turboexl3 gpu decode merges: the indexer projection's finish normalizing and rotating the index queries and pooling the block equals the chain byte for byte (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x4DEC4);
    const rnd = prng.random();
    const OUTI: u32 = (NIDX + 1) * 128;
    const qn = g.keep(try decbBf16(a, &g, rnd, &.{128}, 1.0, 0.5, false, false));
    defer g.release(qn);
    const kn = g.keep(try decbBf16(a, &g, rnd, &.{128}, 1.0, 0.5, false, false));
    defer g.release(kn);
    var tally: Tally = .{};
    // (rows, the block's ring rows): a block completes with `RATIO - ln` of the call's rows
    const Case = struct { rows: u32, ln: u32, offset: c_int };
    const cases = [_]Case{
        .{ .rows = 4, .ln = 0, .offset = 16384 }, .{ .rows = 4, .ln = 1, .offset = 16385 }, .{ .rows = 4, .ln = 2, .offset = 16386 }, .{ .rows = 4, .ln = 3, .offset = 2051 },
        .{ .rows = 1, .ln = 3, .offset = 16387 }, .{ .rows = 2, .ln = 2, .offset = 16386 }, .{ .rows = 2, .ln = 3, .offset = 9999 },  .{ .rows = 3, .ln = 1, .offset = 16385 },
        .{ .rows = 3, .ln = 3, .offset = 16387 }, .{ .rows = 8, .ln = 0, .offset = 4096 },  .{ .rows = 16, .ln = 2, .offset = 30 },
    };
    for ([_]u32{ 80, 112 }) |n| {
        const p = try decbProj(a, &g, rnd, HIDDEN, OUTI, n);
        defer for ([_]mlx.mlx_array{ p.trellis, p.suh, p.svh }) |x| g.release(x);
        const geo: dense.Shape = .{ .in = HIDDEN, .out = OUTI, .rate = format.Rate.fromN(n).?, .dec = .mul1, .out_dtype = .bfloat16 };
        var dref = try decbDense(geo, .chain);
        defer dref.deinit();
        for (cases) |c| {
            const mark = g.mark();
            defer g.resetTo(mark);
            const r = ci(c.rows);
            const x = try decbBf16(a, &g, rnd, &.{ r, ci(HIDDEN) }, 0, 1, false, true);
            const y = try g.adopt(try dref.linear(s, x, p.trellis, p.suh, p.svh, c.rows));
            const y3 = try g.reshape(y, &.{ 1, r, ci(OUTI) });
            // the index queries: slice, reshape, rms_norm (MLX's), transpose, rope (MLX's)
            const qv = try g.slice(y3, &.{ 0, 0, 0 }, &.{ 1, r, ci(NIDX * 128) }, &.{ 1, 1, 1 });
            const q4 = try g.reshape(qv, &.{ 1, r, ci(NIDX), 128 });
            var qnorm = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_rms_norm(&qnorm, q4, qn, EPS, s));
            const qt = try g.transposeAxes(try g.adopt(qnorm), &.{ 0, 2, 1, 3 });
            var qr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_rope(&qr, qt, RD, false, .{ .value = THETA, .has_value = true }, 1.0, c.offset, .{ .ctx = null }, s));
            const want_q = try g.adopt(qr);
            // the pooled key: the ring's leftover rows ++ the call's first rows, f32 mean, bf16, rms_norm, the composed rotation
            const ring = try decbBf16(a, &g, rnd, &.{ 1, ci(@max(c.ln, 1)), 128 }, 0, 1, false, true);
            const kraw = try g.slice(y3, &.{ 0, 0, ci(NIDX * 128) }, &.{ 1, ci(RATIO - c.ln), ci(OUTI) }, &.{ 1, 1, 1 });
            const kb_flat = if (c.ln > 0) try g.concat(&.{ ring, kraw }, 1) else kraw;
            const kb4 = try g.reshape(kb_flat, &.{ 1, 1, ci(RATIO), 128 });
            const pooled_f = try g.mean(try g.astype(kb4, .float32), 2, false);
            var pn = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_rms_norm(&pn, try g.astype(pooled_f, .bfloat16), kn, EPS, s));
            const pn4 = try g.reshape(try g.adopt(pn), &.{ 1, 1, 1, 128 });
            const pcos = try decbBf16(a, &g, rnd, &.{ 1, RD }, 0, 1, false, false);
            const psin = try decbBf16(a, &g, rnd, &.{ 1, RD }, 0, 1, false, false);
            const xr = try g.slice(pn4, &.{ 0, 0, 0, 0 }, &.{ 1, 1, 1, RD }, &.{ 1, 1, 1, 1 });
            const xp = try g.slice(pn4, &.{ 0, 0, 0, RD }, &.{ 1, 1, 1, 128 }, &.{ 1, 1, 1, 1 });
            const x1 = try g.slice(xr, &.{ 0, 0, 0, 0 }, &.{ 1, 1, 1, @divExact(RD, 2) }, &.{ 1, 1, 1, 1 });
            const x2 = try g.slice(xr, &.{ 0, 0, 0, @divExact(RD, 2) }, &.{ 1, 1, 1, RD }, &.{ 1, 1, 1, 1 });
            const rot = try g.concat(&.{ try g.neg(x2), x1 }, -1);
            const roped = try g.add(try g.mul(xr, pcos), try g.mul(rot, psin));
            const want_p = try g.reshape(try g.concat(&.{ roped, xp }, -1), &.{ 1, 1, 128 });
            const angles = try decbAngles(a, &g, s, c.rows, c.offset);
            for (arms) |arm| {
                var d = try decbDense(geo, arm);
                defer d.deinit();
                for ([_][2]bool{ .{ true, true }, .{ true, false }, .{ false, true } }) |mode| {
                    const ep: decb.IdxEpilogue = .{
                        .heads = NIDX,
                        .rope_dims = @intCast(RD),
                        .ratio = RATIO,
                        .qepi = mode[0],
                        .q_norm = qn,
                        .angles = angles,
                        .pool = mode[1],
                        .k_norm = kn,
                        .ring = ring,
                        .ln = c.ln,
                        .pcos = pcos,
                        .psin = psin,
                        .eps = EPS,
                    };
                    const out = decb.indexerQk(s, x, c.rows, .{ .d = &d, .trellis = p.trellis, .suh = p.suh, .svh = p.svh }, ep) catch |e| {
                        // the pooled key needs the block's rows in DEC-ONE's first row chunk
                        if (e == error.Exl3EngineNotImplemented and arm != .chain and mode[1]) {
                            std.debug.print("TURBOEXL3_DECB indexer n={d} rows={d} ln={d} arm={s} qepi={} pool={} refused (block rows past the first chunk)\n", .{ n, c.rows, c.ln, @tagName(arm), mode[0], mode[1] });
                            continue;
                        }
                        return e;
                    };
                    const qk = try g.adopt(out.qk);
                    const dqk = try decbDiff(a, &g, y, qk);
                    const dq: usize = if (mode[0]) try decbDiff(a, &g, want_q, try g.adopt(out.q_rope)) else 0;
                    const dp: usize = if (mode[1]) try decbDiff(a, &g, want_p, try g.adopt(out.pooled)) else 0;
                    const ok = dqk == 0 and dq == 0 and dp == 0;
                    tally.note(ok);
                    std.debug.print("TURBOEXL3_DECB indexer n={d} rows={d} ln={d} offset={d} arm={s} qepi={} pool={} qk_differing={d} q_rope_differing={d} pooled_differing={d} {s}\n", .{ n, c.rows, c.ln, c.offset, @tagName(arm), mode[0], mode[1], dqk, dq, dp, if (ok) "PASS" else "FAIL" });
                }
            }
        }
        // rows 17 refused before any launch (the host keeps its indexer chain there)
        for (arms) |arm| {
            var d = try decbDense(geo, arm);
            defer d.deinit();
            const ep: decb.IdxEpilogue = .{ .heads = NIDX, .rope_dims = @intCast(RD), .ratio = RATIO, .qepi = true, .q_norm = qn, .angles = qn, .pool = true, .k_norm = kn, .ring = kn, .ln = 0, .pcos = kn, .psin = kn, .eps = EPS };
            const ok = decbRefused(decb.indexerQk(s, p.svh, 17, .{ .d = &d, .trellis = p.trellis, .suh = p.suh, .svh = p.svh }, ep));
            tally.note(ok);
            std.debug.print("TURBOEXL3_DECB indexer n={d} rows=17 arm={s} refused={} {s}\n", .{ n, @tagName(arm), ok, if (ok) "PASS" else "FAIL" });
        }
    }
    std.debug.print("TURBOEXL3_DECB_DONE indexer checks={d} {s}\n", .{ tally.checks, if (tally.failed) "FAIL" else "PASS" });
    try testing.expect(!tally.failed);
}
