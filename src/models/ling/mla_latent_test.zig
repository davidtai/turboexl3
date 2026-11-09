//! Ling-3.1-flash's MLA decode in the latent (`mla_latent.zig`). CPU (always): the host reference's absorbed form equals
//! the expanded form (kv_b's key and value halves applied to every cached latent, the naive attention), in f64. GPU
//! (TURBOEXL3_GPU=1, else skipped), at Ling's widths (64 heads, latent 512, rope 64, DN = DV = 128):
//! - `decodeLatent` is within tolerance of the host's f64 absorbed attention: operands f32 (values bf16's) with an f32
//!   output, and bf16 throughout, at rows 1 to 16 and key counts across the sub-chunk (32) and key-group (256) edges;
//! - a call's every row has the bits of a one-row call at its position (rows 4, 6, 16), and cache views read through
//!   their strides (the latent and the rope key interleaved in one buffer) the bits of contiguous ones.
//! - no worse than the served path against the same f64 reference: the expanded cache (kv_b over every cached latent
//!   in bf16, the shared rope key broadcast, MLX's one-row SDPA for each row over the keys up to it, as the exact
//!   verify's rows run), bf16, rows 1 to 4 at decode and verify widths, keys to the served prompt's 18,480.
//! - the outside plugin turboexl3_ling's served operand: kv_b as its loader holds it (the f16 checkpoint's [in, out]
//!   view cast to bf16) transposed back to its stored rows, row-contiguous over the loaded buffer and taken by the
//!   plan; `decodeLatent` over a wider cache buffer's first rows within 2 % of the plugin's expanded attention, which
//!   reads that view through a plain matmul, at the standard prompt's 18,380 keys.
//! - launch 2's variants (`mla_latent.Attn`: rows2, rows4, stage, block) give launch 2's words: each call's output and
//!   launch 2's partials (pm, pl, po) word for word against `decodeLatentParts` on launch 2 as it is, at rows 1 to 4 and
//!   keys 1, 2, 31, 32, 255, 256, 257, 4096, 16,669 and 19,404 (the MTP head's cache at the standard request), on the
//!   plugin's operands (kv_b through the loader's rows view, the head's cache as the first T rows of a wider buffer, the
//!   query halves as views of the transposed heads) and on two strided caches (the latent and the rope key interleaved
//!   in one buffer; each transposed, a key's words a column apart); a planted fault (a key row zeroed, one flipped bit)
//!   must differ.
//! Lines: TURBOEXL3_LING_MLA_LATENT, TURBOEXL3_LING_MLA_LATENT_ROWS, TURBOEXL3_LING_MLA_LATENT_SERVED,
//! TURBOEXL3_LING_MLA_LATENT_LOADER_KVB, TURBOEXL3_LING_MLA_LATENT_LOADER, TURBOEXL3_LING_MLA_LATENT_VARIANT (_RED,
//! _DONE). The variants' test prints an MLX error (a launch MLX refuses) as its variant's FAIL line and goes on.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const ref = @import("../../core/ref.zig");
const ml = @import("mla_latent.zig");

const testing = std.testing;

const DN: usize = ml.dn;
const DR: usize = ml.dr;
const DV: usize = ml.dv;
const DL: usize = ml.dl;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn bfExact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
}

/// One call's operands on the host (f32, values bf16's): q_nope [H, S, DN], q_rot [H, S, DR], the cache lat [T, DL] and
/// rot [T, DR], kv_b [H (DN + DV), DL]
const Ops = struct {
    h: usize,
    s: usize,
    t: usize,
    qn: []f32,
    qr: []f32,
    lat: []f32,
    rot: []f32,
    w: []f32,
    scale: f64,

    fn init(a: std.mem.Allocator, rnd: std.Random, h: usize, s: usize, t: usize, scale: f64) !Ops {
        const o: Ops = .{
            .h = h,
            .s = s,
            .t = t,
            .qn = try a.alloc(f32, h * s * DN),
            .qr = try a.alloc(f32, h * s * DR),
            .lat = try a.alloc(f32, t * DL),
            .rot = try a.alloc(f32, t * DR),
            .w = try a.alloc(f32, h * (DN + DV) * DL),
            .scale = scale,
        };
        for (o.qn) |*v| v.* = bfExact(rnd.floatNorm(f32));
        for (o.qr) |*v| v.* = bfExact(rnd.floatNorm(f32));
        for (o.lat) |*v| v.* = bfExact(rnd.floatNorm(f32));
        for (o.rot) |*v| v.* = bfExact(rnd.floatNorm(f32));
        const ws: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(DL)));
        for (o.w) |*v| v.* = bfExact(rnd.floatNorm(f32) * ws);
        return o;
    }

    fn deinit(o: *Ops, a: std.mem.Allocator) void {
        inline for (.{ o.qn, o.qr, o.lat, o.rot, o.w }) |x| a.free(x);
    }
};

/// The absorbed attention in f64: out [H, S, DV]. Row (h, s) sits at key position T - S + s and sees the keys up to it.
fn refAbsorbed(a: std.mem.Allocator, o: Ops, out: []f64) !void {
    const ql = try a.alloc(f64, DL);
    defer a.free(ql);
    const sc = try a.alloc(f64, o.t);
    defer a.free(sc);
    const ol = try a.alloc(f64, DL);
    defer a.free(ol);
    for (0..o.h) |h| for (0..o.s) |s| {
        const qn = o.qn[(h * o.s + s) * DN ..][0..DN];
        const qr = o.qr[(h * o.s + s) * DR ..][0..DR];
        @memset(ql, 0);
        for (0..DN) |d| {
            const wr = o.w[(h * (DN + DV) + d) * DL ..][0..DL];
            for (ql, wr) |*q, wv| q.* += @as(f64, qn[d]) * wv;
        }
        const lim = o.t - o.s + s;
        var mx: f64 = -std.math.inf(f64);
        for (0..lim + 1) |t| {
            var acc: f64 = 0;
            for (ql, o.lat[t * DL ..][0..DL]) |q, l| acc += q * l;
            for (qr, o.rot[t * DR ..][0..DR]) |q, r| acc += @as(f64, q) * r;
            sc[t] = acc * o.scale;
            mx = @max(mx, sc[t]);
        }
        var l_sum: f64 = 0;
        @memset(ol, 0);
        for (0..lim + 1) |t| {
            const p = @exp(sc[t] - mx);
            l_sum += p;
            for (ol, o.lat[t * DL ..][0..DL]) |*v, l| v.* += p * l;
        }
        for (0..DV) |e| {
            const wr = o.w[(h * (DN + DV) + DN + e) * DL ..][0..DL];
            var acc: f64 = 0;
            for (ol, wr) |v, wv| acc += v / l_sum * wv;
            out[(h * o.s + s) * DV + e] = acc;
        }
    };
}

/// The expanded (naive) attention in f64: every cached latent through kv_b's halves into per-head keys and values
fn refExpanded(a: std.mem.Allocator, o: Ops, out: []f64) !void {
    const kn = try a.alloc(f64, o.t * o.h * DN);
    defer a.free(kn);
    const vv = try a.alloc(f64, o.t * o.h * DV);
    defer a.free(vv);
    for (0..o.t) |t| for (0..o.h) |h| {
        const l = o.lat[t * DL ..][0..DL];
        for (0..DN + DV) |d| {
            var acc: f64 = 0;
            for (l, o.w[(h * (DN + DV) + d) * DL ..][0..DL]) |x, wv| acc += @as(f64, x) * wv;
            if (d < DN) kn[(t * o.h + h) * DN + d] = acc else vv[(t * o.h + h) * DV + d - DN] = acc;
        }
    };
    const sc = try a.alloc(f64, o.t);
    defer a.free(sc);
    for (0..o.h) |h| for (0..o.s) |s| {
        const lim = o.t - o.s + s;
        var mx: f64 = -std.math.inf(f64);
        for (0..lim + 1) |t| {
            var acc: f64 = 0;
            for (o.qn[(h * o.s + s) * DN ..][0..DN], kn[(t * o.h + h) * DN ..][0..DN]) |q, k| acc += @as(f64, q) * k;
            for (o.qr[(h * o.s + s) * DR ..][0..DR], o.rot[t * DR ..][0..DR]) |q, r| acc += @as(f64, q) * r;
            sc[t] = acc * o.scale;
            mx = @max(mx, sc[t]);
        }
        var l_sum: f64 = 0;
        for (0..lim + 1) |t| l_sum += @exp(sc[t] - mx);
        for (0..DV) |e| {
            var acc: f64 = 0;
            for (0..lim + 1) |t| acc += @exp(sc[t] - mx) / l_sum * vv[(t * o.h + h) * DV + e];
            out[(h * o.s + s) * DV + e] = acc;
        }
    };
}

test "turboexl3 ling mla latent reference: the absorbed attention equals the expanded one in f64 (kv_b's halves, the shared rope key, each row's own key positions)" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x3A7E);
    var o = try Ops.init(a, prng.random(), 4, 3, 40, 1.0 / @sqrt(192.0));
    defer o.deinit(a);
    const x = try a.alloc(f64, o.h * o.s * DV);
    defer a.free(x);
    const y = try a.alloc(f64, o.h * o.s * DV);
    defer a.free(y);
    try refAbsorbed(a, o, x);
    try refExpanded(a, o, y);
    var worst: f64 = 0;
    var rms: f64 = 0;
    for (x, y) |u, v| {
        worst = @max(worst, @abs(u - v));
        rms += v * v;
    }
    rms = @sqrt(rms / @as(f64, @floatFromInt(y.len)));
    try testing.expect(rms > 0 and worst <= 1e-12 * @max(rms, 1.0));
}

// ── GPU ──

/// The operands as device arrays in `dt` (q and the cache), kv_b in `wt`; `interleave`: the latent and the rope key one
/// [1, 1, T, DL + DR] buffer, the views its column ranges (strided)
const Dev = struct { qn: mlx.mlx_array, qr: mlx.mlx_array, lat: mlx.mlx_array, rot: mlx.mlx_array, w: mlx.mlx_array };

fn devOps(g: *G, o: Ops, dt: mlx.mlx_dtype, wt: mlx.mlx_dtype, interleave: bool) !Dev {
    const qn = try g.astype(try g.hostArray(std.mem.sliceAsBytes(o.qn), &.{ 1, ci(o.h), ci(o.s), ci(DN) }, .float32), dt);
    const qr = try g.astype(try g.hostArray(std.mem.sliceAsBytes(o.qr), &.{ 1, ci(o.h), ci(o.s), ci(DR) }, .float32), dt);
    const w = try g.astype(try g.hostArray(std.mem.sliceAsBytes(o.w), &.{ ci(o.h * (DN + DV)), ci(DL) }, .float32), wt);
    const lat_c = try g.astype(try g.hostArray(std.mem.sliceAsBytes(o.lat), &.{ 1, 1, ci(o.t), ci(DL) }, .float32), dt);
    const rot_c = try g.astype(try g.hostArray(std.mem.sliceAsBytes(o.rot), &.{ 1, 1, ci(o.t), ci(DR) }, .float32), dt);
    if (!interleave) return .{ .qn = qn, .qr = qr, .lat = lat_c, .rot = rot_c, .w = w };
    const both = try g.concat(&.{ lat_c, rot_c }, 3);
    try mlx.check(mlx.mlx_array_eval(both));
    const lat = try g.slice(both, &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(o.t), ci(DL) }, &.{ 1, 1, 1, 1 });
    const rot = try g.slice(both, &.{ 0, 0, 0, ci(DL) }, &.{ 1, 1, ci(o.t), ci(DL + DR) }, &.{ 1, 1, 1, 1 });
    return .{ .qn = qn, .qr = qr, .lat = lat, .rot = rot, .w = w };
}

fn run(g: *G, s: mlx.mlx_stream, d: Dev, scale: f64, out: []f32) !void {
    const y = try g.adopt(try ml.decodeLatent(s, d.qn, d.qr, d.lat, d.rot, d.w, @floatCast(scale)));
    _ = try g.hostF32(try g.astype(y, .float32), out);
}

/// MLX's SDPA and broadcast (mlx-c's signatures): the consumer's binding where its `sdk.mlx` declares them, else these
/// declarations of the same symbols.
const sdpa = if (@hasDecl(mlx, "mlx_fast_scaled_dot_product_attention")) mlx.mlx_fast_scaled_dot_product_attention else struct {
    extern "c" fn mlx_fast_scaled_dot_product_attention(res: *mlx.mlx_array, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mask_mode: [*:0]const u8, mask_arr: mlx.mlx_array, sinks: mlx.mlx_array, force_fused: bool, s: mlx.mlx_stream) c_int;
}.mlx_fast_scaled_dot_product_attention;
const broadcastTo = if (@hasDecl(mlx, "mlx_broadcast_to")) mlx.mlx_broadcast_to else struct {
    extern "c" fn mlx_broadcast_to(res: *mlx.mlx_array, a: mlx.mlx_array, shape: [*]const c_int, shape_num: usize, s: mlx.mlx_stream) c_int;
}.mlx_broadcast_to;

/// The served (expanded-cache) attention of the call's rows, as the host's MLA serves it today: kv_b over every cached
/// latent in the activation dtype (`x @ kv_bᵀ`), the heads' keys and values cut from it, the shared rope key broadcast
/// to every head, keys = concat(k_nope, rope), the values made contiguous, then each row alone over the keys up to its
/// position through MLX's one-row SDPA (the exact verify's rows). out [H, S, DV] f32.
fn servedExpanded(g: *G, s: mlx.mlx_stream, a: std.mem.Allocator, d: Dev, o: Ops, scale: f64, out: []f32) !void {
    const h = ci(o.h);
    const t = ci(o.t);
    const lat2 = try g.reshape(d.lat, &.{ t, ci(DL) });
    const kv = try g.matmul(lat2, try g.transposeAxes(d.w, &.{ 1, 0 }));
    const kvh = try g.transposeAxes(try g.reshape(kv, &.{ 1, t, h, ci(DN + DV) }), &.{ 0, 2, 1, 3 });
    const kn = try g.slice(kvh, &.{ 0, 0, 0, 0 }, &.{ 1, h, t, ci(DN) }, &.{ 1, 1, 1, 1 });
    const vt = try g.slice(kvh, &.{ 0, 0, 0, ci(DN) }, &.{ 1, h, t, ci(DN + DV) }, &.{ 1, 1, 1, 1 });
    var vc = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&vc, vt, false, s));
    const v = try g.adopt(vc);
    var rb = mlx.mlx_array_new();
    try mlx.check(broadcastTo(&rb, d.rot, &[_]c_int{ 1, h, t, ci(DR) }, 4, s));
    const keys = try g.concat(&.{ kn, try g.adopt(rb) }, 3);
    const q = try g.concat(&.{ d.qn, d.qr }, 3);
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    const one = try a.alloc(f32, o.h * DV);
    defer a.free(one);
    for (0..o.s) |r| {
        const pos = o.t - o.s + r;
        const q1 = try g.slice(q, &.{ 0, 0, ci(r), 0 }, &.{ 1, h, ci(r + 1), ci(DN + DR) }, &.{ 1, 1, 1, 1 });
        const k1 = try g.slice(keys, &.{ 0, 0, 0, 0 }, &.{ 1, h, ci(pos + 1), ci(DN + DR) }, &.{ 1, 1, 1, 1 });
        const v1 = try g.slice(v, &.{ 0, 0, 0, 0 }, &.{ 1, h, ci(pos + 1), ci(DV) }, &.{ 1, 1, 1, 1 });
        var y = mlx.mlx_array_new();
        try mlx.check(sdpa(&y, q1, k1, v1, @floatCast(scale), "", none, .{ .ctx = null }, false, s));
        _ = try g.hostF32(try g.astype(try g.adopt(y), .float32), one);
        for (0..o.h) |hh| @memcpy(out[(hh * o.s + r) * DV ..][0..DV], one[hh * DV ..][0..DV]);
    }
}

/// The RMS of got - want over the reference's RMS
fn rmsRatio(got: []const f32, want: []const f32, rms_ref: f64) f64 {
    var ss: f64 = 0;
    for (got, want) |u, w| ss += (@as(f64, u) - w) * (@as(f64, u) - w);
    const r = @sqrt(ss / @as(f64, @floatFromInt(want.len)));
    return if (rms_ref > 0) r / rms_ref else r;
}

test "turboexl3 gpu ling mla latent: decodeLatent at Ling's widths is within tolerance of the host's f64 absorbed attention, f32 and bf16, rows 1 to 16, keys across the sub-chunk and key-group edges (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const cases = [_][2]usize{ .{ 1, 1 }, .{ 31, 1 }, .{ 32, 1 }, .{ 33, 2 }, .{ 256, 1 }, .{ 257, 6 }, .{ 1000, 3 }, .{ 2049, 16 }, .{ 600, 16 } };
    var failed: usize = 0;
    for (cases) |c| {
        const t = c[0];
        const rows = c[1];
        var prng = std.Random.DefaultPrng.init(0x1A7E ^ (@as(u64, t) << 8) ^ rows);
        // Ling's scale (1 / sqrt(192)), and a peaked one (8x) that puts most of a row's weight on few keys
        for ([_]f64{ 1.0 / @sqrt(192.0), 8.0 / @sqrt(192.0) }) |scale| {
            var o = try Ops.init(a, prng.random(), 64, rows, t, scale);
            defer o.deinit(a);
            const want64 = try a.alloc(f64, o.h * o.s * DV);
            defer a.free(want64);
            try refAbsorbed(a, o, want64);
            const want = try a.alloc(f32, want64.len);
            defer a.free(want);
            for (want, want64) |*u, v| u.* = @floatCast(v);
            const got = try a.alloc(f32, want.len);
            defer a.free(got);
            for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dt| {
                const mark = g.mark();
                defer g.resetTo(mark);
                try run(&g, s, try devOps(&g, o, dt, dt, false), scale, got);
                const e = if (dt == .float32) ref.compare(got, want) else ref.compareRounded(got, want, 1.0 / 256.0);
                const bar: f64 = if (dt == .float32) 1e-4 else 1e-2;
                const ok = std.math.isFinite(e.max_abs) and e.max_ratio <= bar;
                std.debug.print("TURBOEXL3_LING_MLA_LATENT keys={d} rows={d} scale={d:.4} x={s} max_abs={e:.3} rms={e:.3} ratio={e:.3} bar={e:.1} {s}\n", .{ t, rows, scale, @tagName(dt), e.max_abs, e.rms_ref, e.max_ratio, bar, if (ok) "PASS" else "FAIL" });
                failed += @intFromBool(!ok);
            }
        }
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

test "turboexl3 gpu ling mla latent: a call's every row has the bits of a one-row call at its position, and strided cache views the bits of contiguous ones (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const cases = [_][2]usize{ .{ 300, 4 }, .{ 513, 6 }, .{ 64, 16 }, .{ 257, 2 } };
    var failed: usize = 0;
    for (cases) |c| {
        const t = c[0];
        const rows = c[1];
        var prng = std.Random.DefaultPrng.init(0x2B0E ^ (@as(u64, t) << 8) ^ rows);
        var o = try Ops.init(a, prng.random(), 64, rows, t, 1.0 / @sqrt(192.0));
        defer o.deinit(a);
        const n = o.h * o.s * DV;
        const all = try a.alloc(f32, n);
        defer a.free(all);
        const one = try a.alloc(f32, o.h * DV);
        defer a.free(one);
        const strided = try a.alloc(f32, n);
        defer a.free(strided);
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
            const mark = g.mark();
            defer g.resetTo(mark);
            const d = try devOps(&g, o, dt, .bfloat16, false);
            try run(&g, s, d, o.scale, all);
            // each row alone: its query row, the cache up to its position
            var d_rows: usize = 0;
            for (0..rows) |r| {
                const pos = t - rows + r;
                const q1 = try g.slice(d.qn, &.{ 0, 0, ci(r), 0 }, &.{ 1, ci(o.h), ci(r + 1), ci(DN) }, &.{ 1, 1, 1, 1 });
                const r1 = try g.slice(d.qr, &.{ 0, 0, ci(r), 0 }, &.{ 1, ci(o.h), ci(r + 1), ci(DR) }, &.{ 1, 1, 1, 1 });
                const l1 = try g.slice(d.lat, &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(pos + 1), ci(DL) }, &.{ 1, 1, 1, 1 });
                const k1 = try g.slice(d.rot, &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(pos + 1), ci(DR) }, &.{ 1, 1, 1, 1 });
                try run(&g, s, .{ .qn = q1, .qr = r1, .lat = l1, .rot = k1, .w = d.w }, o.scale, one);
                for (0..o.h) |h| for (0..DV) |e| {
                    const u: u32 = @bitCast(all[(h * rows + r) * DV + e]);
                    const v: u32 = @bitCast(one[h * DV + e]);
                    d_rows += @intFromBool(u != v);
                };
            }
            // the cache as column views of one interleaved buffer
            try run(&g, s, try devOps(&g, o, dt, .bfloat16, true), o.scale, strided);
            var d_str: usize = 0;
            for (all, strided) |u, v| d_str += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
            const ok = d_rows == 0 and d_str == 0;
            std.debug.print("TURBOEXL3_LING_MLA_LATENT_ROWS keys={d} rows={d} x={s} rows_vs_alone={d}/{d} strided_vs_contiguous={d}/{d} {s}\n", .{ t, rows, @tagName(dt), d_rows, n, d_str, n, if (ok) "PASS" else "FAIL" });
            failed += @intFromBool(!ok);
        }
    }
    try testing.expectEqual(@as(usize, 0), failed);
}


test "turboexl3 gpu ling mla latent: no worse than the served expanded attention against the same f64 reference, bf16, the exact verify's rows 1 to 4 at keys to 18,480 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const cases = [_][2]usize{ .{ 1, 1 }, .{ 300, 1 }, .{ 1000, 3 }, .{ 4100, 4 }, .{ 18480, 3 }, .{ 18480, 4 } };
    var failed: usize = 0;
    for (cases) |c| {
        const t = c[0];
        const rows = c[1];
        var prng = std.Random.DefaultPrng.init(0x5E7D ^ (@as(u64, t) << 8) ^ rows);
        for ([_]f64{ 1.0 / @sqrt(192.0), 8.0 / @sqrt(192.0) }) |scale| {
            var o = try Ops.init(a, prng.random(), 64, rows, t, scale);
            defer o.deinit(a);
            const want64 = try a.alloc(f64, o.h * o.s * DV);
            defer a.free(want64);
            try refAbsorbed(a, o, want64);
            const want = try a.alloc(f32, want64.len);
            defer a.free(want);
            for (want, want64) |*u, v| u.* = @floatCast(v);
            const got = try a.alloc(f32, want.len);
            defer a.free(got);
            const served = try a.alloc(f32, want.len);
            defer a.free(served);
            const mark = g.mark();
            defer g.resetTo(mark);
            const d = try devOps(&g, o, .bfloat16, .bfloat16, false);
            try run(&g, s, d, scale, got);
            try servedExpanded(&g, s, a, d, o, scale, served);
            const ef = ref.compareRounded(got, want, 1.0 / 256.0);
            const es = ref.compareRounded(served, want, 1.0 / 256.0);
            const rf = rmsRatio(got, want, ef.rms_ref);
            const rs = rmsRatio(served, want, es.rms_ref);
            // no worse than the served path: the RMS within 10%, the worst element within 25% (each with a floor at
            // the output's own rounding scale)
            const ok = std.math.isFinite(ef.max_abs) and std.math.isFinite(rf) and
                rf <= @max(1.10 * rs, rs + 1e-4) and ef.max_ratio <= @max(1.25 * es.max_ratio, es.max_ratio + 1.0 / 512.0);
            std.debug.print("TURBOEXL3_LING_MLA_LATENT_SERVED keys={d} rows={d} scale={d:.4} fused_rms={e:.3} served_rms={e:.3} fused_max={e:.3} served_max={e:.3} {s}\n", .{ t, rows, scale, rf, rs, ef.max_ratio, es.max_ratio, if (ok) "PASS" else "FAIL" });
            failed += @intFromBool(!ok);
        }
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── the plugin's served operand ──

/// kv_b as the outside plugin turboexl3_ling hands it to `decodeLatent`. The checkpoint stores kv_b f16
/// [H (DN + DV), DL]. The plugin's loader (its copied trunk.zig `proj`) holds a dense projection as the [DL, H (DN + DV)]
/// view a plain matmul reads (`transposed`), then casts a plain f16 weight to bf16 and evaluates it (`lin`). `held` is
/// that array. The plugin's `kvRows` (src/ling_mla_latent.zig) transposes it back to the stored rows: `rows`, what its
/// `latentAttn` passes here. `ckpt_bf16` is the checkpoint cast to bf16, the words `rows` must hold.
const LoaderKvB = struct { held: mlx.mlx_array, rows: mlx.mlx_array, ckpt_bf16: mlx.mlx_array };

fn loaderKvB(g: *G, a: std.mem.Allocator, rnd: std.Random, h: usize) !LoaderKvB {
    const host = try a.alloc(f32, h * (DN + DV) * DL);
    defer a.free(host);
    const ws: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(DL)));
    for (host) |*v| v.* = rnd.floatNorm(f32) * ws;
    const ckpt = try g.astype(try g.hostArray(std.mem.sliceAsBytes(host), &.{ ci(h * (DN + DV)), ci(DL) }, .float32), .float16);
    try mlx.check(mlx.mlx_array_eval(ckpt));
    const held = try g.astype(try g.transposeAxes(ckpt, &.{ 1, 0 }), .bfloat16);
    try mlx.check(mlx.mlx_array_eval(held));
    const rows = try g.transposeAxes(held, &.{ 1, 0 });
    try mlx.check(mlx.mlx_array_eval(rows));
    const ckpt_bf16 = try g.astype(ckpt, .bfloat16);
    try mlx.check(mlx.mlx_array_eval(ckpt_bf16));
    return .{ .held = held, .rows = rows, .ckpt_bf16 = ckpt_bf16 };
}

/// normal(0, 1) on the host, as a device array [shape] in `dt`, evaluated
fn devRandn(g: *G, a: std.mem.Allocator, rnd: std.Random, shape: []const c_int, dt: mlx.mlx_dtype) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @as(usize, @intCast(d));
    const host = try a.alloc(f32, n);
    defer a.free(host);
    for (host) |*v| v.* = rnd.floatNorm(f32);
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(host), shape, .float32), dt);
    try mlx.check(mlx.mlx_array_eval(x));
    return x;
}

/// The plugin's expanded-cache attention of the call's rows, the head's path without its latent switch: the latent rows
/// through the loader's `held` view (`self.mm(&fa.kv_b, latent)`, a plain `x @ held`), the heads' keys and values cut
/// from it, the shared rope key broadcast, then each row alone over the keys up to its position through MLX's one-row
/// SDPA. out [H, S, DV] f32.
fn servedLoader(g: *G, s: mlx.mlx_stream, a: std.mem.Allocator, qn: mlx.mlx_array, qr: mlx.mlx_array, lat: mlx.mlx_array, rot: mlx.mlx_array, held: mlx.mlx_array, h: usize, rows: usize, t: usize, scale: f64, out: []f32) !void {
    const hc = ci(h);
    const tc = ci(t);
    const kv = try g.matmul(try g.reshape(lat, &.{ tc, ci(DL) }), held);
    const kvh = try g.transposeAxes(try g.reshape(kv, &.{ 1, tc, hc, ci(DN + DV) }), &.{ 0, 2, 1, 3 });
    const kn = try g.slice(kvh, &.{ 0, 0, 0, 0 }, &.{ 1, hc, tc, ci(DN) }, &.{ 1, 1, 1, 1 });
    const vt = try g.slice(kvh, &.{ 0, 0, 0, ci(DN) }, &.{ 1, hc, tc, ci(DN + DV) }, &.{ 1, 1, 1, 1 });
    var vc = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&vc, vt, false, s));
    const v = try g.adopt(vc);
    var rb = mlx.mlx_array_new();
    try mlx.check(broadcastTo(&rb, rot, &[_]c_int{ 1, hc, tc, ci(DR) }, 4, s));
    const keys = try g.concat(&.{ kn, try g.adopt(rb) }, 3);
    const q = try g.concat(&.{ qn, qr }, 3);
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    const one = try a.alloc(f32, h * DV);
    defer a.free(one);
    for (0..rows) |r| {
        const pos = t - rows + r;
        const q1 = try g.slice(q, &.{ 0, 0, ci(r), 0 }, &.{ 1, hc, ci(r + 1), ci(DN + DR) }, &.{ 1, 1, 1, 1 });
        const k1 = try g.slice(keys, &.{ 0, 0, 0, 0 }, &.{ 1, hc, ci(pos + 1), ci(DN + DR) }, &.{ 1, 1, 1, 1 });
        const v1 = try g.slice(v, &.{ 0, 0, 0, 0 }, &.{ 1, hc, ci(pos + 1), ci(DV) }, &.{ 1, 1, 1, 1 });
        var y = mlx.mlx_array_new();
        try mlx.check(sdpa(&y, q1, k1, v1, @floatCast(scale), "", none, .{ .ctx = null }, false, s));
        _ = try g.hostF32(try g.astype(try g.adopt(y), .float32), one);
        for (0..h) |hh| @memcpy(out[(hh * rows + r) * DV ..][0..DV], one[hh * DV ..][0..DV]);
    }
}

fn yes(b: bool) []const u8 {
    return if (b) "true" else "false";
}

/// ||got - want|| / ||want|| and max |got - want| / max |want|
fn relErr(got: []const f32, want: []const f32) [2]f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    var max_d: f64 = 0;
    var max_w: f64 = 0;
    for (got, want) |u, w| {
        const d = @as(f64, u) - w;
        num += d * d;
        den += @as(f64, w) * w;
        max_d = @max(max_d, @abs(d));
        max_w = @max(max_w, @abs(@as(f64, w)));
    }
    return .{ @sqrt(num / den), max_d / max_w };
}

test "turboexl3 gpu ling mla latent: the outside plugin's served operand, kv_b as its loader holds it (the f16 checkpoint's [in, out] view cast to bf16) transposed back to its stored rows, row-contiguous over the loaded buffer, taken by the plan, and decodeLatent over a wider buffer's first rows within 2 % of the plugin's expanded attention: 18,380 keys at 1 row, 18,383 at 4, an 18,381 rollback at 2 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const h: usize = 64;
    var prng = std.Random.DefaultPrng.init(0x10AD_E4B0);
    const kvb = try loaderKvB(&g, a, prng.random(), h);
    var failed: usize = 0;
    {
        const hs = mlx.getShape(kvb.held);
        const rs = mlx.getShape(kvb.rows);
        const hst = mlx.mlx_array_strides(kvb.held);
        const rst = mlx.mlx_array_strides(kvb.rows);
        const ph = mlx.mlx_array_data_uint8(kvb.held) orelse return error.NoData;
        const pr = mlx.mlx_array_data_uint8(kvb.rows) orelse return error.NoData;
        const pc = mlx.mlx_array_data_uint8(kvb.ckpt_bf16) orelse return error.NoData;
        const nw = mlx.mlx_array_size(kvb.rows);
        // `rows` is row-contiguous, so its buffer in order is its words; `ckpt_bf16` is a fresh row-major array
        var differ: usize = 0;
        for (0..nw) |i| differ += @intFromBool(!std.mem.eql(u8, pr[i * 2 ..][0..2], pc[i * 2 ..][0..2]));
        const same_buffer = @intFromPtr(ph) == @intFromPtr(pr);
        const ok = hs[0] == ci(DL) and hs[1] == ci(h * (DN + DV)) and hst[0] == 1 and hst[1] == DL and
            rs[0] == ci(h * (DN + DV)) and rs[1] == ci(DL) and rst[0] == DL and rst[1] == 1 and same_buffer and differ == 0;
        std.debug.print("TURBOEXL3_LING_MLA_LATENT_LOADER_KVB held [{d}, {d}] strides [{d}, {d}]; rows [{d}, {d}] strides [{d}, {d}], the held buffer: {s}; words differing from the checkpoint in bf16: {d}/{d} {s}\n", .{ hs[0], hs[1], hst[0], hst[1], rs[0], rs[1], rst[0], rst[1], yes(same_buffer), differ, nw, if (ok) "PASS" else "FAIL" });
        failed += @intFromBool(!ok);
    }
    // the misread the plugin's latent decode made before kvRows: the loader's view read row-major as rows
    const misread = try g.reshape(kvb.held, &.{ ci(h * (DN + DV)), ci(DL) });
    const scale: f64 = 1.0 / @sqrt(192.0);
    const cases = [_][3]usize{ .{ 18380, 1, 18432 }, .{ 18383, 4, 18432 }, .{ 18381, 2, 18432 } };
    for (cases) |c| {
        const t = c[0];
        const rows = c[1];
        const cap = c[2];
        const mark = g.mark();
        defer g.resetTo(mark);
        const lat_buf = try devRandn(&g, a, prng.random(), &.{ 1, 1, ci(cap), ci(DL) }, .bfloat16);
        const rot_buf = try devRandn(&g, a, prng.random(), &.{ 1, 1, ci(cap), ci(DR) }, .bfloat16);
        const lat = try g.slice(lat_buf, &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(t), ci(DL) }, &.{ 1, 1, 1, 1 });
        const rot = try g.slice(rot_buf, &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(t), ci(DR) }, &.{ 1, 1, 1, 1 });
        // the plugin's mlaAttn cut: the query heads [1, S, H, DN + DR] transposed to [1, H, S, DN + DR], each half a view
        const qt = try g.transposeAxes(try devRandn(&g, a, prng.random(), &.{ 1, ci(rows), ci(h), ci(DN + DR) }, .bfloat16), &.{ 0, 2, 1, 3 });
        const qn = try g.slice(qt, &.{ 0, 0, 0, 0 }, &.{ 1, ci(h), ci(rows), ci(DN) }, &.{ 1, 1, 1, 1 });
        const qr = try g.slice(qt, &.{ 0, 0, 0, ci(DN) }, &.{ 1, ci(h), ci(rows), ci(DN + DR) }, &.{ 1, 1, 1, 1 });
        const takes_rows = ml.takes(qn, qr, lat, rot, kvb.rows);
        const takes_held = ml.takes(qn, qr, lat, rot, kvb.held);
        const n = h * rows * DV;
        const want = try a.alloc(f32, n);
        defer a.free(want);
        try servedLoader(&g, s, a, qn, qr, lat, rot, kvb.held, h, rows, t, scale, want);
        const got = try a.alloc(f32, n);
        defer a.free(got);
        const ctl = try a.alloc(f32, n);
        defer a.free(ctl);
        var e: [2]f64 = .{ std.math.inf(f64), std.math.inf(f64) };
        var ec: [2]f64 = .{ std.math.inf(f64), std.math.inf(f64) };
        if (takes_rows) {
            try run(&g, s, .{ .qn = qn, .qr = qr, .lat = lat, .rot = rot, .w = kvb.rows }, scale, got);
            e = relErr(got, want);
            try run(&g, s, .{ .qn = qn, .qr = qr, .lat = lat, .rot = rot, .w = misread }, scale, ctl);
            ec = relErr(ctl, want);
        }
        const ok = takes_rows and !takes_held and e[0] <= 0.02 and ec[0] > 0.5;
        std.debug.print("TURBOEXL3_LING_MLA_LATENT_LOADER keys={d} of a {d}-row buffer rows={d}: takes(rows)={s} takes(held)={s}; decodeLatent against the plugin's expanded attention rel_l2={d:.5} rel_max={d:.5} (tolerance 0.02); control (the loader's view read as rows) rel_l2={d:.3} {s}\n", .{ t, cap, rows, yes(takes_rows), yes(takes_held), e[0], e[1], ec[0], if (ok) "PASS" else "FAIL" });
        failed += @intFromBool(!ok);
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── launch 2's variants: each launch 2's words ──

/// The variants of launch 2 the byte test holds to it (`mla_latent.Attn` but `base`)
const variants = [_]ml.Attn{ .rows2, .rows4, .stage, .block };

/// mlx-c's `mlx_set_error_handler`: the consumer's binding where its `sdk.mlx` declares one, else a declaration of the
/// same symbol
const setErrorHandler = if (@hasDecl(mlx, "mlx_set_error_handler")) mlx.mlx_set_error_handler else struct {
    extern "c" fn mlx_set_error_handler(handler: ?*const fn ([*:0]const u8, ?*anyopaque) callconv(.c) void, data: ?*anyopaque, dtor: ?*const fn (?*anyopaque) callconv(.c) void) void;
}.mlx_set_error_handler;

fn reportMlxError(msg: [*:0]const u8, _: ?*anyopaque) callconv(.c) void {
    std.debug.print("MLX error: {s}\n", .{msg});
}

/// `on`: an MLX error is printed and its call returns an error code (mlx.check's error.MlxError), so a variant whose
/// launch MLX refuses (its threadgroup past the pipeline's limit) fails its own lines and the run goes on; mlx-c's
/// default handler ends the process. Off: that default again.
pub fn reportMlxErrors(on: bool) void {
    const Handler = @typeInfo(@TypeOf(setErrorHandler)).@"fn".params[0].type.?;
    setErrorHandler(if (on) @as(Handler, @ptrCast(&reportMlxError)) else null, null, null);
}

/// Words of `got` whose bits differ from `want`'s
fn diffWords(got: []const f32, want: []const f32) usize {
    var n: usize = 0;
    for (got, want) |u, w| n += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return n;
}

/// A call's words on the host: its output (as f32) and launch 2's partials
const Words = struct {
    out: []f32,
    pm: []f32,
    pl: []f32,
    po: []f32,

    fn init(a: std.mem.Allocator, h: usize, rows: usize, nj: usize) !Words {
        const slots = rows * h * nj;
        const out = try a.alloc(f32, h * rows * DV);
        errdefer a.free(out);
        const pm = try a.alloc(f32, slots);
        errdefer a.free(pm);
        const pl = try a.alloc(f32, slots);
        errdefer a.free(pl);
        return .{ .out = out, .pm = pm, .pl = pl, .po = try a.alloc(f32, slots * DL) };
    }

    fn deinit(w: *Words, a: std.mem.Allocator) void {
        inline for (.{ w.out, w.pm, w.pl, w.po }) |x| a.free(x);
    }

    /// The words of the output, pm, pl and po that differ from `want`'s
    fn differing(w: Words, want: Words) [4]usize {
        return .{ diffWords(w.out, want.out), diffWords(w.pm, want.pm), diffWords(w.pl, want.pl), diffWords(w.po, want.po) };
    }
};

/// `decodeLatentParts` with launch 2 on `attn`, its words copied into `dst`
fn partsWords(g: *G, s: mlx.mlx_stream, qn: mlx.mlx_array, qr: mlx.mlx_array, lat: mlx.mlx_array, rot: mlx.mlx_array, w: mlx.mlx_array, scale: f64, attn: ml.Attn, dst: Words) !void {
    const r = try ml.decodeLatentParts(s, qn, qr, lat, rot, w, @floatCast(scale), attn);
    const out = try g.adopt(r.out);
    const pm = try g.adopt(r.pm);
    const pl = try g.adopt(r.pl);
    const po = try g.adopt(r.po);
    _ = try g.hostF32(try g.astype(out, .float32), dst.out);
    _ = try g.hostF32(pm, dst.pm);
    _ = try g.hostF32(pl, dst.pl);
    _ = try g.hostF32(po, dst.po);
}

/// `x` [1, 1, T, D] as a view of a contiguous [1, 1, D, T] copy: the same words, a key's a column apart (strides
/// [.., .., 1, T])
fn transposedView(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array) !mlx.mlx_array {
    var c = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&c, try g.transposeAxes(x, &.{ 0, 1, 3, 2 }), false, s));
    const cm = try g.adopt(c);
    try mlx.check(mlx.mlx_array_eval(cm));
    return g.transposeAxes(cm, &.{ 0, 1, 3, 2 });
}

/// The head's cache as the plugin keeps it: bf16 words `lat_host` [cap, DL] and `rot_host` [cap, DR] as two buffers, the
/// views their first `t` rows
fn headCache(g: *G, lat_host: []const u16, rot_host: []const u16, cap: usize, t: usize) ![2]mlx.mlx_array {
    const lat_buf = try g.hostArray(std.mem.sliceAsBytes(lat_host), &.{ 1, 1, ci(cap), ci(DL) }, .bfloat16);
    const rot_buf = try g.hostArray(std.mem.sliceAsBytes(rot_host), &.{ 1, 1, ci(cap), ci(DR) }, .bfloat16);
    return .{
        try g.slice(lat_buf, &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(t), ci(DL) }, &.{ 1, 1, 1, 1 }),
        try g.slice(rot_buf, &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(t), ci(DR) }, &.{ 1, 1, 1, 1 }),
    };
}

const CacheView = enum { plugin, interleaved, transposed };

test "turboexl3 gpu ling mla latent variants: rows2, rows4, stage and block give launch 2's words, the output and launch 2's partials word for word, at rows 1 to 4 and keys 1 to 19,404, on the plugin's operands (kv_b through the loader's rows view, the head's cache as a wider buffer's first rows, the query halves as views) and two strided caches; a planted fault differs (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    reportMlxErrors(true);
    defer reportMlxErrors(false);
    const h: usize = 64;
    const scale: f64 = 1.0 / @sqrt(192.0);
    var prng = std.Random.DefaultPrng.init(0x7A11_A7E5);
    const rnd = prng.random();
    const kvb = try loaderKvB(&g, a, rnd, h);
    // 1 to 257 keys: the sub-chunk (32) and key-group (256) edges; 4096 and 16,669: groups ending mid sub-chunk; 19,404:
    // the MTP head's cache at the standard request
    const key_counts = [_]usize{ 1, 2, 31, 32, 255, 256, 257, 4096, 16669, 19404 };
    var failed: usize = 0;
    var lines: usize = 0;
    for (key_counts) |t| {
        const km = g.mark();
        defer g.resetTo(km);
        // the buffer's rows past T hold other words, as after a rejected draft's rollback
        const cap = t + 61;
        const lat_host = try a.alloc(u16, cap * DL);
        defer a.free(lat_host);
        const rot_host = try a.alloc(u16, cap * DR);
        defer a.free(rot_host);
        for (lat_host) |*v| v.* = support.bf16Bits(rnd.floatNorm(f32));
        for (rot_host) |*v| v.* = support.bf16Bits(rnd.floatNorm(f32));
        const cache = try headCache(&g, lat_host, rot_host, cap, t);
        const both = try g.concat(&.{ cache[0], cache[1] }, 3);
        try mlx.check(mlx.mlx_array_eval(both));
        const views = [3][2]mlx.mlx_array{
            cache,
            .{
                try g.slice(both, &.{ 0, 0, 0, 0 }, &.{ 1, 1, ci(t), ci(DL) }, &.{ 1, 1, 1, 1 }),
                try g.slice(both, &.{ 0, 0, 0, ci(DL) }, &.{ 1, 1, ci(t), ci(DL + DR) }, &.{ 1, 1, 1, 1 }),
            },
            .{ try transposedView(&g, s, cache[0]), try transposedView(&g, s, cache[1]) },
        };
        for (1..5) |rows| {
            if (rows > t) continue;
            const rm = g.mark();
            defer g.resetTo(rm);
            // the plugin's mlaAttn cut: the query heads [1, S, H, DN + DR] transposed to [1, H, S, DN + DR], each half a view
            const qt = try g.transposeAxes(try devRandn(&g, a, rnd, &.{ 1, ci(rows), ci(h), ci(DN + DR) }, .bfloat16), &.{ 0, 2, 1, 3 });
            const qn = try g.slice(qt, &.{ 0, 0, 0, 0 }, &.{ 1, ci(h), ci(rows), ci(DN) }, &.{ 1, 1, 1, 1 });
            const qr = try g.slice(qt, &.{ 0, 0, 0, ci(DN) }, &.{ 1, ci(h), ci(rows), ci(DN + DR) }, &.{ 1, 1, 1, 1 });
            const nj = ml.groupsOf(@intCast(t));
            var want = try Words.init(a, h, rows, nj);
            defer want.deinit(a);
            var got = try Words.init(a, h, rows, nj);
            defer got.deinit(a);
            try partsWords(&g, s, qn, qr, cache[0], cache[1], kvb.rows, scale, .base, want);
            for (variants) |v| for (views, 0..) |view, vi| {
                const vm = g.mark();
                defer g.resetTo(vm);
                const kind: CacheView = @enumFromInt(vi);
                lines += 1;
                partsWords(&g, s, qn, qr, view[0], view[1], kvb.rows, scale, v, got) catch |e| {
                    std.debug.print("TURBOEXL3_LING_MLA_LATENT_VARIANT keys={d} rows={d} variant={s} cache={s}: {s} FAIL\n", .{ t, rows, @tagName(v), @tagName(kind), @errorName(e) });
                    failed += 1;
                    continue;
                };
                const d = got.differing(want);
                const ok = d[0] == 0 and d[1] == 0 and d[2] == 0 and d[3] == 0;
                std.debug.print("TURBOEXL3_LING_MLA_LATENT_VARIANT keys={d} rows={d} variant={s} cache={s} words differing from launch 2's: out={d}/{d} pm={d}/{d} pl={d}/{d} po={d}/{d} {s}\n", .{ t, rows, @tagName(v), @tagName(kind), d[0], want.out.len, d[1], want.pm.len, d[2], want.pl.len, d[3], want.po.len, if (ok) "PASS" else "FAIL" });
                failed += @intFromBool(!ok);
            };
        }
    }
    // Red checks: a planted fault in the cache (one key row zeroed; one bit of one word flipped), at a key every row
    // sees, run on each variant, must differ from launch 2's words on the clean cache. A check that passes a faulted
    // cache cannot catch a variant that reads a wrong key or word.
    var red_failed: usize = 0;
    for ([_][2]usize{ .{ 1, 1 }, .{ 257, 3 }, .{ 19404, 4 } }) |c| {
        const t = c[0];
        const rows = c[1];
        const rm = g.mark();
        defer g.resetTo(rm);
        const cap = t + 61;
        const lat_host = try a.alloc(u16, cap * DL);
        defer a.free(lat_host);
        const rot_host = try a.alloc(u16, cap * DR);
        defer a.free(rot_host);
        for (lat_host) |*v| v.* = support.bf16Bits(rnd.floatNorm(f32));
        for (rot_host) |*v| v.* = support.bf16Bits(rnd.floatNorm(f32));
        const qt = try g.transposeAxes(try devRandn(&g, a, rnd, &.{ 1, ci(rows), ci(h), ci(DN + DR) }, .bfloat16), &.{ 0, 2, 1, 3 });
        const qn = try g.slice(qt, &.{ 0, 0, 0, 0 }, &.{ 1, ci(h), ci(rows), ci(DN) }, &.{ 1, 1, 1, 1 });
        const qr = try g.slice(qt, &.{ 0, 0, 0, ci(DN) }, &.{ 1, ci(h), ci(rows), ci(DN + DR) }, &.{ 1, 1, 1, 1 });
        const nj = ml.groupsOf(@intCast(t));
        var want = try Words.init(a, h, rows, nj);
        defer want.deinit(a);
        var got = try Words.init(a, h, rows, nj);
        defer got.deinit(a);
        const clean = try headCache(&g, lat_host, rot_host, cap, t);
        try partsWords(&g, s, qn, qr, clean[0], clean[1], kvb.rows, scale, .base, want);
        // the faulted key: one every row of the call sees
        const k = (t - rows) / 2;
        for ([_][]const u8{ "zeroed_row", "flipped_bit" }) |fault| {
            const faulty = try a.dupe(u16, lat_host);
            defer a.free(faulty);
            if (fault[0] == 'z') {
                @memset(faulty[k * DL ..][0..DL], 0);
            } else {
                faulty[k * DL + 7] ^= 1;
            }
            const bad = try headCache(&g, faulty, rot_host, cap, t);
            for (variants) |v| {
                const vm = g.mark();
                defer g.resetTo(vm);
                partsWords(&g, s, qn, qr, bad[0], bad[1], kvb.rows, scale, v, got) catch |e| {
                    std.debug.print("TURBOEXL3_LING_MLA_LATENT_VARIANT_RED keys={d} rows={d} variant={s} fault={s}: {s} FAIL\n", .{ t, rows, @tagName(v), fault, @errorName(e) });
                    red_failed += 1;
                    continue;
                };
                const d = got.differing(want);
                const n = d[0] + d[1] + d[2] + d[3];
                std.debug.print("TURBOEXL3_LING_MLA_LATENT_VARIANT_RED keys={d} rows={d} variant={s} fault={s} at key {d}: words differing from launch 2's on the clean cache out={d} pm={d} pl={d} po={d} (a fault must differ) {s}\n", .{ t, rows, @tagName(v), fault, k, d[0], d[1], d[2], d[3], if (n > 0) "PASS" else "FAIL" });
                red_failed += @intFromBool(n == 0);
            }
        }
    }
    std.debug.print("TURBOEXL3_LING_MLA_LATENT_VARIANT_DONE {d} lines, {d} failed; red checks failed {d}\n", .{ lines, failed, red_failed });
    try testing.expectEqual(@as(usize, 0), failed);
    try testing.expectEqual(@as(usize, 0), red_failed);
}
