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
//! Lines: TURBOEXL3_LING_MLA_LATENT, TURBOEXL3_LING_MLA_LATENT_ROWS, TURBOEXL3_LING_MLA_LATENT_SERVED.
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
