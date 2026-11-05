//! F3 (`root.sharedSwiglu`, `dense.text_swiglu_prep`): DeepSeek-V4.1's shared expert at decode rows with the gate's and
//! up's finishes, the host's clamped SwiGLU and the down's prepare in one launch. The CPU test holds the fused text to the
//! statements it reuses; TURBOEXL3_GPU=1 TURBOEXL3_F3=1 compares it with the host's chain (gate and up `linear`, astype f32,
//! MLX's minimum, clip, sigmoid and multiplies, astype to the activation dtype, down `linear`) word for word at rows 1 to
//! 16 (lines TURBOEXL3_F3_BYTES, then TURBOEXL3_F3_BYTES_DONE); TURBOEXL3_GPU=1 TURBOEXL3_F3_TIME=1 times the 40-layer
//! trunk chain at rows 5, 6 and 8 with the host's SwiGLU as one elementwise launch (what MLX's compiled decode stretch
//! makes of it) against F3, A B B A in 3 rounds, every run's output held to the first's words (lines TURBOEXL3_F3_CHAIN,
//! then TURBOEXL3_F3_CHAIN_SUM).
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ks = sdk.kernel_launch;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const root = @import("../../root.zig");
const dense = @import("../../core/dense.zig");
const decode = @import("../../core/decode.zig");
const ds = @import("deepseek_test.zig");

const testing = std.testing;
const ci = ds.ci;
const A = mlx.mlx_array;

const limit: f32 = 10.0;

test "turboexl3 f3u: the fused shared-expert text is text_dfinish's finishes, MLX's clamped SwiGLU and text_prep_s's prepare" {
    const t = dense.text_swiglu_prep;
    for ([_][]const u8{
        // text_dfinish's statements on both planes
        "float acc = z1[ulong(row) * ulong(DIMC) + o];",
        "for (uint k = 1u; k < KSG; ++k) acc += z1[(ulong(k) * ulong(ROWSC) + row) * ulong(DIMC) + o];",
        "if (DEFER != 0) acc = fma(acc, MUL1_A, MUL1_C * xsum1);",
        "if (DEFER != 0) for (uint q = 0; q < INB; ++q) xsum1 += xs1[row * INB + q];",
        "const float gate = float(OUT_T(float(half(v1[j] * HAD_SCALE * float(svh1[o])))));",
        "const float up = float(OUT_T(float(half(v3[j] * HAD_SCALE * float(svh3[o])))));",
        // MLX's Minimum / Maximum (NaN first), Sigmoid (precise exp of |x|) and Multiply for float
        "float g = metal::isnan(gate) ? gate : (gate < hi ? gate : hi);",
        "float um = metal::isnan(up) ? up : (up > lo ? up : lo);",
        "float u = metal::isnan(um) ? um : (um < hi ? um : hi);",
        "float sy = 1 / (1 + metal::precise::exp(metal::abs(g)));",
        "float sig = (g < 0) ? sy : 1 - sy;",
        "float act = g * sig;",
        "float outv = act * u;",
        // text_prep_s on the activation in its dtype
        "v[j] = float(half(float(ACT_T(outv)))) * float(suh2[o]);",
        "xp[row * DIMC + ((i & ~15u) + 4u * q + m)] = hv;",
        "if (L == 0u) xs[row * NB + b] = sum;",
    }) |line| try testing.expect(std.mem.indexOf(u8, t, line) != null);
    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, t, "for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor("));
    for ([_][]const u8{
        "float acc = z[ulong(row) * ulong(DIMC) + o];",
        "for (uint k = 1u; k < KSC; ++k) acc += z[(ulong(k) * ulong(ROWSC) + row) * ulong(DIMC) + o];",
        "if (DEFER != 0) acc = fma(acc, MUL1_A, MUL1_C * xsum);",
        "v[j] = float(half(acc));",
    }) |line| try testing.expect(std.mem.indexOf(u8, dense.text_dfinish, line) != null);
    try testing.expect(std.mem.indexOf(u8, dense.text_dfinish, "y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))));") != null);
}

fn bitsDiffer(p: []const f32, q: []const f32) usize {
    var n: usize = 0;
    for (p, q) |u, w| n += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return n;
}

fn op2(g: *G, comptime f: anytype, a: A, b: A) !A {
    var r = mlx.mlx_array_new();
    mlx.check(f(&r, a, b, g.s)) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
}

/// The host's shared expert (src/mlx_serve/plugins/deepseek_v41 `shared` at decode rows on EXL3 linears, `clampedSwiglu`
/// and sdk/blocks/mlp/activations.zig `clampedSwiGLU`), op for op
fn hostShared(g: *G, s: mlx.mlx_stream, x: A, p1: root.Proj, p3: root.Proj, p2: root.Proj, act: mlx.mlx_dtype) !A {
    const gate = try g.astype(try g.adopt(try root.linear(s, x, p1.trellis, p1.suh, p1.svh, .mul1)), .float32);
    const up = try g.astype(try g.adopt(try root.linear(s, x, p3.trellis, p3.suh, p3.svh, .mul1)), .float32);
    const hi = try g.scalar(limit, .float32);
    const lo = try g.scalar(-limit, .float32);
    const gm = try op2(g, mlx.mlx_minimum, gate, hi);
    const um = try op2(g, mlx.mlx_minimum, try op2(g, mlx.mlx_maximum, up, lo), hi);
    const sh = try g.mul(try g.mul(gm, try g.sigmoid(gm)), um);
    return g.adopt(try root.linear(s, try g.astype(sh, act), p2.trellis, p2.suh, p2.svh, .mul1));
}

test "turboexl3 gpu f3u bytes: sharedSwiglu gives the host's shared expert's words at rows 1 to 16 (TURBOEXL3_F3=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_F3") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xF3);
    const rnd = prng.random();
    const p1 = try ds.randProj(a, &g, rnd, null, 5120, 2304, ds.N);
    const p3 = try ds.randProj(a, &g, rnd, null, 5120, 2304, ds.N);
    const p2 = try ds.randProj(a, &g, rnd, null, 2304, 5120, ds.N);
    var lines: usize = 0;
    var taken: usize = 0;
    var bad: usize = 0;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| for (1..17) |r| {
        const rows: u32 = @intCast(r);
        if (dt == .float32 and rows != 6 and rows != 12) continue;
        const rm = g.mark();
        defer g.resetTo(rm);
        // rows drawn at three scales (1, 8, 32) so the clamp at 10 and both sigmoid branches are taken
        const xh = try a.alloc(f32, @as(usize, rows) * 5120);
        defer a.free(xh);
        for (xh, 0..) |*v, i| v.* = ds.bf16Exact(rnd.floatNorm(f32) * ([_]f32{ 1, 8, 32 })[(i / 5120) % 3]);
        const x0 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), 5120 }, .float32), .bfloat16);
        const x = if (dt == .bfloat16) x0 else try g.astype(x0, .float32);
        const takes = try root.sharedSwigluTakes(x, p1, p3, p2, dt, .mul1);
        const n = rows * 5120;
        const want = try a.alloc(f32, n);
        defer a.free(want);
        const got = try a.alloc(f32, n);
        defer a.free(got);
        _ = try g.hostF32(try g.astype(try hostShared(&g, s, x, p1, p3, p2, dt), .float32), want);
        var diff: usize = 0;
        if (takes) {
            _ = try g.hostF32(try g.astype(try g.adopt(try root.sharedSwiglu(s, x, p1, p3, p2, limit, dt, .mul1)), .float32), got);
            diff = bitsDiffer(got, want);
            taken += 1;
        }
        lines += 1;
        if (diff != 0 or (takes != (rows >= 5))) bad += 1;
        std.debug.print("TURBOEXL3_F3_BYTES rows={d} x={s} takes={} differing={d}/{d} {s}\n", .{ rows, @tagName(dt), takes, diff, n, if (diff == 0 and takes == (rows >= 5)) "OK" else "DIFFER" });
    };
    std.debug.print("TURBOEXL3_F3_BYTES_DONE lines={d} taken={d} differing_lines={d} {s}\n", .{ lines, taken, bad, if (bad == 0) "BYTES_OK" else "BYTES_DIFFER" });
    try testing.expectEqual(@as(usize, 0), bad);
}

/// The host's clamped SwiGLU as MLX's compiled decode stretch runs it: one elementwise launch over the gate and up (bf16),
/// the functors in clampedSwiGLU's order in f32, the activation dtype out
const text_swiglu_one = @embedFile("kernels/test/swiglu_one.metal");

const ChainCtx = struct {
    k: mlx.mlx_fast_metal_kernel,
    lim: A,
};

fn swigluOne(g: *G, s: mlx.mlx_stream, cx: *const ChainCtx, gate: A, up: A, rows: u32) !A {
    const nel: u32 = rows * 2304;
    const cfg = try ks.buildConfig(.{
        .grid = .{ ((nel + 255) / 256) * 256, 1, 1 },
        .threadgroup = .{ 256, 1, 1 },
        .out_shapes = &.{&[_]c_int{ ci(rows), 2304 }},
        .out_dtypes = &.{.bfloat16},
        .template = &.{.{ .name = "NEL", .value = .{ .int = @intCast(nel) } }},
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [1]A = undefined;
    try ks.applyConfig(cx.k, cfg, &.{ gate, up, cx.lim }, &o, s);
    return g.adopt(o[0]);
}

/// `ds.chainForward`'s trunk arm with the shared expert as the host serves it at decode rows: the gate and up linears,
/// the clamped SwiGLU in one elementwise launch (`f3` false) or all of it bar the down's GEMV and finish in F3's
fn chainShared(g: *G, s: mlx.mlx_stream, cx: *const ChainCtx, layers: []const ds.ChainLayer, x: A, rows: u32, f3: bool) !A {
    const R = ci(rows);
    var h = x;
    for (layers) |ly| {
        const qa = try g.adopt(try root.linear(s, h, ly.wq_a.trellis, ly.wq_a.suh, ly.wq_a.svh, .mul1));
        const q = try g.adopt(try root.linear(s, qa, ly.wq_b.trellis, ly.wq_b.suh, ly.wq_b.svh, .mul1));
        var extra = try g.mean(try g.adopt(try root.linear(s, h, ly.wkv.trellis, ly.wkv.suh, ly.wkv.svh, .mul1)), 1, true);
        if (ly.cwkv) |p| extra = try g.add(extra, try g.mean(try g.adopt(try root.linear(s, h, p.trellis, p.suh, p.svh, .mul1)), 1, true));
        if (ly.cwgate) |p| extra = try g.add(extra, try g.mean(try g.adopt(try root.linear(s, h, p.trellis, p.suh, p.svh, .mul1)), 1, true));
        if (ly.iwq_b) |p| extra = try g.add(extra, try g.mean(try g.adopt(try root.linear(s, qa, p.trellis, p.suh, p.svh, .mul1)), 1, true));
        const oa = try g.adopt(try root.linearGrouped(s, try g.reshape(q, &.{ R, 8, 4096 }), ly.wo_a, .mul1));
        const ob = try g.adopt(try root.linear(s, try g.reshape(oa, &.{ R, 8192 }), ly.wo_b.trellis, ly.wo_b.suh, ly.wo_b.svh, .mul1));
        h = try g.add(try g.add(h, ob), extra);
        const sy = if (f3)
            try g.adopt(try root.sharedSwiglu(s, h, ly.sw1, ly.sw3, ly.sw2, limit, .bfloat16, .mul1))
        else blk: {
            const g1 = try g.adopt(try root.linear(s, h, ly.sw1.trellis, ly.sw1.suh, ly.sw1.svh, .mul1));
            const up = try g.adopt(try root.linear(s, h, ly.sw3.trellis, ly.sw3.suh, ly.sw3.svh, .mul1));
            break :blk try g.adopt(try root.linear(s, try swigluOne(g, s, cx, g1, up, rows), ly.sw2.trellis, ly.sw2.suh, ly.sw2.svh, .mul1));
        };
        h = try g.add(h, sy);
    }
    return h;
}

fn sortedMedian(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return if (xs.len % 2 == 1) xs[xs.len / 2] else 0.5 * (xs[xs.len / 2 - 1] + xs[xs.len / 2]);
}

fn minOf(xs: []const f64) f64 {
    var m: f64 = std.math.inf(f64);
    for (xs) |v| m = @min(m, v);
    return m;
}

fn chainMin(g: *G, s: mlx.mlx_stream, cx: *const ChainCtx, layers: []const ds.ChainLayer, x: A, rows: u32, f3: bool, out: []f32) !f64 {
    var total: [5]f64 = undefined;
    for (0..total.len + 2) |round| {
        const fm = g.mark();
        defer g.resetTo(fm);
        const t0 = ds.nowNs();
        const y = try chainShared(g, s, cx, layers, x, rows, f3);
        try g.evalAll(&.{y});
        const dt = ds.nowNs() - t0;
        if (round >= 2) total[round - 2] = @as(f64, @floatFromInt(dt)) / 1e6;
        if (round == 0) _ = try g.hostF32(try g.astype(y, .float32), out);
    }
    return minOf(&total);
}

test "turboexl3 gpu f3u time: the 40-layer trunk chain with the host's shared expert against F3 at rows 5, 6 and 8, every run the first's words (TURBOEXL3_F3_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_F3_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xF32);
    const rnd = prng.random();
    const k = ks.newKernel("turboexl3_test_swiglu_one", &[_][*:0]const u8{ "gt", "ut", "lim" }, &[_][*:0]const u8{"y"}, text_swiglu_one, "", true) orelse return error.KernelCreateFailed;
    defer decode.freeKernel(k);
    const lim = try g.hostArray(std.mem.asBytes(&limit), &.{1}, .float32);
    const cx: ChainCtx = .{ .k = k, .lim = lim };
    const nl = 40;
    const layers = try a.alloc(ds.ChainLayer, nl);
    defer a.free(layers);
    var chp = [_]ds.HostProj{
        try ds.HostProj.draw(a, rnd, null, 5120, 1280, ds.N), // 0 wq_a
        try ds.HostProj.draw(a, rnd, null, 1280, 32768, ds.N), // 1 wq_b
        try ds.HostProj.draw(a, rnd, null, 5120, 512, ds.N), // 2 wkv, the compressor's wkv and wgate
        try ds.HostProj.draw(a, rnd, null, 1280, 4096, ds.N), // 3 the indexer's wq_b
        try ds.HostProj.draw(a, rnd, ds.wo_a.groups, ds.wo_a.in, ds.wo_a.out, ds.N), // 4 wo_a
        try ds.HostProj.draw(a, rnd, null, 8192, 5120, ds.N), // 5 wo_b
        try ds.HostProj.draw(a, rnd, null, 5120, 2304, ds.N), // 6 the shared w1 and w3
        try ds.HostProj.draw(a, rnd, null, 2304, 5120, ds.N), // 7 the shared w2
    };
    defer for (&chp) |*h| h.deinit(a);
    for (layers, 0..) |*ly, li| {
        const L: u32 = @intCast(li);
        const comp = std.mem.indexOfScalar(u32, &ds.comp_layers, L) != null;
        const idx = std.mem.indexOfScalar(u32, &ds.index_layers, L) != null;
        const wq_a = try chp[0].upload(&g);
        ly.* = .{
            .wq_a = wq_a,
            .wq_b = try chp[1].upload(&g),
            .wkv = try chp[2].upload(&g),
            .cwkv = if (comp) try chp[2].upload(&g) else null,
            .cwgate = if (comp) try chp[2].upload(&g) else null,
            .iwq_b = if (idx) try chp[3].upload(&g) else null,
            .wo_a = try chp[4].upload(&g),
            .wo_b = try chp[5].upload(&g),
            .sw1 = try chp[6].upload(&g),
            .sw3 = try chp[6].upload(&g),
            .sw2 = try chp[7].upload(&g),
            .bank = .{ .gate = wq_a, .up = wq_a, .down = wq_a },
        };
    }
    var failed = false;
    for ([_]u32{ 5, 6, 8 }) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        const x = try ds.randBf16(a, &g, rnd, &.{ ci(rows), 5120 });
        const n_out = rows * 5120;
        const want = try a.alloc(f32, n_out);
        defer a.free(want);
        const got = try a.alloc(f32, n_out);
        defer a.free(got);
        var first = true;
        var t: [2][6]f64 = undefined;
        var nt: [2]usize = .{ 0, 0 };
        var diff: usize = 0;
        for (0..3) |round| for ([_]u1{ 0, 1, 1, 0 }, 0..) |ai, pos| {
            const ms = try chainMin(&g, s, &cx, layers, x, rows, ai == 1, if (first) want else got);
            const dd = if (first) 0 else bitsDiffer(got, want);
            first = false;
            diff += dd;
            t[ai][nt[ai]] = ms;
            nt[ai] += 1;
            std.debug.print("TURBOEXL3_F3_CHAIN rows={d} round={d} pos={d} arm={s} chain_ms={d:.3} differing={d}\n", .{ rows, round, pos, if (ai == 0) "A:host" else "B:f3", ms, dd });
        };
        const amed = sortedMedian(&t[0]);
        const bmed = sortedMedian(&t[1]);
        std.debug.print("TURBOEXL3_F3_CHAIN_SUM rows={d} layers={d} a_min_ms={d:.3} a_med_ms={d:.3} b_min_ms={d:.3} b_med_ms={d:.3} b_minus_a_med={d:.3} b_minus_a_min={d:.3} differing={d} {s}\n", .{ rows, nl, minOf(&t[0]), amed, minOf(&t[1]), bmed, bmed - amed, minOf(&t[1]) - minOf(&t[0]), diff, if (diff == 0) "BYTES_OK" else "BYTES_DIFFER" });
        if (diff != 0) failed = true;
    }
    try testing.expect(!failed);
}
