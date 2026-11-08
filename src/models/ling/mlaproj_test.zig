//! `mlaProjG` (the outside plugin turboexl3_ling's MLX_SERVE_LING_MLA_PROJ_FUSED): an MLA layer's q_a and kv_a (two
//! EXL3 projections of the layer's input) and its head gate g_proj (bf16, 4096 -> 64) in one GEMV launch after one
//! prepare launch, held word for word to the launches it replaces.
//! - GPU (TURBOEXL3_GPU=1, else skipped), at Ling-3.1-flash's MLA shapes (q_a 4096 -> 1536 and kv_a 4096 -> 640 at K 6,
//!   the loader's I16 trellis words; g_proj 4096 -> 64): g's weight as the plugin's loader keeps it (the pack's f16
//!   [64, 4096] viewed [4096, 64] and cast to bf16, its column-major layout kept) and its transpose back for the
//!   member; x the layer's normed input [1, S, 4096] bf16. At rows 1 to 8 (serial decode's one row and an exact
//!   verify's rows), q_a's and kv_a's outputs against each one's own `linear`, and g's against the served g_proj (`m.mm`
//!   at one row: MLX's matmul; at 2 to 8 the exact verify's denseRows: each row stored twice, the first viewed, a
//!   batched matmul), each word for word; and the refusals. Lines: TURBOEXL3_LING_MLA_PROJ.
//! - Timing (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; asserts nothing, guidance only): ten MLA layers' three
//!   projections chained at rows 1 to 4 (each layer's input reads the layer before through its gate's first column
//!   times zero), the served launches against `mlaProjG`; then the last layer's words in both arms. Lines:
//!   TURBOEXL3_LING_MLA_PROJ_TIME.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const T = mlx.mlx_array;
const format = @import("../../core/format.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const mlaproj = @import("mlaproj.zig");
const ds_test = @import("../deepseek_v41/deepseek_test.zig");

const testing = std.testing;

const in: u32 = 4096;
const outs = [2]u32{ 1536, 640 };
const out_g: u32 = 64;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// A synthetic projection and the operands `linear` takes for it
const Served = struct { lin: ref.Linear, p: root.Proj };

/// A projection `linear` takes at (in, out, n 96): its synthetic trellis as the loader hands it (the pack's I16 words)
fn servedProj(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, rnd: std.Random, o: u32) !Served {
    var lin = try ref.synthLinear(a, rnd, 1, in, o, format.Rate.fromN(96).?, .mul1);
    errdefer lin.deinit(a);
    const u = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(in / 16), ci(o / 16), 96 }, .uint16);
    var v = mlx.mlx_array_new();
    mlx.check(mlx.mlx_view(&v, u, .int16, s)) catch |e| {
        _ = mlx.mlx_array_free(v);
        return e;
    };
    return .{ .lin = lin, .p = .{
        .trellis = try g.adopt(v),
        .suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(o)}, .float16),
    } };
}

/// MLX's expand_dims (denseRows): the consumer's binding where its `sdk.mlx` declares it, else libmlxc's own symbol
const expandDims = if (@hasDecl(mlx, "mlx_expand_dims")) mlx.mlx_expand_dims else struct {
    extern "c" fn mlx_expand_dims(res: *mlx.mlx_array, a: mlx.mlx_array, axis: c_int, s: mlx.mlx_stream) c_int;
}.mlx_expand_dims;

/// The served g_proj of x [rows, in] bf16 over the loaded weight w [in, out] (column-major): at one row `m.mm`'s
/// matmul; at 2 to 8 rows the exact verify's denseRows (x as [S, 1, in], each row stored twice and the first viewed, w
/// as [1, in, out], one batched matmul), [rows, out]
fn servedG(g: *G, s: mlx.mlx_stream, x: T, w: T, rows: u32) !T {
    if (rows == 1) return g.matmul(x, w);
    const x3 = try g.reshape(x, &.{ ci(rows), 1, ci(in) });
    const twice = try g.concat(&.{ x3, x3 }, 1);
    const view = try g.slice(twice, &.{ 0, 0, 0 }, &.{ ci(rows), 1, ci(in) }, &.{ 1, 1, 1 });
    var w3 = mlx.mlx_array_new();
    mlx.check(expandDims(&w3, w, 0, s)) catch |e| {
        _ = mlx.mlx_array_free(w3);
        return e;
    };
    return g.reshape(try g.matmul(view, try g.adopt(w3)), &.{ ci(rows), ci(out_g) });
}

/// g's weight as the loader keeps it (the pack's f16 [out, in], viewed [in, out], cast to bf16, its layout kept) and the
/// member's (that view transposed back, [out, in] row-major)
const GateWeights = struct { served: T, member: T };

fn gateWeights(a: std.mem.Allocator, g: *G, rnd: std.Random) !GateWeights {
    const wraw = try a.alloc(f16, out_g * in);
    defer a.free(wraw);
    for (wraw) |*v| v.* = @floatCast(rnd.floatNorm(f32) / 32.0);
    const w16 = try g.hostArray(std.mem.sliceAsBytes(wraw), &.{ ci(out_g), ci(in) }, .float16);
    const served = try g.astype(try g.transposeAxes(w16, &.{ 1, 0 }), .bfloat16);
    try g.evalAll(&.{served});
    return .{ .served = served, .member = try g.transposeAxes(served, &.{ 1, 0 }) };
}

/// Words of `got` and `want` (one size; compared as f32) that differ
fn differ(a: std.mem.Allocator, g: *G, got: T, want: T) !usize {
    const n = mlx.mlx_array_size(want);
    if (mlx.mlx_array_size(got) != n) return error.SizeMismatch;
    const p = try a.alloc(f32, n);
    defer a.free(p);
    const q = try a.alloc(f32, n);
    defer a.free(q);
    _ = try g.hostF32(try g.astype(got, .float32), p);
    _ = try g.hostF32(try g.astype(want, .float32), q);
    var d: usize = 0;
    for (p, q) |u, w| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return d;
}

test "turboexl3 gpu ling mlaproj: mlaProjG's q_a and kv_a are each one's own linear and its gate the served g_proj (one-row matmul, the exact verify's denseRows), word for word at rows 1 to 8, operands as the loader keeps them (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x3A1A_0E5);
    const rnd = prng.random();
    var ms: [2]Served = undefined;
    var made: usize = 0;
    defer for (ms[0..made]) |*m| m.lin.deinit(a);
    for (&ms, outs) |*m, o| {
        m.* = try servedProj(a, &g, s, rnd, o);
        made += 1;
    }
    const P = [2]root.Proj{ ms[0].p, ms[1].p };
    const gw = try gateWeights(a, &g, rnd);
    const sh = try mlaproj.shapeOf(P);
    try testing.expectEqual(@as(?u32, out_g), mlaproj.gTakes(sh.in, .bfloat16, gw.member));
    const xs = try a.alloc(f32, 8 * in);
    defer a.free(xs);
    for (xs) |*v| v.* = rnd.floatNorm(f32);
    var failed: usize = 0;
    for (1..9) |ru| {
        const rows: u32 = @intCast(ru);
        const mark = g.mark();
        defer g.resetTo(mark);
        // the layer's normed input as the MLA block takes it, [1, S, 4096] bf16; `linear` and the gate read its rows
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. rows * in]), &.{ 1, ci(rows), ci(in) }, .float32), .bfloat16);
        const x2 = try g.reshape(x, &.{ ci(rows), ci(in) });
        var want: [2]T = undefined;
        for (P, &want) |mp, *w| w.* = try g.adopt(try root.linear(s, x2, mp.trellis, mp.suh, mp.svh, .mul1));
        const g_want = try servedG(&g, s, x2, gw.served, rows);
        const got = try root.ling.mlaProjG(s, x, P, gw.member, .mul1);
        var gotx: [3]T = undefined;
        for (got, &gotx) |y, *w| w.* = try g.adopt(y);
        try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(outs[0]) }, mlx.getShape(gotx[0]));
        try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(outs[1]) }, mlx.getShape(gotx[1]));
        try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(out_g) }, mlx.getShape(gotx[2]));
        const d_qa = try differ(a, &g, gotx[0], want[0]);
        const d_kva = try differ(a, &g, gotx[1], want[1]);
        const d_g = try differ(a, &g, gotx[2], g_want);
        const ok = d_qa == 0 and d_kva == 0 and d_g == 0;
        std.debug.print("TURBOEXL3_LING_MLA_PROJ rows={d} q_a_differing={d}/{d} kv_a_differing={d}/{d} g_differing={d}/{d} served_g={s} {s}\n", .{ rows, d_qa, rows * outs[0], d_kva, rows * outs[1], d_g, rows * out_g, if (rows == 1) "matmul" else "denseRows", if (ok) "PASS" else "FAIL" });
        if (!ok) failed += 1;
    }
    // the refusals, before any launch: an f32 x, a weight not [out, in] bf16, a width MLX's gemv takes otherwise
    {
        const x32 = try g.zeros(&.{ 1, ci(in) }, .float32);
        try testing.expectError(error.Exl3EngineNotImplemented, root.ling.mlaProjG(s, x32, P, gw.member, .mul1));
        const xb = try g.zeros(&.{ 1, ci(in) }, .bfloat16);
        try testing.expectError(error.Exl3EngineNotImplemented, root.ling.mlaProjG(s, xb, P, gw.served, .mul1));
        const wide = try g.zeros(&.{ 512, ci(in) }, .bfloat16);
        try testing.expectError(error.Exl3EngineNotImplemented, root.ling.mlaProjG(s, xb, P, wide, .mul1));
        try testing.expectError(error.Exl3EngineNotImplemented, root.ling.mlaProjG(s, xb, P, try g.astype(gw.member, .float16), .mul1));
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── timing (TURBOEXL3_LING_TIME=1): guidance for the serving plan, never a speed verdict ──

/// Ling-3.1-flash's MLA layers
const time_layers: usize = 10;

/// One MLA layer's projection operands
const Layer = struct { p: [2]root.Proj, w: GateWeights };

const Arm = enum { served, fused };

/// One round of `arm` over the chained layers at `rows` rows: each layer's q_a, kv_a and gate of the layer's input, the
/// next layer's input that input plus its gate's first column times zero; the arrays to evaluate (every layer's three
/// outputs and the last input) into `outs_`, and the last layer's three
fn chainRound(g: *G, s: mlx.mlx_stream, x0: T, zero: T, layers: []const Layer, rows: u32, arm: Arm, outs_: []T) ![3]T {
    var x = x0;
    var last: [3]T = undefined;
    for (layers, 0..) |l, i| {
        const y: [3]T = switch (arm) {
            .served => .{
                try g.adopt(try root.linear(s, x, l.p[0].trellis, l.p[0].suh, l.p[0].svh, .mul1)),
                try g.adopt(try root.linear(s, x, l.p[1].trellis, l.p[1].suh, l.p[1].svh, .mul1)),
                try servedG(g, s, x, l.w.served, rows),
            },
            .fused => blk: {
                const o = try root.ling.mlaProjG(s, x, l.p, l.w.member, .mul1);
                break :blk .{ try g.adopt(o[0]), try g.adopt(o[1]), try g.adopt(o[2]) };
            },
        };
        for (y, 0..) |v, k| outs_[i * 3 + k] = v;
        const dep = try g.mul(try g.slice(y[2], &.{ 0, 0 }, &.{ ci(rows), 1 }, &.{ 1, 1 }), zero);
        x = try g.add(x, dep);
        last = y;
    }
    outs_[layers.len * 3] = x;
    return last;
}

// Ten MLA layers' projections of the layer's input timed on the device (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; it
// asserts nothing): guidance for the serving plan, never a speed verdict (the standard test through the outside plugin
// decides speed). Random trellises at K6 (deepseek_test's draw), each layer its own weights; x bf16 [rows, 4096]. Per
// round the chain's graph is built and evaluated alone (2 warm-up rounds, then 8): served (q_a's and kv_a's `linear`,
// the gate's matmul or denseRows) against `mlaProjG`; the min and median us of a round, the saving a layer at each, and
// the last layer's differing words between the arms.
test "turboexl3 gpu ling mlaproj timing: ten MLA layers' q_a, kv_a and gate, the served launches against mlaProjG's two, at rows 1 to 4, and their bytes (TURBOEXL3_LING_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_LING_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x3A1A_71E);
    const rnd = prng.random();
    var layers: [time_layers]Layer = undefined;
    for (&layers) |*l| {
        l.p = .{ try ds_test.randProj(a, &g, rnd, null, in, outs[0], 96), try ds_test.randProj(a, &g, rnd, null, in, outs[1], 96) };
        l.w = try gateWeights(a, &g, rnd);
    }
    const zero = try g.zeros(&.{ 1, 1 }, .bfloat16);
    var outs_buf: [time_layers * 3 + 1]T = undefined;
    for ([_]u32{ 1, 2, 3, 4 }) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        const x0 = try ds_test.randBf16(a, &g, rnd, &.{ ci(rows), ci(in) });
        try g.evalAll(&.{x0});
        var res: [2][2]f64 = undefined;
        var last: [2][3]T = undefined;
        for ([_]Arm{ .served, .fused }, 0..) |arm, ai| {
            var t: [8]f64 = undefined;
            for (0..t.len + 2) |round| {
                const m = g.mark();
                const t0 = ds_test.nowNs();
                const y = try chainRound(&g, s, x0, zero, &layers, rows, arm, &outs_buf);
                try g.evalAll(&outs_buf);
                const dt = ds_test.nowNs() - t0;
                if (round >= 2) t[round - 2] = @as(f64, @floatFromInt(dt)) / 1000.0;
                if (round == t.len + 1) {
                    for (y, &last[ai]) |v, *k| k.* = g.keep(v);
                }
                g.resetTo(m);
            }
            std.mem.sort(f64, &t, {}, std.sort.asc(f64));
            res[ai] = .{ t[0], t[t.len / 2] };
        }
        defer for (last) |arm_last| for (arm_last) |v| g.release(v);
        var d: usize = 0;
        for (last[0], last[1]) |u, v| d += try differ(a, &g, v, u);
        const n_words = rows * (outs[0] + outs[1] + out_g);
        const lf: f64 = @floatFromInt(time_layers);
        std.debug.print("TURBOEXL3_LING_MLA_PROJ_TIME rows={d} layers={d} served_us={d:.1}/{d:.1} fused_us={d:.1}/{d:.1} saved_a_layer_us={d:.2}/{d:.2} (min/median) last_layer_differing={d}/{d}\n", .{ rows, time_layers, res[0][0], res[0][1], res[1][0], res[1][1], (res[0][0] - res[1][0]) / lf, (res[0][1] - res[1][1]) / lf, d, n_words });
    }
}
