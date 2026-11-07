//! Ling-3.1-flash's MoE router in two launches (`models/ling/router.zig`) on the GPU (TURBOEXL3_GPU=1, else skipped), at
//! Ling's shapes (hidden 4096, 512 experts, the top-8 of the 4 best of 8 groups of 64, routed_scaling_factor 2.5),
//! against the host's launches word for word at rows 1 to 8:
//! - the logits: each row MLX's one-row `matmul(astype(x_r, f32), w)` over the f32 router table (gemv_t: serial
//!   decode's words, and at 2 to 8 rows the exact verify's, turboexl3_ling's denseRows), with and without a predictor;
//! - the predictor's scores: the host streamer's `nextScores`, MLX's `add(sqrt(logaddexp(astype(matmul(x,
//!   transpose(pw)), f32), 0)), pb)` over the call's rows (gemv at one row, gemv_wide at 2 to 8), and Ling's own
//!   `add(sigmoid(...), pb)` (candidate C1);
//! - the ids and weights: the host's fused router (a reference launch of `router.text_select` at the host's config,
//!   TOUT f32) over the reference logits, then MLX's astype of its weights to bf16; `select` alone and `route`;
//! - the refusals.
//! The rows run from routine to saturating magnitudes (both arms of the host's sigmoid, every branch of its log1p) and
//! include a row of signed zeros. Lines: TURBOEXL3_LING_ROUTER.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ks = sdk.kernel_launch;
const support = @import("../../core/test_support.zig");
const decode = @import("../../core/decode.zig");
const G = support.Graph;
const T = mlx.mlx_array;
const router = @import("router.zig");

const testing = std.testing;
const hidden: usize = 4096;
const ne: usize = 512;
const nk: usize = 8;
/// Ling-3.1-flash's routed_scaling_factor
const scale: f32 = 2.5;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// MLX's logaddexp and sqrt (the host streamer's `nextScores` ops): the consumer's bindings where its `sdk.mlx` declares
/// them, else libmlxc's own symbols
const logaddexpOp = if (@hasDecl(mlx, "mlx_logaddexp")) mlx.mlx_logaddexp else struct {
    extern "c" fn mlx_logaddexp(res: *mlx.mlx_array, a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) c_int;
}.mlx_logaddexp;
const sqrtOp = if (@hasDecl(mlx, "mlx_sqrt")) mlx.mlx_sqrt else struct {
    extern "c" fn mlx_sqrt(res: *mlx.mlx_array, a: mlx.mlx_array, s: mlx.mlx_stream) c_int;
}.mlx_sqrt;

fn adopted(g: *G, rc: c_int, r: T) !T {
    mlx.check(rc) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
}

/// The host's router logits: each row of x its own one-row matmul of the row widened to f32 and the f32 table (MLX's
/// gemv_t), joined
fn hostLogits(g: *G, x: T, w: T, rows: usize) !T {
    var parts: [8]T = undefined;
    for (0..rows) |r| {
        const xr = try g.slice(x, &.{ ci(r), 0 }, &.{ ci(r + 1), ci(hidden) }, &.{ 1, 1 });
        parts[r] = try g.matmul(try g.astype(xr, .float32), w);
    }
    return if (rows == 1) parts[0] else g.concat(parts[0..rows], 0);
}

/// The host streamer's predictor over every row of x: z = astype(matmul(x, transpose(pw)), f32), then
/// add(sqrt(logaddexp(z, 0)), pb) (`nextScores`) or add(sigmoid(z), pb) (C1's), op for op
fn hostScores(g: *G, s: mlx.mlx_stream, x: T, pw: T, pb: T, score: router.Score) !T {
    const z = try g.astype(try g.matmul(x, try g.transposeAxes(pw, &.{ 1, 0 })), .float32);
    const f = switch (score) {
        .softplus_sqrt => blk: {
            const zero = try g.scalar(0, .float32);
            var la = mlx.mlx_array_new();
            const rc_la = logaddexpOp(&la, z, zero, s);
            const soft = try adopted(g, rc_la, la);
            var sq = mlx.mlx_array_new();
            const rc_sq = sqrtOp(&sq, soft, s);
            break :blk try adopted(g, rc_sq, sq);
        },
        .sigmoid => try g.sigmoid(z),
    };
    return g.add(f, pb);
}

/// The host's fused router at its config (`moeRouterTopK`: a threadgroup of 256 a row, TOUT f32) over logits f32
/// [rows, 512], then the MoE block's astype of the weights to x's dtype (bf16): ids u32 and weights bf16 [rows, 8]
fn hostSelect(g: *G, s: mlx.mlx_stream, k: mlx.mlx_fast_metal_kernel, logits: T, bias: T, rows: usize) ![2]T {
    const out = [_]c_int{ ci(rows), ci(nk) };
    const c = try ks.buildConfig(.{
        .grid = .{ 256, @intCast(rows), 1 },
        .threadgroup = .{ 256, 1, 1 },
        .out_shapes = &.{ &out, &out },
        .out_dtypes = &.{ .uint32, .float32 },
        .template = &.{
            .{ .name = "TOUT", .value = .{ .dtype = .float32 } },
            .{ .name = "NE", .value = .{ .int = 512 } },
            .{ .name = "NK", .value = .{ .int = 8 } },
            .{ .name = "TG", .value = .{ .int = 256 } },
            .{ .name = "NORM", .value = .{ .int = 1 } },
            .{ .name = "NG", .value = .{ .int = 8 } },
            .{ .name = "GSZ", .value = .{ .int = 64 } },
            .{ .name = "TOPKG", .value = .{ .int = 4 } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const sc = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(sc);
    var o: [2]T = undefined;
    try ks.applyConfig(k, c, &.{ logits, bias, sc }, &o, s);
    const inds = try g.adopt(o[0]);
    const w32 = try g.adopt(o[1]);
    return .{ inds, try g.astype(w32, .bfloat16) };
}

/// x's rows by pattern: normal values at a scale from routine to saturating (the router's sigmoid at both ends, the
/// predictor's z past logaddexp's every log1p branch), 0 a row of signed zeros
const row_scales = [_]f32{ 1.0, 8.0, 0.0, 0.25, 32.0, 2.0, 64.0, 0.5 };

/// The words of `got` that differ from `want` (f32 or u32), the first few printed
fn differing(label: []const u8, rows: usize, rep: usize, got: anytype, want: anytype, cols: usize) usize {
    var n: usize = 0;
    for (got, want, 0..) |u, v, i| {
        const bu: u32 = @bitCast(u);
        const bv: u32 = @bitCast(v);
        if (bu == bv) continue;
        if (n < 6) std.debug.print("TURBOEXL3_LING_ROUTER {s} rows={d} rep={d} differs at row={d} col={d}: got 0x{x:0>8} want 0x{x:0>8}\n", .{ label, rows, rep, i / cols, i % cols, bu, bv });
        n += 1;
    }
    return n;
}

/// Each row's ids are 8 distinct experts of at most 4 groups of 64 (the reference is the host's rule, not a broken one)
fn idsSane(ids: []const u32, rows: usize) bool {
    for (0..rows) |r| {
        const row = ids[r * nk ..][0..nk];
        var groups: u32 = 0;
        for (row, 0..) |e, j| {
            if (e >= ne) return false;
            for (row[0..j]) |f| if (f == e) return false;
            groups |= @as(u32, 1) << @intCast(e / 64);
        }
        if (@popCount(groups) > 4) return false;
    }
    return true;
}

test "turboexl3 gpu ling router: routerLogits, select and route give the host's logits (one-row gemv_t a row), predictor scores (DeepSeek's and Ling's, gemv at one row and gemv_wide at 2 to 8) and ids and bf16 weights (its fused router, then the astype) word for word at rows 1 to 8 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x11C0_E2E2);
    const rnd = prng.random();

    // the router table f32 [hidden, 512] (f16-exact: Ling's gate weight widened), its bias; the next layer's router bf16
    // [512, hidden] and bias
    const w = try a.alloc(f32, hidden * ne);
    defer a.free(w);
    for (w) |*v| v.* = @as(f16, @floatCast(rnd.floatNorm(f32) * 0.02));
    const pw = try a.alloc(u16, ne * hidden);
    defer a.free(pw);
    for (pw) |*v| v.* = support.bf16Bits(rnd.floatNorm(f32) * 0.02);
    var bias: [ne]f32 = undefined;
    for (&bias) |*v| v.* = (rnd.float(f32) - 0.5) * 0.6;
    var pbias: [ne]f32 = undefined;
    for (&pbias) |*v| v.* = (rnd.float(f32) - 0.5) * 0.6;
    const wa = try g.hostArray(std.mem.sliceAsBytes(w), &.{ ci(hidden), ci(ne) }, .float32);
    const pwa = try g.hostArray(std.mem.sliceAsBytes(pw), &.{ ci(ne), ci(hidden) }, .bfloat16);
    const ba = try g.hostArray(std.mem.sliceAsBytes(&bias), &.{ci(ne)}, .float32);
    const pba = try g.hostArray(std.mem.sliceAsBytes(&pbias), &.{ci(ne)}, .float32);
    try g.evalAll(&.{ wa, pwa, ba, pba });

    const href = ks.newKernel("turboexl3_test_ling_router_host", &router.select_in, &router.select_out, router.text_select, "", true) orelse return error.KernelCreateFailed;
    defer decode.freeKernel(href);

    const sp: router.Predictor = .{ .w = pwa, .bias = pba };
    const sig: router.Predictor = .{ .w = pwa, .bias = pba, .score = .sigmoid };
    var failed = false;
    var calls: usize = 0;
    const xbits = try a.alloc(u16, 8 * hidden);
    defer a.free(xbits);
    const fbuf = try a.alloc(f32, 2 * 8 * ne);
    defer a.free(fbuf);
    var ubuf: [2 * 8 * nk]u32 = undefined;
    var wbuf: [2 * 8 * nk]f32 = undefined;
    for (1..9) |rows| {
        const reps: usize = if (rows == 1) row_scales.len else 2;
        for (0..reps) |rep| {
            const mark = g.mark();
            defer g.resetTo(mark);
            for (0..rows) |r| {
                const sc = row_scales[(r + rep * rows) % row_scales.len];
                for (xbits[r * hidden ..][0..hidden]) |*v| v.* = support.bf16Bits(rnd.floatNorm(f32) * sc);
            }
            const xa = try g.hostArray(std.mem.sliceAsBytes(xbits[0 .. rows * hidden]), &.{ ci(rows), ci(hidden) }, .bfloat16);
            const n = rows * ne;
            const want_l = fbuf[0..n];
            const got = fbuf[n .. 2 * n];
            _ = try g.hostF32(try hostLogits(&g, xa, wa, rows), want_l);
            var d_logits: usize = 0;
            var d_scores: usize = 0;
            // the logits alone (the last MoE layer), then beside each score
            for ([_]?router.Predictor{ null, sp, sig }) |pred| {
                const l = try router.routerLogits(s, xa, wa, pred);
                const lg = try g.adopt(l.logits);
                const ls = if (l.scores) |x| try g.adopt(x) else null;
                try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(ne) }, mlx.getShape(lg));
                _ = try g.hostF32(lg, got);
                d_logits += differing("logits", rows, rep, got, want_l, ne);
                const p = pred orelse continue;
                const want_s = try a.alloc(f32, n);
                defer a.free(want_s);
                _ = try g.hostF32(try hostScores(&g, s, xa, pwa, pba, p.score), want_s);
                _ = try g.hostF32(ls.?, got);
                d_scores += differing(if (p.score == .sigmoid) "scores_sigmoid" else "scores", rows, rep, got, want_s, ne);
            }
            // the select alone over the host's logits, and route end to end, against the host's router and astype
            const ref_logits = try hostLogits(&g, xa, wa, rows);
            const ref = try hostSelect(&g, s, href, ref_logits, ba, rows);
            const m = rows * nk;
            _ = try g.hostU32(ref[0], ubuf[0..m]);
            _ = try g.hostF32(try g.astype(ref[1], .float32), wbuf[0..m]);
            const sel = try router.select(s, ref_logits, ba, scale);
            const rt = try router.route(s, xa, wa, ba, scale, sp);
            var d_ids: usize = 0;
            var d_w: usize = 0;
            for ([_][2]T{ .{ try g.adopt(sel.inds), try g.adopt(sel.weights) }, .{ try g.adopt(rt.inds), try g.adopt(rt.weights) } }) |pair| {
                try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(nk) }, mlx.getShape(pair[0]));
                try testing.expect(mlx.mlx_array_dtype(pair[1]) == .bfloat16);
                _ = try g.hostU32(pair[0], ubuf[m .. 2 * m]);
                _ = try g.hostF32(try g.astype(pair[1], .float32), wbuf[m .. 2 * m]);
                d_ids += differing("ids", rows, rep, ubuf[m .. 2 * m], ubuf[0..m], nk);
                d_w += differing("weights", rows, rep, wbuf[m .. 2 * m], wbuf[0..m], nk);
            }
            const want_r = try a.alloc(f32, n);
            defer a.free(want_r);
            _ = try g.hostF32(try hostScores(&g, s, xa, pwa, pba, .softplus_sqrt), want_r);
            _ = try g.hostF32(try g.adopt(rt.scores.?), got);
            d_scores += differing("route_scores", rows, rep, got, want_r, ne);
            const sane = idsSane(ubuf[0..m], rows);
            const ok = d_logits == 0 and d_scores == 0 and d_ids == 0 and d_w == 0 and sane;
            std.debug.print("TURBOEXL3_LING_ROUTER rows={d} rep={d} logits differing={d}/{d} scores differing={d}/{d} ids differing={d}/{d} weights differing={d}/{d} ids_sane={} {s}\n", .{ rows, rep, d_logits, 3 * n, d_scores, 3 * n, d_ids, 2 * m, d_w, 2 * m, sane, if (ok) "PASS" else "FAIL" });
            if (!ok) failed = true;
            calls += 1;
        }
    }
    std.debug.print("TURBOEXL3_LING_ROUTER {d} calls {s}\n", .{ calls, if (failed) "FAIL" else "PASS" });
    try testing.expect(!failed);
}

test "turboexl3 gpu ling router refusals: rows past 8, a hidden size off the texts' tiles, a shape or dtype off (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(std.heap.page_allocator, s);
    defer g.deinit();
    const x = try g.zeros(&.{ 2, ci(hidden) }, .bfloat16);
    const w = try g.zeros(&.{ ci(hidden), ci(ne) }, .float32);
    const pw = try g.zeros(&.{ ci(ne), ci(hidden) }, .bfloat16);
    const b = try g.zeros(&.{ci(ne)}, .float32);
    const e = error.BadExl3Shape;
    const dt = error.BadExl3Dtype;
    try testing.expectError(e, router.routerLogits(s, try g.zeros(&.{ 9, ci(hidden) }, .bfloat16), w, null));
    try testing.expectError(e, router.routerLogits(s, try g.zeros(&.{ 2, 4000 }, .bfloat16), try g.zeros(&.{ 4000, ci(ne) }, .float32), null));
    try testing.expectError(e, router.routerLogits(s, try g.zeros(&.{ 1, 2, ci(hidden) }, .bfloat16), w, null));
    try testing.expectError(dt, router.routerLogits(s, try g.zeros(&.{ 2, ci(hidden) }, .float32), w, null));
    try testing.expectError(e, router.routerLogits(s, x, try g.zeros(&.{ ci(hidden), 256 }, .float32), null));
    try testing.expectError(dt, router.routerLogits(s, x, try g.zeros(&.{ ci(hidden), ci(ne) }, .bfloat16), null));
    try testing.expectError(e, router.routerLogits(s, x, w, .{ .w = try g.zeros(&.{ ci(ne), 2048 }, .bfloat16), .bias = b }));
    try testing.expectError(dt, router.routerLogits(s, x, w, .{ .w = try g.zeros(&.{ ci(ne), ci(hidden) }, .float16), .bias = b }));
    try testing.expectError(dt, router.routerLogits(s, x, w, .{ .w = pw, .bias = try g.zeros(&.{ci(ne)}, .bfloat16) }));
    try testing.expectError(e, router.select(s, try g.zeros(&.{ 9, ci(ne) }, .float32), b, scale));
    try testing.expectError(e, router.select(s, try g.zeros(&.{ 2, 256 }, .float32), b, scale));
    try testing.expectError(dt, router.select(s, try g.zeros(&.{ 2, ci(ne) }, .bfloat16), b, scale));
    try testing.expectError(e, router.route(s, x, w, try g.zeros(&.{256}, .float32), scale, null));
}
