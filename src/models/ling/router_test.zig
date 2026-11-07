//! Ling-3.1-flash's MoE router in two launches (`models/ling/router.zig`) on the GPU (TURBOEXL3_GPU=1, else skipped),
//! at Ling's shapes (hidden 4096, 512 experts, the top-8 of the 4 best of 8 groups of 64, routed_scaling_factor 2.5),
//! against the host's (turboexl3_ling's) served path word for word. Every operand is built the way the host builds it,
//! not only with its values:
//! - the router weight: f16 [512, hidden] row-major, as the loader reads it from the pack;
//! - the router table: the loader's `transposed` (MLX's transpose of that weight: a [hidden, 512] view) and `f32Table`
//!   (MLX's astype to f32, then `materializedOwnedCopy`: + a 0-dim f32 zero), which keep the view's layout, so the
//!   table is column-major (strides 1, hidden; the test checks it);
//! - what turboexl3_ling's fused router passes: the table transposed back (a [512, hidden] row-major view), or with the
//!   f16 weight kept (MLX_SERVE_LING_ROUTER_F16=1) the weight's transposed view transposed back (the loaded buffer);
//! - x: the MoE block's input [1, S, hidden] bf16 (the stock matmul's operand), reshaped to [S, hidden] for the fused
//!   router and the streamer (its `reshape(x, [B * S, D])`);
//! - the predictor's gate: `router_raw`, the next layer's loaded weight cast to bf16 (`astypeEval`), and its f32 bias.
//! The references are the served ops on those operands: the router's logits at one row are `matmul(x, table)` (Linear's
//! qmatmulBits: MLX promotes x to f32 and runs gemv over the column-major table) and at 2 to 8 rows the exact verify's
//! denseRows (x reshaped [rows, 1, hidden], cast to f32, stored twice and sliced to a batch stride of two rows, matmul
//! with the table expanded [1, hidden, 512]); the predictor's scores the streamer's nextScores; the ids and weights the
//! host's fused router at its config (`moeRouterTopK`, a launch of `router.text_select` at TOUT f32) over the served
//! logits [1, S, 512], then the block's astype to bf16. Lines: TURBOEXL3_LING_ROUTER.
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

/// MLX's logaddexp, sqrt and expand_dims (the streamer's nextScores, denseRows): the consumer's bindings where its
/// `sdk.mlx` declares them, else libmlxc's own symbols
const logaddexpOp = if (@hasDecl(mlx, "mlx_logaddexp")) mlx.mlx_logaddexp else struct {
    extern "c" fn mlx_logaddexp(res: *mlx.mlx_array, a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) c_int;
}.mlx_logaddexp;
const sqrtOp = if (@hasDecl(mlx, "mlx_sqrt")) mlx.mlx_sqrt else struct {
    extern "c" fn mlx_sqrt(res: *mlx.mlx_array, a: mlx.mlx_array, s: mlx.mlx_stream) c_int;
}.mlx_sqrt;
const expandDims = if (@hasDecl(mlx, "mlx_expand_dims")) mlx.mlx_expand_dims else struct {
    extern "c" fn mlx_expand_dims(res: *mlx.mlx_array, a: mlx.mlx_array, axis: c_int, s: mlx.mlx_stream) c_int;
}.mlx_expand_dims;

fn adopted(g: *G, rc: c_int, r: T) !T {
    mlx.check(rc) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
}

/// The router weight as the loader reads it: f16 [512, hidden] row-major from its words
fn loadedWeight(g: *G, words: []const u16) !T {
    return g.hostArray(std.mem.sliceAsBytes(words), &.{ ci(ne), ci(hidden) }, .float16);
}

/// The loader's `transposed`: MLX's transpose of the weight's last two axes (a view, [hidden, 512] column-major)
fn loaderView(g: *G, w: T) !T {
    return g.transposeAxes(w, &.{ 1, 0 });
}

/// The loader's `f32Table` over that view: MLX's astype to f32, then `materializedOwnedCopy` (+ a 0-dim zero of the
/// dtype), evaluated
fn loaderTable(g: *G, w: T) !T {
    const zero = try g.zeros(&.{}, .float32);
    const t = try g.add(try g.astype(try loaderView(g, w), .float32), zero);
    try g.evalAll(&.{t});
    return t;
}

/// The plugin's `router_raw` (the predictor's gate): the next layer's loaded weight cast to bf16 (`astypeEval`)
fn routerRaw(g: *G, w: T) !T {
    const r = try g.astype(w, .bfloat16);
    try g.evalAll(&.{r});
    return r;
}

/// The served router logits [1, rows, 512]: at one row Linear's matmul of the MoE input [1, 1, hidden] and the table,
/// at 2 to 8 rows the exact verify's denseRows (one batched one-row gemv a row)
fn servedLogits(g: *G, s: mlx.mlx_stream, x3: T, table: T, rows: usize) !T {
    if (rows == 1) return g.matmul(x3, table);
    const xr = try g.astype(try g.reshape(x3, &.{ ci(rows), 1, ci(hidden) }), .float32);
    const two = try g.concat(&.{ xr, xr }, 1);
    const xs = try g.slice(two, &.{ 0, 0, 0 }, &.{ ci(rows), 1, ci(hidden) }, &.{ 1, 1, 1 });
    var we = mlx.mlx_array_new();
    const wb = try adopted(g, expandDims(&we, table, 0, s), we);
    return g.reshape(try g.matmul(xs, wb), &.{ 1, ci(rows), ci(ne) });
}

/// The streamer's nextScores over the call's rows xf [rows, hidden]: z = astype(matmul(xf, transpose(gate)), f32), then
/// add(sqrt(logaddexp(z, 0)), bias) or C1's add(sigmoid(z), bias), op for op
fn servedScores(g: *G, s: mlx.mlx_stream, xf: T, gate: T, bias: T, score: router.Score) !T {
    var tr = mlx.mlx_array_new();
    const gt = try adopted(g, mlx.mlx_transpose(&tr, gate, s), tr);
    const z = try g.astype(try g.matmul(xf, gt), .float32);
    const f = switch (score) {
        .softplus_sqrt => blk: {
            const zero = try g.adopt(mlx.mlx_array_new_float(0));
            var la = mlx.mlx_array_new();
            const rc_la = logaddexpOp(&la, z, zero, s);
            const soft = try adopted(g, rc_la, la);
            var sq = mlx.mlx_array_new();
            const rc_sq = sqrtOp(&sq, soft, s);
            break :blk try adopted(g, rc_sq, sq);
        },
        .sigmoid => try g.sigmoid(z),
    };
    return g.add(f, bias);
}

/// The host's fused router at its config (`moeRouterTopK`: the logits' shape with k last, a threadgroup of 256 a row,
/// TOUT f32, the scale a 0-dim array) over the served logits [1, rows, 512], then the MoE block's astype of the weights
/// to x's dtype (bf16): ids u32 and weights bf16 [1, rows, 8]
fn servedSelect(g: *G, s: mlx.mlx_stream, k: mlx.mlx_fast_metal_kernel, logits: T, bias: T, rows: usize) ![2]T {
    const out = [_]c_int{ 1, ci(rows), ci(nk) };
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

/// x's rows by pattern: normal values at a scale from routine to saturating (the router's sigmoid at both ends, the
/// predictor's z past logaddexp's every log1p branch), 0 a row of signed zeros
const row_scales = [_]f32{ 1.0, 8.0, 0.0, 0.25, 32.0, 2.0, 64.0, 0.5 };

/// A router weight's f16 words, router-like: normal(0, 0.02) rounded to f16 (subnormals and signed zeros among them)
fn fillWeight(rnd: std.Random, words: []u16) void {
    for (words) |*v| v.* = @bitCast(@as(f16, @floatCast(rnd.floatNorm(f32) * 0.02)));
}

/// The table stride check: the loader's table is [hidden, 512] column-major
fn tableIsColumnMajor(t: T) bool {
    const sh = mlx.getShape(t);
    const st = mlx.mlx_array_strides(t);
    return sh.len == 2 and sh[0] == ci(hidden) and sh[1] == ci(ne) and st[0] == 1 and st[1] == hidden;
}

test "turboexl3 gpu ling router: routerLogits, select and route over the loader's table (f32, transposed back) and over the loaded f16 weight give the served logits (one-row gemv over the column-major table, denseRows at 2 to 8), predictor scores (DeepSeek's and Ling's, gemv at one row and gemv_wide at 2 to 8) and ids and bf16 weights word for word at rows 1 to 8 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x11C0_E2E3);
    const rnd = prng.random();

    // this layer's router weight and bias, the next layer's (the predictor's gate), each as the host loads them
    const words = try a.alloc(u16, ne * hidden);
    defer a.free(words);
    fillWeight(rnd, words);
    const wl = try loadedWeight(&g, words);
    fillWeight(rnd, words);
    const wn = try loadedWeight(&g, words);
    var bias: [ne]f32 = undefined;
    for (&bias) |*v| v.* = (rnd.float(f32) - 0.5) * 0.6;
    var pbias: [ne]f32 = undefined;
    for (&pbias) |*v| v.* = (rnd.float(f32) - 0.5) * 0.6;
    const ba = try g.hostArray(std.mem.sliceAsBytes(&bias), &.{ci(ne)}, .float32);
    const pba = try g.hostArray(std.mem.sliceAsBytes(&pbias), &.{ci(ne)}, .float32);
    const table = try loaderTable(&g, wl);
    try testing.expect(tableIsColumnMajor(table));
    const pw = try routerRaw(&g, wn);
    // what the fused router reads: the table transposed back (f32) or the loaded weight's view transposed back (f16)
    const w32 = try g.transposeAxes(table, &.{ 1, 0 });
    const w16 = try g.transposeAxes(try loaderView(&g, wl), &.{ 1, 0 });
    try g.evalAll(&.{ pw, ba, pba });

    const href = ks.newKernel("turboexl3_test_ling_router_host", &router.select_in, &router.select_out, router.text_select, "", true) orelse return error.KernelCreateFailed;
    defer decode.freeKernel(href);

    var failed = false;
    var calls: usize = 0;
    const xbits = try a.alloc(u16, 8 * hidden);
    defer a.free(xbits);
    const fbuf = try a.alloc(f32, 3 * 8 * ne);
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
            // the MoE block's input [1, S, hidden], and the [S, hidden] view the fused router and the streamer take
            const x3 = try g.hostArray(std.mem.sliceAsBytes(xbits[0 .. rows * hidden]), &.{ 1, ci(rows), ci(hidden) }, .bfloat16);
            const xf = try g.reshape(x3, &.{ ci(rows), ci(hidden) });
            const n = rows * ne;
            const want_l = fbuf[0..n];
            const want_s = fbuf[n .. 2 * n];
            const got = fbuf[2 * n .. 3 * n];
            const served = try servedLogits(&g, s, x3, table, rows);
            _ = try g.hostF32(served, want_l);
            const m = rows * nk;
            const ref = try servedSelect(&g, s, href, served, ba, rows);
            _ = try g.hostU32(ref[0], ubuf[0..m]);
            _ = try g.hostF32(try g.astype(ref[1], .float32), wbuf[0..m]);
            var d_logits: usize = 0;
            var d_scores: usize = 0;
            var d_ids: usize = 0;
            var d_w: usize = 0;
            var checks: usize = 0;
            for ([_]T{ w32, w16 }, [_][]const u8{ "f32", "f16" }) |w, wname| {
                // the logits alone (the last MoE layer), then beside each score
                for ([_]?router.Predictor{ null, .{ .w = pw, .bias = pba }, .{ .w = pw, .bias = pba, .score = .sigmoid } }) |pred| {
                    const l = try router.routerLogits(s, xf, w, pred);
                    const lg = try g.adopt(l.logits);
                    const ls = if (l.scores) |scr| try g.adopt(scr) else null;
                    try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(ne) }, mlx.getShape(lg));
                    _ = try g.hostF32(lg, got);
                    d_logits += differing(wname, rows, rep, got, want_l, ne);
                    checks += 1;
                    const p = pred orelse continue;
                    _ = try g.hostF32(try servedScores(&g, s, xf, pw, pba, p.score), want_s);
                    _ = try g.hostF32(ls.?, got);
                    d_scores += differing(if (p.score == .sigmoid) "scores_sigmoid" else "scores", rows, rep, got, want_s, ne);
                }
                // route end to end, and the select alone over the served logits (the [S, 512] view of them)
                const rt = try router.route(s, xf, w, ba, scale, .{ .w = pw, .bias = pba });
                const sel = try router.select(s, try g.reshape(served, &.{ ci(rows), ci(ne) }), ba, scale);
                for ([_][2]T{ .{ try g.adopt(sel.inds), try g.adopt(sel.weights) }, .{ try g.adopt(rt.inds), try g.adopt(rt.weights) } }) |pair| {
                    try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(nk) }, mlx.getShape(pair[0]));
                    try testing.expect(mlx.mlx_array_dtype(pair[1]) == .bfloat16);
                    _ = try g.hostU32(pair[0], ubuf[m .. 2 * m]);
                    _ = try g.hostF32(try g.astype(pair[1], .float32), wbuf[m .. 2 * m]);
                    d_ids += differing("ids", rows, rep, ubuf[m .. 2 * m], ubuf[0..m], nk);
                    d_w += differing("weights", rows, rep, wbuf[m .. 2 * m], wbuf[0..m], nk);
                }
                _ = try g.hostF32(try servedScores(&g, s, xf, pw, pba, .softplus_sqrt), want_s);
                _ = try g.hostF32(try g.adopt(rt.scores.?), got);
                d_scores += differing("route_scores", rows, rep, got, want_s, ne);
            }
            const sane = idsSane(ubuf[0..m], rows);
            const ok = d_logits == 0 and d_scores == 0 and d_ids == 0 and d_w == 0 and sane;
            std.debug.print("TURBOEXL3_LING_ROUTER rows={d} rep={d} logits differing={d}/{d} scores differing={d}/{d} ids differing={d}/{d} weights differing={d}/{d} ids_sane={} {s}\n", .{ rows, rep, d_logits, checks * n, d_scores, 6 * n, d_ids, 4 * m, d_w, 4 * m, sane, if (ok) "PASS" else "FAIL" });
            if (!ok) failed = true;
            calls += 1;
        }
    }
    std.debug.print("TURBOEXL3_LING_ROUTER {d} calls, each over the f32 table and the f16 weight {s}\n", .{ calls, if (failed) "FAIL" else "PASS" });
    try testing.expect(!failed);
}

/// The f16 weight's words with every finite f16 word in it (each about 33 times, scrambled; -0.0, +0.0 and every
/// subnormal included), then expert `adv`'s row: 2^-24 at k = 4 l (l 0 to 31, gemv's lanes' first words), -0.0 at
/// k = 1, 1.0 elsewhere. With x's adversarial row (-2^-126 at k = 4 l, 1.0 at k = 1, -0.0 elsewhere) each lane's first
/// product underflows to -0.0, and only lane 0's sum then meets the table's word at k = 1: +0.0 (the loader's + 0) keeps
/// the logit +0.0; a -0.0 there would leave -0.0 (where MLX fuses the multiply into the add).
fn fillAllWords(rnd: std.Random, words: []u16, adv: usize) void {
    var finite: [63488]u16 = undefined;
    var nf: usize = 0;
    for (0..65536) |i| {
        const v: u16 = @intCast(i);
        if ((v & 0x7C00) != 0x7C00) {
            finite[nf] = v;
            nf += 1;
        }
    }
    std.debug.assert(nf == finite.len);
    for (words, 0..) |*w, i| w.* = finite[i % finite.len];
    rnd.shuffle(u16, words);
    const row = words[adv * hidden ..][0..hidden];
    for (row, 0..) |*w, k| w.* = if (k == 1) 0x8000 else if (k % 4 == 0 and k < 128) 0x0001 else 0x3C00;
}

test "turboexl3 gpu ling router f16 words: over a loaded f16 weight holding every finite f16 word (-0.0 and every subnormal among them), routerLogits gives the loader's table's words at rows 1 to 8, the f16 word widened then + 0, as the loader's x + 0 turns -0.0 into +0.0; an adversarial expert shows where a -0.0 would differ (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x11C0_E10E);
    const rnd = prng.random();
    const adv: usize = 7;
    const words = try a.alloc(u16, ne * hidden);
    defer a.free(words);
    fillAllWords(rnd, words, adv);
    const wl = try loadedWeight(&g, words);
    const table = try loaderTable(&g, wl);
    try testing.expect(tableIsColumnMajor(table));
    // the same view widened without the + 0: the table a -0.0 word would leave
    const bare = try g.astype(try loaderView(&g, wl), .float32);
    const w16 = try g.transposeAxes(try loaderView(&g, wl), &.{ 1, 0 });
    const xbits = try a.alloc(u16, 8 * hidden);
    defer a.free(xbits);
    const fbuf = try a.alloc(f32, 3 * 8 * ne);
    defer a.free(fbuf);
    var failed = false;
    var bare_differs: usize = 0;
    for (1..9) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        for (0..rows) |r| {
            const row = xbits[r * hidden ..][0..hidden];
            if (r == 0) {
                // the adversarial row
                for (row, 0..) |*v, k| v.* = if (k == 1) 0x3F80 else if (k % 4 == 0 and k < 128) 0x8080 else 0x8000;
            } else {
                for (row) |*v| v.* = support.bf16Bits(rnd.floatNorm(f32));
            }
        }
        const x3 = try g.hostArray(std.mem.sliceAsBytes(xbits[0 .. rows * hidden]), &.{ 1, ci(rows), ci(hidden) }, .bfloat16);
        const xf = try g.reshape(x3, &.{ ci(rows), ci(hidden) });
        const n = rows * ne;
        const want = fbuf[0..n];
        const got = fbuf[n .. 2 * n];
        const bare_l = fbuf[2 * n .. 3 * n];
        _ = try g.hostF32(try servedLogits(&g, s, x3, table, rows), want);
        _ = try g.hostF32(try servedLogits(&g, s, x3, bare, rows), bare_l);
        const l = try router.routerLogits(s, xf, w16, null);
        _ = try g.hostF32(try g.adopt(l.logits), got);
        const d = differing("f16_words", rows, 0, got, want, ne);
        const want_adv: u32 = @bitCast(want[adv]);
        const bare_adv: u32 = @bitCast(bare_l[adv]);
        if (want_adv != bare_adv) bare_differs += 1;
        std.debug.print("TURBOEXL3_LING_ROUTER f16_words rows={d} logits differing={d}/{d} adversarial expert: table 0x{x:0>8}, without the + 0 0x{x:0>8} {s}\n", .{ rows, d, n, want_adv, bare_adv, if (d == 0) "PASS" else "FAIL" });
        if (d != 0) failed = true;
    }
    std.debug.print("TURBOEXL3_LING_ROUTER f16_words: the + 0 changes the adversarial logit at {d} of 8 row counts {s}\n", .{ bare_differs, if (failed) "FAIL" else "PASS" });
    try testing.expect(!failed);
}

test "turboexl3 gpu ling router stock table: tableF32 rebuilds the loader's table from the f16 weight's view word for word and stride for stride, and MLX's matmul over it gives the table's words at the stock widths (rows 1 to 8, denseRows at 2 to 8, prompt chunks of 1996 and 4096 rows) (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x11C0_57CC);
    const rnd = prng.random();
    const words = try a.alloc(u16, ne * hidden);
    defer a.free(words);
    fillAllWords(rnd, words, 7);
    const wl = try loadedWeight(&g, words);
    const table = try loaderTable(&g, wl);
    const rebuilt = try g.adopt(try router.tableF32(s, try loaderView(&g, wl)));
    try g.evalAll(&.{rebuilt});
    try testing.expect(tableIsColumnMajor(table) and tableIsColumnMajor(rebuilt));
    const tw = try a.alloc(f32, ne * hidden);
    defer a.free(tw);
    const rw = try a.alloc(f32, ne * hidden);
    defer a.free(rw);
    _ = try g.hostF32(table, tw);
    _ = try g.hostF32(rebuilt, rw);
    const d_table = differing("table", hidden, 0, rw, tw, ne);
    std.debug.print("TURBOEXL3_LING_ROUTER stock table: tableF32 differing={d}/{d} strides ({d}, {d}) {s}\n", .{ d_table, ne * hidden, mlx.mlx_array_strides(rebuilt)[0], mlx.mlx_array_strides(rebuilt)[1], if (d_table == 0) "PASS" else "FAIL" });
    var failed = d_table != 0;
    const widths = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 1996, 4096 };
    const xbits = try a.alloc(u16, 4096 * hidden);
    defer a.free(xbits);
    for (xbits) |*v| v.* = support.bf16Bits(rnd.floatNorm(f32));
    const want = try a.alloc(f32, 4096 * ne);
    defer a.free(want);
    const got = try a.alloc(f32, 4096 * ne);
    defer a.free(got);
    for (widths) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        // the stock call's operand: the MoE block's input [1, S, hidden]
        const x3 = try g.hostArray(std.mem.sliceAsBytes(xbits[0 .. rows * hidden]), &.{ 1, ci(rows), ci(hidden) }, .bfloat16);
        const n = rows * ne;
        // Linear's matmul at every width, and at 2 to 8 rows also the exact verify's denseRows
        const forms: []const bool = if (rows >= 2 and rows <= 8) &.{ false, true } else &.{false};
        for (forms) |dense| {
            const ref = if (dense) try servedLogits(&g, s, x3, table, rows) else try g.matmul(x3, table);
            const out = if (dense) try servedLogits(&g, s, x3, rebuilt, rows) else try g.matmul(x3, rebuilt);
            _ = try g.hostF32(ref, want[0..n]);
            _ = try g.hostF32(out, got[0..n]);
            const d = differing(if (dense) "stock_dense" else "stock", rows, 0, got[0..n], want[0..n], ne);
            std.debug.print("TURBOEXL3_LING_ROUTER stock rows={d} {s} differing={d}/{d} {s}\n", .{ rows, if (dense) "denseRows" else "matmul", d, n, if (d == 0) "PASS" else "FAIL" });
            if (d != 0) failed = true;
        }
    }
    try testing.expect(!failed);
}

/// x's rows over the pack's weights: `row_scales`, then two tiny ones whose products and partial sums are f32
/// subnormals: 2^-120 (a normal bf16) and 2^-130 (a subnormal bf16)
const pack_scales = row_scales ++ [_]f32{ 0x1p-120, 0x1p-130 };

/// `buf` from `off` to its end, read from `fd` (a short read is an error)
fn preadAll(fd: c_int, buf: []u8, off: usize) !void {
    var n: usize = 0;
    while (n < buf.len) {
        const r = std.c.pread(fd, buf.ptr + n, buf.len - n, @intCast(off + n));
        if (r <= 0) return error.ShortRead;
        n += @intCast(r);
    }
}

/// A small file's bytes (the pack's index): `buf[0..n]`; the caller frees `buf`
const FileBytes = struct { buf: []u8, n: usize };

fn readFile(a: std.mem.Allocator, path: [*:0]const u8, max: usize) !FileBytes {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    const buf = try a.alloc(u8, max);
    errdefer a.free(buf);
    var n: usize = 0;
    while (n < max) {
        const r = std.c.pread(fd, buf.ptr + n, max - n, @intCast(n));
        if (r < 0) return error.ShortRead;
        if (r == 0) break;
        n += @intCast(r);
    }
    if (n == max) return error.FileTooBig;
    return .{ .buf = buf, .n = n };
}

/// Tensor `name`'s words from the safetensors shard at `path` into `out`, as stored (the loader's f16 [512, hidden]
/// row-major); error.PackTensor unless it is that
fn shardWords(a: std.mem.Allocator, path: [*:0]const u8, name: []const u8, out: []u16) !void {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    var hl_buf: [8]u8 = undefined;
    try preadAll(fd, &hl_buf, 0);
    const hlen: usize = @intCast(std.mem.readInt(u64, &hl_buf, .little));
    const hdr = try a.alloc(u8, hlen);
    defer a.free(hdr);
    try preadAll(fd, hdr, 8);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, hdr, .{});
    defer parsed.deinit();
    const t = (parsed.value.object.get(name) orelse return error.PackTensor).object;
    const dims = t.get("shape").?.array.items;
    if (!std.mem.eql(u8, t.get("dtype").?.string, "F16") or dims.len != 2) return error.PackTensor;
    if (dims[0].integer != @as(i64, @intCast(ne)) or dims[1].integer != @as(i64, @intCast(hidden))) return error.PackTensor;
    const offs = t.get("data_offsets").?.array.items;
    const o0: usize = @intCast(offs[0].integer);
    const o1: usize = @intCast(offs[1].integer);
    const bytes = std.mem.sliceAsBytes(out);
    if (o1 - o0 != bytes.len) return error.PackTensor;
    try preadAll(fd, bytes, 8 + hlen + o0);
}

test "turboexl3 gpu ling router pack weights: over every routed layer's router weight of a Ling-3.1-flash pack (TURBOEXL3_LING_PACK=<pack dir>: the MoE layers' and the NextN layer's, -0.0 words and subnormal words among them), read from its shard as the loader reads it, tableF32 rebuilds the loader's table word for word and routerLogits over the table and over the f16 weight gives the served logits at rows 1 to 8, x's rows down to f32 subnormal products (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const dir_c = std.c.getenv("TURBOEXL3_LING_PACK") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const dir = std.mem.span(dir_c);
    // the pack's index: each tensor's shard
    var ibuf: [4096]u8 = undefined;
    const index_path = try std.fmt.bufPrintSentinel(&ibuf, "{s}/model.safetensors.index.json", .{dir}, 0);
    const index = try readFile(a, index_path.ptr, 64 << 20);
    defer a.free(index.buf);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, index.buf[0..index.n], .{});
    defer parsed.deinit();
    const wmap = (parsed.value.object.get("weight_map") orelse return error.PackTensor).object;
    var prng = std.Random.DefaultPrng.init(0x11C0_9AC4);
    const rnd = prng.random();
    const words = try a.alloc(u16, ne * hidden);
    defer a.free(words);
    const xbits = try a.alloc(u16, 8 * hidden);
    defer a.free(xbits);
    const fbuf = try a.alloc(f32, 2 * 8 * ne);
    defer a.free(fbuf);
    const tw = try a.alloc(f32, ne * hidden);
    defer a.free(tw);
    const rw = try a.alloc(f32, ne * hidden);
    defer a.free(rw);
    var failed = false;
    var n_layers: usize = 0;
    var first_layer: usize = 0;
    var last_layer: usize = 0;
    var all_neg: usize = 0;
    var all_sub: usize = 0;
    var layer: usize = 0;
    while (layer < 256) : (layer += 1) {
        var nbuf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&nbuf, "model.layers.{d}.mlp.gate.weight", .{layer});
        const shard = (wmap.get(name) orelse continue).string;
        var sbuf: [4096]u8 = undefined;
        const shard_path = try std.fmt.bufPrintSentinel(&sbuf, "{s}/{s}", .{ dir, shard }, 0);
        try shardWords(a, shard_path.ptr, name, words);
        var neg: usize = 0;
        var sub: usize = 0;
        for (words) |v| {
            if (v == 0x8000) neg += 1;
            if ((v & 0x7C00) == 0 and (v & 0x03FF) != 0) sub += 1;
        }
        const mark = g.mark();
        defer g.resetTo(mark);
        // the weight as loaded, today's table over its view, and tableF32's rebuild of it
        const wl = try loadedWeight(&g, words);
        const table = try loaderTable(&g, wl);
        const rebuilt = try g.adopt(try router.tableF32(s, try loaderView(&g, wl)));
        try g.evalAll(&.{rebuilt});
        const cm = tableIsColumnMajor(table) and tableIsColumnMajor(rebuilt);
        _ = try g.hostF32(table, tw);
        _ = try g.hostF32(rebuilt, rw);
        const d_table = differing("pack_table", hidden, layer, rw, tw, ne);
        // what the fused router reads: the table transposed back (f32) or the loaded weight's view transposed back (f16)
        const w32 = try g.transposeAxes(table, &.{ 1, 0 });
        const w16 = try g.transposeAxes(try loaderView(&g, wl), &.{ 1, 0 });
        var d32: usize = 0;
        var d16: usize = 0;
        var checked: usize = 0;
        for (1..9) |rows| {
            const reps: usize = if (rows == 1) pack_scales.len else 2;
            for (0..reps) |rep| {
                const m2 = g.mark();
                defer g.resetTo(m2);
                for (0..rows) |r| {
                    const sc = pack_scales[(r + rep * rows) % pack_scales.len];
                    for (xbits[r * hidden ..][0..hidden]) |*v| v.* = support.bf16Bits(rnd.floatNorm(f32) * sc);
                }
                // the MoE block's input [1, S, hidden], and the [S, hidden] view the fused router takes
                const x3 = try g.hostArray(std.mem.sliceAsBytes(xbits[0 .. rows * hidden]), &.{ 1, ci(rows), ci(hidden) }, .bfloat16);
                const xf = try g.reshape(x3, &.{ ci(rows), ci(hidden) });
                const n = rows * ne;
                const want = fbuf[0..n];
                const got = fbuf[n .. 2 * n];
                _ = try g.hostF32(try servedLogits(&g, s, x3, table, rows), want);
                for ([_]T{ w32, w16 }, [_][]const u8{ "pack_f32", "pack_f16" }, [_]*usize{ &d32, &d16 }) |w, wname, count| {
                    const l = try router.routerLogits(s, xf, w, null);
                    _ = try g.hostF32(try g.adopt(l.logits), got);
                    count.* += differing(wname, rows, rep, got, want, ne);
                }
                checked += n;
            }
        }
        const ok = cm and d_table == 0 and d32 == 0 and d16 == 0;
        std.debug.print("TURBOEXL3_LING_ROUTER pack layer={d} neg_zero={d} subnormal={d} tableF32 differing={d}/{d} column_major={} logits differing over the f32 table={d}/{d} over the f16 weight={d}/{d} {s}\n", .{ layer, neg, sub, d_table, ne * hidden, cm, d32, checked, d16, checked, if (ok) "PASS" else "FAIL" });
        if (!ok) failed = true;
        if (n_layers == 0) first_layer = layer;
        last_layer = layer;
        n_layers += 1;
        all_neg += neg;
        all_sub += sub;
    }
    std.debug.print("TURBOEXL3_LING_ROUTER pack: {d} router weights (layers {d} to {d}), {d} -0.0 words, {d} subnormal words {s}\n", .{ n_layers, first_layer, last_layer, all_neg, all_sub, if (failed or n_layers == 0) "FAIL" else "PASS" });
    try testing.expect(n_layers > 0 and !failed);
}

test "turboexl3 gpu ling router refusals: rows past 8, a hidden size off the texts' tiles, a shape or dtype off (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(std.heap.page_allocator, s);
    defer g.deinit();
    const x = try g.zeros(&.{ 2, ci(hidden) }, .bfloat16);
    const w = try g.zeros(&.{ ci(ne), ci(hidden) }, .float32);
    const pw = try g.zeros(&.{ ci(ne), ci(hidden) }, .bfloat16);
    const b = try g.zeros(&.{ci(ne)}, .float32);
    const e = error.BadExl3Shape;
    const dt = error.BadExl3Dtype;
    try testing.expectError(e, router.routerLogits(s, try g.zeros(&.{ 9, ci(hidden) }, .bfloat16), w, null));
    try testing.expectError(e, router.routerLogits(s, try g.zeros(&.{ 2, 4000 }, .bfloat16), try g.zeros(&.{ ci(ne), 4000 }, .float32), null));
    try testing.expectError(e, router.routerLogits(s, try g.zeros(&.{ 1, 2, ci(hidden) }, .bfloat16), w, null));
    try testing.expectError(dt, router.routerLogits(s, try g.zeros(&.{ 2, ci(hidden) }, .float32), w, null));
    try testing.expectError(e, router.routerLogits(s, x, try g.zeros(&.{ ci(hidden), ci(ne) }, .float32), null));
    try testing.expectError(dt, router.routerLogits(s, x, try g.zeros(&.{ ci(ne), ci(hidden) }, .bfloat16), null));
    try testing.expectError(e, router.routerLogits(s, x, w, .{ .w = try g.zeros(&.{ ci(ne), 2048 }, .bfloat16), .bias = b }));
    try testing.expectError(dt, router.routerLogits(s, x, w, .{ .w = try g.zeros(&.{ ci(ne), ci(hidden) }, .float16), .bias = b }));
    try testing.expectError(dt, router.routerLogits(s, x, w, .{ .w = pw, .bias = try g.zeros(&.{ci(ne)}, .bfloat16) }));
    try testing.expectError(e, router.select(s, try g.zeros(&.{ 9, ci(ne) }, .float32), b, scale));
    try testing.expectError(e, router.select(s, try g.zeros(&.{ 2, 256 }, .float32), b, scale));
    try testing.expectError(dt, router.select(s, try g.zeros(&.{ 2, ci(ne) }, .bfloat16), b, scale));
    try testing.expectError(e, router.route(s, x, w, try g.zeros(&.{256}, .float32), scale, null));
    try testing.expectError(dt, router.tableF32(s, try g.zeros(&.{ ci(hidden), ci(ne) }, .float32)));
}
