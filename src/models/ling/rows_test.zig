//! Ling-3.1-flash's prompt call over gathered rows (`root.moeLingRows`, the outside plugin turboexl3_ling's
//! MLX_SERVE_LING_ROWS_PATH=1), held byte for byte to the prompt call it replaces: turboexl3_ling's ling_math.zig
//! `chunks()` and `gemm()`, statement for statement (`pluginChain`): the call's pairs sorted by slot (a counting sort),
//! cut into chunks of whole experts of at most 4096 pairs (the plugin's `prefill_rows`; an expert wider alone), each
//! chunk's rows taken from the layer's MoE input by their act rows and cast to bf16 where they are not, then the call: a
//! chunk of 16 pairs or fewer padded to 17 with copies of its first pair at score 0 (the pad rows dropped after), else
//! weight 1, the rows cast to f32 [1, n, hidden] and `moeLing` at top-1 over the bank's slot arrays, y f32. The rows
//! path (`pluginRows`, the plugin's chunk under the switch) casts the call's x to bf16 once where it is not and gives
//! `moeLingRows` the chunk's act rows as the row map, its slots and the same weights, y f32.
//! - CPU (always): the row-map prepare's text is the prepare's with its row line swapped for the row map's, nothing
//!   else, and the prepare's text is its pieces.
//! - GPU (TURBOEXL3_GPU=1, else skipped), at Ling's shapes (hidden 4096, expert width 1280, MUL1 at n 40 = K 2.5 and
//!   n 32 = K 2, the 2.60 and 2.10 packs' routed rates; TURBOEXL3_LING_N narrows), limits 0 and 10 (the clamp after the
//!   SiLU), on operands as the plugin serves them: the bank a streamer's slot arrays as its loader hands them over (each
//!   code the pack's I16 words viewed as U16, as the plugin's `t3View` views them; rin and rout f16), x the layer's MoE
//!   input [tokens, hidden] in bf16 (the streamer's dtype for every prompt call: experts.zig casts the call's
//!   activation to bf16) and in f32 (the plugin's chunks round a non-bf16 x to bf16 first), x ~ N(0, 8^2) (a part of
//!   the gates and ups pass the limit), the pairs a 6144-token prompt layer's top-8 over 160 slots (lognormal
//!   popularity) in the streamer's order (each token's pairs in turn, its act row the token's): every chunk of the call
//!   (its 4096-pair chunks and its tail) and a 5-pair call padded to 17, each chunk's kept rows bitwise; then
//!   `Prefill.moeWith` and `moeT128With` with the row map against `moe` and `moeT128` over the gathered rows on one
//!   engine (the served options, the 128-row tiles off and on), on the call's first chunk and the padded call.
//!   Lines: TURBOEXL3_LING_ROWS, TURBOEXL3_LING_ROWS_ENGINE.
//! - GPU, the routed chain's variants (`variants`: the 256-row tiles, the remainder rule's cut, gate|up's column blocks,
//!   the finish as the down GEMM's epilogue, the plain launches at bm 128 and 256; each a `prefill.Options` change, off
//!   by default), each held word for word to the served engine (`Prefill.moeT128With` at the served options) on the row
//!   map, at n 40 and 32, limits 0 and 10, over synthetic calls (never a prompt's routing): every bucket edge of an
//!   expert's rows (1-32, 33-64, 65-128, 129-256, 257-288, 289-512), the 4096-pair maximum call, many small experts,
//!   lognormal-skewed counts, one hot expert, a padded 5-pair call. Red checks: a flipped output word, a planted
//!   epilogue fault (one tile's f16 rounding dropped), a planted route fault (one 256-row tile on the wrong expert) and
//!   the GEMM's weight rounding changed (the deferred MUL1 map off) must each show differing words. Lines:
//!   TURBOEXL3_LING_ROWS_VARIANT, TURBOEXL3_LING_ROWS_RED.
//! - Timing (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; asserts nothing, guidance only): the rows path against the
//!   plugin's chain at the plugin's call shape (the first chunk of a 4096- and a 16384-token layer's call over 512
//!   experts at top-8, x bf16), min and median us per call over 8 rounds after 2 warm, each round's graph built and
//!   evaluated alone, and the two outputs' differing words. Lines: TURBOEXL3_LING_ROWS_TIME. Then each variant's arm on
//!   the same call through the row map against the served engine's, and a 16384-token layer at uniform popularity
//!   (about 256 rows an expert): TURBOEXL3_LING_ROWS_TIME_ARM.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ks = sdk.kernel_launch;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const prefill = @import("../../core/prefill.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const ds_test = @import("../deepseek_v41/deepseek_test.zig");

const testing = std.testing;

const hidden: u32 = 4096;
const inter: u32 = 1280;
const top_k: u32 = 8;
const sigma: f32 = 8;
/// The plugin's chunk bound (turboexl3_ling's ling_math.zig `prefill_rows`)
const prefill_rows: usize = 4096;
/// The fewest pairs a plugin prompt call takes (ling_math.zig `gemm_min_rows`): a smaller chunk is padded to it
const gemm_min_rows: usize = root.kernels.DECODE_ROWS_MAX + 1;
/// The byte test's slot bank and the prompt layer's tokens: about 307 pairs a slot, chunks of about 13 experts
const prompt_slots: u32 = 160;
const prompt_tokens: u32 = 6144;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// A comma list of numbers from the environment (at most buf.len of them), else `default`.
fn envList(name: [*:0]const u8, default: []const u32, buf: []u32) []const u32 {
    const v = std.c.getenv(name) orelse return default;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
    while (it.next()) |tok| {
        if (n == buf.len) break;
        buf[n] = std.fmt.parseInt(u32, std.mem.trim(u8, tok, " "), 10) catch continue;
        n += 1;
    }
    return if (n == 0) default else buf[0..n];
}

test "turboexl3 ling rows texts: the row-map prepare is the prepare with its row line swapped for the row map's, nothing else; the prepare's text is its pieces" {
    const pin = comptime support.pins(@embedFile("kernels/test/pins/rows_texts.metal"));
    try testing.expectEqualStrings(prefill.prep_1 ++ prefill.prep_row ++ prefill.prep_2, prefill.text_prep);
    try testing.expectEqualStrings(prefill.prep_1 ++ prefill.prep_rmap ++ prefill.prep_2, prefill.text_prep_rows);
    // the cut: prep_1 ends at the slot's line; each row line is one statement binding r, which prep_2 reads for x's row;
    // only the row map's line reads rmap
    try testing.expect(std.mem.endsWith(u8, prefill.prep_1, pin[0] ++ "\n"));
    for ([_][]const u8{ prefill.prep_row, prefill.prep_rmap }) |line| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, line, "\n"));
        try testing.expect(std.mem.indexOf(u8, line, pin[1]) != null);
    }
    try testing.expect(std.mem.indexOf(u8, prefill.text_prep, pin[2]) == null);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, prefill.text_prep_rows, pin[2]));
}

/// `x` viewed as `dt` (the same bytes; MLX's view, as the plugin's `viewU16`)
fn viewAs(g: *G, x: mlx.mlx_array, dt: mlx.mlx_dtype) !mlx.mlx_array {
    var v = mlx.mlx_array_new();
    mlx.check(mlx.mlx_view(&v, x, dt, g.s)) catch |e| {
        _ = mlx.mlx_array_free(v);
        return e;
    };
    return g.adopt(v);
}

/// A streamer's slot bank as its loader hands it over and the plugin hands it to turboexl3 (turboexl3_ling's `t3View`):
/// each code the pack's I16 words viewed as U16 (the same bits), rin as suh and rout as svh, f16.
fn loaderBank(g: *G, b: *const ref.Bank) !root.Bank {
    const E = ci(b.experts);
    const ht = ci(b.hidden / 16);
    const it = ci(b.inter / 16);
    const n = ci(b.rate.n);
    return .{
        .gate = .{ .trellis = try viewAs(g, try g.hostArray(std.mem.sliceAsBytes(b.gate_t), &.{ E, ht, it, n }, .int16), .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(b.gate_suh), &.{ E, ci(b.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.gate_svh), &.{ E, ci(b.inter) }, .float16) },
        .up = .{ .trellis = try viewAs(g, try g.hostArray(std.mem.sliceAsBytes(b.up_t), &.{ E, ht, it, n }, .int16), .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(b.up_suh), &.{ E, ci(b.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.up_svh), &.{ E, ci(b.inter) }, .float16) },
        .down = .{ .trellis = try viewAs(g, try g.hostArray(std.mem.sliceAsBytes(b.down_t), &.{ E, it, ht, n }, .int16), .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(b.down_suh), &.{ E, ci(b.inter) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.down_svh), &.{ E, ci(b.hidden) }, .float16) },
    };
}

/// A prompt layer's call as the streamer hands its pairs to the plugin: each token's top-8 distinct slots (Gumbel top-k
/// over lognormal popularity, sigma 1.5), the pairs in token order, each pair's act row its token.
fn drawPairs(a: std.mem.Allocator, rnd: std.Random, tokens: u32, n_slots: u32, act: []u32, slot: []u32) !void {
    return drawPairsAt(a, rnd, tokens, n_slots, 1.5, act, slot);
}

/// `drawPairs` at lognormal popularity sigma `skew` (0: uniform)
fn drawPairsAt(a: std.mem.Allocator, rnd: std.Random, tokens: u32, n_slots: u32, skew: f64, act: []u32, slot: []u32) !void {
    const logw = try a.alloc(f64, n_slots);
    defer a.free(logw);
    for (logw) |*w| w.* = skew * rnd.floatNorm(f64);
    const keys = try a.alloc(f64, n_slots);
    defer a.free(keys);
    for (0..tokens) |t| {
        for (keys, logw) |*kv, lw| kv.* = lw - @log(-@log(@max(rnd.float(f64), 1e-300)));
        for (0..top_k) |j| {
            var best: usize = 0;
            for (keys, 0..) |kv, e| {
                if (kv > keys[best]) best = e;
            }
            act[t * top_k + j] = @intCast(t);
            slot[t * top_k + j] = @intCast(best);
            keys[best] = -std.math.inf(f64);
        }
    }
}

/// The plugin's sort of a call's pairs by slot (ling_math.zig `chunks()`: a counting sort, stable): each sorted pair's act
/// row and slot.
fn sortPairs(a: std.mem.Allocator, act: []const u32, slot: []const u32, s_act: []u32, s_slot: []u32) !void {
    var max_slot: u32 = 0;
    for (slot) |sv| max_slot = @max(max_slot, sv);
    const starts = try a.alloc(u32, @as(usize, max_slot) + 2);
    defer a.free(starts);
    @memset(starts, 0);
    for (slot) |sv| starts[sv + 1] += 1;
    for (1..starts.len) |i| starts[i] += starts[i - 1];
    for (slot, act) |sv, av| {
        s_slot[starts[sv]] = sv;
        s_act[starts[sv]] = av;
        starts[sv] += 1;
    }
}

/// The end of the plugin's chunk from sorted pair `c0` (ling_math.zig `chunks()`): whole experts, at most
/// `prefill_rows` pairs, a first expert wider alone.
fn chunkEnd(s_slot: []const u32, c0: usize) usize {
    const n = s_slot.len;
    var c1 = c0;
    while (c1 < n) {
        var run_end = c1 + 1;
        while (run_end < n and s_slot[run_end] == s_slot[c1]) run_end += 1;
        if (c1 > c0 and run_end - c0 > prefill_rows) break;
        c1 = run_end;
    }
    return c1;
}

/// The distinct slots of a chunk (sorted, so its runs)
fn runsOf(s_slot: []const u32) usize {
    var r: usize = 0;
    for (s_slot, 0..) |sv, i| r += @intFromBool(i == 0 or sv != s_slot[i - 1]);
    return r;
}

/// `n` ones, f32, in `shape`
fn onesF32(g: *G, n: usize, shape: []const c_int) !mlx.mlx_array {
    const buf = try std.heap.page_allocator.alloc(f32, n);
    defer std.heap.page_allocator.free(buf);
    @memset(buf, 1.0);
    return g.hostArray(std.mem.sliceAsBytes(buf), shape, .float32);
}

/// turboexl3_ling's prompt chunk today, statement for statement (ling_math.zig `chunks()`'s take and cast, then
/// `gemm()`): the chunk's rows taken from x by its act rows, cast to bf16 where they are not; a chunk of 16 pairs or fewer
/// padded to 17 with copies of its first pair at score 0, else weight 1; the rows cast to f32 [1, n, hidden], `moeLing`
/// at top-1; y f32 [pairs, hidden] (a padded call's pad rows dropped).
fn pluginChain(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, act: []const u32, slot: []const u32, tb: root.Bank, lim: c_int) !mlx.mlx_array {
    const rows = act.len;
    const mc: c_int = @intCast(rows);
    const ai = try g.hostArray(std.mem.sliceAsBytes(act), &.{mc}, .uint32);
    const xr = try g.take(x, ai, 0);
    const xb = if (mlx.mlx_array_dtype(xr) == .bfloat16) xr else try g.astype(xr, .bfloat16);
    const sid = try g.hostArray(std.mem.sliceAsBytes(slot), &.{mc}, .uint32);
    const small = !root.kernels.usesPrefillArm(rows);
    const n_call: usize = if (small) gemm_min_rows else rows;
    const nc: c_int = @intCast(n_call);
    var xc = xb;
    var sc_ids = sid;
    var sc: mlx.mlx_array = undefined;
    if (small) {
        var idx: [gemm_min_rows]u32 = undefined;
        var w: [gemm_min_rows]f32 = undefined;
        for (&idx, &w, 0..) |*ix, *wt, i| {
            ix.* = if (i < rows) @intCast(i) else 0;
            wt.* = if (i < rows) 1.0 else 0.0;
        }
        const ia = try g.hostArray(std.mem.sliceAsBytes(idx[0..]), &.{nc}, .uint32);
        xc = try g.take(xb, ia, 0);
        sc_ids = try g.take(sid, ia, 0);
        sc = try g.hostArray(std.mem.sliceAsBytes(w[0..]), &.{ 1, nc, 1 }, .float32);
    } else sc = try onesF32(g, n_call, &.{ 1, nc, 1 });
    const x3 = try g.reshape(try g.astype(xc, .float32), &.{ 1, nc, ci(hidden) });
    const inds = try g.reshape(sc_ids, &.{ 1, nc, 1 });
    const y3 = try g.adopt(try root.moeLing(s, x3, tb, inds, sc, .mul1, lim));
    const y_all = try g.reshape(y3, &.{ nc, ci(hidden) });
    if (!small) return y_all;
    return g.slice(y_all, &.{ 0, 0 }, &.{ mc, ci(hidden) }, &.{ 1, 1 });
}

/// A rows-path call's operands (the plugin's chunk under the switch): the row map (the chunk's act rows), the slots and
/// the weights, a chunk of 16 pairs or fewer padded to 17 with its first pair at score 0.
const RowsOps = struct { rm: mlx.mlx_array, ids: mlx.mlx_array, sc: mlx.mlx_array, n_call: usize, small: bool };

fn rowsOps(g: *G, act: []const u32, slot: []const u32) !RowsOps {
    const rows = act.len;
    const small = !root.kernels.usesPrefillArm(rows);
    if (!small) {
        const mc: c_int = @intCast(rows);
        return .{
            .rm = try g.hostArray(std.mem.sliceAsBytes(act), &.{mc}, .uint32),
            .ids = try g.hostArray(std.mem.sliceAsBytes(slot), &.{mc}, .uint32),
            .sc = try onesF32(g, rows, &.{mc}),
            .n_call = rows,
            .small = false,
        };
    }
    var r: [gemm_min_rows]u32 = undefined;
    var sl: [gemm_min_rows]u32 = undefined;
    var w: [gemm_min_rows]f32 = undefined;
    for (&r, &sl, &w, 0..) |*rv, *sv, *wv, i| {
        const j = if (i < rows) i else 0;
        rv.* = act[j];
        sv.* = slot[j];
        wv.* = if (i < rows) 1.0 else 0.0;
    }
    const nc: c_int = @intCast(gemm_min_rows);
    return .{
        .rm = try g.hostArray(std.mem.sliceAsBytes(r[0..]), &.{nc}, .uint32),
        .ids = try g.hostArray(std.mem.sliceAsBytes(sl[0..]), &.{nc}, .uint32),
        .sc = try g.hostArray(std.mem.sliceAsBytes(w[0..]), &.{nc}, .float32),
        .n_call = gemm_min_rows,
        .small = true,
    };
}

/// The rows path (the plugin's chunk under MLX_SERVE_LING_ROWS_PATH=1): `xb` the call's x in bf16 (cast once where it is
/// not), `moeLingRows` over the chunk's row map, slots and weights (`rowsOps`), y f32 [pairs, hidden] (a padded call's pad
/// rows dropped).
fn pluginRows(g: *G, s: mlx.mlx_stream, xb: mlx.mlx_array, act: []const u32, slot: []const u32, tb: root.Bank, lim: c_int) !mlx.mlx_array {
    const o = try rowsOps(g, act, slot);
    const y = try g.adopt(try root.moeLingRows(s, xb, o.rm, tb, o.ids, o.sc, .mul1, lim, .float32));
    if (!o.small) return y;
    return g.slice(y, &.{ 0, 0 }, &.{ ci(act.len), ci(hidden) }, &.{ 1, 1 });
}

/// The words of `got` and `want` (both f32 [rows, hidden]) that differ, and `got`'s non-finite words
fn compare(a: std.mem.Allocator, g: *G, got: mlx.mlx_array, want: mlx.mlx_array) !struct { differing: usize, nonfinite: usize, of: usize } {
    const n = mlx.mlx_array_size(want);
    if (mlx.mlx_array_size(got) != n or mlx.mlx_array_dtype(got) != .float32 or mlx.mlx_array_dtype(want) != .float32) return error.SizeMismatch;
    const p = try a.alloc(f32, n);
    defer a.free(p);
    const q = try a.alloc(f32, n);
    defer a.free(q);
    _ = try g.hostF32(got, p);
    _ = try g.hostF32(want, q);
    var d: usize = 0;
    var nf: usize = 0;
    for (p, q) |u, v| {
        d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
        nf += @intFromBool(!std.math.isFinite(u));
    }
    return .{ .differing = d, .nonfinite = nf, .of = n };
}

/// The served prefill options (root's `pfOptions` with its clamps at rate n), the 128-row tiles as asked
fn servedOptions(n: u32, t128: bool) prefill.Options {
    const intr = n % 16 == 0;
    return .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = intr, .defer_affine = true, .ncb = 4, .sintr = intr, .froute = true, .alay = 2, .t128 = t128 };
}

/// A rows-path call's geometry (`root.moeSwigluRows`'s: top-1 over the bank's slots, Ling's clamp after the SiLU)
fn rowsGeo(n: u32, experts: u32, limit: c_int) prefill.Geometry {
    return .{ .hidden = hidden, .inter = inter, .topk = 1, .experts = experts, .rate = format.Rate.fromN(n).?, .dec = .mul1, .limit = @intCast(limit), .clamp = .silu_out };
}

/// A bank's slot arrays as the engine takes them
fn opsOf(b: root.Bank) prefill.Operands {
    return .{ .gate_t = b.gate.trellis, .gate_suh = b.gate.suh, .gate_svh = b.gate.svh, .up_t = b.up.trellis, .up_suh = b.up.suh, .up_svh = b.up.svh, .down_t = b.down.trellis, .down_suh = b.down.suh, .down_svh = b.down.svh };
}

/// One rows-path call on engine `pf` through the row map (`moeT128With` with the 128-row tiles, else `moeWith`), y f32
/// [o.n_call, hidden]
fn engineRows(pf: *prefill.Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, o: RowsOps, ops: prefill.Operands) !mlx.mlx_array {
    if (pf.opts.t128) return pf.moeT128With(s, x, o.ids, o.sc, ops, o.n_call, .float32, o.rm);
    return pf.moeWith(s, x, o.ids, o.sc, ops, o.n_call, .float32, o.rm);
}

/// A variant of the routed chain: its name and its options at rate n
const Variant = struct { name: []const u8, o: prefill.Options };

/// The served options (the 128-row tiles) with the 256-row tiles, the remainder rule's cut, gate|up's column blocks and the
/// fused finish set (`prefill.Options.t256`, `cut`, `gu_ncb`, `dfin`, `dfr`)
fn vT(n: u32, t256: bool, cut: u8, gu_ncb: u8, dfin: bool, dfr: u16) prefill.Options {
    var o = servedOptions(n, true);
    o.t256 = t256;
    o.cut = cut;
    o.gu_ncb = gu_ncb;
    o.dfin = dfin;
    o.dfr = dfr;
    return o;
}

/// The plain chain (no 128-row tiles: one launch a GEMM) at M tile rows `bm`, the register path for <= 32-row remainders
/// on or off (every tile on the shared-decode body), the fused finish set
fn vP(n: u32, bm: u32, mixed: bool, dfin: bool, dfr: u16) prefill.Options {
    var o = servedOptions(n, false);
    o.bm = bm;
    if (!mixed) {
        o.small = .off;
        o.sintr = false;
    }
    o.dfin = dfin;
    o.dfr = dfr;
    return o;
}

const n_variants = 22;

/// The routed chain's variants at rate n (the byte test's and the timing's arms): each one launch-shape change off the
/// served options or one plain launch, every one `moeLingRows`'s bytes by construction (the same per-simdgroup K loop on
/// every body, the finish's own statements on the store's f16 words)
fn variants(n: u32) [n_variants]Variant {
    return .{
        .{ .name = "t256", .o = vT(n, true, 64, 1, false, 64) },
        .{ .name = "t256.cut0", .o = vT(n, true, 0, 1, false, 64) },
        .{ .name = "t256.cut32", .o = vT(n, true, 32, 1, false, 64) },
        .{ .name = "t128.cut0", .o = vT(n, false, 0, 1, false, 64) },
        .{ .name = "t128.cut32", .o = vT(n, false, 32, 1, false, 64) },
        .{ .name = "t128.cut96", .o = vT(n, false, 96, 1, false, 64) },
        .{ .name = "t128.cut127", .o = vT(n, false, 127, 1, false, 64) },
        .{ .name = "gu2", .o = vT(n, false, 64, 2, false, 64) },
        .{ .name = "gu4", .o = vT(n, false, 64, 4, false, 64) },
        .{ .name = "t256.gu2", .o = vT(n, true, 64, 2, false, 64) },
        .{ .name = "t256.gu4", .o = vT(n, true, 64, 4, false, 64) },
        .{ .name = "dfin", .o = vT(n, false, 64, 1, true, 64) },
        .{ .name = "dfin.r32", .o = vT(n, false, 64, 1, true, 32) },
        .{ .name = "t256.dfin", .o = vT(n, true, 64, 1, true, 64) },
        .{ .name = "t256.dfin.r32", .o = vT(n, true, 64, 1, true, 32) },
        .{ .name = "t256.gu2.dfin", .o = vT(n, true, 64, 2, true, 64) },
        .{ .name = "m128", .o = vP(n, 128, true, false, 64) },
        .{ .name = "m256", .o = vP(n, 256, true, false, 64) },
        .{ .name = "p256", .o = vP(n, 256, false, false, 64) },
        .{ .name = "p256.dfin", .o = vP(n, 256, false, true, 64) },
        .{ .name = "p256.dfin.r32", .o = vP(n, 256, false, true, 32) },
        .{ .name = "p128.dfin", .o = vP(n, 128, false, true, 64) },
    };
}

/// A synthetic call (never a prompt's routing): `counts[i]` rows for expert i, each on its own slot (a random choice of
/// distinct slots), each row a random act row of the layer's `tokens`; the pairs sorted by slot as the plugin sorts them.
/// The caller frees `act` and `slot`.
fn synthCall(a: std.mem.Allocator, rnd: std.Random, counts: []const u32, n_slots: u32, tokens: u32) !struct { act: []u32, slot: []u32 } {
    std.debug.assert(counts.len <= n_slots);
    const perm = try a.alloc(u32, n_slots);
    defer a.free(perm);
    for (perm, 0..) |*p, i| p.* = @intCast(i);
    rnd.shuffle(u32, perm);
    var np: usize = 0;
    for (counts) |c| np += c;
    const act = try a.alloc(u32, np);
    errdefer a.free(act);
    const slot = try a.alloc(u32, np);
    errdefer a.free(slot);
    var k: usize = 0;
    for (counts, 0..) |c, i| for (0..c) |_| {
        act[k] = rnd.uintLessThan(u32, tokens);
        slot[k] = perm[i];
        k += 1;
    };
    const s_act = try a.alloc(u32, np);
    defer a.free(s_act);
    const s_slot = try a.alloc(u32, np);
    defer a.free(s_slot);
    try sortPairs(a, act, slot, s_act, s_slot);
    @memcpy(act, s_act);
    @memcpy(slot, s_slot);
    return .{ .act = act, .slot = slot };
}

/// The byte test's calls: an expert's rows at every tile edge, the plugin's 4096-pair maximum, many small experts,
/// lognormal-skewed counts, one hot expert, a padded 5-pair call (each count list at most 4096 pairs and 160 experts)
const SynthKind = enum { buckets, max4096, small, skewed, hot, padded };

fn synthCounts(kind: SynthKind, rnd: std.Random, buf: []u32) []const u32 {
    switch (kind) {
        // each bucket's edges: 1-32, 33-64, 65-128, 129-256, 257-288, 289-512 (3904 pairs, 21 experts)
        .buckets => {
            const edges = [_]u32{ 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 255, 256, 257, 287, 288, 289, 383, 511, 512 };
            @memcpy(buf[0..edges.len], &edges);
            return buf[0..edges.len];
        },
        // the plugin's largest call: 4096 pairs over 14 experts (8 of 293 rows, 6 of 292)
        .max4096 => {
            for (buf[0..14], 0..) |*c, i| c.* = if (i < 8) 293 else 292;
            return buf[0..14];
        },
        // many small experts: 120 of 1 to 32 rows
        .small => {
            for (buf[0..120]) |*c| c.* = 1 + rnd.uintLessThan(u32, 32);
            return buf[0..120];
        },
        // lognormal counts (median ~90, sigma 1.2), at most 1500 an expert, until 4096 pairs
        .skewed => {
            var total: u32 = 0;
            var k: usize = 0;
            while (k < 60) : (k += 1) {
                const c: u32 = @intFromFloat(@min(1500.0, @max(1.0, @exp(4.5 + 1.2 * rnd.floatNorm(f64)))));
                if (total + c > 4096) break;
                buf[k] = c;
                total += c;
            }
            return buf[0..k];
        },
        // one hot expert of 2600 rows beside 30 of 1 to 48
        .hot => {
            buf[0] = 2600;
            for (buf[1..31]) |*c| c.* = 1 + rnd.uintLessThan(u32, 48);
            return buf[0..31];
        },
        // a 5-pair call: the plugin pads it to 17 at score 0
        .padded => {
            @memset(buf[0..5], 1);
            return buf[0..5];
        },
    }
}

/// `got`'s words with word `at`'s lowest bit flipped, as a new array (a red check's planted output fault)
fn flipOne(a: std.mem.Allocator, g: *G, got: mlx.mlx_array, at: usize) !mlx.mlx_array {
    const n = mlx.mlx_array_size(got);
    const buf = try a.alloc(f32, n);
    defer a.free(buf);
    _ = try g.hostF32(got, buf);
    buf[at] = @bitCast(@as(u32, @bitCast(buf[at])) ^ 1);
    return g.hostArray(std.mem.sliceAsBytes(buf), mlx.getShape(got), .float32);
}

test "turboexl3 gpu ling rows: moeLingRows is the plugin's prompt chunk (take, bf16, f32, moeLing at top-1) byte for byte on every chunk of a prompt layer's call and a padded call, at n 40 and 32, limits 0 and 10, x bf16 and f32, on the loader's views (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_LING_N", &.{ 40, 32 }, &nbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x11A6_0E5 ^ @as(u64, n));
        const rnd = prng.random();
        var hb = try ref.synthBank(a, rnd, prompt_slots, hidden, inter, rate);
        defer hb.deinit(a);
        const tb = try loaderBank(&g, &hb);
        // the layer's MoE input: f32 values (not bf16-exact, so the plugin's bf16 rounding of an f32 x counts), and the
        // same values in bf16 (the streamer's dtype)
        const xh = try a.alloc(f32, prompt_tokens * hidden);
        defer a.free(xh);
        for (xh) |*v| v.* = sigma * rnd.floatNorm(f32);
        const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(prompt_tokens), ci(hidden) }, .float32);
        const xbf = try g.astype(x32, .bfloat16);
        // the call's pairs as the streamer passes them, then as the plugin sorts them
        const np = prompt_tokens * top_k;
        const act = try a.alloc(u32, np);
        defer a.free(act);
        const slot = try a.alloc(u32, np);
        defer a.free(slot);
        try drawPairs(a, rnd, prompt_tokens, prompt_slots, act, slot);
        const s_act = try a.alloc(u32, np);
        defer a.free(s_act);
        const s_slot = try a.alloc(u32, np);
        defer a.free(s_slot);
        try sortPairs(a, act, slot, s_act, s_slot);
        // the chunks the plugin issues for the call, then a 5-pair call (its first 5 pairs, sorted as the plugin sorts a
        // call of its own): padded to 17
        var bounds: [64][2]usize = undefined;
        var nb: usize = 0;
        var c0: usize = 0;
        while (c0 < np) {
            const c1 = chunkEnd(s_slot, c0);
            if (nb == bounds.len) return error.TooManyChunks;
            bounds[nb] = .{ c0, c1 };
            nb += 1;
            c0 = c1;
        }
        var p_act: [5]u32 = undefined;
        var p_slot: [5]u32 = undefined;
        try sortPairs(a, act[0..5], slot[0..5], &p_act, &p_slot);
        for ([_]c_int{ 0, 10 }) |lim| for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |xd| {
            const x = if (xd == .bfloat16) xbf else x32;
            // the rows path's x: the call's x cast to bf16 once where it is not
            const xb = if (mlx.mlx_array_dtype(x) == .bfloat16) x else try g.astype(x, .bfloat16);
            for (0..nb + 1) |ck| {
                const m2 = g.mark();
                defer g.resetTo(m2);
                const ca: []const u32 = if (ck < nb) s_act[bounds[ck][0]..bounds[ck][1]] else &p_act;
                const cs: []const u32 = if (ck < nb) s_slot[bounds[ck][0]..bounds[ck][1]] else &p_slot;
                const want = try pluginChain(&g, s, x, ca, cs, tb, lim);
                const got = try pluginRows(&g, s, xb, ca, cs, tb, lim);
                try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
                const c = try compare(a, &g, got, want);
                const ok = c.differing == 0 and c.nonfinite == 0;
                std.debug.print("TURBOEXL3_LING_ROWS n={d} limit={d} x={s} chunk={d}/{d} pairs={d} experts={d} padded={} differing={d}/{d} nonfinite={d} {s}\n", .{ n, lim, @tagName(xd), ck + 1, nb + 1, ca.len, runsOf(cs), !root.kernels.usesPrefillArm(ca.len), c.differing, c.of, c.nonfinite, if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            }
        };
        // the row map on one engine: moeWith / moeT128With against moe / moeT128 over the gathered rows (the served
        // options at this rate, the 128-row tiles off and on), the call's first chunk and the padded call, x bf16
        for ([_]c_int{ 0, 10 }) |lim| for ([_]bool{ false, true }) |t128| {
            const geo: prefill.Geometry = .{ .hidden = hidden, .inter = inter, .topk = 1, .experts = prompt_slots, .rate = rate, .dec = .mul1, .limit = @intCast(lim), .clamp = .silu_out };
            var pf = try prefill.Prefill.init(geo, servedOptions(n, t128));
            defer pf.deinit();
            const ops: prefill.Operands = .{
                .gate_t = tb.gate.trellis,
                .gate_suh = tb.gate.suh,
                .gate_svh = tb.gate.svh,
                .up_t = tb.up.trellis,
                .up_suh = tb.up.suh,
                .up_svh = tb.up.svh,
                .down_t = tb.down.trellis,
                .down_suh = tb.down.suh,
                .down_svh = tb.down.svh,
            };
            for (0..2) |which| {
                const m2 = g.mark();
                defer g.resetTo(m2);
                const ca: []const u32 = if (which == 0) s_act[bounds[0][0]..bounds[0][1]] else &p_act;
                const cs: []const u32 = if (which == 0) s_slot[bounds[0][0]..bounds[0][1]] else &p_slot;
                const o = try rowsOps(&g, ca, cs);
                const xg = try g.take(xbf, o.rm, 0);
                const want = try g.adopt(if (t128) try pf.moeT128(s, xg, o.ids, o.sc, ops, o.n_call, .float32) else try pf.moe(s, xg, o.ids, o.sc, ops, o.n_call, .float32));
                const got = try g.adopt(if (t128) try pf.moeT128With(s, xbf, o.ids, o.sc, ops, o.n_call, .float32, o.rm) else try pf.moeWith(s, xbf, o.ids, o.sc, ops, o.n_call, .float32, o.rm));
                const c = try compare(a, &g, got, want);
                const ok = c.differing == 0 and c.nonfinite == 0;
                std.debug.print("TURBOEXL3_LING_ROWS_ENGINE n={d} limit={d} t128={} pairs={d} row_slots={d} differing={d}/{d} nonfinite={d} {s}\n", .{ n, lim, t128, ca.len, o.n_call, c.differing, c.of, c.nonfinite, if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            }
        };
    }
    try testing.expect(!failed);
}

test "turboexl3 gpu ling rows variants: the routed chain's variants (256-row tiles, the remainder cut, gate-up column blocks, the fused finish, the plain launches) are the served engine's words on synthetic calls at every row bucket, n 40 and 32, limits 0 and 10; planted faults are caught (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_LING_N", &.{ 40, 32 }, &nbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x11A6_7A2 ^ @as(u64, n));
        const rnd = prng.random();
        var hb = try ref.synthBank(a, rnd, prompt_slots, hidden, inter, rate);
        defer hb.deinit(a);
        const tb = try loaderBank(&g, &hb);
        const ops = opsOf(tb);
        // the layer's MoE input in bf16 (the streamer's dtype), x ~ N(0, 8^2): a part of the gates and ups pass the limit
        const xh = try a.alloc(f32, prompt_tokens * hidden);
        defer a.free(xh);
        for (xh) |*v| v.* = sigma * rnd.floatNorm(f32);
        const xb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(prompt_tokens), ci(hidden) }, .float32), .bfloat16);
        const vs = variants(n);
        for ([_]c_int{ 0, 10 }) |lim| {
            const geo = rowsGeo(n, prompt_slots, lim);
            var served = try prefill.Prefill.init(geo, servedOptions(n, true));
            defer served.deinit();
            var engines: [n_variants]prefill.Prefill = undefined;
            var made: usize = 0;
            defer for (engines[0..made]) |*e| e.deinit();
            for (vs, &engines) |v, *e| {
                e.* = try prefill.Prefill.init(geo, v.o);
                made += 1;
            }
            var cbuf: [160]u32 = undefined;
            for (std.enums.values(SynthKind)) |kind| {
                const m2 = g.mark();
                defer g.resetTo(m2);
                const counts = synthCounts(kind, rnd, &cbuf);
                const call = try synthCall(a, rnd, counts, prompt_slots, prompt_tokens);
                defer a.free(call.act);
                defer a.free(call.slot);
                const o = try rowsOps(&g, call.act, call.slot);
                const want = try g.adopt(try engineRows(&served, s, xb, o, ops));
                for (vs, &engines) |v, *e| {
                    // fresh buffers for the variant's arrays: no word it fails to write can hold a served value
                    _ = mlx.mlx_clear_cache();
                    const m3 = g.mark();
                    defer g.resetTo(m3);
                    const got = try g.adopt(try engineRows(e, s, xb, o, ops));
                    const c = try compare(a, &g, got, want);
                    const ok = c.differing == 0 and c.nonfinite == 0;
                    std.debug.print("TURBOEXL3_LING_ROWS_VARIANT n={d} limit={d} call={s} pairs={d} experts={d} variant={s} differing={d}/{d} nonfinite={d} {s}\n", .{ n, lim, @tagName(kind), call.act.len, counts.len, v.name, c.differing, c.of, c.nonfinite, if (ok) "PASS" else "FAIL" });
                    if (!ok) failed = true;
                }
            }
        }
        // the red checks (limit 0, the bucket call): each planted fault must show differing words, or the byte lines above
        // could not have caught it
        {
            const m2 = g.mark();
            defer g.resetTo(m2);
            var cbuf: [160]u32 = undefined;
            const counts = synthCounts(.buckets, rnd, &cbuf);
            const call = try synthCall(a, rnd, counts, prompt_slots, prompt_tokens);
            defer a.free(call.act);
            defer a.free(call.slot);
            const o = try rowsOps(&g, call.act, call.slot);
            const geo = rowsGeo(n, prompt_slots, 0);
            var served = try prefill.Prefill.init(geo, servedOptions(n, true));
            defer served.deinit();
            const want = try g.adopt(try engineRows(&served, s, xb, o, ops));
            // 1. one output word flipped (the comparison counts exactly it)
            {
                var pf = try prefill.Prefill.init(geo, vT(n, true, 64, 1, true, 64));
                defer pf.deinit();
                const got = try g.adopt(try engineRows(&pf, s, xb, o, ops));
                const c = try compare(a, &g, try flipOne(a, &g, got, (mlx.mlx_array_size(got) / 2) | 1), want);
                const caught = c.differing == 1;
                std.debug.print("TURBOEXL3_LING_ROWS_RED n={d} fault=flipped_word differing={d}/{d} caught={}\n", .{ n, c.differing, c.of, caught });
                if (!caught) failed = true;
            }
            // 2. the fused finish with tile 0's f16 rounding dropped (a planted epilogue fault in the kernel)
            {
                var pf = try prefill.Prefill.init(geo, vT(n, false, 64, 1, true, 64));
                defer pf.deinit();
                const fault = ks.newKernel("turboexl3_test_pf_dfin_fault", &prefill.dfin_in, &prefill.dfin_out, comptime prefill.dfinText(@embedFile("kernels/test/fault/dfin_4.metal")), prefill.header_nax_dfin, true) orelse return error.KernelCreateFailed;
                defer root.decode.freeKernel(fault);
                const kept = pf.dfin_k;
                pf.dfin_k = fault;
                defer pf.dfin_k = kept;
                _ = mlx.mlx_clear_cache();
                const got = try g.adopt(try engineRows(&pf, s, xb, o, ops));
                const c = try compare(a, &g, got, want);
                const caught = c.differing > 0;
                std.debug.print("TURBOEXL3_LING_ROWS_RED n={d} fault=dfin_tile0_unrounded differing={d}/{d} caught={}\n", .{ n, c.differing, c.of, caught });
                if (!caught) failed = true;
            }
            // 3. the 256-row tiles' route with its first tile on the next expert (a planted route fault in the kernel)
            {
                var pf = try prefill.Prefill.init(geo, vT(n, true, 64, 1, false, 64));
                defer pf.deinit();
                const fault = ks.newKernel("turboexl3_test_pf_t256_route_fault", &prefill.route_t128_in, &prefill.route_t128_out, comptime prefill.froute256With(@embedFile("kernels/test/fault/t256_store.metal")), prefill.header_small, true) orelse return error.KernelCreateFailed;
                defer root.decode.freeKernel(fault);
                const kept = pf.t128_k.?.route;
                pf.t128_k.?.route = fault;
                defer pf.t128_k.?.route = kept;
                _ = mlx.mlx_clear_cache();
                const got = try g.adopt(try engineRows(&pf, s, xb, o, ops));
                const c = try compare(a, &g, got, want);
                const caught = c.differing > 0;
                std.debug.print("TURBOEXL3_LING_ROWS_RED n={d} fault=t256_tile0_wrong_expert differing={d}/{d} caught={}\n", .{ n, c.differing, c.of, caught });
                if (!caught) failed = true;
            }
            // 4. the GEMM's weights rounded to f16 before the products (the deferred MUL1 map off): a rounding change inside
            // the K loop, within tolerance of the served words, never equal to them
            {
                var off = servedOptions(n, true);
                off.defer_affine = false;
                var pf = try prefill.Prefill.init(geo, off);
                defer pf.deinit();
                _ = mlx.mlx_clear_cache();
                const got = try g.adopt(try engineRows(&pf, s, xb, o, ops));
                const c = try compare(a, &g, got, want);
                const caught = c.differing > 0;
                std.debug.print("TURBOEXL3_LING_ROWS_RED n={d} fault=weights_rounded_first differing={d}/{d} caught={}\n", .{ n, c.differing, c.of, caught });
                if (!caught) failed = true;
            }
        }
    }
    try testing.expect(!failed);
}

/// A timed call's operands: the call's x (bf16), its sorted pairs' act rows and slots, the bank
const TimeCtx = struct { s: mlx.mlx_stream, x: mlx.mlx_array, act: []const u32, slot: []const u32, bank: root.Bank };

/// A timed engine call's operands: the engine, the call's x and row-map operands, the bank's
const ArmCtx = struct { s: mlx.mlx_stream, pf: *prefill.Prefill, x: mlx.mlx_array, o: RowsOps, ops: prefill.Operands };

/// `engineRows` timed as `timeCall` times: per round its graph built and evaluated alone, 2 warm-up rounds, then the min
/// and median us over 8
fn timeArm(g: *G, c: ArmCtx) ![2]f64 {
    var t: [8]f64 = undefined;
    for (0..t.len + 2) |round| {
        const m = g.mark();
        defer g.resetTo(m);
        const t0 = ds_test.nowNs();
        const y = try g.adopt(try engineRows(c.pf, c.s, c.x, c.o, c.ops));
        try g.evalAll(&.{y});
        const dt = ds_test.nowNs() - t0;
        if (round >= 2) t[round - 2] = @as(f64, @floatFromInt(dt)) / 1000.0;
    }
    std.mem.sort(f64, &t, {}, std.sort.asc(f64));
    return .{ t[0], t[t.len / 2] };
}

/// `f(g, ctx)` timed on the device: per round its graph built and evaluated alone (the round's arrays freed after), 2
/// warm-up rounds, then the min and median us over 8
fn timeCall(g: *G, ctx: TimeCtx, comptime f: fn (*G, TimeCtx) anyerror!mlx.mlx_array) ![2]f64 {
    var t: [8]f64 = undefined;
    for (0..t.len + 2) |round| {
        const m = g.mark();
        defer g.resetTo(m);
        const t0 = ds_test.nowNs();
        const y = try f(g, ctx);
        try g.evalAll(&.{y});
        const dt = ds_test.nowNs() - t0;
        if (round >= 2) t[round - 2] = @as(f64, @floatFromInt(dt)) / 1000.0;
    }
    std.mem.sort(f64, &t, {}, std.sort.asc(f64));
    return .{ t[0], t[t.len / 2] };
}

// Timing (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; it asserts nothing): guidance for the serving plan, never a speed
// verdict (the standard test through the outside plugin decides speed). The plugin's prompt call at its call shape: a
// 4096- and a 16384-token layer's top-8 over 512 experts (random trellises at the packs' rates, `ds_test.randProj`), the
// pairs sorted by slot as the plugin sorts them, the first chunk the plugin cuts (whole experts, at most 4096 pairs; about
// 64 and about 16 experts), x bf16 (the streamer's dtype). The chain (take, f32, moeLing) against the rows path
// (moeLingRows), each round's graph built and evaluated alone, then both outputs' differing words.
test "turboexl3 gpu ling timing: the prompt call over gathered rows (moeLingRows) against the plugin's chain at its call shape, and their bytes (TURBOEXL3_LING_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_LING_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x11A6_71E);
    const rnd = prng.random();
    const experts: u32 = 512;
    const expert_flops = 2.0 * 3.0 * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter));
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_LING_N", &.{ 40, 32 }, &nbuf);
    for (rates) |n| {
        const mark = g.mark();
        defer g.resetTo(mark);
        defer _ = mlx.mlx_clear_cache();
        const bank: root.Bank = .{
            .gate = try ds_test.randProj(a, &g, rnd, experts, hidden, inter, n),
            .up = try ds_test.randProj(a, &g, rnd, experts, hidden, inter, n),
            .down = try ds_test.randProj(a, &g, rnd, experts, inter, hidden, n),
        };
        for ([_]u32{ 4096, 16384 }) |tokens| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const np = tokens * top_k;
            const act = try a.alloc(u32, np);
            defer a.free(act);
            const slot = try a.alloc(u32, np);
            defer a.free(slot);
            try drawPairs(a, rnd, tokens, experts, act, slot);
            const s_act = try a.alloc(u32, np);
            defer a.free(s_act);
            const s_slot = try a.alloc(u32, np);
            defer a.free(s_slot);
            try sortPairs(a, act, slot, s_act, s_slot);
            const c1 = chunkEnd(s_slot, 0);
            const xb = try ds_test.randBf16(a, &g, rnd, &.{ ci(tokens), ci(hidden) });
            const ctx: TimeCtx = .{ .s = s, .x = xb, .act = s_act[0..c1], .slot = s_slot[0..c1], .bank = bank };
            const chain = try timeCall(&g, ctx, struct {
                fn f(gg: *G, cc: TimeCtx) anyerror!mlx.mlx_array {
                    return pluginChain(gg, cc.s, cc.x, cc.act, cc.slot, cc.bank, 0);
                }
            }.f);
            const rows = try timeCall(&g, ctx, struct {
                fn f(gg: *G, cc: TimeCtx) anyerror!mlx.mlx_array {
                    return pluginRows(gg, cc.s, cc.x, cc.act, cc.slot, cc.bank, 0);
                }
            }.f);
            const bm = g.mark();
            defer g.resetTo(bm);
            const c = try compare(a, &g, try pluginRows(&g, s, xb, ctx.act, ctx.slot, bank, 0), try pluginChain(&g, s, xb, ctx.act, ctx.slot, bank, 0));
            const flops = @as(f64, @floatFromInt(c1)) * expert_flops;
            std.debug.print("TURBOEXL3_LING_ROWS_TIME n={d} of_tokens={d} pairs={d} experts={d} chain_min_us={d:.1} chain_med_us={d:.1} rows_min_us={d:.1} rows_med_us={d:.1} chain_tflops={d:.2} rows_tflops={d:.2} differing={d}/{d}\n", .{ n, tokens, c1, runsOf(s_slot[0..c1]), chain[0], chain[1], rows[0], rows[1], flops / (chain[0] * 1e6), flops / (rows[0] * 1e6), c.differing, c.of });
            try timeVariants(a, &g, s, n, experts, "lognormal", tokens, xb, ctx.act, ctx.slot, bank);
        }
        // a 16384-token layer at uniform popularity (about 256 rows an expert), its own draws (the lines above keep theirs)
        {
            const um = g.mark();
            defer g.resetTo(um);
            var uprng = std.Random.DefaultPrng.init(0x11A6_0F1 ^ @as(u64, n));
            const urnd = uprng.random();
            const tokens: u32 = 16384;
            const np = tokens * top_k;
            const act = try a.alloc(u32, np);
            defer a.free(act);
            const slot = try a.alloc(u32, np);
            defer a.free(slot);
            try drawPairsAt(a, urnd, tokens, experts, 0, act, slot);
            const s_act = try a.alloc(u32, np);
            defer a.free(s_act);
            const s_slot = try a.alloc(u32, np);
            defer a.free(s_slot);
            try sortPairs(a, act, slot, s_act, s_slot);
            const c1 = chunkEnd(s_slot, 0);
            const xb = try ds_test.randBf16(a, &g, urnd, &.{ ci(tokens), ci(hidden) });
            try timeVariants(a, &g, s, n, experts, "uniform", tokens, xb, s_act[0..c1], s_slot[0..c1], bank);
        }
    }
}

/// The variants' arms on one call (guidance only): the served engine (the 128-row tiles at the served options) and each
/// variant's (`variants`), each through the row map on its own engine, timed by `timeArm`, with the arm's differing words
/// against the served engine's y. Lines: TURBOEXL3_LING_ROWS_TIME_ARM.
fn timeVariants(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, n: u32, experts: u32, routing: []const u8, tokens: u32, xb: mlx.mlx_array, act: []const u32, slot: []const u32, bank: root.Bank) !void {
    const m = g.mark();
    defer g.resetTo(m);
    const geo = rowsGeo(n, experts, 0);
    const ops = opsOf(bank);
    const o = try rowsOps(g, act, slot);
    const flops = @as(f64, @floatFromInt(act.len)) * 2.0 * 3.0 * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter));
    var served = try prefill.Prefill.init(geo, servedOptions(n, true));
    defer served.deinit();
    const want = try g.adopt(try engineRows(&served, s, xb, o, ops));
    const base = try timeArm(g, .{ .s = s, .pf = &served, .x = xb, .o = o, .ops = ops });
    std.debug.print("TURBOEXL3_LING_ROWS_TIME_ARM n={d} routing={s} of_tokens={d} pairs={d} experts={d} arm=served min_us={d:.1} med_us={d:.1} tflops={d:.2} vs_served_min=1.000 differing=0/0\n", .{ n, routing, tokens, act.len, runsOf(slot), base[0], base[1], flops / (base[0] * 1e6) });
    for (variants(n)) |v| {
        const vm = g.mark();
        defer g.resetTo(vm);
        var pf = try prefill.Prefill.init(geo, v.o);
        defer pf.deinit();
        const t = try timeArm(g, .{ .s = s, .pf = &pf, .x = xb, .o = o, .ops = ops });
        const c = try compare(a, g, try g.adopt(try engineRows(&pf, s, xb, o, ops)), want);
        std.debug.print("TURBOEXL3_LING_ROWS_TIME_ARM n={d} routing={s} of_tokens={d} pairs={d} experts={d} arm={s} min_us={d:.1} med_us={d:.1} tflops={d:.2} vs_served_min={d:.3} differing={d}/{d}\n", .{ n, routing, tokens, act.len, runsOf(slot), v.name, t[0], t[1], flops / (t[0] * 1e6), t[0] / base[0], c.differing, c.of });
    }
}
