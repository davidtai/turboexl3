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
//! - GPU, the routed chain's variants (`variants`: the finish as the down GEMM's epilogue over the 128-row tiles' two
//!   tables at the served options, also at 128-deep, two-block and MMA-first stages; each a `prefill.Options` change,
//!   off by default), each held word for word to the served engine
//!   (`Prefill.moeT128With` at the served options) on the row map, at n 40 and 32, limits 0 and 10, over synthetic
//!   calls (never a prompt's routing): every bucket edge of an expert's rows (1-32, 33-64, 65-128, 129-256, 257-288,
//!   289-512), the 4096-pair maximum call, many small experts, lognormal-skewed counts, one hot expert, a padded 5-pair
//!   call, experts on both sides of 64, 80 and 96 rows (63 to 65, 79 to 81, 95 to 97), a call drawn from lognormal
//!   popularity. Red checks: a flipped output word, a planted epilogue fault (one tile's f16 rounding dropped), the
//!   GEMM's weight rounding changed (the deferred MUL1 map off) and two planted order faults (one tile's H128 stages
//!   reordered, one tile's K stages in reverse order: the same values in exact arithmetic) must each show differing
//!   words, the order faults' all finite and within `order_tol` (a reordering, not garbage). Lines:
//!   TURBOEXL3_LING_ROWS_VARIANT, TURBOEXL3_LING_ROWS_RED.
//! - GPU, PF_DFIN's whole-call switch (`prefill.Call`; the outside plugin's MLX_SERVE_LING_PF_DFIN_CALL_MIN): a call
//!   that declines the fused finish runs the served ending on an engine built with it. Both endings give the served
//!   words, so each call's line also names the ending it ran, from the engine's counts (`prefill.Steps`, `endRan`). On
//!   the fused finish's engine (every synthetic call and every rung of the ladder), both choices, each the served
//!   engine's words and its ending the choice's; then the plugin's rule (`callTakes`: a wide call of R rows at top-8
//!   over 512 experts takes the fused finish at R * 8 >= 80 * 512, at every R with call_min 0) at R 1,120, 5,119, 5,120
//!   and 16,384, each on a chunk at its call's mean rows an expert, its ending held to the side the case states. Three
//!   red checks the engagement check alone can see: the choice ignored (the 5,119-row call run on the engine's own
//!   ending) and the rule off by one (5,119 rows a row early; 5,120 rows under `>`). Lines: TURBOEXL3_LING_DFIN_CALL,
//!   TURBOEXL3_LING_ROWS_RED.
//! - Timing (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; asserts nothing, guidance only): the rows path against the
//!   plugin's chain at the plugin's call shape (the first chunk of a 4096- and a 16384-token layer's call over 512
//!   experts at top-8, x bf16), min and median us per call over 8 rounds after 2 warm, each round's graph built and
//!   evaluated alone, and the two outputs' differing words. Lines: TURBOEXL3_LING_ROWS_TIME. Then each variant's arm on
//!   the same call through the row map against the served engine's, a 16384-token layer at uniform popularity (about
//!   256 rows an expert) and a ladder of rows per expert (8 to 512, every expert of a call at one count, about 4096
//!   pairs a call: the sweep the fused finish's whole-call threshold is chosen from): TURBOEXL3_LING_ROWS_TIME_ARM.
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
pub fn envList(name: [*:0]const u8, default: []const u32, buf: []u32) []const u32 {
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
pub fn loaderBank(g: *G, b: *const ref.Bank) !root.Bank {
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
pub fn drawPairsAt(a: std.mem.Allocator, rnd: std.Random, tokens: u32, n_slots: u32, skew: f64, act: []u32, slot: []u32) !void {
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
pub fn chunkEnd(s_slot: []const u32, c0: usize) usize {
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
pub fn runsOf(s_slot: []const u32) usize {
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
pub const RowsOps = struct { rm: mlx.mlx_array, ids: mlx.mlx_array, sc: mlx.mlx_array, n_call: usize, small: bool };

pub fn rowsOps(g: *G, act: []const u32, slot: []const u32) !RowsOps {
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
pub fn servedOptions(n: u32, t128: bool) prefill.Options {
    const intr = n % 16 == 0;
    return .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = intr, .defer_affine = true, .ncb = 4, .sintr = intr, .froute = true, .alay = 2, .t128 = t128 };
}

/// A rows-path call's geometry (`root.moeSwigluRows`'s: top-1 over the bank's slots, Ling's clamp after the SiLU)
pub fn rowsGeo(n: u32, experts: u32, limit: c_int) prefill.Geometry {
    return .{ .hidden = hidden, .inter = inter, .topk = 1, .experts = experts, .rate = format.Rate.fromN(n).?, .dec = .mul1, .limit = @intCast(limit), .clamp = .silu_out };
}

/// A bank's slot arrays as the engine takes them
pub fn opsOf(b: root.Bank) prefill.Operands {
    return .{ .gate_t = b.gate.trellis, .gate_suh = b.gate.suh, .gate_svh = b.gate.svh, .up_t = b.up.trellis, .up_suh = b.up.suh, .up_svh = b.up.svh, .down_t = b.down.trellis, .down_suh = b.down.suh, .down_svh = b.down.svh };
}

/// One rows-path call on engine `pf` through the row map (`moeT128With` with the 128-row tiles, else `moeWith`), y f32
/// [o.n_call, hidden]
fn engineRows(pf: *prefill.Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, o: RowsOps, ops: prefill.Operands) !mlx.mlx_array {
    if (pf.opts.t128) return pf.moeT128With(s, x, o.ids, o.sc, ops, o.n_call, .float32, o.rm);
    return pf.moeWith(s, x, o.ids, o.sc, ops, o.n_call, .float32, o.rm);
}

/// A variant of the routed chain: its name, its options at rate n, whether the TIME test times it (the byte test runs
/// every variant), the chain its bytes are held to: the served engine (the unfused chain at the served options), or
/// (`base`) the same stage shape with the variant's change off, where that stage shape is not the served chain's bytes
/// by itself, so the line isolates the change
const Variant = struct { name: []const u8, o: prefill.Options, time: bool = true, base: ?prefill.Options = null };

/// The served options (the 128-row tiles) with the fused finish set (`prefill.Options.dfin`)
fn vT(n: u32, dfin: bool) prefill.Options {
    var o = servedOptions(n, true);
    o.dfin = dfin;
    return o;
}

/// `vT`'s fused finish (the 128-row tiles, 32-row rounds in the freed stage buffer; `dfin` false: its base) at another
/// stage shape: 128-deep down stages (`bk128`: two sub-stages a buffer), two column blocks a threadgroup (`ncb` 2: the
/// pair is the last block's), or the MMA before the next stage's decode (`mma_first`)
fn vS(n: u32, bk128: bool, ncb: u8, mma_first: bool, dfin: bool) prefill.Options {
    var o = vT(n, dfin);
    o.bk128 = bk128;
    o.ncb = ncb;
    o.mma_first = mma_first;
    return o;
}

const n_variants = 4;

/// The routed chain's variants at rate n (the byte test's and the timing's arms): each one launch-shape change off the
/// served options, every one `moeLingRows`'s bytes by construction (the same per-simdgroup K loop on every body, the
/// finish's own statements on the store's f16 words)
fn variants(n: u32) [n_variants]Variant {
    return .{
        .{ .name = "dfin", .o = vT(n, true) },
        // the fused finish's staging at the other stage shapes it takes (byte arms, not timed), each held to its own stage
        // shape with the fused finish off
        .{ .name = "dfin.k128", .o = vS(n, true, 4, false, true), .time = false, .base = vS(n, true, 4, false, false) },
        .{ .name = "dfin.n2", .o = vS(n, false, 2, false, true), .time = false, .base = vS(n, false, 2, false, false) },
        .{ .name = "dfin.mf", .o = vS(n, false, 4, true, true), .time = false, .base = vS(n, false, 4, true, false) },
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
/// lognormal-skewed counts, one hot expert, a padded 5-pair call, experts on both sides of 64, 80 and 96 rows,
/// a call drawn from lognormal popularity (each count list at most 4096 pairs and 160 experts)
const SynthKind = enum { buckets, max4096, small, skewed, hot, padded, threshold, lognormal };

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
        // experts on both sides of 64, 80 and 96 rows in one call, beside 1, 32, 128, 287 and 512 (1680 pairs, 14
        // experts)
        .threshold => {
            const rows = [_]u32{ 79, 80, 81, 63, 64, 65, 95, 96, 97, 1, 32, 128, 287, 512 };
            @memcpy(buf[0..rows.len], &rows);
            return buf[0..rows.len];
        },
        // lognormal popularity (sigma 1.5, the timing's routing=lognormal) over 160 slots: 4096 pairs, each pair's slot
        // drawn by popularity, the slots that drew none dropped (a few large experts, many under 80 rows)
        .lognormal => {
            var w: [160]f64 = undefined;
            var tot: f64 = 0;
            for (&w) |*x| {
                x.* = @exp(1.5 * rnd.floatNorm(f64));
                tot += x.*;
            }
            var cnt: [160]u32 = @splat(0);
            for (0..4096) |_| {
                var u = rnd.float(f64) * tot;
                var e: usize = 0;
                while (e + 1 < w.len and u >= w[e]) : (e += 1) u -= w[e];
                cnt[e] += 1;
            }
            var k: usize = 0;
            for (cnt) |c| {
                if (c == 0) continue;
                buf[k] = c;
                k += 1;
            }
            return buf[0..k];
        },
    }
}

/// The ladder of rows per expert, every expert of a call at one count: the TIME ladder (`timeLadder`) and the byte
/// calls of PF_DFIN's whole-call switch
const ladder_rows = [_]u32{ 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 272, 288, 320, 384, 448, 512 };

/// A planted order fault's largest difference over the served words' RMS: a reordering's words differ by about an f16 ulp
/// (about 2^-10 of a value), so well under this; a garbage kernel's by about 1
const order_tol: f64 = 2e-2;

/// `compare` with `ref.compare`'s largest difference over `want`'s RMS (a red check of a reordering: its words differ,
/// every one finite and within `order_tol`)
fn compareTol(a: std.mem.Allocator, g: *G, got: mlx.mlx_array, want: mlx.mlx_array) !struct { differing: usize, nonfinite: usize, of: usize, max_ratio: f64 } {
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
    return .{ .differing = d, .nonfinite = nf, .of = n, .max_ratio = ref.compare(p, q).max_ratio };
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

/// PF_DFIN's whole-call switch: one rows-path call on engine `pf` through the row map at the call's own choice
/// (`prefill.Call`: `moeT128WithCall` with the 128-row tiles, else `moeWithCall`), y f32 [o.n_call, hidden]
fn engineCall(pf: *prefill.Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, o: RowsOps, ops: prefill.Operands, choice: prefill.Call) !mlx.mlx_array {
    if (pf.opts.t128) return pf.moeT128WithCall(s, x, o.ids, o.sc, ops, o.n_call, .float32, o.rm, choice);
    return pf.moeWithCall(s, x, o.ids, o.sc, ops, o.n_call, .float32, o.rm, choice);
}

/// The ending a call ran on its engine, from the engine's step counts before and after it (`prefill.Steps`): the served
/// route, down step and finish once each; or the served route and the fused finish once each; nothing else. Null for
/// any other mix (no ending, two, a step without its pair).
fn endRan(before: prefill.Steps, after: prefill.Steps) ?prefill.End {
    const d: prefill.Steps = .{
        .route = after.route -% before.route,
        .down = after.down -% before.down,
        .fin = after.fin -% before.fin,
        .dfin = after.dfin -% before.dfin,
    };
    if (std.meta.eql(d, prefill.Steps{ .route = 1, .down = 1, .fin = 1 })) return .served;
    if (std.meta.eql(d, prefill.Steps{ .route = 1, .dfin = 1 })) return .dfin;
    return null;
}

/// An ending's name on a line (`mixed`: `endRan`'s null)
fn endName(e: ?prefill.End) []const u8 {
    return if (e) |v| @tagName(v) else "mixed";
}

/// Whether `ran` is the ending `want`
fn endIs(ran: ?prefill.End, want: prefill.End) bool {
    return if (ran) |v| v == want else false;
}

/// Ling-3.1-flash's routed experts and PF_DFIN's whole-call threshold in rows an expert (the outside plugin's
/// MLX_SERVE_LING_PF_DFIN_CALL_MIN, 80 by default: the TIME ladder's step where the fused finish breaks even)
const route_experts: u64 = 512;
const dfin_call_min: u64 = 80;

/// A rule off by one (a red check's plant): `early` a row early, (rows + 1) * top_k; `strict` `>` for `>=`
const RuleSlip = enum { none, early, strict };

/// The plugin's whole-call rule (turboexl3_ling's src/ling_switches.zig `dfinCallTakes`, decided once a wide call from
/// its rows alone): a wide routed call of `rows` rows takes the fused finish when rows * top_k >= call_min * 512 (its mean
/// rows an expert at least call_min), at every call with call_min 0; `slip` plants a rule off by one
fn callTakes(rows: u64, call_min: u64, slip: RuleSlip) bool {
    if (call_min == 0) return true;
    const need = call_min * route_experts;
    return switch (slip) {
        .none => rows * top_k >= need,
        .early => (rows + 1) * top_k >= need,
        .strict => rows * top_k > need,
    };
}

/// A wide call's mean rows an expert at top-8 over 512 experts (rows * 8 / 512), for its line
fn rowsAnExpert(rows: u64) f64 {
    return @as(f64, @floatFromInt(rows * top_k)) / @as(f64, @floatFromInt(route_experts));
}

/// A chunk of a wide call of `rows` rows at its mean rows an expert: every expert at rows * 8 / 512 rows (at least 1), as
/// many experts as one chunk holds (`prefill_rows` pairs) and `buf` (the bank's slots); the counts in `buf`
fn chunkAtMean(rows: u64, buf: []u32) []const u32 {
    const rpe: u32 = @intCast(@max(1, rows * top_k / route_experts));
    const ne: usize = @min(buf.len, prefill_rows / rpe);
    @memset(buf[0..ne], rpe);
    return buf[0..ne];
}

/// The rule's cases: a wide call's rows, its call_min, and the ending the case states on the fused finish's engine:
/// either side of 80 rows an expert (5,119 rows: 79.98; 5,120: 80.00), a layer-major call of 16,384 rows (256.00), a short
/// call of 1,120 (17.50), and call_min 0, which takes the fused finish at every call
const RuleCase = struct { rows: u64, call_min: u64, want: prefill.End };
const rule_cases = [_]RuleCase{
    .{ .rows = 1120, .call_min = dfin_call_min, .want = .served },
    .{ .rows = 5119, .call_min = dfin_call_min, .want = .served },
    .{ .rows = 5120, .call_min = dfin_call_min, .want = .dfin },
    .{ .rows = 16384, .call_min = dfin_call_min, .want = .dfin },
    .{ .rows = 1120, .call_min = 0, .want = .dfin },
    .{ .rows = 5119, .call_min = 0, .want = .dfin },
    .{ .rows = 16384, .call_min = 0, .want = .dfin },
};

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

test "turboexl3 gpu ling rows variants: the routed chain's fused finish (over both tables at the served options, and at each other stage shape it takes) is the served engine's words on synthetic calls at every row bucket, n 40 and 32, limits 0 and 10; planted faults are caught (TURBOEXL3_GPU=1)" {
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
            var bases: [n_variants]prefill.Prefill = undefined;
            var has_base: [n_variants]bool = @splat(false);
            var made: usize = 0;
            defer for (engines[0..made], bases[0..made], has_base[0..made]) |*e, *bse, hb2| {
                e.deinit();
                if (hb2) bse.deinit();
            };
            for (vs, &engines, &bases, &has_base) |v, *e, *bse, *hb2| {
                e.* = try prefill.Prefill.init(geo, v.o);
                made += 1;
                if (v.base) |bo| {
                    bse.* = try prefill.Prefill.init(geo, bo);
                    hb2.* = true;
                }
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
                for (vs, &engines, &bases, has_base) |v, *e, *bse, hb2| {
                    // fresh buffers for the variant's arrays: no word it fails to write can hold a served value
                    _ = mlx.mlx_clear_cache();
                    const m3 = g.mark();
                    defer g.resetTo(m3);
                    const held = if (hb2) try g.adopt(try engineRows(bse, s, xb, o, ops)) else want;
                    _ = mlx.mlx_clear_cache();
                    const got = try g.adopt(try engineRows(e, s, xb, o, ops));
                    const c = try compare(a, &g, got, held);
                    const ok = c.differing == 0 and c.nonfinite == 0;
                    std.debug.print("TURBOEXL3_LING_ROWS_VARIANT n={d} limit={d} call={s} pairs={d} experts={d} variant={s} against={s} differing={d}/{d} nonfinite={d} {s}\n", .{ n, lim, @tagName(kind), call.act.len, counts.len, v.name, if (hb2) "base" else "served", c.differing, c.of, c.nonfinite, if (ok) "PASS" else "FAIL" });
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
            // 1. one output word flipped: the served engine's own y with one word's lowest bit flipped (the comparison counts
            // exactly it)
            {
                const c = try compare(a, &g, try flipOne(a, &g, want, (mlx.mlx_array_size(want) / 2) | 1), want);
                const caught = c.differing == 1;
                std.debug.print("TURBOEXL3_LING_ROWS_RED n={d} fault=flipped_word differing={d}/{d} caught={}\n", .{ n, c.differing, c.of, caught });
                if (!caught) failed = true;
            }
            // 2. the fused finish with tile 0's f16 rounding dropped (a planted epilogue fault in the kernel)
            {
                var pf = try prefill.Prefill.init(geo, vT(n, true));
                defer pf.deinit();
                const fault = ks.newKernel("turboexl3_test_pf_dfin_fault", &prefill.dfin_in, &prefill.dfin_out, comptime prefill.dfinText(prefill.dfin_gemm_4, @embedFile("kernels/test/fault/dfin_4.metal")), prefill.header_nax_dfin, true) orelse return error.KernelCreateFailed;
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
            // 3. the GEMM's weights rounded to f16 before the products (the deferred MUL1 map off): a rounding change inside
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
            // 4. a reduction's order changed and nothing else: the fused finish with tile 0's H128 stages in another order
            // (the same transform in exact arithmetic, its f32 sums reassociated; a planted fault in the kernel)
            {
                var pf = try prefill.Prefill.init(geo, vT(n, true));
                defer pf.deinit();
                const fault = ks.newKernel("turboexl3_test_pf_dfin_order_fault", &prefill.dfin_in, &prefill.dfin_out, comptime prefill.dfinText(prefill.dfin_gemm_4, @embedFile("kernels/test/fault/dfin_4_order.metal")), prefill.header_nax_dfin, true) orelse return error.KernelCreateFailed;
                defer root.decode.freeKernel(fault);
                const kept = pf.dfin_k;
                pf.dfin_k = fault;
                defer pf.dfin_k = kept;
                _ = mlx.mlx_clear_cache();
                const got = try g.adopt(try engineRows(&pf, s, xb, o, ops));
                const c = try compareTol(a, &g, got, want);
                const caught = c.differing > 0 and c.nonfinite == 0 and c.max_ratio <= order_tol;
                std.debug.print("TURBOEXL3_LING_ROWS_RED n={d} fault=dfin_tile0_h128_reordered differing={d}/{d} nonfinite={d} max_ratio={e:.3} caught={}\n", .{ n, c.differing, c.of, c.nonfinite, c.max_ratio, caught });
                if (!caught) failed = true;
            }
            // 5. the K loop's order changed and nothing else: the fused down GEMM with tile 0's K stages in reverse order
            // (the same products, its accumulators summed in another order; a planted fault in the kernel)
            {
                var pf = try prefill.Prefill.init(geo, vT(n, true));
                defer pf.deinit();
                const fault = ks.newKernel("turboexl3_test_pf_dfin_korder_fault", &prefill.dfin_in, &prefill.dfin_out, comptime prefill.dfinText(@embedFile("kernels/test/fault/gemm_4_krev.metal"), prefill.dfin_finish_4), prefill.header_nax_dfin, true) orelse return error.KernelCreateFailed;
                defer root.decode.freeKernel(fault);
                const kept = pf.dfin_k;
                pf.dfin_k = fault;
                defer pf.dfin_k = kept;
                _ = mlx.mlx_clear_cache();
                const got = try g.adopt(try engineRows(&pf, s, xb, o, ops));
                const c = try compareTol(a, &g, got, want);
                const caught = c.differing > 0 and c.nonfinite == 0 and c.max_ratio <= order_tol;
                std.debug.print("TURBOEXL3_LING_ROWS_RED n={d} fault=dfin_tile0_k_reversed differing={d}/{d} nonfinite={d} max_ratio={e:.3} caught={}\n", .{ n, c.differing, c.of, c.nonfinite, c.max_ratio, caught });
                if (!caught) failed = true;
            }
        }
        // PF_DFIN's whole-call switch (`prefill.Call`; the outside plugin's MLX_SERVE_LING_PF_DFIN_CALL_MIN): a call keeps
        // or declines the fused finish on an engine built with it. Both endings give the served words, so each line also
        // names the ending the call ran, from the engine's counts (`endRan`), and holds it to the call's choice.
        for ([_]c_int{ 0, 10 }) |lim| {
            const geo = rowsGeo(n, prompt_slots, lim);
            var served = try prefill.Prefill.init(geo, servedOptions(n, true));
            defer served.deinit();
            // the plugin's engine: PF_DFIN (the fused finish); a call that keeps the fused finish runs the
            // engine's own ending, one that declines it the served one
            var plain = try prefill.Prefill.init(geo, vT(n, true));
            defer plain.deinit();
            const DfinArm = struct { name: []const u8, pf: *prefill.Prefill, own: prefill.End, ladder: bool };
            const dfin_arms = [_]DfinArm{
                .{ .name = "dfin", .pf = &plain, .own = .dfin, .ladder = true },
            };
            var cbuf: [160]u32 = undefined;
            // every synthetic call, then every rung of the ladder (that engine's): both choices on each engine, each
            // the served engine's words, each ending the choice's
            const kinds = std.enums.values(SynthKind);
            for (0..kinds.len + ladder_rows.len) |at| {
                const m2 = g.mark();
                defer g.resetTo(m2);
                const rung = at >= kinds.len;
                const counts: []const u32 = if (rung) blk: {
                    const rpe = ladder_rows[at - kinds.len];
                    const ne: usize = @min(cbuf.len, 4096 / rpe);
                    @memset(cbuf[0..ne], rpe);
                    break :blk cbuf[0..ne];
                } else synthCounts(kinds[at], rnd, &cbuf);
                var name_buf: [16]u8 = undefined;
                const call_name: []const u8 = if (rung) try std.fmt.bufPrint(&name_buf, "ladder{d}", .{ladder_rows[at - kinds.len]}) else @tagName(kinds[at]);
                const call = try synthCall(a, rnd, counts, prompt_slots, prompt_tokens);
                defer a.free(call.act);
                defer a.free(call.slot);
                const o = try rowsOps(&g, call.act, call.slot);
                _ = mlx.mlx_clear_cache();
                const want = try g.adopt(try engineRows(&served, s, xb, o, ops));
                for (dfin_arms) |arm| {
                    if (rung and !arm.ladder) continue;
                    for ([_]bool{ false, true }) |keep| {
                        // fresh buffers for the call's arrays: no word it fails to write can hold a served value
                        _ = mlx.mlx_clear_cache();
                        const m3 = g.mark();
                        defer g.resetTo(m3);
                        const before = arm.pf.steps;
                        const got = try g.adopt(try engineCall(arm.pf, s, xb, o, ops, .{ .dfin = keep }));
                        const ran = endRan(before, arm.pf.steps);
                        const want_end: prefill.End = if (keep) arm.own else .served;
                        const c = try compare(a, &g, got, want);
                        const ok = endIs(ran, want_end) and c.differing == 0 and c.nonfinite == 0;
                        std.debug.print("TURBOEXL3_LING_DFIN_CALL n={d} limit={d} call={s} pairs={d} experts={d} arm={s} take={s} engine={s} want={s} differing={d}/{d} nonfinite={d} {s}\n", .{ n, lim, call_name, call.act.len, counts.len, arm.name, if (keep) "dfin" else "served", endName(ran), @tagName(want_end), c.differing, c.of, c.nonfinite, if (ok) "PASS" else "FAIL" });
                        if (!ok) failed = true;
                    }
                }
            }
            // the plugin's rule (`callTakes`) on either side of 80 rows an expert and at call_min 0, on that engine:
            // each case's call decided by the rule and run on a chunk at its call's mean rows an expert (`chunkAtMean`),
            // the served engine's words, its ending the side the case states
            for (rule_cases) |rc| {
                const m2 = g.mark();
                defer g.resetTo(m2);
                const counts = chunkAtMean(rc.rows, &cbuf);
                const call = try synthCall(a, rnd, counts, prompt_slots, prompt_tokens);
                defer a.free(call.act);
                defer a.free(call.slot);
                const o = try rowsOps(&g, call.act, call.slot);
                _ = mlx.mlx_clear_cache();
                const want = try g.adopt(try engineRows(&served, s, xb, o, ops));
                const keep = callTakes(rc.rows, rc.call_min, .none);
                _ = mlx.mlx_clear_cache();
                const before = plain.steps;
                const got = try g.adopt(try engineCall(&plain, s, xb, o, ops, .{ .dfin = keep }));
                const ran = endRan(before, plain.steps);
                const c = try compare(a, &g, got, want);
                const ok = endIs(ran, rc.want) and c.differing == 0 and c.nonfinite == 0;
                std.debug.print("TURBOEXL3_LING_DFIN_CALL n={d} limit={d} call=rule rows={d} call_min={d} rows_an_expert={d:.2} pairs={d} experts={d} arm=dfin take={s} engine={s} want={s} differing={d}/{d} nonfinite={d} {s}\n", .{ n, lim, rc.rows, rc.call_min, rowsAnExpert(rc.rows), call.act.len, counts.len, if (keep) "dfin" else "served", endName(ran), @tagName(rc.want), c.differing, c.of, c.nonfinite, if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            }
            // the red checks (limit 0): a call on the wrong side leaves the bytes alone, both endings giving the served
            // words, so each must show in the engagement check alone: the choice ignored (the 5,119-row call, which keeps
            // the served chain, run on the engine's own ending: `engineRows`), the rule a row early (5,119 rows taking the
            // fused finish) and the rule under `>` (5,120 rows keeping the served chain)
            if (lim != 0) continue;
            const Red = struct { name: []const u8, rows: u64, slip: RuleSlip, ignored: bool, want: prefill.End };
            const reds = [_]Red{
                .{ .name = "dfin_call_choice_ignored", .rows = 5119, .slip = .none, .ignored = true, .want = .served },
                .{ .name = "dfin_call_rule_early", .rows = 5119, .slip = .early, .ignored = false, .want = .served },
                .{ .name = "dfin_call_rule_strict", .rows = 5120, .slip = .strict, .ignored = false, .want = .dfin },
            };
            for (reds) |rd| {
                const m2 = g.mark();
                defer g.resetTo(m2);
                const counts = chunkAtMean(rd.rows, &cbuf);
                const call = try synthCall(a, rnd, counts, prompt_slots, prompt_tokens);
                defer a.free(call.act);
                defer a.free(call.slot);
                const o = try rowsOps(&g, call.act, call.slot);
                _ = mlx.mlx_clear_cache();
                const want = try g.adopt(try engineRows(&served, s, xb, o, ops));
                const keep = callTakes(rd.rows, dfin_call_min, rd.slip);
                _ = mlx.mlx_clear_cache();
                const before = plain.steps;
                const got = try g.adopt(if (rd.ignored) try engineRows(&plain, s, xb, o, ops) else try engineCall(&plain, s, xb, o, ops, .{ .dfin = keep }));
                const ran = endRan(before, plain.steps);
                const c = try compare(a, &g, got, want);
                const caught = !endIs(ran, rd.want);
                std.debug.print("TURBOEXL3_LING_ROWS_RED n={d} fault={s} rows={d} call_min={d} take={s} engine={s} held_to={s} differing={d}/{d} caught={}\n", .{ n, rd.name, rd.rows, dfin_call_min, if (keep) "dfin" else "served", endName(ran), @tagName(rd.want), c.differing, c.of, caught });
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
            try timeLadder(a, &g, s, n, experts, urnd, tokens, xb, bank);
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
        if (!v.time) continue;
        const vm = g.mark();
        defer g.resetTo(vm);
        var pf = try prefill.Prefill.init(geo, v.o);
        defer pf.deinit();
        const t = try timeArm(g, .{ .s = s, .pf = &pf, .x = xb, .o = o, .ops = ops });
        const c = try compare(a, g, try g.adopt(try engineRows(&pf, s, xb, o, ops)), want);
        std.debug.print("TURBOEXL3_LING_ROWS_TIME_ARM n={d} routing={s} of_tokens={d} pairs={d} experts={d} arm={s} min_us={d:.1} med_us={d:.1} tflops={d:.2} vs_served_min={d:.3} differing={d}/{d}\n", .{ n, routing, tokens, act.len, runsOf(slot), v.name, t[0], t[1], flops / (t[0] * 1e6), t[0] / base[0], c.differing, c.of });
    }
}

/// The variants' arms over a ladder of rows per expert (guidance only: the sweep the fused finish's whole-call threshold
/// is chosen from, over row counts, never a prompt's routing): for each count r, a call of min(4096 / r, 512)
/// experts of r rows each (about 4096 pairs), act rows drawn at random, through `timeVariants`. Lines:
/// TURBOEXL3_LING_ROWS_TIME_ARM with routing=ladder<r>.
fn timeLadder(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, n: u32, experts: u32, rnd: std.Random, tokens: u32, xb: mlx.mlx_array, bank: root.Bank) !void {
    var counts: [512]u32 = undefined;
    for (ladder_rows) |rpe| {
        const lm = g.mark();
        defer g.resetTo(lm);
        const ne: usize = @min(@min(counts.len, experts), 4096 / rpe);
        @memset(counts[0..ne], rpe);
        const call = try synthCall(a, rnd, counts[0..ne], experts, tokens);
        defer a.free(call.act);
        defer a.free(call.slot);
        var name_buf: [16]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "ladder{d}", .{rpe});
        try timeVariants(a, g, s, n, experts, name, tokens, xb, call.act, call.slot, bank);
    }
}
