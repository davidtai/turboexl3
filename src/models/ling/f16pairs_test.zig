//! L1, the outside plugin turboexl3_ling's MLX_SERVE_LING_PF_F16_PAIRS: a prompt call's routed pairs stored as f16
//! instead of f32 (`ling.moeLingRowsPairsF16`), and the JOINLESS combine over f16 sources (`ling.combineParts`,
//! `text_parts_h`), held word for word to the served f32 pairs and combine.
//! - CPU (always): the exactness rule (`ling.pairsF16Exact`) takes a top-1 call at weights 1 and +0 only; the host
//!   round-trip this file's checks use keeps a planted -0 and catches a value off the f16 grid.
//! - GPU (TURBOEXL3_GPU=1, else skipped), at Ling's shapes (hidden 4096, expert width 1280, MUL1 at n 40, the 2.60 pack's
//!   routed rate; TURBOEXL3_LING_N names others), limits 0 and 10 (the clamp after the SiLU), on the loader's views of a
//!   512-slot bank (rows_test.zig's `loaderBank`), x the layer's MoE input in bf16 (the streamer's dtype), at the served
//!   sizes: the standard request's prompt layer (18,380 tokens at top-8 over the 512 slots: 147,040 pairs in the plugin's
//!   chunks of whole experts of at most 4,096 pairs), a 2,048-token call and a 2-token call (16 pairs, padded to 17).
//!   Every chunk on the served engine's two endings (the fused finish, PF_DFIN's, and the served finish) in f32 and in
//!   f16: each f16 word widens to the f32 word, and every f32 word round-trips through f16 (TURBOEXL3_LING_F16_PAIRS).
//!   Then each call's combine (the plugin's ending: the fused finish at 5,120 tokens or more, else the served one) over
//!   its JOINLESS sources (above 24 the smallest merged, as the streamer merges them): f16 sources, and a mix of f16 and
//!   f32 ones, against the f32 sources, word for word (TURBOEXL3_LING_F16_PAIRS_COMBINE); the entry itself
//!   (`moeLingRowsPairsF16` with the plugin's host weights) against `moeLingRowsWith` at f32 on the first chunk and the
//!   small call, both endings asked (TURBOEXL3_LING_F16_PAIRS_API). Red checks (TURBOEXL3_LING_F16_PAIRS_RED): a weight
//!   other than 1 (0.3, 0.5, -0, 2) is refused before any launch, and at 0.3 the served f32 rows hold words off the f16
//!   grid (so the refusal is needed); a two-slot call (the split of partials a finish would sum) is refused by the rule,
//!   and its f32 rows hold words off the f16 grid; a flipped f16 word in one source shows in the combine.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const prefill = @import("../../core/prefill.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const api = @import("api.zig");
const combine = @import("combine.zig");
const rt = @import("rows_test.zig");
const ct = @import("combine_test.zig");

const testing = std.testing;
const hidden: u32 = 4096;
const inter: u32 = 1280;
const top_k: u32 = 8;
const sigma: f32 = 8;
/// The standard request's prompt layer: its tokens, and the slots its top-8 routes over
const std_tokens: u32 = 18_380;
const std_slots: u32 = 512;
/// A call under PF_DFIN's whole-call cut (5,120 rows at top-8 over 512): the served finish in the plugin
const tail_tokens: u32 = 2_048;
/// A call of 16 pairs: the plugin pads it to 17 with +0 weights
const small_tokens: u32 = 2;
/// The plugin's whole-call cut (turboexl3_ling's PF_DFIN_CALL_MIN 80 at top-8 over 512: R * 8 >= 80 * 512)
const dfin_call_rows: u32 = 5_120;
const gemm_min_rows: usize = root.kernels.DECODE_ROWS_MAX + 1;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// `v` stored as f16 and widened back gives `v`'s word (a planted -0 keeps its sign; NaN payloads aside, every value
/// on the f16 grid does)
fn roundTrips(v: f32) bool {
    const h: f16 = @floatCast(v);
    const back: f32 = h;
    return @as(u32, @bitCast(back)) == @as(u32, @bitCast(v));
}

test "turboexl3 ling f16 pairs: the exactness rule takes a top-1 call at weights 1 and +0 only; another weight, a second slot or no weight is refused; the round-trip keeps -0 and catches a value off the f16 grid" {
    const ones = [_]f32{ 1, 1, 1 };
    const padded = [_]f32{ 1, 1, 0, 0, 0 };
    try testing.expect(api.pairsF16Exact(.{ .topk = 1, .weights = &ones }));
    try testing.expect(api.pairsF16Exact(.{ .topk = 1, .weights = &padded }));
    // a second slot: the finish sums two weighted terms, which f16 does not hold
    for ([_]u32{ 0, 2, 6, 8 }) |k| try testing.expect(!api.pairsF16Exact(.{ .topk = k, .weights = &ones }));
    try testing.expect(!api.pairsF16Exact(.{ .topk = 1, .weights = &.{} }));
    const off = [_]f32{ 0.5, 0.3, 2.0, -1.0, -0.0, std.math.inf(f32), std.math.nan(f32), 1.0 + std.math.floatEps(f32), std.math.floatMin(f32) };
    for (off) |w| {
        const ws = [_]f32{ 1, w, 1 };
        try testing.expect(!api.pairsF16Exact(.{ .topk = 1, .weights = &ws }));
    }
    // the host round-trip: f16 values (and both zeros) keep their words, a value between two f16 values does not
    for ([_]f32{ 0.0, -0.0, 1.0, -2.5, 65504.0, 0x1p-14, 0x1p-24, std.math.inf(f32) }) |v| try testing.expect(roundTrips(v));
    try testing.expect(@as(u16, @bitCast(@as(f16, @floatCast(@as(f32, -0.0))))) == 0x8000);
    for ([_]f32{ 1.0 + 0x1p-20, 0.1, 65504.0 + 16.0, 1.0e-9 }) |v| try testing.expect(!roundTrips(v));
}

/// The plugin's counting sort of a call's pairs by slot (ling_math.zig `chunks()`, stable): `order[i]` the call's pair at
/// sorted position i, with its act row and slot
fn sortOrder(a: std.mem.Allocator, act: []const u32, slot: []const u32, order: []u32, s_act: []u32, s_slot: []u32) !void {
    var max_slot: u32 = 0;
    for (slot) |sv| max_slot = @max(max_slot, sv);
    const starts = try a.alloc(u32, @as(usize, max_slot) + 2);
    defer a.free(starts);
    @memset(starts, 0);
    for (slot) |sv| starts[sv + 1] += 1;
    for (1..starts.len) |i| starts[i] += starts[i - 1];
    for (slot, 0..) |sv, i| {
        order[starts[sv]] = @intCast(i);
        starts[sv] += 1;
    }
    for (order, s_act, s_slot) |pi, *ao, *so| {
        ao.* = act[pi];
        so.* = slot[pi];
    }
}

/// The plugin's host weights for a rows-path call of `rows` pairs (ling_math.zig `rowsCall`): 1 a pair, and a call of
/// 16 pairs or fewer padded to 17 with +0 for each pad row
fn hostWeights(rows: usize, buf: []f32) []const f32 {
    const n = if (root.kernels.usesPrefillArm(rows)) rows else gemm_min_rows;
    for (buf[0..n], 0..) |*w, i| w.* = if (i < rows) 1.0 else 0.0;
    return buf[0..n];
}

/// One chunk's f16 rows against its f32 rows: the words of the widened f16 rows that differ from the f32 rows, the f32
/// words that do not round-trip through f16, and the f32 rows' non-finite words
const Check = struct { differing: usize = 0, not_f16: usize = 0, nonfinite: usize = 0, of: usize = 0 };

fn checkPairs(a: std.mem.Allocator, g: *G, got16: mlx.mlx_array, want32: mlx.mlx_array) !Check {
    const n = mlx.mlx_array_size(want32);
    if (mlx.mlx_array_size(got16) != n or mlx.mlx_array_dtype(got16) != .float16 or mlx.mlx_array_dtype(want32) != .float32) return error.SizeMismatch;
    const p = try a.alloc(f32, n);
    defer a.free(p);
    const q = try a.alloc(f32, n);
    defer a.free(q);
    _ = try g.hostF32(try g.astype(got16, .float32), p);
    _ = try g.hostF32(want32, q);
    var c: Check = .{ .of = n };
    for (p, q) |u, v| {
        c.differing += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
        c.not_f16 += @intFromBool(!roundTrips(v));
        c.nonfinite += @intFromBool(!std.math.isFinite(v));
    }
    return c;
}

/// The words of two f32 arrays that differ
fn differingWords(a: std.mem.Allocator, g: *G, got: mlx.mlx_array, want: mlx.mlx_array) !usize {
    const n = mlx.mlx_array_size(want);
    if (mlx.mlx_array_size(got) != n) return error.SizeMismatch;
    const p = try a.alloc(f32, n);
    defer a.free(p);
    const q = try a.alloc(f32, n);
    defer a.free(q);
    _ = try g.hostF32(got, p);
    _ = try g.hostF32(want, q);
    var d: usize = 0;
    for (p, q) |u, v| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
    return d;
}

/// The f32 rows' words off the f16 grid
fn notF16Words(a: std.mem.Allocator, g: *G, y32: mlx.mlx_array) !usize {
    const n = mlx.mlx_array_size(y32);
    const p = try a.alloc(f32, n);
    defer a.free(p);
    _ = try g.hostF32(y32, p);
    var k: usize = 0;
    for (p) |v| k += @intFromBool(!roundTrips(v));
    return k;
}

/// A call's JOINLESS combine (the streamer's plan: up to 24 outputs in place, above that the smallest merged in join
/// order into the last source) over `outs`, each pair's routed position in `pos` (the call's pairs in join order) and
/// the weights `wa` f32 [n, K]. With `alt` (the same outputs in the other dtype), every other source read in place is
/// `alt`'s (a mix of f16 and f32 sources; the merged source stays `outs`', one dtype)
fn combineOf(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, outs: []const mlx.mlx_array, alt: ?[]const mlx.mlx_array, rows: []const u32, pos: []const u32, n: usize, wa: mlx.mlx_array) !struct { y: mlx.mlx_array, sources: usize, merged: usize } {
    const n_out = outs.len;
    const order = try a.alloc(u32, n_out);
    defer a.free(order);
    const src = try a.alloc(u8, n_out);
    defer a.free(src);
    const off = try a.alloc(u32, n_out);
    defer a.free(off);
    const n_src = ct.planJoinless(rows, order, src, off);
    const loc = try a.alloc(i32, 2 * n * top_k);
    defer a.free(loc);
    ct.fillJoinlessLoc(rows, src, off, pos, loc);
    var sources: std.ArrayList(mlx.mlx_array) = .empty;
    defer sources.deinit(a);
    var merge_in: std.ArrayList(mlx.mlx_array) = .empty;
    defer merge_in.deinit(a);
    for (outs, src, 0..) |x, sv, i| {
        if (n_src < n_out and sv == n_src - 1) {
            try merge_in.append(a, x);
        } else if (alt) |al| {
            try sources.append(a, if (sources.items.len % 2 == 1) al[i] else x);
        } else try sources.append(a, x);
    }
    if (merge_in.items.len > 0) try sources.append(a, try g.concat(merge_in.items, 0));
    if (sources.items.len != n_src) return error.JoinlessPlan;
    const la = try g.hostArray(std.mem.sliceAsBytes(loc), &.{ ci(n), ci(top_k), 2 }, .int32);
    const y = try g.adopt(try combine.combineParts(s, sources.items, la, wa));
    return .{ .y = y, .sources = n_src, .merged = merge_in.items.len };
}

/// A call's pairs as the streamer hands them to the plugin (each token's top-8 slots, token order), the routing weights
/// the combine reads (bf16 values in (0, 2.5), a router's after its scale), and the plugin's sort and chunks
const CallPairs = struct {
    name: []const u8,
    tokens: u32,
    act: []u32,
    slot: []u32,
    w: []f32,
    order: []u32,
    s_act: []u32,
    s_slot: []u32,
    bounds: std.ArrayList([2]usize) = .empty,

    fn init(a: std.mem.Allocator, rnd: std.Random, name: []const u8, tokens: u32) !CallPairs {
        const np = @as(usize, tokens) * top_k;
        var c: CallPairs = .{
            .name = name,
            .tokens = tokens,
            .act = try a.alloc(u32, np),
            .slot = try a.alloc(u32, np),
            .w = try a.alloc(f32, np),
            .order = try a.alloc(u32, np),
            .s_act = try a.alloc(u32, np),
            .s_slot = try a.alloc(u32, np),
        };
        try rt.drawPairsAt(a, rnd, tokens, std_slots, 1.5, c.act, c.slot);
        for (c.w) |*v| v.* = @bitCast(@as(u32, support.bf16Bits(0.01 + 2.49 * rnd.float(f32))) << 16);
        try sortOrder(a, c.act, c.slot, c.order, c.s_act, c.s_slot);
        var c0: usize = 0;
        while (c0 < np) {
            const c1 = rt.chunkEnd(c.s_slot, c0);
            try c.bounds.append(a, .{ c0, c1 });
            c0 = c1;
        }
        return c;
    }

    fn deinit(c: *CallPairs, a: std.mem.Allocator) void {
        for ([_][]u32{ c.act, c.slot, c.order, c.s_act, c.s_slot }) |x| a.free(x);
        a.free(c.w);
        c.bounds.deinit(a);
    }
};

/// One chunk on engine `pf` at `out_dtype` through the row map, the call's ending as `call` asks: y [pairs, hidden] (a
/// padded call's pad rows dropped)
fn chunkRows(g: *G, pf: *prefill.Prefill, s: mlx.mlx_stream, xb: mlx.mlx_array, o: rt.RowsOps, ops: prefill.Operands, pairs: usize, out_dtype: mlx.mlx_dtype, call: prefill.Call) !mlx.mlx_array {
    const y = try g.adopt(try pf.moeT128WithCall(s, xb, o.ids, o.sc, ops, o.n_call, out_dtype, o.rm, call));
    if (!o.small) return y;
    return g.slice(y, &.{ 0, 0 }, &.{ ci(pairs), ci(hidden) }, &.{ 1, 1 });
}

const ending_names = [_][]const u8{ "dfin", "served" };

test "turboexl3 gpu ling f16 pairs: moeLingRowsPairsF16's f16 rows widen to the served f32 rows, and the JOINLESS combine over f16 sources gives the f32 sources' words, at the standard request's call (18,380 tokens at top-8 over 512), a 2,048-token call and a padded call, on both endings, with red checks (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = rt.envList("TURBOEXL3_LING_N", &.{40}, &nbuf);
    var failed = false;
    var reds_run: usize = 0;
    for (rates, 0..) |n, ri| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x11F1_6A15 ^ @as(u64, n));
        const rnd = prng.random();
        var hb = try ref.synthBank(a, rnd, std_slots, hidden, inter, rate);
        defer hb.deinit(a);
        const tb = try rt.loaderBank(&g, &hb);
        const ops = rt.opsOf(tb);
        // the layer's MoE input: N(0, 8^2) (a part of the gates and ups pass the clamp), bf16 as the streamer hands it
        const xh = try a.alloc(f32, @as(usize, std_tokens) * hidden);
        defer a.free(xh);
        for (xh) |*v| v.* = sigma * rnd.floatNorm(f32);
        const xb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(std_tokens), ci(hidden) }, .float32), .bfloat16);
        var calls = [_]CallPairs{
            try CallPairs.init(a, rnd, "standard", std_tokens),
            try CallPairs.init(a, rnd, "tail2048", tail_tokens),
            try CallPairs.init(a, rnd, "small", small_tokens),
        };
        defer for (&calls) |*c| c.deinit(a);
        for ([_]c_int{ 0, 10 }) |lim| {
            // the served engine (the 128-row tiles) built with the fused finish: a call takes it or declines it, as the
            // plugin's PF_DFIN whole-call rule asks
            var opts = rt.servedOptions(n, true);
            opts.dfin = true;
            var pf = try prefill.Prefill.init(rt.rowsGeo(n, std_slots, lim), opts);
            defer pf.deinit();
            for (&calls) |*c| {
                const m1 = g.mark();
                defer g.resetTo(m1);
                // the plugin's ending for this call (V1c: the fused finish at 5,120 tokens or more)
                const plugin_dfin = c.tokens >= dfin_call_rows;
                var keep32: std.ArrayList(mlx.mlx_array) = .empty;
                defer {
                    for (keep32.items) |x| g.release(x);
                    keep32.deinit(a);
                }
                var keep16: std.ArrayList(mlx.mlx_array) = .empty;
                defer {
                    for (keep16.items) |x| g.release(x);
                    keep16.deinit(a);
                }
                var rows: std.ArrayList(u32) = .empty;
                defer rows.deinit(a);
                const nb = c.bounds.items.len;
                for (c.bounds.items, 0..) |bd, ck| {
                    const m2 = g.mark();
                    defer g.resetTo(m2);
                    const ca = c.s_act[bd[0]..bd[1]];
                    const cs = c.s_slot[bd[0]..bd[1]];
                    const o = try rt.rowsOps(&g, ca, cs);
                    for ([_]bool{ true, false }, 0..) |dfin, ei| {
                        const want = try chunkRows(&g, &pf, s, xb, o, ops, ca.len, .float32, .{ .dfin = dfin });
                        const got = try chunkRows(&g, &pf, s, xb, o, ops, ca.len, .float16, .{ .dfin = dfin });
                        const k = try checkPairs(a, &g, got, want);
                        const ok = k.differing == 0 and k.not_f16 == 0 and k.nonfinite == 0;
                        std.debug.print("TURBOEXL3_LING_F16_PAIRS n={d} limit={d} call={s} ending={s} chunk={d}/{d} pairs={d} experts={d} padded={} f16_differing={d}/{d} f32_off_f16_grid={d} nonfinite={d} {s}\n", .{ n, lim, c.name, ending_names[ei], ck + 1, nb, ca.len, rt.runsOf(cs), o.small, k.differing, k.of, k.not_f16, k.nonfinite, if (ok) "PASS" else "FAIL" });
                        if (!ok) failed = true;
                        if (dfin == plugin_dfin) {
                            try keep32.append(a, g.keep(want));
                            try keep16.append(a, g.keep(got));
                        }
                    }
                    try rows.append(a, @intCast(ca.len));
                    // the entry the plugin calls, on the first chunk and the small call, both endings asked
                    if (ck == 0) for ([_]bool{ true, false }) |dfin| {
                        const m3 = g.mark();
                        defer g.resetTo(m3);
                        const wbuf = try a.alloc(f32, @max(ca.len, gemm_min_rows));
                        defer a.free(wbuf);
                        const ws = hostWeights(ca.len, wbuf);
                        const y16 = try g.adopt(try api.moeLingRowsPairsF16(s, xb, o.rm, tb, o.ids, ws, .mul1, lim, .{ .dfin = dfin }));
                        const y32 = try g.adopt(try api.moeLingRowsWith(s, xb, o.rm, tb, o.ids, o.sc, .mul1, lim, .float32, .{ .dfin = dfin }));
                        const k = try checkPairs(a, &g, y16, y32);
                        const ok = k.differing == 0 and k.not_f16 == 0 and mlx.mlx_array_dtype(y16) == .float16;
                        std.debug.print("TURBOEXL3_LING_F16_PAIRS_API n={d} limit={d} call={s} dfin_asked={} row_slots={d} differing={d}/{d} f32_off_f16_grid={d} {s}\n", .{ n, lim, c.name, dfin, ws.len, k.differing, k.of, k.not_f16, if (ok) "PASS" else "FAIL" });
                        if (!ok) failed = true;
                    };
                }
                // the call's combine over its chunks (the plugin's ending): f16 sources and a mix, against the f32 ones
                const wa = try g.hostArray(std.mem.sliceAsBytes(c.w), &.{ ci(c.tokens), ci(top_k) }, .float32);
                const want = try combineOf(a, &g, s, keep32.items, null, rows.items, c.order, c.tokens, wa);
                const got = try combineOf(a, &g, s, keep16.items, null, rows.items, c.order, c.tokens, wa);
                const mix = try combineOf(a, &g, s, keep16.items, keep32.items, rows.items, c.order, c.tokens, wa);
                for ([_][]const u8{ "f16", "mixed" }, [_]mlx.mlx_array{ got.y, mix.y }) |label, y| {
                    const d = try differingWords(a, &g, y, want.y);
                    std.debug.print("TURBOEXL3_LING_F16_PAIRS_COMBINE n={d} limit={d} call={s} ending={s} rows={d} outputs={d} sources={d} merged={d} sources_dtype={s} differing={d}/{d} {s}\n", .{ n, lim, c.name, if (plugin_dfin) "dfin" else "served", c.tokens, rows.items.len, want.sources, want.merged, label, d, @as(usize, c.tokens) * hidden, if (d == 0) "PASS" else "FAIL" });
                    if (d != 0) failed = true;
                }
                // red checks, once: on the small call at limit 0 of the first rate
                if (ri == 0 and lim == 0 and std.mem.eql(u8, c.name, "small")) {
                    reds_run += try redChecks(a, &g, s, n, xb, tb, &pf, ops, c, keep32.items, keep16.items, rows.items, wa, want.y, &failed);
                }
            }
        }
    }
    try testing.expect(reds_run == 4);
    try testing.expect(!failed);
}

/// The red checks (each must catch its fault; a miss fails the test): weights off 1 refused (and needed), a two-slot
/// call refused (and needed), a flipped f16 word caught by the combine's words. Returns the checks run.
fn redChecks(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, n: u32, xb: mlx.mlx_array, tb: root.Bank, pf: *prefill.Prefill, ops: prefill.Operands, c: *const CallPairs, outs32: []const mlx.mlx_array, outs16: []const mlx.mlx_array, rows: []const u32, wa: mlx.mlx_array, want_y: mlx.mlx_array, failed: *bool) !usize {
    const m = g.mark();
    defer g.resetTo(m);
    var run: usize = 0;
    const bd = c.bounds.items[0];
    const ca = c.s_act[bd[0]..bd[1]];
    const cs = c.s_slot[bd[0]..bd[1]];
    const o = try rt.rowsOps(g, ca, cs);
    if (!o.small or o.n_call != gemm_min_rows) return error.RedCallNotSmall;
    // 1. a weight other than 1 (or +0): refused before any launch, at 0.3, 0.5, -0 and 2
    {
        var refused = true;
        for ([_]f32{ 0.3, 0.5, -0.0, 2.0 }) |w| {
            var wt: [gemm_min_rows]f32 = undefined;
            const ws = hostWeights(ca.len, &wt);
            wt[1] = w;
            _ = api.moeLingRowsPairsF16(s, xb, o.rm, tb, o.ids, ws, .mul1, 0, .{}) catch |e| {
                if (e != error.Exl3PairsNotExact) refused = false;
                continue;
            };
            refused = false;
        }
        // ... and needed: the served f32 rows at weight 0.3 hold words off the f16 grid
        var wv: [gemm_min_rows]f32 = undefined;
        _ = hostWeights(ca.len, &wv);
        for (wv[0..ca.len]) |*w| w.* = 0.3;
        const sc = try g.hostArray(std.mem.sliceAsBytes(wv[0..o.n_call]), &.{ci(o.n_call)}, .float32);
        const y = try g.adopt(try pf.moeT128WithCall(s, xb, o.ids, sc, ops, o.n_call, .float32, o.rm, .{ .dfin = false }));
        const off = try notF16Words(a, g, y);
        const caught = refused and off > 0;
        std.debug.print("TURBOEXL3_LING_F16_PAIRS_RED n={d} fault=weight_not_1 refused={} f32_off_f16_grid_at_0.3={d}/{d} caught={}\n", .{ n, refused, off, mlx.mlx_array_size(y), caught });
        if (!caught) failed.* = true;
        run += 1;
    }
    // 2. a split of partials: a two-slot call (each row's finish sums two terms) is refused by the rule, and its f32
    // rows (`moeLing` at top-2, weight 1 each) hold words off the f16 grid
    {
        const toks: usize = 32;
        const ones2 = [_]f32{ 1, 1 };
        const refused = !api.pairsF16Exact(.{ .topk = 2, .weights = &ones2 });
        var ids: [toks * 2]u32 = undefined;
        var sc2: [toks * 2]f32 = undefined;
        var rr: [toks]u32 = undefined;
        for (0..toks) |t| {
            rr[t] = @intCast(t);
            ids[2 * t] = c.slot[t * top_k];
            ids[2 * t + 1] = c.slot[t * top_k + 1];
            sc2[2 * t] = 1;
            sc2[2 * t + 1] = 1;
        }
        const ra = try g.hostArray(std.mem.sliceAsBytes(&rr), &.{ci(toks)}, .uint32);
        const x3 = try g.reshape(try g.astype(try g.take(xb, ra, 0), .float32), &.{ 1, ci(toks), ci(hidden) });
        const ia = try g.hostArray(std.mem.sliceAsBytes(&ids), &.{ 1, ci(toks), 2 }, .uint32);
        const sa = try g.hostArray(std.mem.sliceAsBytes(&sc2), &.{ 1, ci(toks), 2 }, .float32);
        const y = try g.adopt(try root.moeLing(s, x3, tb, ia, sa, .mul1, 0));
        const off = try notF16Words(a, g, y);
        const caught = refused and off > 0;
        std.debug.print("TURBOEXL3_LING_F16_PAIRS_RED n={d} fault=two_slots_summed refused={} f32_off_f16_grid={d}/{d} caught={}\n", .{ n, refused, off, mlx.mlx_array_size(y), caught });
        if (!caught) failed.* = true;
        run += 1;
    }
    // 3. a flipped f16 word (its last bit, in the first source's first normal value) shows in the combine's words
    {
        const src0 = outs16[0];
        const size: usize = mlx.mlx_array_size(src0);
        const vals = try a.alloc(f32, size);
        defer a.free(vals);
        _ = try g.hostF32(try g.astype(src0, .float32), vals);
        const bits = try a.alloc(u16, size);
        defer a.free(bits);
        for (vals, bits) |v, *b| b.* = @bitCast(@as(f16, @floatCast(v)));
        const at = for (vals, 0..) |v, i| {
            if (std.math.isFinite(v) and @abs(v) >= 0x1p-14 and @abs(v) < 32768.0) break i;
        } else return error.RedNoNormalWord;
        bits[at] ^= 1;
        const flipped = try g.hostArray(std.mem.sliceAsBytes(bits), mlx.getShape(src0), .float16);
        const outs = try a.alloc(mlx.mlx_array, outs16.len);
        defer a.free(outs);
        @memcpy(outs, outs16);
        outs[0] = flipped;
        const y = try combineOf(a, g, s, outs, null, rows, c.order, c.tokens, wa);
        const d = try differingWords(a, g, y.y, want_y);
        const caught = d > 0;
        std.debug.print("TURBOEXL3_LING_F16_PAIRS_RED n={d} fault=flipped_f16_word differing={d}/{d} caught={}\n", .{ n, d, @as(usize, c.tokens) * hidden, caught });
        if (!caught) failed.* = true;
        run += 1;
    }
    // 4. a planted -0 in a source (f32 -0 and f16 0x8000) round-trips and gives the f32 source's combine; a value off the
    // f16 grid in an f32 source is caught by the round-trip check
    {
        const src0 = outs32[0];
        const size: usize = mlx.mlx_array_size(src0);
        const vals = try a.alloc(f32, size);
        defer a.free(vals);
        _ = try g.hostF32(src0, vals);
        vals[0] = -0.0;
        const bits = try a.alloc(u16, size);
        defer a.free(bits);
        for (vals, bits) |v, *b| b.* = @bitCast(@as(f16, @floatCast(v)));
        const p32 = try g.hostArray(std.mem.sliceAsBytes(vals), mlx.getShape(src0), .float32);
        const p16 = try g.hostArray(std.mem.sliceAsBytes(bits), mlx.getShape(src0), .float16);
        const o32 = try a.alloc(mlx.mlx_array, outs32.len);
        defer a.free(o32);
        @memcpy(o32, outs32);
        o32[0] = p32;
        const o16 = try a.alloc(mlx.mlx_array, outs16.len);
        defer a.free(o16);
        @memcpy(o16, outs16);
        o16[0] = p16;
        const y32 = try combineOf(a, g, s, o32, null, rows, c.order, c.tokens, wa);
        const y16 = try combineOf(a, g, s, o16, null, rows, c.order, c.tokens, wa);
        const d = try differingWords(a, g, y16.y, y32.y);
        const kz = try checkPairs(a, g, p16, p32);
        // a value off the f16 grid in the f32 rows: the round-trip check must count it
        vals[1] = 1.0 + 0x1p-20;
        const poff = try g.hostArray(std.mem.sliceAsBytes(vals), mlx.getShape(src0), .float32);
        const ko = try checkPairs(a, g, p16, poff);
        const caught = d == 0 and kz.differing == 0 and kz.not_f16 == 0 and bits[0] == 0x8000 and ko.not_f16 == 1;
        std.debug.print("TURBOEXL3_LING_F16_PAIRS_RED n={d} fault=planted_zero_and_off_grid neg_zero_combine_differing={d}/{d} neg_zero_round_trip_off={d} off_grid_counted={d} caught={}\n", .{ n, d, @as(usize, c.tokens) * hidden, kz.not_f16, ko.not_f16, caught });
        if (!caught) failed.* = true;
        run += 1;
    }
    return run;
}
