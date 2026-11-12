//! The routed prefill chain at GLM-5.3-Flash's routed geometry (hidden 4096, expert width 2048, `swiglu_limit` 10) in the
//! plugin's prompt calls, on `root.moeClampedRows`' row-map path, over two routings of a bank group: 36 experts' 455-row
//! runs (16,384 row-slots), and 48 experts at lognormal popularity (21,840 row-slots, sigma 1.0: the sc_3.10 pack's usage
//! profile has a per-layer median sigma of 0.97 over its log counts). Each routing is cut into calls of at most 4096
//! row-slots two ways: fixed (every 4096) and tile-aligned (`alignedEnd`, the plugin's TURBOEXL3_GLM53F_PF_ALIGN rule:
//! the latest point within 4096 rows that is an expert boundary or a multiple of 128 rows into the expert crossing it, so
//! no cut splits an expert's 128-row run).
//! - CPU: `alignedEnd`'s calls cover the row-slots in order, each at most the cap, each cut an expert boundary or a
//!   multiple of 128 rows into its expert, and no expert gains a part-full tile from a cut.
//! - GPU (TURBOEXL3_GPU=1): the aligned calls' y is the fixed calls' y byte for byte (a row's bytes do not depend on its
//!   call), at every SC rate, on `root.moeClampedRows` (TURBOEXL3_GLM_PF_ALIGN_BYTES, one line a rate and routing, then
//!   their total).
//! - TURBOEXL3_GLM_TIME=1 (with TURBOEXL3_GPU=1): at K3 on an engine with the plugin's control options (the served set and
//!   bk128): each routing and cut's calls with their 128-row tiles, part-full tiles and mixed rows counted on the CPU
//!   (TURBOEXL3_GLM_CHAIN_TIME); at the fixed cut, the chain's stages cumulatively (route, prepare, gate|up, mid, down,
//!   finish: TURBOEXL3_GLM_CHAIN_STAGE, each with its time over the stage before it) and the GEMM probes
//!   (`prefill.Options.probe` 1: no trellis read or decode; 4: no A loads; 8: no B loads; outputs wrong:
//!   TURBOEXL3_GLM_CHAIN_PROBE, each with the share of the served chain's time it removes). Every call's graph is built,
//!   then all are evaluated at once; min and median ms over 8 rounds after 2 warm. Steering only: it asserts nothing.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const prefill = @import("../../core/prefill.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const routed = @import("routed_test.zig");
const fused = @import("fused_test.zig");

const testing = std.testing;

const hidden = routed.hidden;
const inter = routed.inter;
const limit = routed.limit;

/// A prompt call's rows at most (the plugin's default `prompt_rows`, TURBOEXL3_GLM53F_PF_CALL_ROWS unset)
const cap: usize = 4096;
/// The routed prefill's 128-row tiles: an aligned cut inside an expert lands on a multiple of this many of its rows
const tile_rows: usize = 128;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// The end of the tile-aligned call from `c0` over slot-sorted row-slots `slots`, at most `max_rows` of them: `c0 +
/// max_rows` (or the end) where that is an expert boundary, else the latest point before it that is a multiple of 128 rows
/// into the expert crossing it (that expert's first row when it starts after c0). Every cut before keeps that rule, so the
/// expert a call starts in is cut at a multiple of 128 rows into it, and a call of 128 rows or more always moves.
pub fn alignedEnd(slots: []const u32, c0: usize, max_rows: usize) usize {
    const lim = @min(c0 + max_rows, slots.len);
    if (lim == slots.len or slots[lim - 1] != slots[lim]) return lim;
    var start = lim - 1;
    while (start > 0 and slots[start - 1] == slots[lim - 1]) start -= 1;
    const cut = start + (lim - start) / tile_rows * tile_rows;
    return if (cut > c0) cut else lim;
}

/// A cut's calls over `slots`: their ends, in order (fixed: every `max_rows`; aligned: `alignedEnd`)
fn cutCalls(a: std.mem.Allocator, slots: []const u32, max_rows: usize, aligned: bool) ![]usize {
    var ends: std.ArrayList(usize) = .empty;
    errdefer ends.deinit(a);
    var c0: usize = 0;
    while (c0 < slots.len) {
        const c1 = if (aligned) alignedEnd(slots, c0, max_rows) else @min(c0 + max_rows, slots.len);
        try ends.append(a, c1);
        c0 = c1;
    }
    return ends.toOwnedSlice(a);
}

/// The 128-row tile route's counts over a cut's calls (`text_froute_t128`'s rule: each expert's run in a call takes its
/// full 128-row tiles, one part-full tile for a remainder of more than `prefill.t128_cut` rows, the mixed launch for the
/// rest)
const TileCount = struct { full: usize = 0, part: usize = 0, mixed_rows: usize = 0 };
fn tileCount(slots: []const u32, ends: []const usize) TileCount {
    var tc: TileCount = .{};
    var c0: usize = 0;
    for (ends) |c1| {
        var i = c0;
        while (i < c1) {
            var j = i + 1;
            while (j < c1 and slots[j] == slots[i]) j += 1;
            const r = j - i;
            tc.full += r / tile_rows;
            const rem = r % tile_rows;
            if (rem > prefill.t128_cut) tc.part += 1 else tc.mixed_rows += rem;
            i = j;
        }
        c0 = c1;
    }
    return tc;
}

/// The two routings: `runs455` (36 experts' 455-row runs, the last 459: 16,384 row-slots) and `lognorm48` (48 experts,
/// counts in proportion to exp(N(0, 1)) draws, 21,840 row-slots), every expert two slots apart in the 100-slot bank, its
/// rows drawn from the activation's; slot-sorted, as the plugin sorts a call's rows
const Routing = struct { name: []const u8, rows: []u32, slots: []u32 };
fn makeRouting(a: std.mem.Allocator, rnd: std.Random, name: []const u8) !Routing {
    const lognorm = std.mem.eql(u8, name, "lognorm48");
    const experts: usize = if (lognorm) 48 else 36;
    const S: usize = if (lognorm) 48 * 455 else 16384;
    var counts: [48]usize = @splat(0);
    if (lognorm) {
        var w: [48]f64 = undefined;
        var sum: f64 = 0;
        for (&w) |*v| {
            v.* = @exp(rnd.floatNorm(f64));
            sum += v.*;
        }
        var given: usize = 0;
        for (counts[0..experts], w) |*c, v| {
            c.* = @intFromFloat(@floor(@as(f64, @floatFromInt(S)) * v / sum));
            given += c.*;
        }
        // the rounding's remainder one row each from the first expert on
        var e: usize = 0;
        while (given < S) : (e = (e + 1) % experts) {
            counts[e] += 1;
            given += 1;
        }
    } else {
        for (counts[0..experts]) |*c| c.* = 455;
        counts[experts - 1] += S - 455 * experts;
    }
    const rows = try a.alloc(u32, S);
    errdefer a.free(rows);
    const slots = try a.alloc(u32, S);
    var p: usize = 0;
    for (counts[0..experts], 0..) |c, e| for (0..c) |_| {
        rows[p] = rnd.uintLessThan(u32, fused.act_rows);
        slots[p] = @intCast(2 * e);
        p += 1;
    };
    std.debug.assert(p == S);
    return .{ .name = name, .rows = rows, .slots = slots };
}

test "turboexl3 glm53f chain cuts: tile-aligned calls cover the row-slots in order, each at most the cap, each cut an expert boundary or a multiple of 128 rows into its expert, and no cut adds a part-full tile" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA11C);
    const rnd = prng.random();
    for ([_][]const u8{ "runs455", "lognorm48" }) |name| {
        const rt = try makeRouting(a, rnd, name);
        defer a.free(rt.rows);
        defer a.free(rt.slots);
        const fixed = try cutCalls(a, rt.slots, cap, false);
        defer a.free(fixed);
        const al = try cutCalls(a, rt.slots, cap, true);
        defer a.free(al);
        var c0: usize = 0;
        for (al) |c1| {
            try testing.expect(c1 > c0 and c1 - c0 <= cap);
            if (c1 < rt.slots.len and rt.slots[c1 - 1] == rt.slots[c1]) {
                var start = c1 - 1;
                while (start > 0 and rt.slots[start - 1] == rt.slots[c1]) start -= 1;
                try testing.expect((c1 - start) % tile_rows == 0);
            }
            c0 = c1;
        }
        try testing.expectEqual(rt.slots.len, c0);
        // no expert loses a full tile or gains a part-full one: the counts are the whole run's, as in one call
        const whole = tileCount(rt.slots, &.{rt.slots.len});
        const t_al = tileCount(rt.slots, al);
        try testing.expectEqual(whole.full, t_al.full);
        try testing.expectEqual(whole.part, t_al.part);
        try testing.expectEqual(whole.mixed_rows, t_al.mixed_rows);
        try testing.expect(al.len >= fixed.len);
    }
    // edges: an expert crossing the cap is cut at the latest multiple of 128 rows into it (expert 1: 3968 rows in the first
    // call beside expert 0's 100, then 4096, then its last 636 with expert 2); a boundary at the cap is kept; the last call
    // is the rest
    var big: [9000]u32 = undefined;
    for (&big, 0..) |*v, i| v.* = if (i < 100) 0 else if (i < 8800) 1 else 2;
    const cuts = try cutCalls(a, &big, cap, true);
    defer a.free(cuts);
    try testing.expectEqualSlices(usize, &.{ 100 + 3968, 100 + 3968 + 4096, 9000 }, cuts);
    var exact: [8192]u32 = undefined;
    for (&exact, 0..) |*v, i| v.* = if (i < 4096) 3 else 4;
    const ex = try cutCalls(a, &exact, cap, true);
    defer a.free(ex);
    try testing.expectEqualSlices(usize, &.{ 4096, 8192 }, ex);
}

/// One call's y over `rows` / `slots` on `root.moeClampedRows`, bf16
fn rowsCall(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, bank: root.Bank, rows: []const u32, slots: []const u32, ones_h: []const f32) !mlx.mlx_array {
    const S = rows.len;
    const ones = try g.hostArray(std.mem.sliceAsBytes(ones_h[0..S]), &.{ci(S)}, .float32);
    const rows_a = try g.hostArray(std.mem.sliceAsBytes(rows), &.{ci(S)}, .uint32);
    const inds = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(S)}, .uint32);
    return g.adopt(try root.moeClampedRows(s, x, rows_a, bank, inds, ones, .mul1, limit, .bfloat16));
}

test "turboexl3 gpu glm53f chain cuts: a routing's tile-aligned calls give its fixed 4096-row calls' y byte for byte on moeClampedRows, over 455-row runs and a lognormal routing, at every SC rate (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = routed.envList("TURBOEXL3_GLM_ROUTED_N", &routed.rates_n, &nbuf);
    const smax: usize = 48 * 455;
    const want_buf = try a.alloc(f32, smax * hidden);
    defer a.free(want_buf);
    const got_buf = try a.alloc(f32, smax * hidden);
    defer a.free(got_buf);
    const ones_h = try a.alloc(f32, smax);
    defer a.free(ones_h);
    @memset(ones_h, 1.0);
    var lines: usize = 0;
    var fails: usize = 0;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        defer root.releaseEngines();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0xA11C0 ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, fused.prompt_slots, hidden, inter, rate);
        defer bank.deinit(a);
        const rb = try routed.rootBank(&g, &bank, .uint16);
        const xh = try a.alloc(f32, fused.act_rows * hidden);
        defer a.free(xh);
        for (xh) |*v| v.* = routed.bfExact(fused.sigma * rnd.floatNorm(f32));
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(fused.act_rows), ci(hidden) }, .float32), .bfloat16);
        try g.evalAll(&.{x});
        for ([_][]const u8{ "runs455", "lognorm48" }) |name| {
            const rt = try makeRouting(a, rnd, name);
            defer a.free(rt.rows);
            defer a.free(rt.slots);
            const S = rt.slots.len;
            var ncalls: [2]usize = undefined;
            for ([_]bool{ false, true }, 0..) |aligned, k| {
                const ends = try cutCalls(a, rt.slots, cap, aligned);
                defer a.free(ends);
                ncalls[k] = ends.len;
                const buf = if (aligned) got_buf else want_buf;
                var c0: usize = 0;
                for (ends) |c1| {
                    const m2 = g.mark();
                    defer g.resetTo(m2);
                    const y = try rowsCall(&g, s, x, rb, rt.rows[c0..c1], rt.slots[c0..c1], ones_h);
                    _ = try fused.bitsOf(&g, y, buf[c0 * hidden .. c1 * hidden]);
                    c0 = c1;
                }
            }
            const want: []const u32 = @as([*]const u32, @ptrCast(want_buf.ptr))[0 .. S * hidden];
            const got: []const u32 = @as([*]const u32, @ptrCast(got_buf.ptr))[0 .. S * hidden];
            const d = fused.countDiff(got, want, got_buf);
            const ok = d[0] == 0 and d[1] == 0;
            std.debug.print("TURBOEXL3_GLM_PF_ALIGN_BYTES n={d} routing={s} S={d} calls_fixed={d} calls_aligned={d} against=fixed differing={d}/{d} nonfinite={d} {s}\n", .{ n, name, S, ncalls[0], ncalls[1], d[0], S * hidden, d[1], if (ok) "PASS" else "FAIL" });
            lines += 1;
            if (!ok) fails += 1;
        }
    }
    std.debug.print("TURBOEXL3_GLM_PF_ALIGN_BYTES lines={d} fails={d} {s}\n", .{ lines, fails, if (fails == 0) "PASS" else "FAIL" });
    try testing.expect(fails == 0);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// The plugin's control options: root.pfOptions' served set (bm 64, the mixed launch, the fragment-order double-buffered
/// stage, the 128-row tiles) with bk128 (TURBOEXL3_GLM53F_PF_OPTS=bk128), and a timing probe (0: none)
fn controlOpts(probe: u8) prefill.Options {
    return .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .defer_affine = true, .ncb = 4, .sintr = true, .froute = true, .alay = 2, .t128 = true, .bk128 = true, .probe = probe };
}

/// The chain's stages, cumulatively: a call's arrays through `stage` (1 route, 2 the prepare, 3 gate|up, 4 mid, 5 down, 6
/// finish), adopted by `g`; its last stage's outputs in `out` (their count returned)
const stage_names = [_][]const u8{ "route", "prep", "gate_up", "mid", "down", "finish" };
fn chainStages(g: *G, pf: *prefill.Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, ops: prefill.Operands, rows: []const u32, slots: []const u32, ones_h: []const f32, stage: usize, out: *[4]mlx.mlx_array) !usize {
    const S = rows.len;
    const Su: u32 = @intCast(S);
    const ones = try g.hostArray(std.mem.sliceAsBytes(ones_h[0..S]), &.{ci(S)}, .float32);
    const rows_a = try g.hostArray(std.mem.sliceAsBytes(rows), &.{ci(S)}, .uint32);
    const inds = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(S)}, .uint32);
    var rt = try pf.routeT128(s, inds, Su);
    for ([_]mlx.mlx_array{ rt.order, rt.tiles, rt.tiles128, rt.xbase }) |r| _ = try g.adopt(r);
    rt.zcap = prefill.t128ZCap(Su, null);
    if (stage == 1) {
        out.* = .{ rt.order, rt.tiles, rt.tiles128, rt.xbase };
        return 4;
    }
    const po = try pf.prepWith(s, x, inds, rt.order, ops, Su, rows_a);
    for (po) |p| _ = try g.adopt(p);
    if (stage == 2) {
        out.* = .{ po[0], po[1], po[3], po[4] };
        return 4;
    }
    const zgu = try pf.gemmGUT128(s, rt, po[0], po[1], po[3], po[4], ops, Su);
    for (zgu.m[0..zgu.n]) |m| _ = try g.adopt(m);
    for (zgu.t[0..zgu.n]) |t| _ = try g.adopt(t);
    if (stage == 3) {
        out.* = .{ zgu.m[0], zgu.m[1], zgu.t[0], zgu.t[1] };
        return 4;
    }
    const xd = try pf.midT128(s, zgu, rt, inds, ops, Su);
    for (xd) |p| _ = try g.adopt(p);
    if (stage == 4) {
        out[0] = xd[0];
        out[1] = xd[1];
        return 2;
    }
    const zd = try pf.gemmDT128(s, rt, xd[0], xd[1], ops, Su);
    for (zd.m[0..zd.n]) |m| _ = try g.adopt(m);
    for (zd.t[0..zd.n]) |t| _ = try g.adopt(t);
    if (stage == 5) {
        out[0] = zd.m[0];
        out[1] = zd.t[0];
        return 2;
    }
    out[0] = try g.adopt(try pf.finishT128(s, zd, rt.xbase, inds, po[2], ones, ops, S, .bfloat16));
    return 1;
}

/// A routing's calls (`ends`) built through `stage` on `pf` and evaluated at once, 2 warm rounds then 8 timed: min and
/// median ms
fn timeCalls(g: *G, pf: *prefill.Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, ops: prefill.Operands, rt: Routing, ends: []const usize, ones_h: []const f32, stage: usize) ![2]f64 {
    var t: [8]f64 = undefined;
    for (0..t.len + 2) |round| {
        const m = g.mark();
        defer g.resetTo(m);
        var outs: [64]mlx.mlx_array = undefined;
        var no: usize = 0;
        const t0 = nowNs();
        var c0: usize = 0;
        for (ends) |c1| {
            var o: [4]mlx.mlx_array = undefined;
            const k = try chainStages(g, pf, s, x, ops, rt.rows[c0..c1], rt.slots[c0..c1], ones_h, stage, &o);
            @memcpy(outs[no..][0..k], o[0..k]);
            no += k;
            c0 = c1;
        }
        try g.evalAll(outs[0..no]);
        const dt = nowNs() - t0;
        if (round >= 2) t[round - 2] = @as(f64, @floatFromInt(dt)) / 1e6;
    }
    std.mem.sort(f64, &t, {}, std.sort.asc(f64));
    return .{ t[0], t[t.len / 2] };
}

test "turboexl3 gpu glm53f chain timing: the plugin's control chain over 455-row runs and a lognormal routing, fixed and tile-aligned 4096-row calls, its stages cumulatively and the GEMM probes (TURBOEXL3_GLM_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_GLM_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    var prng = std.Random.DefaultPrng.init(0xC4A1);
    const rnd = prng.random();
    // K3, the sc_3.10bpw pack's routed layers
    const n: u32 = 48;
    const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
    const mark = g.mark();
    defer g.resetTo(mark);
    var bank = try ref.synthBank(a, rnd, fused.prompt_slots, hidden, inter, rate);
    defer bank.deinit(a);
    const rb = try routed.rootBank(&g, &bank, .uint16);
    const ops: prefill.Operands = .{ .gate_t = rb.gate.trellis, .gate_suh = rb.gate.suh, .gate_svh = rb.gate.svh, .up_t = rb.up.trellis, .up_suh = rb.up.suh, .up_svh = rb.up.svh, .down_t = rb.down.trellis, .down_suh = rb.down.suh, .down_svh = rb.down.svh };
    const xh = try a.alloc(f32, fused.act_rows * hidden);
    defer a.free(xh);
    for (xh) |*v| v.* = routed.bfExact(fused.sigma * rnd.floatNorm(f32));
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(fused.act_rows), ci(hidden) }, .float32), .bfloat16);
    try g.evalAll(&.{x});
    const smax: usize = 48 * 455;
    const ones_h = try a.alloc(f32, smax);
    defer a.free(ones_h);
    @memset(ones_h, 1.0);
    const geo: prefill.Geometry = .{ .hidden = hidden, .inter = inter, .topk = 1, .experts = fused.prompt_slots, .rate = rate, .dec = .mul1, .limit = @intCast(limit) };
    var pf = try prefill.Prefill.init(geo, controlOpts(0));
    defer pf.deinit();
    for ([_][]const u8{ "runs455", "lognorm48" }) |name| {
        const rt = try makeRouting(a, rnd, name);
        defer a.free(rt.rows);
        defer a.free(rt.slots);
        const S = rt.slots.len;
        const flops = 2.0 * 3.0 * @as(f64, @floatFromInt(S)) * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter));
        const fixed = try cutCalls(a, rt.slots, cap, false);
        defer a.free(fixed);
        var served: [2]f64 = undefined;
        // each cut's calls through the whole chain, with their tiles
        for ([_]bool{ false, true }) |aligned| {
            const ends = if (aligned) try cutCalls(a, rt.slots, cap, true) else fixed;
            defer if (aligned) a.free(ends);
            const tc = tileCount(rt.slots, ends);
            const t = try timeCalls(&g, &pf, s, x, ops, rt, ends, ones_h, stage_names.len);
            if (!aligned) served = t;
            std.debug.print("TURBOEXL3_GLM_CHAIN_TIME n={d} routing={s} S={d} cut={s} calls={d} tiles128={d} part128={d} mixed_rows={d} min_ms={d:.2} med_ms={d:.2} tflops={d:.2}\n", .{ n, name, S, if (aligned) "aligned" else "fixed", ends.len, tc.full, tc.part, tc.mixed_rows, t[0], t[1], flops / (t[0] * 1e9) });
        }
        // the stages, cumulatively, at the fixed cut
        var prev: f64 = 0;
        for (stage_names, 1..) |sn, k| {
            const t = try timeCalls(&g, &pf, s, x, ops, rt, fixed, ones_h, k);
            std.debug.print("TURBOEXL3_GLM_CHAIN_STAGE n={d} routing={s} S={d} cut=fixed calls={d} through={s} min_ms={d:.2} med_ms={d:.2} stage_ms={d:.2}\n", .{ n, name, S, fixed.len, sn, t[0], t[1], t[0] - prev });
            prev = t[0];
        }
        // the GEMM probes at the fixed cut (outputs wrong; timing only)
        for ([_]u8{ 1, 4, 8 }) |probe| {
            var pp = try prefill.Prefill.init(geo, controlOpts(probe));
            defer pp.deinit();
            const t = try timeCalls(&g, &pp, s, x, ops, rt, fixed, ones_h, stage_names.len);
            std.debug.print("TURBOEXL3_GLM_CHAIN_PROBE n={d} routing={s} S={d} cut=fixed probe={d} min_ms={d:.2} med_ms={d:.2} served_min_ms={d:.2} share={d:.1}%\n", .{ n, name, S, probe, t[0], t[1], served[0], 100.0 * (served[0] - t[0]) / served[0] });
        }
    }
}
