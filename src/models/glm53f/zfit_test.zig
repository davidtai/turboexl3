//! The mixed launch's outputs at the rows they hold (`prefill.Options.t128_zfit`, `Config.pf_t128_zfit`: the plugin's
//! TURBOEXL3_GLM53F_PF_ZFIT), at GLM-5.3-Flash's routed geometry:
//! - CPU: the rows `prefill.t128ZCap` gives a call hold every remainder the 128-row tiles' route leaves (its CPU rule,
//!   `prefill.t128Tables`), over every remainder edge and random calls; the option needs t128 and is off by default.
//! - GPU (TURBOEXL3_GPU=1): a prompt call on `root.moeClampedRows` with t128_zfit gives the words it gives without, with
//!   the call's experts named (`root.moeClampedRowsFit`: outputs of t128ZCap rows) and without (S rows), under the served
//!   options, bk128 and bk128.tk (the plugin's PF_OPTS values it serves or A/Bs; fused_test.zig holds those to the served
//!   words), 16384 row-slots in 4096-row calls over a drawn routing, 36 experts' 455-row runs and the remainder edges, at
//!   every SC rate. One line a rate, option set, routing and naming: TURBOEXL3_GLM_PF_ZFIT_BYTES (zcap_min the smallest
//!   call's rows: below 4096 where the outputs shrank).
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const prefill = @import("../../core/prefill.zig");
const config = @import("../../core/config.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const routed = @import("routed_test.zig");
const fused = @import("fused_test.zig");

const testing = std.testing;

const hidden = routed.hidden;
const inter = routed.inter;
const limit = routed.limit;

/// root.pfOptions' served set with the 128-row tiles (bm 64, the mixed launch, fragment-order double-buffered stage)
const served: prefill.Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .defer_affine = true, .ncb = 4, .sintr = true, .froute = true, .alay = 2, .t128 = true };

/// Every count an expert's rows meet the cut (64) and the 128-row tiles at
const edges = [_]u32{ 1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 455 };

fn ci(x: anytype) c_int {
    return @intCast(x);
}

test "turboexl3 glm53f prefill t128_zfit: every remainder the 128-row tiles' route leaves fits t128ZCap's rows for the call's experts (64 an expert, at most S); the option needs t128 and the config carries it off by default" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x7F17);
    const rnd = prng.random();
    // the remainder edges, then random calls of 1 to 288 experts with 0 to 600 rows each
    var counts: [288]u32 = undefined;
    for (0..201) |round| {
        const ne: usize = if (round == 0) edges.len else 1 + rnd.uintLessThan(usize, counts.len);
        for (counts[0..ne], 0..) |*c, i| c.* = if (round == 0) edges[i] else rnd.uintLessThan(u32, 601);
        var S: u32 = 0;
        var live: u32 = 0;
        for (counts[0..ne]) |c| {
            S += c;
            live += @intFromBool(c > 0);
        }
        if (S == 0) continue;
        var t = try prefill.t128Tables(a, counts[0..ne], served.bm, prefill.t128_cut, true);
        defer t.deinit(a);
        var mixed: u32 = 0;
        for (t.big.items) |b| mixed += b[2];
        for (t.small.items) |sm| mixed += sm[2];
        try testing.expect(mixed <= prefill.t128ZCap(S, live));
        try testing.expect(prefill.t128ZCap(S, live) <= S);
    }
    try testing.expectEqual(@as(u32, 4096), prefill.t128ZCap(4096, null));
    try testing.expectEqual(@as(u32, 640), prefill.t128ZCap(4096, 10));
    try testing.expectEqual(@as(u32, 1), prefill.t128ZCap(0, null));
    var o = served;
    o.t128_zfit = true;
    try testing.expect(o.valid());
    o.t128 = false;
    try testing.expect(!o.valid());
    const dc: config.Config = .{};
    try testing.expect(!dc.pf_t128_zfit);
    try testing.expect((config.Config{ .pf_t128_zfit = true }).valid());
}

/// The routed prefill options the plugin serves or A/Bs (its TURBOEXL3_GLM53F_PF_OPTS: unset, bk128, bk128.tk), each
/// checked with t128_zfit against itself without it
const ZOpts = struct { name: []const u8, bk128: bool, t128_bk128: bool };
const zopts = [_]ZOpts{ .{ .name = "served", .bk128 = false, .t128_bk128 = false }, .{ .name = "bk128", .bk128 = true, .t128_bk128 = false }, .{ .name = "bk128.tk", .bk128 = true, .t128_bk128 = true } };

/// `configure` with the options `o` and Config.pf_t128_zfit `on` (every other field `saved`'s), every engine released
/// first so the next call's prefill engine is made under them
fn zfitConfigure(saved: config.Config, o: ZOpts, on: bool) !void {
    root.releaseEngines();
    var c = saved;
    c.pf_bk128 = o.bk128;
    c.pf_t128_bk128 = o.t128_bk128;
    c.pf_t128_zfit = on;
    try root.configure(c);
}

/// `saved` configured again, every engine released first
fn zfitRestore(saved: config.Config) !void {
    root.releaseEngines();
    try root.configure(saved);
}

/// The distinct slots of a slot-sorted call: the plugin's count for `moeClampedRowsFit`
fn sortedDistinct(slots: []const u32) u32 {
    var d: u32 = 0;
    for (slots, 0..) |sl, i| d += @intFromBool(i == 0 or sl != slots[i - 1]);
    return d;
}

/// One prompt call on `moeClampedRows`, or with `call_experts` on `moeClampedRowsFit`, y bf16
fn zfitCall(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, bank: root.Bank, rows: []const u32, slots: []const u32, ones_h: []const f32, call_experts: ?u32) !mlx.mlx_array {
    const S = rows.len;
    const ones = try g.hostArray(std.mem.sliceAsBytes(ones_h[0..S]), &.{ci(S)}, .float32);
    const rows_a = try g.hostArray(std.mem.sliceAsBytes(rows), &.{ci(S)}, .uint32);
    const inds = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(S)}, .uint32);
    if (call_experts) |d| return g.adopt(try root.moeClampedRowsFit(s, x, rows_a, bank, inds, ones, .mul1, limit, .bfloat16, d));
    return g.adopt(try root.moeClampedRows(s, x, rows_a, bank, inds, ones, .mul1, limit, .bfloat16));
}

/// The routings a rate's calls take: a drawn routing (slot-sorted), 36 experts' 455-row runs, the remainder edges
const routings = [_][]const u8{ "drawn", "runs455", "edges" };

test "turboexl3 gpu glm53f prefill zfit: the mixed launch's outputs at the rows they hold give a prompt call's words on moeClampedRows, with the call's experts named (moeClampedRowsFit) and without, under the served options, bk128 and bk128.tk, 16384 row-slots in 4096-row calls over drawn, 455-row and remainder-edge routings, at every SC rate (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = config.get().*;
    defer zfitRestore(saved) catch unreachable;
    var nbuf: [8]u32 = undefined;
    const rates = routed.envList("TURBOEXL3_GLM_ROUTED_N", &routed.rates_n, &nbuf);
    const S: usize = 16384;
    const chunk: usize = 4096;
    // each routing's words without t128_zfit, then one call set's with it
    var want_bufs: [routings.len][]f32 = undefined;
    for (&want_bufs) |*w| w.* = try a.alloc(f32, S * hidden);
    defer for (want_bufs) |w| a.free(w);
    const got_buf = try a.alloc(f32, S * hidden);
    defer a.free(got_buf);
    const ones_h = try a.alloc(f32, S);
    defer a.free(ones_h);
    @memset(ones_h, 1.0);
    var rows_r: [routings.len][]u32 = undefined;
    var slots_r: [routings.len][]u32 = undefined;
    for (&rows_r, &slots_r) |*r, *sl| {
        r.* = try a.alloc(u32, S);
        sl.* = try a.alloc(u32, S);
    }
    defer for (rows_r, slots_r) |r, sl| {
        a.free(r);
        a.free(sl);
    };
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        defer root.releaseEngines();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x2F17 ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, fused.prompt_slots, hidden, inter, rate);
        defer bank.deinit(a);
        const rb = try routed.rootBank(&g, &bank, .uint16);
        const xh = try a.alloc(f32, fused.act_rows * hidden);
        defer a.free(xh);
        for (xh) |*v| v.* = routed.bfExact(fused.sigma * rnd.floatNorm(f32));
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(fused.act_rows), ci(hidden) }, .float32), .bfloat16);
        try g.evalAll(&.{x});
        // drawn (slot-sorted); 36 experts' 455-row runs; expert e taking edges[e % 14] rows in turn (98 experts)
        fused.drawCall(rnd, rows_r[0], slots_r[0], fused.prompt_slots, 0);
        try fused.sortBySlot(a, rows_r[0], slots_r[0], fused.prompt_slots);
        for (rows_r[1], slots_r[1], 0..) |*r, *sl, i| {
            r.* = rnd.uintLessThan(u32, fused.act_rows);
            // the run's index in u32 first: @min with a comptime bound narrows its type, and 35 * 2 needs 7 bits
            const run: u32 = @intCast(@min(i / 455, 35));
            sl.* = run * 2;
        }
        var e: u32 = 0;
        var left: u32 = edges[0];
        for (rows_r[2], slots_r[2]) |*r, *sl| {
            while (left == 0) {
                e += 1;
                left = edges[e % edges.len];
            }
            r.* = rnd.uintLessThan(u32, fused.act_rows);
            sl.* = @min(e, fused.prompt_slots - 1);
            left -= 1;
        }
        for (zopts) |o| {
            // the options' words without t128_zfit, then with it, the calls' experts named and not
            for ([_]u8{ 0, 1, 2 }) |mode| {
                try zfitConfigure(saved, o, mode != 0);
                for (routings, 0..) |routing, ri| {
                    const buf = if (mode == 0) want_bufs[ri] else got_buf;
                    var cap_min: u32 = std.math.maxInt(u32);
                    var c0: usize = 0;
                    while (c0 < S) : (c0 += chunk) {
                        const c1 = @min(c0 + chunk, S);
                        const m2 = g.mark();
                        defer g.resetTo(m2);
                        const named: ?u32 = if (mode == 1) sortedDistinct(slots_r[ri][c0..c1]) else null;
                        cap_min = @min(cap_min, prefill.t128ZCap(@intCast(c1 - c0), named));
                        const y = try zfitCall(&g, s, x, rb, rows_r[ri][c0..c1], slots_r[ri][c0..c1], ones_h, named);
                        _ = try fused.bitsOf(&g, y, buf[c0 * hidden .. c1 * hidden]);
                    }
                    if (mode == 0) continue;
                    const want: []const u32 = @as([*]const u32, @ptrCast(want_bufs[ri].ptr))[0 .. S * hidden];
                    const got: []const u32 = @as([*]const u32, @ptrCast(got_buf.ptr))[0 .. S * hidden];
                    const dd = fused.countDiff(got, want, got_buf);
                    const ok = dd[0] == 0 and dd[1] == 0;
                    std.debug.print("TURBOEXL3_GLM_PF_ZFIT_BYTES n={d} S={d} opts={s} routing={s} chunk={d} experts={s} zcap_min={d} against=zfit_off differing={d}/{d} nonfinite={d} {s}\n", .{ n, S, o.name, routing, chunk, if (mode == 1) "named" else "unnamed", cap_min, dd[0], S * hidden, dd[1], if (ok) "PASS" else "FAIL" });
                    if (!ok) failed = true;
                }
            }
        }
        try zfitRestore(saved);
    }
    try testing.expect(!failed);
}
