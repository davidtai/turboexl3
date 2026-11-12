//! wave3 (`Config.banked_wave3`; models/deepseek_v41/banked_wave3.zig, the plugin's TURBOEXL3_GLM53F_BANKED_WAVE3): the
//! whole banked call in three launches where five run, at GLM-5.3-Flash's routed geometry (hidden 4096, expert width
//! 2048, `swiglu_limit` 10, MUL1) over three slot banks of 20, 12 and 8 experts (routed_test.zig's):
//! - CPU (always): the option is off by default and needs `banked_group` 4; and a finish unit's arrivals are exactly the
//!   count its text waits for. At every row-slot count 1 to 64, over one id at every row-slot, every row-slot skipped and
//!   drawn waves (a verify's rows, repeats, skipped row-slots), and at every tile count and K threadgroup count the texts
//!   take, the launch grids (each lead's column threadgroups over the width and its K threadgroups, one count a member at
//!   its unit; the leads and members `turboexl3_members_g`'s) give each unit of a row-slot that runs (8 / BT) x (KS /
//!   SPT) counts, a skipped row-slot's units none, and every count lands among the zeroed counters.
//! - GPU (TURBOEXL3_GPU=1, else skipped): at every SC rate (K2, K2.5, K3, K3.5, K4, K5; TURBOEXL3_GLM_ROUTED_N narrows
//!   them) and 13 waves (one to four decode and verify rows of top-8: 8, 16, 24 and 32 pairs; 1, 2, 7, 48 and 64 pairs;
//!   1, 8, 24 and 32 pairs with 1, 2, 5 and 32 row-slots skipped), wave3's call gives the five launches' words:
//!   - selected per call (`deepseek_v41.moeClampedBankedWave`, wave3 against the five launches on the same texts and
//!     variants): hw4b's texts (g4m3) and today's (g4) at the grouped GEMV variants t4.t4 (the served schedule's) and
//!     0.0, y f32 and bf16, and the gate|up's hd and hs; a skipped row-slot's y row 0 in both, its hd and hs rows (left
//!     unwritten by both) not compared;
//!   - configured as the plugin configures it (`banked_group` 4, `banked_m3`, t4.t4 and `banked_wave3`, a geometry's
//!     banked state made after it): `root.moeClampedBanked` (y f32) and `root.moeClampedBankedAs` at bf16, against the
//!     five launches' y of the same schedule, each call counted on the geometry's wave3 state (so it ran the three).
//!   One line a rate, wave, route, texts, variants and output dtype, then the total: TURBOEXL3_GLM_WAVE3_BYTES (130 lines
//!   a rate: 780 at the six rates, and the total line).
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const config = @import("../../core/config.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const routed = @import("routed_test.zig");
const fused = @import("fused_test.zig");

const testing = std.testing;

const hidden = routed.hidden;
const inter = routed.inter;
const limit = routed.limit;
const skip_id = root.banked_skip_id;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// A wave: `pairs` row-slots, `skips` of them skipped (drawn among them)
const Wave = struct { pairs: u32, skips: u32 };
/// One to four decode and verify rows of top-8, drawn as a verify's rows route (the grouped texts' members: one to four
/// of an expert); 1 pair (GROUP 1), 2 and 7 (drawn with repeats), 48 and 64 (past four rows: two leads an expert); and
/// skipped row-slots: the lone row-slot, a few, every one
const waves = [_]Wave{
    .{ .pairs = 8, .skips = 0 },  .{ .pairs = 16, .skips = 0 }, .{ .pairs = 24, .skips = 0 }, .{ .pairs = 32, .skips = 0 },
    .{ .pairs = 1, .skips = 0 },  .{ .pairs = 2, .skips = 0 },  .{ .pairs = 7, .skips = 0 },  .{ .pairs = 48, .skips = 0 },
    .{ .pairs = 64, .skips = 0 }, .{ .pairs = 1, .skips = 1 },  .{ .pairs = 8, .skips = 2 },  .{ .pairs = 24, .skips = 5 },
    .{ .pairs = 32, .skips = 32 },
};

/// A wave's packed ids (bank << 24 | slot row) over the three banks: at a multiple of 8 pairs a verify's rows (8
/// distinct experts a row, each pick its previous row's at that position half the time, a fresh one in the base bank 7
/// times in 10, the ext 2 and the transient 1: fused_test.zig's waves), at another count drawn with repeats
/// (`routed.drawPairs`); then `skips` row-slots, drawn, skipped (`banked_skip_id`; `skipped` marks them).
fn drawIds(rnd: std.Random, ids: []u32, skips: u32, skipped: []bool) void {
    const n = ids.len;
    if (n % 8 != 0) {
        var bank_of: [root.banked_rows_max]u32 = undefined;
        var slot_of: [root.banked_rows_max]u32 = undefined;
        routed.drawPairs(rnd, bank_of[0..n], slot_of[0..n]);
        for (ids, bank_of[0..n], slot_of[0..n]) |*d, b, sl| d.* = (b << 24) | sl;
    } else {
        for (0..n / 8) |r| {
            const row = ids[r * 8 ..][0..8];
            for (row, 0..) |*d, j| {
                while (true) {
                    const cand: u32 = if (r > 0 and rnd.boolean()) ids[(r - 1) * 8 + j] else blk: {
                        const u = rnd.uintLessThan(u32, 10);
                        const bk: u32 = if (u < 7) 0 else if (u < 9) 1 else 2;
                        break :blk (bk << 24) | rnd.uintLessThan(u32, routed.bank_experts[bk]);
                    };
                    if (std.mem.indexOfScalar(u32, row[0..j], cand) == null) {
                        d.* = cand;
                        break;
                    }
                }
            }
        }
    }
    @memset(skipped, false);
    var left = @min(skips, @as(u32, @intCast(n)));
    while (left > 0) {
        const r = rnd.uintLessThan(usize, n);
        if (skipped[r]) continue;
        skipped[r] = true;
        ids[r] = skip_id;
        left -= 1;
    }
}

/// The members `turboexl3_members_g` gives row-slot r (gemv_group.metal): none where r is skipped or leads nothing (its
/// occurrence index of its id not a multiple of `grp`), else r and the next grp - 1 row-slots of its id in slot order
fn membersOf(ids: []const u32, r: usize, grp: u32, out: *[4]u32) u32 {
    const e = ids[r];
    if (e == skip_id) return 0;
    var before: u32 = 0;
    for (ids[0..r]) |v| before += @intFromBool(v == e);
    if (before % grp != 0) return 0;
    out[0] = @intCast(r);
    var m: u32 = 1;
    for (ids[r + 1 ..], r + 1..) |v, i| {
        if (m == grp) break;
        if (v != e) continue;
        out[m] = @intCast(i);
        m += 1;
    }
    return m;
}

/// wave3's arrivals at one launch, as its grid makes them: each lead's column threadgroups over `width` (width / (16 bt))
/// at each of its `z` K threadgroups count one arrival a member at the member's unit (the column threadgroup's 128-wide
/// block, the member's row-slot), at counter `base` + row-slot x (width / 128) + block (the texts' index). Whether every
/// unit of a row-slot that runs got (8 / bt) x z counts (the texts' expected arrivals, KS / SPT being z), a skipped
/// row-slot's none, and every counter lies below `ncnt` (the zeroed ones).
fn arrivalsExact(a: std.mem.Allocator, ids: []const u32, width: u32, bt: u32, z: u32, base: u32, ncnt: u32) !bool {
    const rows = ids.len;
    const nb = width / 128;
    const grp: u32 = if (rows >= 2) 4 else 1;
    const counts = try a.alloc(u32, rows * nb);
    defer a.free(counts);
    @memset(counts, 0);
    for (0..rows) |r| {
        var members: [4]u32 = undefined;
        const m = membersOf(ids, r, grp, &members);
        for (0..width / (16 * bt)) |x| for (0..z) |_| for (members[0..m]) |mr| {
            const idx = mr * nb + @as(u32, @intCast(x)) / (8 / bt);
            if (base + idx >= ncnt) return false;
            counts[idx] += 1;
        };
    }
    for (0..rows) |r| for (0..nb) |b| {
        const want: u32 = if (ids[r] == skip_id) 0 else (8 / bt) * z;
        if (counts[r * nb + b] != want) return false;
    };
    return true;
}

test "turboexl3 glm53f wave3: the option is off by default and needs banked_group 4; a finish unit's arrivals are exactly the count its text waits for, a skipped row-slot's none, every counter among the zeroed ones" {
    const t4 = root.groupVariantOf("t4").?;
    const d: config.Config = .{};
    try testing.expect(!d.banked_wave3);
    try testing.expect(d.valid());
    try testing.expect((config.Config{ .banked_group = 4, .banked_wave3 = true }).valid());
    try testing.expect((config.Config{ .banked_group = 4, .banked_m3 = true, .banked_wave3 = true, .banked_variant_gu = t4, .banked_variant_dn = t4 }).valid());
    try testing.expect(!(config.Config{ .banked_wave3 = true }).valid());
    try testing.expect(!(config.Config{ .banked_group = 2, .banked_m3 = false, .banked_wave3 = true }).valid());
    // beside the persistent grouped GEMVs (`banked_ptg`, where this build has that field) it is refused: wave3 runs GEMV
    // grids of its own
    if (@hasField(config.Config, "banked_ptg")) {
        var both: config.Config = .{ .banked_group = 4, .banked_wave3 = true };
        @field(both, "banked_ptg") = true;
        try testing.expect(!both.valid());
        @field(both, "banked_wave3") = false;
        try testing.expect(both.valid());
    }
    var prng = std.Random.DefaultPrng.init(0x3A7E3);
    const rnd = prng.random();
    var ids: [root.banked_rows_max]u32 = undefined;
    var skipped: [root.banked_rows_max]bool = undefined;
    var checked: usize = 0;
    for (1..root.banked_rows_max + 1) |rows| {
        const r_ids = ids[0..rows];
        const rows_u: u32 = @intCast(rows);
        const ncnt: u32 = @max(rows_u * (inter / 128 + hidden / 128), 8);
        for (0..10) |trial| {
            // one id at every row-slot (a lead every fourth), every row-slot skipped, then drawn waves with up to a third
            // of their row-slots skipped
            switch (trial) {
                0 => @memset(r_ids, 5),
                1 => @memset(r_ids, skip_id),
                else => drawIds(rnd, r_ids, rnd.uintAtMost(u32, rows_u / 3), skipped[0..rows]),
            }
            for ([_]u32{ 1, 2, 4, 8 }) |bt| for ([_]u32{ 1, 2, 4, 8 }) |z| {
                try testing.expect(try arrivalsExact(testing.allocator, r_ids, inter, bt, z, 0, ncnt));
                try testing.expect(try arrivalsExact(testing.allocator, r_ids, hidden, bt, z, rows_u * (inter / 128), ncnt));
                checked += 2;
            };
        }
    }
    try testing.expectEqual(@as(usize, 64 * 10 * 16 * 2), checked);
}

/// The words of `got` against `want` (`row_w` a row): on a row a row-slot ran, the differing ones and got's non-finite
/// values (`vals`); on a skipped row (`skipped`), with `zero_skipped` (y's: 0 in both) the differing ones and the nonzero
/// words of either, else none (hd's and hs's rows, unwritten in both)
const Diff = struct { words: usize = 0, differing: usize = 0, nonfinite: usize = 0, skipped_nonzero: usize = 0 };
fn wordDiff(got: []const u32, want: []const u32, vals: []const f32, row_w: usize, skipped: []const bool, zero_skipped: bool) Diff {
    var d: Diff = .{};
    for (got, want, vals, 0..) |u, v, f, i| {
        const sk = skipped[i / row_w];
        if (sk and !zero_skipped) continue;
        d.words += 1;
        d.differing += @intFromBool(u != v);
        d.nonfinite += @intFromBool(!std.math.isFinite(f));
        if (sk) d.skipped_nonzero += @as(usize, @intFromBool(u != 0)) + @as(usize, @intFromBool(v != 0));
    }
    return d;
}

/// A grouped GEMV variant's name for a line (`0` today's)
fn varName(v: u8) []const u8 {
    return if (v == 0) "0" else root.group_variants[v].name;
}

/// A y dtype's name for a line
fn outName(dt: mlx.mlx_dtype) []const u8 {
    return if (dt == .float32) "f32" else "bf16";
}

test "turboexl3 gpu glm53f wave3: the whole banked call in three launches gives the five launches' words (y, hd, hs; a skipped row-slot's y row 0), selected (g4m3 and g4 at t4.t4 and 0.0) and configured as the plugin configures it (g4m3.t4.t4), y f32 and bf16, at GLM-5.3-Flash's routed geometry, every SC rate, 1 to 4 rows of top-8 and other waves (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = root.config.get().*;
    defer root.configure(saved) catch unreachable;
    var nbuf: [8]u32 = undefined;
    const rates = routed.envList("TURBOEXL3_GLM_ROUTED_N", &routed.rates_n, &nbuf);
    const t4 = root.groupVariantOf("t4").?;
    const Arm = struct { sel: root.BankedSel, gu: u8, dn: u8 };
    const arms = [_]Arm{ .{ .sel = .g4m3, .gu = t4, .dn = t4 }, .{ .sel = .g4, .gu = t4, .dn = t4 }, .{ .sel = .g4m3, .gu = 0, .dn = 0 }, .{ .sel = .g4, .gu = 0, .dn = 0 } };
    const dtypes = [_]mlx.mlx_dtype{ .float32, .bfloat16 };
    const max_y = root.banked_rows_max * hidden;
    const max_h = root.banked_rows_max * inter;
    const max_s = root.banked_rows_max * (inter / 128);
    var bufs: [6][]f32 = undefined;
    for (&bufs, [_]usize{ max_y, max_y, max_h, max_h, max_s, max_s }) |*b, len| b.* = try a.alloc(f32, len);
    defer for (bufs) |b| a.free(b);
    var lines: usize = 0;
    var fails: usize = 0;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        defer root.releaseEngines();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const k: f32 = @as(f32, @floatFromInt(n)) / 16.0;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x3A7E3 ^ @as(u64, n));
        const rnd = prng.random();
        var hb: [3]ref.Bank = undefined;
        var made: usize = 0;
        defer for (hb[0..made]) |*b| b.deinit(a);
        for (&hb, routed.bank_experts) |*b, e| {
            b.* = try ref.synthBank(a, rnd, e, hidden, inter, rate);
            made += 1;
        }
        var rb: [3]root.Bank = undefined;
        for (&hb, &rb) |*b, *r| r.* = try routed.rootBank(&g, b, .int16);
        // each wave's inputs, and the served schedule's five launches' y at each output dtype (the configured route's
        // reference), kept to the rate's end
        const Case = struct { w: Wave, x: mlx.mlx_array, ids: mlx.mlx_array, skipped: [root.banked_rows_max]bool, want: [2]mlx.mlx_array };
        var cases: [waves.len]Case = undefined;
        for (waves, &cases) |w, *c| {
            const xh = try a.alloc(f32, w.pairs * hidden);
            defer a.free(xh);
            for (xh) |*v| v.* = routed.bfExact(fused.sigma * rnd.floatNorm(f32));
            var ids: [root.banked_rows_max]u32 = undefined;
            c.w = w;
            c.skipped = @splat(false);
            drawIds(rnd, ids[0..w.pairs], w.skips, c.skipped[0..w.pairs]);
            c.x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(w.pairs), ci(hidden) }, .float32), .bfloat16);
            c.ids = try g.hostArray(std.mem.sliceAsBytes(ids[0..w.pairs]), &.{ci(w.pairs)}, .uint32);
            for (dtypes, &c.want) |dt, *wy| {
                const o = try root.deepseek_v41.moeClampedBankedWave(s, c.x, &rb, c.ids, .mul1, limit, .g4m3, t4, t4, dt, false);
                _ = try g.adopt(o.hd);
                _ = try g.adopt(o.hs);
                wy.* = try g.adopt(o.y);
            }
            try g.evalAll(&.{ c.x, c.ids, c.want[0], c.want[1] });
        }
        // selected per call: wave3's three launches against the five on each texts and variants
        for (&cases) |*c| for (arms) |arm| for (dtypes) |dt| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            const P = c.w.pairs;
            const geo_calls = try wave3Calls(rate, dt);
            const r = try root.deepseek_v41.moeClampedBankedWave(s, c.x, &rb, c.ids, .mul1, limit, arm.sel, arm.gu, arm.dn, dt, false);
            const want = [3]mlx.mlx_array{ try g.adopt(r.y), try g.adopt(r.hd), try g.adopt(r.hs) };
            const o = try root.deepseek_v41.moeClampedBankedWave(s, c.x, &rb, c.ids, .mul1, limit, arm.sel, arm.gu, arm.dn, dt, true);
            const got = [3]mlx.mlx_array{ try g.adopt(o.y), try g.adopt(o.hd), try g.adopt(o.hs) };
            try testing.expectEqual(dt, mlx.mlx_array_dtype(got[0]));
            const ran = (try wave3Calls(rate, dt)) - geo_calls;
            var dy: Diff = .{};
            var dh: Diff = .{};
            for (want, got, [_]usize{ hidden, inter, inter / 128 }, 0..) |wa, ga, row_w, i| {
                const nw = P * row_w;
                const wb = try fused.bitsOf(&g, wa, bufs[2 * i][0..nw]);
                const gb = try fused.bitsOf(&g, ga, bufs[2 * i + 1][0..nw]);
                const df = wordDiff(gb, wb, bufs[2 * i + 1][0..nw], row_w, c.skipped[0..P], i == 0);
                if (i == 0) dy = df else {
                    dh.words += df.words;
                    dh.differing += df.differing;
                }
            }
            const ok = dy.differing == 0 and dh.differing == 0 and dy.nonfinite == 0 and dy.skipped_nonzero == 0 and ran == 1;
            lines += 1;
            fails += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_GLM_WAVE3_BYTES k={d} pairs={d} skipped={d} route=selected texts={t} gu={s} dn={s} out={s} differing={d}/{d} hd_hs_differing={d}/{d} nonfinite={d} skipped_nonzero={d} wave3_calls={d} {s}\n", .{ k, P, c.w.skips, arm.sel, varName(arm.gu), varName(arm.dn), outName(dt), dy.differing, dy.words, dh.differing, dh.words, dy.nonfinite, dy.skipped_nonzero, ran, if (ok) "PASS" else "FAIL" });
        };
        // configured, as the plugin sets it before its first banked call (a geometry's banked state made after it): the
        // configured call's y at f32 (moeClampedBanked) and at bf16 (moeClampedBankedAs), each against the five launches'
        root.releaseEngines();
        var cfg = saved;
        cfg.banked_group = 4;
        cfg.banked_m3 = true;
        cfg.banked_variant_gu = t4;
        cfg.banked_variant_dn = t4;
        cfg.banked_wave3 = true;
        try root.configure(cfg);
        defer {
            root.configure(saved) catch unreachable;
            root.releaseEngines();
        }
        for (&cases) |*c| for (dtypes, c.want) |dt, wy| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            const P = c.w.pairs;
            const geo_calls = try wave3Calls(rate, dt);
            const y = try g.adopt(if (dt == .float32)
                try root.moeClampedBanked(s, c.x, &rb, c.ids, .mul1, limit)
            else
                try root.moeClampedBankedAs(s, c.x, &rb, c.ids, .mul1, limit, .bfloat16));
            try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
            const ran = (try wave3Calls(rate, dt)) - geo_calls;
            const wb = try fused.bitsOf(&g, wy, bufs[0][0 .. P * hidden]);
            const gb = try fused.bitsOf(&g, y, bufs[1][0 .. P * hidden]);
            const dy = wordDiff(gb, wb, bufs[1][0 .. P * hidden], hidden, c.skipped[0..P], true);
            const ok = dy.differing == 0 and dy.nonfinite == 0 and dy.skipped_nonzero == 0 and ran == 1;
            lines += 1;
            fails += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_GLM_WAVE3_BYTES k={d} pairs={d} skipped={d} route=configured texts=g4m3 gu=t4 dn=t4 out={s} differing={d}/{d} hd_hs_differing=0/0 nonfinite={d} skipped_nonzero={d} wave3_calls={d} {s}\n", .{ k, P, c.w.skips, outName(dt), dy.differing, dy.words, dy.nonfinite, dy.skipped_nonzero, ran, if (ok) "PASS" else "FAIL" });
        };
    }
    std.debug.print("TURBOEXL3_GLM_WAVE3_BYTES lines={d} fails={d} {s}\n", .{ lines, fails, if (fails == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 0), fails);
    try testing.expectEqual(rates.len * waves.len * (arms.len + 1) * dtypes.len, lines);
}

/// The wave3 calls run so far on the banked state of the banked call's geometry at `rate` and y dtype `dt` (top-1, MUL1,
/// GLM's clamp: `root.kernelsFor`'s Kernels3, made here if no call made it yet)
fn wave3Calls(rate: format.Rate, dt: mlx.mlx_dtype) !u64 {
    const k3 = try root.kernelsFor(.{ .hidden = hidden, .inter = inter, .topk = 1, .rate = rate, .dec = .mul1, .limit = @intCast(limit) }, dt);
    return (try root.banked.forKernels(k3)).wave3.calls;
}
