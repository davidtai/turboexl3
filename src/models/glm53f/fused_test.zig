//! GLM-5.3-Flash's fused paths on turboexl3, each the opt-in switch of the outside plugin turboexl3_glm53f, and each held
//! byte for byte to the chain of launches it replaces, at GLM-5.3-Flash's routed geometry (hidden 4096, expert width 2048,
//! `swiglu_limit` 10) and every routed rate of the SC packs:
//! - `root.moeClampedRows` (the plugin's TURBOEXL3_GLM53F_PF_ROWS): a prompt call over an expert streamer's slot bank, row
//!   p of x (bf16 [n, hidden], the layer's activation) through slot `inds[p]`, the gather in the chain's prepare
//!   (`prefill.text_prep_rows`) and y stored bf16 by the chain's finish. The chain it replaces is the plugin's prompt
//!   call today: `take(x, rows)`, a cast to f32, `moeClamped` at top-1 over [1, S, hidden], a cast of its f32 y to bf16.
//! - `root.moeClampedBankedAs` at bf16 (the plugin's TURBOEXL3_GLM53F_DEC_BF16): a decode wave over the three slot banks with
//!   y stored bf16 by the banked down finish. The chain it replaces: `moeClampedBanked` (y f32), then a cast to bf16.
//! - CPU (always): the row-map prepare's text is the prepare's with its one row line swapped for the row map's, and the
//!   prepare's own text is unchanged by the cut.
//! - GPU (TURBOEXL3_GPU=1, else skipped): the bytes above, x ~ N(0, 8^2) bf16-exact (about a tenth of the gates and a
//!   quarter of the ups pass the limit). TURBOEXL3_GLM_ROUTED_N (rates in halfwords a tile, a comma list; default
//!   32,40,48,56,64,80) narrows both; TURBOEXL3_GLM_ROWS_S (default 17,64,1328,4096) the prompt calls' row-slots,
//!   TURBOEXL3_GLM_BANKED_ROWS (default 1,2,7,8,16,32,48,64) the decode waves' pairs. Each rate's engines are freed after it.
//!   Lines: TURBOEXL3_GLM_FUSED_ROWS_BYTES, TURBOEXL3_GLM_FUSED_BANKED_BYTES.
//! - TURBOEXL3_GLM_TIME=1 (with TURBOEXL3_GPU=1, else skipped): each fused path against its chain on the device at the
//!   served shapes (a 4096-row prompt call over 9 experts of a 100-slot bank; a 4-row verify's 32 pairs), and every
//!   sc_3.10bpw trunk linear (K6, `root.linear`) at decode rows 1 and 4 and a 16384-row prompt chunk with its count in a
//!   forward (the trunk's share of a forward: each linear alone, so launch overlap and the forward's other ops are not in
//!   it); min and median us per call over 8 rounds after 2 warm. It asserts nothing (a microbenchmark steers, the standard
//!   test decides). Lines: TURBOEXL3_GLM_FUSED_TIME, TURBOEXL3_GLM_TRUNK_TIME, TURBOEXL3_GLM_TRUNK_TOTAL.
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

const testing = std.testing;

const hidden = routed.hidden;
const inter = routed.inter;
const limit = routed.limit;
const sigma: f32 = 8;
/// The prompt calls' slot bank: a base bank's rows (the plugin serves 100 to 145 prompt rows a layer at the 110 GiB ceiling)
const prompt_slots: u32 = 100;
/// The layer activation's rows the prompt calls gather from
const act_rows: u32 = 600;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

test "turboexl3 glm53f fused texts: the row-map prepare is the prepare with its row line swapped for the row map's, nothing else; the prepare's text is its pieces" {
    const pin = comptime support.pins(@embedFile("kernels/test/pins/fused_texts.metal"));
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

/// `x`'s 32-bit words (an f32 array's bits, or a bf16 array's bits through its exact f32 widening).
fn bitsOf(g: *G, x: mlx.mlx_array, out: []f32) ![]const u32 {
    const f = if (mlx.mlx_array_dtype(x) == .float32) x else try g.astype(x, .float32);
    _ = try g.hostF32(f, out);
    const p: [*]const u32 = @ptrCast(out.ptr);
    return p[0..out.len];
}

/// A prompt call's row-slots: each an x row (`rows`) and a slot (`slots`), the slots drawn over `n_experts` with lognormal
/// popularity (sigma 1.5); `pads` trailing row-slots repeat row-slot 0 (the plugin's padding of a call of 16 rows or fewer)
fn drawCall(rnd: std.Random, rows: []u32, slots: []u32, n_experts: u32, pads: usize) void {
    var logw: [512]f64 = undefined;
    for (logw[0..n_experts]) |*w| w.* = 1.5 * rnd.floatNorm(f64);
    for (rows, slots) |*r, *sl| {
        r.* = rnd.uintLessThan(u32, act_rows);
        var best: u32 = 0;
        var bk: f64 = -std.math.inf(f64);
        for (logw[0..n_experts], 0..) |lw, e| {
            const k = lw - @log(-@log(@max(rnd.float(f64), 1e-300)));
            if (k > bk) {
                bk = k;
                best = @intCast(e);
            }
        }
        sl.* = best;
    }
    const n = rows.len;
    for (n - pads..n) |i| {
        rows[i] = rows[0];
        slots[i] = slots[0];
    }
}

/// The plugin's prompt call today: `take(x, rows)`, f32, `moeClamped` at top-1 over [1, S, hidden], y cast to `out`
fn seamChain(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, rows: mlx.mlx_array, bank: root.Bank, inds: mlx.mlx_array, sc: mlx.mlx_array, S: usize, out: mlx.mlx_dtype) !mlx.mlx_array {
    const xr = try g.astype(try g.take(x, rows, 0), .float32);
    const x3 = try g.reshape(xr, &.{ 1, ci(S), ci(hidden) });
    const y3 = try g.adopt(try root.moeClamped(s, x3, bank, try g.reshape(inds, &.{ 1, ci(S), 1 }), try g.reshape(sc, &.{ 1, ci(S), 1 }), .mul1, limit));
    const y = try g.reshape(y3, &.{ ci(S), ci(hidden) });
    return if (out == .float32) y else g.astype(y, out);
}

test "turboexl3 gpu glm53f fused rows: root.moeClampedRows over gathered rows is the plugin's prompt call (take, f32, moeClamped at top-1, cast) byte for byte, at every SC rate, bf16 and f32 out, padded calls included (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = routed.envList("TURBOEXL3_GLM_ROUTED_N", &routed.rates_n, &nbuf);
    var sbuf: [8]u32 = undefined;
    const calls = routed.envList("TURBOEXL3_GLM_ROWS_S", &.{ 17, 64, 1328, 4096 }, &sbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        defer root.releaseEngines();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x61F7500 ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, prompt_slots, hidden, inter, rate);
        defer bank.deinit(a);
        const rb = try routed.rootBank(&g, &bank, .uint16);
        const xh = try a.alloc(f32, act_rows * hidden);
        defer a.free(xh);
        for (xh) |*v| v.* = routed.bfExact(sigma * rnd.floatNorm(f32));
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(act_rows), ci(hidden) }, .float32), .bfloat16);
        // each call size once, and a padded call: 5 row-slots padded to 17 with row-slot 0's row and slot at score 0
        var cases: [9]struct { S: u32, pads: u32 } = undefined;
        var nc: usize = 0;
        for (calls) |S| {
            if (S <= root.kernels.DECODE_ROWS_MAX or nc == cases.len) continue;
            cases[nc] = .{ .S = S, .pads = 0 };
            nc += 1;
        }
        if (nc < cases.len) {
            cases[nc] = .{ .S = root.kernels.DECODE_ROWS_MAX + 1, .pads = root.kernels.DECODE_ROWS_MAX + 1 - 5 };
            nc += 1;
        }
        for (cases[0..nc]) |cs| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            const S: usize = cs.S;
            const rows = try a.alloc(u32, S);
            defer a.free(rows);
            const slots = try a.alloc(u32, S);
            defer a.free(slots);
            drawCall(rnd, rows, slots, prompt_slots, cs.pads);
            const w = try a.alloc(f32, S);
            defer a.free(w);
            for (w, 0..) |*v, i| v.* = if (i < S - cs.pads) 1.0 else 0.0;
            const rows_a = try g.hostArray(std.mem.sliceAsBytes(rows), &.{ci(S)}, .uint32);
            const inds = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(S)}, .uint32);
            const sc = try g.hostArray(std.mem.sliceAsBytes(w), &.{ci(S)}, .float32);
            const want_buf = try a.alloc(f32, S * hidden);
            defer a.free(want_buf);
            const got_buf = try a.alloc(f32, S * hidden);
            defer a.free(got_buf);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const want = try bitsOf(&g, try seamChain(&g, s, x, rows_a, rb, inds, sc, S, dt), want_buf);
                const y = try g.adopt(try root.moeClampedRows(s, x, rows_a, rb, inds, sc, .mul1, limit, dt));
                try testing.expectEqualSlices(c_int, &.{ ci(S), ci(hidden) }, mlx.getShape(y));
                try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
                const got = try bitsOf(&g, y, got_buf);
                // the rows a call keeps: a padded call's pads are dropped by its caller
                const kept = (S - cs.pads) * hidden;
                var differing: usize = 0;
                var nonfinite: usize = 0;
                for (got[0..kept], want[0..kept], got_buf[0..kept]) |u, v, f| {
                    differing += @intFromBool(u != v);
                    nonfinite += @intFromBool(!std.math.isFinite(f));
                }
                const ok = differing == 0 and nonfinite == 0;
                std.debug.print("TURBOEXL3_GLM_FUSED_ROWS_BYTES n={d} S={d} pads={d} slots={d} out={s} differing={d}/{d} nonfinite={d} {s}\n", .{ n, S, cs.pads, prompt_slots, @tagName(dt), differing, kept, nonfinite, if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            }
        }
    }
    try testing.expect(!failed);
}

test "turboexl3 gpu glm53f fused banked: root.moeClampedBankedAs at bf16 is moeClampedBanked's f32 y cast to bf16, byte for byte, over three slot banks at every SC rate (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = routed.envList("TURBOEXL3_GLM_ROUTED_N", &routed.rates_n, &nbuf);
    var rbuf: [16]u32 = undefined;
    const pair_counts = routed.envList("TURBOEXL3_GLM_BANKED_ROWS", &.{ 1, 2, 7, 8, 16, 32, 48, 64 }, &rbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        defer root.releaseEngines();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x61FB16 ^ @as(u64, n));
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
        for (pair_counts) |rows| {
            if (rows == 0 or rows > root.banked_rows_max) continue;
            const m2 = g.mark();
            defer g.resetTo(m2);
            const xh = try a.alloc(f32, rows * hidden);
            defer a.free(xh);
            for (xh) |*v| v.* = routed.bfExact(sigma * rnd.floatNorm(f32));
            var bank_of: [root.banked_rows_max]u32 = undefined;
            var slot_of: [root.banked_rows_max]u32 = undefined;
            routed.drawPairs(rnd, bank_of[0..rows], slot_of[0..rows]);
            var ids: [root.banked_rows_max]u32 = undefined;
            for (ids[0..rows], bank_of[0..rows], slot_of[0..rows]) |*d, b, sl| d.* = (b << 24) | sl;
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(hidden) }, .float32), .bfloat16);
            const ids_a = try g.hostArray(std.mem.sliceAsBytes(ids[0..rows]), &.{ci(rows)}, .uint32);
            const want_buf = try a.alloc(f32, rows * hidden);
            defer a.free(want_buf);
            const got_buf = try a.alloc(f32, rows * hidden);
            defer a.free(got_buf);
            const y32 = try g.adopt(try root.moeClampedBanked(s, x, &rb, ids_a, .mul1, limit));
            const want = try bitsOf(&g, try g.astype(y32, .bfloat16), want_buf);
            const y = try g.adopt(try root.moeClampedBankedAs(s, x, &rb, ids_a, .mul1, limit, .bfloat16));
            try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(hidden) }, mlx.getShape(y));
            try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y));
            const got = try bitsOf(&g, y, got_buf);
            var differing: usize = 0;
            var nonfinite: usize = 0;
            for (got, want, got_buf) |u, v, f| {
                differing += @intFromBool(u != v);
                nonfinite += @intFromBool(!std.math.isFinite(f));
            }
            const ok = differing == 0 and nonfinite == 0;
            std.debug.print("TURBOEXL3_GLM_FUSED_BANKED_BYTES n={d} pairs={d} out=bf16 differing={d}/{d} nonfinite={d} {s}\n", .{ n, rows, differing, rows * hidden, nonfinite, if (ok) "PASS" else "FAIL" });
            if (!ok) failed = true;
        }
    }
    try testing.expect(!failed);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// `f(ctx)` built and evaluated alone, 2 warm rounds then 8 timed: min and median us per call (graph build included, as a
/// served call pays it).
fn timeCall(g: *G, ctx: anytype, comptime f: anytype) ![2]f64 {
    var t: [8]f64 = undefined;
    for (0..t.len + 2) |round| {
        const m = g.mark();
        defer g.resetTo(m);
        const t0 = nowNs();
        const y = try f(g, ctx);
        try g.evalAll(&.{y});
        const dt = nowNs() - t0;
        if (round >= 2) t[round - 2] = @as(f64, @floatFromInt(dt)) / 1000.0;
    }
    std.mem.sort(f64, &t, {}, std.sort.asc(f64));
    return .{ t[0], t[t.len / 2] };
}

const RowsCtx = struct { s: mlx.mlx_stream, x: mlx.mlx_array, rows: mlx.mlx_array, bank: root.Bank, inds: mlx.mlx_array, sc: mlx.mlx_array, S: usize };
const BankedCtx = struct { s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]root.Bank, ids: mlx.mlx_array };

test "turboexl3 gpu glm53f fused timing: each fused path against the chain it replaces at the served shapes (TURBOEXL3_GLM_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_GLM_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    defer root.releaseEngines();
    var prng = std.Random.DefaultPrng.init(0x61F71E);
    const rnd = prng.random();
    // K3, the sc_3.10bpw pack's routed layers
    const rate = format.Rate.fromN(48) orelse return error.Exl3RateUnsupported;
    {
        const mark = g.mark();
        defer g.resetTo(mark);
        var bank = try ref.synthBank(a, rnd, prompt_slots, hidden, inter, rate);
        defer bank.deinit(a);
        const rb = try routed.rootBank(&g, &bank, .uint16);
        const xh = try a.alloc(f32, act_rows * hidden);
        defer a.free(xh);
        for (xh) |*v| v.* = routed.bfExact(sigma * rnd.floatNorm(f32));
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(act_rows), ci(hidden) }, .float32), .bfloat16);
        // a 4096-row prompt call as the plugin sorts its rows: 9 experts' runs of about 455 rows (a 16K prompt's rows per
        // expert at top-8 of 288)
        const S: usize = 4096;
        const rows = try a.alloc(u32, S);
        defer a.free(rows);
        const slots = try a.alloc(u32, S);
        defer a.free(slots);
        for (rows, slots, 0..) |*r, *sl, i| {
            r.* = rnd.uintLessThan(u32, act_rows);
            sl.* = @intCast(@min(i / 455, 8) * 11);
        }
        const ones: [S]f32 = @splat(1.0);
        const c: RowsCtx = .{
            .s = s,
            .x = x,
            .rows = try g.hostArray(std.mem.sliceAsBytes(rows), &.{ci(S)}, .uint32),
            .bank = rb,
            .inds = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(S)}, .uint32),
            .sc = try g.hostArray(std.mem.sliceAsBytes(ones[0..]), &.{ci(S)}, .float32),
            .S = S,
        };
        const chain = try timeCall(&g, c, struct {
            fn f(gg: *G, cc: RowsCtx) !mlx.mlx_array {
                return seamChain(gg, cc.s, cc.x, cc.rows, cc.bank, cc.inds, cc.sc, cc.S, .bfloat16);
            }
        }.f);
        const fused = try timeCall(&g, c, struct {
            fn f(gg: *G, cc: RowsCtx) !mlx.mlx_array {
                return gg.adopt(try root.moeClampedRows(cc.s, cc.x, cc.rows, cc.bank, cc.inds, cc.sc, .mul1, limit, .bfloat16));
            }
        }.f);
        std.debug.print("TURBOEXL3_GLM_FUSED_TIME path=prompt_rows n=48 S={d} experts=9 chain_min_us={d:.1} chain_med_us={d:.1} fused_min_us={d:.1} fused_med_us={d:.1}\n", .{ S, chain[0], chain[1], fused[0], fused[1] });
    }
    {
        const mark = g.mark();
        defer g.resetTo(mark);
        var hb: [3]ref.Bank = undefined;
        var made: usize = 0;
        defer for (hb[0..made]) |*b| b.deinit(a);
        for (&hb, routed.bank_experts) |*b, e| {
            b.* = try ref.synthBank(a, rnd, e, hidden, inter, rate);
            made += 1;
        }
        var rb: [3]root.Bank = undefined;
        for (&hb, &rb) |*b, *r| r.* = try routed.rootBank(&g, b, .int16);
        // a 4-row verify's 32 pairs
        const P: u32 = 32;
        const xh = try a.alloc(f32, P * hidden);
        defer a.free(xh);
        for (xh) |*v| v.* = routed.bfExact(sigma * rnd.floatNorm(f32));
        var bank_of: [P]u32 = undefined;
        var slot_of: [P]u32 = undefined;
        routed.drawPairs(rnd, &bank_of, &slot_of);
        var ids: [P]u32 = undefined;
        for (&ids, bank_of, slot_of) |*d, b, sl| d.* = (b << 24) | sl;
        const c: BankedCtx = .{
            .s = s,
            .x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(P), ci(hidden) }, .float32), .bfloat16),
            .banks = &rb,
            .ids = try g.hostArray(std.mem.sliceAsBytes(ids[0..]), &.{ci(P)}, .uint32),
        };
        const chain = try timeCall(&g, c, struct {
            fn f(gg: *G, cc: BankedCtx) !mlx.mlx_array {
                return gg.astype(try gg.adopt(try root.moeClampedBanked(cc.s, cc.x, cc.banks, cc.ids, .mul1, limit)), .bfloat16);
            }
        }.f);
        const fused = try timeCall(&g, c, struct {
            fn f(gg: *G, cc: BankedCtx) !mlx.mlx_array {
                return gg.adopt(try root.moeClampedBankedAs(cc.s, cc.x, cc.banks, cc.ids, .mul1, limit, .bfloat16));
            }
        }.f);
        std.debug.print("TURBOEXL3_GLM_FUSED_TIME path=banked_bf16 n=48 pairs={d} chain_min_us={d:.1} chain_med_us={d:.1} fused_min_us={d:.1} fused_med_us={d:.1}\n", .{ P, chain[0], chain[1], fused[0], fused[1] });
    }
}

/// One trunk linear of the sc_3.10bpw pack (K6) for the timing below: its shape and how many a forward runs.
const TrunkShape = struct { name: []const u8, in: u32, out: u32, per_forward: u32, prefill: bool = true };
/// The sc_3.10bpw trunk a forward runs (34 KDA layers, 11 DSA layers, 42 MoE layers' shared experts, the 3 dense
/// layers' MLP, the head; config.json's layer_types): every EXL3 linear but the routed experts, K6.
const trunk_shapes = [_]TrunkShape{
    .{ .name = "kda.qkv_proj", .in = 4096, .out = 24576, .per_forward = 34 },
    .{ .name = "kda.o_proj", .in = 8192, .out = 4096, .per_forward = 34 },
    .{ .name = "dsa.q_a_proj", .in = 4096, .out = 1536, .per_forward = 11 },
    .{ .name = "dsa.q_b_proj", .in = 1536, .out = 16384, .per_forward = 11 },
    .{ .name = "dsa.kv_a_proj_with_mqa", .in = 4096, .out = 512, .per_forward = 11 },
    .{ .name = "dsa.o_proj", .in = 16384, .out = 4096, .per_forward = 11 },
    .{ .name = "indexer.wq_b", .in = 1536, .out = 4096, .per_forward = 11 },
    .{ .name = "shared.gate_proj/up_proj", .in = 4096, .out = 2048, .per_forward = 84 },
    .{ .name = "shared.down_proj", .in = 2048, .out = 4096, .per_forward = 42 },
    .{ .name = "dense.gate_proj/up_proj", .in = 4096, .out = 12288, .per_forward = 6 },
    .{ .name = "dense.down_proj", .in = 12288, .out = 4096, .per_forward = 3 },
    // a prompt chunk evaluates the head on its last row only
    .{ .name = "lm_head", .in = 4096, .out = 154880, .per_forward = 1, .prefill = false },
};

const LinCtx = struct { s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj };

test "turboexl3 gpu glm53f trunk timing: every sc_3.10bpw trunk linear (K6) at decode rows 1 and 4 and a 16384-row prompt chunk, and its sum over a forward (TURBOEXL3_GLM_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_GLM_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    defer root.releaseEngines();
    var prng = std.Random.DefaultPrng.init(0x61F7E7);
    const rnd = prng.random();
    const n: u32 = 96; // K6
    // x's values from a pool of bf16-exact normals (a prompt chunk's x is up to 268M values: drawing each would take
    // seconds of host time; the timing does not depend on the values)
    var pool: [8191]f32 = undefined;
    for (&pool) |*v| v.* = routed.bfExact(rnd.floatNorm(f32));
    for ([_]u32{ 1, 4, 16384 }) |rows| {
        var total_ms: f64 = 0;
        for (trunk_shapes) |t| {
            if (rows > root.kernels.DECODE_ROWS_MAX and !t.prefill) continue;
            const mark = g.mark();
            defer g.resetTo(mark);
            // random words (every pattern a codeword), suh in [0.5, 1.5], svh in [0.5, 1.5] / sqrt(in): values for timing only
            const tw = try a.alloc(u16, (t.in / 16) * (t.out / 16) * n);
            defer a.free(tw);
            rnd.bytes(std.mem.sliceAsBytes(tw));
            const su = try a.alloc(u16, t.in);
            defer a.free(su);
            for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
            const sv = try a.alloc(u16, t.out);
            defer a.free(sv);
            for (sv) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(t.in))));
            const p: root.Proj = .{
                .trellis = try g.hostArray(std.mem.sliceAsBytes(tw), &.{ ci(t.in / 16), ci(t.out / 16), ci(n) }, .uint16),
                .suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(t.in)}, .float16),
                .svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(t.out)}, .float16),
            };
            const xh = try a.alloc(f32, @as(usize, rows) * t.in);
            defer a.free(xh);
            for (xh, 0..) |*v, i| v.* = pool[i % pool.len];
            const c: LinCtx = .{ .s = s, .x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(t.in) }, .float32), .bfloat16), .p = p };
            try g.evalAll(&.{ c.x, p.trellis, p.suh, p.svh });
            const st = try timeCall(&g, c, struct {
                fn f(gg: *G, cc: LinCtx) !mlx.mlx_array {
                    return gg.adopt(try root.linear(cc.s, cc.x, cc.p.trellis, cc.p.suh, cc.p.svh, .mul1));
                }
            }.f);
            const flops = 2.0 * @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(t.in)) * @as(f64, @floatFromInt(t.out));
            const forward_ms = st[0] * @as(f64, @floatFromInt(t.per_forward)) / 1000.0;
            total_ms += forward_ms;
            std.debug.print("TURBOEXL3_GLM_TRUNK_TIME name={s} in={d} out={d} n={d} rows={d} min_us={d:.1} med_us={d:.1} tflops={d:.2} per_forward={d} forward_ms={d:.2}\n", .{ t.name, t.in, t.out, n, rows, st[0], st[1], flops / (st[0] * 1e6), t.per_forward, forward_ms });
            root.releaseEngines();
        }
        std.debug.print("TURBOEXL3_GLM_TRUNK_TOTAL rows={d} forward_ms={d:.2} (each linear alone, min of 8, times its count; the head at decode rows only)\n", .{ rows, total_ms });
    }
}
