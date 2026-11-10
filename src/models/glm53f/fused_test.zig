//! GLM-5.3-Flash's fused paths on turboexl3, each the opt-in switch of the outside plugin turboexl3_glm53f, and each held
//! byte for byte to the chain of launches it replaces, at GLM-5.3-Flash's routed geometry (hidden 4096, expert width 2048,
//! `swiglu_limit` 10) and every routed rate of the SC packs:
//! - `root.moeClampedRows` (the plugin's TURBOEXL3_GLM53F_PF_ROWS): a prompt call over an expert streamer's slot bank, row
//!   p of x (bf16 [n, hidden], the layer's activation) through slot `inds[p]`, the gather in the chain's prepare
//!   (`prefill.text_prep_rows`) and y stored bf16 by the chain's finish. The chain it replaces is the plugin's prompt
//!   call today: `take(x, rows)`, a cast to f32, `moeClamped` at top-1 over [1, S, hidden], a cast of its f32 y to bf16.
//! - `root.moeClampedBankedAs` at bf16 (the plugin's TURBOEXL3_GLM53F_DEC_BF16): a decode wave over the three slot banks with
//!   y stored bf16 by the banked down finish. The chain it replaces: `moeClampedBanked` (y f32), then a cast to bf16.
//! And the plugin's opt-in schedules, each held to the default's bytes:
//! - the banked decode's schedules (the plugin's TURBOEXL3_GLM53F_BANKED): the grouped texts (`banked_group` 4), hw4b's
//!   (`banked_m3`) and every grouped GEMV variant on each but the plugin's refused EXL3_GSH ones (`sh`, `pfsh`: other bytes
//!   at the K5 rate, g53fz2), selected per call (`moeClampedBankedSelV`: each gate|up
//!   variant with today's down GEMV, each down variant with today's gate|up GEMV) and configured as the plugin configures
//!   them (`configure`, a geometry's banked state made after it: y f32, and bf16 from `moeClampedBankedAs`), against the
//!   served call (`banked_group` 2), over waves of 1 to 64 pairs (a verify's rows drawn routing the same experts);
//! - a prompt call in calls of 2048, 8192 or 16384 row-slots (the plugin's TURBOEXL3_GLM53F_PF_CALL_ROWS) against its
//!   4096-row calls, 16384 row-slots in the plugin's slot order, on the default path and on `moeClampedRows`;
//! - the same call (4096-row calls) under the routed prefill's bk128, half_rows and both (`Config.pf_bk128`,
//!   `pf_half_rows`), and the 128-row tile launches' 128-deep stages alone and with bk128 (`pf_t128_bk128`; all the
//!   plugin's TURBOEXL3_GLM53F_PF_OPTS) against the served options;
//! - every pg2 candidate tile (`pgemm.candidates`, as the plugin's TURBOEXL3_GLM53F_PG2 names them by code) at every
//!   trunk linear of `api.zig` (`root.glm53f.trunk`) against the chain (dequantize, MLX's matmul, the finish), word for
//!   word, bf16 x and y, at 100, 2048, 4100 and 8200 rows where MLX runs its regular kernel.
//! And the routed gate|up and down on int8 (`Config.pf_int8` 1: a scale each activation 128-block; 2: one scale a row;
//! lossy), held to agreement with the exact chain instead, both with the size classes as the plugin serves them: the
//! same 4096-row calls' y on both paths at every SC rate, cosine at least 0.99 and the error's rms at most a tenth of y's
//! (TURBOEXL3_GLM_PF_INT8_AGREE); timed beside the exact chain under TURBOEXL3_GLM_TIME (lines TURBOEXL3_GLM_PF_INT8_TIME
//! for the call, TURBOEXL3_GLM_PF_INT8_STAGE for its route, prepare and gate|up alone and through mid and down).
//! - CPU (always): the row-map prepare's text is the prepare's with its one row line swapped for the row map's, and the
//!   prepare's own text is unchanged by the cut.
//! - GPU (TURBOEXL3_GPU=1, else skipped): the bytes above, x ~ N(0, 8^2) bf16-exact (about a tenth of the gates and a
//!   quarter of the ups pass the limit). TURBOEXL3_GLM_ROUTED_N (rates in halfwords a tile, a comma list; default
//!   32,40,48,56,64,80) narrows both; TURBOEXL3_GLM_ROWS_S (default 17,64,1328,4096) the prompt calls' row-slots,
//!   TURBOEXL3_GLM_BANKED_ROWS (default 1,2,7,8,16,32,48,64; the schedules' 1,2,7,8,16,24,32,48,64) the decode waves'
//!   pairs. Each rate's engines are freed after it. Lines: TURBOEXL3_GLM_FUSED_ROWS_BYTES, TURBOEXL3_GLM_FUSED_BANKED_BYTES,
//!   TURBOEXL3_GLM_BANKED_SCHED_BYTES, TURBOEXL3_GLM_PF_CHUNK_BYTES, TURBOEXL3_GLM_PF_OPTS_BYTES, TURBOEXL3_GLM_PG2_BYTES.
//! - TURBOEXL3_GLM_TIME=1 (with TURBOEXL3_GPU=1, else skipped): each fused path against its chain on the device at the
//!   served shapes (a 4096-row prompt call over 9 experts of a 100-slot bank; a 4-row verify's 32 pairs), and every
//!   sc_3.10bpw trunk linear (K6, `root.linear`) at decode rows 1 and 4 and a 16384-row prompt chunk with its count in a
//!   forward (the trunk's share of a forward: each linear alone, so launch overlap and the forward's other ops are not in
//!   it); min and median us per call over 8 rounds after 2 warm. Also the banked decode's schedules at a verify's 24 and
//!   32 pairs (the served call, then each texts at each variant for both GEMVs), and a 16384-row prompt bank call (36
//!   experts' runs of about 455 rows) in calls of 2048 to 16384 rows on each path; that call (and one over a drawn
//!   routing) under the served options and each of the plugin's TURBOEXL3_GLM53F_PF_OPTS values, in calls of 4096 and
//!   16384 rows; each pg2 candidate tile against the chain at every trunk linear at 2048, 8192 and 16384 rows, and the
//!   code each linear gets by pg2u's rule (pgemm.zig: at or below the chain at every row count, the fastest at 16384). It
//!   asserts nothing (a microbenchmark steers, the standard test decides). Lines: TURBOEXL3_GLM_FUSED_TIME,
//!   TURBOEXL3_GLM_TRUNK_TIME, TURBOEXL3_GLM_TRUNK_TOTAL, TURBOEXL3_GLM_BANKED_SCHED_TIME, TURBOEXL3_GLM_PF_CHUNK_TIME,
//!   TURBOEXL3_GLM_PF_OPTS_TIME, TURBOEXL3_GLM_PG2_TIME, TURBOEXL3_GLM_PG2_PICK, TURBOEXL3_GLM_PG2_CODES.
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
pub const sigma: f32 = 8;
/// The prompt calls' slot bank: a base bank's rows (the plugin serves 100 to 145 prompt rows a layer at the 110 GiB ceiling)
pub const prompt_slots: u32 = 100;
/// The layer activation's rows the prompt calls gather from
pub const act_rows: u32 = 600;

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
pub fn bitsOf(g: *G, x: mlx.mlx_array, out: []f32) ![]const u32 {
    const f = if (mlx.mlx_array_dtype(x) == .float32) x else try g.astype(x, .float32);
    _ = try g.hostF32(f, out);
    const p: [*]const u32 = @ptrCast(out.ptr);
    return p[0..out.len];
}

/// A prompt call's row-slots: each an x row (`rows`) and a slot (`slots`), the slots drawn over `n_experts` with lognormal
/// popularity (sigma 1.5); `pads` trailing row-slots repeat row-slot 0 (the plugin's padding of a call of 16 rows or fewer)
pub fn drawCall(rnd: std.Random, rows: []u32, slots: []u32, n_experts: u32, pads: usize) void {
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

/// A decode wave's packed ids (bank << 24 | slot row) over three banks of `routed.bank_experts`: at a multiple of 8 pairs
/// a verify's rows, 8 distinct experts a row, each pick its previous row's at that position half the time (a verify's
/// rows route the same experts often: the grouped texts' members), a fresh pick in the base bank 7 times in 10, the ext
/// 2 and the transient 1; at another count drawn with repeats (`routed.drawPairs`)
pub fn drawWave(rnd: std.Random, ids: []u32) void {
    const n = ids.len;
    if (n % 8 != 0) {
        var bank_of: [root.banked_rows_max]u32 = undefined;
        var slot_of: [root.banked_rows_max]u32 = undefined;
        routed.drawPairs(rnd, bank_of[0..n], slot_of[0..n]);
        for (ids, bank_of[0..n], slot_of[0..n]) |*d, b, sl| d.* = (b << 24) | sl;
        return;
    }
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

/// The differing words of `got` against `want`, and the non-finite values of `vals` (got's values)
pub fn countDiff(got: []const u32, want: []const u32, vals: []const f32) [2]usize {
    var d: [2]usize = .{ 0, 0 };
    for (got, want, vals) |u, v, f| {
        d[0] += @intFromBool(u != v);
        d[1] += @intFromBool(!std.math.isFinite(f));
    }
    return d;
}

/// A grouped GEMV variant's name for a line (`0` today's)
pub fn varName(v: u8) []const u8 {
    return if (v == 0) "0" else root.group_variants[v].name;
}

/// The grouped GEMV variants the plugin refuses (its `BankedSchedule.refused_variants`): the EXL3_GSH ones give other
/// bytes than the served banked call at the K5 rate (g53fz2: 4095 of 4096 words at one pair, on either GEMV and either
/// texts), so this file holds only the others to the served call and times only them
const refused_variants = [_][]const u8{ "sh", "pfsh" };

pub fn refusedVariant(v: usize) bool {
    for (refused_variants) |r| if (std.mem.eql(u8, root.group_variants[v].name, r)) return true;
    return false;
}

test "turboexl3 gpu glm53f banked schedules: the grouped texts (g4), hw4b's (g4m3) and every grouped GEMV variant on each, selected per call or configured, give the served banked call's rows byte for byte (the bf16 store its rounding), at GLM-5.3-Flash's routed geometry and every SC rate (TURBOEXL3_GPU=1)" {
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
    var rbuf: [16]u32 = undefined;
    const pair_counts = routed.envList("TURBOEXL3_GLM_BANKED_ROWS", &.{ 1, 2, 7, 8, 16, 24, 32, 48, 64 }, &rbuf);
    const n_var = root.group_variants.len;
    // the plugin's TURBOEXL3_GLM53F_BANKED as it configures turboexl3: each texts, a variant pair of each GEMV
    const Configured = struct { m3: bool, gu: []const u8, dn: []const u8 };
    const configured = [_]Configured{ .{ .m3 = false, .gu = "pf", .dn = "xh" }, .{ .m3 = true, .gu = "xh", .dn = "pf" } };
    var failed = false;
    var checked: usize = 0;
    const max_w = root.banked_rows_max * hidden;
    const want_buf = try a.alloc(f32, max_w);
    defer a.free(want_buf);
    const got_buf = try a.alloc(f32, max_w);
    defer a.free(got_buf);
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        defer root.releaseEngines();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x61FD3 ^ @as(u64, n));
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
        // each case's inputs and the served call's rows (the default configuration: banked_group 2), kept to the rate's end
        const Case = struct { pairs: u32, x: mlx.mlx_array, ids: mlx.mlx_array, want: mlx.mlx_array };
        var cases: [16]Case = undefined;
        var nc: usize = 0;
        for (pair_counts) |pairs| {
            if (pairs == 0 or pairs > root.banked_rows_max or nc == cases.len) continue;
            const xh = try a.alloc(f32, pairs * hidden);
            defer a.free(xh);
            for (xh) |*v| v.* = routed.bfExact(sigma * rnd.floatNorm(f32));
            var ids: [root.banked_rows_max]u32 = undefined;
            drawWave(rnd, ids[0..pairs]);
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(pairs), ci(hidden) }, .float32), .bfloat16);
            const ids_a = try g.hostArray(std.mem.sliceAsBytes(ids[0..pairs]), &.{ci(pairs)}, .uint32);
            const want = try g.adopt(try root.moeClampedBanked(s, x, &rb, ids_a, .mul1, limit));
            try g.evalAll(&.{ x, ids_a, want });
            cases[nc] = .{ .pairs = pairs, .x = x, .ids = ids_a, .want = want };
            nc += 1;
        }
        // selected per call: each texts at each gate|up variant (the down GEMV today's) and each down variant (the gate|up
        // GEMV today's): any pair of them is a composition of the two
        for (cases[0..nc]) |c| {
            const w_n = c.pairs * hidden;
            const want = try bitsOf(&g, c.want, want_buf[0..w_n]);
            for ([_]root.BankedSel{ .g4, .g4m3 }) |sel| for (0..n_var) |vi| for ([_]bool{ false, true }) |down| {
                if (down and vi == 0) continue;
                if (refusedVariant(vi)) continue;
                const m2 = g.mark();
                defer g.resetTo(m2);
                const v: u8 = @intCast(vi);
                const vgu: u8 = if (down) 0 else v;
                const vdn: u8 = if (down) v else 0;
                const y = try g.adopt(try root.moeClampedBankedSelV(s, c.x, &rb, c.ids, .mul1, limit, sel, vgu, vdn));
                const got = try bitsOf(&g, y, got_buf[0..w_n]);
                const d = countDiff(got, want, got_buf[0..w_n]);
                const ok = d[0] == 0 and d[1] == 0;
                checked += w_n;
                std.debug.print("TURBOEXL3_GLM_BANKED_SCHED_BYTES n={d} pairs={d} route=selected texts={t} gu={s} dn={s} out=f32 differing={d}/{d} nonfinite={d} {s}\n", .{ n, c.pairs, sel, varName(vgu), varName(vdn), d[0], w_n, d[1], if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            };
        }
        // configured, as the plugin sets it before its first banked call (a geometry's banked state made after it): the
        // configured call's y at f32 (moeClampedBanked) and at bf16 (moeClampedBankedAs: the served y rounded once)
        for (configured) |cf| {
            root.releaseEngines();
            var cfg = saved;
            cfg.banked_group = 4;
            cfg.banked_m3 = cf.m3;
            cfg.banked_variant_gu = root.groupVariantOf(cf.gu).?;
            cfg.banked_variant_dn = root.groupVariantOf(cf.dn).?;
            try root.configure(cfg);
            defer {
                root.configure(saved) catch unreachable;
                root.releaseEngines();
            }
            for (cases[0..nc]) |c| {
                const w_n = c.pairs * hidden;
                for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dt| {
                    const m2 = g.mark();
                    defer g.resetTo(m2);
                    const want = try bitsOf(&g, if (dt == .float32) c.want else try g.astype(c.want, .bfloat16), want_buf[0..w_n]);
                    const y = try g.adopt(if (dt == .float32)
                        try root.moeClampedBanked(s, c.x, &rb, c.ids, .mul1, limit)
                    else
                        try root.moeClampedBankedAs(s, c.x, &rb, c.ids, .mul1, limit, .bfloat16));
                    try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
                    const got = try bitsOf(&g, y, got_buf[0..w_n]);
                    const d = countDiff(got, want, got_buf[0..w_n]);
                    const ok = d[0] == 0 and d[1] == 0;
                    checked += w_n;
                    std.debug.print("TURBOEXL3_GLM_BANKED_SCHED_BYTES n={d} pairs={d} route=configured texts={s} gu={s} dn={s} out={s} differing={d}/{d} nonfinite={d} {s}\n", .{ n, c.pairs, if (cf.m3) "g4m3" else "g4", cf.gu, cf.dn, @tagName(dt), d[0], w_n, d[1], if (ok) "PASS" else "FAIL" });
                    if (!ok) failed = true;
                }
            }
        }
    }
    std.debug.print("TURBOEXL3_GLM_BANKED_SCHED_BYTES total_words={d} {s}\n", .{ checked, if (failed) "FAIL" else "PASS" });
    try testing.expect(!failed);
}

/// A prompt call's row-slots as the plugin orders them (a stable counting sort by slot): `rows` and `slots` permuted
pub fn sortBySlot(a: std.mem.Allocator, rows: []u32, slots: []u32, n_slots: u32) !void {
    const n = rows.len;
    const starts = try a.alloc(u32, n_slots + 1);
    defer a.free(starts);
    @memset(starts, 0);
    for (slots) |sl| starts[sl + 1] += 1;
    for (1..starts.len) |i| starts[i] += starts[i - 1];
    const r2 = try a.alloc(u32, n);
    defer a.free(r2);
    const s2 = try a.alloc(u32, n);
    defer a.free(s2);
    for (rows, slots) |r, sl| {
        r2[starts[sl]] = r;
        s2[starts[sl]] = sl;
        starts[sl] += 1;
    }
    @memcpy(rows, r2);
    @memcpy(slots, s2);
}

/// One prompt call over the row-slots `rows` and `slots` (host values, each at weight 1: `ones_h` at least as long) on
/// the plugin's default path (`seamChain`) or on `moeClampedRows`, y bf16
fn chunkCall(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, bank: root.Bank, rows: []const u32, slots: []const u32, ones_h: []const f32, on_rows: bool) !mlx.mlx_array {
    const S = rows.len;
    const ones = try g.hostArray(std.mem.sliceAsBytes(ones_h[0..S]), &.{ci(S)}, .float32);
    const rows_a = try g.hostArray(std.mem.sliceAsBytes(rows), &.{ci(S)}, .uint32);
    const inds = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(S)}, .uint32);
    if (on_rows) return g.adopt(try root.moeClampedRows(s, x, rows_a, bank, inds, ones, .mul1, limit, .bfloat16));
    return seamChain(g, s, x, rows_a, bank, inds, ones, S, .bfloat16);
}

test "turboexl3 gpu glm53f chunked prompt calls: a prompt call's row-slots in calls of 2048, 8192 or 16384 are its 4096-row calls' byte for byte, on the plugin's default path and on moeClampedRows, at every SC rate (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = routed.envList("TURBOEXL3_GLM_ROUTED_N", &routed.rates_n, &nbuf);
    // a 16K prompt's group of 48 experts routes about 22,000 row-slots a bank call
    const S: usize = 16384;
    const base_chunk: usize = 4096;
    const chunks = [_]usize{ 2048, 8192, 16384 };
    const want_buf = try a.alloc(f32, S * hidden);
    defer a.free(want_buf);
    const got_buf = try a.alloc(f32, S * hidden);
    defer a.free(got_buf);
    const ones_h = try a.alloc(f32, S);
    defer a.free(ones_h);
    @memset(ones_h, 1.0);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        defer root.releaseEngines();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x61FC4 ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, prompt_slots, hidden, inter, rate);
        defer bank.deinit(a);
        const rb = try routed.rootBank(&g, &bank, .uint16);
        const xh = try a.alloc(f32, act_rows * hidden);
        defer a.free(xh);
        for (xh) |*v| v.* = routed.bfExact(sigma * rnd.floatNorm(f32));
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(act_rows), ci(hidden) }, .float32), .bfloat16);
        try g.evalAll(&.{x});
        const rows = try a.alloc(u32, S);
        defer a.free(rows);
        const slots = try a.alloc(u32, S);
        defer a.free(slots);
        drawCall(rnd, rows, slots, prompt_slots, 0);
        try sortBySlot(a, rows, slots, prompt_slots);
        for ([_]bool{ false, true }) |on_rows| {
            // the 4096-row calls' rows (the plugin's default chunk), then each other chunk size's
            for ([_]usize{base_chunk} ++ chunks) |chunk| {
                const buf = if (chunk == base_chunk) want_buf else got_buf;
                var c0: usize = 0;
                while (c0 < S) : (c0 += chunk) {
                    const c1 = @min(c0 + chunk, S);
                    const m2 = g.mark();
                    defer g.resetTo(m2);
                    const y = try chunkCall(&g, s, x, rb, rows[c0..c1], slots[c0..c1], ones_h, on_rows);
                    _ = try bitsOf(&g, y, buf[c0 * hidden .. c1 * hidden]);
                }
                if (chunk == base_chunk) continue;
                const want: []const u32 = @as([*]const u32, @ptrCast(want_buf.ptr))[0 .. S * hidden];
                const got: []const u32 = @as([*]const u32, @ptrCast(got_buf.ptr))[0 .. S * hidden];
                const d = countDiff(got, want, got_buf);
                const ok = d[0] == 0 and d[1] == 0;
                std.debug.print("TURBOEXL3_GLM_PF_CHUNK_BYTES n={d} S={d} path={s} chunk={d} against={d} differing={d}/{d} nonfinite={d} {s}\n", .{ n, S, if (on_rows) "rows" else "chain", chunk, base_chunk, d[0], S * hidden, d[1], if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            }
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
            // the run's index in u32 first: @min with a comptime bound narrows its type (u4 here), and 8 * 11 needs 7 bits
            const run: u32 = @intCast(@min(i / 455, 8));
            sl.* = run * 11;
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

/// A banked call for the schedule timing: `sel` null the configured (served) call, else `sel`'s texts at variant `v` for
/// both GEMVs
const SchedCtx = struct { s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]root.Bank, ids: mlx.mlx_array, sel: ?root.BankedSel, v: u8 };

/// A prompt bank call's row-slots for the chunk timing, on one path, in calls of `chunk` rows
const ChunkCtx = struct { s: mlx.mlx_stream, x: mlx.mlx_array, bank: root.Bank, rows: []const u32, slots: []const u32, ones: []const f32, on_rows: bool, chunk: usize };

test "turboexl3 gpu glm53f schedule timing: the banked decode's schedules at a verify's waves, and a prompt bank call in calls of 2048 to 16384 rows (TURBOEXL3_GLM_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_GLM_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    defer root.releaseEngines();
    var prng = std.Random.DefaultPrng.init(0x61F5C7);
    const rnd = prng.random();
    // K3, the sc_3.10bpw pack's routed layers
    const n: u32 = 48;
    const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
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
        // a depth-2 verify's 3 rows and a 4-row verify's (24 and 32 pairs)
        for ([_]u32{ 24, 32 }) |pairs| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            const xh = try a.alloc(f32, pairs * hidden);
            defer a.free(xh);
            for (xh) |*v| v.* = routed.bfExact(sigma * rnd.floatNorm(f32));
            var ids: [root.banked_rows_max]u32 = undefined;
            drawWave(rnd, ids[0..pairs]);
            var c: SchedCtx = .{
                .s = s,
                .x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(pairs), ci(hidden) }, .float32), .bfloat16),
                .banks = &rb,
                .ids = try g.hostArray(std.mem.sliceAsBytes(ids[0..pairs]), &.{ci(pairs)}, .uint32),
                .sel = null,
                .v = 0,
            };
            try g.evalAll(&.{ c.x, c.ids });
            const f = struct {
                fn f(gg: *G, cc: SchedCtx) !mlx.mlx_array {
                    return gg.adopt(if (cc.sel) |sel| try root.moeClampedBankedSelV(cc.s, cc.x, cc.banks, cc.ids, .mul1, limit, sel, cc.v, cc.v) else try root.moeClampedBanked(cc.s, cc.x, cc.banks, cc.ids, .mul1, limit));
                }
            }.f;
            const served = try timeCall(&g, c, f);
            std.debug.print("TURBOEXL3_GLM_BANKED_SCHED_TIME n={d} pairs={d} texts=served gu=0 dn=0 min_us={d:.1} med_us={d:.1}\n", .{ n, pairs, served[0], served[1] });
            for ([_]root.BankedSel{ .g4, .g4m3, .g4p, .g4m3p }) |sel| for (0..root.group_variants.len) |vi| {
                if (refusedVariant(vi)) continue;
                c.sel = sel;
                c.v = @intCast(vi);
                const t = try timeCall(&g, c, f);
                std.debug.print("TURBOEXL3_GLM_BANKED_SCHED_TIME n={d} pairs={d} texts={t} gu={s} dn={s} min_us={d:.1} med_us={d:.1}\n", .{ n, pairs, sel, varName(c.v), varName(c.v), t[0], t[1] });
            };
        }
    }
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
        try g.evalAll(&.{x});
        // a 16K prompt's bank call: 36 experts' runs of about 455 rows, in slot order (the plugin's)
        const S: usize = 16384;
        const rows = try a.alloc(u32, S);
        defer a.free(rows);
        const slots = try a.alloc(u32, S);
        defer a.free(slots);
        for (rows, slots, 0..) |*r, *sl, i| {
            r.* = rnd.uintLessThan(u32, act_rows);
            // the run's index in u32 first: @min with a comptime bound narrows its type (u6 here), and 35 * 2 needs 7 bits
            const run: u32 = @intCast(@min(i / 455, 35));
            sl.* = run * 2;
        }
        const ones_h = try a.alloc(f32, S);
        defer a.free(ones_h);
        @memset(ones_h, 1.0);
        for ([_]bool{ false, true }) |on_rows| for ([_]usize{ 2048, 4096, 8192, 16384 }) |chunk| {
            const c: ChunkCtx = .{ .s = s, .x = x, .bank = rb, .rows = rows, .slots = slots, .ones = ones_h, .on_rows = on_rows, .chunk = chunk };
            var t: [8]f64 = undefined;
            for (0..t.len + 2) |round| {
                const m = g.mark();
                defer g.resetTo(m);
                var outs: [8]mlx.mlx_array = undefined;
                var no: usize = 0;
                const t0 = nowNs();
                var c0: usize = 0;
                while (c0 < S) : (c0 += c.chunk) {
                    const c1 = @min(c0 + c.chunk, S);
                    outs[no] = try chunkCall(&g, c.s, c.x, c.bank, c.rows[c0..c1], c.slots[c0..c1], c.ones, c.on_rows);
                    no += 1;
                }
                try g.evalAll(outs[0..no]);
                const dt = nowNs() - t0;
                if (round >= 2) t[round - 2] = @as(f64, @floatFromInt(dt)) / 1e6;
            }
            std.mem.sort(f64, &t, {}, std.sort.asc(f64));
            const flops = 2.0 * 3.0 * @as(f64, @floatFromInt(S)) * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter));
            std.debug.print("TURBOEXL3_GLM_PF_CHUNK_TIME n={d} S={d} experts=36 path={s} chunk={d} calls={d} min_ms={d:.2} med_ms={d:.2} tflops={d:.2}\n", .{ n, S, if (on_rows) "rows" else "chain", chunk, (S + chunk - 1) / chunk, t[0], t[t.len / 2], flops / (t[0] * 1e9) });
        };
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

/// A trunk linear's K6 projection for the pg2 tests: random words (every pattern a codeword), suh in [0.5, 1.5], svh in
/// [0.5, 1.5] / sqrt(in), the trunk timing's values
fn trunkProj(g: *G, a: std.mem.Allocator, rnd: std.Random, in: u32, out: u32) !root.Proj {
    const n: u32 = 96; // K6
    const tw = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(tw);
    rnd.bytes(std.mem.sliceAsBytes(tw));
    const su = try a.alloc(u16, in);
    defer a.free(su);
    for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, out);
    defer a.free(sv);
    for (sv) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    const p: root.Proj = .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(tw), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(out)}, .float16),
    };
    try g.evalAll(&.{ p.trellis, p.suh, p.svh });
    return p;
}

/// A trunk linear's x: bf16 [rows, in] from `pool`'s bf16-exact normals in turn
fn trunkX(g: *G, a: std.mem.Allocator, pool: []const f32, rows: u32, in: u32) !mlx.mlx_array {
    const xh = try a.alloc(f32, @as(usize, rows) * in);
    defer a.free(xh);
    for (xh, 0..) |*v, i| v.* = pool[i % pool.len];
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(in) }, .float32), .bfloat16);
    try g.evalAll(&.{x});
    return x;
}

/// root.linear at the pg2 switch `on` with `entry` the host's table (one linear's tile), restored after; whether the
/// fused launch served it
fn pg2Linear(g: *G, c: LinCtx, entry: ?*const [1]root.pgemm.Entry) !struct { y: mlx.mlx_array, fused: bool } {
    const saved_mode = root.pgemm.mode.get().*;
    const saved_table = root.pgemm.host_table.get().*;
    defer {
        root.pgemm.mode.get().* = saved_mode;
        root.pgemm.host_table.get().* = saved_table;
    }
    root.pgemm.mode.get().* = if (entry != null) .on else .off;
    root.pgemm.host_table.get().* = if (entry) |e| @as([]const root.pgemm.Entry, e) else null;
    const before = root.pgemm.calls.get().*.load(.monotonic);
    const y = try g.adopt(try root.linear(c.s, c.x, c.p.trellis, c.p.suh, c.p.svh, .mul1));
    try g.evalAll(&.{y});
    return .{ .y = y, .fused = root.pgemm.calls.get().*.load(.monotonic) > before };
}

/// The pg2 tests' rows: unaligned and aligned M; a row count runs where MLX would run its regular kernel
/// (`pgemm.mlxRegular`), the only rows pg2 takes (dsa.o_proj, K 16384, from 5462 rows)
const pg2_rows = [_]u32{ 100, 2048, 4100, 8200 };

test "turboexl3 gpu glm53f pg2: every candidate tile at every GLM-5.3-Flash trunk linear gives the chain's words (dequantize, MLX's matmul, the finish), bf16 x and y, at the rows MLX runs its regular kernel (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    defer root.releaseEngines();
    var prng = std.Random.DefaultPrng.init(0x9632F4);
    const rnd = prng.random();
    var pool: [8191]f32 = undefined;
    for (&pool) |*v| v.* = routed.bfExact(rnd.floatNorm(f32));
    var max_words: usize = 0;
    for (root.glm53f.trunk) |l| max_words = @max(max_words, @as(usize, pg2_rows[pg2_rows.len - 1]) * l.n);
    const want_buf = try a.alloc(f32, max_words);
    defer a.free(want_buf);
    const got_buf = try a.alloc(f32, max_words);
    defer a.free(got_buf);
    var failed = false;
    for (root.glm53f.trunk) |l| {
        defer root.releaseEngines();
        const mark = g.mark();
        defer g.resetTo(mark);
        const p = try trunkProj(&g, a, rnd, l.k, l.n);
        var fused_at: [root.pgemm.candidates.len]usize = @splat(0);
        for (pg2_rows) |rows| {
            if (!root.pgemm.mlxRegular(rows, l.n, l.k)) continue;
            const rm = g.mark();
            defer g.resetTo(rm);
            const c: LinCtx = .{ .s = s, .x = try trunkX(&g, a, &pool, rows, l.k), .p = p };
            const words = @as(usize, rows) * l.n;
            const want = try pg2Linear(&g, c, null);
            const wb = try bitsOf(&g, want.y, want_buf[0..words]);
            for (root.pgemm.candidates, 0..) |t, ti| {
                const cm = g.mark();
                defer g.resetTo(cm);
                const entry = [1]root.pgemm.Entry{.{ .k = l.k, .n = l.n, .tile = t }};
                const got = try pg2Linear(&g, c, &entry);
                const gb = try bitsOf(&g, got.y, got_buf[0..words]);
                const d = countDiff(gb, wb, got_buf[0..words]);
                const ok = got.fused and d[0] == 0 and d[1] == 0;
                fused_at[ti] += @intFromBool(got.fused);
                std.debug.print("TURBOEXL3_GLM_PG2_BYTES name={s} k={d} n={d} tile={c} rows={d} fused={} differing={d}/{d} nonfinite={d} {s}\n", .{ l.name, l.k, l.n, root.glm53f.codeOf(ti), rows, got.fused, d[0], words, d[1], if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            }
        }
        for (fused_at, 0..) |f, ti| if (f == 0) {
            std.debug.print("TURBOEXL3_GLM_PG2_BYTES name={s} k={d} n={d} tile={c} no row count fused FAIL\n", .{ l.name, l.k, l.n, root.glm53f.codeOf(ti) });
            failed = true;
        };
    }
    try testing.expect(!failed);
}

/// The pg2 timing's rows (pgemm.zig's pg2u rule: a tile at or below the chain at each of them)
const pg2_time_rows = [_]u32{ 2048, 8192, 16384 };

test "turboexl3 gpu glm53f pg2 timing: each candidate tile against the chain at every GLM-5.3-Flash trunk linear, at 2048, 8192 and 16384 rows, and the tile code a linear gets by pg2u's rule (TURBOEXL3_GLM_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_GLM_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    defer root.releaseEngines();
    var prng = std.Random.DefaultPrng.init(0x96327E);
    const rnd = prng.random();
    var pool: [8191]f32 = undefined;
    for (&pool) |*v| v.* = routed.bfExact(rnd.floatNorm(f32));
    const nc = root.pgemm.candidates.len;
    var codes: [root.glm53f.trunk.len]u8 = undefined;
    var saved_ms: f64 = 0;
    for (root.glm53f.trunk, &codes) |l, *code| {
        defer root.releaseEngines();
        const mark = g.mark();
        defer g.resetTo(mark);
        const p = try trunkProj(&g, a, rnd, l.k, l.n);
        var per_forward: u32 = 0;
        for (trunk_shapes) |t| if (std.mem.eql(u8, t.name, l.name)) {
            per_forward = t.per_forward;
        };
        var chain: [pg2_time_rows.len]f64 = undefined;
        var tt: [nc][pg2_time_rows.len]f64 = undefined;
        var taken: [pg2_time_rows.len]bool = undefined;
        for (pg2_time_rows, 0..) |rows, ri| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const c: LinCtx = .{ .s = s, .x = try trunkX(&g, a, &pool, rows, l.k), .p = p };
            taken[ri] = root.pgemm.mlxRegular(rows, l.n, l.k);
            const flops = 2.0 * @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(l.k)) * @as(f64, @floatFromInt(l.n));
            chain[ri] = (try timeCall(&g, c, struct {
                fn f(gg: *G, cc: LinCtx) !mlx.mlx_array {
                    return (try pg2Linear(gg, cc, null)).y;
                }
            }.f))[0];
            std.debug.print("TURBOEXL3_GLM_PG2_TIME name={s} k={d} n={d} tile=chain rows={d} min_us={d:.1} tflops={d:.2}\n", .{ l.name, l.k, l.n, rows, chain[ri], flops / (chain[ri] * 1e6) });
            for (root.pgemm.candidates, 0..) |t, ti| {
                if (!taken[ri]) {
                    tt[ti][ri] = chain[ri];
                    continue;
                }
                const Ctx = struct { c: LinCtx, e: *const [1]root.pgemm.Entry };
                const entry = [1]root.pgemm.Entry{.{ .k = l.k, .n = l.n, .tile = t }};
                tt[ti][ri] = (try timeCall(&g, Ctx{ .c = c, .e = &entry }, struct {
                    fn f(gg: *G, cc: Ctx) !mlx.mlx_array {
                        const r = try pg2Linear(gg, cc.c, cc.e);
                        if (!r.fused) return error.Pg2NotFused;
                        return r.y;
                    }
                }.f))[0];
                std.debug.print("TURBOEXL3_GLM_PG2_TIME name={s} k={d} n={d} tile={c} rows={d} min_us={d:.1} tflops={d:.2} ratio={d:.3}\n", .{ l.name, l.k, l.n, root.glm53f.codeOf(ti), rows, tt[ti][ri], flops / (tt[ti][ri] * 1e6), tt[ti][ri] / chain[ri] });
            }
        }
        // pg2u's rule: a tile at or below the chain at every row count it takes, the fastest at the widest of them
        const last = pg2_time_rows.len - 1;
        var best: ?usize = null;
        for (0..nc) |ti| {
            var ok = true;
            for (0..pg2_time_rows.len) |ri| ok = ok and tt[ti][ri] <= chain[ri];
            if (ok and taken[last] and (best == null or tt[ti][last] < tt[best.?][last])) best = ti;
        }
        code.* = if (best) |b| root.glm53f.codeOf(b) else '0';
        const cut_ms = if (best) |b| (chain[last] - tt[b][last]) * @as(f64, @floatFromInt(per_forward)) / 1000.0 else 0;
        saved_ms += cut_ms;
        std.debug.print("TURBOEXL3_GLM_PG2_PICK name={s} k={d} n={d} tile={c} ratios={d:.3}/{d:.3}/{d:.3} per_forward={d} forward_ms_cut={d:.2}\n", .{ l.name, l.k, l.n, code.*, if (best) |b| tt[b][0] / chain[0] else 1.0, if (best) |b| tt[b][1] / chain[1] else 1.0, if (best) |b| tt[b][last] / chain[last] else 1.0, per_forward, cut_ms });
    }
    std.debug.print("TURBOEXL3_GLM_PG2_CODES codes={s} forward_ms_cut={d:.2} (one code a linear in root.glm53f.trunk's order; each linear alone at 16384 rows times its count in a forward)\n", .{ &codes, saved_ms });
}

/// The routed prefill's opt-in options (the plugin's TURBOEXL3_GLM53F_PF_OPTS): `bk128`, `half_rows`, both; the 128-row
/// tile launches' 128-deep stages (`t128_bk128`, `tk`) alone and with `bk128`
const PfOpts = struct { name: []const u8, bk128: bool, half_rows: bool, t128_bk128: bool = false };
const pf_opts = [_]PfOpts{ .{ .name = "bk128", .bk128 = true, .half_rows = false }, .{ .name = "hr", .bk128 = false, .half_rows = true }, .{ .name = "bk128.hr", .bk128 = true, .half_rows = true }, .{ .name = "tk", .bk128 = false, .half_rows = false, .t128_bk128 = true }, .{ .name = "bk128.tk", .bk128 = true, .half_rows = false, .t128_bk128 = true } };
/// The served options (null), then each of `pf_opts`
const pf_sets = [_]?PfOpts{ null, pf_opts[0], pf_opts[1], pf_opts[2], pf_opts[3], pf_opts[4] };

/// `configure` with the routed prefill options `o` (null: the served ones), every engine released first so the next
/// call's prefill engine is made under them
fn pfOptsConfigure(saved: root.config.Config, o: ?PfOpts) !void {
    root.releaseEngines();
    var c = saved;
    c.pf_bk128 = if (o) |v| v.bk128 else saved.pf_bk128;
    c.pf_half_rows = if (o) |v| v.half_rows else saved.pf_half_rows;
    c.pf_t128_bk128 = if (o) |v| v.t128_bk128 else saved.pf_t128_bk128;
    try root.configure(c);
}

test "turboexl3 gpu glm53f prefill options: bk128, half_rows and both give a prompt call's served words, 16384 row-slots in 4096-row calls on the plugin's default path and on moeClampedRows, at every SC rate (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = root.config.get().*;
    defer pfOptsConfigure(saved, null) catch unreachable;
    var nbuf: [8]u32 = undefined;
    const rates = routed.envList("TURBOEXL3_GLM_ROUTED_N", &routed.rates_n, &nbuf);
    const S: usize = 16384;
    const chunk: usize = 4096;
    const want_buf = try a.alloc(f32, S * hidden);
    defer a.free(want_buf);
    const got_buf = try a.alloc(f32, S * hidden);
    defer a.free(got_buf);
    const ones_h = try a.alloc(f32, S);
    defer a.free(ones_h);
    @memset(ones_h, 1.0);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        defer root.releaseEngines();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x61F0B7 ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, prompt_slots, hidden, inter, rate);
        defer bank.deinit(a);
        const rb = try routed.rootBank(&g, &bank, .uint16);
        const xh = try a.alloc(f32, act_rows * hidden);
        defer a.free(xh);
        for (xh) |*v| v.* = routed.bfExact(sigma * rnd.floatNorm(f32));
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(act_rows), ci(hidden) }, .float32), .bfloat16);
        try g.evalAll(&.{x});
        const rows = try a.alloc(u32, S);
        defer a.free(rows);
        const slots = try a.alloc(u32, S);
        defer a.free(slots);
        drawCall(rnd, rows, slots, prompt_slots, 0);
        try sortBySlot(a, rows, slots, prompt_slots);
        for ([_]bool{ false, true }) |on_rows| {
            // the served options' words, then each option set's
            for (pf_sets) |o| {
                try pfOptsConfigure(saved, o);
                const buf = if (o == null) want_buf else got_buf;
                var c0: usize = 0;
                while (c0 < S) : (c0 += chunk) {
                    const c1 = @min(c0 + chunk, S);
                    const m2 = g.mark();
                    defer g.resetTo(m2);
                    const y = try chunkCall(&g, s, x, rb, rows[c0..c1], slots[c0..c1], ones_h, on_rows);
                    _ = try bitsOf(&g, y, buf[c0 * hidden .. c1 * hidden]);
                }
                const v = o orelse continue;
                const want: []const u32 = @as([*]const u32, @ptrCast(want_buf.ptr))[0 .. S * hidden];
                const got: []const u32 = @as([*]const u32, @ptrCast(got_buf.ptr))[0 .. S * hidden];
                const d = countDiff(got, want, got_buf);
                const ok = d[0] == 0 and d[1] == 0;
                std.debug.print("TURBOEXL3_GLM_PF_OPTS_BYTES n={d} S={d} path={s} opts={s} against=served differing={d}/{d} nonfinite={d} {s}\n", .{ n, S, if (on_rows) "rows" else "chain", v.name, d[0], S * hidden, d[1], if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            }
        }
        try pfOptsConfigure(saved, null);
    }
    try testing.expect(!failed);
}

test "turboexl3 gpu glm53f prefill options timing: a 16384-row prompt bank call under the served options, bk128, half_rows and both, in calls of 4096 and 16384 rows on each path, over 36 experts' 455-row runs and over a drawn routing (TURBOEXL3_GLM_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_GLM_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    const saved = root.config.get().*;
    defer pfOptsConfigure(saved, null) catch unreachable;
    var prng = std.Random.DefaultPrng.init(0x61F0B8);
    const rnd = prng.random();
    // K3, the sc_3.10bpw pack's routed layers
    const n: u32 = 48;
    const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
    const mark = g.mark();
    defer g.resetTo(mark);
    var bank = try ref.synthBank(a, rnd, prompt_slots, hidden, inter, rate);
    defer bank.deinit(a);
    const rb = try routed.rootBank(&g, &bank, .uint16);
    const xh = try a.alloc(f32, act_rows * hidden);
    defer a.free(xh);
    for (xh) |*v| v.* = routed.bfExact(sigma * rnd.floatNorm(f32));
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(act_rows), ci(hidden) }, .float32), .bfloat16);
    try g.evalAll(&.{x});
    const S: usize = 16384;
    const ones_h = try a.alloc(f32, S);
    defer a.free(ones_h);
    @memset(ones_h, 1.0);
    for ([_][]const u8{ "runs455", "drawn" }) |routing| {
        const rows = try a.alloc(u32, S);
        defer a.free(rows);
        const slots = try a.alloc(u32, S);
        defer a.free(slots);
        if (std.mem.eql(u8, routing, "drawn")) {
            drawCall(rnd, rows, slots, prompt_slots, 0);
            try sortBySlot(a, rows, slots, prompt_slots);
        } else for (rows, slots, 0..) |*r, *sl, i| {
            r.* = rnd.uintLessThan(u32, act_rows);
            // the run's index in u32 first: @min with a comptime bound narrows its type (u6 here), and 35 * 2 needs 7 bits
            const run: u32 = @intCast(@min(i / 455, 35));
            sl.* = run * 2;
        }
        for (pf_sets) |o| {
            try pfOptsConfigure(saved, o);
            for ([_]bool{ false, true }) |on_rows| for ([_]usize{ 4096, 16384 }) |chunk| {
                var t: [8]f64 = undefined;
                for (0..t.len + 2) |round| {
                    const m = g.mark();
                    defer g.resetTo(m);
                    var outs: [8]mlx.mlx_array = undefined;
                    var no: usize = 0;
                    const t0 = nowNs();
                    var c0: usize = 0;
                    while (c0 < S) : (c0 += chunk) {
                        const c1 = @min(c0 + chunk, S);
                        outs[no] = try chunkCall(&g, s, x, rb, rows[c0..c1], slots[c0..c1], ones_h, on_rows);
                        no += 1;
                    }
                    try g.evalAll(outs[0..no]);
                    const dt = nowNs() - t0;
                    if (round >= 2) t[round - 2] = @as(f64, @floatFromInt(dt)) / 1e6;
                }
                std.mem.sort(f64, &t, {}, std.sort.asc(f64));
                const flops = 2.0 * 3.0 * @as(f64, @floatFromInt(S)) * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter));
                std.debug.print("TURBOEXL3_GLM_PF_OPTS_TIME n={d} S={d} routing={s} path={s} chunk={d} opts={s} min_ms={d:.2} med_ms={d:.2} tflops={d:.2}\n", .{ n, S, routing, if (on_rows) "rows" else "chain", chunk, if (o) |v| v.name else "served", t[0], t[t.len / 2], flops / (t[0] * 1e9) });
            };
        }
        try pfOptsConfigure(saved, null);
    }
}

/// `configure` with the routed gate|up and down on int8 at `v` (`Config.pf_int8`; 0 the exact chain) and the size classes
/// (`Config.pf_size_class`, the plugin's PF_REUSE), every engine released first so the next call's prefill engine is made
/// under it
fn pfI8Configure(saved: root.config.Config, v: u8) !void {
    root.releaseEngines();
    var c = saved;
    c.pf_int8 = v;
    c.pf_size_class = true;
    try root.configure(c);
}

/// `saved` back in force, every engine released first
fn pfI8Restore(saved: root.config.Config) void {
    root.releaseEngines();
    root.configure(saved) catch unreachable;
}

test "turboexl3 glm53f prefill int8 options: int8 1 or 2 rides the 128-row tiles on the deferred MUL1 map, the mixed outputs at their sorted positions, beside the size classes" {
    const t = testing;
    const served: prefill.Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .defer_affine = true, .ncb = 4, .sintr = true, .froute = true, .alay = 2, .t128 = true };
    try t.expect(served.valid() and served.int8 == 0);
    for ([_]u8{ 1, 2 }) |v| {
        var o = served;
        o.int8 = v;
        try t.expect(o.valid());
        var z = o;
        z.t128_zfit = true;
        try t.expect(!z.valid());
        var d = o;
        d.defer_affine = false;
        try t.expect(!d.valid());
        var q = o;
        q.t128 = false;
        try t.expect(!q.valid());
        var c = o;
        c.size_class = true;
        try t.expect(c.valid());
    }
    var o3 = served;
    o3.int8 = 3;
    try t.expect(!o3.valid());
}

test "turboexl3 gpu glm53f prefill int8: the routed gate|up and down on int8 (a scale each activation 128-block; one scale a row) agree with the exact chain (cosine >= 0.99, error rms <= 0.1 of y's, finite), 16384 row-slots in 4096-row calls with the size classes, on the plugin's default path and on moeClampedRows, at every SC rate (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = root.config.get().*;
    defer pfI8Restore(saved);
    var nbuf: [8]u32 = undefined;
    const rates = routed.envList("TURBOEXL3_GLM_ROUTED_N", &routed.rates_n, &nbuf);
    const S: usize = 16384;
    const chunk: usize = 4096;
    const want = try a.alloc(f32, S * hidden);
    defer a.free(want);
    const got = try a.alloc(f32, S * hidden);
    defer a.free(got);
    const ones_h = try a.alloc(f32, S);
    defer a.free(ones_h);
    @memset(ones_h, 1.0);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        defer root.releaseEngines();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x618A9 ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, prompt_slots, hidden, inter, rate);
        defer bank.deinit(a);
        const rb = try routed.rootBank(&g, &bank, .uint16);
        const xh = try a.alloc(f32, act_rows * hidden);
        defer a.free(xh);
        for (xh) |*v| v.* = routed.bfExact(sigma * rnd.floatNorm(f32));
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(act_rows), ci(hidden) }, .float32), .bfloat16);
        try g.evalAll(&.{x});
        const rows = try a.alloc(u32, S);
        defer a.free(rows);
        const slots = try a.alloc(u32, S);
        defer a.free(slots);
        drawCall(rnd, rows, slots, prompt_slots, 0);
        try sortBySlot(a, rows, slots, prompt_slots);
        for ([_]bool{ false, true }) |on_rows| {
            // the exact chain's y, then each int8 scheme's
            for ([_]u8{ 0, 1, 2 }) |v| {
                try pfI8Configure(saved, v);
                const buf = if (v == 0) want else got;
                var c0: usize = 0;
                while (c0 < S) : (c0 += chunk) {
                    const c1 = @min(c0 + chunk, S);
                    const m2 = g.mark();
                    defer g.resetTo(m2);
                    const y = try chunkCall(&g, s, x, rb, rows[c0..c1], slots[c0..c1], ones_h, on_rows);
                    _ = try g.hostF32(try g.astype(y, .float32), buf[c0 * hidden .. c1 * hidden]);
                }
                if (v == 0) continue;
                var dot: f64 = 0;
                var gg: f64 = 0;
                var ww: f64 = 0;
                var ee: f64 = 0;
                var nonfinite: usize = 0;
                for (got, want) |gv, wv| {
                    if (!std.math.isFinite(gv) or !std.math.isFinite(wv)) {
                        nonfinite += 1;
                        continue;
                    }
                    dot += @as(f64, gv) * wv;
                    gg += @as(f64, gv) * gv;
                    ww += @as(f64, wv) * wv;
                    ee += (@as(f64, gv) - wv) * (@as(f64, gv) - wv);
                }
                const cos = dot / @sqrt(gg * ww);
                const rms = @sqrt(ee / ww);
                const ok = nonfinite == 0 and cos >= 0.99 and rms <= 0.1;
                std.debug.print("TURBOEXL3_GLM_PF_INT8_AGREE n={d} S={d} int8={d} path={s} cos={d:.6} rms_ratio={d:.5} nonfinite={d} {s}\n", .{ n, S, v, if (on_rows) "rows" else "chain", cos, rms, nonfinite, if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            }
        }
    }
    try testing.expect(!failed);
}

/// The routed prefill's route, prepare and gate|up launches alone over a prompt call's row-slots (`Prefill.routeT128`,
/// then `prepWith` and `gemmGUT128`, or with `q8` `prepI8` and `gemmGUI8`), every array adopted by `g`: the gate|up
/// outputs of both launches
fn guStage(g: *G, pf: *prefill.Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, rows_a: mlx.mlx_array, inds: mlx.mlx_array, ops: prefill.Operands, S: u32, q8: bool) ![4]mlx.mlx_array {
    const rt = try pf.routeT128(s, inds, S);
    for ([_]mlx.mlx_array{ rt.order, rt.tiles, rt.tiles128, rt.xbase }) |r| _ = try g.adopt(r);
    var zt: prefill.ZT128 = undefined;
    if (q8) {
        const po = try pf.prepI8(s, x, inds, rt.order, ops, S, rows_a);
        for (po) |p| _ = try g.adopt(p);
        zt = try pf.gemmGUI8(s, rt, po[0], po[1], po[5], po[6], po[3], po[4], ops, S);
    } else {
        const po = try pf.prepWith(s, x, inds, rt.order, ops, S, rows_a);
        for (po) |p| _ = try g.adopt(p);
        zt = try pf.gemmGUT128(s, rt, po[0], po[1], po[3], po[4], ops, S);
    }
    return .{ try g.adopt(zt.m[0]), try g.adopt(zt.m[1]), try g.adopt(zt.t[0]), try g.adopt(zt.t[1]) };
}

/// The routed prefill's launches through mid and the down GEMM over a prompt call's row-slots (`guStage`'s, then
/// `midT128` and `gemmDT128`, or with `q8` `midI8` and `gemmDI8`), every array adopted by `g`: the down outputs of both
/// launches
fn dnStage(g: *G, pf: *prefill.Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, rows_a: mlx.mlx_array, inds: mlx.mlx_array, ops: prefill.Operands, S: u32, q8: bool) ![2]mlx.mlx_array {
    const rt = try pf.routeT128(s, inds, S);
    for ([_]mlx.mlx_array{ rt.order, rt.tiles, rt.tiles128, rt.xbase }) |r| _ = try g.adopt(r);
    var zd: prefill.ZT128 = undefined;
    if (q8) {
        const po = try pf.prepI8(s, x, inds, rt.order, ops, S, rows_a);
        for (po) |p| _ = try g.adopt(p);
        const zt = try pf.gemmGUI8(s, rt, po[0], po[1], po[5], po[6], po[3], po[4], ops, S);
        for ([_]mlx.mlx_array{ zt.m[0], zt.m[1], zt.t[0], zt.t[1] }) |z| _ = try g.adopt(z);
        const xd = try pf.midI8(s, zt, rt, inds, ops, S);
        for (xd) |d| _ = try g.adopt(d);
        zd = try pf.gemmDI8(s, rt, xd[0], xd[2], xd[1], ops, S);
    } else {
        const po = try pf.prepWith(s, x, inds, rt.order, ops, S, rows_a);
        for (po) |p| _ = try g.adopt(p);
        const zt = try pf.gemmGUT128(s, rt, po[0], po[1], po[3], po[4], ops, S);
        for ([_]mlx.mlx_array{ zt.m[0], zt.m[1], zt.t[0], zt.t[1] }) |z| _ = try g.adopt(z);
        const xd = try pf.midT128(s, zt, rt, inds, ops, S);
        for (xd) |d| _ = try g.adopt(d);
        zd = try pf.gemmDT128(s, rt, xd[0], xd[1], ops, S);
    }
    return .{ try g.adopt(zd.m[0]), try g.adopt(zd.t[0]) };
}

test "turboexl3 gpu glm53f prefill int8 timing: a 16384-row prompt bank call with the routed gate|up and down on int8 (1: a scale each activation 128-block; 2: one scale a row) against the exact chain, in calls of 4096 and 16384 rows on each path with the size classes, and its route, prepare and gate|up alone and through mid and down, over 36 experts' 455-row runs and over a drawn routing (TURBOEXL3_GLM_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_GLM_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    const saved = root.config.get().*;
    defer pfI8Restore(saved);
    var prng = std.Random.DefaultPrng.init(0x618AA);
    const rnd = prng.random();
    // K3, the sc_3.10bpw pack's routed layers
    const n: u32 = 48;
    const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
    const mark = g.mark();
    defer g.resetTo(mark);
    var bank = try ref.synthBank(a, rnd, prompt_slots, hidden, inter, rate);
    defer bank.deinit(a);
    const rb = try routed.rootBank(&g, &bank, .uint16);
    const xh = try a.alloc(f32, act_rows * hidden);
    defer a.free(xh);
    for (xh) |*v| v.* = routed.bfExact(sigma * rnd.floatNorm(f32));
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(act_rows), ci(hidden) }, .float32), .bfloat16);
    try g.evalAll(&.{x});
    const S: usize = 16384;
    const ones_h = try a.alloc(f32, S);
    defer a.free(ones_h);
    @memset(ones_h, 1.0);
    const arms = [_][]const u8{ "exact", "i8", "i8t" };
    // the stages' own engine: the served options (root.pfOptions with the plugin's bk128) with int8 on; the exact
    // launches beside it, the scheme by its option
    var pf = try prefill.Prefill.init(.{ .hidden = hidden, .inter = inter, .topk = 1, .experts = prompt_slots, .rate = rate, .dec = .mul1, .limit = @intCast(limit) }, .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .defer_affine = true, .ncb = 4, .sintr = true, .froute = true, .alay = 2, .t128 = true, .bk128 = true, .int8 = 1 });
    defer pf.deinit();
    const ops: prefill.Operands = .{ .gate_t = rb.gate.trellis, .gate_suh = rb.gate.suh, .gate_svh = rb.gate.svh, .up_t = rb.up.trellis, .up_suh = rb.up.suh, .up_svh = rb.up.svh, .down_t = rb.down.trellis, .down_suh = rb.down.suh, .down_svh = rb.down.svh };
    for ([_][]const u8{ "runs455", "drawn" }) |routing| {
        const rows = try a.alloc(u32, S);
        defer a.free(rows);
        const slots = try a.alloc(u32, S);
        defer a.free(slots);
        if (std.mem.eql(u8, routing, "drawn")) {
            drawCall(rnd, rows, slots, prompt_slots, 0);
            try sortBySlot(a, rows, slots, prompt_slots);
        } else for (rows, slots, 0..) |*r, *sl, i| {
            r.* = rnd.uintLessThan(u32, act_rows);
            const run: u32 = @intCast(@min(i / 455, 35));
            sl.* = run * 2;
        }
        for (arms, 0..) |arm, v| {
            try pfI8Configure(saved, @intCast(v));
            for ([_]bool{ false, true }) |on_rows| for ([_]usize{ 4096, 16384 }) |chunk| {
                var t: [8]f64 = undefined;
                for (0..t.len + 2) |round| {
                    const m = g.mark();
                    defer g.resetTo(m);
                    var outs: [8]mlx.mlx_array = undefined;
                    var no: usize = 0;
                    const t0 = nowNs();
                    var c0: usize = 0;
                    while (c0 < S) : (c0 += chunk) {
                        const c1 = @min(c0 + chunk, S);
                        outs[no] = try chunkCall(&g, s, x, rb, rows[c0..c1], slots[c0..c1], ones_h, on_rows);
                        no += 1;
                    }
                    try g.evalAll(outs[0..no]);
                    const dt = nowNs() - t0;
                    if (round >= 2) t[round - 2] = @as(f64, @floatFromInt(dt)) / 1e6;
                }
                std.mem.sort(f64, &t, {}, std.sort.asc(f64));
                const flops = 2.0 * 3.0 * @as(f64, @floatFromInt(S)) * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter));
                std.debug.print("TURBOEXL3_GLM_PF_INT8_TIME n={d} S={d} routing={s} path={s} chunk={d} arm={s} min_ms={d:.2} med_ms={d:.2} tflops={d:.2}\n", .{ n, S, routing, if (on_rows) "rows" else "chain", chunk, arm, t[0], t[t.len / 2], flops / (t[0] * 1e9) });
            };
        }
        try pfI8Configure(saved, 0);
        // the route, prepare and gate|up launches alone, then through mid and down, in 4096-row calls over x's rows
        // (moeClampedRows' row map)
        for ([_]bool{ false, true }) |down| for (arms, 0..) |arm, v| {
            pf.opts.int8 = if (v == 0) 1 else @intCast(v);
            var t: [8]f64 = undefined;
            for (0..t.len + 2) |round| {
                const m = g.mark();
                defer g.resetTo(m);
                var outs: [16]mlx.mlx_array = undefined;
                var no: usize = 0;
                const t0 = nowNs();
                var c0: usize = 0;
                while (c0 < S) : (c0 += 4096) {
                    const c1 = @min(c0 + 4096, S);
                    const rows_a = try g.hostArray(std.mem.sliceAsBytes(rows[c0..c1]), &.{ci(c1 - c0)}, .uint32);
                    const inds = try g.hostArray(std.mem.sliceAsBytes(slots[c0..c1]), &.{ci(c1 - c0)}, .uint32);
                    if (down) {
                        const z = try dnStage(&g, &pf, s, x, rows_a, inds, ops, @intCast(c1 - c0), v != 0);
                        @memcpy(outs[no..][0..2], &z);
                        no += 2;
                    } else {
                        const z = try guStage(&g, &pf, s, x, rows_a, inds, ops, @intCast(c1 - c0), v != 0);
                        @memcpy(outs[no..][0..4], &z);
                        no += 4;
                    }
                }
                try g.evalAll(outs[0..no]);
                const dt = nowNs() - t0;
                if (round >= 2) t[round - 2] = @as(f64, @floatFromInt(dt)) / 1e6;
            }
            std.mem.sort(f64, &t, {}, std.sort.asc(f64));
            const flops = 2.0 * @as(f64, if (down) 3.0 else 2.0) * @as(f64, @floatFromInt(S)) * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter));
            std.debug.print("TURBOEXL3_GLM_PF_INT8_STAGE n={d} S={d} routing={s} chunk=4096 stage={s} arm={s} min_ms={d:.2} med_ms={d:.2} tflops={d:.2}\n", .{ n, S, routing, if (down) "route+prep+gate_up+mid+down" else "route+prep+gate_up", arm, t[0], t[t.len / 2], flops / (t[0] * 1e9) });
        };
        pf.opts.int8 = 1;
    }
}
