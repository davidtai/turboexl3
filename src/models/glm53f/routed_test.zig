//! GLM-5.3-Flash's routed experts on turboexl3: hidden 4096, expert width 2048, 288 experts, top-8, `swiglu_limit` 10 (the
//! clamp DeepSeek-V4 and V4.1 share: the gate clamped from above, the up to [-10, 10], before the SiLU), MUL1 at every
//! rate turboderp/GLM-5.3-Flash-exl3's self-calibrated branches store a routed layer at: K2 (sc_2.10bpw) and its one
//! K3 layer, K2.5 (sc_2.60bpw: exllamav3 1.6's 40-halfword rate) and its one K3.5 layer (56 halfwords), K3 (sc_3.10bpw)
//! and its one K4 layer, K4 (sc_4.10bpw) and its one K5 layer; the MTP layer's experts at K3 and K4 (a resident bank).
//! - CPU (always): the routed geometry is admitted by the decode and prefill engines at every rate, its stacked
//!   trellises are admitted at and under each rate's ceiling, and the packs' `expert_quant` K values parse to those rates.
//! - GPU (TURBOEXL3_GPU=1, else skipped):
//!   - `root.moeClamped` at that geometry against the clamped reference (`ref.moeRowClamped`) on sampled rows, at decode
//!     rows 1, 4 (the MTP verify), 16 and prefill rows 32, 1328, bf16 and f32 x, within 5e-3 (the clamped test's bar);
//!   - `root.moeClampedBanked` over three slot banks of 20, 12 and 8 experts (an expert streamer's base, ext and
//!     transient rows) at every rate: each pair's row byte for byte `moeClamped`'s on its bank at top-1 with score 1, at
//!     pair counts 1 to 64 (a 4-row verify is 32 pairs), x bf16 and the trellis I16 as the streamer's code arrays come.
//!   x ~ N(0, 8^2): about a tenth of the gates and a quarter of the ups pass the limit; each check requires 1% of each.
//!   TURBOEXL3_GLM_ROUTED_N (rates in halfwords a tile, a comma list; default 32,40,48,56,64,80) and
//!   TURBOEXL3_GLM_ROUTED_ROWS (default 1,4,16,32,1328) narrow the first, TURBOEXL3_GLM_BANKED_ROWS (default
//!   1,2,7,8,16,32,48,64) the second. One rate's bank is held twice (host and device): 2 x 2.3 GB at K2 up to 2 x 5.7 GB at K5.
//!   Lines: TURBOEXL3_GLM_ROUTED_PARITY, TURBOEXL3_GLM_BANKED_BYTES.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const spec = @import("../../core/spec.zig");
const decode = @import("../../core/decode.zig");
const prefill = @import("../../core/prefill.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");

const testing = std.testing;

pub const hidden: u32 = 4096;
pub const inter: u32 = 2048;
pub const experts: u32 = 288;
pub const top_k: u32 = 8;
pub const limit: c_int = 10;
const sigma: f32 = 8;
/// Every routed rate of the four SC packs, in halfwords a 256-weight tile (16 K): K2, K2.5, K3, K3.5, K4, K5.
pub const rates_n = [_]u32{ 32, 40, 48, 56, 64, 80 };
/// The banked parity test's slot banks: base, ext, transient.
const bank_experts = [3]u32{ 20, 12, 8 };

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// `v` rounded to bf16 (as MLX rounds), back in f32: a value both sides read alike.
fn bfExact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
}

test "turboexl3 glm53f routed: GLM-5.3-Flash's routed geometry (4096 x 2048, 288 experts, top-8, limit 10) is admitted at every SC rate, its stacked trellises at and under each ceiling, its expert_quant K values parse to those rates" {
    const ks = [_][]const u8{ "2", "2.5", "3", "3.5", "4", "5" };
    for (rates_n, ks) |n, k| {
        const rate = format.Rate.fromN(n) orelse return error.TestUnexpectedResult;
        try testing.expectEqual(n, rate.n);
        const dgeo: decode.Geometry = .{ .hidden = hidden, .inter = inter, .topk = top_k, .rate = rate, .dec = .mul1, .limit = limit };
        try testing.expect(dgeo.admitted());
        // the MTP verify's routed pairs (4 rows of 8) and a top-1 pair call both fit the decode chain's reduce bank
        try spec.admitTopK(top_k);
        const pgeo: prefill.Geometry = .{ .hidden = hidden, .inter = inter, .topk = top_k, .experts = experts, .rate = rate, .dec = .mul1, .limit = limit };
        try testing.expect(pgeo.admitted());
        const one: prefill.Geometry = .{ .hidden = hidden, .inter = inter, .topk = 1, .experts = experts, .rate = rate, .dec = .mul1, .limit = limit };
        try testing.expect(one.admitted());
        // gate/up [288, 256, 128, n], down [288, 128, 256, n]
        try testing.expect(spec.trellisAdmitted(&.{ ci(experts), ci(hidden / 16), ci(inter / 16), ci(n) }, experts, hidden, inter, rate));
        try testing.expect(spec.trellisAdmitted(&.{ ci(experts), ci(inter / 16), ci(hidden / 16), ci(n) }, experts, inter, hidden, rate));
        // a layer below the ceiling (the packs' K2.5 layers under a K3.5 ceiling) is admitted; one above it is not
        if (n > 32) try testing.expect(spec.trellisAdmitted(&.{ ci(experts), ci(hidden / 16), ci(inter / 16), ci(n - 8) }, experts, hidden, inter, rate));
        try testing.expect(!spec.trellisAdmitted(&.{ ci(experts), ci(hidden / 16), ci(inter / 16), ci(n + 8) }, experts, hidden, inter, rate));
        var buf: [96]u8 = undefined;
        const raw = try std.fmt.bufPrint(&buf, "{{\"expert_quant\":{{\"format\":\"exl3\",\"k\":{s},\"codebook\":\"mul1\"}}}}", .{k});
        const sp = try spec.specFromConfigJson(testing.allocator, raw);
        try testing.expectEqual(n, sp.rate.n);
    }
}

/// The bank as `root` takes it: each projection's arrays from the host bank, its trellis in `td` (U16 for `moeClamped`,
/// I16 as an expert streamer's code arrays come: the same words).
fn rootBank(g: *G, b: *const ref.Bank, td: mlx.mlx_dtype) !root.Bank {
    const E = ci(b.experts);
    const ht = ci(b.hidden / 16);
    const it = ci(b.inter / 16);
    const n = ci(b.rate.n);
    return .{
        .gate = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(b.gate_t), &.{ E, ht, it, n }, td), .suh = try g.hostArray(std.mem.sliceAsBytes(b.gate_suh), &.{ E, ci(b.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.gate_svh), &.{ E, ci(b.inter) }, .float16) },
        .up = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(b.up_t), &.{ E, ht, it, n }, td), .suh = try g.hostArray(std.mem.sliceAsBytes(b.up_suh), &.{ E, ci(b.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.up_svh), &.{ E, ci(b.inter) }, .float16) },
        .down = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(b.down_t), &.{ E, it, ht, n }, td), .suh = try g.hostArray(std.mem.sliceAsBytes(b.down_suh), &.{ E, ci(b.inter) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.down_svh), &.{ E, ci(b.hidden) }, .float16) },
    };
}

/// One call's routing: each row's top-k distinct experts by Gumbel top-k over lognormal popularity (sigma `skew`), scores
/// positive and summing to one per row before their bf16 rounding; row 0 takes the bank's last expert.
fn route(a: std.mem.Allocator, rnd: std.Random, rows: u32, n_experts: u32, topk: u32, skew: f64, slots: []u32, scores: []f32) !void {
    const logw = try a.alloc(f64, n_experts);
    defer a.free(logw);
    for (logw) |*w| w.* = skew * rnd.floatNorm(f64);
    const keys = try a.alloc(f64, n_experts);
    defer a.free(keys);
    for (0..rows) |r| {
        for (keys, logw) |*kv, lw| kv.* = lw - @log(-@log(@max(rnd.float(f64), 1e-300)));
        for (slots[r * topk ..][0..topk]) |*sl| {
            var best: usize = 0;
            for (keys, 0..) |kv, e| {
                if (kv > keys[best]) best = e;
            }
            sl.* = @intCast(best);
            keys[best] = -std.math.inf(f64);
        }
        const ws = scores[r * topk ..][0..topk];
        var sum: f32 = 0;
        for (ws) |*w| {
            w.* = 0.05 + rnd.float(f32);
            sum += w.*;
        }
        for (ws) |*w| w.* = bfExact(w.* / sum);
    }
    if (std.mem.indexOfScalar(u32, slots[0..topk], n_experts - 1) == null) slots[topk - 1] = n_experts - 1;
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

/// The rows a call's reference covers (it decodes every weight of their experts on the CPU): the first and the last, and
/// the middle one past 32 rows.
fn sampleRows(rows: u32, buf: *[3]u32) []const u32 {
    buf.* = .{ 0, rows - 1, rows / 2 };
    if (rows == 1) return buf[0..1];
    return if (rows <= 32) buf[0..2] else buf;
}

test "turboexl3 gpu glm53f routed: root.moeClamped at GLM-5.3-Flash's routed geometry (4096 x 2048, 288 experts, top-8, limit 10) equals the clamped reference at every SC rate (K2, K2.5, K3, K3.5, K4, K5), decode rows 1, 4, 16 and prefill rows 32, 1328 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_GLM_ROUTED_N", &rates_n, &nbuf);
    var rbuf: [8]u32 = undefined;
    const row_counts = envList("TURBOEXL3_GLM_ROUTED_ROWS", &.{ 1, 4, 16, 32, 1328 }, &rbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x61A5300 ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
        defer bank.deinit(a);
        const rb = try rootBank(&g, &bank, .uint16);
        for (row_counts) |rows| {
            if (rows == 0) continue;
            const m2 = g.mark();
            defer g.resetTo(m2);
            const x = try a.alloc(f32, rows * hidden);
            defer a.free(x);
            for (x) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
            const slots = try a.alloc(u32, rows * top_k);
            defer a.free(slots);
            const scores = try a.alloc(f32, rows * top_k);
            defer a.free(scores);
            try route(a, rnd, rows, experts, top_k, 1.5, slots, scores);
            const inds = try a.alloc(i32, rows * top_k);
            defer a.free(inds);
            for (slots, inds) |sl, *d| d.* = @intCast(sl);
            const x3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(hidden) }, .float32), .bfloat16);
            const inds3 = try g.hostArray(std.mem.sliceAsBytes(inds), &.{ 1, ci(rows), ci(top_k) }, .int32);
            const sc3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, ci(rows), ci(top_k) }, .float32), .bfloat16);
            var sbuf: [3]u32 = undefined;
            const sample = sampleRows(rows, &sbuf);
            const want = try a.alloc(f32, sample.len * hidden);
            defer a.free(want);
            var hits: ref.Hits = .{};
            for (sample, 0..) |r, i| {
                const h = try ref.moeRowClamped(a, &bank, .mul1, @floatFromInt(limit), x[r * hidden ..][0..hidden], slots[r * top_k ..][0..top_k], scores[r * top_k ..][0..top_k], want[i * hidden ..][0..hidden]);
                hits.gate += h.gate;
                hits.up += h.up;
                hits.total += h.total;
            }
            const hit = hits.gate * 100 >= hits.total and hits.up * 100 >= hits.total;
            const got = try a.alloc(f32, rows * hidden);
            defer a.free(got);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const y = try g.adopt(try root.moeClamped(s, try g.astype(x3, dt), rb, inds3, sc3, .mul1, limit));
                try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(hidden) }, mlx.getShape(y));
                try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
                _ = try g.hostF32(try g.astype(y, .float32), got);
                var bad: usize = 0;
                for (got) |v| bad += @intFromBool(!std.math.isFinite(v));
                // a bf16 output is judged beyond its store's rounding (2^-8 of each value), as the clamped test does
                var worst: f64 = 0;
                for (sample, 0..) |r, i| {
                    const gr = got[r * hidden ..][0..hidden];
                    const wr = want[i * hidden ..][0..hidden];
                    worst = @max(worst, (if (dt == .float32) ref.compare(gr, wr) else ref.compareRounded(gr, wr, 0x1p-8)).max_ratio);
                }
                const bar: f64 = 5e-3;
                const ok = hit and bad == 0 and worst <= bar;
                const path: []const u8 = if (rows <= root.kernels.DECODE_ROWS_MAX) "decode" else "prefill";
                std.debug.print("TURBOEXL3_GLM_ROUTED_PARITY n={d} rows={d} path={s} out={s} worst_vs_ref={e:.3} bar={e:.1} nonfinite={d} gates_clamped={d}/{d} ups_clamped={d}/{d} {s}\n", .{ n, rows, path, @tagName(dt), worst, bar, bad, hits.gate, hits.total, hits.up, hits.total, if (ok) "PASS" else "FAIL" });
                if (!ok) failed = true;
            }
        }
    }
    try testing.expect(!failed);
}

/// A call's pairs: row-slot r on bank `bank[r]` at slot row `slot[r]`, drawn over the banks; from 4 rows row-slot 1
/// repeats row-slot 0's id and row-slot 2 takes row-slot 0's slot row in the next bank.
fn drawPairs(rnd: std.Random, bank: []u32, slot: []u32) void {
    for (bank, slot) |*b, *sl| {
        b.* = rnd.uintLessThan(u32, 3);
        sl.* = rnd.uintLessThan(u32, bank_experts[b.*]);
    }
    if (bank.len >= 4) {
        slot[0] = rnd.uintLessThan(u32, @min(bank_experts[0], @min(bank_experts[1], bank_experts[2])));
        bank[1] = bank[0];
        slot[1] = slot[0];
        bank[2] = (bank[0] + 1) % 3;
        slot[2] = slot[0];
    }
}

test "turboexl3 gpu glm53f banked: every pair of moeClampedBanked over three slot banks is moeClamped's on its bank at top-1 with score 1, byte for byte, at GLM-5.3-Flash's routed geometry and every SC rate (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_GLM_ROUTED_N", &rates_n, &nbuf);
    var rbuf: [16]u32 = undefined;
    const row_counts = envList("TURBOEXL3_GLM_BANKED_ROWS", &.{ 1, 2, 7, 8, 16, 32, 48, 64 }, &rbuf);
    var failed = false;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x61AB4C ^ @as(u64, n));
        const rnd = prng.random();
        var hb: [3]ref.Bank = undefined;
        var made: usize = 0;
        defer for (hb[0..made]) |*b| b.deinit(a);
        for (&hb, bank_experts) |*b, e| {
            b.* = try ref.synthBank(a, rnd, e, hidden, inter, rate);
            made += 1;
        }
        var rb_u: [3]root.Bank = undefined;
        var rb_i: [3]root.Bank = undefined;
        for (&hb, &rb_u, &rb_i) |*b, *u, *i| {
            u.* = try rootBank(&g, b, .uint16);
            i.* = try rootBank(&g, b, .int16);
        }
        for (row_counts) |rows| {
            if (rows == 0 or rows > root.banked_rows_max) continue;
            const m2 = g.mark();
            defer g.resetTo(m2);
            const x = try a.alloc(f32, rows * hidden);
            defer a.free(x);
            for (x) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
            var bank_of: [root.banked_rows_max]u32 = undefined;
            var slot_of: [root.banked_rows_max]u32 = undefined;
            drawPairs(rnd, bank_of[0..rows], slot_of[0..rows]);
            var ids: [root.banked_rows_max]u32 = undefined;
            for (ids[0..rows], bank_of[0..rows], slot_of[0..rows]) |*d, b, sl| d.* = (b << 24) | sl;
            const x32 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ ci(rows), ci(hidden) }, .float32);
            const xbf = try g.astype(x32, .bfloat16);
            const ids_a = try g.hostArray(std.mem.sliceAsBytes(ids[0..rows]), &.{ci(rows)}, .uint32);
            // the reference bytes: each bank's pairs through moeClamped at top-1 with score 1, 16 rows a call (its decode
            // launches), x in f32 (x is bf16-exact, so the banked call's bf16 x holds the same values)
            const want = try a.alloc(f32, rows * hidden);
            defer a.free(want);
            const buf = try a.alloc(f32, 16 * hidden);
            defer a.free(buf);
            const ones: [16]f32 = @splat(1.0);
            for (0..3) |b| {
                var pos: [root.banked_rows_max]u32 = undefined;
                var np: usize = 0;
                for (bank_of[0..rows], 0..) |bk, r| if (bk == b) {
                    pos[np] = @intCast(r);
                    np += 1;
                };
                var c0: usize = 0;
                while (c0 < np) : (c0 += 16) {
                    const c = @min(16, np - c0);
                    const P = pos[c0 .. c0 + c];
                    var sl: [16]i32 = undefined;
                    for (P, sl[0..c]) |p, *d| d.* = @intCast(slot_of[p]);
                    const xr = try g.take(x32, try g.hostArray(std.mem.sliceAsBytes(P), &.{ci(c)}, .uint32), 0);
                    const x3 = try g.reshape(xr, &.{ 1, ci(c), ci(hidden) });
                    const inds = try g.hostArray(std.mem.sliceAsBytes(sl[0..c]), &.{ 1, ci(c), 1 }, .int32);
                    const sc = try g.hostArray(std.mem.sliceAsBytes(ones[0..c]), &.{ 1, ci(c), 1 }, .float32);
                    const y = try g.adopt(try root.moeClamped(s, x3, rb_u[b], inds, sc, .mul1, limit));
                    _ = try g.hostF32(y, buf[0 .. c * hidden]);
                    for (P, 0..) |p, i| @memcpy(want[p * hidden ..][0..hidden], buf[i * hidden ..][0..hidden]);
                }
            }
            const got = try a.alloc(f32, rows * hidden);
            defer a.free(got);
            const y = try g.adopt(try root.moeClampedBanked(s, xbf, &rb_i, ids_a, .mul1, limit));
            try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(hidden) }, mlx.getShape(y));
            try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(y));
            _ = try g.hostF32(y, got);
            var differing: usize = 0;
            var nonfinite: usize = 0;
            for (got, want) |u, v| {
                differing += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
                nonfinite += @intFromBool(!std.math.isFinite(u));
            }
            const ok = differing == 0 and nonfinite == 0;
            std.debug.print("TURBOEXL3_GLM_BANKED_BYTES n={d} rows={d} x=bf16 trellis=i16 differing={d}/{d} nonfinite={d} {s}\n", .{ n, rows, differing, rows * hidden, nonfinite, if (ok) "PASS" else "FAIL" });
            if (!ok) failed = true;
        }
    }
    try testing.expect(!failed);
}
