//! hw4b, the grouped banked GEMVs with an M = 3 member path (`decode.text_gu3bgm3`, `text_down3bgm3`; `Kernels3.BankedSel`;
//! `Config.banked_m3`): a lead with three members runs it, where today's grouped texts run the M = 4 path with a fourth
//! member's FMAs padding. CPU (always): the M = 3 texts are today's with the one M = 3 line added. GPU (TURBOEXL3_GPU=1,
//! else skipped), at DeepSeek-V4.1's routed geometry (hidden 5120, expert width 2304, swiglu_limit 10, MUL1) over three
//! synthetic banks of 20, 12 and 8 experts at K3 and K2 (TURBOEXL3_BANKED_N):
//! - g4m3's gate|up outputs (hd f16, hs f32) and down rows (y f32) are today's (g4), byte for byte (differing words 0,
//!   max_ulps 0; a skipped row-slot's y row 0), at rows 1 to 9, 12, 16, 24, 36, 42, 48 and 64 (TURBOEXL3_BANKED_ROWS), x
//!   bf16 and f32, the trellis I16 and U16, the pairs drawn so one id takes 1 to 16 row-slots, others 3 to 8, in shuffled
//!   slot order, two row-slots skipped.
//! With TURBOEXL3_BANKED_GRP_TIME=1 a further test times the whole banked call, K3 and K2, x bf16, trellis I16, on banks
//! of 48, 16 and 8 experts, for g4 (today's) and g4m3: one call an evaluation, and a chain of 40 dependent calls an
//! evaluation (call k + 1's x is call k's y), its time a call; 2 warm then 9 timed rounds, the arms interleaved with a
//! rotated start. The profiles (`count[c]` experts of c row-slots each): the recorded 36-pair verify profiles (rec),
//! width-7 and width-8 profiles from the recorded pairs-per-expert histogram (w7, w8), only 1-, 2- and 4-pair experts
//! (rp: the M = 3 text's own cost where no lead runs it), and the member sweep (D 16 at 1-4 row-slots an expert, D 8 at
//! 1-8).
//! TURBOEXL3_BANKED_GRP_ONLY_TIME skips the byte test (a window's timing run, after its cool gate).
//! Lines: TURBOEXL3_BANKED_GRP_BYTES, TURBOEXL3_BANKED_GRP_TIME.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const decode = @import("../../core/decode.zig");

const testing = std.testing;

const hidden: u32 = 5120;
const inter: u32 = 2304;
const limit: c_int = 10;
const sigma: f32 = 8;
/// the byte test's banks (base, ext, transient), and the timing test's (72 experts)
const bank_experts = [3]u32{ 20, 12, 8 };
const time_experts = [3]u32{ 48, 16, 8 };
const Sel = decode.Kernels3.BankedSel;
/// the arms: today's grouped texts first
const sels = [_]Sel{ .g4, .g4m3 };

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// `v` rounded to bf16 (as MLX rounds), back in f32: a value both sides read alike.
fn bfExact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
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

test "turboexl3 banked grp: the M = 3 texts are today's grouped texts with the one M = 3 line added after the M = 2 line" {
    for ([_][2][:0]const u8{ .{ decode.text_gu3bg, decode.text_gu3bgm3 }, .{ decode.text_down3bg, decode.text_down3bgm3 } }) |p| {
        const today = p[0];
        const m3 = p[1];
        try testing.expect(std.mem.indexOf(u8, today, "BTC, 3u, uint(DEFER)>") == null);
        const at = std.mem.indexOf(u8, m3, "        else if (m == 3u) turboexl3_gemv_group<").?;
        const line_end = std.mem.indexOfScalarPos(u8, m3, at, '\n').? + 1;
        try testing.expect(std.mem.indexOf(u8, m3[at..line_end], "BTC, 3u, uint(DEFER)>") != null);
        // the line follows the M = 2 line, and the text without it is today's
        try testing.expect(std.mem.endsWith(u8, m3[0..at], "BTC, 2u, uint(DEFER)>(wbase, xbase, ulong(HIDC), members, m, tk0, tk0 + per, blk, L, red[si][sg]);\n") or
            std.mem.endsWith(u8, m3[0..at], "BTC, 2u, uint(DEFER)>(wbase, xbase, ulong(INTC), members, m, tk0, tk0 + per, blk, L, red[si][sg]);\n"));
        try testing.expectEqual(today.len, m3.len - (line_end - at));
        try testing.expectEqualStrings(today[0..at], m3[0..at]);
        try testing.expectEqualStrings(today[at..], m3[line_end..]);
    }
}

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

fn evalBank(g: *G, b: root.Bank) !void {
    try g.evalAll(&.{ b.gate.trellis, b.gate.suh, b.gate.svh, b.up.trellis, b.up.suh, b.up.svh, b.down.trellis, b.down.suh, b.down.svh });
}

/// The banked launches' banks of three root banks.
fn kernelBanks(rb: *const [3]root.Bank) [3]decode.Kernels.Bank {
    var kb: [3]decode.Kernels.Bank = undefined;
    for (rb, &kb) |b, *o| o.* = .{ .gate_t = b.gate.trellis, .gate_suh = b.gate.suh, .gate_svh = b.gate.svh, .up_t = b.up.trellis, .up_suh = b.up.suh, .up_svh = b.up.svh, .down_t = b.down.trellis, .down_suh = b.down.suh, .down_svh = b.down.svh };
    return kb;
}

/// A Kernels3 at DeepSeek-V4.1's routed geometry (top-1, clamped, MUL1) as `root` builds it for a banked call, its
/// banked GEMVs grouped (4 row-slots a lead); the arms pick their texts per call (`BankedSel`).
fn groupedKernels3(rate: format.Rate) !decode.Kernels3 {
    var k = try decode.Kernels3.initWith(.{ .hidden = hidden, .inter = inter, .topk = 1, .rate = rate, .dec = .mul1, .limit = @intCast(limit) }, 1, 4, 2);
    k.out_dtype = .float32;
    k.banked_group = 4;
    return k;
}

/// Two arrays' f32 words compared (`row_w` words a row): the compared words, the differing words, the largest ulp
/// distance (|bits(a) - bits(b)| as integers), the non-finite words of `got`; a skipped row's words are 0 in both
/// (`zero_skipped`: y's, counted as skipped_nonzero) or not compared (hd's and hs's: the launches leave them unwritten).
const Diff = struct { words: usize = 0, differing: usize = 0, max_ulps: u64 = 0, nonfinite: usize = 0, skipped_nonzero: usize = 0 };
fn wordDiff(got: []const f32, want: []const f32, row_w: usize, skipped: []const usize, zero_skipped: bool) Diff {
    var r: Diff = .{};
    for (got, want, 0..) |u, v, i| {
        const bu: u32 = @bitCast(u);
        const bv: u32 = @bitCast(v);
        const row = i / row_w;
        var skip = false;
        for (skipped) |sk| skip = skip or sk == row;
        if (skip and !zero_skipped) continue;
        r.words += 1;
        if (bu != bv) {
            r.differing += 1;
            const iu: i64 = @as(i32, @bitCast(bu));
            const iv: i64 = @as(i32, @bitCast(bv));
            r.max_ulps = @max(r.max_ulps, @abs(iu - iv));
        }
        r.nonfinite += @intFromBool(!std.math.isFinite(u));
        if (skip) r.skipped_nonzero += @as(usize, @intFromBool(bu != 0)) + @as(usize, @intFromBool(bv != 0));
    }
    return r;
}

/// The packed id of global expert index e over the byte test's banks (base, then ext, then transient).
fn packedId(e: u32) u32 {
    if (e < bank_experts[0]) return e;
    if (e < bank_experts[0] + bank_experts[1]) return (1 << 24) | (e - bank_experts[0]);
    return (2 << 24) | (e - bank_experts[0] - bank_experts[1]);
}

/// A byte test call's pairs at `rows` row-slots (ids packed bank << 24 | slot row; skipped row-slots `banked_skip_id`).
/// From 10 rows: id A (base row 1) takes min(rows / 3, 16) row-slots (up to four GROUP 4 leads), id B (ext row 0) rows / 6
/// and id C (transient row 2) 3, the rest drawn, two row-slots skipped, the slot order shuffled; at 5 to 9 rows one id takes
/// rows - 1 row-slots (up to 8: two GROUP 4 leads from 5), the last skipped; below that drawn, row-slot 1
/// repeating row-slot 0 from 2 rows. Returns the skipped row-slots.
fn drawIds(rnd: std.Random, rows: usize, ids: []u32, skipped: *[2]usize) []const usize {
    const n_e = bank_experts[0] + bank_experts[1] + bank_experts[2];
    for (0..rows) |p| ids[p] = packedId(rnd.uintLessThan(u32, n_e));
    if (rows >= 10) {
        const a_n = @min(rows / 3, 16);
        const b_n = rows / 6;
        var p: usize = 0;
        for (0..a_n) |_| {
            ids[p] = packedId(1);
            p += 1;
        }
        for (0..b_n) |_| {
            ids[p] = packedId(bank_experts[0]);
            p += 1;
        }
        for (0..3) |_| {
            ids[p] = packedId(bank_experts[0] + bank_experts[1] + 2);
            p += 1;
        }
        // the drawn rest never names A, B or C
        while (p < rows) : (p += 1) {
            var e = rnd.uintLessThan(u32, n_e);
            while (e == 1 or e == bank_experts[0] or e == bank_experts[0] + bank_experts[1] + 2) e = rnd.uintLessThan(u32, n_e);
            ids[p] = packedId(e);
        }
        rnd.shuffle(u32, ids[0..rows]);
        skipped.* = .{ rnd.uintLessThan(usize, rows), 0 };
        skipped[1] = (skipped[0] + 1 + rnd.uintLessThan(usize, rows - 1)) % rows;
        for (skipped) |q| ids[q] = root.banked_skip_id;
        return skipped[0..2];
    }
    if (rows >= 5) {
        for (0..rows - 1) |p| ids[p] = packedId(3);
        ids[rows - 1] = root.banked_skip_id;
        skipped[0] = rows - 1;
        return skipped[0..1];
    }
    if (rows >= 2) ids[1] = ids[0];
    return skipped[0..0];
}

/// The banked call's three outputs on `sel`'s texts: hd (f16, as f32), hs and y; KEPT by `g`.
const Outs = struct { hd: mlx.mlx_array, hs: mlx.mlx_array, y: mlx.mlx_array };
fn callSel(g: *G, k: *decode.Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, ids: mlx.mlx_array, kb: *const [3]decode.Kernels.Bank, rows: usize, sel: Sel) !Outs {
    const h = try k.bankedGateUpSel(s, x, ids, kb, rows, sel);
    const hd = try g.adopt(h[0]);
    const hs = try g.adopt(h[1]);
    const y = try g.adopt(try k.bankedDownSel(s, hd, hs, ids, kb, rows, sel));
    return .{ .hd = try g.astype(hd, .float32), .hs = hs, .y = y };
}

/// The most row-slots one id takes in `ids` (skipped row-slots aside).
fn maxMembers(ids: []const u32) usize {
    var best: usize = 0;
    for (ids, 0..) |e, i| {
        if (e == root.banked_skip_id) continue;
        var c: usize = 0;
        for (ids[i..]) |f| c += @intFromBool(f == e);
        best = @max(best, c);
    }
    return best;
}

test "turboexl3 gpu banked grp: the M = 3 texts' gate|up outputs and rows are today's, byte for byte, a skipped row-slot's row 0, at DeepSeek-V4.1's routed geometry (K3, K2) (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    if (std.c.getenv("TURBOEXL3_BANKED_GRP_ONLY_TIME") != null) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_BANKED_N", &.{ 48, 32 }, &nbuf);
    var rbuf: [24]u32 = undefined;
    const row_counts = envList("TURBOEXL3_BANKED_ROWS", &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 36, 42, 48, 64 }, &rbuf);
    var failed = false;
    var checked: usize = 0;
    const max_w = root.banked_rows_max * hidden;
    const want = try a.alloc(f32, max_w);
    defer a.free(want);
    const got = try a.alloc(f32, max_w);
    defer a.free(got);
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x4A11 ^ @as(u64, n));
        const rnd = prng.random();
        var rb_u: [3]root.Bank = undefined;
        var rb_i: [3]root.Bank = undefined;
        for (&rb_u, &rb_i, bank_experts) |*u, *i, e| {
            var hb = try ref.synthBank(a, rnd, e, hidden, inter, rate);
            defer hb.deinit(a);
            u.* = try rootBank(&g, &hb, .uint16);
            i.* = try rootBank(&g, &hb, .int16);
            try evalBank(&g, u.*);
            try evalBank(&g, i.*);
        }
        const kb_u = kernelBanks(&rb_u);
        const kb_i = kernelBanks(&rb_i);
        var k = try groupedKernels3(rate);
        defer k.deinit();
        for (row_counts) |rows| {
            if (rows == 0 or rows > root.banked_rows_max) continue;
            const m2 = g.mark();
            defer g.resetTo(m2);
            var ids: [root.banked_rows_max]u32 = undefined;
            var sbuf: [2]usize = undefined;
            const skipped = drawIds(rnd, rows, ids[0..rows], &sbuf);
            const top = maxMembers(ids[0..rows]);
            const xh = try a.alloc(f32, rows * hidden);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
            const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(hidden) }, .float32);
            const xbf = try g.astype(x32, .bfloat16);
            const ids_a = try g.hostArray(std.mem.sliceAsBytes(ids[0..rows]), &.{ci(rows)}, .uint32);
            const Arm = struct { x: mlx.mlx_array, kb: *const [3]decode.Kernels.Bank, name: []const u8 };
            for ([_]Arm{
                .{ .x = xbf, .kb = &kb_i, .name = "x=bf16 trellis=i16" },
                .{ .x = x32, .kb = &kb_u, .name = "x=f32 trellis=u16" },
                .{ .x = xbf, .kb = &kb_u, .name = "x=bf16 trellis=u16" },
                .{ .x = x32, .kb = &kb_i, .name = "x=f32 trellis=i16" },
            }) |arm| {
                const m3 = g.mark();
                defer g.resetTo(m3);
                const o0 = try callSel(&g, &k, s, arm.x, ids_a, arm.kb, rows, .g4);
                try g.evalAll(&.{ o0.hd, o0.hs, o0.y });
                for (sels[1..]) |sel| {
                    const m4 = g.mark();
                    defer g.resetTo(m4);
                    const o = try callSel(&g, &k, s, arm.x, ids_a, arm.kb, rows, sel);
                    try g.evalAll(&.{ o.hd, o.hs, o.y });
                    var line: Diff = .{};
                    var ok = true;
                    for ([_]struct { w: mlx.mlx_array, v: mlx.mlx_array, row_w: usize, y: bool }{
                        .{ .w = o0.hd, .v = o.hd, .row_w = inter, .y = false },
                        .{ .w = o0.hs, .v = o.hs, .row_w = inter / 128, .y = false },
                        .{ .w = o0.y, .v = o.y, .row_w = hidden, .y = true },
                    }) |c| {
                        const nw = rows * c.row_w;
                        const w = try g.hostF32(c.w, want[0..nw]);
                        const u = try g.hostF32(c.v, got[0..nw]);
                        const df = wordDiff(u, w, c.row_w, skipped, c.y);
                        line.words += df.words;
                        line.differing += df.differing;
                        line.max_ulps = @max(line.max_ulps, df.max_ulps);
                        line.skipped_nonzero += df.skipped_nonzero;
                        line.nonfinite += df.nonfinite;
                        if (df.differing != 0) ok = false;
                    }
                    if (line.nonfinite != 0 or line.skipped_nonzero != 0) ok = false;
                    if (!ok) failed = true;
                    checked += line.words;
                    std.debug.print("TURBOEXL3_BANKED_GRP_BYTES n={d} rows={d} members_max={d} {s} sel={t} words={d} differing={d} max_ulps={d} nonfinite={d} skipped={d} skipped_nonzero={d} {s}\n", .{ n, rows, top, arm.name, sel, line.words, line.differing, line.max_ulps, line.nonfinite, skipped.len, line.skipped_nonzero, if (ok) "OK" else "FAIL" });
                }
            }
        }
    }
    std.debug.print("TURBOEXL3_BANKED_GRP_BYTES total_words={d} {s}\n", .{ checked, if (failed) "FAIL" else "OK" });
    try testing.expect(!failed);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// A timed call: `count[c]` hit experts of c row-slots each (c 1..8), `skip` skipped row-slots (misses), in shuffled
/// slot order.
const Profile = struct { set: []const u8, name: []const u8, count: [9]u32, skip: u32 };
const profiles = [_]Profile{
    // the recorded 36-pair verify profiles (banked_test.zig's p10, median, p90): 34 hits, 2 misses
    .{ .set = "rec", .name = "d18", .count = .{ 0, 10, 4, 2, 1, 0, 1, 0, 0 }, .skip = 2 },
    .{ .set = "rec", .name = "d22", .count = .{ 0, 14, 5, 2, 1, 0, 0, 0, 0 }, .skip = 2 },
    .{ .set = "rec", .name = "d29", .count = .{ 0, 25, 3, 1, 0, 0, 0, 0, 0 }, .skip = 2 },
    // widths 7 and 8 (40 and 46 hits, 2 misses), from the recorded pairs-per-expert histogram
    .{ .set = "w7", .name = "w7a", .count = .{ 0, 15, 5, 2, 1, 1, 0, 0, 0 }, .skip = 2 },
    .{ .set = "w7", .name = "w7b", .count = .{ 0, 16, 4, 2, 1, 0, 1, 0, 0 }, .skip = 2 },
    .{ .set = "w7", .name = "w7c", .count = .{ 0, 15, 4, 2, 1, 0, 0, 1, 0 }, .skip = 2 },
    .{ .set = "w8", .name = "w8a", .count = .{ 0, 15, 5, 2, 1, 1, 1, 0, 0 }, .skip = 2 },
    .{ .set = "w8", .name = "w8b", .count = .{ 0, 17, 6, 2, 1, 0, 0, 1, 0 }, .skip = 2 },
    .{ .set = "w8", .name = "w8c", .count = .{ 0, 14, 5, 2, 2, 0, 0, 0, 1 }, .skip = 2 },
    // only today's member paths (M 1, 2, 4): the M = 3 text's own cost
    .{ .set = "rp", .name = "rp36", .count = .{ 0, 18, 6, 0, 1, 0, 0, 0, 0 }, .skip = 2 },
    .{ .set = "rp", .name = "rp48", .count = .{ 0, 22, 10, 0, 1, 0, 0, 0, 0 }, .skip = 2 },
    // the member sweep: D experts of c row-slots each
    .{ .set = "sweep", .name = "d16c1", .count = .{ 0, 16, 0, 0, 0, 0, 0, 0, 0 }, .skip = 0 },
    .{ .set = "sweep", .name = "d16c2", .count = .{ 0, 0, 16, 0, 0, 0, 0, 0, 0 }, .skip = 0 },
    .{ .set = "sweep", .name = "d16c3", .count = .{ 0, 0, 0, 16, 0, 0, 0, 0, 0 }, .skip = 0 },
    .{ .set = "sweep", .name = "d16c4", .count = .{ 0, 0, 0, 0, 16, 0, 0, 0, 0 }, .skip = 0 },
    .{ .set = "sweep", .name = "d8c1", .count = .{ 0, 8, 0, 0, 0, 0, 0, 0, 0 }, .skip = 0 },
    .{ .set = "sweep", .name = "d8c2", .count = .{ 0, 0, 8, 0, 0, 0, 0, 0, 0 }, .skip = 0 },
    .{ .set = "sweep", .name = "d8c3", .count = .{ 0, 0, 0, 8, 0, 0, 0, 0, 0 }, .skip = 0 },
    .{ .set = "sweep", .name = "d8c4", .count = .{ 0, 0, 0, 0, 8, 0, 0, 0, 0 }, .skip = 0 },
    .{ .set = "sweep", .name = "d8c5", .count = .{ 0, 0, 0, 0, 0, 8, 0, 0, 0 }, .skip = 0 },
    .{ .set = "sweep", .name = "d8c6", .count = .{ 0, 0, 0, 0, 0, 0, 8, 0, 0 }, .skip = 0 },
    .{ .set = "sweep", .name = "d8c7", .count = .{ 0, 0, 0, 0, 0, 0, 0, 8, 0 }, .skip = 0 },
    .{ .set = "sweep", .name = "d8c8", .count = .{ 0, 0, 0, 0, 0, 0, 0, 0, 8 }, .skip = 0 },
};

fn timedCall(k: *decode.Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, ids: mlx.mlx_array, kb: *const [3]decode.Kernels.Bank, rows: usize, sel: Sel) !mlx.mlx_array {
    const h = try k.bankedGateUpSel(s, x, ids, kb, rows, sel);
    defer _ = mlx.mlx_array_free(h[0]);
    defer _ = mlx.mlx_array_free(h[1]);
    return k.bankedDownSel(s, h[0], h[1], ids, kb, rows, sel);
}

test "turboexl3 gpu banked grp time: the M = 3 texts against today's grouped texts, recorded and swept profiles (TURBOEXL3_GPU=1 TURBOEXL3_BANKED_GRP_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_BANKED_GRP_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_BANKED_N", &.{ 48, 32 }, &nbuf);
    const warm: usize = 2;
    const timed: usize = 9;
    const chain: usize = 40;
    const n_arms = sels.len;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x4A17 ^ @as(u64, n));
        const rnd = prng.random();
        var rb: [3]root.Bank = undefined;
        for (&rb, time_experts) |*b, e| {
            var hb = try ref.synthBank(a, rnd, e, hidden, inter, rate);
            defer hb.deinit(a);
            b.* = try rootBank(&g, &hb, .int16);
            try evalBank(&g, b.*);
        }
        const kb = kernelBanks(&rb);
        var k = try groupedKernels3(rate);
        defer k.deinit();
        for (profiles) |pf| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            // the hit experts: distinct global indexes over base rows 1..47 and the ext bank, never base row 0
            var order: [63]u32 = undefined;
            for (&order, 0..) |*o, i| o.* = @intCast(i + 1);
            rnd.shuffle(u32, &order);
            var ids: [root.banked_rows_max]u32 = undefined;
            var nh: usize = 0;
            var d: usize = 0;
            for (pf.count, 0..) |cnt, c| for (0..cnt) |_| {
                const e = order[d];
                d += 1;
                const id: u32 = if (e < 48) e else (1 << 24) | (e - 48);
                for (0..c) |_| {
                    ids[nh] = id;
                    nh += 1;
                }
            };
            for (0..pf.skip) |_| {
                ids[nh] = root.banked_skip_id;
                nh += 1;
            }
            const rows = nh;
            std.debug.assert(rows <= root.banked_rows_max);
            rnd.shuffle(u32, ids[0..rows]);
            const xh = try a.alloc(f32, rows * hidden);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(hidden) }, .float32), .bfloat16);
            const ids_a = try g.hostArray(std.mem.sliceAsBytes(ids[0..rows]), &.{ci(rows)}, .uint32);
            try g.evalAll(&.{ x, ids_a });
            var single: [n_arms][timed]u64 = undefined;
            var chained: [n_arms][timed]u64 = undefined;
            var ys: [chain]mlx.mlx_array = undefined;
            for (0..warm + timed) |round| {
                for (0..n_arms) |kk| {
                    const ai = (kk + round) % n_arms;
                    const sel = sels[ai];
                    const t0 = nowNs();
                    {
                        const y = try timedCall(&k, s, x, ids_a, &kb, rows, sel);
                        defer _ = mlx.mlx_array_free(y);
                        try mlx.check(mlx.mlx_array_eval(y));
                    }
                    const t1 = nowNs();
                    var made_y: usize = 0;
                    defer for (ys[0..made_y]) |y| {
                        _ = mlx.mlx_array_free(y);
                    };
                    var xc = x;
                    for (0..chain) |c| {
                        ys[c] = try timedCall(&k, s, xc, ids_a, &kb, rows, sel);
                        made_y += 1;
                        xc = ys[c];
                    }
                    try mlx.check(mlx.mlx_array_eval(ys[chain - 1]));
                    const t2 = nowNs();
                    if (round >= warm) {
                        single[ai][round - warm] = t1 - t0;
                        chained[ai][round - warm] = (t2 - t1) / chain;
                    }
                }
            }
            for (0..n_arms) |ai| {
                std.mem.sort(u64, &single[ai], {}, std.sort.asc(u64));
                std.mem.sort(u64, &chained[ai], {}, std.sort.asc(u64));
                const us = struct {
                    fn of(v: u64) f64 {
                        return @as(f64, @floatFromInt(v)) / 1000.0;
                    }
                };
                std.debug.print("TURBOEXL3_BANKED_GRP_TIME n={d} set={s} profile={s} distinct={d} pairs={d} rows={d} sel={t} single_min_us={d:.1} single_med_us={d:.1} chain_min_us={d:.1} chain_med_us={d:.1}\n", .{ n, pf.set, pf.name, d, rows - pf.skip, rows, sels[ai], us.of(single[ai][0]), us.of(single[ai][timed / 2]), us.of(chained[ai][0]), us.of(chained[ai][timed / 2]) });
            }
        }
    }
}
