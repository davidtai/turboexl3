//! The int8 options at DeepSeek-V4.1's shapes (the outside plugin turboexl3_ds41's TURBOEXL3_DS41_PF_INT8): the routed
//! prefill's int8 GEMMs at its routed geometry (5120 -> 2304, top-6, the clamped SwiGLU at 10) and trunk int8 at every
//! linear of `api.trunk` with its inner products, and the grouped `wo_a` on trunk int8, each at the plugin's 509.5 center
//! (trunk_q8_test.zig's batteries).
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const t8 = @import("../../core/trunk_q8_test.zig");
const root = @import("../../root.zig");
const api = @import("api.zig");

const testing = std.testing;

/// The routed geometry: hidden, the experts' intermediate width, the top-k, the SwiGLU's clamp
const hidden: u32 = 5120;
const inter: u32 = 2304;
const topk: u32 = 6;
const limit: c_int = 10;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

test "turboexl3 deepseek_v41 int8 shapes: every trunk linear takes trunk int8 (in and out multiples of 128), level 3 names the attention projections alone, level 4 every linear; one scale a row does not fit the routed geometry (hidden past 32 blocks)" {
    for (api.trunk) |l| {
        try testing.expect(l.k % 128 == 0 and l.n % root.trunk_q8.bn == 0);
    }
    var b3: [api.trunk.len]root.trunk_q8.Shape = undefined;
    var b4: [api.trunk.len]root.trunk_q8.Shape = undefined;
    const lv3 = api.trunkShapes(3, &b3);
    const lv4 = api.trunkShapes(4, &b4);
    try testing.expectEqual(@as(usize, 0), api.trunkShapes(2, &b3).len);
    try testing.expectEqual(api.trunk.len, lv4.len);
    for (api.trunk) |l| {
        var in3 = false;
        for (lv3) |sh| in3 = in3 or root.trunk_q8.names(sh, l.k, l.n, 96);
        try testing.expectEqual(l.attn, in3);
    }
    // the routed int8's one-scale-a-row scheme needs a row of hidden in one threadgroup: 5120 is 40 blocks
    try testing.expect(hidden / 128 > 32);
}

/// Routed calls past the decode rows: a 2048-row chunk over 32 experts (12288 row-slots, ~384 a expert)
const routed_rows: u32 = 2048;

test "turboexl3 gpu deepseek_v41 int8 routed: the routed gate|up and down on int8 at DeepSeek-V4.1's geometry (5120 -> 2304, top-6, clamped at 10; K2.5 and K2) agree with the exact chain at a scale each 128-block, at both weight centers, 509.5's error below 510's; one scale a row is refused by name (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    defer root.releaseEngines();
    var prng = std.Random.DefaultPrng.init(0xD541_18);
    const rnd = prng.random();
    var ok = true;
    for ([_]u32{ 40, 32 }) |n| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const c = try t8.routedCase(&g, a, rnd, 32, hidden, inter, n, routed_rows, topk);
        ok = (try t8.routedAgree(&g, a, s, "TURBOEXL3_DS41_PF_INT8", c, limit, n, &.{1})) and ok;
        // one scale a row: the prefill engine refuses the geometry (Prefill.init: hidden past 32 blocks)
        const sv = t8.save();
        defer t8.restore(sv);
        try t8.setRouted(sv.cfg, 2, 1019);
        const refused = if (t8.routedRun(&g, s, c, limit)) |_| false else |e| e == error.BadExl3Shape;
        std.debug.print("TURBOEXL3_DS41_PF_INT8_REFUSED n={d} int8=2 {s}\n", .{ n, if (refused) "PASS" else "FAIL" });
        ok = ok and refused;
    }
    try testing.expect(ok);
}

/// The rows of trunk int8's checks (a short call, a chunk's worth, past one 4096-row GEMM tile run)
const rows_list = [_]u32{ 100, 2048, 4100 };

/// The rate a test gives a linear: the 2.25 pack's (DSpark's main_proj K4, the rest K6)
fn rateOf(l: api.Linear) u32 {
    return if (std.mem.startsWith(u8, l.name, "mtp.")) 64 else 96;
}

test "turboexl3 gpu deepseek_v41 int8 trunk: every DeepSeek-V4.1 trunk linear on trunk int8 at the 509.5 center agrees with the chain at 100 to 4100 rows; the switch off is the chain word for word; a call's rows split over two calls give its words; the inner products a host splits off give linear's int8 words, and the chain's with the inner switch off (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    defer root.releaseEngines();
    var prng = std.Random.DefaultPrng.init(0xD541_7A);
    const rnd = prng.random();
    var pool: [8191]f32 = undefined;
    t8.normalPool(rnd, &pool);
    var max_words: usize = 0;
    for (api.trunk) |l| max_words = @max(max_words, @as(usize, rows_list[rows_list.len - 1]) * l.n);
    const bufs = try t8.Bufs.init(a, max_words);
    defer bufs.deinit(a);
    var ok = true;
    for (api.trunk) |l| {
        defer root.releaseEngines();
        const mark = g.mark();
        defer g.resetTo(mark);
        const p = try t8.synthProj(&g, a, rnd, l.k, l.n, rateOf(l));
        ok = (try t8.linearBattery(&g, a, s, "TURBOEXL3_DS41_TRUNK_INT8", l.name, p, l.k, l.n, &rows_list, &pool, 1019, bufs)) and ok;
    }
    try testing.expect(ok);
}

/// `wo_a`: eight groups of 4096 -> 1024 at K6
const groups: u32 = 8;
const woa_in: u32 = 4096;
const woa_out: u32 = 1024;

test "turboexl3 gpu deepseek_v41 int8 grouped: wo_a's prefill rows (linearGrouped) on trunk int8 give each group's linear int8 words and agree with the exact grouped linear; a group's rows prepared once for every group and run as inner products (preparePrepared + gemmInner, a host's joined prepare and finish) give them too (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    defer root.releaseEngines();
    var prng = std.Random.DefaultPrng.init(0xD541_0A);
    const rnd = prng.random();
    var pool: [8191]f32 = undefined;
    t8.normalPool(rnd, &pool);
    const sv = t8.save();
    defer t8.restore(sv);
    const one = [1]root.trunk_q8.Shape{.{ .k = woa_in, .n = woa_out }};
    // the groups' projections, and the stacked operand linearGrouped takes
    var ps: [groups]root.Proj = undefined;
    var ts: [groups]mlx.mlx_array = undefined;
    var us: [groups]mlx.mlx_array = undefined;
    var vs: [groups]mlx.mlx_array = undefined;
    for (&ps, &ts, &us, &vs) |*p, *t, *u, *v| {
        p.* = try t8.synthProj(&g, a, rnd, woa_in, woa_out, 96);
        t.* = try g.reshape(p.trellis, &.{ 1, ci(woa_in / 16), ci(woa_out / 16), 96 });
        u.* = try g.reshape(p.suh, &.{ 1, ci(woa_in) });
        v.* = try g.reshape(p.svh, &.{ 1, ci(woa_out) });
    }
    const stacked: root.Proj = .{ .trellis = try g.concat(&ts, 0), .suh = try g.concat(&us, 0), .svh = try g.concat(&vs, 0) };
    try g.evalAll(&.{ stacked.trellis, stacked.suh, stacked.svh });
    var ok = true;
    for ([_]u32{ 100, 2048 }) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const x2 = try t8.poolX(&g, a, &pool, rows, groups * woa_in);
        const x = try g.reshape(x2, &.{ ci(rows), ci(groups), ci(woa_in) });
        const words = @as(usize, rows) * groups * woa_out;
        const want = try a.alloc(f32, words);
        defer a.free(want);
        const got = try a.alloc(f32, words);
        defer a.free(got);
        const ref8 = try a.alloc(f32, words);
        defer a.free(ref8);
        // the exact grouped linear
        try t8.setTrunk(sv.cfg, false, false, 1019, null);
        const y0 = try g.adopt(try root.linearGrouped(s, x, stacked, .mul1));
        _ = try t8.bitsOf(&g, y0, want);
        // on trunk int8: linearGrouped, and each group's linear (a slice of x) joined
        try t8.setTrunk(sv.cfg, true, false, 1019, &one);
        const before = t8.int8Calls();
        const y8 = try g.adopt(try root.linearGrouped(s, x, stacked, .mul1));
        const grouped_int8 = t8.int8Calls() >= before + groups;
        const gb = try t8.bitsOf(&g, y8, ref8);
        var ys: [groups]mlx.mlx_array = undefined;
        for (&ys, ps, 0..) |*y, p, gi| {
            const xg = try g.reshape(try g.slice(x, &.{ 0, ci(gi), 0 }, &.{ ci(rows), ci(gi + 1), ci(woa_in) }, &.{ 1, 1, 1 }), &.{ ci(rows), ci(woa_in) });
            y.* = try g.adopt(try root.linear(s, xg, p.trellis, p.suh, p.svh, .mul1));
        }
        const yj = try g.reshape(try g.concat(&ys, 1), &.{ ci(rows), ci(groups), ci(woa_out) });
        const dj = t8.countDiff(try t8.bitsOf(&g, yj, got), gb, got);
        // a host's joined run: each group's rows prepared once (linear's prepare, natural order), then its inner product on
        // trunk int8 (preparePrepared + gemmInner against its wq, decoded once) and linear's finish
        try t8.setTrunk(sv.cfg, true, true, 1019, &one);
        var yi: [groups]mlx.mlx_array = undefined;
        for (&yi, ps, 0..) |*y, p, gi| {
            const xg = try g.reshape(try g.slice(x, &.{ 0, ci(gi), 0 }, &.{ ci(rows), ci(gi + 1), ci(woa_in) }, &.{ 1, 1, 1 }), &.{ ci(rows), ci(woa_in) });
            const d = try root.denseOf(p, woa_in, .bfloat16, .mul1);
            const xp = try g.adopt(try root.linearPrep(s, xg, p, .mul1));
            const wq = try g.adopt(try root.trunk_q8.decodeW(s, d, p.trellis));
            const act = try root.trunk_q8.preparePrepared(s, d, xp, rows);
            defer act.deinit();
            const z = try g.adopt(try root.trunk_q8.gemmInner(s, act, wq, rows));
            y.* = try g.adopt(try root.linearFinish(s, z, p, .mul1, .bfloat16));
        }
        const yh = try g.reshape(try g.concat(&yi, 1), &.{ ci(rows), ci(groups), ci(woa_out) });
        const dh = t8.countDiff(try t8.bitsOf(&g, yh, got), gb, got);
        const ag = t8.agreeOf(ref8, want);
        const ok_rows = grouped_int8 and dj[0] == 0 and dj[1] == 0 and dh[0] == 0 and dh[1] == 0 and ag.ok();
        std.debug.print("TURBOEXL3_DS41_WOA_INT8 rows={d} groups={d} int8={} joined_differing={d}/{d} inner_differing={d}/{d} cos={d:.6} rms_ratio={d:.5} nonfinite={d} {s}\n", .{ rows, groups, grouped_int8, dj[0], words, dh[0], words, ag.cos, ag.rms, ag.nonfinite, if (ok_rows) "PASS" else "FAIL" });
        ok = ok and ok_rows;
    }
    try testing.expect(ok);
}
