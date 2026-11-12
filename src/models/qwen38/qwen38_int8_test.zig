//! The int8 options at Qwen3.8-Flash-Next's shapes (the outside plugin turboexl3_qwen38's TURBOEXL3_QWEN38_PF_INT8): the
//! routed prefill's int8 GEMMs at its routed geometry (2560 -> 640, top-10) and trunk int8 at every linear of
//! `api.trunk`, with its inner products, each at the plugin's 509.5 center (trunk_q8_test.zig's batteries). ATTN-IN's and
//! ATTN-OUT's int8 paths are attn_io_test.zig's.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const t8 = @import("../../core/trunk_q8_test.zig");
const root = @import("../../root.zig");
const api = @import("api.zig");

const testing = std.testing;

/// The routed geometry: hidden, the experts' intermediate width, the top-k
const hidden: u32 = 2560;
const inter: u32 = 640;
const topk: u32 = 10;

test "turboexl3 qwen38 int8 shapes: every trunk linear takes trunk int8 (in and out multiples of 128), level 3 names the attention projections with the indexer at its K5 alone, level 4 every linear; one scale a row fits the routed geometry" {
    for (api.trunk) |l| {
        try testing.expect(l.k % 128 == 0 and l.n % root.trunk_q8.bn == 0);
    }
    var b3: [api.trunk.len]root.trunk_q8.Shape = undefined;
    var b4: [api.trunk.len]root.trunk_q8.Shape = undefined;
    const lv3 = api.trunkShapes(3, &b3);
    const lv4 = api.trunkShapes(4, &b4);
    try testing.expectEqual(@as(usize, 0), api.trunkShapes(1, &b3).len);
    try testing.expectEqual(api.trunk.len, lv4.len);
    const Case = struct { k: u32, n: u32, rate_n: u32, lv3: bool };
    // the indexer (K5) at level 3, the shared gate|up (K7) of the same shape only at 4; the attention's and the GDN's
    // projections at 3 (K7), the shared down and the MTP fc (K6) at 4
    const cases = [_]Case{
        .{ .k = 2560, .n = 640, .rate_n = 80, .lv3 = true },
        .{ .k = 2560, .n = 640, .rate_n = 112, .lv3 = false },
        .{ .k = 2560, .n = 10240, .rate_n = 112, .lv3 = true },
        .{ .k = 2560, .n = 6144, .rate_n = 112, .lv3 = true },
        .{ .k = 6144, .n = 2560, .rate_n = 112, .lv3 = true },
        .{ .k = 2560, .n = 12288, .rate_n = 112, .lv3 = true },
        .{ .k = 2560, .n = 512, .rate_n = 112, .lv3 = true },
        .{ .k = 640, .n = 2560, .rate_n = 112, .lv3 = false },
        .{ .k = 2560, .n = 2560, .rate_n = 96, .lv3 = false },
    };
    for (cases) |c| {
        var in3 = false;
        for (lv3) |sh| in3 = in3 or root.trunk_q8.names(sh, c.k, c.n, c.rate_n);
        var in4 = false;
        for (lv4) |sh| in4 = in4 or root.trunk_q8.names(sh, c.k, c.n, c.rate_n);
        try testing.expectEqual(c.lv3, in3);
        try testing.expect(in4);
    }
    // the routed int8's one-scale-a-row scheme holds a row of hidden and of inter in one threadgroup (at most 32 blocks)
    try testing.expect(hidden / 128 <= 32 and inter / 128 <= 32);
}

test "turboexl3 gpu qwen38 int8 routed: the routed gate|up and down on int8 at Qwen3.8's geometry (2560 -> 640, top-10, K5) agree with the exact chain, a scale each 128-block and one a row, at both weight centers, 509.5's error below 510's (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    defer root.releaseEngines();
    var prng = std.Random.DefaultPrng.init(0x0E38_18);
    const rnd = prng.random();
    // 64 experts over 1638 rows (16380 row-slots: ~256 a expert, so 128-row tiles and remainders both run)
    const n: u32 = 80;
    const c = try t8.routedCase(&g, a, rnd, 64, hidden, inter, n, 1638, topk);
    const ok = try t8.routedAgree(&g, a, s, "TURBOEXL3_QWEN38_PF_INT8", c, 0, n, &.{ 1, 2 });
    try testing.expect(ok);
}

/// The rows of trunk int8's checks (a short call, a chunk's worth, past one 4096-row GEMM tile run)
const rows_list = [_]u32{ 100, 2048, 4100 };

/// The rate a test gives a linear: its named rate, else the 5.05 pack's (the MTP fc K6, the rest K7)
fn rateOf(l: api.Linear) u32 {
    if (l.rate_n != 0) return l.rate_n;
    return if (std.mem.startsWith(u8, l.name, "mtp.")) 96 else 112;
}

test "turboexl3 gpu qwen38 int8 trunk: every Qwen3.8 trunk linear on trunk int8 at the 509.5 center agrees with the chain at 100 to 4100 rows; the switch off is the chain word for word; a call's rows split over two calls give its words; the inner products a host splits off give linear's int8 words, and the chain's with the inner switch off (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    defer root.releaseEngines();
    var prng = std.Random.DefaultPrng.init(0x0E38_7A);
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
        ok = (try t8.linearBattery(&g, a, s, "TURBOEXL3_QWEN38_TRUNK_INT8", l.name, p, l.k, l.n, &rows_list, &pool, 1019, bufs)) and ok;
    }
    try testing.expect(ok);
}
