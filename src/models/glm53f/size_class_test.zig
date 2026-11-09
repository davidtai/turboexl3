//! The routed prefill's size classes (`prefill.Options.size_class`, `Config.pf_size_class`: the plugin's
//! TURBOEXL3_GLM53F_PF_REUSE), at GLM-5.3-Flash's routed geometry:
//! - CPU: `prefill.classRows` rounds a call's rows up to a multiple of `prefill.size_class_rows` (256); the option needs
//!   the 128-row tiles without t128_zfit, and the config carries it off by default.
//! - GPU (TURBOEXL3_GPU=1): prompt calls whose row-slot counts are not multiples of 256 (257, 1000, 1777, 2500, 4095,
//!   3333 and 3422: 16384 row-slots) give, with the size classes, the words they give without, on the plugin's default
//!   path (take, f32, `moeClamped` at top-1, cast) and on `root.moeClampedRows`, over a drawn routing and 36 experts'
//!   455-row runs, at every SC rate. One line a rate, path and routing (TURBOEXL3_GLM_PF_SIZE_CLASS_BYTES), then their
//!   total.
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

/// The calls' row-slot counts: none a multiple of 256, 16384 together
const sizes = [_]usize{ 257, 1000, 1777, 2500, 4095, 3333, 3422 };

fn ci(x: anytype) c_int {
    return @intCast(x);
}

test "turboexl3 glm53f prefill size classes: classRows rounds a call's rows up to a multiple of 256; the option needs the 128-row tiles without t128_zfit and the config carries it off by default" {
    try testing.expectEqual(@as(u32, 256), prefill.size_class_rows);
    for ([_][2]u32{ .{ 0, 0 }, .{ 1, 256 }, .{ 255, 256 }, .{ 256, 256 }, .{ 257, 512 }, .{ 1000, 1024 }, .{ 4095, 4096 }, .{ 4096, 4096 }, .{ 4097, 4352 }, .{ 16384, 16384 }, .{ 131071, 131072 } }) |c|
        try testing.expectEqual(c[1], prefill.classRows(c[0]));
    var n: u32 = 1;
    while (n < 20000) : (n += 37) {
        const r = prefill.classRows(n);
        try testing.expect(r >= n and r - n < prefill.size_class_rows and r % prefill.size_class_rows == 0);
    }
    for (sizes) |s| try testing.expect(s % prefill.size_class_rows != 0);
    var o = served;
    o.size_class = true;
    try testing.expect(o.valid());
    o.t128_zfit = true;
    try testing.expect(!o.valid());
    o.t128_zfit = false;
    o.t128 = false;
    try testing.expect(!o.valid());
    const dc: config.Config = .{};
    try testing.expect(!dc.pf_size_class);
    try testing.expect((config.Config{ .pf_size_class = true }).valid());
}

/// `configure` with Config.pf_size_class `on` (every other field `saved`'s), every engine released first so the next
/// call's prefill engine is made under it
fn classConfigure(saved: config.Config, on: bool) !void {
    root.releaseEngines();
    var c = saved;
    c.pf_size_class = on;
    try root.configure(c);
}

/// One prompt call over `rows` / `slots`: on `root.moeClampedRows`, or the plugin's default path (take, f32,
/// `moeClamped` at top-1 over [1, S, hidden], y cast), y bf16
fn classCall(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, bank: root.Bank, rows: []const u32, slots: []const u32, ones_h: []const f32, on_rows: bool) !mlx.mlx_array {
    const S = rows.len;
    const ones = try g.hostArray(std.mem.sliceAsBytes(ones_h[0..S]), &.{ci(S)}, .float32);
    const rows_a = try g.hostArray(std.mem.sliceAsBytes(rows), &.{ci(S)}, .uint32);
    const inds = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(S)}, .uint32);
    if (on_rows) return g.adopt(try root.moeClampedRows(s, x, rows_a, bank, inds, ones, .mul1, limit, .bfloat16));
    const xr = try g.astype(try g.take(x, rows_a, 0), .float32);
    const x3 = try g.reshape(xr, &.{ 1, ci(S), ci(hidden) });
    const y3 = try g.adopt(try root.moeClamped(s, x3, bank, try g.reshape(inds, &.{ 1, ci(S), 1 }), try g.reshape(ones, &.{ 1, ci(S), 1 }), .mul1, limit));
    return g.astype(try g.reshape(y3, &.{ ci(S), ci(hidden) }), .bfloat16);
}

/// The routings a rate's calls take: a drawn routing (each call's row-slots slot-sorted), 36 experts' 455-row runs
const routings = [_][]const u8{ "drawn", "runs455" };

test "turboexl3 gpu glm53f prefill size classes: prompt calls of 257 to 4095 row-slots (none a multiple of 256) give the same words with the size classes as without, on the plugin's default path and on moeClampedRows, over drawn and 455-row routings, at every SC rate (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = config.get().*;
    defer classConfigure(saved, saved.pf_size_class) catch unreachable;
    var nbuf: [8]u32 = undefined;
    const rates = routed.envList("TURBOEXL3_GLM_ROUTED_N", &routed.rates_n, &nbuf);
    comptime var total: usize = 0;
    inline for (sizes) |c| total += c;
    const S: usize = total;
    const want_buf = try a.alloc(f32, S * hidden);
    defer a.free(want_buf);
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
    var lines: usize = 0;
    var fails: usize = 0;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        defer root.releaseEngines();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x5C1A55 ^ @as(u64, n));
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, fused.prompt_slots, hidden, inter, rate);
        defer bank.deinit(a);
        const rb = try routed.rootBank(&g, &bank, .uint16);
        const xh = try a.alloc(f32, fused.act_rows * hidden);
        defer a.free(xh);
        for (xh) |*v| v.* = routed.bfExact(fused.sigma * rnd.floatNorm(f32));
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(fused.act_rows), ci(hidden) }, .float32), .bfloat16);
        try g.evalAll(&.{x});
        // drawn: each call's row-slots slot-sorted (a call is one bank wave's); 36 experts' 455-row runs over the 16384
        fused.drawCall(rnd, rows_r[0], slots_r[0], fused.prompt_slots, 0);
        var c0: usize = 0;
        for (sizes) |c| {
            try fused.sortBySlot(a, rows_r[0][c0 .. c0 + c], slots_r[0][c0 .. c0 + c], fused.prompt_slots);
            c0 += c;
        }
        for (rows_r[1], slots_r[1], 0..) |*r, *sl, i| {
            r.* = rnd.uintLessThan(u32, fused.act_rows);
            // the run's index in u32 first: @min with a comptime bound narrows its type, and 35 * 2 needs 7 bits
            const run: u32 = @intCast(@min(i / 455, 35));
            sl.* = run * 2;
        }
        for ([_]bool{ false, true }) |on_rows| {
            for (routings, 0..) |routing, ri| {
                // the words without the size classes, then with them
                for ([_]bool{ false, true }) |on| {
                    try classConfigure(saved, on);
                    const buf = if (on) got_buf else want_buf;
                    var b0: usize = 0;
                    for (sizes) |c| {
                        const m2 = g.mark();
                        defer g.resetTo(m2);
                        const y = try classCall(&g, s, x, rb, rows_r[ri][b0 .. b0 + c], slots_r[ri][b0 .. b0 + c], ones_h, on_rows);
                        _ = try fused.bitsOf(&g, y, buf[b0 * hidden .. (b0 + c) * hidden]);
                        b0 += c;
                    }
                }
                const want: []const u32 = @as([*]const u32, @ptrCast(want_buf.ptr))[0 .. S * hidden];
                const got: []const u32 = @as([*]const u32, @ptrCast(got_buf.ptr))[0 .. S * hidden];
                const dd = fused.countDiff(got, want, got_buf);
                const ok = dd[0] == 0 and dd[1] == 0;
                std.debug.print("TURBOEXL3_GLM_PF_SIZE_CLASS_BYTES n={d} S={d} path={s} routing={s} calls={d} against=class_off differing={d}/{d} nonfinite={d} {s}\n", .{ n, S, if (on_rows) "rows" else "chain", routing, sizes.len, dd[0], S * hidden, dd[1], if (ok) "PASS" else "FAIL" });
                lines += 1;
                if (!ok) fails += 1;
            }
        }
        try classConfigure(saved, saved.pf_size_class);
    }
    std.debug.print("TURBOEXL3_GLM_PF_SIZE_CLASS_BYTES lines={d} fails={d} {s}\n", .{ lines, fails, if (fails == 0) "PASS" else "FAIL" });
    try testing.expect(fails == 0);
}
