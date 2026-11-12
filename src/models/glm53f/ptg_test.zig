//! The persistent grouped banked GEMVs (`Config.banked_ptg`; banked_ptg.zig's `text_gu3bgp`, `text_down3bgp` and hw4b's
//! `text_gu3bgm3p`, `text_down3bgm3p`: the plugin's TURBOEXL3_GLM53F_BANKED g4p and g4m3p) at GLM-5.3-Flash's routed
//! geometry:
//! - CPU: the option needs `banked_group` 4 and is off by default, and the persistent GEMVs' kernel names are their own
//!   (MLX keys a custom kernel's library by its name).
//! - GPU (TURBOEXL3_GPU=1): each persistent text set (g4p, g4m3p) at every grouped GEMV variant the plugin takes, selected
//!   per call (each variant on one GEMV, the other GEMV today's), and configured as the plugin sets it (g4m3p at t4.t4,
//!   the served schedule's variants, and g4p at t4.t4), gives the served banked call's rows byte for byte (the bf16 store
//!   its rounding) at every SC rate and the banked schedule test's pair counts (1 to 64: one row's 8 pairs, a depth-2
//!   verify's 24, a 4-row verify's 32, and the partial and full waves). One line a rate, pair count, texts, variants and
//!   route: TURBOEXL3_GLM_BANKED_PTG_BYTES.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const config = @import("../../core/config.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const banked = @import("../deepseek_v41/banked.zig");
const banked_ptg = @import("../deepseek_v41/banked_ptg.zig");
const routed = @import("routed_test.zig");
const fused = @import("fused_test.zig");

const testing = std.testing;

const hidden = routed.hidden;
const limit = routed.limit;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

test "turboexl3 glm53f banked ptg: the persistent grouped GEMVs need banked_group 4 and are off by default, under kernel names of their own" {
    const d: config.Config = .{};
    try testing.expect(!d.banked_ptg);
    try testing.expect((config.Config{ .banked_group = 4, .banked_ptg = true }).valid());
    try testing.expect((config.Config{ .banked_group = 4, .banked_m3 = true, .banked_ptg = true, .banked_variant_gu = root.groupVariantOf("t4").?, .banked_variant_dn = root.groupVariantOf("t4").? }).valid());
    try testing.expect(!(config.Config{ .banked_ptg = true }).valid());
    // every grouped GEMV kernel name distinct, the persistent ones beside the grouped texts' own
    const names = banked.group_kernel_names ++ banked_ptg.kernel_names;
    for (names, 0..) |a, i| for (names[i + 1 ..]) |b| try testing.expect(!std.mem.eql(u8, a, b));
    var found: usize = 0;
    for (names) |n| {
        for ([_][]const u8{ "turboexl3_gu3bgp", "turboexl3_down3bgp", "turboexl3_gu3bgm3p", "turboexl3_down3bgm3p" }) |p| found += @intFromBool(std.mem.eql(u8, n, p));
    }
    try testing.expectEqual(@as(usize, 4), found);
}

test "turboexl3 gpu glm53f banked ptg: the persistent grouped GEMVs (g4p, g4m3p) at every grouped GEMV variant the plugin takes, selected per call or configured, give the served banked call's rows byte for byte (the bf16 store its rounding), at GLM-5.3-Flash's routed geometry, every SC rate and pair counts 1 to 64 (TURBOEXL3_GPU=1)" {
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
    // the plugin's TURBOEXL3_GLM53F_BANKED g4m3p.t4.t4 (the served g4m3.t4.t4's persistent form) and g4p.t4.t4, as it
    // configures turboexl3
    const Configured = struct { m3: bool, gu: []const u8, dn: []const u8 };
    const configured = [_]Configured{ .{ .m3 = true, .gu = "t4", .dn = "t4" }, .{ .m3 = false, .gu = "t4", .dn = "t4" } };
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
        var prng = std.Random.DefaultPrng.init(0x7F7D3 ^ @as(u64, n));
        const rnd = prng.random();
        var hb: [3]ref.Bank = undefined;
        var made: usize = 0;
        defer for (hb[0..made]) |*b| b.deinit(a);
        for (&hb, routed.bank_experts) |*b, e| {
            b.* = try ref.synthBank(a, rnd, e, hidden, routed.inter, rate);
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
            for (xh) |*v| v.* = routed.bfExact(fused.sigma * rnd.floatNorm(f32));
            var ids: [root.banked_rows_max]u32 = undefined;
            fused.drawWave(rnd, ids[0..pairs]);
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(pairs), ci(hidden) }, .float32), .bfloat16);
            const ids_a = try g.hostArray(std.mem.sliceAsBytes(ids[0..pairs]), &.{ci(pairs)}, .uint32);
            const want = try g.adopt(try root.moeClampedBanked(s, x, &rb, ids_a, .mul1, limit));
            try g.evalAll(&.{ x, ids_a, want });
            cases[nc] = .{ .pairs = pairs, .x = x, .ids = ids_a, .want = want };
            nc += 1;
        }
        // selected per call: each persistent text set at each gate|up variant (the down GEMV today's) and each down variant
        // (the gate|up GEMV today's)
        for (cases[0..nc]) |c| {
            const w_n = c.pairs * hidden;
            const want = try fused.bitsOf(&g, c.want, want_buf[0..w_n]);
            for ([_]root.BankedSel{ .g4p, .g4m3p }) |sel| for (0..n_var) |vi| for ([_]bool{ false, true }) |down| {
                if (down and vi == 0) continue;
                if (fused.refusedVariant(vi)) continue;
                const m2 = g.mark();
                defer g.resetTo(m2);
                const v: u8 = @intCast(vi);
                const vgu: u8 = if (down) 0 else v;
                const vdn: u8 = if (down) v else 0;
                const y = try g.adopt(try root.moeClampedBankedSelV(s, c.x, &rb, c.ids, .mul1, limit, sel, vgu, vdn));
                const got = try fused.bitsOf(&g, y, got_buf[0..w_n]);
                const d = fused.countDiff(got, want, got_buf[0..w_n]);
                const ok = d[0] == 0 and d[1] == 0;
                checked += w_n;
                std.debug.print("TURBOEXL3_GLM_BANKED_PTG_BYTES n={d} pairs={d} route=selected texts={t} gu={s} dn={s} out=f32 differing={d}/{d} nonfinite={d} {s}\n", .{ n, c.pairs, sel, fused.varName(vgu), fused.varName(vdn), d[0], w_n, d[1], if (ok) "PASS" else "FAIL" });
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
            cfg.banked_ptg = true;
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
                    const want = try fused.bitsOf(&g, if (dt == .float32) c.want else try g.astype(c.want, .bfloat16), want_buf[0..w_n]);
                    const y = try g.adopt(if (dt == .float32)
                        try root.moeClampedBanked(s, c.x, &rb, c.ids, .mul1, limit)
                    else
                        try root.moeClampedBankedAs(s, c.x, &rb, c.ids, .mul1, limit, .bfloat16));
                    try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
                    const got = try fused.bitsOf(&g, y, got_buf[0..w_n]);
                    const d = fused.countDiff(got, want, got_buf[0..w_n]);
                    const ok = d[0] == 0 and d[1] == 0;
                    checked += w_n;
                    std.debug.print("TURBOEXL3_GLM_BANKED_PTG_BYTES n={d} pairs={d} route=configured texts={s} gu={s} dn={s} out={s} differing={d}/{d} nonfinite={d} {s}\n", .{ n, c.pairs, if (cf.m3) "g4m3p" else "g4p", cf.gu, cf.dn, @tagName(dt), d[0], w_n, d[1], if (ok) "PASS" else "FAIL" });
                    if (!ok) failed = true;
                }
            }
        }
    }
    std.debug.print("TURBOEXL3_GLM_BANKED_PTG_BYTES total_words={d} {s}\n", .{ checked, if (failed) "FAIL" else "PASS" });
    try testing.expect(!failed);
}
