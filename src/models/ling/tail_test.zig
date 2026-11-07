//! `root.ling.moeTail` on the device (TURBOEXL3_GPU=1, else skipped): a MoE layer's tail at decode and verify rows in
//! one launch (tail.zig) against the composed ops it replaces in turboexl3_ling's served path, word for word: the
//! source's routed sum over the call's joined rows (MLX's astype of the weights to f32, its multiply of the rows by
//! them expanded on the last axis, its sum over the top-k axis), the MoE block's astype of the sum to bf16 and its add
//! of the shared expert, then mlx-serve's add_norm, whose words are MLX's add and MLX's fast rms_norm (its text mirrors
//! that kernel; addnorm_test.zig checks the same pair). The operands are built as the plugin builds them: the joined rows
//! are banked outputs (`moeLingBankedSel` on hw4b's texts, the control's BANKED_T4 with the configured variant t4, over a
//! 24-slot bank at n 40, the pack's I16 trellis words, limits 0 and 10; the texts named, as a process's banked state keeps
//! the group it was made at), given as the join gives them ([R, 8, 4096]) and flat ([R * 8, 4096]), and where the
//! turboexl3 has it, DEVROUTE's one-call wave's (`moeLingBankedLutSel`, turboexl3_ling's MLX_SERVE_LING_DEVWAVE_LUT=1:
//! x by token, each pair's id through the layer's LUT); the weights are the router's select (bf16,
//! normalized and scaled); the shared expert is SHARED_F3's two launches on the MoE input [1, R, 4096] at the rows they
//! take (1 to 4), and past them random bf16 rows [1, R, 4096] (the served path's stock chain there, any bf16 words to the
//! tail). Then planted signed zeros: -0.0 in the rows and the shared expert, zeros in the residual. Every case compares
//! the residual and the normed rows. Lines: TURBOEXL3_LING_TAIL.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const root = @import("../../root.zig");
const config = @import("../../core/config.zig");
const banked = @import("../deepseek_v41/banked.zig");
const ds_test = @import("../deepseek_v41/deepseek_test.zig");

const testing = std.testing;
const hidden: c_int = 4096;
const eps: f32 = 1e-6;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

const sumAxis = if (@hasDecl(mlx, "mlx_sum_axis")) mlx.mlx_sum_axis else struct {
    extern "c" fn mlx_sum_axis(res: *mlx.mlx_array, a: mlx.mlx_array, axis: c_int, keepdims: bool, s: mlx.mlx_stream) c_int;
}.mlx_sum_axis;
const expandDims = if (@hasDecl(mlx, "mlx_expand_dims")) mlx.mlx_expand_dims else struct {
    extern "c" fn mlx_expand_dims(res: *mlx.mlx_array, a: mlx.mlx_array, axis: c_int, s: mlx.mlx_stream) c_int;
}.mlx_expand_dims;

fn adopted(g: *G, rc: c_int, r: mlx.mlx_array) !mlx.mlx_array {
    mlx.check(rc) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
}

/// A projection's trellis as the plugin's loader hands it over: the pack's I16 words (the same bits as `p`'s U16)
fn packTrellis(g: *G, p: root.Proj) !root.Proj {
    var v = mlx.mlx_array_new();
    mlx.check(mlx.mlx_view(&v, p.trellis, .int16, g.s)) catch |e| {
        _ = mlx.mlx_array_free(v);
        return e;
    };
    return .{ .trellis = try g.adopt(v), .suh = p.suh, .svh = p.svh };
}

fn bf16Host(a: std.mem.Allocator, g: *G, vals: []const f32, shape: []const c_int) !mlx.mlx_array {
    const bits = try a.alloc(u16, vals.len);
    defer a.free(bits);
    for (vals, bits) |v, *b| b.* = support.bf16Bits(v);
    return g.hostArray(std.mem.sliceAsBytes(bits), shape, .bfloat16);
}

fn randFill(rnd: std.Random, xs: []f32, scale: f32, offset: f32) void {
    for (xs) |*v| v.* = offset + scale * rnd.floatNorm(f32);
}

fn wordsDiffer(p: []const f32, q: []const f32) usize {
    var d: usize = 0;
    for (p, q) |u, w| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return d;
}

/// The served chain after the join: the source's sum (`weightsOf`, `weigh`), the block's `sourceSum` and its add of
/// the shared expert, then the add_norm's words (MLX's add, MLX's fast rms_norm)
fn chain(g: *G, s: mlx.mlx_stream, h: mlx.mlx_array, ro: mlx.mlx_array, wts: mlx.mlx_array, shx: mlx.mlx_array, wn: mlx.mlx_array, rows: u32) !struct { h: mlx.mlx_array, normed: mlx.mlx_array } {
    const R = ci(rows);
    const w32 = try g.astype(wts, .float32);
    var w3 = mlx.mlx_array_new();
    w3 = try adopted(g, expandDims(&w3, w32, -1, s), w3);
    const prod = try g.mul(try g.astype(try g.reshape(ro, &.{ R, 8, hidden }), .float32), w3);
    var sm = mlx.mlx_array_new();
    sm = try adopted(g, sumAxis(&sm, prod, -2, false, s), sm);
    const mlp = try g.add(try g.reshape(try g.astype(sm, .bfloat16), &.{ 1, R, hidden }), try g.reshape(shx, &.{ 1, R, hidden }));
    const hn = try g.add(h, mlp);
    var n = mlx.mlx_array_new();
    n = try adopted(g, mlx.mlx_fast_rms_norm(&n, hn, wn, eps, s), n);
    return .{ .h = hn, .normed = n };
}

/// The residual's and the normed rows' differing words, moeTail against the chain
fn check(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, h: mlx.mlx_array, ro: mlx.mlx_array, wts: mlx.mlx_array, shx: mlx.mlx_array, wn: mlx.mlx_array, eps_a: mlx.mlx_array, rows: u32) ![2]usize {
    const want = try chain(g, s, h, ro, wts, shx, wn, rows);
    const got = try root.ling.moeTail(s, h, ro, wts, shx, wn, eps_a);
    const gh = try g.adopt(got.h);
    const gn = try g.adopt(got.normed);
    const n = @as(usize, rows) * 4096;
    const buf = try a.alloc(f32, 4 * n);
    defer a.free(buf);
    _ = try g.hostF32(try g.astype(try g.reshape(want.h, &.{ci(n)}), .float32), buf[0..n]);
    _ = try g.hostF32(try g.astype(try g.reshape(gh, &.{ci(n)}), .float32), buf[n .. 2 * n]);
    _ = try g.hostF32(try g.astype(try g.reshape(want.normed, &.{ci(n)}), .float32), buf[2 * n .. 3 * n]);
    _ = try g.hostF32(try g.astype(try g.reshape(gn, &.{ci(n)}), .float32), buf[3 * n ..]);
    return .{ wordsDiffer(buf[0..n], buf[n .. 2 * n]), wordsDiffer(buf[2 * n .. 3 * n], buf[3 * n ..]) };
}

test "turboexl3 gpu ling tail: moeTail gives the served chain's words (the source's routed sum, the astype, the shared add, then MLX's add and fast rms_norm, the add_norm's words) at rows 1 to 8, limits 0 and 10, the join's rows and flat rows, and planted signed zeros (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    const saved = config.get().*;
    defer config.configure(saved) catch {};
    var cfg = saved;
    cfg.banked_group = 4;
    cfg.banked_m3 = true;
    cfg.banked_gu8 = false;
    cfg.banked_variant_gu = banked.groupVariantOf("t4") orelse return error.NoT4Variant;
    cfg.banked_variant_dn = cfg.banked_variant_gu;
    try config.configure(cfg);
    var prng = std.Random.DefaultPrng.init(0x7A11_0E);
    const rnd = prng.random();
    const HostProj = ds_test.HostProj;
    var hp = [_]HostProj{
        try HostProj.draw(a, rnd, 24, 4096, 1280, 40),
        try HostProj.draw(a, rnd, 24, 1280, 4096, 40),
        try HostProj.draw(a, rnd, null, 4096, 1280, 96),
        try HostProj.draw(a, rnd, null, 1280, 4096, 96),
    };
    defer for (&hp) |*x| x.deinit(a);
    const bank: root.Bank = .{ .gate = try packTrellis(&g, try hp[0].upload(&g)), .up = try packTrellis(&g, try hp[0].upload(&g)), .down = try packTrellis(&g, try hp[1].upload(&g)) };
    const banks = [3]root.Bank{ bank, bank, bank };
    const sgp = try packTrellis(&g, try hp[2].upload(&g));
    const sup = try packTrellis(&g, try hp[2].upload(&g));
    const sdp = try packTrellis(&g, try hp[3].upload(&g));
    const tab = try g.adopt(try root.lingSigmoidTable(s, .bfloat16, false));
    const bias_h = try a.alloc(f32, 512);
    defer a.free(bias_h);
    randFill(rnd, bias_h, 0.01, 0);
    const bias = try g.hostArray(std.mem.sliceAsBytes(bias_h), &.{512}, .float32);
    const eps_a = try g.scalar(eps, .float32);
    const wn_h = try a.alloc(f32, 4096);
    defer a.free(wn_h);
    randFill(rnd, wn_h, 0.05, 1.0);
    const wn = try bf16Host(a, &g, wn_h, &.{4096});
    var checks: usize = 0;
    var failed: usize = 0;
    for ([_]c_int{ 0, 10 }) |lim| for (1..9) |ru| {
        const rows: u32 = @intCast(ru);
        const rm = g.mark();
        defer g.resetTo(rm);
        const R = ci(rows);
        const np: usize = rows * 8;
        const xs = try a.alloc(f32, 3 * rows * 4096 + rows * 512);
        defer a.free(xs);
        randFill(rnd, xs, 1.0, 0);
        const x = try bf16Host(a, &g, xs[0 .. rows * 4096], &.{ 1, R, 4096 });
        const hres = try bf16Host(a, &g, xs[rows * 4096 .. 2 * rows * 4096], &.{ 1, R, 4096 });
        const logits = try g.hostArray(std.mem.sliceAsBytes(xs[3 * rows * 4096 ..]), &.{ R, 512 }, .float32);
        const sel = try root.ling.router.select(s, logits, bias, 2.5);
        const wts = try g.adopt(sel.weights);
        _ = try g.adopt(sel.inds);
        var ids_h: [64]u32 = undefined;
        var tok_h: [64]i32 = undefined;
        for (ids_h[0..np], tok_h[0..np], 0..) |*d, *t, p| {
            d.* = rnd.uintLessThan(u32, 24);
            t.* = @intCast(p / 8);
        }
        const ids = try g.hostArray(std.mem.sliceAsBytes(ids_h[0..np]), &.{ci(np)}, .uint32);
        const xp = try g.take(try g.reshape(x, &.{ R, 4096 }), try g.hostArray(std.mem.sliceAsBytes(tok_h[0..np]), &.{ci(np)}, .int32), 0);
        const yr = try g.adopt(try root.moeLingBankedSel(s, xp, &banks, ids, .mul1, lim, .g4m3));
        // the shared expert: SHARED_F3's launches where they take the call (rows 1 to 4), else bf16 rows of their own
        const shx = if (try root.ling.sharedTakes(x, sgp, sup, sdp, .mul1))
            try g.adopt(try root.lingSharedExpert(s, x, sgp, sup, sdp, 0, tab, .mul1))
        else
            try bf16Host(a, &g, xs[2 * rows * 4096 .. 3 * rows * 4096], &.{ 1, R, 4096 });
        for ([_]bool{ true, false }) |joined| {
            const ro = if (joined) try g.reshape(yr, &.{ R, 8, 4096 }) else yr;
            const d = try check(a, &g, s, hres, ro, wts, shx, wn, eps_a, rows);
            const ok = d[0] == 0 and d[1] == 0;
            checks += 1;
            failed += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_LING_TAIL rows={d} limit={d} routed={s} residual_differing={d}/{d} normed_differing={d}/{d} {s}\n", .{ rows, lim, if (joined) "[R,8,D]" else "[R*8,D]", d[0], rows * 4096, d[1], rows * 4096, if (ok) "PASS" else "FAIL" });
        }
        // DEVROUTE's device wave in one call (turboexl3_ling's MLX_SERVE_LING_DEVWAVE_LUT=1), where this turboexl3 has it:
        // x by token, each pair's id the LUT's at its routed expert (the identity into bank 0 here), the rows as the
        // streamer's join hands them over
        if (comptime @hasDecl(root, "moeLingBankedLutSel")) {
            var lut_h: [24]u32 = undefined;
            for (&lut_h, 0..) |*v, e| v.* = @intCast(e);
            const lut = try g.hostArray(std.mem.sliceAsBytes(&lut_h), &.{24}, .uint32);
            const yl = try g.adopt(try root.moeLingBankedLutSel(s, try g.reshape(x, &.{ R, 4096 }), ids, lut, &banks, .mul1, lim, .g4m3));
            const d = try check(a, &g, s, hres, try g.reshape(yl, &.{ R, 8, 4096 }), wts, shx, wn, eps_a, rows);
            const ok = d[0] == 0 and d[1] == 0;
            checks += 1;
            failed += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_LING_TAIL rows={d} limit={d} routed=devwave_lut residual_differing={d}/{d} normed_differing={d}/{d} {s}\n", .{ rows, lim, d[0], rows * 4096, d[1], rows * 4096, if (ok) "PASS" else "FAIL" });
        }
    };
    // planted signed zeros: -0.0 in the routed rows and the shared expert, exact zeros in the residual
    for ([_]u32{ 3, 8 }) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        const R = ci(rows);
        const np: usize = rows * 8;
        const ro_h = try a.alloc(f32, np * 4096);
        defer a.free(ro_h);
        for (ro_h, 0..) |*v, i| v.* = switch (i % 7) {
            0 => -0.0,
            1 => 0.0,
            else => rnd.floatNorm(f32),
        };
        const sh_h = try a.alloc(f32, 2 * rows * 4096 + rows * 512);
        defer a.free(sh_h);
        randFill(rnd, sh_h, 1.0, 0);
        for (sh_h[0 .. rows * 4096], 0..) |*v, i| {
            if (i % 5 == 0) v.* = -0.0;
        }
        for (sh_h[rows * 4096 .. 2 * rows * 4096], 0..) |*v, i| {
            if (i % 3 == 0) v.* = 0.0;
        }
        const ro = try g.hostArray(std.mem.sliceAsBytes(ro_h), &.{ R, 8, 4096 }, .float32);
        const shx = try bf16Host(a, &g, sh_h[0 .. rows * 4096], &.{ 1, R, 4096 });
        const hres = try bf16Host(a, &g, sh_h[rows * 4096 .. 2 * rows * 4096], &.{ 1, R, 4096 });
        const logits = try g.hostArray(std.mem.sliceAsBytes(sh_h[2 * rows * 4096 ..]), &.{ R, 512 }, .float32);
        const sel = try root.ling.router.select(s, logits, bias, 2.5);
        const wts = try g.adopt(sel.weights);
        _ = try g.adopt(sel.inds);
        const d = try check(a, &g, s, hres, ro, wts, shx, wn, eps_a, rows);
        const ok = d[0] == 0 and d[1] == 0;
        checks += 1;
        failed += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_LING_TAIL rows={d} planted_signed_zeros residual_differing={d}/{d} normed_differing={d}/{d} {s}\n", .{ rows, d[0], rows * 4096, d[1], rows * 4096, if (ok) "PASS" else "FAIL" });
    }
    std.debug.print("TURBOEXL3_LING_TAIL_DONE checks={d} failed={d} {s}\n", .{ checks, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 0), failed);
}
