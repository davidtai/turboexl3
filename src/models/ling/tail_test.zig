//! `root.ling.moeTail` on the device (TURBOEXL3_GPU=1, else skipped): a MoE layer's tail at decode and verify rows in
//! one launch (tail.zig) against the composed ops it replaces in turboexl3_ling's served path, word for word: the
//! source's routed sum over the call's joined rows (MLX's astype of the weights to f32, its multiply of the rows by
//! them expanded on the last axis, its sum over the top-k axis), the MoE block's astype of the sum to bf16 and its add
//! of the shared expert, then mlx-serve's add_norm, whose words are MLX's add and MLX's fast rms_norm (its text mirrors
//! that kernel). The operands are built as the plugin builds them: the joined rows
//! are banked outputs (`moeLingBankedSel` on hw4b's texts, the control's BANKED_T4 with the configured variant t4, over a
//! 24-slot bank at n 40, the pack's I16 trellis words, limits 0 and 10; the texts named, as a process's banked state keeps
//! the group it was made at), given as the join gives them ([R, 8, 4096]) and flat ([R * 8, 4096]); the weights are the
//! router's select (bf16,
//! normalized and scaled); the shared expert is SHARED_F3's two launches on the MoE input [1, R, 4096] at the rows they
//! take (1 to 4), and past them random bf16 rows [1, R, 4096] (the served path's stock chain there, any bf16 words to the
//! tail). Then planted signed zeros: -0.0 in the rows and the shared expert, zeros in the residual. Every case compares
//! the residual and the normed rows. Lines: TURBOEXL3_LING_TAIL.
//! `moeTailParts` over a decode call's unjoined outputs (the device wave and one to seven miss parts, banked outputs
//! on the same texts) against the chain and `moeTail` over the streamer's join of them: TURBOEXL3_LING_TAIL_PARTS.
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

/// A decode call's join as the streamer makes it (experts.zig `join`): `take(concatenate(sources), order)` as
/// [rows, 8, D], where order[p] is the last row of the concatenation carrying position p (`joinOrder`: a miss part's row
/// replaces the device wave's for its pair), and moeTailParts' loc for the same rows: each position's (source, row)
const Join = struct { joined: mlx.mlx_array, loc: mlx.mlx_array };

fn joinOf(g: *G, sources: []const mlx.mlx_array, pos: []const []const u32, rows: u32) !Join {
    const np: usize = rows * 8;
    var order: [64]u32 = undefined;
    var loc_h: [128]i32 = undefined;
    var off: u32 = 0;
    for (pos, 0..) |ps, si| {
        for (ps, 0..) |p, j| {
            order[p] = off + @as(u32, @intCast(j));
            loc_h[2 * p] = @intCast(si);
            loc_h[2 * p + 1] = @intCast(j);
        }
        off += @intCast(ps.len);
    }
    const cat = try g.concat(sources, 0);
    const ord = try g.hostArray(std.mem.sliceAsBytes(order[0..np]), &.{ci(np)}, .uint32);
    const joined = try g.reshape(try g.take(cat, ord, 0), &.{ ci(rows), 8, hidden });
    const loc = try g.hostArray(std.mem.sliceAsBytes(loc_h[0 .. 2 * np]), &.{ ci(rows), 8, 2 }, .int32);
    return .{ .joined = joined, .loc = loc };
}

/// moeTailParts over the sources against the served chain over their join and against moeTail over it: the residual's
/// and the normed rows' differing words, [chain residual, chain normed, moeTail residual, moeTail normed]
fn checkParts(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, h: mlx.mlx_array, sources: []const mlx.mlx_array, j: Join, wts: mlx.mlx_array, shx: mlx.mlx_array, wn: mlx.mlx_array, eps_a: mlx.mlx_array, rows: u32) ![4]usize {
    const want = try chain(g, s, h, j.joined, wts, shx, wn, rows);
    const tj = try root.ling.moeTail(s, h, j.joined, wts, shx, wn, eps_a);
    const th = try g.adopt(tj.h);
    const tn = try g.adopt(tj.normed);
    const got = try root.ling.moeTailParts(s, h, sources, j.loc, wts, shx, wn, eps_a);
    const gh = try g.adopt(got.h);
    const gn = try g.adopt(got.normed);
    const n = @as(usize, rows) * 4096;
    const buf = try a.alloc(f32, 6 * n);
    defer a.free(buf);
    const arrs = [_]mlx.mlx_array{ want.h, want.normed, th, tn, gh, gn };
    for (arrs, 0..) |x, i| _ = try g.hostF32(try g.astype(try g.reshape(x, &.{ci(n)}), .float32), buf[i * n .. (i + 1) * n]);
    return .{
        wordsDiffer(buf[4 * n .. 5 * n], buf[0..n]),
        wordsDiffer(buf[5 * n .. 6 * n], buf[n .. 2 * n]),
        wordsDiffer(buf[4 * n .. 5 * n], buf[2 * n .. 3 * n]),
        wordsDiffer(buf[5 * n .. 6 * n], buf[3 * n .. 4 * n]),
    };
}

// Bar: moeTailParts over a decode call's unjoined outputs gives the words of the served chain and of moeTail over the
// streamer's join of them, wherever the miss parts' rows replace the device wave's; and it refuses what it cannot take.
test "turboexl3 gpu ling tail parts: moeTailParts reads each pair's row in place from the call's unjoined outputs, the words of the served chain and of moeTail over the streamer's join (take of the concatenate at each pair's last carrier), at rows 1 to 8, limits 0 and 10, one to three sources and all eight, and its refusals (TURBOEXL3_GPU=1)" {
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
    cfg.banked_variant_gu = banked.groupVariantOf("t4") orelse return error.NoT4Variant;
    cfg.banked_variant_dn = cfg.banked_variant_gu;
    try config.configure(cfg);
    var prng = std.Random.DefaultPrng.init(0x7A11_0F);
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
    // (rows, sources): 1 to 3 sources at every row count, all eight at 8 rows
    const Case = struct { rows: u32, n_src: usize };
    var cases: [25]Case = undefined;
    var n_cases: usize = 0;
    for (1..9) |ru| for (1..4) |ns| {
        cases[n_cases] = .{ .rows = @intCast(ru), .n_src = ns };
        n_cases += 1;
    };
    cases[n_cases] = .{ .rows = 8, .n_src = 8 };
    n_cases += 1;
    for ([_]c_int{ 0, 10 }) |lim| for (cases[0..n_cases]) |cs| {
        const rows = cs.rows;
        const rm = g.mark();
        defer g.resetTo(rm);
        const R = ci(rows);
        const np: usize = rows * 8;
        const xs = try a.alloc(f32, 3 * rows * 4096 + rows * 512);
        defer a.free(xs);
        randFill(rnd, xs, 1.0, 0);
        const x = try bf16Host(a, &g, xs[0 .. rows * 4096], &.{ 1, R, 4096 });
        const xr = try g.reshape(x, &.{ R, 4096 });
        const hres = try bf16Host(a, &g, xs[rows * 4096 .. 2 * rows * 4096], &.{ 1, R, 4096 });
        const logits = try g.hostArray(std.mem.sliceAsBytes(xs[3 * rows * 4096 ..]), &.{ R, 512 }, .float32);
        const sel = try root.ling.router.select(s, logits, bias, 2.5);
        const wts = try g.adopt(sel.weights);
        _ = try g.adopt(sel.inds);
        const shx = if (try root.ling.sharedTakes(x, sgp, sup, sdp, .mul1))
            try g.adopt(try root.lingSharedExpert(s, x, sgp, sup, sdp, 0, tab, .mul1))
        else
            try bf16Host(a, &g, xs[2 * rows * 4096 .. 3 * rows * 4096], &.{ 1, R, 4096 });
        // the device wave: every pair on its routed slot, as DEVROUTE's wave gives the rows (hw4b's texts named)
        var ids_h: [64]u32 = undefined;
        var all_pos: [64]u32 = undefined;
        var tok_h: [64]i32 = undefined;
        for (ids_h[0..np], all_pos[0..np], tok_h[0..np], 0..) |*d, *p, *t, i| {
            d.* = rnd.uintLessThan(u32, 24);
            p.* = @intCast(i);
            t.* = @intCast(i / 8);
        }
        var sources: [8]mlx.mlx_array = undefined;
        var pos_store: [8][16]u32 = undefined;
        var pos: [8][]const u32 = undefined;
        sources[0] = try g.adopt(try root.moeLingBankedSel(s, try g.take(xr, try g.hostArray(std.mem.sliceAsBytes(tok_h[0..np]), &.{ci(np)}, .int32), 0), &banks, try g.hostArray(std.mem.sliceAsBytes(ids_h[0..np]), &.{ci(np)}, .uint32), .mul1, lim, .g4m3));
        pos[0] = all_pos[0..np];
        // the miss parts: some pairs again on another slot (as a miss part's row replaces the device wave's), the later
        // part's row last where two parts carry a pair
        for (1..cs.n_src) |si| {
            var m: usize = 0;
            for (0..np) |p| {
                const take_it = if (cs.n_src == 8) (p / 9 == si - 1 and p % 9 < 3) or (si == 7 and p == 1) else p % (si + 2) == si;
                if (take_it and m < 16) {
                    pos_store[si][m] = @intCast(p);
                    m += 1;
                }
            }
            if (m == 0) {
                pos_store[si][0] = @intCast(np - 1);
                m = 1;
            }
            var mids: [16]u32 = undefined;
            var mtok: [16]i32 = undefined;
            for (pos_store[si][0..m], mids[0..m], mtok[0..m]) |p, *d, *t| {
                d.* = (ids_h[p] + 5 * @as(u32, @intCast(si)) + 1) % 24;
                t.* = @intCast(p / 8);
            }
            sources[si] = try g.adopt(try root.moeLingBankedSel(s, try g.take(xr, try g.hostArray(std.mem.sliceAsBytes(mtok[0..m]), &.{ci(m)}, .int32), 0), &banks, try g.hostArray(std.mem.sliceAsBytes(mids[0..m]), &.{ci(m)}, .uint32), .mul1, lim, .g4m3));
            pos[si] = pos_store[si][0..m];
        }
        const j = try joinOf(&g, sources[0..cs.n_src], pos[0..cs.n_src], rows);
        const d = try checkParts(a, &g, s, hres, sources[0..cs.n_src], j, wts, shx, wn, eps_a, rows);
        const ok = d[0] == 0 and d[1] == 0 and d[2] == 0 and d[3] == 0;
        checks += 1;
        failed += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_LING_TAIL_PARTS rows={d} limit={d} sources={d} vs_chain residual_differing={d}/{d} normed_differing={d}/{d} vs_moeTail residual_differing={d} normed_differing={d} {s}\n", .{ rows, lim, cs.n_src, d[0], rows * 4096, d[1], rows * 4096, d[2], d[3], if (ok) "PASS" else "FAIL" });
    };
    // the refusals: no source, nine, a bf16 source, a source of another width, loc of the wrong size or dtype
    {
        const rm = g.mark();
        defer g.resetTo(rm);
        const R: c_int = 2;
        const np: c_int = 16;
        const vals = try a.alloc(f32, 3 * 2 * 4096 + 2 * 512);
        defer a.free(vals);
        randFill(rnd, vals, 1.0, 0);
        const hres = try bf16Host(a, &g, vals[0 .. 2 * 4096], &.{ 1, R, 4096 });
        const shx = try bf16Host(a, &g, vals[2 * 4096 .. 4 * 4096], &.{ 1, R, 4096 });
        const logits = try g.hostArray(std.mem.sliceAsBytes(vals[6 * 4096 ..]), &.{ R, 512 }, .float32);
        const sel = try root.ling.router.select(s, logits, bias, 2.5);
        const wts = try g.adopt(sel.weights);
        _ = try g.adopt(sel.inds);
        const src = try g.zeros(&.{ np, 4096 }, .float32);
        var loc_h: [32]i32 = undefined;
        for (0..16) |p| {
            loc_h[2 * p] = 0;
            loc_h[2 * p + 1] = @intCast(p);
        }
        const loc = try g.hostArray(std.mem.sliceAsBytes(&loc_h), &.{ R, 8, 2 }, .int32);
        const loc_short = try g.hostArray(std.mem.sliceAsBytes(loc_h[0..30]), &.{30}, .int32);
        const loc_u32 = try g.hostArray(std.mem.sliceAsBytes(&loc_h), &.{ R, 8, 2 }, .uint32);
        const nine: [9]mlx.mlx_array = @splat(src);
        const parts = root.ling.moeTailParts;
        var refused: usize = 0;
        refused += @intFromBool(if (parts(s, hres, &.{}, loc, wts, shx, wn, eps_a)) |_| false else |e| e == error.BadExl3Shape);
        refused += @intFromBool(if (parts(s, hres, &nine, loc, wts, shx, wn, eps_a)) |_| false else |e| e == error.BadExl3Shape);
        refused += @intFromBool(if (parts(s, hres, &.{try g.astype(src, .bfloat16)}, loc, wts, shx, wn, eps_a)) |_| false else |e| e == error.BadExl3Dtype);
        refused += @intFromBool(if (parts(s, hres, &.{try g.zeros(&.{ np, 2048 }, .float32)}, loc, wts, shx, wn, eps_a)) |_| false else |e| e == error.BadExl3Shape);
        refused += @intFromBool(if (parts(s, hres, &.{src}, loc_short, wts, shx, wn, eps_a)) |_| false else |e| e == error.BadExl3Shape);
        refused += @intFromBool(if (parts(s, hres, &.{src}, loc_u32, wts, shx, wn, eps_a)) |_| false else |e| e == error.BadExl3Dtype);
        const ok = refused == 6;
        checks += 1;
        failed += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_LING_TAIL_PARTS refusals={d}/6 {s}\n", .{ refused, if (ok) "PASS" else "FAIL" });
    }
    std.debug.print("TURBOEXL3_LING_TAIL_PARTS_DONE checks={d} failed={d} {s}\n", .{ checks, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── L3: the MoE tail at prompt widths (`moeTailWide`, turboexl3_ling's MLX_SERVE_LING_PF_MOE_TAIL) ──
//
// CPU: the wide texts' compositions (kernels/test/pins/tail_wide.metal). GPU (TURBOEXL3_GPU=1): `moeTailWide` against
// the served chain at prompt widths, word for word, at the standard request's spans (4,096 x 4, 1,965 and 31 rows of
// one routed call of 18,380 tokens at top-8 over 512 slots, lognormal popularity sigma 1.5): the call's outputs in the
// plugin's chunks of whole experts (at most 4,096 pairs), the streamer's JOINLESS plan (above 24 outputs the smallest
// merged into the last source), each output's rows random f16 words (PF_F16_PAIRS's rows) and the same words widened
// to f32; the routes' weights bf16 values read in f32; the served chain is the combine over the parts
// (`combine.combineParts`), its astype to bf16, the span's slice, the block's add of the shared expert, MLX's add of
// the residual and MLX's fast rms_norm (TURBOEXL3_LING_PF_TAIL). Red checks (TURBOEXL3_LING_PF_TAIL_RED): a planted
// reordering of the norm's threadgroup sum (its 32 simdgroup sums added one by one in reverse) must show in the normed
// rows of the whole call, and the refusals. A thread's own four squares are no such check: four squares of bf16 words
// (16-bit significands) round at most once in practice, so their order gives the same partial sum and the same words
// (a reversal planted there changed none of 16,777,216 normed words on the box).

const tail = @import("tail.zig");
const combine = @import("combine.zig");
const rt = @import("rows_test.zig");
const ct = @import("combine_test.zig");
const ks = sdk.kernel_launch;
const decode = @import("../../core/decode.zig");

/// The statements the wide texts' checks pin (kernels/test/pins/tail_wide.metal): 0 the f32 source pointer, 1 the f16
/// one, 2 the combine's pick call, 3 its f16 pick's, 4 the decode parts delta's pick, 5 the wide delta's pick, 6 the
/// delta's block pragma, 7 W from WE, 8 eps from WE, 9 the fold's first statement, 10 the norm's threadgroup sum (the
/// simd sum over the simdgroups' sums), 11 the fault's reversed threadgroup sum, 12 the norm's rsqrt, 13 the fault's
/// first statement
const wide_pin = support.pins(@embedFile("kernels/test/pins/tail_wide.metal"));
/// The planted fault (kernels/test/fault/): tail_2 with the threadgroup's simdgroup sums added one by one in reverse
const fault_tgsum_rev = @embedFile("kernels/test/fault/tail_2_tgsum_rev.metal");
const text_wide_tgsum_rev: [:0]const u8 = tail.tail_we ++ tail.tail_1 ++ tail.tail_delta_wide_h ++ fault_tgsum_rev;

/// `s` up to its first `needle`
fn upTo(comptime s: []const u8, comptime needle: []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(10_000_000);
        const j = std.mem.indexOf(u8, s, needle) orelse @compileError("needle not in the text");
        return s[0..j];
    }
}

/// `s` with every `needle` replaced by `with`, at comptime (the checks compare pieces swapped this way; no kernel is
/// built from it)
fn swapAll(comptime s: []const u8, comptime needle: []const u8, comptime with: []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(10_000_000);
        var out: []const u8 = "";
        var i: usize = 0;
        while (std.mem.indexOfPos(u8, s, i, needle)) |j| {
            out = out ++ s[i..j] ++ with;
            i = j + needle.len;
        }
        return out ++ s[i..];
    }
}

/// `s` from the first `needle` on
fn fromFirst(comptime s: []const u8, comptime needle: []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(10_000_000);
        const j = std.mem.indexOf(u8, s, needle) orelse @compileError("needle not in the text");
        return s[j..];
    }
}

fn holds(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

test "turboexl3 ling tail wide texts: the prompt-width launch is the parts launch's composition with W and eps read from WE and the combine's 24-source pick; the f16 text is the f32 one with the source pointer half and pick_h; the headers are the combine's picks with no file-scope pragma; the fault swaps tail_2's threadgroup sum alone" {
    try testing.expect(std.mem.eql(u8, tail.text_wide, tail.tail_we ++ tail.tail_1 ++ tail.tail_delta_wide ++ tail.tail_2));
    try testing.expect(std.mem.eql(u8, tail.text_wide_h, tail.tail_we ++ tail.tail_1 ++ tail.tail_delta_wide_h ++ tail.tail_2));
    try testing.expectEqualStrings(combine.pick, tail.header_wide);
    try testing.expectEqualStrings(combine.pick_h, tail.header_wide_h);
    // the f16 delta is the f32 one with the source pointer half and the pick renamed, nothing else
    try testing.expectEqualStrings(tail.tail_delta_wide_h, comptime swapAll(swapAll(tail.tail_delta_wide, wide_pin[0], wide_pin[1]), wide_pin[2], wide_pin[3]));
    try testing.expect(!holds(tail.tail_delta_wide_h, wide_pin[0]) and !holds(tail.tail_delta_wide, wide_pin[1]));
    // the wide delta's statements are the decode parts delta's with the 24-source pick for the 8-source one
    try testing.expectEqualStrings(comptime fromFirst(tail.tail_delta_parts, wide_pin[9]), comptime swapAll(fromFirst(tail.tail_delta_wide, wide_pin[9]), wide_pin[5], wide_pin[4]));
    // the combine's rounding points by the delta's own block pragma; the norm compiles as the add_norm's (no pragma in
    // the headers, none in tail_we, tail_1 or tail_2)
    try testing.expect(holds(tail.tail_delta_wide, wide_pin[6]) and holds(tail.tail_delta_wide_h, wide_pin[6]));
    for ([_][]const u8{ tail.header_wide, tail.header_wide_h, tail.tail_we, tail.tail_1, tail.tail_2 }) |piece| try testing.expect(!holds(piece, wide_pin[6]));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, tail.text_wide, wide_pin[6]));
    // W and eps from the one buffer WE
    try testing.expect(holds(tail.tail_we, wide_pin[7]) and holds(tail.tail_we, wide_pin[8]));
    // every operand a wide launch binds is a name its text or header reads; no name takes MLX's shape / strides suffixes
    for ([_][:0]const u8{ tail.text_wide, tail.text_wide_h }, [_][:0]const u8{ tail.header_wide, tail.header_wide_h }) |txt, hdr| {
        for (tail.wide_in_names) |n| try testing.expect(holds(txt, std.mem.span(n)) or holds(hdr, std.mem.span(n)));
        for (tail.out_names) |n| try testing.expect(holds(txt, std.mem.span(n)));
        for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| try testing.expect(!holds(txt, suf));
    }
    // the fault: tail_2 with the simd sum over the simdgroups' sums swapped for their sum one by one in reverse; up to
    // it (the squares, the simd sums, local_sums) and from rsqrt on (the stores) it is tail_2's
    try testing.expect(!std.mem.eql(u8, fault_tgsum_rev, tail.tail_2));
    try testing.expect(holds(tail.tail_2, wide_pin[10]) and !holds(fault_tgsum_rev, wide_pin[10]) and holds(fault_tgsum_rev, wide_pin[11]));
    try testing.expect(!holds(tail.tail_2, wide_pin[11]) and !holds(tail.tail_2, wide_pin[13]));
    try testing.expectEqualStrings(comptime upTo(tail.tail_2, wide_pin[10]), comptime upTo(fault_tgsum_rev, wide_pin[13]));
    try testing.expectEqualStrings(comptime fromFirst(tail.tail_2, wide_pin[12]), comptime fromFirst(fault_tgsum_rev, wide_pin[12]));
    try testing.expect(std.mem.endsWith(u8, text_wide_tgsum_rev, fault_tgsum_rev) and std.mem.startsWith(u8, text_wide_tgsum_rev, tail.tail_we ++ tail.tail_1 ++ tail.tail_delta_wide_h));
}

test "turboexl3 ling tail wide takes: any row count from 1, K 1 to 8, a width a multiple of 4 up to 4096, bf16 or f16" {
    try testing.expect(tail.wideTakes(1, 4096, 8, .bfloat16) and tail.wideTakes(18_380, 4096, 8, .bfloat16) and tail.wideTakes(4607, 4096, 8, .float16));
    try testing.expect(!tail.wideTakes(0, 4096, 8, .bfloat16));
    try testing.expect(!tail.wideTakes(31, 4096, 9, .bfloat16) and !tail.wideTakes(31, 4096, 0, .bfloat16));
    try testing.expect(!tail.wideTakes(31, 4100, 8, .bfloat16) and !tail.wideTakes(31, 8192, 8, .bfloat16) and !tail.wideTakes(31, 4094, 8, .bfloat16));
    try testing.expect(!tail.wideTakes(31, 4096, 8, .float32));
    try testing.expectEqual(@as(usize, 24), tail.wide_max_sources);
    try testing.expectEqual(@as(usize, 31), tail.wide_in_names.len + tail.out_names.len);
}

/// The plugin's sort of a call's pairs by slot (ling_math.zig `chunks()`: a counting sort, stable): each sorted pair's
/// position and slot
fn slotOrder(a: std.mem.Allocator, slot: []const u32, order: []u32, s_slot: []u32) !void {
    var max_slot: u32 = 0;
    for (slot) |sv| max_slot = @max(max_slot, sv);
    const starts = try a.alloc(u32, @as(usize, max_slot) + 2);
    defer a.free(starts);
    @memset(starts, 0);
    for (slot) |sv| starts[sv + 1] += 1;
    for (1..starts.len) |i| starts[i] += starts[i - 1];
    for (slot, 0..) |sv, i| {
        order[starts[sv]] = @intCast(i);
        s_slot[starts[sv]] = sv;
        starts[sv] += 1;
    }
}

/// A random array on the device (normal, scale 1) of `shape` in `dt`
fn devRandom(g: *G, s: mlx.mlx_stream, shape: []const c_int, dt: mlx.mlx_dtype) !mlx.mlx_array {
    const nokey = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nokey);
    var r = mlx.mlx_array_new();
    return adopted(g, mlx.mlx_random_normal(&r, shape.ptr, shape.len, dt, 0, 1, nokey, s), r);
}

/// The residual's and the normed rows' differing words of a wide launch's output against the chain's (n words each)
fn wideDiffer(a: std.mem.Allocator, g: *G, got: tail.Out, want_h: mlx.mlx_array, want_n: mlx.mlx_array, n: usize) ![2]usize {
    const gh = try g.adopt(got.h);
    const gn = try g.adopt(got.normed);
    const buf = try a.alloc(f32, 4 * n);
    defer a.free(buf);
    _ = try g.hostF32(try g.astype(try g.reshape(want_h, &.{ci(n)}), .float32), buf[0..n]);
    _ = try g.hostF32(try g.astype(try g.reshape(gh, &.{ci(n)}), .float32), buf[n .. 2 * n]);
    _ = try g.hostF32(try g.astype(try g.reshape(want_n, &.{ci(n)}), .float32), buf[2 * n .. 3 * n]);
    _ = try g.hostF32(try g.astype(try g.reshape(gn, &.{ci(n)}), .float32), buf[3 * n ..]);
    return .{ wordsDiffer(buf[0..n], buf[n .. 2 * n]), wordsDiffer(buf[2 * n .. 3 * n], buf[3 * n ..]) };
}

/// One refusal: the call must return `want` before any launch
fn refuses(r: anyerror!tail.Out, want: anyerror) bool {
    if (r) |o| {
        _ = mlx.mlx_array_free(o.h);
        _ = mlx.mlx_array_free(o.normed);
        return false;
    } else |e| return e == want;
}

test "turboexl3 gpu ling tail wide: moeTailWide gives the served chain's words at prompt widths (the combine over the call's JOINLESS parts, the block's astype and shared add, MLX's add and fast rms_norm) at the standard request's spans 4096 x 4, 1965 and 31 over 24 merged sources, f16 and f32; a planted reordering of the norm shows, and its refusals (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    var prng = std.Random.DefaultPrng.init(0x13_7A11_E0);
    const rnd = prng.random();
    const spans = [_]u32{ 4096, 4096, 4096, 4096, 1965, 31 };
    var tokens: u32 = 0;
    for (spans) |r| tokens += r;
    const np: usize = @as(usize, tokens) * 8;
    // the call's pairs (each token's top-8 of 512 slots), the plugin's stable sort by slot and its chunks
    const act = try a.alloc(u32, np);
    defer a.free(act);
    const slot = try a.alloc(u32, np);
    defer a.free(slot);
    try rt.drawPairsAt(a, rnd, tokens, 512, 1.5, act, slot);
    const order = try a.alloc(u32, np);
    defer a.free(order);
    const s_slot = try a.alloc(u32, np);
    defer a.free(s_slot);
    try slotOrder(a, slot, order, s_slot);
    var out_rows: std.ArrayList(u32) = .empty;
    defer out_rows.deinit(a);
    var c0: usize = 0;
    while (c0 < np) {
        const c1 = rt.chunkEnd(s_slot, c0);
        try out_rows.append(a, @intCast(c1 - c0));
        c0 = c1;
    }
    const n_out = out_rows.items.len;
    // the streamer's JOINLESS plan and each pair's (source, row)
    const jorder = try a.alloc(u32, n_out);
    defer a.free(jorder);
    const src_of = try a.alloc(u8, n_out);
    defer a.free(src_of);
    const off = try a.alloc(u32, n_out);
    defer a.free(off);
    const n_src = ct.planJoinless(out_rows.items, jorder, src_of, off);
    const loc_h = try a.alloc(i32, 2 * np);
    defer a.free(loc_h);
    ct.fillJoinlessLoc(out_rows.items, src_of, off, order, loc_h);
    // each output's rows random f16 words; the merged outputs concatenated in join order into the last source
    var srcs16: std.ArrayList(mlx.mlx_array) = .empty;
    defer srcs16.deinit(a);
    var merge_in: std.ArrayList(mlx.mlx_array) = .empty;
    defer merge_in.deinit(a);
    for (out_rows.items, src_of) |r, sv| {
        const y = try devRandom(&g, s, &.{ ci(r), hidden }, .float16);
        if (n_src < n_out and sv == n_src - 1) try merge_in.append(a, y) else try srcs16.append(a, y);
    }
    if (merge_in.items.len > 0) try srcs16.append(a, try g.concat(merge_in.items, 0));
    try testing.expectEqual(n_src, srcs16.items.len);
    const srcs32 = try a.alloc(mlx.mlx_array, srcs16.items.len);
    defer a.free(srcs32);
    for (srcs16.items, srcs32) |x, *y| y.* = try g.astype(x, .float32);
    // the routes' weights: bf16 values (a router's after its scale) read in f32, as the combine reads them
    const w_h = try a.alloc(f32, np);
    defer a.free(w_h);
    for (w_h) |*v| v.* = @bitCast(@as(u32, support.bf16Bits(0.01 + 2.49 * rnd.float(f32))) << 16);
    const wa = try g.hostArray(std.mem.sliceAsBytes(w_h), &.{ ci(tokens), 8 }, .float32);
    const loc_all = try g.hostArray(std.mem.sliceAsBytes(loc_h), &.{ ci(tokens), 8, 2 }, .int32);
    // the next norm's weight (bf16 near 1), eps, and WE f32 [D + 1]
    const wn_h = try a.alloc(f32, 4096);
    defer a.free(wn_h);
    randFill(rnd, wn_h, 0.05, 1.0);
    const wn = try bf16Host(a, &g, wn_h, &.{4096});
    const eps_a = try g.scalar(eps, .float32);
    const we = try g.concat(&.{ try g.astype(wn, .float32), try g.reshape(eps_a, &.{1}) }, 0);
    // the served chain's routed sum over the whole call (the combine over the parts, its astype), per source dtype
    const y16 = try g.reshape(try g.astype(try g.adopt(try combine.combineParts(s, srcs16.items, loc_all, wa)), .bfloat16), &.{ 1, ci(tokens), hidden });
    const y32 = try g.reshape(try g.astype(try g.adopt(try combine.combineParts(s, srcs32, loc_all, wa)), .bfloat16), &.{ 1, ci(tokens), hidden });
    var checks: usize = 0;
    var failed: usize = 0;
    var at: c_int = 0;
    for (spans, 0..) |r, j| {
        const S = ci(r);
        const sm = g.mark();
        defer g.resetTo(sm);
        const h_j = try devRandom(&g, s, &.{ 1, S, hidden }, .bfloat16);
        const sh_j = try devRandom(&g, s, &.{ 1, S, hidden }, .bfloat16);
        const loc_j = try g.slice(loc_all, &.{ at, 0, 0 }, &.{ at + S, 8, 2 }, &.{ 1, 1, 1 });
        const w_j = try g.slice(wa, &.{ at, 0 }, &.{ at + S, 8 }, &.{ 1, 1 });
        // f16 sources at every span; f32 at the first, the 1,965-row and the 31-row spans
        for ([_]bool{ true, false }) |half| {
            if (!half and !(j == 0 or r == 1965 or r == 31)) continue;
            const part = try g.slice(if (half) y16 else y32, &.{ 0, at, 0 }, &.{ 1, at + S, hidden }, &.{ 1, 1, 1 });
            const hn = try g.add(h_j, try g.add(part, sh_j));
            var nrm = mlx.mlx_array_new();
            nrm = try adopted(&g, mlx.mlx_fast_rms_norm(&nrm, hn, wn, eps, s), nrm);
            const n_words = @as(usize, r) * 4096;
            const got = root.ling.moeTailWide(s, h_j, if (half) srcs16.items else srcs32, loc_j, w_j, sh_j, we) catch |e| {
                std.debug.print("TURBOEXL3_LING_PF_TAIL sources={s} span={d} rows={d} at={d}: {s} FAIL\n", .{ if (half) "f16" else "f32", j, r, at, @errorName(e) });
                checks += 1;
                failed += 1;
                continue;
            };
            const d = try wideDiffer(a, &g, got, hn, nrm, n_words);
            const ok = d[0] == 0 and d[1] == 0;
            checks += 1;
            failed += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_LING_PF_TAIL sources={s} n_src={d} outputs={d} merged={d} span={d} rows={d} at={d} residual_differing={d}/{d} normed_differing={d}/{d} {s}\n", .{ if (half) "f16" else "f32", srcs16.items.len, n_out, merge_in.items.len, j, r, at, d[0], n_words, d[1], n_words, if (ok) "PASS" else "FAIL" });
        }
        at += S;
    }
    // red check 1: the planted reordering of the norm's threadgroup sum over the whole call's rows (one launch of 18,380
    // rows over the f16 sources): the residual keeps the chain's words, the normed rows must not. A reordered sum moves
    // a row's rsqrt by an ulp at most, which turns about one bf16 word in 65,536, so the check reads every row.
    {
        const sm = g.mark();
        defer g.resetTo(sm);
        const kern = ks.newKernel("turboexl3_test_ling_moe_tail_wide_tgsum_rev_fault", &tail.wide_in_names, &tail.out_names, text_wide_tgsum_rev, tail.header_wide_h, true) orelse return error.KernelCreateFailed;
        defer decode.freeKernel(kern);
        const S = ci(tokens);
        const h_c = try devRandom(&g, s, &.{ 1, S, hidden }, .bfloat16);
        const sh_c = try devRandom(&g, s, &.{ 1, S, hidden }, .bfloat16);
        const hn = try g.add(h_c, try g.add(y16, sh_c));
        var nrm = mlx.mlx_array_new();
        nrm = try adopted(&g, mlx.mlx_fast_rms_norm(&nrm, hn, wn, eps, s), nrm);
        const got = try tail.moeTailWideWith(s, kern, h_c, srcs16.items, loc_all, wa, sh_c, we);
        const n_words: usize = @as(usize, tokens) * 4096;
        const d = try wideDiffer(a, &g, got, hn, nrm, n_words);
        const ok = d[0] == 0 and d[1] > 0;
        checks += 1;
        failed += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_LING_PF_TAIL_RED fault=tgsum_rev rows={d} normed_differing={d}/{d} (a fault must differ) residual_differing={d}/{d} (must not) {s}\n", .{ tokens, d[1], n_words, d[0], n_words, if (ok) "PASS" else "FAIL" });
    }
    // red check 2: the refusals, before any launch
    {
        const sm = g.mark();
        defer g.resetTo(sm);
        const R: c_int = 2;
        const h2 = try devRandom(&g, s, &.{ 1, R, hidden }, .bfloat16);
        const sh2 = try devRandom(&g, s, &.{ 1, R, hidden }, .bfloat16);
        const src16 = try devRandom(&g, s, &.{ 16, hidden }, .float16);
        var lh: [32]i32 = undefined;
        for (0..16) |p| {
            lh[2 * p] = 0;
            lh[2 * p + 1] = @intCast(p);
        }
        const loc2 = try g.hostArray(std.mem.sliceAsBytes(&lh), &.{ R, 8, 2 }, .int32);
        const w2 = try g.slice(wa, &.{ 0, 0 }, &.{ R, 8 }, &.{ 1, 1 });
        const wide = tail.moeTailWide;
        const many: [25]mlx.mlx_array = @splat(src16);
        var refused: usize = 0;
        refused += @intFromBool(refuses(wide(s, try g.astype(h2, .float32), &.{src16}, loc2, w2, try g.astype(sh2, .float32), we), error.BadExl3Shape));
        refused += @intFromBool(refuses(wide(s, try devRandom(&g, s, &.{ 1, R, 4100 }, .bfloat16), &.{try devRandom(&g, s, &.{ 16, 4100 }, .float16)}, loc2, w2, try devRandom(&g, s, &.{ 1, R, 4100 }, .bfloat16), try g.zeros(&.{4101}, .float32)), error.BadExl3Shape));
        refused += @intFromBool(refuses(wide(s, h2, &.{src16}, try g.zeros(&.{ R, 9, 2 }, .int32), try g.zeros(&.{ R, 9 }, .float32), sh2, we), error.BadExl3Shape));
        refused += @intFromBool(refuses(wide(s, h2, &many, loc2, w2, sh2, we), error.BadExl3Shape));
        refused += @intFromBool(refuses(wide(s, h2, &.{ src16, try g.astype(src16, .float32) }, loc2, w2, sh2, we), error.BadExl3Dtype));
        refused += @intFromBool(refuses(wide(s, h2, &.{try g.astype(src16, .bfloat16)}, loc2, w2, sh2, we), error.BadExl3Dtype));
        refused += @intFromBool(refuses(wide(s, h2, &.{src16}, try g.zeros(&.{ R, 8, 3 }, .int32), w2, sh2, we), error.BadExl3Shape));
        refused += @intFromBool(refuses(wide(s, h2, &.{src16}, try g.astype(loc2, .uint32), w2, sh2, we), error.BadExl3Dtype));
        refused += @intFromBool(refuses(wide(s, h2, &.{src16}, loc2, w2, sh2, try g.zeros(&.{4096}, .float32)), error.BadExl3Shape));
        refused += @intFromBool(refuses(wide(s, h2, &.{src16}, loc2, w2, sh2, try g.astype(we, .bfloat16)), error.BadExl3Dtype));
        refused += @intFromBool(refuses(wide(s, h2, &.{src16}, loc2, w2, try g.astype(sh2, .float32), we), error.BadExl3Dtype));
        const ok = refused == 11;
        checks += 1;
        failed += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_LING_PF_TAIL_RED refusals={d}/11 (an f32 residual, width 4100, top-9, 25 sources, mixed f16 and f32 sources, a bf16 source, loc of shape [R, 8, 3], loc as u32, WE of D words, a bf16 WE, an f32 shared) {s}\n", .{ refused, if (ok) "PASS" else "FAIL" });
    }
    std.debug.print("TURBOEXL3_LING_PF_TAIL_DONE checks={d} failed={d} {s}\n", .{ checks, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 0), failed);
}
