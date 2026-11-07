//! `root.ling.addNormLogits` on the device (TURBOEXL3_GPU=1, else skipped): a MoE layer's post-attention add_norm and
//! the router's launch 1 in one launch (anroute.zig) against the two launches it replaces in turboexl3_ling's served path
//! under ROUTER_FUSED (E2) and ROUTER_F16 (E10), word for word: mlx-serve's add_norm, whose words are MLX's add and MLX's
//! fast rms_norm (its text mirrors that kernel; addnorm_test.zig checks the same pair), then launch 1
//! (`router.routerLogits`, router_test.zig's served words) over its normed rows [R, 4096]. The operands are the
//! loader's: the router weight f16 [512, 4096] as loaded, its transposed view [4096, 512] and the f32 table over that
//! view (`router.tableF32`: astype, + 0), each passed transposed back as the plugin passes it; the predictor's gate the
//! next layer's weight cast to bf16 and its f32 bias; the residual and the mixer's output bf16 [1, R, 4096], with rows at
//! scales from routine to saturating, a row of signed zeros, and columns where the mixer's output is the residual's
//! negation; the norm weight bf16 [4096]; eps 1e-6. Every case compares the residual, the normed rows, the logits and
//! the scores, at rows 1 to 8, over the f32 table and the f16 weight, with no predictor and with each score. Lines:
//! TURBOEXL3_LING_ANROUTE.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const root = @import("../../root.zig");

const testing = std.testing;
const hidden: c_int = 4096;
const eps: f32 = 1e-6;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn adopted(g: *G, rc: c_int, r: mlx.mlx_array) !mlx.mlx_array {
    mlx.check(rc) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
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

/// The words of `got` that differ from `want` (each widened to f32: bf16 and f32 words alike)
fn differ(a: std.mem.Allocator, g: *G, got: mlx.mlx_array, want: mlx.mlx_array) !usize {
    const n: usize = @intCast(mlx.mlx_array_size(want));
    if (@as(usize, @intCast(mlx.mlx_array_size(got))) != n) return n;
    const buf = try a.alloc(f32, 2 * n);
    defer a.free(buf);
    _ = try g.hostF32(try g.astype(try g.reshape(got, &.{ci(n)}), .float32), buf[0..n]);
    _ = try g.hostF32(try g.astype(try g.reshape(want, &.{ci(n)}), .float32), buf[n..]);
    var d: usize = 0;
    for (buf[0..n], buf[n..]) |u, w| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return d;
}

/// x's rows by pattern: scales from routine to saturating, 0 a row of signed zeros
const row_scales = [_]f32{ 1.0, 8.0, 0.0, 0.25, 32.0, 2.0, 64.0, 0.5 };

test "turboexl3 gpu ling anroute: addNormLogits gives the served chain's words (the add_norm's: MLX's add and fast rms_norm; then routerLogits over its normed rows): the residual, the normed rows, the logits and the predictor's scores at rows 1 to 8, over the loader's f32 table and the loaded f16 weight, with no predictor and each score, on rows of every scale, a row of signed zeros and cancelling columns (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    var prng = std.Random.DefaultPrng.init(0xA2_0E_7E);
    const rnd = prng.random();
    // the router weight and the next layer's, f16 [512, 4096] as loaded (normal(0, 0.02) rounded to f16)
    const wwords = try a.alloc(u16, 2 * 512 * 4096);
    defer a.free(wwords);
    for (wwords) |*v| v.* = @bitCast(@as(f16, @floatCast(rnd.floatNorm(f32) * 0.02)));
    const w16 = try g.hostArray(std.mem.sliceAsBytes(wwords[0 .. 512 * 4096]), &.{ 512, 4096 }, .float16);
    const next16 = try g.hostArray(std.mem.sliceAsBytes(wwords[512 * 4096 ..]), &.{ 512, 4096 }, .float16);
    const view = try g.transposeAxes(w16, &.{ 1, 0 });
    const table = try g.adopt(try root.ling.router.tableF32(s, view));
    try g.evalAll(&.{table});
    const wt = [2]mlx.mlx_array{ try g.transposeAxes(table, &.{ 1, 0 }), try g.transposeAxes(view, &.{ 1, 0 }) };
    const pw = try g.astype(next16, .bfloat16);
    try g.evalAll(&.{pw});
    const pb_h = try a.alloc(f32, 512);
    defer a.free(pb_h);
    randFill(rnd, pb_h, 0.01, 0);
    const pb = try g.hostArray(std.mem.sliceAsBytes(pb_h), &.{512}, .float32);
    const nw_h = try a.alloc(f32, 4096);
    defer a.free(nw_h);
    randFill(rnd, nw_h, 0.05, 1.0);
    const nw = try bf16Host(a, &g, nw_h, &.{4096});
    const eps_a = try g.scalar(eps, .float32);
    const preds = [_]?root.ling.router.Predictor{ null, .{ .w = pw, .bias = pb, .score = .softplus_sqrt }, .{ .w = pw, .bias = pb, .score = .sigmoid } };
    var checks: usize = 0;
    var failed: usize = 0;
    for (1..9) |ru| {
        const rows: u32 = @intCast(ru);
        const R = ci(rows);
        const rm = g.mark();
        defer g.resetTo(rm);
        const hx = try a.alloc(f32, 2 * rows * 4096);
        defer a.free(hx);
        for (0..rows) |r| {
            const sc = row_scales[(r + ru) % row_scales.len];
            for (0..4096) |c| {
                const hv: f32 = if (sc == 0) -0.0 else sc * rnd.floatNorm(f32);
                const dv: f32 = if (sc == 0) -0.0 else if (c % 97 == 5) -hv else sc * rnd.floatNorm(f32);
                hx[r * 4096 + c] = hv;
                hx[rows * 4096 + r * 4096 + c] = dv;
            }
        }
        const h = try bf16Host(a, &g, hx[0 .. rows * 4096], &.{ 1, R, hidden });
        const xd = try bf16Host(a, &g, hx[rows * 4096 ..], &.{ 1, R, hidden });
        // the add_norm's words: MLX's add, MLX's fast rms_norm
        const h_ref = try g.add(h, xd);
        var n_ref = mlx.mlx_array_new();
        n_ref = try adopted(&g, mlx.mlx_fast_rms_norm(&n_ref, h_ref, nw, eps, s), n_ref);
        const xn2 = try g.reshape(n_ref, &.{ R, hidden });
        for (wt, 0..) |w, wi| for (preds) |pr| {
            const want = try root.ling.router.routerLogits(s, xn2, w, pr);
            const want_l = try g.adopt(want.logits);
            const want_s: ?mlx.mlx_array = if (want.scores) |sc| try g.adopt(sc) else null;
            const got = try root.ling.addNormLogits(s, try g.reshape(h, &.{ R, hidden }), try g.reshape(xd, &.{ R, hidden }), nw, eps_a, w, pr);
            const got_l = try g.adopt(got.logits);
            const got_s: ?mlx.mlx_array = if (got.scores) |sc| try g.adopt(sc) else null;
            const got_h = try g.adopt(got.h);
            const got_n = try g.adopt(got.normed);
            const dh = try differ(a, &g, got_h, h_ref);
            const dn = try differ(a, &g, got_n, n_ref);
            const dl = try differ(a, &g, got_l, want_l);
            const ds: usize = if (got_s != null and want_s != null) try differ(a, &g, got_s.?, want_s.?) else @intFromBool((got_s == null) != (want_s == null));
            const ok = dh == 0 and dn == 0 and dl == 0 and ds == 0;
            checks += 1;
            failed += @intFromBool(!ok);
            const pname = if (pr) |p| @tagName(p.score) else "none";
            std.debug.print("TURBOEXL3_LING_ANROUTE rows={d} weight={s} predictor={s} residual_differing={d} normed_differing={d} logits_differing={d} scores_differing={d} {s}\n", .{ rows, if (wi == 0) "f32_table" else "f16", pname, dh, dn, dl, ds, if (ok) "PASS" else "FAIL" });
        };
    }
    std.debug.print("TURBOEXL3_LING_ANROUTE_DONE checks={d} failed={d} {s}\n", .{ checks, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 0), failed);
}
