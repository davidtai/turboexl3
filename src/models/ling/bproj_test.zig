//! `kdaProjB` (the outside plugin turboexl3_ling's MLX_SERVE_LING_KDA_BPROJ): a KDA layer's b_proj (bf16, 4096 -> 64)
//! as a fourth member of kdaProj's GEMV launch, held word for word to the launches it replaces.
//! - CPU (always): turboexl3_ling_kdaproj_dec0b's text is `kdaproj/bmember` before dec0's text, nothing else; the member
//!   names its operands and template arguments and none of MLX's shape, strides or ndim buffers; its MLX gemv statements
//!   (pins in kernels/test/pins/bproj_texts.metal) appear once each.
//! - GPU (TURBOEXL3_GPU=1, else skipped), at Ling-3.1-flash's KDA shapes (q|k|v 4096 -> 24576, f and g 4096 -> 8192 at
//!   K 6, the loader's I16 trellis words; b_proj 4096 -> 64): b's weight as the plugin's loader keeps it (the pack's f16
//!   [64, 4096] viewed [4096, 64] and cast to bf16, its column-major layout kept) and its transpose back for the
//!   member; x the layer's normed input [1, S, 4096] bf16. At rows 1 to 8 (serial decode's one row and an exact verify's
//!   rows), b's output against the served b_proj (`m.mm` at one row: MLX's matmul; at 2 to 8 the exact verify's
//!   denseRows: each row stored twice, the first viewed, a batched matmul), and the three members' outputs against
//!   `kdaProj`'s, each word for word; and the refusals. Lines: TURBOEXL3_LING_KDA_BPROJ.
//! - GPU, D35: the variant launch (`kdaproj.variantFor`: b first, the members' tiles apart, DEC-ONE's GEMV variants)
//!   against the served launch, its four outputs word for word, with red checks. Lines: TURBOEXL3_LING_KDA_VARIANT.
//! - TURBOEXL3_LING_TIME (from ling_test.zig's timing test, guidance only): `timeKdaVariants`.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const T = mlx.mlx_array;
const format = @import("../../core/format.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const kdaproj = @import("kdaproj.zig");

const testing = std.testing;

const in: u32 = 4096;
const outs = [3]u32{ 24576, 8192, 8192 };
const out_b: u32 = 64;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

test "turboexl3 ling bproj texts: dec0b is the b_proj member before dec0's text, nothing else; the member names its operands and arguments, no shape buffers, and MLX's gemv statements once each" {
    const pin = comptime support.pins(@embedFile("kernels/test/pins/bproj_texts.metal"));
    try testing.expectEqualStrings(kdaproj.kbmember ++ kdaproj.text_kdec0, kdaproj.text_kdec0b);
    for ([_][]const u8{ "xb", "wb", "yb", "FB", "OUTB", "IN", "RB", "ROWS" }) |n| try testing.expect(count(kdaproj.kbmember, n) > 0);
    for ([_][]const u8{ "xb", "wb", "yb" }) |n| for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| {
        var buf: [24]u8 = undefined;
        try testing.expectEqual(@as(usize, 0), count(kdaproj.text_kdec0b, std.fmt.bufPrint(&buf, "{s}{s}", .{ n, suf }) catch unreachable));
    };
    for (pin) |p| try testing.expectEqual(@as(usize, 1), count(kdaproj.kbmember, p));
}

/// A synthetic projection and the operands `linear` takes for it
const Served = struct { lin: ref.Linear, p: root.Proj };

/// A projection `linear` takes at (in, out, n): its synthetic trellis as the loader hands it (the pack's I16 words)
fn servedProj(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, rnd: std.Random, o: u32) !Served {
    var lin = try ref.synthLinear(a, rnd, 1, in, o, format.Rate.fromN(96).?, .mul1);
    errdefer lin.deinit(a);
    const u = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(in / 16), ci(o / 16), 96 }, .uint16);
    var v = mlx.mlx_array_new();
    mlx.check(mlx.mlx_view(&v, u, .int16, s)) catch |e| {
        _ = mlx.mlx_array_free(v);
        return e;
    };
    return .{ .lin = lin, .p = .{
        .trellis = try g.adopt(v),
        .suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(o)}, .float16),
    } };
}

/// MLX's expand_dims (denseRows): the consumer's binding where its `sdk.mlx` declares it, else libmlxc's own symbol
const expandDims = if (@hasDecl(mlx, "mlx_expand_dims")) mlx.mlx_expand_dims else struct {
    extern "c" fn mlx_expand_dims(res: *mlx.mlx_array, a: mlx.mlx_array, axis: c_int, s: mlx.mlx_stream) c_int;
}.mlx_expand_dims;

/// The served b_proj of x [1, S, in] bf16 over the loaded weight w [in, out] (column-major): at one row `m.mm`'s
/// matmul; at 2 to 8 rows the exact verify's denseRows (x as [S, 1, in], each row stored twice and the first viewed, w
/// as [1, in, out], one batched matmul), reshaped [1, S, out]
fn servedB(g: *G, s: mlx.mlx_stream, x: T, w: T, rows: u32) !T {
    if (rows == 1) return g.matmul(x, w);
    const x3 = try g.reshape(x, &.{ ci(rows), 1, ci(in) });
    const twice = try g.concat(&.{ x3, x3 }, 1);
    const view = try g.slice(twice, &.{ 0, 0, 0 }, &.{ ci(rows), 1, ci(in) }, &.{ 1, 1, 1 });
    var w3 = mlx.mlx_array_new();
    mlx.check(expandDims(&w3, w, 0, s)) catch |e| {
        _ = mlx.mlx_array_free(w3);
        return e;
    };
    return g.reshape(try g.matmul(view, try g.adopt(w3)), &.{ 1, ci(rows), ci(out_b) });
}

/// Words of `got` and `want` (one size; compared as f32) that differ
fn differ(a: std.mem.Allocator, g: *G, got: T, want: T) !usize {
    const n = mlx.mlx_array_size(want);
    if (mlx.mlx_array_size(got) != n) return error.SizeMismatch;
    const p = try a.alloc(f32, n);
    defer a.free(p);
    const q = try a.alloc(f32, n);
    defer a.free(q);
    _ = try g.hostF32(try g.astype(got, .float32), p);
    _ = try g.hostF32(try g.astype(want, .float32), q);
    var d: usize = 0;
    for (p, q) |u, w| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return d;
}

test "turboexl3 gpu ling bproj: kdaProjB's b_proj member is the served b_proj (one-row matmul, the exact verify's denseRows) and its three members kdaProj's, word for word at rows 1 to 8, operands as the loader keeps them (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xB9A0E1);
    const rnd = prng.random();
    var ms: [3]Served = undefined;
    var made: usize = 0;
    defer for (ms[0..made]) |*m| m.lin.deinit(a);
    for (&ms, outs) |*m, o| {
        m.* = try servedProj(a, &g, s, rnd, o);
        made += 1;
    }
    const P = [3]root.Proj{ ms[0].p, ms[1].p, ms[2].p };
    // b's weight as the loader keeps it: the pack's f16 [out, in], viewed [in, out], cast to bf16 (its layout kept)
    const wraw = try a.alloc(f16, out_b * in);
    defer a.free(wraw);
    for (wraw) |*v| v.* = @floatCast(rnd.floatNorm(f32) / 32.0);
    const w16 = try g.hostArray(std.mem.sliceAsBytes(wraw), &.{ ci(out_b), ci(in) }, .float16);
    const w_served = try g.astype(try g.transposeAxes(w16, &.{ 1, 0 }), .bfloat16);
    try g.evalAll(&.{w_served});
    // the member's: the served weight transposed back, [out, in] row-major
    const wb = try g.transposeAxes(w_served, &.{ 1, 0 });
    const sh = try kdaproj.shapeOf(P);
    try testing.expectEqual(@as(?u32, out_b), kdaproj.bTakes(sh, .bfloat16, wb));
    const xs = try a.alloc(f32, 8 * in);
    defer a.free(xs);
    for (xs) |*v| v.* = rnd.floatNorm(f32);
    var failed: usize = 0;
    for (1..9) |ru| {
        const rows: u32 = @intCast(ru);
        const mark = g.mark();
        defer g.resetTo(mark);
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. rows * in]), &.{ 1, ci(rows), ci(in) }, .float32), .bfloat16);
        const b_want = try servedB(&g, s, x, w_served, rows);
        const ys = try root.ling.kdaProj(s, x, P, .mul1);
        var want: [3]T = undefined;
        for (ys, &want) |y, *w| w.* = try g.adopt(y);
        const got = try root.ling.kdaProjB(s, x, P, wb, .mul1);
        var gotx: [4]T = undefined;
        for (got, &gotx) |y, *w| w.* = try g.adopt(y);
        try testing.expectEqualSlices(c_int, &.{ ci(rows), ci(out_b) }, mlx.getShape(gotx[3]));
        const d_b = try differ(a, &g, gotx[3], b_want);
        var d_m: usize = 0;
        for (gotx[0..3], want) |u, w| d_m += try differ(a, &g, u, w);
        const n_m = rows * (outs[0] + outs[1] + outs[2]);
        const ok = d_b == 0 and d_m == 0;
        std.debug.print("TURBOEXL3_LING_KDA_BPROJ rows={d} b_differing={d}/{d} members_differing={d}/{d} served_b={s} {s}\n", .{ rows, d_b, rows * out_b, d_m, n_m, if (rows == 1) "matmul" else "denseRows", if (ok) "PASS" else "FAIL" });
        if (!ok) failed += 1;
    }
    // the refusals, before any launch: an f32 x, a weight not [out, in] bf16, a width MLX's gemv takes otherwise
    {
        const x32 = try g.zeros(&.{ 1, ci(in) }, .float32);
        try testing.expectError(error.Exl3EngineNotImplemented, root.ling.kdaProjB(s, x32, P, wb, .mul1));
        const xb = try g.zeros(&.{ 1, ci(in) }, .bfloat16);
        try testing.expectError(error.Exl3EngineNotImplemented, root.ling.kdaProjB(s, xb, P, w_served, .mul1));
        const wide = try g.zeros(&.{ 512, ci(in) }, .bfloat16);
        try testing.expectError(error.Exl3EngineNotImplemented, root.ling.kdaProjB(s, xb, P, wide, .mul1));
        try testing.expectError(error.Exl3EngineNotImplemented, root.ling.kdaProjB(s, xb, P, try g.astype(wb, .float16), .mul1));
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── D35: the variant launch (`kdaproj.variantFor`: Config.kda_bfirst, kda_bt0 / kda_bt1, dec_variant) ──

const config = @import("../../core/config.zig");
const dense = @import("../../core/dense.zig");
const ds_test = @import("../deepseek_v41/deepseek_test.zig");

/// The variant launch's arms (each a configuration `kdaProjBWith` takes): b first; q|k|v's (bt0) and f and g's (bt1)
/// output tiles a threadgroup apart at a verify's rows; DEC-ONE's GEMV variants (t4, pf, pft4); all together
pub const VariantArm = struct { name: []const u8, c: config.Config };
pub const variant_arms = [_]VariantArm{
    .{ .name = "bfirst", .c = .{ .kda_bfirst = true } },
    .{ .name = "bt1_2", .c = .{ .kda_bt0 = 1, .kda_bt1 = 2 } },
    .{ .name = "bt2_1", .c = .{ .kda_bt0 = 2, .kda_bt1 = 1 } },
    .{ .name = "bt1_1", .c = .{ .kda_bt0 = 1, .kda_bt1 = 1 } },
    .{ .name = "bt2_4", .c = .{ .kda_bt0 = 2, .kda_bt1 = 4 } },
    .{ .name = "bt4_2", .c = .{ .kda_bt0 = 4, .kda_bt1 = 2 } },
    .{ .name = "t4", .c = .{ .dec_variant = 1 } },
    .{ .name = "pf", .c = .{ .dec_variant = 2 } },
    .{ .name = "pft4", .c = .{ .dec_variant = 3 } },
    .{ .name = "all", .c = .{ .kda_bfirst = true, .kda_bt0 = 1, .kda_bt1 = 2, .dec_variant = 3 } },
};

/// The words of `p` that differ from `q` (f32 bit patterns)
fn wordsApart(p: []const f32, q: []const f32) usize {
    var d: usize = 0;
    for (p, q) |u, w| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return d;
}

/// b's weight from its f16 words as the loader keeps it (the pack's [out, in], viewed [in, out] and cast to bf16, its
/// column-major layout kept), and the member's operand: that view transposed back, [out, in] row-major
fn bWeights(g: *G, words: []const f16) ![2]T {
    const w16 = try g.hostArray(std.mem.sliceAsBytes(words), &.{ ci(out_b), ci(in) }, .float16);
    const w_served = try g.astype(try g.transposeAxes(w16, &.{ 1, 0 }), .bfloat16);
    try g.evalAll(&.{w_served});
    return .{ w_served, try g.transposeAxes(w_served, &.{ 1, 0 }) };
}

// Each variant arm's four outputs against the served launch's (`kdaProjBWith` at the default configuration, which the
// test above holds to the served b_proj and to kdaProj), word for word, at rows 1 to 8 and 16 (serial decode's row, an
// exact verify's rows, DEC-ONE's widest), x bf16 [1, S, 4096], the members' trellises as the loader's I16 words and b's
// weight as the loader keeps it, transposed back. Each arm checks that the variant launch ran where `variantFor` asks one
// (the tiles apart act at 2 rows or more only) and not elsewhere. Red checks at 3 rows: a flipped output word is exactly
// one differing word; each arm with q|k|v at another K split (2, the same products summed in another order) must differ
// in q|k|v's words and in no other member's or b's; with one word of b's weight negated, the b-first, t4 and all arms
// must differ in b's words and in no member's. Lines: TURBOEXL3_LING_KDA_VARIANT (_RED, _DONE).
test "turboexl3 gpu ling bproj variants: each variant launch (b first; q|k|v's and f and g's tiles apart; t4, pf, pft4; all together) gives kdaProjB's words, b's and the three members', at rows 1 to 8 and 16, operands as the loader keeps them; red checks: a flipped word, q|k|v at another K split, one word of b's weight (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xD35B_0E1);
    const rnd = prng.random();
    var ms: [3]Served = undefined;
    var made: usize = 0;
    defer for (ms[0..made]) |*m| m.lin.deinit(a);
    for (&ms, outs) |*m, o| {
        m.* = try servedProj(a, &g, s, rnd, o);
        made += 1;
    }
    const P = [3]root.Proj{ ms[0].p, ms[1].p, ms[2].p };
    const sh = try kdaproj.shapeOf(P);
    const wraw = try a.alloc(f16, out_b * in);
    defer a.free(wraw);
    for (wraw) |*v| v.* = @floatCast(rnd.floatNorm(f32) / 32.0);
    const wbs = try bWeights(&g, wraw);
    const wb = wbs[1];
    const xs = try a.alloc(f32, 16 * in);
    defer a.free(xs);
    for (xs) |*v| v.* = rnd.floatNorm(f32);
    const served_cfg: config.Config = .{};
    const widths = [4]u32{ outs[0], outs[1], outs[2], out_b };
    const names = [4][]const u8{ "qkv", "f", "g", "b" };
    // host copies of the served words and an arm's, per output
    var want: [4][]f32 = undefined;
    var got: [4][]f32 = undefined;
    for (&want, &got, widths) |*w, *gt, wd| {
        w.* = try a.alloc(f32, 16 * wd);
        gt.* = try a.alloc(f32, 16 * wd);
    }
    defer for (want, got) |w, gt| {
        a.free(w);
        a.free(gt);
    };
    var failed: usize = 0;
    var checks: usize = 0;
    var reds: usize = 0;
    for ([_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 16 }) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. rows * in]), &.{ 1, ci(rows), ci(in) }, .float32), .bfloat16);
        const pl = (try kdaproj.plan(sh, .mul1, rows, .bfloat16)).?;
        const base = try kdaproj.kdaProjBWith(s, x, P, wb, .mul1, served_cfg);
        for (base, want, widths) |y, w, wd| _ = try g.hostF32(try g.astype(try g.adopt(y), .float32), w[0 .. rows * wd]);
        for (variant_arms) |arm| {
            const vv = kdaproj.variantFor(arm.c, pl, rows);
            const runs0 = kdaproj.variantRuns();
            const ys = try kdaproj.kdaProjBWith(s, x, P, wb, .mul1, arm.c);
            var d: [4]usize = undefined;
            for (ys, got, want, widths, &d) |y, gt, w, wd, *dd| {
                _ = try g.hostF32(try g.astype(try g.adopt(y), .float32), gt[0 .. rows * wd]);
                dd.* = wordsApart(gt[0 .. rows * wd], w[0 .. rows * wd]);
            }
            const ran = kdaproj.variantRuns() - runs0;
            const ok = d[0] == 0 and d[1] == 0 and d[2] == 0 and d[3] == 0 and ran == @intFromBool(vv != null);
            checks += 1;
            std.debug.print("TURBOEXL3_LING_KDA_VARIANT arm={s} rows={d} launch={s} bfirst={s} bt0={d} bt1={d} gemv={s} qkv_differing={d}/{d} f_differing={d}/{d} g_differing={d}/{d} b_differing={d}/{d} ran={d} {s}\n", .{ arm.name, rows, if (vv != null) "variant" else "served", if (vv) |v| (if (v.bfirst) "on" else "off") else "-", if (vv) |v| v.bt0 else pl.bt, if (vv) |v| v.bt1 else pl.bt, if (vv) |v| (if (v.dv == 0) "dec_one" else dense.dec_variants[v.dv].name) else "-", d[0], rows * widths[0], d[1], rows * widths[1], d[2], rows * widths[2], d[3], rows * widths[3], ran, if (ok) "PASS" else "FAIL" });
            if (!ok) failed += 1;
        }
        if (rows != 3) continue;
        // red: one flipped word of the served output is one differing word
        {
            const n = rows * widths[0];
            @memcpy(got[0][0..n], want[0][0..n]);
            got[0][n / 2] = @bitCast(@as(u32, @bitCast(got[0][n / 2])) ^ 1);
            const d_flip = wordsApart(got[0][0..n], want[0][0..n]);
            reds += 1;
            std.debug.print("TURBOEXL3_LING_KDA_VARIANT_RED check=flipped_word rows={d} differing={d}/{d} expect=1 {s}\n", .{ rows, d_flip, n, if (d_flip == 1) "PASS" else "FAIL" });
            if (d_flip != 1) failed += 1;
        }
        // red: q|k|v at another K split (its cached Dense's, which plan reads), each arm: q|k|v's words differ, no other's
        {
            const d0 = try root.denseCached(.{ .in = in, .out = outs[0], .rate = format.Rate.fromN(96).?, .dec = .mul1, .out_dtype = .bfloat16 });
            const ks_served = d0.ks_split;
            d0.ks_split = 2;
            defer {
                d0.ks_split = ks_served;
                d0.resetCfgs();
            }
            for (variant_arms) |arm| {
                const ys = try kdaproj.kdaProjBWith(s, x, P, wb, .mul1, arm.c);
                var d: [4]usize = undefined;
                for (ys, got, want, widths, &d) |y, gt, w, wd, *dd| {
                    _ = try g.hostF32(try g.astype(try g.adopt(y), .float32), gt[0 .. rows * wd]);
                    dd.* = wordsApart(gt[0 .. rows * wd], w[0 .. rows * wd]);
                }
                const ok = ks_served != 2 and d[0] > 0 and d[1] == 0 and d[2] == 0 and d[3] == 0;
                reds += 1;
                std.debug.print("TURBOEXL3_LING_KDA_VARIANT_RED check=other_ks arm={s} rows={d} qkv_ks={d}->2 qkv_differing={d}/{d} others_differing={d} expect=qkv_nonzero_others_0 {s}\n", .{ arm.name, rows, ks_served, d[0], rows * widths[0], d[1] + d[2] + d[3], if (ok) "PASS" else "FAIL" });
                if (!ok) failed += 1;
            }
        }
        // red: one word of b's weight negated (the largest of output row 5), the arms whose launch runs b's statements under
        // its own choice: b's words differ, no member's
        {
            const w2 = try a.dupe(f16, wraw);
            defer a.free(w2);
            const row5 = w2[5 * @as(usize, in) ..][0..in];
            var kmax: usize = 0;
            for (row5, 0..) |wv, k| if (@abs(wv) > @abs(row5[kmax])) {
                kmax = k;
            };
            row5[kmax] = -row5[kmax];
            const wbs2 = try bWeights(&g, w2);
            for ([_][]const u8{ "bfirst", "t4", "all" }) |nm| {
                const arm = for (variant_arms) |va| {
                    if (std.mem.eql(u8, va.name, nm)) break va;
                } else unreachable;
                const ys = try kdaproj.kdaProjBWith(s, x, P, wbs2[1], .mul1, arm.c);
                var d: [4]usize = undefined;
                for (ys, got, want, widths, &d) |y, gt, w, wd, *dd| {
                    _ = try g.hostF32(try g.astype(try g.adopt(y), .float32), gt[0 .. rows * wd]);
                    dd.* = wordsApart(gt[0 .. rows * wd], w[0 .. rows * wd]);
                }
                const ok = d[3] > 0 and d[0] == 0 and d[1] == 0 and d[2] == 0;
                reds += 1;
                std.debug.print("TURBOEXL3_LING_KDA_VARIANT_RED check=b_weight_word arm={s} rows={d} b_differing={d}/{d} members_differing={d} expect=b_nonzero_members_0 {s}\n", .{ arm.name, rows, d[3], rows * widths[3], d[0] + d[1] + d[2], if (ok) "PASS" else "FAIL" });
                if (!ok) failed += 1;
            }
        }
    }
    for (names, widths) |nm, wd| std.debug.print("TURBOEXL3_LING_KDA_VARIANT output {s} width={d}\n", .{ nm, wd });
    std.debug.print("TURBOEXL3_LING_KDA_VARIANT_DONE checks={d} red={d} {s}\n", .{ checks, reds, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 0), failed);
}

/// TURBOEXL3_LING_TIME's arm for the KDA projections' variant launch (`kdaproj.variantFor`), guidance only: `kdaProjB`
/// (`kdaProjBWith`) at rows 1 to 4 over random K6 members (q|k|v 4096 -> 24576, f and g 4096 -> 8192) and a bf16 b_proj
/// [64, 4096], the served launch and each of `variant_arms`: min and median us a call (8 calls an evaluation over the
/// same weights), the weights' GB/s (the three trellises and b's words), the arm's min over the served launch's. One row
/// at the one-row table's tiles for q|k|v (the plugin's ONE_ROW_TABLE). Its own draw. Lines: TURBOEXL3_LING_TIME
/// kda_variant.
pub fn timeKdaVariants(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream) !void {
    var prng = std.Random.DefaultPrng.init(0xD35_7DA0);
    const rnd = prng.random();
    const mark = g.mark();
    defer g.resetTo(mark);
    defer _ = mlx.mlx_clear_cache();
    var P: [3]root.Proj = undefined;
    for (&P, outs) |*p, o| p.* = try ds_test.randProj(a, g, rnd, null, in, o, 96);
    const wb = try ds_test.randBf16(a, g, rnd, &.{ ci(out_b), ci(in) });
    // one row at the outside plugin's tiles (ONE_ROW_TABLE: member 0's one-row table entry, which the plan reads)
    const qkv_shape: dense.Shape = .{ .in = in, .out = outs[0], .rate = format.Rate.fromN(96).?, .dec = .mul1, .out_dtype = .bfloat16 };
    const d0 = try root.denseCached(qkv_shape);
    const bt1_served = d0.bt_set[1];
    defer {
        d0.bt_set[1] = bt1_served;
        d0.resetCfgs();
    }
    if (dense.oneRowFor(qkv_shape)) |e| {
        d0.bt_set[1] = e.bt;
        d0.resetCfgs();
    }
    const wbytes = @as(f64, @floatFromInt(in)) * @as(f64, @floatFromInt(outs[0] + outs[1] + outs[2])) * 96.0 / 128.0 + 2.0 * @as(f64, @floatFromInt(out_b * in));
    const Ctx = struct {
        s: mlx.mlx_stream,
        x: T,
        p: [3]root.Proj,
        wb: T,
        c: config.Config,
        pub fn call(cx: @This()) !T {
            const o = try kdaproj.kdaProjBWith(cx.s, cx.x, cx.p, cx.wb, .mul1, cx.c);
            for (o[1..]) |y| _ = mlx.mlx_array_free(y);
            return o[0];
        }
    };
    for ([_]u32{ 1, 2, 3, 4 }) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        const x = try ds_test.randBf16(a, g, rnd, &.{ 1, ci(rows), ci(in) });
        const served = try ds_test.timeIt(g, Ctx{ .s = s, .x = x, .p = P, .wb = wb, .c = .{} }, 8);
        std.debug.print("TURBOEXL3_LING_TIME kda_variant rows={d} arm=served min_us={d:.1} med_us={d:.1} GBps={d:.0} of_served=1.000\n", .{ rows, served.min, served.med, wbytes / (served.min * 1e3) });
        for (variant_arms) |arm| {
            const st = try ds_test.timeIt(g, Ctx{ .s = s, .x = x, .p = P, .wb = wb, .c = arm.c }, 8);
            std.debug.print("TURBOEXL3_LING_TIME kda_variant rows={d} arm={s} min_us={d:.1} med_us={d:.1} GBps={d:.0} of_served={d:.3}\n", .{ rows, arm.name, st.min, st.med, wbytes / (st.min * 1e3), st.min / served.min });
        }
    }
}
