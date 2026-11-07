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
