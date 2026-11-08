//! P4: a KDA layer's q|k|v, f and g at prompt rows (`kdaProjPrompt`, the outside plugin turboexl3_ling's
//! MLX_SERVE_LING_KDA_PROJ_PROMPT): three EXL3 projections of one input, each member's prepare and W_hat by its own
//! `linear`'s launches (`dense.Dense.prefillOperands`), then their three products and finishes in ONE launch,
//! `turboexl3_ling_kdaproj_pf`: pg2's fused NAX GEMM with the dense finish as its epilogue (core/pgemm.zig), the grid's
//! N tiles running member 0's, 1's and 2's. `linear` at these rows is three launches past the operands (MLX's matmul,
//! z half written, and the finish reading it back); here z never reaches device memory, and the three members share a
//! launch.
//! Member m's y is its own `linear`'s bytes: MLX's NAX loop keeps each column's k order whatever the tile (pg2's
//! claim, held by its probe over every shape, tile and row count), and pg2's epilogue is the finish's statements. A call
//! is taken only where `linear` runs that chain at these rows and MLX runs its regular NAX kernel (no split K); else
//! error.Exl3EngineNotImplemented, nothing launched.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const config = @import("../../core/config.zig");
const format = @import("../../core/format.zig");
const pgemm = @import("../../core/pgemm.zig");
const root = @import("../../root.zig");

const A = mlx.mlx_array;

// ── texts ──

/// the members' shapes and the launch's N tiles
pub const kpf_1 = @embedFile("kernels/kdaproj/pf_1.metal");
/// the tile's member: its width, its first tile, its operands and its output
pub const kpf_3 = @embedFile("kernels/kdaproj/pf_3.metal");
/// the tile's column within its member
pub const kpf_5 = @embedFile("kernels/kdaproj/pf_5.metal");
/// the header's `kernel` macro undefined after the body (as pg2's body ends)
pub const kpf_undef = @embedFile("kernels/kdaproj/pf_undef.metal");

/// pg2's body with the member choice in place of its single operand pair
pub const text_pf: [:0]const u8 = kpf_1 ++ pgemm.fused_2 ++ kpf_3 ++ pgemm.fused_4 ++ kpf_5 ++ pgemm.fused_6 ++ kpf_undef;

const pf_in = [_][*:0]const u8{ "x0", "x1", "x2", "w0", "w1", "w2", "svh0", "svh1", "svh2" };
const pf_out = [_][*:0]const u8{ "y0", "y1", "y2" };

// ── the kernel: made at its first call ──

const kernel_lock = Global(std.atomic.Value(u32), "ling_kdaproj_pf_kernel_lock", std.atomic.Value(u32).init(0));
const kernel_cache = Global(?mlx.mlx_fast_metal_kernel, "ling_kdaproj_pf_kernel", null);

fn kernelGet() !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    if (kernel_cache.get().*) |k| return k;
    const k = ks.newKernel("turboexl3_ling_kdaproj_pf", &pf_in, &pf_out, text_pf, pgemm.header, true) orelse return error.KernelCreateFailed;
    kernel_cache.get().* = k;
    return k;
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernel_cache };

// ── the tile ──

/// The launch's tile at a K: this machine's (M5 Max) choice for Ling-3.1-flash's KDA projections (K 4096; q|k|v 24576,
/// f and g 8192 wide), from the TURBOEXL3_LING_TIME arm in kdaproj_pf_test.zig; null at any other K
pub fn tileFor(k: u32) ?pgemm.Tile {
    if (k != 4096) return null;
    return tile_override.get().* orelse ling_tile;
}

/// The tile the TIME arm picked (pg2's shape for its K 5120 and 8192 shapes until it says otherwise)
pub const ling_tile: pgemm.Tile = .{ .bm = 32, .bn = 256, .bk = 512, .wm = 1, .wn = 4, .swz = 2 };

/// Test seam: the TIME arm's sweep sets the tile a call runs at (null: `ling_tile`)
pub const tile_override = Global(?pgemm.Tile, "ling_kdaproj_pf_tile", null);

// ── the call ──

/// The rows `linear` runs on its register-path GEMM on NAX (dense's `small_rows`): the grouped launch starts past them
const register_rows: usize = 32;

/// Whether `kdaProjPrompt` runs x [..., in] through `p` at `rows` rows, its tile `t`: NAX, three EXL3 members of one
/// input width and rate with x's dtype a float, every width a multiple of the tile's N, rows past the GEMV and register
/// paths (each member's `prefillOperands` has the last word), and MLX's regular kernel at every member's (rows, out, in)
pub fn takes(p: [3]root.Proj, rows: usize, xd: mlx.mlx_dtype, t: pgemm.Tile) bool {
    if (!config.get().nax or !t.admitted()) return false;
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return false;
    if (rows <= register_rows) return false;
    const in0 = mlx.getShape(p[0].trellis);
    if (in0.len != 3) return false;
    for (p) |m| {
        const sh = mlx.getShape(m.trellis);
        if (sh.len != 3 or sh[0] != in0[0] or sh[2] != in0[2]) return false;
        const out: u32 = @intCast(sh[1] * 16);
        if (out % t.bn != 0) return false;
        if (!pgemm.mlxRegular(@intCast(rows), out, @intCast(in0[0] * 16))) return false;
    }
    return true;
}

/// y_m [..., out_m] in x's dtype for x [..., in] through member m of `p` (a KDA layer's q|k|v, f and g: one input, three
/// EXL3 projections), at prompt rows: each `linear(x, p[m])`'s bytes. Owned results. error.Exl3EngineNotImplemented
/// where it does not take the call (`takes`, or a member's `linear` taking another arm at these rows), nothing launched.
pub fn kdaProjPrompt(s: mlx.mlx_stream, x: A, p: [3]root.Proj, dec: format.Decode) ![3]A {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8) return error.BadExl3Shape;
    var rows: usize = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows *= @as(usize, @intCast(d));
    const xd = mlx.mlx_array_dtype(x);
    const tsh0 = mlx.getShape(p[0].trellis);
    if (tsh0.len != 3) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(tsh0[0] * 16);
    if (xsh[xsh.len - 1] != @as(c_int, @intCast(in_dim))) return error.BadExl3Shape;
    const t = tileFor(in_dim) orelse return error.Exl3EngineNotImplemented;
    if (!takes(p, rows, xd, t)) return error.Exl3EngineNotImplemented;
    const rate = format.Rate.fromN(@intCast(tsh0[2])) orelse return error.BadExl3Shape;
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ @intCast(rows), @intCast(in_dim) }, 2, s));
    // each member's prepared rows and W_hat, by its own `linear`'s launches
    var ops: [3][2]A = undefined;
    var made: usize = 0;
    defer for (ops[0..made]) |o| for (o) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var outs: [3]u32 = undefined;
    for (p, 0..) |m, i| {
        const sh = mlx.getShape(m.trellis);
        outs[i] = @intCast(sh[1] * 16);
        const d = try root.denseCached(.{ .in = in_dim, .out = outs[i], .rate = rate, .dec = dec, .out_dtype = xd });
        ops[i] = (try d.prefillOperands(s, x2, m.trellis, m.suh, rows)) orelse return error.Exl3EngineNotImplemented;
        made += 1;
    }
    const ys = try launch(s, .{ ops[0][0], ops[1][0], ops[2][0] }, .{ ops[0][1], ops[1][1], ops[2][1] }, .{ p[0].svh, p[1].svh, p[2].svh }, outs, @intCast(rows), @intCast(in_dim), xd, t);
    defer for (ys) |y| {
        _ = mlx.mlx_array_free(y);
    };
    var shape: [8]c_int = undefined;
    for (xsh, 0..) |dd, i| shape[i] = dd;
    var res: [3]A = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (res) |r| {
        _ = mlx.mlx_array_free(r);
    };
    for (&res, ys, outs) |*r, y, o| {
        shape[xsh.len - 1] = @intCast(o);
        try mlx.check(mlx.mlx_reshape(r, y, &shape, xsh.len, s));
    }
    return res;
}

/// The grouped launch: y_m [M, out_m] of `out_dtype` = finish(xp_m [M, K] w_m [K, out_m]) at tile `t`, pg2's grid over
/// the members' N tiles end to end (tiles_n << swz by ceil(tiles_m >> swz) threadgroups) in MLX's threadgroup shape
/// (32, WN, WM); the caller owns the three y
fn launch(s: mlx.mlx_stream, xp: [3]A, w: [3]A, svh: [3]A, outs: [3]u32, m: u32, kd: u32, out_dtype: mlx.mlx_dtype, t: pgemm.Tile) ![3]A {
    const kern = try kernelGet();
    const tile: u32 = @as(u32, 1) << @intCast(t.swz);
    const tiles_m = (m + t.bm - 1) / t.bm;
    const tiles_n = (outs[0] + outs[1] + outs[2]) / t.bn;
    const cfg = try ks.buildConfig(.{
        .grid = .{ tiles_n * tile * 32, ((tiles_m + tile - 1) / tile) * t.wn, t.wm },
        .threadgroup = .{ 32, t.wn, t.wm },
        .out_shapes = &.{ &[_]c_int{ @intCast(m), @intCast(outs[0]) }, &[_]c_int{ @intCast(m), @intCast(outs[1]) }, &[_]c_int{ @intCast(m), @intCast(outs[2]) } },
        .out_dtypes = &.{ out_dtype, out_dtype, out_dtype },
        .template = &.{
            .{ .name = "T", .value = .{ .dtype = .float16 } },
            .{ .name = "BM", .value = .{ .int = t.bm } },
            .{ .name = "BN", .value = .{ .int = t.bn } },
            .{ .name = "BK", .value = .{ .int = t.bk } },
            .{ .name = "WM", .value = .{ .int = t.wm } },
            .{ .name = "WN", .value = .{ .int = t.wn } },
            .{ .name = "SWZ", .value = .{ .int = t.swz } },
            .{ .name = "AM", .value = .{ .int = @intFromBool(m % t.bm == 0) } },
            .{ .name = "AN", .value = .{ .int = 1 } },
            .{ .name = "AK", .value = .{ .int = @intFromBool(kd % t.bk == 0) } },
            .{ .name = "OUT_T", .value = .{ .dtype = out_dtype } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [3]A = undefined;
    try ks.applyConfig(kern, cfg, &.{ xp[0], xp[1], xp[2], w[0], w[1], w[2], svh[0], svh[1], svh[2] }, &o, s);
    return o;
}

// ── CPU tests: the texts ──

const testing = std.testing;

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

test "turboexl3 ling kdaproj prompt texts: pg2's body is its six pieces; the grouped body is pg2's pieces 2, 4 and 6 with the member choice, its column and the macro's end around them" {
    try testing.expectEqualStrings(pgemm.fused_1 ++ pgemm.fused_2 ++ pgemm.fused_3 ++ pgemm.fused_4 ++ pgemm.fused_5 ++ pgemm.fused_6, pgemm.text_fused);
    try testing.expectEqualStrings(kpf_1 ++ pgemm.fused_2 ++ kpf_3 ++ pgemm.fused_4 ++ kpf_5 ++ pgemm.fused_6 ++ kpf_undef, text_pf);
    // the grouped body names its members' operands and outputs, never pg2's single pair, and reads no strides
    for ([_][]const u8{ "x0", "x1", "x2", "w0", "w1", "w2", "svh0", "svh1", "svh2", "y0", "y1", "y2" }) |n| try testing.expect(count(text_pf, n) > 0);
    for ([_][]const u8{ "x_shape", "w_shape", "_strides" }) |n| try testing.expectEqual(@as(usize, 0), count(text_pf, n));
}

test "turboexl3 ling kdaproj prompt: the tile is admitted, divides Ling's KDA widths, and the shapes are MLX's regular kernel at prompt rows" {
    try testing.expect(ling_tile.admitted());
    for ([_]u32{ 24576, 8192 }) |o| try testing.expectEqual(@as(u32, 0), o % ling_tile.bn);
    for ([_]u32{ 33, 512, 2630, 4096, 16384 }) |rows| for ([_]u32{ 24576, 8192 }) |o| try testing.expect(pgemm.mlxRegular(rows, o, 4096));
    try testing.expect(tileFor(4096) != null);
    try testing.expect(tileFor(8192) == null);
}
