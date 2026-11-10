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
/// the tile's member: its width, its first tile and its operands, then its output (cut where the KDA gate's launches
/// name their own outputs: `kpf_3_2g`)
pub const kpf_3_1 = @embedFile("kernels/kdaproj/pf_3_1.metal");
pub const kpf_3_2 = @embedFile("kernels/kdaproj/pf_3_2.metal");
pub const kpf_3 = kpf_3_1 ++ kpf_3_2;
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

/// Whether member `dm`'s `prefillOperands` hands out its operands at `rows` rows for `trellis` (no launch): past the
/// GEMV's and the register path's rows, no fused-GEMM option, no un-rotated resident W_std for the trellis
fn operandsTaken(dm: *const root.dense.Dense, trellis: A, rows: usize) bool {
    if (rows <= root.decode.Kernels.max_rows or rows <= dm.small_rows or dm.fused_prefill) return false;
    return !(rows > root.dense.unrot_min_rows and root.dense.residentUnrot(trellis));
}

/// `kdaProjPrompt` over W_hats the caller decoded: `w[m]` half [in, out_m], what `Dense.dequantize` makes of member m's
/// trellis, so a caller that decodes them once can run several calls over them (the outside plugin turboexl3_ling's
/// MLX_SERVE_LING_LAYER_WHAT: a layer-major layer's wide spans over its first wide span's W_hats). Each member's
/// prepared rows by its `linear`'s prepare at these rows (`Dense.prepare`: `prefillOperands`' first launch, its config),
/// then the grouped launch at `kdaProjPrompt`'s tile over the given W_hats: with each trellis's own W_hat, every
/// member's y is `kdaProjPrompt`'s word for word. Owned results. error.Exl3EngineNotImplemented where `kdaProjPrompt`
/// does not take the call (`takes`, or a member's `prefillOperands` declining at these rows), error.BadExl3Shape for a
/// W_hat that is not half [in, out_m], both before any launch. `kdaProjPrompt` itself is unchanged.
pub fn kdaProjPromptW(s: mlx.mlx_stream, x: A, p: [3]root.Proj, w: [3]A, dec: format.Decode) ![3]A {
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
    // every member's `linear` Dense (the one `kdaProjPrompt` takes), its operands' declines and its W_hat's shape, before
    // any launch
    var dms: [3]*root.dense.Dense = undefined;
    var outs: [3]u32 = undefined;
    for (p, w, &dms, &outs) |m, wm, *dm, *o| {
        o.* = @intCast(mlx.getShape(m.trellis)[1] * 16);
        dm.* = try root.denseCached(.{ .in = in_dim, .out = o.*, .rate = rate, .dec = dec, .out_dtype = xd });
        if (!operandsTaken(dm.*, m.trellis, rows)) return error.Exl3EngineNotImplemented;
        if (mlx.mlx_array_dtype(wm) != .float16 or !std.mem.eql(c_int, mlx.getShape(wm), &[_]c_int{ @intCast(in_dim), @intCast(o.*) })) return error.BadExl3Shape;
    }
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ @intCast(rows), @intCast(in_dim) }, 2, s));
    // each member's prepared rows by its `linear`'s prepare launch; its W_hat the caller's
    var xp: [3]A = undefined;
    var made: usize = 0;
    defer for (xp[0..made]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (dms, p) |dm, m| {
        xp[made] = try dm.prepare(s, x2, m.suh, rows);
        made += 1;
    }
    const ys = try launch(s, xp, w, .{ p[0].svh, p[1].svh, p[2].svh }, outs, @intCast(rows), @intCast(in_dim), xd, t);
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

// ── L6: member 1's (f's) bounded gate in the launch's epilogue (`kdaProjPromptGate`) ──
//
// At prompt rows the outside plugin turboexl3_ling runs a KDA layer's f through its bounded gate in one more launch
// (its `ling_kda_gate`: the ops of mlx-serve's `kdaGateChain`), which reads y_1 back and writes g f32. Here member 1's
// tiles run that gate's statements on y_1's words in the epilogue and store g: y_1 never reaches device memory, and the
// gate's launch is gone. The grid, the tile, the N tiles' members and members 0's and 2's statements are P4's.

/// the gate's launches' output line, in kpf_3_2's place: y names member 0's and 2's outputs (OUT_T); member 1's tiles
/// write y1 (f32) in the store loop
pub const kpf_3_2g = @embedFile("kernels/kdaproj/pf_3_2g.metal");
/// the gate's store loop (G0), in pg2's fused_6_2's place: members 0 and 2 store as fused_6_2 does; member 1 rounds its
/// word as that store does, then runs the bounded gate's statements on it in their order, exp(A_log) for each element
/// as the gate takes it, g into y1
pub const kpf_6g0 = @embedFile("kernels/kdaproj/pf_6g0.metal");
/// G1: kpf_6g0 with the gate's first statement read from `gea` (exp(A_log), made once by `gateEa`)
pub const kpf_6g1 = @embedFile("kernels/kdaproj/pf_6g1.metal");
/// `gateEa`'s launch: exp(A_log) per channel, the gate's first statement
pub const kgate_ea = @embedFile("kernels/kdaproj/gate_ea.metal");

/// (tests first: the launches' texts and calls come with the next commit; until then the texts are empty, no gate fits
/// and every call declines)
pub const text_pfg0: [:0]const u8 = "";
pub const text_pfg1: [:0]const u8 = "";
pub const text_gate_ea: [:0]const u8 = "";

/// G0's operands: P4's, then A_log and dt_bias (f32 [N1]) and the bound (an f32 scalar)
pub const pfg0_in = [_][*:0]const u8{ "x0", "x1", "x2", "w0", "w1", "w2", "svh0", "svh1", "svh2", "A_log", "dt_bias", "lb" };
/// G1's: P4's, then exp(A_log) from `gateEa`, dt_bias and the bound
pub const pfg1_in = [_][*:0]const u8{ "x0", "x1", "x2", "w0", "w1", "w2", "svh0", "svh1", "svh2", "gea", "dt_bias", "lb" };
/// The gate launches' outputs: P4's names, y1 f32
pub const pfg_out = pf_out;

/// The epilogue's two forms: G0 takes exp(A_log) for each element as the served gate does, G1 reads it from `gateEa`'s
/// array (one launch per layer, made once by the caller)
pub const GateForm = enum { g0, g1 };

/// Test seam: a kernel the gate launches run in place of their form's (a planted fault's red check; null: the form's)
pub const gate_kernel_override = Global(?mlx.mlx_fast_metal_kernel, "ling_kdaproj_pfg_kernel_override", null);

/// Member 1's bounded gate for `kdaProjPromptGate`: `a` the KDA layer's per-channel A_log (G0) or its exp from `gateEa`
/// (G1), `dt_bias` its per-channel bias, both f32 [N1] (N1 member 1's width), and the bound (the config's
/// kda_gate_lower_bound: Ling-3.1-flash's -5)
pub const Gate = struct {
    form: GateForm,
    a: A,
    dt_bias: A,
    bound: f32,
};

/// (tests first) Whether a gate's operands fit member 1 of width `n1`: none yet
pub fn gateShapeOk(n1: u32, a_dtype: mlx.mlx_dtype, a_shape: []const c_int, dt_dtype: mlx.mlx_dtype, dt_shape: []const c_int, bound: f32) bool {
    _ = .{ n1, a_dtype, a_shape, dt_dtype, dt_shape, bound };
    return false;
}

/// (tests first) Whether `kdaProjPromptGate` takes `gate`: never yet
pub fn gateTakes(p: [3]root.Proj, gate: Gate) bool {
    _ = .{ p, gate };
    return false;
}

/// (tests first) G1's operand: declined until the next commit
pub fn gateEa(s: mlx.mlx_stream, A_log: A) !A {
    _ = .{ s, A_log };
    return error.Exl3EngineNotImplemented;
}

/// (tests first) `kdaProjPrompt` with member 1's output its bounded gate: declined until the next commit
pub fn kdaProjPromptGate(s: mlx.mlx_stream, x: A, p: [3]root.Proj, gate: Gate, dec: format.Decode) ![3]A {
    _ = .{ s, x, p, gate, dec };
    return error.Exl3EngineNotImplemented;
}

/// (tests first) over given W_hats: declined until the next commit
pub fn kdaProjPromptWGate(s: mlx.mlx_stream, x: A, p: [3]root.Proj, w: [3]A, gate: Gate, dec: format.Decode) ![3]A {
    _ = .{ s, x, p, w, gate, dec };
    return error.Exl3EngineNotImplemented;
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

// ── CPU tests: L6, the KDA gate in the launch's epilogue ──

test "turboexl3 ling kdaproj gate texts: G0's and G1's launches are P4's pieces with the gate's output line in pf_3_2's place and its store loop in pg2's fused_6_2's, the cuts keep P4's and pg2's pieces whole, G0 reads A_log and G1 gea" {
    try testing.expectEqualStrings(kpf_1 ++ pgemm.fused_2 ++ kpf_3_1 ++ kpf_3_2g ++ pgemm.fused_4 ++ kpf_5 ++ pgemm.fused_6_1 ++ kpf_6g0 ++ kpf_undef, text_pfg0);
    try testing.expectEqualStrings(kpf_1 ++ pgemm.fused_2 ++ kpf_3_1 ++ kpf_3_2g ++ pgemm.fused_4 ++ kpf_5 ++ pgemm.fused_6_1 ++ kpf_6g1 ++ kpf_undef, text_pfg1);
    try testing.expectEqualStrings(kpf_3_1 ++ kpf_3_2, kpf_3);
    try testing.expectEqualStrings(pgemm.fused_6_1 ++ pgemm.fused_6_2, pgemm.fused_6);
    // P4's output line names its three outputs; the gate's names members 0's and 2's alone (y1 is its store loop's)
    try testing.expectEqual(@as(usize, 1), count(kpf_3_2, "y1"));
    try testing.expectEqual(@as(usize, 0), count(kpf_3_2g, "y1"));
    for ([_][]const u8{ "A_log", "dt_bias", "lb", "y1" }) |n| try testing.expect(count(kpf_6g0, n) > 0);
    for ([_][]const u8{ "gea", "dt_bias", "lb", "y1" }) |n| try testing.expect(count(kpf_6g1, n) > 0);
    try testing.expectEqual(@as(usize, 0), count(kpf_6g1, "A_log"));
    for ([_][]const u8{ text_pfg0, text_pfg1 }) |txt| for ([_][]const u8{ "x_shape", "w_shape", "_strides" }) |n| try testing.expectEqual(@as(usize, 0), count(txt, n));
    try testing.expectEqual(@as(usize, 1), count(text_gate_ea, "A_log_shape"));
}

test "turboexl3 ling kdaproj gate operands: f32 [8192] A_log (or its exp) and dt_bias with the bound -5 fit member 1 of 8192 channels; another dtype, rank or length, and a bound of 0 or not finite do not" {
    const v8192 = [_]c_int{8192};
    try testing.expect(gateShapeOk(8192, .float32, &v8192, .float32, &v8192, -5.0));
    try testing.expect(!gateShapeOk(8192, .bfloat16, &v8192, .float32, &v8192, -5.0));
    try testing.expect(!gateShapeOk(8192, .float32, &v8192, .float16, &v8192, -5.0));
    try testing.expect(!gateShapeOk(8192, .float32, &[_]c_int{ 1, 8192 }, .float32, &v8192, -5.0));
    try testing.expect(!gateShapeOk(8192, .float32, &v8192, .float32, &[_]c_int{ 1, 8192 }, -5.0));
    try testing.expect(!gateShapeOk(8192, .float32, &[_]c_int{8191}, .float32, &v8192, -5.0));
    try testing.expect(!gateShapeOk(8192, .float32, &v8192, .float32, &[_]c_int{4096}, -5.0));
    try testing.expect(!gateShapeOk(8192, .float32, &v8192, .float32, &v8192, 0.0));
    try testing.expect(!gateShapeOk(8192, .float32, &v8192, .float32, &v8192, std.math.nan(f32)));
    try testing.expect(!gateShapeOk(8192, .float32, &v8192, .float32, &v8192, -std.math.inf(f32)));
    try testing.expect(!gateShapeOk(0, .float32, &[_]c_int{0}, .float32, &[_]c_int{0}, -5.0));
}

/// `st` with the served gate's element, channel and output words for the store loop's (pins 17 to 24): member 1's word
/// for the element, channel o for the channel, y1's word for the output
fn gateStatement(a: std.mem.Allocator, pin: []const []const u8, st: []const u8) ![]u8 {
    var cur = try a.dupe(u8, st);
    var k: usize = 17;
    while (k < 25) : (k += 2) {
        const next = try std.mem.replaceOwned(u8, a, cur, pin[k], pin[k + 1]);
        a.free(cur);
        cur = next;
    }
    return cur;
}

/// Whether every needle is in `hay` once, in their order
fn onceInOrder(hay: []const u8, needles: []const []const u8) bool {
    var at: usize = 0;
    for (needles) |n| {
        if (count(hay, n) != 1) return false;
        const i = std.mem.indexOf(u8, hay, n).?;
        if (i < at) return false;
        at = i + n.len;
    }
    return true;
}

test "turboexl3 ling kdaproj gate statements: the store loop's gate is the served one-launch gate's statements in their order (its test-only copy) on member 1's word as P4 stores it, channel o, g into y1; G1's first reads gea, and gateEa's is the gate's first" {
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/kdaproj_gate.metal"));
    const copy = @embedFile("kernels/test/copy/ling_kda_gate.metal");
    const a = testing.allocator;
    // the served gate's seven statements (pins 2 to 8), once each in its copy, in this order
    try testing.expect(onceInOrder(copy, pin[2..9]));
    // G0's (pins 9 to 15): each the served statement with the store loop's words, at the loop body's indent (pin 25)
    for (pin[2..9], pin[9..16]) |st, want| {
        const got = try gateStatement(a, pin, st);
        defer a.free(got);
        const line = try std.mem.concat(a, u8, &.{ pin[25], got });
        defer a.free(line);
        try testing.expectEqualStrings(want, line);
    }
    try testing.expect(onceInOrder(kpf_6g0, pin[9..16]));
    // G1: the first statement reads gea (pin 16), the other six G0's
    const g1_want = [_][]const u8{ pin[16], pin[10], pin[11], pin[12], pin[13], pin[14], pin[15] };
    try testing.expect(onceInOrder(kpf_6g1, &g1_want));
    try testing.expectEqual(@as(usize, 0), count(kpf_6g1, pin[9]));
    // member 1's word is the store's value (pin 1) as fused_6_2 stores it (pin 0), once before the gate (pin 26); members
    // 0 and 2 store as fused_6_2 does (pin 27 holds pin 0)
    try testing.expectEqual(@as(usize, 1), count(pgemm.fused_6_2, pin[0]));
    try testing.expect(count(pin[0], pin[1]) == 1 and count(pin[26], pin[1]) == 1 and count(pin[27], pin[0]) == 1);
    for ([_][]const u8{ kpf_6g0, kpf_6g1 }) |txt| {
        try testing.expectEqual(@as(usize, 1), count(txt, pin[26]));
        try testing.expectEqual(@as(usize, 1), count(txt, pin[27]));
        try testing.expect(std.mem.indexOf(u8, txt, pin[26]).? < std.mem.indexOf(u8, txt, pin[10]).?);
    }
    // gateEa's statement (pin 28) takes the gate's first right-hand side (pin 29)
    try testing.expectEqual(@as(usize, 1), count(text_gate_ea, pin[28]));
    try testing.expect(count(pin[2], pin[29]) == 1 and count(pin[28], pin[29]) == 1);
}
