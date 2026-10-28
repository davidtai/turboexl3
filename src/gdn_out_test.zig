//! The GDN tail's GPU tests (`gdn_out.zig`; TURBOEXL3_GPU=1, else skipped): `root.gdnOutPrep` against the chain it
//! replaces, byte for byte, on synthetic trellises at Qwen3.8-Flash-Next's GatedDeltaNet geometry (hidden 2560, 48 value
//! heads of 128: in_proj_z 2560 -> 6144, out_proj 6144 -> 2560). The chain is `root.linearFinish` of z's inner product,
//! then MLX's rms_norm over each head, MLX's sigmoid of z and MLX's multiply, then `root.linearPrep`. Row counts: 17 (the
//! smallest prefill width the tail takes), 32 and 33 (either side of the register-path GEMM), a 1328-row chunk, an
//! 8192-row chunk and a batch of two 17-row sequences, in bf16, and 33 rows in f16. Lines: TURBOEXL3_GDN_OUT per case
//! (differing elements / total, PASS when 0).
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("test_support.zig");
const G = support.Graph;
const bf16Bits = support.bf16Bits;
const format = @import("format.zig");
const root = @import("root.zig");

const testing = std.testing;

const HIDDEN: u32 = 2560;
const HV: u32 = 48;
const DV: u32 = 128;
const VD: u32 = HV * DV;
const EPS: f32 = 1.0e-6;
/// The projections' rate (K5).
const N: u32 = 80;

const Cell = struct { batch: u32, seq: u32, dt: mlx.mlx_dtype };
const cells = [_]Cell{
    .{ .batch = 1, .seq = 17, .dt = .bfloat16 },
    .{ .batch = 1, .seq = 32, .dt = .bfloat16 },
    .{ .batch = 1, .seq = 33, .dt = .bfloat16 },
    .{ .batch = 1, .seq = 1328, .dt = .bfloat16 },
    .{ .batch = 1, .seq = 8192, .dt = .bfloat16 },
    .{ .batch = 2, .seq = 17, .dt = .bfloat16 },
    .{ .batch = 1, .seq = 33, .dt = .float16 },
};

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// A dense EXL3 projection in -> out at rate n as the host holds it: trellis U16 [in/16, out/16, n] (every pattern is a valid
/// codeword), suh F16 [in] in [0.5, 1.5], svh F16 [out] in [0.5, 1.5] / sqrt(in) (outputs O(1)).
fn synthProj(a: std.mem.Allocator, g: *G, rnd: std.Random, in: u32, out: u32, n: u32) !root.Proj {
    const words = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(words);
    for (words) |*w| w.* = rnd.int(u16);
    const suh = try a.alloc(u16, in);
    defer a.free(suh);
    for (suh) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const svh = try a.alloc(u16, out);
    defer a.free(svh);
    const sc: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(in)));
    for (svh) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * sc);
    return .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(words), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(suh), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(svh), &.{ci(out)}, .float16),
    };
}

/// An array of `shape` in `dt` (bf16, else f16) holding lo + span * N(0,1), each value exact in `dt`.
fn randArray(a: std.mem.Allocator, g: *G, rnd: std.Random, shape: []const c_int, dt: mlx.mlx_dtype, lo: f32, span: f32) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const bits = try a.alloc(u16, n);
    defer a.free(bits);
    for (bits) |*b| {
        const v = lo + span * rnd.floatNorm(f32);
        b.* = if (dt == .bfloat16) bf16Bits(v) else format.f32ToF16Bits(v);
    }
    return g.hostArray(std.mem.sliceAsBytes(bits), shape, dt);
}

/// MLX's sigmoid of every 16-bit pattern of `dt` (the host's table).
fn sigTable(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, dt: mlx.mlx_dtype) !mlx.mlx_array {
    const pats = try a.alloc(u16, 1 << 16);
    defer a.free(pats);
    for (pats, 0..) |*v, i| v.* = @intCast(i);
    const vals = try g.hostArray(std.mem.sliceAsBytes(pats), &.{1 << 16}, dt);
    var t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&t, vals, s));
    const tab = try g.adopt(t);
    try g.evalAll(&.{tab});
    return tab;
}

/// The number of elements whose bits differ (bf16 / f16 -> f32 is one-to-one on bit patterns), after a shape check.
fn bitsDiff(a: std.mem.Allocator, g: *G, want: mlx.mlx_array, got: mlx.mlx_array) !usize {
    if (!std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got)) or mlx.mlx_array_dtype(want) != mlx.mlx_array_dtype(got)) return std.math.maxInt(usize);
    const n = mlx.mlx_array_size(want);
    const wb = try a.alloc(f32, n);
    defer a.free(wb);
    const gb = try a.alloc(f32, n);
    defer a.free(gb);
    _ = try g.hostF32(try g.astype(want, .float32), wb);
    _ = try g.hostF32(try g.astype(got, .float32), gb);
    var diff: usize = 0;
    for (wb, gb) |p, q| {
        if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) diff += 1;
    }
    return diff;
}

/// The chain the tail replaces: z = linearFinish(zq), MLX's rms_norm(y) * sigmoid(z), out_proj's linearPrep of it.
fn chain(g: *G, s: mlx.mlx_stream, y: mlx.mlx_array, zq: mlx.mlx_array, z: root.Proj, norm_w: mlx.mlx_array, o: root.Proj, c: Cell) !mlx.mlx_array {
    const zf = try g.adopt(try root.linearFinish(s, zq, z, .mul1, c.dt));
    const z4 = try g.reshape(zf, &.{ ci(c.batch), ci(c.seq), ci(HV), ci(DV) });
    var n = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&n, y, norm_w, EPS, s));
    const normed = try g.adopt(n);
    var sig = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&sig, z4, s));
    var prod = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&prod, normed, try g.adopt(sig), s));
    const gated = try g.reshape(try g.adopt(prod), &.{ ci(c.batch), ci(c.seq), ci(VD) });
    return g.adopt(try root.linearPrep(s, gated, o, .mul1));
}

test "turboexl3 gpu gdn_out: z's finish, the gated norm and out_proj's prepare in one launch equal linearFinish, MLX's rms_norm, sigmoid and multiply, and linearPrep byte for byte (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const eps = mlx.mlx_array_new_float(EPS);
    defer _ = mlx.mlx_array_free(eps);
    var prng = std.Random.DefaultPrng.init(0x6D0_0A7);
    const rnd = prng.random();
    const top = g.mark();
    defer g.resetTo(top);
    const z = try synthProj(a, &g, rnd, HIDDEN, VD, N);
    const o = try synthProj(a, &g, rnd, VD, HIDDEN, N);
    var failed = false;
    for (cells) |c| {
        const cell_mark = g.mark();
        defer g.resetTo(cell_mark);
        const rows = c.batch * c.seq;
        const y = try randArray(a, &g, rnd, &.{ ci(c.batch), ci(c.seq), ci(HV), ci(DV) }, c.dt, 0.0, 1.0);
        // in_proj_z's inner product as its finish reads it (f16; z comes out O(1), so the sigmoid spans its range)
        const zq = try randArray(a, &g, rnd, &.{ ci(rows), ci(VD) }, .float16, 0.0, 64.0);
        const norm_w = try randArray(a, &g, rnd, &.{ci(DV)}, c.dt, 1.0, 0.25);
        const sigtab = try sigTable(a, &g, s, c.dt);
        const want = try chain(&g, s, y, zq, z, norm_w, o, c);
        const got = try g.adopt(try root.gdnOutPrep(s, y, zq, z, .{ .w = norm_w, .eps = eps, .sigtab = sigtab }, o));
        const diff = try bitsDiff(a, &g, want, got);
        const total = mlx.mlx_array_size(want);
        std.debug.print("TURBOEXL3_GDN_OUT batch={d} seq={d} dt={s} differ={d}/{d} {s}\n", .{ c.batch, c.seq, @tagName(c.dt), diff, total, if (diff == 0) "PASS" else "FAIL" });
        if (diff != 0) failed = true;
    }
    try testing.expect(!failed);
}

test "turboexl3 gpu gdn_out: what it does not serve is refused before any launch (decode rows, another head width, f32) (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const eps = mlx.mlx_array_new_float(EPS);
    defer _ = mlx.mlx_array_free(eps);
    var prng = std.Random.DefaultPrng.init(0x6D0_0A8);
    const rnd = prng.random();
    const z = try synthProj(a, &g, rnd, HIDDEN, VD, N);
    const o = try synthProj(a, &g, rnd, VD, HIDDEN, N);
    const sigtab = try sigTable(a, &g, s, .bfloat16);
    const norm_w = try randArray(a, &g, rnd, &.{ci(DV)}, .bfloat16, 1.0, 0.25);
    const norm: root.gdn_out.Norm = .{ .w = norm_w, .eps = eps, .sigtab = sigtab };
    // 16 rows: `linear` takes the GEMV there
    const y16 = try randArray(a, &g, rnd, &.{ 1, 16, ci(HV), ci(DV) }, .bfloat16, 0.0, 1.0);
    const zq16 = try randArray(a, &g, rnd, &.{ 16, ci(VD) }, .float16, 0.0, 4.0);
    try testing.expectError(error.Exl3EngineNotImplemented, root.gdnOutPrep(s, y16, zq16, z, norm, o));
    // 64-wide heads
    const y64 = try randArray(a, &g, rnd, &.{ 1, 33, ci(2 * HV), 64 }, .bfloat16, 0.0, 1.0);
    const zq33 = try randArray(a, &g, rnd, &.{ 33, ci(VD) }, .float16, 0.0, 4.0);
    try testing.expectError(error.Exl3EngineNotImplemented, root.gdnOutPrep(s, y64, zq33, z, norm, o));
    // f32 activations
    const y32 = try g.astype(try randArray(a, &g, rnd, &.{ 1, 33, ci(HV), ci(DV) }, .bfloat16, 0.0, 1.0), .float32);
    try testing.expectError(error.BadExl3Dtype, root.gdnOutPrep(s, y32, zq33, z, norm, o));
}
