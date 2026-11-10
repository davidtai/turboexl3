//! Trunk int8 (`Config.pf_trunk_int8`; off by default, lossy): a dense linear's prefill rows on int8 at the host's shapes
//! (`shapes`), in `Dense.linearOn`'s matmul path in place of its prepare, W_hat decode, matmul and finish. Three launches:
//! - `prepare` (kernels/trunk_q8/prep_1, dense's prep_2, kernels/trunk_q8/prep_2): one threadgroup a row; each 128-block's
//!   f16 values are text_prep's (x's f16 rounding, suh, the H128: the chain's xp), the row at one scale (its amax / 127,
//!   routed int8 2's scheme), int8 rint(xp 127 / amax) in natural order: xq int8 [MP, in] (MP the rows rounded up to `bm`,
//!   so the GEMM reads whole tiles; the pad rows are never written or stored), sx f32 [rows] the scales, xsum f32 [rows]
//!   xp's sums (each block's sum, the blocks in order);
//! - `decodeW` (dense's dequant_1, kernels/trunk_q8/dequant_2): dequantize's tile decode with each MUL1 codeword's weight
//!   stored as int8 round((t - 510) / 4) of its byte sum t (`exl3pf_q8`, the routed int8's): wq int8 [out, in] (k fastest),
//!   one scale for every weight (4 A);
//! - `gemm` (kernels/trunk_q8/gemm_1, then pg2's epilogue pgemm.fused_2): one mpp matmul2d of `bm` x `bn` x 64 int8 a
//!   threadgroup (exact int32 sums over K; mlx-serve's w8a8 GEMM body), z half = 4 A sx (xq . wq) + (510 A + C) xsum (the
//!   routed int8 GEMM's epilogue at one scale a row) staged in threadgroup memory, then the finish y = f16(H128(f16(z)) svh)
//!   per (row, 128-column block), pg2's statements.
//! A row's y depends on its own row alone (the prepare is per row and the sums exact), so rows split across calls give
//! the one call's bytes. MUL1 packs on the NAX path; out a multiple of `bn`, in of 128.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ks = sdk.kernel_launch;
const config = @import("config.zig");
const decode = @import("decode.zig");
const dense = @import("dense.zig");
const pgemm = @import("pgemm.zig");
const prefill = @import("prefill.zig");
const Global = @import("global.zig").Global;

const A = mlx.mlx_array;

fn sentinel(comptime s: []const u8) [:0]const u8 {
    comptime {
        var buf: [s.len:0]u8 = undefined;
        @memcpy(buf[0..s.len], s);
        const final = buf;
        return &final;
    }
}

/// The int8 prepare's own pieces: prep_1 its constants, the row's threadgroup and the loop over a simdgroup's blocks (dense's
/// prep_2 inside it), prep_2 the block's halves held, then the row's scale and the int8 store
pub const prep_1 = @embedFile("kernels/trunk_q8/prep_1.metal");
pub const prep_2 = @embedFile("kernels/trunk_q8/prep_2.metal");
/// The int8 decode's store, after dense's dequant_1
pub const dequant_2 = @embedFile("kernels/trunk_q8/dequant_2.metal");
/// The int8 GEMM to z's halves in threadgroup memory, before pg2's epilogue (pgemm.fused_2)
pub const gemm_1 = @embedFile("kernels/trunk_q8/gemm_1.metal");

pub const text_prep: [:0]const u8 = sentinel(prep_1 ++ dense.prep_2 ++ prep_2);
pub const text_dequant: [:0]const u8 = sentinel(dense.dequant_1 ++ dequant_2);
pub const text_gemm: [:0]const u8 = sentinel(gemm_1 ++ pgemm.fused_2);

const prep_in = [_][*:0]const u8{ "x", "suh" };
const prep_out = [_][*:0]const u8{ "xq", "sx", "xsum" };
const dequant_in = [_][*:0]const u8{"w"};
const dequant_out = [_][*:0]const u8{"wq"};
const gemm_in = [_][*:0]const u8{ "xq", "wq", "sx", "xsum", "svh" };
const gemm_out = [_][*:0]const u8{"y"};

/// The GEMM's tile: `bm` rows by `bn` outputs a threadgroup on `wm` x `wn` simdgroups (mlx-serve's w8a8 tile past 8192
/// rows; `bn` one 128-column finish block)
pub const bm: u32 = 64;
pub const bn: u32 = 128;
const wm: u32 = 2;
const wn: u32 = 2;

/// One linear's (K, N)
pub const Shape = struct { k: u32, n: u32 };

/// The host's shapes on int8 under `Config.pf_trunk_int8`, set at load before the first prompt (the slice outlives every
/// call); null: none
pub const shapes = Global(?[]const Shape, "trunk_q8_shapes", null);

/// GEMM launches since load, and their output elements
pub const calls = Global(std.atomic.Value(u64), "trunk_q8_calls", std.atomic.Value(u64).init(0));
pub const elems = Global(std.atomic.Value(u64), "trunk_q8_elems", std.atomic.Value(u64).init(0));
/// the engage line, once a process
const said = Global(bool, "trunk_q8_said", false);

const Kernels = struct { prep: mlx.mlx_fast_metal_kernel, dequant: mlx.mlx_fast_metal_kernel, gemm: mlx.mlx_fast_metal_kernel };
const kernel_cache = Global(?Kernels, "trunk_q8_kernels", null);
const kernel_lock = Global(std.atomic.Value(u32), "trunk_q8_kernel_lock", std.atomic.Value(u32).init(0));

/// The three kernels, made on first use and kept for the process (pgemm's kernel's lifetime)
fn kernels() !Kernels {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    if (kernel_cache.get().*) |k| return k;
    const p = ks.newKernel("turboexl3_trunk_q8_prep", &prep_in, &prep_out, text_prep, decode.header, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(p);
    const d = ks.newKernel("turboexl3_trunk_q8_dequant", &dequant_in, &dequant_out, text_dequant, prefill.header_i8, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(d);
    const g = ks.newKernel("turboexl3_trunk_q8_gemm", &gemm_in, &gemm_out, text_gemm, prefill.header_i8, true) orelse return error.KernelCreateFailed;
    const k: Kernels = .{ .prep = p, .dequant = d, .gemm = g };
    kernel_cache.get().* = k;
    return k;
}

/// Whether `d`'s prefill rows on its matmul path take int8: the switch on, the NAX path, MUL1, its (in, out) one of the
/// host's shapes, out a multiple of `bn` and in of 128
pub fn takes(d: *const dense.Dense) bool {
    const c = config.get();
    if (!c.pf_trunk_int8 or !c.nax or d.shape.dec.codebook != .mul1) return false;
    if (d.shape.out % bn != 0 or d.shape.in % 128 != 0) return false;
    const list = shapes.get().* orelse return false;
    for (list) |sh| if (sh.k == d.shape.in and sh.n == d.shape.out) return true;
    return false;
}

/// `rows` rounded up to whole GEMM tiles: xq's rows
pub fn padRows(rows: usize) usize {
    return (rows + bm - 1) / bm * bm;
}

/// The int8 prepared rows of one call (`prepare`); the caller frees them
pub const Act = struct {
    xq: A,
    sx: A,
    xsum: A,

    pub fn deinit(a: Act) void {
        for ([_]A{ a.xq, a.sx, a.xsum }) |x| _ = mlx.mlx_array_free(x);
    }
};

/// x [rows, in] (bf16, f16 or f32) prepared as int8 on `s`: xq int8 [padRows(rows), in], sx and xsum f32 [rows]
pub fn prepare(s: mlx.mlx_stream, d: *const dense.Dense, x: A, suh: A, rows: usize) !Act {
    const k = try kernels();
    const cfg = try ks.buildConfig(.{
        .grid = .{ 256, @intCast(rows), 1 },
        .threadgroup = .{ 256, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ @intCast(padRows(rows)), @intCast(d.shape.in) }, &[_]c_int{@intCast(rows)}, &[_]c_int{@intCast(rows)} },
        .out_dtypes = &.{ .int8, .float32, .float32 },
        .template = &.{.{ .name = "DIM", .value = .{ .int = @intCast(d.shape.in) } }},
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [3]A = undefined;
    try ks.applyConfig(k.prep, cfg, &.{ x, suh }, &o, s);
    return .{ .xq = o[0], .sx = o[1], .xsum = o[2] };
}

/// wq int8 [out, in] decoded from the trellis on `s`; the caller frees it (a caller over several calls of one linear, the
/// plugin's KDA spans, decodes it once)
pub fn decodeW(s: mlx.mlx_stream, d: *const dense.Dense, trellis: A) !A {
    const k = try kernels();
    const tiles = (d.shape.in / 16) * (d.shape.out / 16);
    const t = d.templ();
    const cfg = try ks.buildConfig(.{
        .grid = .{ ((tiles + 3) / 4) * 128, 1, 1 },
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = &.{&[_]c_int{ @intCast(d.shape.out), @intCast(d.shape.in) }},
        .out_dtypes = &.{.int8},
        .template = &.{ t[0], t[1], t[2], .{ .name = "IN", .value = .{ .int = @intCast(d.shape.in) } }, .{ .name = "OUT", .value = .{ .int = @intCast(d.shape.out) } } },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [1]A = undefined;
    try ks.applyConfig(k.dequant, cfg, &.{trellis}, &o, s);
    return o[0];
}

/// y [rows, out] of `out_dtype` = finish(z) of `act`'s rows by wq [out, in] on `s`; the caller owns y
pub fn gemm(s: mlx.mlx_stream, act: Act, wq: A, svh: A, rows: usize, out_dtype: mlx.mlx_dtype) !A {
    const k = try kernels();
    const n: u32 = @intCast(mlx.getShape(wq)[0]);
    const tiles_m: u32 = @intCast((rows + bm - 1) / bm);
    const cfg = try ks.buildConfig(.{
        .grid = .{ (n / bn) * (wm * wn * 32), tiles_m, 1 },
        .threadgroup = .{ wm * wn * 32, 1, 1 },
        .out_shapes = &.{&[_]c_int{ @intCast(rows), @intCast(n) }},
        .out_dtypes = &.{out_dtype},
        .template = &.{
            .{ .name = "BM", .value = .{ .int = @intCast(bm) } },
            .{ .name = "BN", .value = .{ .int = @intCast(bn) } },
            .{ .name = "WM", .value = .{ .int = @intCast(wm) } },
            .{ .name = "WN", .value = .{ .int = @intCast(wn) } },
            .{ .name = "OUT_T", .value = .{ .dtype = out_dtype } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [1]A = undefined;
    try ks.applyConfig(k.gemm, cfg, &.{ act.xq, wq, act.sx, act.xsum, svh }, &o, s);
    _ = calls.get().*.fetchAdd(1, .monotonic);
    _ = elems.get().*.fetchAdd(@as(u64, rows) * n, .monotonic);
    return o[0];
}

/// One call's rows x2 [rows, in] against a decoded wq (`decodeW`): the prepare and the GEMM on `s`, y [rows, out] in
/// `d.shape.out_dtype`
pub fn rowsOn(s: mlx.mlx_stream, d: *const dense.Dense, x2: A, suh: A, svh: A, wq: A, rows: usize) !A {
    const act = try prepare(s, d, x2, suh, rows);
    defer act.deinit();
    const y = try gemm(s, act, wq, svh, rows, d.shape.out_dtype);
    engaged(d, rows);
    return y;
}

/// `Dense.linearOn`'s matmul path on int8 (`takes`): the prepare on `s`, the decode and the GEMM on `rs` (the chain's
/// streams: MLX joins them where xq crosses), y [rows, out] in `d.shape.out_dtype`
pub fn linearOn(d: *const dense.Dense, s: mlx.mlx_stream, rs: mlx.mlx_stream, x: A, trellis: A, suh: A, svh: A, rows: usize) !A {
    const act = try prepare(s, d, x, suh, rows);
    defer act.deinit();
    const wq = try decodeW(rs, d, trellis);
    defer _ = mlx.mlx_array_free(wq);
    const y = try gemm(rs, act, wq, svh, rows, d.shape.out_dtype);
    engaged(d, rows);
    return y;
}

fn engaged(d: *const dense.Dense, rows: usize) void {
    if (said.get().*) return;
    said.get().* = true;
    std.debug.print("[turboexl3] trunk int8 engaged (Config.pf_trunk_int8; lossy): int8 prepare at one scale a row, int8 decode, int8 GEMM with the finish; the first {d} x {d} at {d} rows\n", .{ d.shape.in, d.shape.out, rows });
}

const testing = std.testing;

test "turboexl3 trunk int8 texts: the prepare runs text_prep's block statements, the decode dequantize's tile load, the GEMM pg2's epilogue; the cut texts are their pieces" {
    try testing.expectEqualStrings(dense.text_prep, dense.prep_1 ++ dense.prep_2 ++ dense.prep_3);
    try testing.expectEqualStrings(dense.text_dequant, dense.dequant_1 ++ dense.dequant_2);
    try testing.expectEqualStrings(pgemm.text_fused, pgemm.fused_1 ++ pgemm.fused_2);
    try testing.expect(std.mem.indexOf(u8, text_prep, dense.prep_2) != null);
    try testing.expect(std.mem.startsWith(u8, text_dequant, dense.dequant_1));
    try testing.expect(std.mem.endsWith(u8, text_gemm, pgemm.fused_2));
    // pg2's epilogue is the GEMM's last piece whole (pgemm's test holds the finish's statements in it)
    try testing.expect(bn % 128 == 0 and bm % 64 == 0 and wm * wn == 4);
    try testing.expectEqual(@as(usize, 128), padRows(100));
    try testing.expectEqual(@as(usize, 4096), padRows(4096));
}
