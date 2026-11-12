//! pg2: a dense linear's prefill matmul and finish as one launch, at the shapes this machine runs faster that way.
//! `linearOn`'s matmul path at prefill rows is three launches past the prepare: the W_hat decode, MLX's f16 matmul (z half
//! [M, N] written, MLX's fused NAX kernel) and the finish (z read, y written: f16(H128(f16(z)) * svh) per 128-column
//! block, dense.text_finish). pg2 runs the matmul and the finish as one kernel (kernels/pg2/gemm_fused_*.metal):
//! MLX's fused NAX GEMM body (steel/gemm/kernels/steel_gemm_fused_nax.h `gemm`, MLX 0.32.3; its gemm_loop verbatim,
//! kernels/pg2/gemm_loop.metal: SK 32, the 16 x 32 x 16 multiply-accumulate on the f16 operands, f32 accumulation, k in
//! MLX's order) at a tile of this machine's choosing, then the finish as its epilogue: the accumulators rounded to half as
//! MLX's store rounds them (NAXFrag::store's static_cast), staged in threadgroup memory, and the finish's statements per
//! (row, 128-column block) with z's halves read from threadgroup memory. The tile only nests BK's k-steps and moves which
//! simdgroup computes which output (each simdgroup's N extent a multiple of 32: the MMA's N pairs as MLX's); a call is
//! taken only where MLX runs that regular kernel (no split K) on the NAX path, so y is the three-launch chain's y word for
//! word (the pg2u probe, w-pg2u-20261102-0609: 0 differing words over every shape, tile and row count, 2048 to 16384).
//! The launch is MLX's own threadgroup shape, (32, WN, WM): the probe's copy of MLX's kernel ran ~16% slower at MLX's own
//! geometry launched as (threads, 1, 1).
//! `table` holds this machine's (M5 Max) tile per (K, N), the pg2u probe's: each tile at or below the chain's time at
//! 16384, 7168 and 2048 rows, one tile a shape whatever the rows. The host's switch (`mode`): off (the chain), on (the
//! tabled shapes fused), check (the chain served, the fused launch beside it handed with the chain's y to `check_hook`).
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ks = sdk.kernel_launch;
const config = @import("config.zig");
const decode = @import("decode.zig");
const Global = @import("global.zig").Global;
const steel_nax = @import("steel.zig").nax;

const A = mlx.mlx_array;

pub const text_loop = @embedFile("kernels/pg2/gemm_loop.metal");
pub const text_fused = fused_1 ++ fused_2;
/// `text_fused` in pieces: fused_1 the GEMM and z's halves staged in threadgroup memory, fused_2 the finish over them (the
/// epilogue trunk_q8.zig's int8 GEMM runs too)
pub const fused_1 = fused_1_1 ++ fused_1_2 ++ fused_1_3 ++ fused_1_4 ++ fused_1_5 ++ fused_1_6;
pub const fused_2 = fused_2_1 ++ fused_2_2;
/// fused_1 in pieces, cut where a grouped launch (Ling's KDA prompt projections, models/ling/kdaproj_pf.zig) puts its own
/// statements: the shapes, the block and its bounds, the operands, the block's row, its column, then the GEMM to z's
/// staged halves
pub const fused_1_1 = @embedFile("kernels/pg2/gemm_fused_1_1.metal");
pub const fused_1_2 = @embedFile("kernels/pg2/gemm_fused_1_2.metal");
pub const fused_1_3 = @embedFile("kernels/pg2/gemm_fused_1_3.metal");
pub const fused_1_4 = @embedFile("kernels/pg2/gemm_fused_1_4.metal");
pub const fused_1_5 = @embedFile("kernels/pg2/gemm_fused_1_5.metal");
pub const fused_1_6 = @embedFile("kernels/pg2/gemm_fused_1_6.metal");
/// fused_2 in pieces: the finish up to its store loop, then the store loop and the row loop's close: cut where the KDA
/// gate's launches (models/ling/kdaproj_pf.zig `kdaProjPromptGate`) put their own store loop
pub const fused_2_1 = @embedFile("kernels/pg2/gemm_fused_2_1.metal");
pub const fused_2_2 = @embedFile("kernels/pg2/gemm_fused_2_2.metal");

fn sentinel(comptime s: []const u8) [:0]const u8 {
    comptime {
        var buf: [s.len:0]u8 = undefined;
        @memcpy(buf[0..s.len], s);
        const final = buf;
        return &final;
    }
}

/// MLX's kernel attribute on the generated definition: `[[kernel]]` becomes `[[kernel, max_total_threads_per_threadgroup(WM
/// * WN * 32)]]` (WM and WN the definition's template parameters, MLX's own expression); the body's last line undefines
/// the macro, so the explicit instantiation MLX appends keeps its plain `[[kernel]]`.
pub const header: [:0]const u8 = sentinel(steel_nax ++ "using namespace mlx::steel;\n" ++ text_loop ++ "\n#define kernel kernel, max_total_threads_per_threadgroup(WM * WN * 32)\n");
const body: [:0]const u8 = sentinel(text_fused ++ "\n#undef kernel\n");

const fused_in = [_][*:0]const u8{ "x", "w", "svh" };
const fused_out = [_][*:0]const u8{"y"};

/// One tile: MLX's BM x BN, WM x WN simdgroups, BK, swizzle_log
pub const Tile = struct {
    bm: u16,
    bn: u16,
    bk: u16,
    wm: u8,
    wn: u8,
    swz: u8,

    pub fn threads(t: Tile) u32 {
        return @as(u32, t.wm) * t.wn * 32;
    }

    /// Exact by construction and staged within 32 KB: each simdgroup's tile whole 16-row fragments and N pairs of 16
    /// columns, BK whole k-steps, the tile whole 128-column blocks
    pub fn admitted(t: Tile) bool {
        if (t.wm == 0 or t.wn == 0 or t.bm % t.wm != 0 or t.bn % t.wn != 0) return false;
        const sm = t.bm / t.wm;
        const sn = t.bn / t.wn;
        return sm % 16 == 0 and sn % 32 == 0 and t.bk % 32 == 0 and t.bk > 0 and t.bn % 128 == 0 and
            @as(u32, t.bm) * t.bn * 2 <= 32768 and t.threads() <= 1024 and t.swz <= 4;
    }
};

pub const Entry = struct { k: u32, n: u32, tile: Tile };

/// This machine's tiles (M5 Max), the pg2u probe's (w-pg2u-20261102-0609; the fused launch's time over the chain's at
/// 16384 / 7168 / 2048 rows)
pub const table = [_]Entry{
    // wq_b 0.866 / 0.870 / 0.900
    .{ .k = 1280, .n = 32768, .tile = .{ .bm = 32, .bn = 256, .bk = 256, .wm = 1, .wn = 4, .swz = 2 } },
    // wo_b 0.957 / 0.973 / 0.899
    .{ .k = 8192, .n = 5120, .tile = .{ .bm = 32, .bn = 256, .bk = 512, .wm = 1, .wn = 4, .swz = 2 } },
    // the shared expert's w1 and w3 0.933 / 0.984 / 0.982
    .{ .k = 5120, .n = 2304, .tile = .{ .bm = 32, .bn = 256, .bk = 512, .wm = 1, .wn = 4, .swz = 2 } },
    // its w2 0.938 / 0.933 / 0.950
    .{ .k = 2304, .n = 5120, .tile = .{ .bm = 32, .bn = 256, .bk = 256, .wm = 1, .wn = 4, .swz = 2 } },
    // wq_a 0.963 / 0.920 / 0.963
    .{ .k = 5120, .n = 1280, .tile = .{ .bm = 32, .bn = 256, .bk = 512, .wm = 1, .wn = 4, .swz = 2 } },
    // wkv and the compressors' wkv and wgate 0.986 / 0.951 / 0.962
    .{ .k = 5120, .n = 512, .tile = .{ .bm = 64, .bn = 256, .bk = 512, .wm = 2, .wn = 4, .swz = 2 } },
};

/// Tiles a host times on its own machine, to pick one a (K, N) for `host_table` (each admitted, every N that is a
/// multiple of 512 divisible by its BN): `table`'s, their BK and swizzle neighbors, and wider and taller tiles. A
/// host names them by index, so the order is fixed: new tiles go at the end.
pub const candidates = [_]Tile{
    .{ .bm = 32, .bn = 256, .bk = 256, .wm = 1, .wn = 4, .swz = 2 },
    .{ .bm = 32, .bn = 256, .bk = 512, .wm = 1, .wn = 4, .swz = 2 },
    .{ .bm = 64, .bn = 256, .bk = 256, .wm = 2, .wn = 4, .swz = 2 },
    .{ .bm = 64, .bn = 256, .bk = 512, .wm = 2, .wn = 4, .swz = 2 },
    .{ .bm = 64, .bn = 128, .bk = 512, .wm = 2, .wn = 2, .swz = 2 },
    .{ .bm = 128, .bn = 128, .bk = 512, .wm = 4, .wn = 4, .swz = 2 },
    .{ .bm = 128, .bn = 128, .bk = 256, .wm = 4, .wn = 2, .swz = 2 },
    .{ .bm = 32, .bn = 512, .bk = 256, .wm = 1, .wn = 8, .swz = 2 },
    .{ .bm = 32, .bn = 256, .bk = 512, .wm = 1, .wn = 4, .swz = 1 },
    .{ .bm = 32, .bn = 256, .bk = 512, .wm = 1, .wn = 4, .swz = 3 },
    .{ .bm = 64, .bn = 256, .bk = 512, .wm = 1, .wn = 4, .swz = 2 },
    .{ .bm = 32, .bn = 128, .bk = 512, .wm = 1, .wn = 4, .swz = 2 },
};

/// A host's own tiles a (K, N), from `candidates` as timed on its machine (a model's linears), looked up before `table`;
/// null: `table` alone. Set at load, before the first prompt, with `mode`; the slice outlives every call.
pub const host_table = Global(?[]const Entry, "pgemm_host_table", null);

/// The host's switch: off, on, check (set at load, before the first prompt)
pub const Mode = enum(u8) { off, on, check };
pub const mode = Global(Mode, "pgemm_mode", .off);

/// check: the chain's y (served) and the fused launch's on the same operands, both lazy; the host compares them
pub const CheckFn = *const fn (s: mlx.mlx_stream, want: A, got: A) void;
pub const check_hook = Global(?CheckFn, "pgemm_check_hook", null);

/// Fused launches served (on) or run beside the chain (check) since load, and their output elements
pub const calls = Global(std.atomic.Value(u64), "pgemm_calls", std.atomic.Value(u64).init(0));
pub const elems = Global(std.atomic.Value(u64), "pgemm_elems", std.atomic.Value(u64).init(0));

/// Whether MLX runs [m, k] x [k, n] on its regular fused kernel (not the NAX split-K: K >= 3 max(M, N), or max(M, N) <=
/// 1024 and K > 2 max(M, N))
pub fn mlxRegular(m: u32, n: u32, k: u32) bool {
    const mx = @max(m, n);
    return !(k >= 3 * mx or (mx <= 1024 and k > 2 * mx));
}

/// The table's tile for xp half [M, K] by W half [K, N] where MLX runs its regular fused kernel on the NAX path; null
/// elsewhere (the chain)
pub fn tileFor(xp: A, w: A) ?Tile {
    if (!config.get().nax) return null;
    const xs = mlx.getShape(xp);
    const wsh = mlx.getShape(w);
    if (xs.len != 2 or wsh.len != 2 or xs[0] < 1 or xs[1] < 1 or wsh[1] < 1 or xs[1] != wsh[0]) return null;
    if (mlx.mlx_array_dtype(xp) != .float16 or mlx.mlx_array_dtype(w) != .float16) return null;
    const m: u32 = @intCast(xs[0]);
    const k: u32 = @intCast(xs[1]);
    const n: u32 = @intCast(wsh[1]);
    if (!mlxRegular(m, n, k)) return null;
    return tileOf(k, n);
}

/// The tile a (K, N): the host's table first, then `table`; null where neither holds an admitted tile for it
pub fn tileOf(k: u32, n: u32) ?Tile {
    if (host_table.get().*) |ht| for (ht) |e| if (e.k == k and e.n == n) return if (e.tile.admitted() and n % e.tile.bn == 0) e.tile else null;
    for (table) |e| if (e.k == k and e.n == n) return if (e.tile.admitted() and n % e.tile.bn == 0) e.tile else null;
    return null;
}

const kernel_cache = Global(?mlx.mlx_fast_metal_kernel, "pgemm_kernel", null);
const kernel_lock = Global(std.atomic.Value(u32), "pgemm_kernel_lock", std.atomic.Value(u32).init(0));

fn kernelGet() !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    if (kernel_cache.get().*) |k| return k;
    const k = ks.newKernel("turboexl3_pgemm_fused", &fused_in, &fused_out, body, header, true) orelse return error.KernelCreateFailed;
    kernel_cache.get().* = k;
    return k;
}

/// The fused launch: y [M, N] of `out_dtype` = finish(xp [M, K] W [K, N]) at tile `t`, MLX's grid (tiles_n << swz by
/// ceil(tiles_m >> swz) threadgroups) in MLX's threadgroup shape (32, WN, WM); the caller owns y
pub fn launch(s: mlx.mlx_stream, xp: A, w: A, svh: A, out_dtype: mlx.mlx_dtype, t: Tile) !A {
    const kern = try kernelGet();
    const xs = mlx.getShape(xp);
    const m: u32 = @intCast(xs[0]);
    const kd: u32 = @intCast(xs[1]);
    const n: u32 = @intCast(mlx.getShape(w)[1]);
    const tile: u32 = @as(u32, 1) << @intCast(t.swz);
    const tiles_m = (m + t.bm - 1) / t.bm;
    const tiles_n = n / t.bn;
    const cfg = try ks.buildConfig(.{
        .grid = .{ tiles_n * tile * 32, ((tiles_m + tile - 1) / tile) * t.wn, t.wm },
        .threadgroup = .{ 32, t.wn, t.wm },
        .out_shapes = &.{&[_]c_int{ @intCast(m), @intCast(n) }},
        .out_dtypes = &.{out_dtype},
        .template = &.{
            .{ .name = "T", .value = .{ .dtype = .float16 } },
            .{ .name = "BM", .value = .{ .int = t.bm } },
            .{ .name = "BN", .value = .{ .int = t.bn } },
            .{ .name = "BK", .value = .{ .int = t.bk } },
            .{ .name = "WM", .value = .{ .int = t.wm } },
            .{ .name = "WN", .value = .{ .int = t.wn } },
            .{ .name = "SWZ", .value = .{ .int = t.swz } },
            .{ .name = "AM", .value = .{ .int = @intFromBool(m % t.bm == 0) } },
            .{ .name = "AN", .value = .{ .int = @intFromBool(n % t.bn == 0) } },
            .{ .name = "AK", .value = .{ .int = @intFromBool(kd % t.bk == 0) } },
            .{ .name = "OUT_T", .value = .{ .dtype = out_dtype } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [1]A = undefined;
    try ks.applyConfig(kern, cfg, &.{ xp, w, svh }, &o, s);
    _ = calls.get().*.fetchAdd(1, .monotonic);
    _ = elems.get().*.fetchAdd(@as(u64, m) * n, .monotonic);
    return o[0];
}

/// On, at a tabled shape: the fused launch's y (the caller owns it); null elsewhere (the caller runs the chain)
pub fn fusedOn(s: mlx.mlx_stream, xp: A, w: A, svh: A, out_dtype: mlx.mlx_dtype) !?A {
    if (mode.get().* != .on) return null;
    const t = tileFor(xp, w) orelse return null;
    return try launch(s, xp, w, svh, out_dtype, t);
}

/// Check, at a tabled shape: the fused launch beside the chain's `want` (served), both handed to the host's hook
pub fn check(s: mlx.mlx_stream, xp: A, w: A, svh: A, out_dtype: mlx.mlx_dtype, want: A) void {
    if (mode.get().* != .check) return;
    const hook = check_hook.get().* orelse return;
    const t = tileFor(xp, w) orelse return;
    const got = launch(s, xp, w, svh, out_dtype, t) catch return;
    defer _ = mlx.mlx_array_free(got);
    hook(s, want, got);
}

/// At load (on or check): every tabled tile's kernel (the host's table's and `table`'s, or `only`'s (K, N)) compiled for
/// `out_dtype` (M aligned and not, N and K as the entry's), on small zero operands, evaluated here, so a prompt's first
/// call pays no JIT
pub fn warm(s: mlx.mlx_stream, out_dtype: mlx.mlx_dtype, only: ?[2]u32) !void {
    if (mode.get().* == .off or !config.get().nax) return;
    const hosts: []const Entry = host_table.get().* orelse &.{};
    for ([_][]const Entry{ hosts, &table }) |entries| for (entries) |e| {
        const t = e.tile;
        if (!t.admitted()) continue;
        if (only) |kn| if (kn[0] != e.k or kn[1] != e.n) continue;
        const kd: u32 = if (e.k % t.bk == 0) t.bk else t.bk + 32;
        for ([_]u32{ t.bm, t.bm + 1 }) |m| {
            var z: [3]A = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
            defer for (z) |a| {
                _ = mlx.mlx_array_free(a);
            };
            try mlx.check(mlx.mlx_zeros(&z[0], &[_]c_int{ @intCast(m), @intCast(kd) }, 2, .float16, s));
            try mlx.check(mlx.mlx_zeros(&z[1], &[_]c_int{ @intCast(kd), @intCast(t.bn) }, 2, .float16, s));
            try mlx.check(mlx.mlx_zeros(&z[2], &[_]c_int{@intCast(t.bn)}, 1, .float16, s));
            const y = try launch(s, z[0], z[1], z[2], out_dtype, t);
            defer _ = mlx.mlx_array_free(y);
            try mlx.check(mlx.mlx_array_eval(y));
        }
    };
    // the warm launches are not the prompt's
    calls.get().*.store(0, .monotonic);
    elems.get().*.store(0, .monotonic);
}

const testing = std.testing;

test "pgemm: the epilogue runs the finish's statements, the loop is MLX's gemm_loop by its statements, the table's tiles exact by construction" {
    const finish = @embedFile("kernels/dense/finish_v.metal") ++ @embedFile("kernels/dense/finish_store.metal");
    var it = std.mem.splitScalar(u8, finish, '\n');
    var lines: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        lines += 1;
        if (std.mem.indexOf(u8, line, "constexpr uint DIMC") != null) continue;
        if (std.mem.indexOf(u8, line, "threadgroup_position_in_grid") != null) continue;
        if (std.mem.indexOf(u8, line, "v[j] = float(half(float(z[") != null) {
            try testing.expect(std.mem.indexOf(u8, text_fused, "for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(zt[rt * uint(BN) + bt * 128u + L + 32u * j])));") != null);
            continue;
        }
        try testing.expect(std.mem.indexOf(u8, text_fused, std.mem.trim(u8, line, " ")) != null);
    }
    try testing.expectEqual(@as(usize, 10), lines);
    try testing.expect(std.mem.indexOf(u8, text_fused, "Dtile.template store<half, BN, 1>(&zt[tm * BN + tn]);") != null);
    try testing.expect(std.mem.indexOf(u8, text_loop, "auto pg1_gemm_loop(") != null);
    try testing.expect(std.mem.indexOf(u8, text_loop, "NAXTile<T, RA, CA> Atile;") != null);
    try testing.expect(std.mem.indexOf(u8, text_loop, "tile_matmad_nax(") != null);
    try testing.expect(std.mem.endsWith(u8, body, "#undef kernel\n"));
    for (table) |e| try testing.expect(e.tile.admitted() and e.n % e.tile.bn == 0);
    try testing.expect(mlxRegular(16384, 512, 5120));
    try testing.expect(!mlxRegular(1024, 512, 5120));
    try testing.expect(mlxRegular(2048, 32768, 1280));
}
