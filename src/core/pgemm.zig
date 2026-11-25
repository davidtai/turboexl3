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
//! Split K (`split_k`, the host's, with `mode` on): where MLX runs its NAX split-K kernel and its accumulation (K >= 3
//! max(M, N), or max(M, N) <= 1024 and K > 2 max(M, N)) at a tabled (K, N), the fused launch runs MLX's partitions in
//! registers (kernels/pg2/gemm_splitk.metal in the GEMM's place): each partition's gemm_loop from a cleared tile on the
//! same operands, added in partition order to a cleared f32 tile, the accumulation's adds, then the store's rounding to
//! half and the finish. So y is the chain's word for word there too, and the f32 partials [P, M, N] never reach device
//! memory. Taken only where MLX's partition size does not depend on the device (`splitKAt`).
//! A caller's epilogue (`launchWith`): pg2's GEMM and the finish up to its store loop (`fused_1 ++ fused_2_1`), the
//! caller's store loop in `fused_2_2_1`'s place, then the row loop's close (`fused_2_2_2`) and `undef`; the hy4f arms'
//! prompt-row finishes (models/hy4f/gate.zig, qrope.zig, swiglu.zig) take that form at their tabled shapes.
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
/// fused_1_6 in pieces: the operands at the simdgroup's tile and its bounds, the GEMM (cut where split K runs MLX's
/// partitions in its place: `text_splitk`), then z's halves staged
pub const fused_1_6_1 = @embedFile("kernels/pg2/gemm_fused_1_6_1.metal");
pub const fused_1_6_2 = @embedFile("kernels/pg2/gemm_fused_1_6_2.metal");
pub const fused_1_6_3 = @embedFile("kernels/pg2/gemm_fused_1_6_3.metal");
pub const fused_1_6 = fused_1_6_1 ++ fused_1_6_2 ++ fused_1_6_3;
/// fused_2 in pieces: the finish up to its store loop, then the store loop and the row loop's close: cut where the KDA
/// gate's launches (models/ling/kdaproj_pf.zig `kdaProjPromptGate`) put their own store loop
pub const fused_2_1 = @embedFile("kernels/pg2/gemm_fused_2_1.metal");
pub const fused_2_2 = fused_2_2_1 ++ fused_2_2_2;
/// fused_2_2 in pieces: the store loop, then the row loop's close (a caller's epilogue puts its own store loop before it)
pub const fused_2_2_1 = @embedFile("kernels/pg2/gemm_fused_2_2_1.metal");
pub const fused_2_2_2 = @embedFile("kernels/pg2/gemm_fused_2_2_2.metal");
/// The header's `kernel` macro undefined after a body (`header` defines it), so the instantiation MLX appends keeps its
/// plain `[[kernel]]`
pub const undef = @embedFile("kernels/pg2/gemm_undef.metal");
/// Split K in registers: MLX's partitions and its accumulation's adds, in `fused_1_6_2`'s place
pub const text_splitk = @embedFile("kernels/pg2/gemm_splitk.metal");
/// fused_1 with split K in registers: the GEMM to z's staged halves over MLX's partitions
pub const fused_1_sk = fused_1_1 ++ fused_1_2 ++ fused_1_3 ++ fused_1_4 ++ fused_1_5 ++ fused_1_6_1 ++ text_splitk ++ fused_1_6_3;
pub const text_fused_sk = fused_1_sk ++ fused_2;

/// `s` with a 0 sentinel (a kernel text the API takes as a C string), at comptime
pub fn sentinel(comptime s: []const u8) [:0]const u8 {
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
const body: [:0]const u8 = sentinel(text_fused ++ undef);
const body_sk: [:0]const u8 = sentinel(text_fused_sk ++ undef);

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
    // MLX's own regular tile on Apple M5 Max, 128 GB (matmul.cpp's 's' routing: bm 64, bn 128, wm 2, wn 4, swizzle 2;
    // bk 256 below K 8192), its bk 512 neighbour, and a 32-row tile of 128-column blocks for the narrow outputs (Hy4's
    // kv_a, 640)
    .{ .bm = 64, .bn = 128, .bk = 256, .wm = 2, .wn = 4, .swz = 2 },
    .{ .bm = 64, .bn = 128, .bk = 512, .wm = 2, .wn = 4, .swz = 2 },
    .{ .bm = 32, .bn = 128, .bk = 256, .wm = 1, .wn = 4, .swz = 2 },
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

/// Split K in registers (the host's switch, with `mode` on; set at load, before the first prompt): the tabled shapes MLX
/// runs on its NAX split-K kernel fused too, its partitions in registers (`SplitK`)
pub const split_k = Global(bool, "pgemm_split_k", false);

/// Whether MLX runs [m, k] x [k, n] on its regular fused kernel (not the NAX split-K: K >= 3 max(M, N), or max(M, N) <=
/// 1024 and K > 2 max(M, N))
pub fn mlxRegular(m: u32, n: u32, k: u32) bool {
    const mx = @max(m, n);
    return !(k >= 3 * mx or (mx <= 1024 and k > 2 * mx));
}

/// MLX's NAX split-K partitions of K (matmul.cpp `steel_gemm_splitk_axpby_nax`): `parts` partitions of `size` k, the
/// last up to K
pub const SplitK = struct { parts: u32, size: u32 };

/// MLX's partitions for [m, k] x [k, n] where it splits K and its partition size is the same on every device: null where
/// MLX runs its regular kernel, and where M and N are both at most 2048 (its Ultra routing, the architecture's 'd', then
/// takes 2048 k a partition). Elsewhere max(M, N) > 2048, so MLX splits K only from K >= 3 max(M, N) > 6144, where its
/// size is 4096 on every device (its smaller sizes start at K <= 4096): ceil(K / 4096) partitions.
pub fn splitKAt(m: u32, n: u32, k: u32) ?SplitK {
    if (mlxRegular(m, n, k)) return null;
    if (m <= 2048 and n <= 2048) return null;
    return .{ .parts = (k + 4095) / 4096, .size = 4096 };
}

/// What a fused launch runs for xp half [M, K] by W half [K, N]: the table's tile, and MLX's partitions where MLX splits
/// K (null: its regular kernel)
pub const Plan = struct { tile: Tile, sk: ?SplitK };

/// The plan at (M, K, N) on the NAX path: the table's tile where MLX runs its regular kernel, and with `split_k` where
/// it splits K at a device-independent partition size (`splitKAt`); null elsewhere (the chain)
pub fn planAt(m: u32, k: u32, n: u32) ?Plan {
    if (!config.get().nax or m < 1) return null;
    const t = tileOf(k, n) orelse return null;
    if (mlxRegular(m, n, k)) return .{ .tile = t, .sk = null };
    if (!split_k.get().*) return null;
    return .{ .tile = t, .sk = splitKAt(m, n, k) orelse return null };
}

/// `planAt` for the operands: xp half [M, K] and W half [K, N]; null for any other shape or dtype
pub fn planFor(xp: A, w: A) ?Plan {
    const xs = mlx.getShape(xp);
    const wsh = mlx.getShape(w);
    if (xs.len != 2 or wsh.len != 2 or xs[0] < 1 or xs[1] < 1 or wsh[1] < 1 or xs[1] != wsh[0]) return null;
    if (mlx.mlx_array_dtype(xp) != .float16 or mlx.mlx_array_dtype(w) != .float16) return null;
    return planAt(@intCast(xs[0]), @intCast(xs[1]), @intCast(wsh[1]));
}

/// The table's tile for xp half [M, K] by W half [K, N] where MLX runs its regular fused kernel on the NAX path; null
/// elsewhere (the chain)
pub fn tileFor(xp: A, w: A) ?Tile {
    const p = planFor(xp, w) orelse return null;
    return if (p.sk == null) p.tile else null;
}

/// The tile a (K, N): the host's table first, then `table`; null where neither holds an admitted tile for it
pub fn tileOf(k: u32, n: u32) ?Tile {
    if (host_table.get().*) |ht| for (ht) |e| if (e.k == k and e.n == n) return if (e.tile.admitted() and n % e.tile.bn == 0) e.tile else null;
    for (table) |e| if (e.k == k and e.n == n) return if (e.tile.admitted() and n % e.tile.bn == 0) e.tile else null;
    return null;
}

const kernel_cache = Global(?mlx.mlx_fast_metal_kernel, "pgemm_kernel", null);
const kernel_sk_cache = Global(?mlx.mlx_fast_metal_kernel, "pgemm_kernel_sk", null);
const kernel_lock = Global(std.atomic.Value(u32), "pgemm_kernel_lock", std.atomic.Value(u32).init(0));

/// The fused kernel, MLX's regular GEMM (`sk` false) or its partitions in registers (`sk` true), made at its first call
fn kernelGet(sk: bool) !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    const slot = if (sk) kernel_sk_cache.get() else kernel_cache.get();
    if (slot.*) |k| return k;
    const k = (if (sk)
        ks.newKernel("turboexl3_pgemm_fused_sk", &fused_in, &fused_out, body_sk, header, true)
    else
        ks.newKernel("turboexl3_pgemm_fused", &fused_in, &fused_out, body, header, true)) orelse return error.KernelCreateFailed;
    slot.* = k;
    return k;
}

/// The most template arguments a caller's epilogue adds (`launchWith`)
pub const max_extra = 8;

/// One launch of `kern`, a pg2 body (pg2's GEMM, then the finish's statements or a caller's store loop in its place) on
/// `ins` = xp half [M, K], W half [K, N], then the epilogue's operands: y [M, N] of `out_dtype` at plan `p`'s tile (and
/// its split K: the body must be `fused_1_sk`'s then), MLX's grid (tiles_n << swz by ceil(tiles_m >> swz)
/// threadgroups) in MLX's threadgroup shape (32, WN, WM); the template arguments are the tile's, then SKP and SKS where
/// `p` splits K, then `extra` (the epilogue's). The caller owns y.
pub fn launchWith(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, ins: []const A, out_dtype: mlx.mlx_dtype, p: Plan, extra: []const ks.TemplateArg) !A {
    if (ins.len < 2 or extra.len > max_extra) return error.BadExl3Shape;
    const t = p.tile;
    const xs = mlx.getShape(ins[0]);
    const m: u32 = @intCast(xs[0]);
    const kd: u32 = @intCast(xs[1]);
    const n: u32 = @intCast(mlx.getShape(ins[1])[1]);
    const tile: u32 = @as(u32, 1) << @intCast(t.swz);
    const tiles_m = (m + t.bm - 1) / t.bm;
    const tiles_n = n / t.bn;
    var targs: [12 + max_extra]ks.TemplateArg = undefined;
    const base = [_]ks.TemplateArg{
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
    };
    @memcpy(targs[0..base.len], &base);
    var nt: usize = base.len;
    if (p.sk) |sk| {
        targs[nt] = .{ .name = "SKP", .value = .{ .int = @intCast(sk.parts) } };
        targs[nt + 1] = .{ .name = "SKS", .value = .{ .int = @intCast(sk.size) } };
        nt += 2;
    }
    @memcpy(targs[nt..][0..extra.len], extra);
    nt += extra.len;
    const cfg = try ks.buildConfig(.{
        .grid = .{ tiles_n * tile * 32, ((tiles_m + tile - 1) / tile) * t.wn, t.wm },
        .threadgroup = .{ 32, t.wn, t.wm },
        .out_shapes = &.{&[_]c_int{ @intCast(m), @intCast(n) }},
        .out_dtypes = &.{out_dtype},
        .template = targs[0..nt],
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [1]A = undefined;
    try ks.applyConfig(kern, cfg, ins, &o, s);
    _ = calls.get().*.fetchAdd(1, .monotonic);
    _ = elems.get().*.fetchAdd(@as(u64, m) * n, .monotonic);
    return o[0];
}

/// The fused launch: y [M, N] of `out_dtype` = finish(xp [M, K] W [K, N]) at plan `p` (its tile, and MLX's partitions
/// in registers where it splits K); the caller owns y
pub fn launch(s: mlx.mlx_stream, xp: A, w: A, svh: A, out_dtype: mlx.mlx_dtype, p: Plan) !A {
    const kern = try kernelGet(p.sk != null);
    return launchWith(s, kern, &.{ xp, w, svh }, out_dtype, p, &.{.{ .name = "OUT_T", .value = .{ .dtype = out_dtype } }});
}

/// On, at a tabled shape: the fused launch's y (the caller owns it); null elsewhere (the caller runs the chain)
pub fn fusedOn(s: mlx.mlx_stream, xp: A, w: A, svh: A, out_dtype: mlx.mlx_dtype) !?A {
    if (mode.get().* != .on) return null;
    const p = planFor(xp, w) orelse return null;
    return try launch(s, xp, w, svh, out_dtype, p);
}

/// Check, at a tabled shape: the fused launch beside the chain's `want` (served), both handed to the host's hook
pub fn check(s: mlx.mlx_stream, xp: A, w: A, svh: A, out_dtype: mlx.mlx_dtype, want: A) void {
    if (mode.get().* != .check) return;
    const hook = check_hook.get().* orelse return;
    const p = planFor(xp, w) orelse return;
    const got = launch(s, xp, w, svh, out_dtype, p) catch return;
    defer _ = mlx.mlx_array_free(got);
    hook(s, want, got);
}

/// At load (on or check): every tabled tile's kernel (the host's table's and `table`'s, or `only`'s (K, N)) compiled for
/// `out_dtype` (M aligned and not, N and K as the entry's), on small zero operands, evaluated here, so a prompt's first
/// call pays no JIT; with `split_k`, an entry MLX splits K at at those rows also its split kernel, at the entry's K
pub fn warm(s: mlx.mlx_stream, out_dtype: mlx.mlx_dtype, only: ?[2]u32) !void {
    if (mode.get().* == .off or !config.get().nax) return;
    const hosts: []const Entry = host_table.get().* orelse &.{};
    for ([_][]const Entry{ hosts, &table }) |entries| for (entries) |e| {
        const t = e.tile;
        if (!t.admitted()) continue;
        if (only) |kn| if (kn[0] != e.k or kn[1] != e.n) continue;
        for ([_]u32{ t.bm, t.bm + 1 }) |m| {
            const sk: ?SplitK = if (split_k.get().*) splitKAt(m, e.n, e.k) else null;
            const kd: u32 = if (sk != null) e.k else if (e.k % t.bk == 0) t.bk else t.bk + 32;
            var z: [3]A = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
            defer for (z) |a| {
                _ = mlx.mlx_array_free(a);
            };
            try mlx.check(mlx.mlx_zeros(&z[0], &[_]c_int{ @intCast(m), @intCast(kd) }, 2, .float16, s));
            try mlx.check(mlx.mlx_zeros(&z[1], &[_]c_int{ @intCast(kd), @intCast(t.bn) }, 2, .float16, s));
            try mlx.check(mlx.mlx_zeros(&z[2], &[_]c_int{@intCast(t.bn)}, 1, .float16, s));
            const y = try launch(s, z[0], z[1], z[2], out_dtype, .{ .tile = t, .sk = sk });
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

test "pgemm pg2arms: the pieces join to pg2's body, split K is pg2's body with MLX's partitions in the GEMM's place, its rule MLX's at device-independent sizes, a plan only at a tabled shape" {
    try testing.expectEqualStrings(fused_1_6_1 ++ fused_1_6_2 ++ fused_1_6_3, fused_1_6);
    try testing.expectEqualStrings(fused_2_2_1 ++ fused_2_2_2, fused_2_2);
    try testing.expectEqualStrings(text_fused ++ undef, body);
    try testing.expectEqualStrings(fused_1_1 ++ fused_1_2 ++ fused_1_3 ++ fused_1_4 ++ fused_1_5 ++ fused_1_6_1 ++ text_splitk ++ fused_1_6_3 ++ fused_2 ++ undef, body_sk);
    const pin = comptime @import("test_support.zig").pins(@embedFile("kernels/test/pins/pgemm_splitk.metal"));
    // the partitions' loop is MLX's gemm_loop call at the partition's k (its own count and alignment), added in order
    for (pin[0..3]) |line| try testing.expectEqual(@as(usize, 1), std.mem.count(u8, text_splitk, line));
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, text_splitk, pin[3]));
    // the store loop is fused_2_2's first piece alone; the close is the row loop's
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, fused_2_2_1, pin[4]));
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, fused_2_2_2, pin[5]));
    // Hy4-flash's split shapes at prompt rows: o_proj (16384 -> 3072) and the dense MLP's down (13824 -> 3072), four
    // partitions of 4096 (the down's last 1536); q_a (3072 -> 1024) at 1024 rows splits on a device-dependent size
    for ([_]u32{ 33, 1024, 3125, 3584 }) |m| {
        try testing.expectEqual(SplitK{ .parts = 4, .size = 4096 }, splitKAt(m, 3072, 16384).?);
        try testing.expectEqual(SplitK{ .parts = 4, .size = 4096 }, splitKAt(m, 3072, 13824).?);
        try testing.expect(splitKAt(m, 3072, 8192) == null);
    }
    try testing.expect(!mlxRegular(1024, 1024, 3072) and splitKAt(1024, 1024, 3072) == null);
    try testing.expect(splitKAt(5462, 3072, 16384) == null);
    // a plan: the host's tile at a tabled (K, N) only; a split shape only with split_k
    const saved_table = host_table.get().*;
    const saved_split = split_k.get().*;
    defer {
        host_table.get().* = saved_table;
        split_k.get().* = saved_split;
    }
    const t: Tile = .{ .bm = 32, .bn = 256, .bk = 512, .wm = 1, .wn = 4, .swz = 2 };
    const entries = [_]Entry{ .{ .k = 16384, .n = 3072, .tile = t }, .{ .k = 1024, .n = 16384, .tile = t } };
    host_table.get().* = &entries;
    split_k.get().* = false;
    try testing.expect(planAt(3584, 16384, 3072) == null);
    try testing.expectEqual(t, planAt(3584, 1024, 16384).?.tile);
    try testing.expect(planAt(3584, 1024, 16384).?.sk == null);
    try testing.expect(planAt(3584, 1024, 8192) == null);
    split_k.get().* = true;
    try testing.expectEqual(SplitK{ .parts = 4, .size = 4096 }, planAt(3584, 16384, 3072).?.sk.?);
}
