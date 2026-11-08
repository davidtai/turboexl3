//! Ling-3.1-flash's MoE router at prompt widths in one launch, the host's words. At a prompt span's rows the host
//! (turboexl3_ling) runs Linear's matmul of the MoE input x [1, S, 4096] bf16 and the router's f32 table (with
//! MLX_SERVE_LING_ROUTER_F16=1 the table rebuilt for the call from the f16 weight, `router.tableF32`): MLX casts x to
//! f32 and runs its fused NAX GEMM (TF32 on the NAX) over the table's column-major memory, so a call writes and reads
//! x's f32 copy, and the rebuild its table. `logits` runs that GEMM alone: MLX's kernel (steel_gemm_fused_nax.h `gemm`
//! at nt, f32 tiles and out) with its gemm_loop's tiles loaded from x's bf16 words and w's own words, each widened in
//! the load, w's -0.0 stored +0.0 as the table's + 0 stores it (kernels/router_prompt/), at the tile MLX picks for the
//! call (`mlxTile`). The tiles hold the f32 values MLX's hold and run the same MMAs in the same k order, so y is the
//! served logits word for word. Taken only where MLX runs that regular kernel on the NAX path; elsewhere null.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ks = sdk.kernel_launch;
const config = @import("../../core/config.zig");
const Global = @import("../../core/global.zig").Global;
const pgemm = @import("../../core/pgemm.zig");
const steel_nax = @import("../../core/steel.zig").nax;

const A = mlx.mlx_array;

/// MLX's gemm_loop with each tile widened from its operand's type (the header's tail: the kernel attribute's macro)
pub const text_loop = @embedFile("kernels/router_prompt/loop.metal");
/// MLX's fused NAX kernel body at nt over x bf16 and w as loaded, y f32 (its last line undefines the macro)
pub const text_gemm = @embedFile("kernels/router_prompt/gemm.metal");

fn sentinel(comptime s: []const u8) [:0]const u8 {
    comptime {
        var buf: [s.len:0]u8 = undefined;
        @memcpy(buf[0..s.len], s);
        const final = buf;
        return &final;
    }
}

const header: [:0]const u8 = sentinel(steel_nax ++ text_loop);
const body: [:0]const u8 = sentinel(text_gemm);

const gemm_in = [_][*:0]const u8{ "x", "w" };
const gemm_out = [_][*:0]const u8{"y"};

/// One tile: MLX's BM x BN, WM x WN simdgroups, BK, swizzle_log
pub const Tile = struct {
    bm: u16,
    bn: u16,
    bk: u16,
    wm: u8,
    wn: u8,
    swz: u8,
};

/// The tile MLX's regular fused NAX kernel takes for [m, k] x [k, n] on a GPU whose architecture name ends in `devc`
/// (matmul.cpp `steel_matmul_regular_axpby_nax`, MLX 0.32): 's' and 'c' 64 x 128 at WM 2, WN 4, BK 64 where K >= 8192
/// and K > M + N, else 256, swizzle 2; 'd' 128 x 64 (WM 4, WN 2) where N > M, else 64 x 128 (WM 2, WN 4), BK 512,
/// swizzle 0 where M > N, else 2; others 128 x 128 at WM 4, WN 4, BK 512, swizzle 0 up to 3 row tiles, else 1.
pub fn mlxTile(devc: u8, m: u32, n: u32, k: u32) Tile {
    var t: Tile = .{ .bm = 128, .bn = 128, .bk = 512, .wm = 4, .wn = 4, .swz = 0 };
    if (devc == 's' or devc == 'c') {
        t.bk = if (k >= 8192 and k > m + n) 64 else 256;
        t.bm = 64;
        t.wm = 2;
    } else if (devc == 'd') {
        if (n > m) {
            t.bn = 64;
            t.wn = 2;
        } else {
            t.bm = 64;
            t.wm = 2;
        }
    }
    const tiles_m = (m + t.bm - 1) / t.bm;
    t.swz = if (tiles_m <= 3) 0 else 1;
    if (devc == 's' or devc == 'c') {
        t.swz = 2;
    } else if (devc == 'd') {
        t.swz = if (m > n) 0 else 2;
    }
    return t;
}

/// MLX's NAX gate for an f32 matmul: MLX_ENABLE_TF32 unset or a nonzero integer (MLX's atoi of the variable, default 1)
pub fn tf32From(v: ?[]const u8) bool {
    const s = v orelse return true;
    var i: usize = 0;
    while (i < s.len and (s[i] == ' ' or s[i] == '\t' or s[i] == '\n')) i += 1;
    if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
    var any = false;
    while (i < s.len and s[i] >= '0' and s[i] <= '9') : (i += 1) {
        if (s[i] != '0') any = true;
    }
    return any;
}

fn tf32() bool {
    return tf32From(if (std.c.getenv("MLX_ENABLE_TF32")) |p| std.mem.span(p) else null);
}

/// The default device's architecture name's last letter (MLX's `devc`), read once; 1 = unknown
const arch_cache = Global(u8, "ling_router_prompt_arch", 0);

/// libmlxc's device info (mlx-c device.h): the consumer's bindings where its `sdk.mlx` declares them (the host's), else
/// libmlxc's own symbols
const device_c = struct {
    const Info = if (@hasDecl(mlx, "mlx_device_info")) mlx.mlx_device_info else extern struct { ctx: ?*anyopaque = null };
    extern "c" fn mlx_get_default_device(dev: *mlx.mlx_device) c_int;
    extern "c" fn mlx_device_info_new() Info;
    extern "c" fn mlx_device_info_get(info: *Info, dev: mlx.mlx_device) c_int;
    extern "c" fn mlx_device_info_get_string(value: *[*:0]const u8, info: Info, key: [*:0]const u8) c_int;
    extern "c" fn mlx_device_info_free(info: Info) c_int;
};
const getDefaultDevice = if (@hasDecl(mlx, "mlx_get_default_device")) mlx.mlx_get_default_device else device_c.mlx_get_default_device;
const infoNew = if (@hasDecl(mlx, "mlx_device_info_new")) mlx.mlx_device_info_new else device_c.mlx_device_info_new;
const infoGet = if (@hasDecl(mlx, "mlx_device_info_get")) mlx.mlx_device_info_get else device_c.mlx_device_info_get;
const infoGetString = if (@hasDecl(mlx, "mlx_device_info_get_string")) mlx.mlx_device_info_get_string else device_c.mlx_device_info_get_string;
const infoFree = if (@hasDecl(mlx, "mlx_device_info_free")) mlx.mlx_device_info_free else device_c.mlx_device_info_free;

/// The device handle is the process's default device's, read once and kept (as the host's own `gpuArchitecture`)
fn readArchClass() u8 {
    var dev = mlx.mlx_device{ .ctx = null };
    if (getDefaultDevice(&dev) != 0) return 1;
    var info = infoNew();
    defer _ = infoFree(info);
    if (infoGet(&info, dev) != 0) return 1;
    var cstr: [*:0]const u8 = undefined;
    if (infoGetString(&cstr, info, "architecture") != 0) return 1;
    const arch = std.mem.span(cstr);
    if (arch.len == 0 or arch[arch.len - 1] < 'a' or arch[arch.len - 1] > 'z') return 1;
    return arch[arch.len - 1];
}

/// The architecture name's last letter `takes` read, or null where none was read (for the tests' lines)
pub fn archLetter() ?u8 {
    return archClass();
}

fn archClass() ?u8 {
    const c = arch_cache.get();
    if (c.* == 0) c.* = readArchClass();
    return if (c.* == 1) null else c.*;
}

const kernel_cache = Global(?mlx.mlx_fast_metal_kernel, "ling_router_prompt_kernel", null);
const kernel_lock = Global(std.atomic.Value(u32), "ling_router_prompt_lock", std.atomic.Value(u32).init(0));

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ arch_cache, kernel_cache, kernel_lock };

fn kernelGet() !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    if (kernel_cache.get().*) |k| return k;
    const k = ks.newKernel("turboexl3_ling_router_prompt", &gemm_in, &gemm_out, body, header, true) orelse return error.KernelCreateFailed;
    kernel_cache.get().* = k;
    return k;
}

/// A call `logits` takes: x [.., S, K] bf16 (S rows, every axis before K), w f16 or f32 [K, N] with strides (1, K) (the
/// memory of [N, K] row-major: the loader's view of the router weight, or its column-major table), on a NAX device
/// with MLX_ENABLE_TF32 on, where MLX's matmul runs its regular fused kernel (two rows or more, no split K); the call's
/// rows, N, K and MLX's tile
pub const Call = struct { m: u32, n: u32, k: u32, tile: Tile };

pub fn takes(x: A, w: A) ?Call {
    if (!config.get().nax or !tf32()) return null;
    const devc = archClass() orelse return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const wd = mlx.mlx_array_dtype(w);
    if (wd != .float16 and wd != .float32) return null;
    const xs = mlx.getShape(x);
    const wsh = mlx.getShape(w);
    if (xs.len < 2 or wsh.len != 2) return null;
    const kx = xs[xs.len - 1];
    if (kx < 1 or wsh[0] != kx or wsh[1] < 1) return null;
    const wst = mlx.mlx_array_strides(w);
    if (wst[0] != 1 or wst[1] != @as(usize, @intCast(kx))) return null;
    var rows: u64 = 1;
    for (xs[0 .. xs.len - 1]) |d| rows *= @intCast(d);
    if (rows < 2 or rows > std.math.maxInt(i32)) return null;
    const m: u32 = @intCast(rows);
    const n: u32 = @intCast(wsh[1]);
    const k: u32 = @intCast(kx);
    if (!pgemm.mlxRegular(m, n, k)) return null;
    return .{ .m = m, .n = n, .k = k, .tile = mlxTile(devc, m, n, k) };
}

/// The launch: y f32 [M, N] = astype(x, f32) times w's f32 table at `c`'s tile, MLX's grid (tiles_n << swz by
/// ceil(tiles_m >> swz) threadgroups) in MLX's threadgroup shape (32, WN, WM); the caller owns y
fn launch(s: mlx.mlx_stream, x2: A, wt: A, c: Call) !A {
    const kern = try kernelGet();
    const t = c.tile;
    const tile: u32 = @as(u32, 1) << @intCast(t.swz);
    const tiles_m = (c.m + t.bm - 1) / t.bm;
    const tiles_n = (c.n + t.bn - 1) / t.bn;
    const cfg = try ks.buildConfig(.{
        .grid = .{ tiles_n * tile * 32, ((tiles_m + tile - 1) / tile) * t.wn, t.wm },
        .threadgroup = .{ 32, t.wn, t.wm },
        .out_shapes = &.{&[_]c_int{ @intCast(c.m), @intCast(c.n) }},
        .out_dtypes = &.{.float32},
        .template = &.{
            .{ .name = "BM", .value = .{ .int = t.bm } },
            .{ .name = "BN", .value = .{ .int = t.bn } },
            .{ .name = "BK", .value = .{ .int = t.bk } },
            .{ .name = "WM", .value = .{ .int = t.wm } },
            .{ .name = "WN", .value = .{ .int = t.wn } },
            .{ .name = "SWZ", .value = .{ .int = t.swz } },
            .{ .name = "AM", .value = .{ .int = @intFromBool(c.m % t.bm == 0) } },
            .{ .name = "AN", .value = .{ .int = @intFromBool(c.n % t.bn == 0) } },
            .{ .name = "AK", .value = .{ .int = @intFromBool(c.k % t.bk == 0) } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [1]A = undefined;
    try ks.applyConfig(kern, cfg, &.{ x2, wt }, &o, s);
    return o[0];
}

/// The router's logits at a prompt span's rows: y f32 [.., S, N] (x's leading shape), the served matmul's words (MLX's
/// astype of x to f32, then its fused NAX GEMM over the f32 table: w's words widened, then + 0). Null where `takes`
/// declines (the caller runs its chain). The caller owns y.
pub fn logits(s: mlx.mlx_stream, x: A, w: A) !?A {
    const c = takes(x, w) orelse return null;
    const xs = mlx.getShape(x);
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ @intCast(c.m), @intCast(c.k) }, 2, s));
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, w, s));
    const y2 = try launch(s, x2, wt, c);
    defer _ = mlx.mlx_array_free(y2);
    var shape: [8]c_int = undefined;
    if (xs.len > shape.len) return error.BadExl3Shape;
    for (xs[0 .. xs.len - 1], 0..) |d, i| shape[i] = d;
    shape[xs.len - 1] = @intCast(c.n);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &shape, xs.len, s));
    return y;
}

const testing = std.testing;

test "turboexl3 ling router_prompt: MLX's tile per architecture class, MLX's TF32 gate, and the regular kernel only where MLX runs it" {
    // the M5 Max ('s'): 64 x 128 at WM 2, WN 4, BK 256 for the router (K 4096 < 8192), swizzle 2
    try testing.expectEqual(Tile{ .bm = 64, .bn = 128, .bk = 256, .wm = 2, .wn = 4, .swz = 2 }, mlxTile('s', 4096, 512, 4096));
    try testing.expectEqual(Tile{ .bm = 64, .bn = 128, .bk = 64, .wm = 2, .wn = 4, .swz = 2 }, mlxTile('c', 2048, 512, 8192));
    try testing.expectEqual(Tile{ .bm = 64, .bn = 128, .bk = 256, .wm = 2, .wn = 4, .swz = 2 }, mlxTile('s', 8192, 512, 8192));
    try testing.expectEqual(Tile{ .bm = 128, .bn = 64, .bk = 512, .wm = 4, .wn = 2, .swz = 2 }, mlxTile('d', 256, 512, 4096));
    try testing.expectEqual(Tile{ .bm = 64, .bn = 128, .bk = 512, .wm = 2, .wn = 4, .swz = 0 }, mlxTile('d', 4096, 512, 4096));
    try testing.expectEqual(Tile{ .bm = 128, .bn = 128, .bk = 512, .wm = 4, .wn = 4, .swz = 1 }, mlxTile('g', 4096, 512, 4096));
    try testing.expectEqual(Tile{ .bm = 128, .bn = 128, .bk = 512, .wm = 4, .wn = 4, .swz = 0 }, mlxTile('g', 384, 512, 4096));
    try testing.expect(tf32From(null) and tf32From("1") and tf32From(" 2x") and tf32From("-1"));
    try testing.expect(!tf32From("0") and !tf32From("") and !tf32From("true") and !tf32From("00"));
    // a prompt chunk and a span of 1366 rows or more take the regular kernel; 1365 and fewer MLX's NAX split K
    try testing.expect(pgemm.mlxRegular(4096, 512, 4096) and pgemm.mlxRegular(1996, 512, 4096) and pgemm.mlxRegular(1366, 512, 4096));
    try testing.expect(!pgemm.mlxRegular(1365, 512, 4096) and !pgemm.mlxRegular(31, 512, 4096));
}
