//! The HIP launches of turboexl3's kernels, from a shape alone: kernel symbol, grid, block, LDS bytes and arguments by
//! role. Pure Zig with no runtime: turboexl3's tests bind the roles to their buffers (hip_tests.zig), and a host with its
//! own HIP runtime (TensorFold) binds them to its own. The partition rules are dense.zig's (`Dense.ksFor`, `btFor`,
//! `rbCap`, `chainRb`), so a row's sums do not depend on the rows beside it.
const std = @import("std");
const format = @import("../format.zig");

/// A buffer a launch reads or writes, bound by the host.
pub const Role = enum {
    x,
    suh,
    svh,
    trellis,
    xp,
    xs,
    z,
    y,
    // the routed experts: the routing and the bank
    slots,
    scores,
    gate_t,
    up_t,
    down_t,
    gate_suh,
    up_suh,
    gate_svh,
    up_svh,
    down_suh,
    down_svh,
    // the decode chain's scratch
    xg,
    xu,
    sg,
    su,
    zg,
    zu,
    hd,
    hs,
    zd,
    // the prefill chain's route
    order,
    inv,
    items,
    counts,
};

pub const Arg = union(enum) { buf: Role, u: u32, i: i32, f: f32 };

pub const max_args = 20;

pub const Launch = struct {
    /// the kernel's name in its group's code object
    symbol: [:0]const u8,
    /// the group that holds it (flags.zig groups' `name`)
    group: []const u8,
    grid: [3]u32,
    block: u32,
    shared: u32 = 0,
    args: [max_args]Arg = undefined,
    nargs: usize = 0,

    fn add(l: *Launch, a: Arg) void {
        l.args[l.nargs] = a;
        l.nargs += 1;
    }

    pub fn argList(l: *const Launch) []const Arg {
        return l.args[0..l.nargs];
    }
};

/// The kernels' activation numbering (common.hpp Dtype).
pub const Dtype = enum(u2) { f32 = 0, f16 = 1, bf16 = 2 };

fn dtypeName(d: Dtype) []const u8 {
    return switch (d) {
        .f32 => "f32",
        .f16 => "f16",
        .bf16 => "bf16",
    };
}

// ---- dense projections (docs/KERNELS.md section 4) ----------------------------------------------------------------

/// The GEMV text's most rows a chunk, and the lean text's (dense.zig `Dense.max_rb`, `lean_rb`).
pub const max_rb: u32 = 5;
pub const lean_rb: u32 = 8;

pub fn rbCap(rows: u32, cap: u32) u32 {
    const chunks = (rows + cap - 1) / cap;
    return (rows + chunks - 1) / chunks;
}

/// output tiles a threadgroup for RB rows (dense.hip bt_for)
pub fn btFor(rb: u32) u32 {
    return if (rb == 1) 4 else if (rb <= 4) 2 else 1;
}

/// the smallest K split that puts 160 threadgroups in flight at one row (BT 4); fixed per shape (dense.zig ksFor)
pub fn ksFor(in: u32, out: u32) u32 {
    const q = (in / 16) / 4;
    const groups = (out / 16) / 4;
    var d: u32 = 1;
    while (d <= q) : (d += 1) {
        if (q % d != 0) continue;
        if (groups * d >= 160) return d;
    }
    return q;
}

pub const Dense = struct {
    in: u32,
    out: u32,
    rows: u32,
    rate: format.Rate,
    dec: format.Decode,
    dtype: Dtype,
    /// MUL1's deferred map (dense.zig deferFor: on for MUL1; the tests also run it off)
    defer_map: bool,
    /// chunks of 6-8 rows on the lean text (Config.dense_lean)
    lean: bool = true,

    pub fn check(d: Dense) error{BadExl3Shape}!void {
        if (d.in == 0 or d.out == 0 or d.in % 128 != 0 or d.out % 128 != 0 or (d.in / 16) % 4 != 0) return error.BadExl3Shape;
        if (d.rows == 0 or d.rows > 16) return error.BadExl3Shape;
        if (d.defer_map and d.dec.codebook != .mul1) return error.BadExl3Shape;
    }

    pub fn ks(d: Dense) u32 {
        return ksFor(d.in, d.out);
    }

    pub fn rb(d: Dense) u32 {
        return rbCap(d.rows, if (d.lean) lean_rb else max_rb);
    }

    pub fn xpBytes(d: Dense) usize {
        return @as(usize, d.rows) * d.in * 2;
    }
    pub fn xsBytes(d: Dense) usize {
        return @as(usize, d.rows) * (d.in / 128) * 4;
    }
    pub fn zBytes(d: Dense) usize {
        return @as(usize, d.ks()) * d.rows * d.out * 4;
    }
};

const dgemv_names = blk: {
    var t: [3][lean_rb][:0]const u8 = undefined;
    for ([_][]const u8{ "mcg", "mul1", "mul1d" }, 0..) |cb, i| {
        for (0..lean_rb) |r| t[i][r] = std.fmt.comptimePrint("t3_dense_dgemv_{s}_{d}", .{ cb, r + 1 });
    }
    break :blk t;
};

fn cbIndex(dec: format.Decode, defer_map: bool) usize {
    return if (dec.codebook == .mcg) 0 else if (defer_map) 2 else 1;
}

/// The three-launch chain prep_s -> dgemv -> dfinish (rows 1-16).
pub fn denseChain(d: Dense) error{BadExl3Shape}![3]Launch {
    try d.check();
    const rb = d.rb();
    // a chunk's text: dgemv up to max_rb rows, the lean text past it (its kernels are named by their rows)
    if (rb > max_rb and !d.lean) return error.BadExl3Shape;
    const chunks = (d.rows + rb - 1) / rb;
    const ks = d.ks();
    var prep: Launch = .{ .symbol = switch (d.dtype) {
        .f32 => "t3_dense_prep_s_f32",
        .f16 => "t3_dense_prep_s_f16",
        .bf16 => "t3_dense_prep_s_bf16",
    }, .group = "dense", .grid = .{ d.in / 128, d.rows, 1 }, .block = 32 };
    prep.add(.{ .buf = .x });
    prep.add(.{ .buf = .suh });
    prep.add(.{ .buf = .xp });
    prep.add(.{ .buf = .xs });
    prep.add(.{ .u = d.in });
    prep.add(.{ .i = 1 });
    var gemv: Launch = .{ .symbol = dgemv_names[cbIndex(d.dec, d.defer_map)][rb - 1], .group = "dense", .grid = .{ d.out / 16 / btFor(rb), chunks, ks }, .block = 128 };
    gemv.add(.{ .buf = .trellis });
    gemv.add(.{ .buf = .xp });
    gemv.add(.{ .buf = .z });
    gemv.add(.{ .u = d.in });
    gemv.add(.{ .u = d.out });
    gemv.add(.{ .u = d.rows });
    gemv.add(.{ .u = d.rate.n });
    gemv.add(.{ .u = d.dec.window.mask() });
    var fin: Launch = .{ .symbol = switch (d.dtype) {
        .f32 => "t3_dense_dfinish_f32",
        .f16 => "t3_dense_dfinish_f16",
        .bf16 => "t3_dense_dfinish_bf16",
    }, .group = "dense", .grid = .{ d.out / 128, d.rows, 1 }, .block = 32 };
    fin.add(.{ .buf = .z });
    fin.add(.{ .buf = .xs });
    fin.add(.{ .buf = .svh });
    fin.add(.{ .buf = .y });
    fin.add(.{ .u = d.out });
    fin.add(.{ .u = d.rows });
    fin.add(.{ .u = ks });
    fin.add(.{ .u = d.in / 128 });
    fin.add(.{ .i = @intFromBool(d.defer_map) });
    _ = dtypeName;
    return .{ prep, gemv, fin };
}

test "the dense plan's partition: dense.zig's K split, tiles and chunks" {
    try std.testing.expectEqual(@as(u32, 5), rbCap(5, max_rb));
    try std.testing.expectEqual(@as(u32, 3), rbCap(6, max_rb));
    try std.testing.expectEqual(@as(u32, 8), rbCap(16, lean_rb));
    try std.testing.expectEqual(@as(u32, 6), rbCap(11, lean_rb));
    try std.testing.expectEqual(@as(u32, 4), btFor(1));
    try std.testing.expectEqual(@as(u32, 1), btFor(5));
    // 2560 x 640: KS divides 40 and 10 groups of 4 output tiles take KS 20 to reach 160 threadgroups (dense.zig's 20)
    try std.testing.expectEqual(@as(u32, 20), ksFor(2560, 640));
}

// ---- routed experts at decode rows (docs/KERNELS.md section 1) ------------------------------------------------------

/// The rounding profile (decode.hip PROF): format.project's, or GLM's (f32 z, GLM's bf16 SwiGLU and clamp, per-slot f32
/// outputs with no score; slots >= experts skipped).
pub const Profile = enum { exl3, glm };

/// How the host holds the routed bank's trellises (gate_t, up_t, down_t). `contiguous`: one buffer [E][K/16][N/16][n/2
/// words] a projection. `pointers`: the role binds a u64 [E] table of each expert's own trellis address (a loader that
/// keeps each expert's tensor apart). The bytes are the same.
pub const Bank = enum { contiguous, pointers };

pub const Decode = struct {
    hidden: u32,
    inter: u32,
    topk: u32,
    rows: u32,
    /// routed experts; a slot id >= experts (GLM's shared slot) is skipped under the GLM profile
    experts: u32,
    rate: format.Rate,
    dec: format.Decode,
    dtype: Dtype,
    defer_map: bool,
    profile: Profile = .exl3,
    /// the SwiGLU clamp (GLM profile; 0 none)
    limit: f32 = 0,
    /// the partitions' first K split (decode.zig Kernels3.init's ks_max; served 4)
    ks_max: u32 = 4,
    bank: Bank = .contiguous,

    pub fn check(d: Decode) error{BadExl3Shape}!void {
        if (d.hidden == 0 or d.inter == 0 or d.hidden % 128 != 0 or d.inter % 128 != 0) return error.BadExl3Shape;
        if (d.rows == 0 or d.rows > 16 or d.topk == 0 or d.topk > 32 or d.experts == 0) return error.BadExl3Shape;
        if (d.defer_map and d.dec.codebook != .mul1) return error.BadExl3Shape;
        if (d.profile == .glm and d.dtype != .bf16) return error.BadExl3Shape;
        if (d.ks_max != 1 and d.ks_max != 2 and d.ks_max != 4 and d.ks_max != 8) return error.BadExl3Shape;
    }

    /// gate/up: hidden / 16 tiles in KS x 2 parts (decode.zig Kernels3.initWith)
    pub fn ksGu(d: Decode) u32 {
        var k = d.ks_max;
        while (k > 1 and (d.hidden / 16) % (k * 2) != 0) k /= 2;
        return k;
    }

    /// down: inter / 16 tiles in KS2 x 4 parts of at least 2 tiles
    pub fn ksDn(d: Decode) u32 {
        var k = d.ks_max;
        while (k > 1 and ((d.inter / 16) % (k * 4) != 0 or (d.inter / 16) / (k * 4) < 2)) k /= 2;
        return k;
    }

    pub fn rk(d: Decode) u32 {
        return d.rows * d.topk;
    }

    /// The bytes of each scratch role.
    pub fn bytes(d: Decode, r: Role) usize {
        const rkn: usize = d.rk();
        return switch (r) {
            .xg, .xu => rkn * d.hidden * 2,
            .sg, .su => rkn * (d.hidden / 128) * 4,
            .zg, .zu => @as(usize, d.ksGu()) * rkn * d.inter * 4,
            .hd => rkn * d.inter * 2,
            .hs => rkn * (d.inter / 128) * 4,
            .zd => @as(usize, d.ksDn()) * rkn * d.hidden * 4,
            // y: [rows][hidden] in the dtype; the GLM profile: f32 [RK][hidden]
            .y => if (d.profile == .glm) rkn * d.hidden * 4 else @as(usize, d.rows) * d.hidden * @as(usize, if (d.dtype == .f32) 4 else 2),
            else => 0,
        };
    }
};

const kBT = 4; // decode.hip kBT

fn gemvName(comptime kind: []const u8, dec: format.Decode, defer_map: bool) [:0]const u8 {
    return switch (cbIndex(dec, defer_map)) {
        0 => "t3_dec_" ++ kind ++ "_mcg",
        1 => "t3_dec_" ++ kind ++ "_mul1",
        else => "t3_dec_" ++ kind ++ "_mul1d",
    };
}

/// Form 3: prep2 -> gu3 -> fin3 -> down3 -> dfin3.
pub fn decodeForm3(d: Decode) error{BadExl3Shape}![5]Launch {
    try d.check();
    const rkn = d.rk();
    const glm = d.profile == .glm;
    var prep: Launch = .{ .symbol = if (glm) "t3_dec_prep2_glm_bf16" else switch (d.dtype) {
        .f32 => "t3_dec_prep2_exl3_f32",
        .f16 => "t3_dec_prep2_exl3_f16",
        .bf16 => "t3_dec_prep2_exl3_bf16",
    }, .group = "decode", .grid = .{ d.hidden / 128, rkn, 1 }, .block = 32 };
    for ([_]Role{ .x, .slots, .gate_suh, .up_suh, .xg, .xu, .sg, .su }) |r| prep.add(.{ .buf = r });
    prep.add(.{ .u = d.hidden });
    prep.add(.{ .u = d.topk });
    prep.add(.{ .u = d.experts });
    var gu: Launch = .{ .symbol = gemvName("gu3", d.dec, d.defer_map), .group = "decode", .grid = .{ d.inter / 16 / kBT, rkn, d.ksGu() }, .block = 128 };
    for ([_]Role{ .xg, .xu, .slots, .gate_t, .up_t, .zg, .zu }) |r| gu.add(.{ .buf = r });
    for ([_]u32{ d.hidden, d.inter, rkn, d.experts, d.rate.n, d.dec.window.mask() }) |v| gu.add(.{ .u = v });
    gu.add(.{ .i = @intFromBool(d.bank == .pointers) });
    var fin: Launch = .{ .symbol = if (glm) "t3_dec_fin3_glm" else "t3_dec_fin3_exl3", .group = "decode", .grid = .{ d.inter / 128, rkn, 1 }, .block = 128 };
    for ([_]Role{ .zg, .zu, .slots, .gate_svh, .up_svh, .down_suh, .sg, .su, .hd, .hs }) |r| fin.add(.{ .buf = r });
    for ([_]u32{ d.hidden, d.inter, rkn, d.ksGu(), d.experts }) |v| fin.add(.{ .u = v });
    fin.add(.{ .i = @intFromBool(d.defer_map) });
    fin.add(.{ .f = d.limit });
    var down: Launch = .{ .symbol = gemvName("down3", d.dec, d.defer_map), .group = "decode", .grid = .{ d.hidden / 16 / kBT, rkn, d.ksDn() }, .block = 128 };
    for ([_]Role{ .hd, .slots, .down_t, .zd }) |r| down.add(.{ .buf = r });
    for ([_]u32{ d.hidden, d.inter, rkn, d.experts, d.rate.n, d.dec.window.mask() }) |v| down.add(.{ .u = v });
    down.add(.{ .i = @intFromBool(d.bank == .pointers) });
    var dfin: Launch = .{ .symbol = if (glm) "t3_dec_dfin3_glm_f32" else switch (d.dtype) {
        .f32 => "t3_dec_dfin3_exl3_f32",
        .f16 => "t3_dec_dfin3_exl3_f16",
        .bf16 => "t3_dec_dfin3_exl3_bf16",
    }, .group = "decode", .grid = .{ d.hidden / 128, d.rows, 1 }, .block = 128 };
    for ([_]Role{ .zd, .slots, .scores, .down_svh, .hs, .y }) |r| dfin.add(.{ .buf = r });
    for ([_]u32{ d.hidden, d.inter, d.topk, rkn, d.ksDn(), d.experts }) |v| dfin.add(.{ .u = v });
    dfin.add(.{ .i = @intFromBool(d.defer_map) });
    return .{ prep, gu, fin, down, dfin };
}

/// gu8 takes one row when its threadgroup's K span (both halves of a gate/up slice) is whole 128-blocks.
pub fn gu8Admitted(d: Decode) bool {
    const per = (d.hidden / 16) / (d.ksGu() * 2);
    const span = 2 * per * 16;
    return d.rows == 1 and span % 128 == 0 and 4 * span <= 48 * 1024;
}

/// Form 9 (served): one row gu8 -> fin7 -> down3 -> dfin7 where gu8 is admitted; else prep2 -> gu3 -> fin7 -> down3
/// -> dfin7. The bytes of form 3.
pub fn decodeForm9(d: Decode) error{BadExl3Shape}!struct { launches: [5]Launch, len: usize } {
    const f3 = try decodeForm3(d);
    const glm = d.profile == .glm;
    var fin7 = f3[2];
    fin7.symbol = if (glm) "t3_dec_fin7_glm" else "t3_dec_fin7_exl3";
    fin7.block = 32;
    var dfin7 = f3[4];
    dfin7.symbol = if (glm) "t3_dec_dfin7_glm_f32" else switch (d.dtype) {
        .f32 => "t3_dec_dfin7_exl3_f32",
        .f16 => "t3_dec_dfin7_exl3_f16",
        .bf16 => "t3_dec_dfin7_exl3_bf16",
    };
    dfin7.block = 32 * d.topk;
    if (!gu8Admitted(d)) return .{ .launches = .{ f3[0], f3[1], fin7, f3[3], dfin7 }, .len = 5 };
    const names = [4][3][:0]const u8{
        .{ "t3_dec_gu8_exl3_f32_mcg", "t3_dec_gu8_exl3_f32_mul1", "t3_dec_gu8_exl3_f32_mul1d" },
        .{ "t3_dec_gu8_exl3_f16_mcg", "t3_dec_gu8_exl3_f16_mul1", "t3_dec_gu8_exl3_f16_mul1d" },
        .{ "t3_dec_gu8_exl3_bf16_mcg", "t3_dec_gu8_exl3_bf16_mul1", "t3_dec_gu8_exl3_bf16_mul1d" },
        .{ "t3_dec_gu8_glm_bf16_mcg", "t3_dec_gu8_glm_bf16_mul1", "t3_dec_gu8_glm_bf16_mul1d" },
    };
    const row: usize = if (glm) 3 else @backingInt(d.dtype);
    const per = (d.hidden / 16) / (d.ksGu() * 2);
    var gu8: Launch = .{ .symbol = names[row][cbIndex(d.dec, d.defer_map)], .group = "decode", .grid = f3[1].grid, .block = 128, .shared = 4 * (2 * per * 16) };
    for ([_]Role{ .x, .slots, .gate_suh, .up_suh, .gate_t, .up_t, .zg, .zu, .sg, .su }) |r| gu8.add(.{ .buf = r });
    for ([_]u32{ d.hidden, d.inter, d.topk, d.experts, d.rate.n, d.dec.window.mask() }) |v| gu8.add(.{ .u = v });
    gu8.add(.{ .i = @intFromBool(d.bank == .pointers) });
    return .{ .launches = .{ gu8, fin7, f3[3], dfin7, undefined }, .len = 4 };
}

test "the decode plan's partitions: decode.zig's KS and KS2 at the served ks_max 4" {
    const base: Decode = .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 1, .experts = 512, .rate = format.Rate.fromK(4), .dec = .mcg, .dtype = .bf16, .defer_map = false };
    // KERNELS.md: at 2560 x 640, KS is 4 (20 input tiles a part) and KS2 is 2 (5 tiles a part)
    try std.testing.expectEqual(@as(u32, 4), base.ksGu());
    try std.testing.expectEqual(@as(u32, 2), base.ksDn());
}

// ---- routed experts at prefill rows (docs/KERNELS.md section 3) and dense rows above 16 ---------------------------

/// The GEMM's rows a pass and a route tile (prefill.hip kBM).
pub const gemm_bm: u32 = 64;

fn gemmName(dec: format.Decode, defer_map: bool) [:0]const u8 {
    return switch (cbIndex(dec, defer_map)) {
        0 => "t3_pf_gemm_mcg",
        1 => "t3_pf_gemm_mul1",
        else => "t3_pf_gemm_mul1d",
    };
}

pub const Prefill = struct {
    hidden: u32,
    inter: u32,
    topk: u32,
    rows: u32,
    experts: u32,
    rate: format.Rate,
    dec: format.Decode,
    dtype: Dtype,
    defer_map: bool,
    profile: Profile = .exl3,
    limit: f32 = 0,
    bank: Bank = .contiguous,

    pub fn check(d: Prefill) error{BadExl3Shape}!void {
        if (d.hidden == 0 or d.inter == 0 or d.hidden % 128 != 0 or d.inter % 128 != 0) return error.BadExl3Shape;
        if (d.rows == 0 or d.topk == 0 or d.topk > 32 or d.experts == 0 or d.experts > 1024) return error.BadExl3Shape;
        if (d.defer_map and d.dec.codebook != .mul1) return error.BadExl3Shape;
        if (d.profile == .glm and d.dtype != .bf16) return error.BadExl3Shape;
    }

    /// the position capacity: every pair (a skipped slot leaves its position unused)
    pub fn positions(d: Prefill) u32 {
        return d.rows * d.topk;
    }

    /// item slots: an item a used expert plus one a further 64 positions
    pub fn cap(d: Prefill) u32 {
        const p = d.positions();
        return @min(p, d.experts) + p / gemm_bm;
    }

    pub fn bytes(d: Prefill, r: Role) usize {
        const p: usize = d.positions();
        return switch (r) {
            .order, .inv => p * 4,
            .items => @as(usize, d.cap()) * 12,
            .counts => 8,
            .xg, .xu => p * d.hidden * 2,
            .sg, .su => p * (d.hidden / 128) * 4,
            .zg => 2 * p * d.inter * 4,
            .hd => p * d.inter * 2,
            .hs => p * (d.inter / 128) * 4,
            .zd => p * d.hidden * 4,
            .y => if (d.profile == .glm) p * d.hidden * 4 else @as(usize, d.rows) * d.hidden * @as(usize, if (d.dtype == .f32) 4 else 2),
            else => 0,
        };
    }
};

/// route -> prep -> GEMM (gate + up) -> mid -> GEMM (down) -> fin.
pub fn prefillChain(d: Prefill) error{BadExl3Shape}![6]Launch {
    try d.check();
    const p = d.positions();
    const glm = d.profile == .glm;
    var route: Launch = .{ .symbol = "t3_pf_route", .group = "prefill", .grid = .{ 1, 1, 1 }, .block = 1024 };
    route.add(.{ .buf = .slots });
    route.add(.{ .u = p });
    route.add(.{ .u = d.experts });
    route.add(.{ .u = gemm_bm });
    for ([_]Role{ .order, .inv, .items }) |r| route.add(.{ .buf = r });
    route.add(.{ .u = d.cap() });
    route.add(.{ .buf = .counts });
    var prep: Launch = .{ .symbol = if (glm) "t3_pf_prep_glm_bf16" else switch (d.dtype) {
        .f32 => "t3_pf_prep_exl3_f32",
        .f16 => "t3_pf_prep_exl3_f16",
        .bf16 => "t3_pf_prep_exl3_bf16",
    }, .group = "prefill", .grid = .{ d.hidden / 128, p, 1 }, .block = 32 };
    for ([_]Role{ .x, .order, .slots, .counts, .gate_suh, .up_suh, .xg, .xu, .sg, .su }) |r| prep.add(.{ .buf = r });
    prep.add(.{ .u = d.hidden });
    prep.add(.{ .u = d.topk });
    var gu: Launch = .{ .symbol = gemmName(d.dec, d.defer_map), .group = "prefill", .grid = .{ d.inter / 128, d.cap(), 2 }, .block = 256 };
    for ([_]Role{ .xg, .xu, .gate_t, .up_t, .items, .zg }) |r| gu.add(.{ .buf = r });
    for ([_]u32{ d.hidden, d.inter, p, d.rate.n, d.dec.window.mask() }) |v| gu.add(.{ .u = v });
    gu.add(.{ .i = @intFromBool(d.bank == .pointers) });
    var mid: Launch = .{ .symbol = if (glm) "t3_pf_mid_glm" else "t3_pf_mid_exl3", .group = "prefill", .grid = .{ d.inter / 128, p, 1 }, .block = 128 };
    for ([_]Role{ .zg, .order, .slots, .counts, .gate_svh, .up_svh, .down_suh, .sg, .su, .hd, .hs }) |r| mid.add(.{ .buf = r });
    for ([_]u32{ d.hidden, d.inter, p }) |v| mid.add(.{ .u = v });
    mid.add(.{ .i = @intFromBool(d.defer_map) });
    mid.add(.{ .f = d.limit });
    var down: Launch = .{ .symbol = gemmName(d.dec, d.defer_map), .group = "prefill", .grid = .{ d.hidden / 128, d.cap(), 1 }, .block = 256 };
    for ([_]Role{ .hd, .hd, .down_t, .down_t, .items, .zd }) |r| down.add(.{ .buf = r });
    for ([_]u32{ d.inter, d.hidden, p, d.rate.n, d.dec.window.mask() }) |v| down.add(.{ .u = v });
    down.add(.{ .i = @intFromBool(d.bank == .pointers) });
    var fin: Launch = .{ .symbol = if (glm) "t3_pf_fin_glm_f32" else switch (d.dtype) {
        .f32 => "t3_pf_fin_exl3_f32",
        .f16 => "t3_pf_fin_exl3_f16",
        .bf16 => "t3_pf_fin_exl3_bf16",
    }, .group = "prefill", .grid = .{ d.hidden / 128, d.rows, 1 }, .block = 128 };
    for ([_]Role{ .zd, .inv, .slots, .scores, .down_svh, .hs, .y }) |r| fin.add(.{ .buf = r });
    for ([_]u32{ d.hidden, d.inter, d.topk, p }) |v| fin.add(.{ .u = v });
    fin.add(.{ .i = @intFromBool(d.defer_map) });
    return .{ route, prep, gu, mid, down, fin };
}

/// Dense rows above 16 (no MLX matmul on AMD): prep_s (natural order) -> the WMMA GEMM over 64-row tiles of the one
/// matrix -> dfinish (one slice). The host fills `items` with `denseItems`.
pub fn denseGemm(d: Dense) error{BadExl3Shape}![3]Launch {
    if (d.in == 0 or d.out == 0 or d.in % 128 != 0 or d.out % 128 != 0 or d.rows == 0) return error.BadExl3Shape;
    if (d.defer_map and d.dec.codebook != .mul1) return error.BadExl3Shape;
    const chain = try denseChain(.{ .in = d.in, .out = d.out, .rows = 1, .rate = d.rate, .dec = d.dec, .dtype = d.dtype, .defer_map = d.defer_map });
    var prep = chain[0];
    prep.grid = .{ d.in / 128, d.rows, 1 };
    prep.args[5] = .{ .i = 0 }; // natural order: the GEMM's A
    var gemm: Launch = .{ .symbol = gemmName(d.dec, d.defer_map), .group = "prefill", .grid = .{ d.out / 128, denseTiles(d.rows), 1 }, .block = 256 };
    for ([_]Role{ .xp, .xp, .trellis, .trellis, .items, .z }) |r| gemm.add(.{ .buf = r });
    for ([_]u32{ d.in, d.out, d.rows, d.rate.n, d.dec.window.mask() }) |v| gemm.add(.{ .u = v });
    gemm.add(.{ .i = 0 }); // the one matrix: no table
    var fin = chain[2];
    fin.grid = .{ d.out / 128, d.rows, 1 };
    fin.args[5] = .{ .u = d.rows };
    fin.args[6] = .{ .u = 1 }; // one slice
    return .{ prep, gemm, fin };
}

pub fn denseTiles(rows: u32) u32 {
    return (rows + gemm_bm - 1) / gemm_bm;
}

/// The dense GEMM's items: (matrix 0, first row, rows) of 64-row tiles.
pub fn denseItems(rows: u32, out: []i32) void {
    for (0..denseTiles(rows)) |t| {
        const first: u32 = @intCast(t * gemm_bm);
        out[3 * t] = 0;
        out[3 * t + 1] = @intCast(first);
        out[3 * t + 2] = @intCast(@min(gemm_bm, rows - first));
    }
}
