//! The HIP launches of turboexl3's kernels, from a shape alone: kernel symbol, grid, block, LDS bytes and arguments by
//! role. Pure Zig with no runtime: turboexl3's tests bind the roles to their buffers (hip_tests.zig), and a host with its
//! own HIP runtime (TensorFold) binds them to its own. The partition rules are dense.zig's (`Dense.ksFor`, `btFor`,
//! `rbCap`, `chainRb`), so a row's sums do not depend on the rows beside it.
const std = @import("std");
const format = @import("../format.zig");

/// A buffer a launch reads or writes, bound by the host.
pub const Role = enum { x, suh, svh, trellis, xp, xs, z, y };

pub const Arg = union(enum) { buf: Role, u: u32, i: i32 };

pub const max_args = 12;

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
    for (.{ "mcg", "mul1", "mul1d" }, 0..) |cb, i| {
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
