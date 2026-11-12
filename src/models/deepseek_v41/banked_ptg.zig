//! The persistent grouped banked GEMVs (`Config.banked_ptg`, at `banked_group` 4; banked.zig's `Banked.ptg`): the grouped
//! gate|up and down GEMVs as at most `threadgroups` threadgroups (NPT; one a work item where the call has fewer), on
//! today's grouped texts (gu3bgp, down3bgp) or hw4b's (gu3bgm3p, down3bgm3p), each grouped GEMV variant on them too.
//!
//! A threadgroup builds the call's lead list once (every row-slot's `turboexl3_members_g`, the grouped GEMVs' own scan, so
//! the same leads, members and member order), then runs the work items at its index, its index + NPT, ...: an item is
//! what one lead threadgroup of the grouped GEMV runs (its lead, output tile block and K threadgroup), on that GEMV's own
//! pieces (gu3g_3 and down3g_3, or hw4b's m3 ones). So every output takes the same reads, decodes, FMAs and sums in the
//! same order: a row-slot's bytes are the grouped call's. No threadgroup runs for a row-slot that leads nothing, and no
//! item scans its members. prep2bg, fin7cbg and dfin3bg are the grouped texts' kernels unchanged.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const decode = @import("../../core/decode.zig");
const banked = @import("banked.zig");
const Kernels3 = decode.Kernels3;
const Cfg = Kernels3.Cfg;
const freeKernel = decode.freeKernel;
const group_variants = banked.group_variants;
const GroupGemvs = banked.GroupGemvs;

/// The persistent texts: the grouped GEMVs' first piece (gu3_1, down3_1: the constants and blk, here the threadgroup's
/// index), the lead list and work item loop around their weight base and their body (banked.zig's grouped pieces)
pub const text_gu3bgp = decode.gu3_1 ++ ptg_1 ++ ptg_gu ++ ptg_2 ++ gu3b_w ++ gu3g_3 ++ ptg_3;
pub const text_down3bgp = decode.down3_1 ++ ptg_1 ++ ptg_2 ++ down3b_w ++ down3g_3 ++ ptg_3;
pub const text_gu3bgm3p = decode.gu3_1 ++ ptg_1 ++ ptg_gu ++ ptg_2 ++ gu3b_w ++ gu3g_3m3 ++ ptg_3;
pub const text_down3bgm3p = decode.down3_1 ++ ptg_1 ++ ptg_2 ++ down3b_w ++ down3g_3m3 ++ ptg_3;
/// The preamble (gu3_1's and down3_1's blk as the threadgroup's index, the lead list), the gate|up GEMV's projection and
/// K half, the work item loop's head and its end
const ptg_1 = @embedFile("kernels/banked/ptg_1.metal");
const ptg_gu = @embedFile("kernels/banked/ptg_gu.metal");
const ptg_2 = @embedFile("kernels/banked/ptg_2.metal");
const ptg_3 = @embedFile("kernels/banked/ptg_3.metal");
/// banked.zig's grouped pieces the work items run: the weight base on the row-slot's bank and the GEMVs' bodies
const gu3b_w = @embedFile("kernels/banked/gu3b_w.metal");
const down3b_w = @embedFile("kernels/banked/down3b_w.metal");
const gu3g_3 = @embedFile("kernels/banked/gu3g_3.metal");
const down3g_3 = @embedFile("kernels/banked/down3g_3.metal");
const gu3g_3m3 = @embedFile("kernels/banked/gu3g_3m3.metal");
const down3g_3m3 = @embedFile("kernels/banked/down3g_3m3.metal");

/// The persistent GEMVs' threadgroups at most (NPT): 640, sixteen 128-thread threadgroups a core of the M5 Max's 40. At
/// least the threadgroups a core keeps resident at these texts' registers: past it the rest wait their turn as whole
/// threadgroups (no work is lost), short of it a core idles, so every core stays full while a threadgroup still runs
/// several work items at a verify's waves (24 pairs: about 8 gate|up and 16 down items)
pub const threadgroups: u32 = 640;

/// Each grouped GEMV variant's header (banked.zig's: its defines before `header_group`)
const headers: [group_variants.len][:0]const u8 = blk: {
    var h: [group_variants.len][:0]const u8 = undefined;
    for (group_variants, 0..) |v, i| h[i] = v.defines ++ banked.header_group;
    break :blk h;
};
/// The persistent GEMVs' kernel names on today's grouped texts and on hw4b's, variant 0's then each variant's
const gu_names: [group_variants.len][:0]const u8 = blk: {
    var n: [group_variants.len][:0]const u8 = undefined;
    n[0] = "turboexl3_gu3bgp";
    for (group_variants[1..], 1..) |v, i| n[i] = "turboexl3_gu3bgp_" ++ v.name;
    break :blk n;
};
const down_names: [group_variants.len][:0]const u8 = blk: {
    var n: [group_variants.len][:0]const u8 = undefined;
    n[0] = "turboexl3_down3bgp";
    for (group_variants[1..], 1..) |v, i| n[i] = "turboexl3_down3bgp_" ++ v.name;
    break :blk n;
};
const gu_m3_names: [group_variants.len][:0]const u8 = blk: {
    var n: [group_variants.len][:0]const u8 = undefined;
    n[0] = "turboexl3_gu3bgm3p";
    for (group_variants[1..], 1..) |v, i| n[i] = "turboexl3_gu3bgm3p_" ++ v.name;
    break :blk n;
};
const down_m3_names: [group_variants.len][:0]const u8 = blk: {
    var n: [group_variants.len][:0]const u8 = undefined;
    n[0] = "turboexl3_down3bgm3p";
    for (group_variants[1..], 1..) |v, i| n[i] = "turboexl3_down3bgm3p_" ++ v.name;
    break :blk n;
};
/// Every persistent GEMV kernel name, for the tests (distinct from banked.zig's `group_kernel_names`)
pub const kernel_names = gu_names ++ down_names ++ gu_m3_names ++ down_m3_names;

/// A geometry's persistent GEMVs beside its banked state: each variant's gate|up and down kernels on today's grouped texts
/// (`gv`) and on hw4b's (`gvm3`), and one config table for both (the grouped configs with the GEMVs' launches at NPT
/// threadgroups), each made on first use.
pub const Ptg = struct {
    gv: [group_variants.len]?GroupGemvs = @splat(null),
    gvm3: [group_variants.len]?GroupGemvs = @splat(null),
    cfg: [banked.banked_rows_max + 1]?Cfg = @splat(null),

    pub fn deinit(self: *Ptg) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.gu, cc.fin, cc.down, cc.dfin, cc.fin7, cc.dfin7, cc.gu8 }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        for (&self.gv) |*g| if (g.*) |gg| {
            freeKernel(gg.gu);
            freeKernel(gg.down);
            g.* = null;
        };
        for (&self.gvm3) |*g| if (g.*) |gg| {
            freeKernel(gg.gu);
            freeKernel(gg.down);
            g.* = null;
        };
    }

    /// Variant `v`'s persistent gate|up and down GEMVs (`group_variants`; 0 today's) on today's grouped texts or, `m3`,
    /// on hw4b's, made on first use: the texts on the variant's header.
    pub fn gemvs(self: *Ptg, v: u8, m3: bool) !GroupGemvs {
        if (v >= group_variants.len) return error.BadExl3Shape;
        const slot = if (m3) &self.gvm3[v] else &self.gv[v];
        if (slot.*) |g| return g;
        const gu = (if (m3)
            ks.newKernel(gu_m3_names[v].ptr, &banked.gu3b_inputs, &decode.gu3_outputs, text_gu3bgm3p, headers[v].ptr, true)
        else
            ks.newKernel(gu_names[v].ptr, &banked.gu3b_inputs, &decode.gu3_outputs, text_gu3bgp, headers[v].ptr, true)) orelse return error.KernelCreateFailed;
        errdefer freeKernel(gu);
        const down = (if (m3)
            ks.newKernel(down_m3_names[v].ptr, &banked.down3b_inputs, &decode.down3_outputs, text_down3bgm3p, headers[v].ptr, true)
        else
            ks.newKernel(down_names[v].ptr, &banked.down3b_inputs, &decode.down3_outputs, text_down3bgp, headers[v].ptr, true)) orelse return error.KernelCreateFailed;
        slot.* = .{ .gu = gu, .down = down };
        return slot.*.?;
    }

    /// The configs at `rows` row-slots: the grouped configs (`buildCfgsGroup` at 4 row-slots per lead, form 9's shape:
    /// bt1 at one row-slot, bt9 above, kdiv9's K threadgroups) with the gate|up and down GEMVs' launches at NPT
    /// threadgroups (`launch`), in a table of their own.
    pub fn cfgs(self: *Ptg, k3: *Kernels3, rows: usize) !Cfg {
        if (rows == 0 or rows > banked.banked_rows_max) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const bt = if (rows == 1) k3.bt1 else k3.bt9;
        if (bt != 1 and bt != 2 and bt != 4 and bt != 8) return error.BadExl3Shape;
        var c = try k3.buildCfgsGroup(rows, bt, k3.kdiv9, k3.kdiv9d orelse k3.kdiv9, 4);
        errdefer {
            inline for (.{ c.prep_h, c.gu, c.fin, c.down, c.dfin, c.fin7, c.dfin7, c.gu8 }) |x| _ = mlx.mlx_fast_metal_kernel_config_free(x);
        }
        const gu = try launch(k3, rows, bt, false);
        _ = mlx.mlx_fast_metal_kernel_config_free(c.gu);
        c.gu = gu;
        const down = try launch(k3, rows, bt, true);
        _ = mlx.mlx_fast_metal_kernel_config_free(c.down);
        c.down = down;
        self.cfg[rows] = c;
        return c;
    }
};

/// The persistent GEMVs' threadgroups (NPT) at `rows` row-slots: one a work item of the grouped GEMV's grid (output tile
/// blocks x row-slots x K threadgroups) when every row-slot leads, at most `threadgroups`
pub fn threadgroupsFor(k3: *const Kernels3, rows: usize, bt: u32, down: bool) u32 {
    const g = k3.geo;
    const ks_n = if (down) k3.ks_dn else k3.ks_gu;
    const z = k3.kThreadgroups(ks_n, rows, if (down) (k3.kdiv9d orelse k3.kdiv9) else k3.kdiv9);
    const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
    const blocks = (if (down) g.hidden else g.inter) / (16 * bt);
    // u32 first: @min with a comptime bound narrows its result type, and the caller's npt * 128 needs 17 bits
    const npt: u32 = @min(rk * blocks * z, threadgroups);
    return npt;
}

/// A persistent GEMV's launch at `rows` row-slots: `buildCfgsGroup`'s gate|up (`down` false) or down GEMV launch, its
/// outputs and constants (GROUP 4 from two row-slots), at NPT threadgroups of 128 threads (`threadgroupsFor`).
fn launch(k3: *Kernels3, rows: usize, bt: u32, down: bool) !mlx.mlx_fast_metal_kernel_config {
    const g = k3.geo;
    const ks_n = if (down) k3.ks_dn else k3.ks_gu;
    const z = k3.kThreadgroups(ks_n, rows, if (down) (k3.kdiv9d orelse k3.kdiv9) else k3.kdiv9);
    const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
    const npt = threadgroupsFor(k3, rows, bt, down);
    const rkc: c_int = @intCast(rk);
    const width: c_int = @intCast(if (down) g.hidden else g.inter);
    const tmpl = [_]ks.TemplateArg{
        .{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } },
        .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } },
        .{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } },
        .{ .name = "HID", .value = .{ .int = @intCast(g.hidden) } },
        .{ .name = "INTER", .value = .{ .int = @intCast(g.inter) } },
        .{ .name = "RKC", .value = .{ .int = rkc } },
        .{ .name = "KS", .value = .{ .int = @intCast(ks_n) } },
        .{ .name = "SPT", .value = .{ .int = @intCast(ks_n / z) } },
        .{ .name = "BT", .value = .{ .int = @intCast(bt) } },
        .{ .name = "GROUP", .value = .{ .int = if (rows >= 2) 4 else 1 } },
        .{ .name = "DEFER", .value = .{ .int = if (k3.defer_affine) 1 else 0 } },
        .{ .name = "NPT", .value = .{ .int = @intCast(npt) } },
    };
    const plane = [_]c_int{ @intCast(ks_n), rkc, width };
    const planes = [_][]const c_int{ &plane, &plane };
    const dtypes = [_]mlx.mlx_dtype{ .float32, .float32 };
    const nout: usize = if (down) 1 else 2;
    return ks.buildConfig(.{ .grid = .{ npt * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = planes[0..nout], .out_dtypes = dtypes[0..nout], .template = &tmpl });
}

const testing = std.testing;

test "turboexl3 banked ptg: the persistent texts are the grouped GEMVs' first pieces, the lead list and work item loop around the grouped bodies, under names of their own" {
    try testing.expect(std.mem.startsWith(u8, text_gu3bgp, decode.gu3_1 ++ ptg_1 ++ ptg_gu ++ ptg_2));
    try testing.expect(std.mem.endsWith(u8, text_gu3bgm3p, gu3b_w ++ gu3g_3m3 ++ ptg_3));
    try testing.expect(std.mem.startsWith(u8, text_down3bgp, decode.down3_1 ++ ptg_1 ++ ptg_2));
    try testing.expect(std.mem.endsWith(u8, text_down3bgm3p, down3b_w ++ down3g_3m3 ++ ptg_3));
    for (kernel_names, 0..) |a, i| for (kernel_names[i + 1 ..]) |b| try testing.expect(!std.mem.eql(u8, a, b));
    try testing.expect(threadgroups >= 1 and threadgroups * 128 < (1 << 31));
}
