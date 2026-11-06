//! DeepSeek-V4.1's banked decode: an expert streamer's routed pairs over three slot banks (`root.moeClampedBanked`), form
//! 7's launches at top-1 with the lines that read a bank's arrays swapped in, its grouped texts (`Config.banked_group` 4)
//! and their GEMV variants (hw2, hw4b). It extends a geometry's `decode.Kernels3` (the routed decode's launch shapes and
//! configs) with the banked kernels and configs of its own (`Banked`), every launch the base launch's on its bank.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const config = @import("../../core/config.zig");
const decode = @import("../../core/decode.zig");
const Kernels3 = decode.Kernels3;
const Kernels = decode.Kernels;
const Cfg = Kernels3.Cfg;
const header = decode.header;
const header5 = decode.header5;
const freeKernel = decode.freeKernel;
const one = Kernels3.one;

// the core's pieces and launch names these texts compose
const prep2_1 = decode.prep2_1;
const prep2_row = decode.prep2_row;
const prep2_body = decode.prep2_body;
const gu3_1 = decode.gu3_1;
const rk_line = decode.rk_line;
const gu3_2 = decode.gu3_2;
const gu3_3 = decode.gu3_3;
const gu3_stores = decode.gu3_stores;
const gu3_4 = decode.gu3_4;
const fin7_1 = decode.fin7_1;
const fin7_2 = decode.fin7_2;
const fin7_planes = decode.fin7_planes;
const fin7_3 = decode.fin7_3;
const fin7_sums = decode.fin7_sums;
const fin7_4 = decode.fin7_4;
const fin7_swiglu_clamp = decode.fin7_swiglu_clamp;
const fin7_swiglu = decode.fin7_swiglu;
const fin7_swiglu_clamp_out = decode.fin7_swiglu_clamp_out;
const fin7_5 = decode.fin7_5;
const down3_1 = decode.down3_1;
const down3_2 = decode.down3_2;
const down3_3 = decode.down3_3;
const dfin3_1 = decode.dfin3_1;
const dfin3_2 = decode.dfin3_2;
const prep2_outputs = decode.prep2_outputs;
const gu3_outputs = decode.gu3_outputs;
const fin3_outputs = decode.fin3_outputs;
const down3_outputs = decode.down3_outputs;
const down_outputs = decode.down_outputs;

// ── The banked decode (`Kernels3.bankedGateUp` / `bankedDown`) ──
//
// A call's row-slots over three expert banks (an expert streamer's slot banks: base, ext, transient, each its own arrays
// with its own slot count), a row-slot's id packed (bank << 24 | slot row). Form 7's launches at top-1 (prep2, gu3, fin7
// clamped, down3, dfin3), each its base text with the lines that read a bank's arrays replaced and every other statement
// kept: where the base reads the expert line, the banked text takes the slot row and selects the row-slot's suh / svh
// arrays by its bank (`prep2b_expert`, `fin7b_expert`, `dfin3b_expert`); where the GEMVs take their weight base, the
// bank's trellis at the slot row (`gu3b_w`, `down3b_w`). The GEMVs' members compare the packed ids, so two row-slots
// share a decode only on one bank's one slot. A row-slot's statements, so its bytes, are the base launch's on its bank.
// Each bank's arrays are separate inputs (at most 14 inputs and 2 outputs a launch, within Metal's 31 buffers).
pub const text_prep2b = prep2_1 ++ prep2b_expert ++ prep2_row ++ prep2_body;
pub const text_gu3b = gu3_1 ++ rk_line ++ gu3_2 ++ gu3b_w ++ gu3_3 ++ gu3_stores ++ gu3_4;
pub const text_fin7cb = fin7_1 ++ fin7b_expert ++ fin7_2 ++ fin7_planes ++ fin7_3 ++ fin7_sums ++ fin7_4 ++ fin7_swiglu_clamp ++ fin7_5;
/// Ling-3.1 (`models/ling/api.zig`): fin7cb with fin7's own SwiGLU line (an unclamped geometry, `Geometry.limit` 0) and
/// with Ling's clamp after the SiLU (`Geometry.clamp` `silu_out`), every other statement fin7cb's.
pub const text_fin7b = fin7_1 ++ fin7b_expert ++ fin7_2 ++ fin7_planes ++ fin7_3 ++ fin7_sums ++ fin7_4 ++ fin7_swiglu ++ fin7_5;
pub const text_fin7cob = fin7_1 ++ fin7b_expert ++ fin7_2 ++ fin7_planes ++ fin7_3 ++ fin7_sums ++ fin7_4 ++ fin7_swiglu_clamp_out ++ fin7_5;
pub const text_down3b = down3_1 ++ rk_line ++ down3_2 ++ down3b_w ++ down3_3;
pub const text_dfin3b = dfin3_1 ++ dfin3b_expert ++ dfin3_2;
const prep2b_expert = @embedFile("kernels/banked/prep2b_expert.metal");
const gu3b_w = @embedFile("kernels/banked/gu3b_w.metal");
const fin7b_expert = @embedFile("kernels/banked/fin7b_expert.metal");
const down3b_w = @embedFile("kernels/banked/down3b_w.metal");
const dfin3b_expert = @embedFile("kernels/banked/dfin3b_expert.metal");
/// The banked launches' inputs: each banked array once per bank (0 base, 1 ext, 2 transient), in the base's order
pub const prep2b_inputs = [_][*:0]const u8{ "x", "slots", "gate_suh0", "gate_suh1", "gate_suh2", "up_suh0", "up_suh1", "up_suh2" };
pub const gu3b_inputs = [_][*:0]const u8{ "xgp", "xup", "slots", "gate_t0", "gate_t1", "gate_t2", "up_t0", "up_t1", "up_t2" };
pub const fin7b_inputs = [_][*:0]const u8{ "zg", "zu", "slots", "gate_svh0", "gate_svh1", "gate_svh2", "up_svh0", "up_svh1", "up_svh2", "down_suh0", "down_suh1", "down_suh2", "xsg", "xsu" };
pub const down3b_inputs = [_][*:0]const u8{ "hdp", "slots", "down_t0", "down_t1", "down_t2" };
pub const dfin3b_inputs = [_][*:0]const u8{ "zd", "slots", "scores", "down_svh0", "down_svh1", "down_svh2", "hsx" };

// ── The grouped banked decode (`Kernels3.banked_group` 4) ──
//
// The banked launches with the GEMVs' members widened from two row-slots to four of one packed id per lead threadgroup
// (`turboexl3_members_g`, `turboexl3_gemv_group`: every weight read and decoded once and fed to each member in the
// one-member order, so a row-slot's bytes are the banked call's), and a row-slot whose id is `skip_id` skipped by every
// launch: no decode, its y row 0 (DEVROUTE's pairs whose expert no bank holds; the caller replaces their rows).
pub const skip_id: u32 = 0xFFFFFFFF;
/// The grouped GEMVs' members and core, after the base header
pub const header_group: [:0]const u8 = header ++ gemv_group;
const gemv_group = @embedFile("kernels/banked/gemv_group.metal");
/// gu8b's header: the grouped GEMVs' members and core after form 8's (the register H128 its prepare runs)
pub const header5_group: [:0]const u8 = header5 ++ gemv_group;
/// hw2 (the hit wave's ILP retune): the grouped banked GEMVs' variants (`Kernels3.bankedGateUpSelV`, `bankedDownSelV`;
/// `Config.banked_variant_gu`, `banked_variant_dn`). Each is Metal defines put before `header_group` for the grouped
/// gate|up and down GEMVs alone (gu3bg and down3bg, or hw4b's gu3bgm3 and down3bgm3; gemv_group.metal and the MUL1
/// helper read them), under kernel names of its own (MLX keys a custom kernel's library by its name). Every variant reads,
/// decodes and sums each weight as variant 0 (today's) does: the same codewords, the same decoded values, the same FMA
/// sequence into the same f32 accumulators and the same folds, so the same bytes (banked_ilp_test.zig checks every word,
/// on today's grouped texts and on hw4b's). They differ in when the tile words load and how many lanes load them, the
/// registers the loop holds, the passes over a tile and the instructions from a codeword to t.
pub const GroupVariant = struct { name: [:0]const u8, defines: [:0]const u8 };
pub const group_variants = [_]GroupVariant{
    .{ .name = "", .defines = "" },
    .{ .name = "ku", .defines = "#define EXL3_KU 1\n" },
    .{ .name = "pf", .defines = "#define EXL3_GPF 1\n" },
    .{ .name = "xh", .defines = "#define EXL3_GXH 1\n" },
    .{ .name = "m2", .defines = "#define EXL3_GM2 1\n" },
    .{ .name = "sh", .defines = "#define EXL3_GSH 1\n" },
    .{ .name = "t4", .defines = "#define EXL3_T1V 4\n" },
    .{ .name = "t1", .defines = "#define EXL3_T1V 1\n" },
    .{ .name = "t0", .defines = "#define EXL3_T1V 0\n" },
    .{ .name = "pfxh", .defines = "#define EXL3_GPF 1\n#define EXL3_GXH 1\n" },
    .{ .name = "pfsh", .defines = "#define EXL3_GPF 1\n#define EXL3_GSH 1\n" },
    .{ .name = "xhm2", .defines = "#define EXL3_GXH 1\n#define EXL3_GM2 1\n" },
};
comptime {
    if (group_variants.len != @import("../../core/config.zig").banked_variants) @compileError("config.banked_variants is not group_variants.len");
}
/// `group_variants`' index by name (null: none)
pub fn groupVariantOf(name: []const u8) ?u8 {
    for (group_variants, 0..) |v, i| if (std.mem.eql(u8, v.name, name)) return @intCast(i);
    return null;
}
/// `group_variants[i]`'s header, and its grouped GEMVs' kernel names on today's grouped texts (gu3bg, down3bg) and on
/// hw4b's (gu3bgm3, down3bgm3); variant 0's the stock names
const group_headers: [group_variants.len][:0]const u8 = blk: {
    var h: [group_variants.len][:0]const u8 = undefined;
    for (group_variants, 0..) |v, i| h[i] = v.defines ++ header_group;
    break :blk h;
};
const group_gu_names: [group_variants.len][:0]const u8 = blk: {
    var n: [group_variants.len][:0]const u8 = undefined;
    n[0] = "turboexl3_gu3bg";
    for (group_variants[1..], 1..) |v, i| n[i] = "turboexl3_gu3bg_" ++ v.name;
    break :blk n;
};
const group_down_names: [group_variants.len][:0]const u8 = blk: {
    var n: [group_variants.len][:0]const u8 = undefined;
    n[0] = "turboexl3_down3bg";
    for (group_variants[1..], 1..) |v, i| n[i] = "turboexl3_down3bg_" ++ v.name;
    break :blk n;
};
const group_gu_m3_names: [group_variants.len][:0]const u8 = blk: {
    var n: [group_variants.len][:0]const u8 = undefined;
    n[0] = "turboexl3_gu3bgm3";
    for (group_variants[1..], 1..) |v, i| n[i] = "turboexl3_gu3bgm3_" ++ v.name;
    break :blk n;
};
const group_down_m3_names: [group_variants.len][:0]const u8 = blk: {
    var n: [group_variants.len][:0]const u8 = undefined;
    n[0] = "turboexl3_down3bgm3";
    for (group_variants[1..], 1..) |v, i| n[i] = "turboexl3_down3bgm3_" ++ v.name;
    break :blk n;
};
/// Every grouped GEMV kernel name a variant or a text set uses, for the tests: distinct names, distinct libraries.
pub const group_kernel_names = group_gu_names ++ group_down_names ++ group_gu_m3_names ++ group_down_m3_names;
pub const text_prep2bg = prep2_1 ++ skip_rk ++ prep2b_expert ++ prep2_row ++ prep2_body;
pub const text_gu3bg = gu3_1 ++ rk_line ++ gu3g_2 ++ gu3b_w ++ gu3g_3;
pub const text_fin7cbg = fin7_1 ++ skip_rk ++ fin7b_expert ++ fin7_2 ++ fin7_planes ++ fin7_3 ++ fin7_sums ++ fin7_4 ++ fin7_swiglu_clamp ++ fin7_5;
/// fin7cbg's Ling-3.1 forms, as `text_fin7b` and `text_fin7cob` are fin7cb's.
pub const text_fin7bg = fin7_1 ++ skip_rk ++ fin7b_expert ++ fin7_2 ++ fin7_planes ++ fin7_3 ++ fin7_sums ++ fin7_4 ++ fin7_swiglu ++ fin7_5;
pub const text_fin7cobg = fin7_1 ++ skip_rk ++ fin7b_expert ++ fin7_2 ++ fin7_planes ++ fin7_3 ++ fin7_sums ++ fin7_4 ++ fin7_swiglu_clamp_out ++ fin7_5;
pub const text_down3bg = down3_1 ++ rk_line ++ down3g_2 ++ down3b_w ++ down3g_3;
pub const text_dfin3bg = dfin3_1 ++ skip_rk_next ++ dfin3b_expert ++ dfin3_2;
/// hw4b: the grouped GEMVs with an M = 3 member path (`gu3g_3m3`, `down3g_3m3`): a lead with three members runs it, where
/// today's run the M = 4 path with a fourth member's FMAs padding. Every member is fed in the one-member order
/// (`turboexl3_gemv_group`), so a row-slot's bytes are the grouped call's. prep2bg, fin7cbg and dfin3bg are the grouped
/// texts unchanged.
pub const text_gu3bgm3 = gu3_1 ++ rk_line ++ gu3g_2 ++ gu3b_w ++ gu3g_3m3;
pub const text_down3bgm3 = down3_1 ++ rk_line ++ down3g_2 ++ down3b_w ++ down3g_3m3;
/// gu8b (`Config.banked_gu8`): the grouped gate|up with its prepare inside (form 8's idiom, `decode.text_gu8`): each lead
/// threadgroup's four simdgroups first run prep2b's arithmetic over the threadgroup's K span for every member and both
/// projections into threadgroup memory (the members' bank's suh at their slot row; the first output-tile threadgroups
/// write each block's input sums, xsg / xsu, for fin7's deferred map), then gu3bg's GEMV (`turboexl3_gemv_group`, every
/// member in the one-member order) reads the input there, the bank's trellis based at the span's first tile; then
/// gu3bg's planes. prep2bg + gu3bg's bytes in one launch; `text_gu8bgm3` the same with hw4b's M = 3 member path.
pub const text_gu8bg = decode.gu8_1 ++ rk_line ++ gu8bg_2 ++ gu8bg_3 ++ gu8bg_4;
pub const text_gu8bgm3 = decode.gu8_1 ++ rk_line ++ gu8bg_2 ++ gu8bg_3m3 ++ gu8bg_4;
pub const gu8bg_2 = @embedFile("kernels/banked/gu8bg_2.metal");
pub const gu8bg_3 = @embedFile("kernels/banked/gu8bg_3.metal");
pub const gu8bg_3m3 = @embedFile("kernels/banked/gu8bg_3m3.metal");
pub const gu8bg_4 = @embedFile("kernels/banked/gu8bg_4.metal");
/// gu8b's inputs: prep2b's and gu3b's, each banked array once per bank; its outputs form 8's (`decode.gu8_outputs`:
/// the planes and the blocks' input sums)
pub const gu8b_inputs = [_][*:0]const u8{ "x", "slots", "gate_suh0", "gate_suh1", "gate_suh2", "up_suh0", "up_suh1", "up_suh2", "gate_t0", "gate_t1", "gate_t2", "up_t0", "up_t1", "up_t2" };
/// The skipped row-slot's line where a launch starts on one (prep2, fin7) and in dfin3's slot loop
const skip_rk = @embedFile("kernels/banked/skip_rk.metal");
const skip_rk_next = @embedFile("kernels/banked/skip_rk_next.metal");
/// gu3's and down3's pieces after the constants and grid line and around the weight base: four members per lead, the
/// reduction buffer for each and the planes' stores for each
const gu3g_2 = @embedFile("kernels/banked/gu3g_2.metal");
const gu3g_3 = @embedFile("kernels/banked/gu3g_3.metal");
const down3g_2 = @embedFile("kernels/banked/down3g_2.metal");
const down3g_3 = @embedFile("kernels/banked/down3g_3.metal");
const gu3g_3m3 = @embedFile("kernels/banked/gu3g_3m3.metal");
const down3g_3m3 = @embedFile("kernels/banked/down3g_3m3.metal");

/// The most row-slots a banked call takes (an expert streamer's route: 8 verify rows of top-6 = 48 pairs)
pub const banked_rows_max: usize = 64;
/// The banked texts' kernels (`text_prep2b` .. `text_dfin3b`)
pub const BankedKernels = struct {
    prep: mlx.mlx_fast_metal_kernel,
    gu: mlx.mlx_fast_metal_kernel,
    fin: mlx.mlx_fast_metal_kernel,
    down: mlx.mlx_fast_metal_kernel,
    dfin: mlx.mlx_fast_metal_kernel,
};
/// A grouped GEMV variant's two kernels (`banked_gv`, `banked_gvm3`)
pub const GroupGemvs = struct { gu: mlx.mlx_fast_metal_kernel, down: mlx.mlx_fast_metal_kernel };

/// A geometry's banked state beside its `Kernels3`: the banked kernels and configs, made on first use.
pub const Banked = struct {
    /// The banked decode (`bankedGateUp`, `bankedDown`; a top-1 clamped geometry): its kernels, built on first use, one
    /// config set per row-slot count up to `banked_rows_max` (form 9's launch shape), and dfin3's unit scores
    banked: ?BankedKernels = null,
    cfgb: [banked_rows_max + 1]?Cfg = @splat(null),
    banked_ones: ?mlx.mlx_array = null,
    /// The banked GEMVs' row-slots of one packed id per lead threadgroup: 2 the banked texts (`group`'s pairing), 4 the
    /// grouped texts (`text_gu3bg` .. `text_dfin3bg`), which also skip a row-slot whose id is `skip_id`. The same bytes
    /// for every other row-slot. Set before the first banked call.
    banked_group: u32 = 2,
    banked_g: ?BankedKernels = null,
    cfgbg: [banked_rows_max + 1]?Cfg = @splat(null),
    /// hw4b: at `banked_group` 4, the grouped texts with the M = 3 member path (`text_gu3bgm3`, `text_down3bgm3`); their
    /// kernels and their configs, a table of its own. The same bytes for every row-slot.
    banked_m3: bool = false,
    banked_gm3: ?BankedKernels = null,
    cfgbgm3: [banked_rows_max + 1]?Cfg = @splat(null),
    /// gu8b: at `banked_group` 4, the gate|up half's prepare inside its GEMV launch (`text_gu8bg`, or `text_gu8bgm3` on
    /// hw4b's texts) where a call fits (`gu8Fits`); the same bytes, one dependent launch fewer. Its two kernels, made on
    /// first use; the configs are the grouped texts' (`cfgbg`, `cfgbgm3`: their `gu8`).
    banked_gu8: bool = false,
    banked_g8: ?mlx.mlx_fast_metal_kernel = null,
    banked_g8m3: ?mlx.mlx_fast_metal_kernel = null,
    /// hw2: each grouped GEMV variant's gate|up and down kernels on today's grouped texts (`banked_gv`) and on hw4b's
    /// (`banked_gvm3`) (`group_variants`; `bankedGateUpSelV`, `bankedDownSelV`), made on first use (variant 0's are
    /// `banked_g`'s and `banked_gm3`'s). The configs are their texts' (`cfgbg`, `cfgbgm3`).
    banked_gv: [group_variants.len]?GroupGemvs = @splat(null),
    banked_gvm3: [group_variants.len]?GroupGemvs = @splat(null),

    /// The banked state for the process's `Kernels3` of a geometry (`forKernels`): `Config.banked_group` and
    /// `banked_m3` as configured when it is made
    pub fn init() Banked {
        return .{ .banked_group = config.get().banked_group, .banked_m3 = config.get().banked_m3, .banked_gu8 = config.get().banked_gu8 };
    }

    pub fn deinit(self: *Banked) void {
        for ([_]*[banked_rows_max + 1]?Cfg{ &self.cfgb, &self.cfgbg, &self.cfgbgm3 }) |tbl| for (tbl) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.gu, cc.fin, cc.down, cc.dfin, cc.fin7, cc.dfin7, cc.gu8 }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        if (self.banked) |b| inline for (.{ b.prep, b.gu, b.fin, b.down, b.dfin }) |k| freeKernel(k);
        if (self.banked_g) |b| inline for (.{ b.prep, b.gu, b.fin, b.down, b.dfin }) |k| freeKernel(k);
        if (self.banked_gm3) |b| inline for (.{ b.prep, b.gu, b.fin, b.down, b.dfin }) |k| freeKernel(k);
        for (&self.banked_gv) |*g| if (g.*) |gg| {
            freeKernel(gg.gu);
            freeKernel(gg.down);
            g.* = null;
        };
        for (&self.banked_gvm3) |*g| if (g.*) |gg| {
            freeKernel(gg.gu);
            freeKernel(gg.down);
            g.* = null;
        };
        if (self.banked_ones) |o| _ = mlx.mlx_array_free(o);
        for ([_]?mlx.mlx_fast_metal_kernel{ self.banked_g8, self.banked_g8m3 }) |k| if (k) |kk| freeKernel(kk);
        self.* = undefined;
    }

    /// The banked decode's kernels, made on first use: a top-1 geometry's (each row-slot a pair at weight 1), its SwiGLU
    /// the geometry's (DeepSeek-V4.1's clamp, none, or Ling-3.1's clamp after the SiLU); error.Exl3EngineNotImplemented for any other.
    fn bankedKernels(self: *Banked, k3: *Kernels3) !*const BankedKernels {
        if (self.banked) |*b| return b;
        if (k3.geo.topk != 1) return error.Exl3EngineNotImplemented;
        const prep = ks.newKernel("turboexl3_prep2b", &prep2b_inputs, &prep2_outputs, text_prep2b, header, true) orelse return error.KernelCreateFailed;
        errdefer freeKernel(prep);
        const gu = ks.newKernel("turboexl3_gu3b", &gu3b_inputs, &gu3_outputs, text_gu3b, header, true) orelse return error.KernelCreateFailed;
        errdefer freeKernel(gu);
        // the geometry's SwiGLU: DeepSeek-V4.1's clamp (fin7cb), none (fin7b) or Ling-3.1's clamp after the SiLU (fin7cob)
        const fin = (if (k3.geo.limit != 0 and k3.geo.clamp == .gate_in)
            ks.newKernel("turboexl3_fin7cb", &fin7b_inputs, &fin3_outputs, text_fin7cb, header5, true)
        else if (k3.geo.limit == 0)
            ks.newKernel("turboexl3_fin7b", &fin7b_inputs, &fin3_outputs, text_fin7b, header5, true)
        else
            ks.newKernel("turboexl3_fin7cob", &fin7b_inputs, &fin3_outputs, text_fin7cob, header5, true)) orelse return error.KernelCreateFailed;
        errdefer freeKernel(fin);
        const down = ks.newKernel("turboexl3_down3b", &down3b_inputs, &down3_outputs, text_down3b, header, true) orelse return error.KernelCreateFailed;
        errdefer freeKernel(down);
        const dfin = ks.newKernel("turboexl3_dfin3b", &dfin3b_inputs, &down_outputs, text_dfin3b, header, true) orelse return error.KernelCreateFailed;
        self.banked = .{ .prep = prep, .gu = gu, .fin = fin, .down = down, .dfin = dfin };
        return &self.banked.?;
    }

    /// The banked decode's configs at `rows` row-slots: form 9's launch shape at that many row-slots (`cfgs9`'s: bt1 at
    /// one, bt9 above, kdiv9's K threadgroups), in a table of its own up to `banked_rows_max`.
    fn cfgsBanked(self: *Banked, k3: *Kernels3, rows: usize) !Cfg {
        if (rows == 0 or rows > banked_rows_max) return error.BadExl3Shape;
        if (self.cfgb[rows]) |c| return c;
        const bt = if (rows == 1) k3.bt1 else k3.bt9;
        if (bt != 1 and bt != 2 and bt != 4 and bt != 8) return error.BadExl3Shape;
        const c = try k3.buildCfgsSplit(rows, bt, k3.kdiv9, k3.kdiv9d orelse k3.kdiv9);
        self.cfgb[rows] = c;
        return c;
    }

    /// The grouped banked kernels (`banked_group` 4), made on first use; the geometry as `bankedKernels` takes it.
    fn bankedKernelsG(self: *Banked, k3: *Kernels3) !*const BankedKernels {
        if (self.banked_g) |*b| return b;
        if (k3.geo.topk != 1) return error.Exl3EngineNotImplemented;
        const prep = ks.newKernel("turboexl3_prep2bg", &prep2b_inputs, &prep2_outputs, text_prep2bg, header, true) orelse return error.KernelCreateFailed;
        errdefer freeKernel(prep);
        const gu = ks.newKernel("turboexl3_gu3bg", &gu3b_inputs, &gu3_outputs, text_gu3bg, header_group, true) orelse return error.KernelCreateFailed;
        errdefer freeKernel(gu);
        // the geometry's SwiGLU: fin7cbg, fin7bg or fin7cobg (as bankedKernels' fin7cb, fin7b or fin7cob)
        const fin = (if (k3.geo.limit != 0 and k3.geo.clamp == .gate_in)
            ks.newKernel("turboexl3_fin7cbg", &fin7b_inputs, &fin3_outputs, text_fin7cbg, header5, true)
        else if (k3.geo.limit == 0)
            ks.newKernel("turboexl3_fin7bg", &fin7b_inputs, &fin3_outputs, text_fin7bg, header5, true)
        else
            ks.newKernel("turboexl3_fin7cobg", &fin7b_inputs, &fin3_outputs, text_fin7cobg, header5, true)) orelse return error.KernelCreateFailed;
        errdefer freeKernel(fin);
        const down = ks.newKernel("turboexl3_down3bg", &down3b_inputs, &down3_outputs, text_down3bg, header_group, true) orelse return error.KernelCreateFailed;
        errdefer freeKernel(down);
        const dfin = ks.newKernel("turboexl3_dfin3bg", &dfin3b_inputs, &down_outputs, text_dfin3bg, header, true) orelse return error.KernelCreateFailed;
        self.banked_g = .{ .prep = prep, .gu = gu, .fin = fin, .down = down, .dfin = dfin };
        return &self.banked_g.?;
    }

    /// `cfgsBanked` at 4 row-slots per lead (the grouped texts' GROUP), in a table of its own.
    fn cfgsBankedG(self: *Banked, k3: *Kernels3, rows: usize) !Cfg {
        if (rows == 0 or rows > banked_rows_max) return error.BadExl3Shape;
        if (self.cfgbg[rows]) |c| return c;
        const bt = if (rows == 1) k3.bt1 else k3.bt9;
        if (bt != 1 and bt != 2 and bt != 4 and bt != 8) return error.BadExl3Shape;
        const c = try k3.buildCfgsGroup(rows, bt, k3.kdiv9, k3.kdiv9d orelse k3.kdiv9, 4);
        self.cfgbg[rows] = c;
        return c;
    }

    /// hw4b's kernels (`text_gu3bgm3`, `text_down3bgm3`; prep2bg, fin7cbg and dfin3bg the grouped texts), made on first use.
    fn bankedKernelsGM3(self: *Banked, k3: *Kernels3) !*const BankedKernels {
        if (self.banked_gm3) |*b| return b;
        if (k3.geo.topk != 1) return error.Exl3EngineNotImplemented;
        const prep = ks.newKernel("turboexl3_prep2bg", &prep2b_inputs, &prep2_outputs, text_prep2bg, header, true) orelse return error.KernelCreateFailed;
        errdefer freeKernel(prep);
        const gu = ks.newKernel("turboexl3_gu3bgm3", &gu3b_inputs, &gu3_outputs, text_gu3bgm3, header_group, true) orelse return error.KernelCreateFailed;
        errdefer freeKernel(gu);
        // the geometry's SwiGLU: fin7cbg, fin7bg or fin7cobg (as bankedKernels' fin7cb, fin7b or fin7cob)
        const fin = (if (k3.geo.limit != 0 and k3.geo.clamp == .gate_in)
            ks.newKernel("turboexl3_fin7cbg", &fin7b_inputs, &fin3_outputs, text_fin7cbg, header5, true)
        else if (k3.geo.limit == 0)
            ks.newKernel("turboexl3_fin7bg", &fin7b_inputs, &fin3_outputs, text_fin7bg, header5, true)
        else
            ks.newKernel("turboexl3_fin7cobg", &fin7b_inputs, &fin3_outputs, text_fin7cobg, header5, true)) orelse return error.KernelCreateFailed;
        errdefer freeKernel(fin);
        const down = ks.newKernel("turboexl3_down3bgm3", &down3b_inputs, &down3_outputs, text_down3bgm3, header_group, true) orelse return error.KernelCreateFailed;
        errdefer freeKernel(down);
        const dfin = ks.newKernel("turboexl3_dfin3bg", &dfin3b_inputs, &down_outputs, text_dfin3bg, header, true) orelse return error.KernelCreateFailed;
        self.banked_gm3 = .{ .prep = prep, .gu = gu, .fin = fin, .down = down, .dfin = dfin };
        return &self.banked_gm3.?;
    }

    /// hw4b's configs at 4 row-slots per lead, in a table of its own.
    fn cfgsBankedGM3(self: *Banked, k3: *Kernels3, rows: usize) !Cfg {
        if (rows == 0 or rows > banked_rows_max) return error.BadExl3Shape;
        if (self.cfgbgm3[rows]) |c| return c;
        const bt = if (rows == 1) k3.bt1 else k3.bt9;
        if (bt != 1 and bt != 2 and bt != 4 and bt != 8) return error.BadExl3Shape;
        const c = try k3.buildCfgsGroup(rows, bt, k3.kdiv9, k3.kdiv9d orelse k3.kdiv9, 4);
        self.cfgbgm3[rows] = c;
        return c;
    }

    /// Which banked texts a call runs (`bankedGateUpSel`, `bankedDownSel`): `configured` (`banked_group`, `banked_m3`),
    /// today's grouped texts (`g4`) or hw4b's (`g4m3`: the M = 3 path). Every selection gives a row-slot the same bytes.
    pub const BankedSel = enum { configured, g4, g4m3, g4gu8, g4m3gu8 };
    /// A set's texts: the banked ones (GROUP 2), today's grouped ones or hw4b's (the texts hw2's variants run on)
    pub const Texts = enum { banked, g4, g4m3 };

    /// The texts a call on `sel` runs (`bankedSetSel`'s choice, read without a launch): a test's check that an arm ran the
    /// texts it names (`configured` follows `banked_group` and `banked_m3` as this state took them at its first call).
    pub fn selTexts(self: *const Banked, sel: BankedSel) Texts {
        return switch (sel) {
            .configured => if (self.banked_group == 4) (if (self.banked_m3) .g4m3 else .g4) else .banked,
            .g4, .g4gu8 => .g4,
            .g4m3, .g4m3gu8 => .g4m3,
        };
    }

    /// Whether a call of `rows` row-slots on `sel` runs gu8b (its gate|up's prepare inside the launch): gu8b's selections,
    /// or `configured` with `banked_gu8` at `banked_group` 4, where the call fits (`gu8Fits`); the tests' check that an arm
    /// ran what it names
    pub fn selGu8(self: *const Banked, k3: *const Kernels3, sel: BankedSel, rows: usize) bool {
        const want = switch (sel) {
            .configured => self.banked_group == 4 and self.banked_gu8,
            .g4, .g4m3 => false,
            .g4gu8, .g4m3gu8 => true,
        };
        return want and gu8Fits(k3, rows);
    }

    /// gu8b's admission at a call of `rows` row-slots (form 9's launch shape there: bt1 at one row-slot, bt9 above,
    /// kdiv9's K threadgroups): the threadgroup's K span (SPT slices of hidden / KS) whole 128-blocks, and its members'
    /// prepared spans (GROUP 4 from two row-slots, both projections, f16) beside the reduction buffer within Metal's 32 KB
    pub fn gu8Fits(k3: *const Kernels3, rows: usize) bool {
        if (rows == 0 or rows > banked_rows_max) return false;
        const hid = k3.geo.hidden;
        if (hid % k3.ks_gu != 0) return false;
        const spt = k3.ks_gu / k3.kThreadgroups(k3.ks_gu, rows, k3.kdiv9);
        const span = hid / k3.ks_gu * spt;
        if (span % 128 != 0) return false;
        const grp: u32 = if (rows >= 2) 4 else 1;
        const bt: u32 = if (rows == 1) k3.bt1 else k3.bt9;
        return grp * 2 * span * 2 + spt * 4 * grp * 16 * bt * 4 <= 32768;
    }

    /// gu8b's kernel on today's grouped texts or (`m3`) hw4b's, made on first use
    fn gu8Kernel(self: *Banked, m3: bool) !mlx.mlx_fast_metal_kernel {
        const slot = if (m3) &self.banked_g8m3 else &self.banked_g8;
        if (slot.*) |k| return k;
        const k = (if (m3)
            ks.newKernel("turboexl3_gu8bgm3", &gu8b_inputs, &decode.gu8_outputs, text_gu8bgm3, header5_group, true)
        else
            ks.newKernel("turboexl3_gu8bg", &gu8b_inputs, &decode.gu8_outputs, text_gu8bg, header5_group, true)) orelse return error.KernelCreateFailed;
        slot.* = k;
        return k;
    }
    const BankedSet = struct { k: *const BankedKernels, c: Cfg, texts: Texts, gu8: bool = false };

    /// The banked launches and configs at `banked_group`: the banked texts at 2, the grouped texts at 4 (hw4b's with
    /// `banked_m3`).
    fn bankedSet(self: *Banked, k3: *Kernels3, rows: usize) !BankedSet {
        return switch (self.banked_group) {
            2 => .{ .k = try self.bankedKernels(k3), .c = try self.cfgsBanked(k3, rows), .texts = .banked },
            4 => blk: {
                var set = if (self.banked_m3) try self.gm3Set(k3, rows) else try self.g4Set(k3, rows);
                set.gu8 = self.banked_gu8;
                break :blk set;
            },
            else => error.BadExl3Shape,
        };
    }

    /// Today's grouped texts at GROUP 4.
    fn g4Set(self: *Banked, k3: *Kernels3, rows: usize) !BankedSet {
        return .{ .k = try self.bankedKernelsG(k3), .c = try self.cfgsBankedG(k3, rows), .texts = .g4 };
    }

    /// hw4b's texts at GROUP 4.
    fn gm3Set(self: *Banked, k3: *Kernels3, rows: usize) !BankedSet {
        return .{ .k = try self.bankedKernelsGM3(k3), .c = try self.cfgsBankedGM3(k3, rows), .texts = .g4m3 };
    }

    /// `sel`'s launches and configs (`configured`: `bankedSet`'s).
    fn bankedSetSel(self: *Banked, k3: *Kernels3, rows: usize, sel: BankedSel) !BankedSet {
        return switch (sel) {
            .configured => self.bankedSet(k3, rows),
            .g4 => self.g4Set(k3, rows),
            .g4m3 => self.gm3Set(k3, rows),
            .g4gu8 => blk: {
                var set = try self.g4Set(k3, rows);
                set.gu8 = true;
                break :blk set;
            },
            .g4m3gu8 => blk: {
                var set = try self.gm3Set(k3, rows);
                set.gu8 = true;
                break :blk set;
            },
        };
    }

    /// `bankedSetSel` with the grouped GEMVs of `variant` (`group_variants`; 0: the set's own) on the set's grouped texts
    /// (today's or hw4b's); any but 0 needs grouped texts (`banked_group` 4, or `sel` g4 or g4m3).
    fn bankedSetSelV(self: *Banked, k3: *Kernels3, rows: usize, sel: BankedSel, variant: u8) !struct { k: BankedKernels, c: Cfg, texts: Texts, gu8: bool } {
        if (variant >= group_variants.len) return error.BadExl3Shape;
        const set = try self.bankedSetSel(k3, rows, sel);
        var k = set.k.*;
        if (variant != 0) {
            if (set.texts == .banked) return error.BadExl3Shape;
            const g = try self.groupGemvs(variant, set.texts == .g4m3);
            k.gu = g.gu;
            k.down = g.down;
        }
        // gu8b's gate|up is its own text (no hw2 variant of it): a variant's call keeps prep2bg + gu3bg
        return .{ .k = k, .c = set.c, .texts = set.texts, .gu8 = set.gu8 and variant == 0 and gu8Fits(k3, rows) };
    }

    /// Variant `v`'s grouped gate|up and down GEMVs (`group_variants`, v >= 1) on today's grouped texts or, `m3`, on
    /// hw4b's, made on first use: the texts on the variant's header.
    fn groupGemvs(self: *Banked, v: u8, m3: bool) !GroupGemvs {
        const slot = if (m3) &self.banked_gvm3[v] else &self.banked_gv[v];
        if (slot.*) |g| return g;
        const gu = (if (m3)
            ks.newKernel(group_gu_m3_names[v].ptr, &gu3b_inputs, &gu3_outputs, text_gu3bgm3, group_headers[v].ptr, true)
        else
            ks.newKernel(group_gu_names[v].ptr, &gu3b_inputs, &gu3_outputs, text_gu3bg, group_headers[v].ptr, true)) orelse return error.KernelCreateFailed;
        errdefer freeKernel(gu);
        const down = (if (m3)
            ks.newKernel(group_down_m3_names[v].ptr, &down3b_inputs, &down3_outputs, text_down3bgm3, group_headers[v].ptr, true)
        else
            ks.newKernel(group_down_names[v].ptr, &down3b_inputs, &down3_outputs, text_down3bg, group_headers[v].ptr, true)) orelse return error.KernelCreateFailed;
        slot.* = .{ .gu = gu, .down = down };
        return slot.*.?;
    }

    /// dfin3's unit scores in the banked down (f32 [banked_rows_max], made and evaluated on first use).
    fn bankedOnes(self: *Banked) !mlx.mlx_array {
        if (self.banked_ones) |o| return o;
        const buf: [banked_rows_max]f32 = @splat(1.0);
        const o = mlx.mlx_array_new_data(&buf, &[_]c_int{banked_rows_max}, 1, .float32);
        if (o.ctx == null) return error.MlxError;
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_array_eval(o));
        self.banked_ones = o;
        return o;
    }

    /// The banked decode's gate|up (prep2b, gu3b, fin7cb): x [rows, hidden] (bf16, f16 or f32: row-slot r's row), `ids`
    /// [rows] u32 or i32, row-slot r's id packed (bank << 24 | slot row) -> the down's prepared input hd f16 [rows, inter]
    /// (lane order) and its 128-block sums hs f32 [rows, inter / 128]. `banks`: base, ext, transient, each its own slot
    /// count (a bank no id names may repeat another's arrays). Needs a clamped top-1 geometry; rows 1..banked_rows_max.
    /// The caller frees both.
    pub fn bankedGateUp(self: *Banked, k3: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [3]Kernels.Bank, rows: usize) ![2]mlx.mlx_array {
        return self.bankedGateUpSel(k3, s, x, ids, banks, rows, .configured);
    }

    /// `bankedGateUp` on `sel`'s texts (hw4b's check and tests): the same bytes for every selection.
    pub fn bankedGateUpSel(self: *Banked, k3: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [3]Kernels.Bank, rows: usize, sel: BankedSel) ![2]mlx.mlx_array {
        return self.bankedGateUpSelV(k3, s, x, ids, banks, rows, sel, 0);
    }

    /// `bankedGateUp` with the grouped gate|up GEMV of `variant` (hw2's `group_variants`; any but 0 needs `banked_group`
    /// 4): the same bytes.
    pub fn bankedGateUpV(self: *Banked, k3: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [3]Kernels.Bank, rows: usize, variant: u8) ![2]mlx.mlx_array {
        return self.bankedGateUpSelV(k3, s, x, ids, banks, rows, .configured, variant);
    }

    /// `bankedGateUp` on `sel`'s texts with the grouped gate|up GEMV of `variant` on them (`bankedSetSelV`): the same
    /// bytes for every selection and variant.
    pub fn bankedGateUpSelV(self: *Banked, k3: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [3]Kernels.Bank, rows: usize, sel: BankedSel, variant: u8) ![2]mlx.mlx_array {
        const set = try self.bankedSetSelV(k3, rows, sel, variant);
        const k = &set.k;
        const c = set.c;
        const b = banks;
        if (set.gu8) {
            // gu8b: prep2bg's arithmetic and gu3bg's GEMV in one launch, then fin7's
            const k8 = try self.gu8Kernel(set.texts == .g4m3);
            var o: [4]mlx.mlx_array = undefined; // the split-K gate|up planes, then the blocks' input sums
            try ks.applyConfig(k8, c.gu8, &.{ x, ids, b[0].gate_suh, b[1].gate_suh, b[2].gate_suh, b[0].up_suh, b[1].up_suh, b[2].up_suh, b[0].gate_t, b[1].gate_t, b[2].gate_t, b[0].up_t, b[1].up_t, b[2].up_t }, &o, s);
            defer for (o) |a| {
                _ = mlx.mlx_array_free(a);
            };
            var hd8: [2]mlx.mlx_array = undefined;
            try ks.applyConfig(k.fin, c.fin7, &.{ o[0], o[1], ids, b[0].gate_svh, b[1].gate_svh, b[2].gate_svh, b[0].up_svh, b[1].up_svh, b[2].up_svh, b[0].down_suh, b[1].down_suh, b[2].down_suh, o[2], o[3] }, &hd8, s);
            return hd8;
        }
        var xp: [4]mlx.mlx_array = undefined; // xg, xu, and their per-block input sums
        try ks.applyConfig(k.prep, c.prep_h, &.{ x, ids, b[0].gate_suh, b[1].gate_suh, b[2].gate_suh, b[0].up_suh, b[1].up_suh, b[2].up_suh }, &xp, s);
        defer for (xp) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var z: [2]mlx.mlx_array = undefined; // the split-K gate|up planes
        try ks.applyConfig(k.gu, c.gu, &.{ xp[0], xp[1], ids, b[0].gate_t, b[1].gate_t, b[2].gate_t, b[0].up_t, b[1].up_t, b[2].up_t }, &z, s);
        defer for (z) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var hd: [2]mlx.mlx_array = undefined;
        try ks.applyConfig(k.fin, c.fin7, &.{ z[0], z[1], ids, b[0].gate_svh, b[1].gate_svh, b[2].gate_svh, b[0].up_svh, b[1].up_svh, b[2].up_svh, b[0].down_suh, b[1].down_suh, b[2].down_suh, xp[2], xp[3] }, &hd, s);
        return hd;
    }

    /// The banked decode's down (down3b, then dfin3b at top-1 with unit scores): `hd`, `hs` (`bankedGateUp`'s), the same
    /// ids and banks -> y f32 [rows, hidden], row-slot r's expert output at weight 1. The caller frees y.
    pub fn bankedDown(self: *Banked, k3: *Kernels3, s: mlx.mlx_stream, hd: mlx.mlx_array, hs: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [3]Kernels.Bank, rows: usize) !mlx.mlx_array {
        return self.bankedDownSel(k3, s, hd, hs, ids, banks, rows, .configured);
    }

    /// `bankedDown` on `sel`'s texts (hw4b's check and tests): the same bytes for every selection.
    pub fn bankedDownSel(self: *Banked, k3: *Kernels3, s: mlx.mlx_stream, hd: mlx.mlx_array, hs: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [3]Kernels.Bank, rows: usize, sel: BankedSel) !mlx.mlx_array {
        return self.bankedDownSelV(k3, s, hd, hs, ids, banks, rows, sel, 0);
    }

    /// `bankedDown` with the grouped down GEMV of `variant` (hw2's `group_variants`; any but 0 needs `banked_group` 4):
    /// the same bytes.
    pub fn bankedDownV(self: *Banked, k3: *Kernels3, s: mlx.mlx_stream, hd: mlx.mlx_array, hs: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [3]Kernels.Bank, rows: usize, variant: u8) !mlx.mlx_array {
        return self.bankedDownSelV(k3, s, hd, hs, ids, banks, rows, .configured, variant);
    }

    /// `bankedDown` on `sel`'s texts with the grouped down GEMV of `variant` on them (`bankedSetSelV`): the same bytes for
    /// every selection and variant.
    pub fn bankedDownSelV(self: *Banked, k3: *Kernels3, s: mlx.mlx_stream, hd: mlx.mlx_array, hs: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [3]Kernels.Bank, rows: usize, sel: BankedSel, variant: u8) !mlx.mlx_array {
        const set = try self.bankedSetSelV(k3, rows, sel, variant);
        const k = &set.k;
        const c = set.c;
        const ones = try self.bankedOnes();
        const b = banks;
        const zd = try one(k.down, c.down, &.{ hd, ids, b[0].down_t, b[1].down_t, b[2].down_t }, s);
        defer _ = mlx.mlx_array_free(zd);
        return one(k.dfin, c.dfin, &.{ zd, ids, ones, b[0].down_svh, b[1].down_svh, b[2].down_svh, hs }, s);
    }
};

/// A geometry's `Kernels3` and its banked state owned together (a caller that builds its own, as the tests do)
pub const Owned = struct {
    k: Kernels3,
    b: Banked,

    pub fn deinit(self: *Owned) void {
        self.b.deinit();
        self.k.deinit();
    }
};

// ── the process's banked state, one per `Kernels3` (root's cache of geometries) ──

const Entry = struct { key: *const Kernels3, b: Banked };
const cache_lock = Global(std.atomic.Value(u32), "deepseek_v41_banked_lock", std.atomic.Value(u32).init(0));
const cache = Global([16]?Entry, "deepseek_v41_banked_cache", @splat(null));

/// The banked state beside `k3` (root's `Kernels3` of the call's geometry), made at its first banked call
pub fn forKernels(k3: *const Kernels3) !*Banked {
    while (cache_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer cache_lock.get().*.store(0, .release);
    var free_slot: ?usize = null;
    for (&cache.get().*, 0..) |*slot, i| {
        if (slot.*) |*e| {
            if (e.key == k3) return &e.b;
        } else if (free_slot == null) free_slot = i;
    }
    const i = free_slot orelse return error.Exl3EngineCacheFull;
    cache.get().*[i] = .{ .key = k3, .b = Banked.init() };
    return &cache.get().*[i].?.b;
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ cache_lock, cache };

