//! wave3 (`Config.banked_wave3`, at `banked_group` 4; banked.zig's `Banked.wave3`, the plugin's
//! TURBOEXL3_GLM53F_BANKED_WAVE3): the grouped banked call's five launches a wave (prep2bg, the gate|up GEMV, fin7cbg,
//! the down GEMV, dfin3bg) as three, on today's grouped texts (g4) or hw4b's (g4m3), at any grouped GEMV variant:
//! - prep2bg zeroing the call's arrival counters in its own launch (`text_prep2bgw3`: `w3_zero` before its skip line);
//! - the gate|up GEMV with fin7cbg after it (`text_gu3bgw3`, `text_gu3bgm3w3`). A finish unit is one of fin7cbg's
//!   threadgroups: a 128-wide block of intermediate of one row-slot. Its planes come from the 8 / BT column
//!   threadgroups over the block at each of the KS / SPT K threadgroups of the row-slot's lead (a lead threadgroup
//!   computes for its members, up to four row-slots of one packed id; every row-slot but a skipped one is in exactly one
//!   lead). Each GEMV threadgroup stores its planes through coherent(device) pointers, fences them device-wide and counts
//!   one arrival a member at the member's unit (DEC-ONE's idiom, dense.zig): the member's simdgroup in the threadgroup
//!   whose count completes the unit runs fin7cbg's statements for it;
//! - the down GEMV with dfin3bg after it (`text_down3bgw3`, `text_down3bgm3w3`). A unit is one of dfin3bg's
//!   threadgroups: a 128-wide block of hidden of one row, at the banked call's top-1 one row-slot, counted the same way
//!   after the gate|up units' counters; the threadgroup whose count completes a unit runs dfin3bg's statements for it
//!   (its H128 takes the whole threadgroup: its units in member order). A skipped row-slot's threadgroups store its y
//!   row's zeros, dfin3bg's row there, and return.
//! Every value-setting statement is the five launches' own (the GEMVs' bodies; fin7cbg and dfin3bg with their grid lines,
//! `decode.fin7_1b` and `dfin3_1b`, in place of the unit's), so each rounding happens where the five launches round it:
//! the same hd, hs and y, word for word (glm53f/wave3_test.zig holds them so on the GPU). Nothing waits: the last
//! arriver finishes.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const config = @import("../../core/config.zig");
const decode = @import("../../core/decode.zig");
const banked = @import("banked.zig");
const Kernels3 = decode.Kernels3;
const Kernels = decode.Kernels;
const freeKernel = decode.freeKernel;
const group_variants = banked.group_variants;

// ── The texts ──

/// prep2bg with the call's arrival counters zeroed before its skip line
pub const text_prep2bgw3: [:0]const u8 = decode.prep2_1 ++ w3_zero ++ skip_rk ++ prep2b_expert ++ decode.prep2_row ++ decode.prep2_body;
/// the gate|up GEMV (today's grouped body or hw4b's) with fin7cbg for each unit its counts complete
pub const text_gu3bgw3: [:0]const u8 = gu_head ++ gu3g_3 ++ gu_tail;
pub const text_gu3bgm3w3: [:0]const u8 = gu_head ++ gu3g_3m3 ++ gu_tail;
/// the down GEMV (today's grouped body or hw4b's) with dfin3bg for each unit its counts complete
pub const text_down3bgw3: [:0]const u8 = dn_head ++ down3g_3 ++ dn_tail;
pub const text_down3bgm3w3: [:0]const u8 = dn_head ++ down3g_3m3 ++ dn_tail;

/// the grouped gate|up GEMV to its body: gu3bg's pieces with the coherent planes after its constants
const gu_head = decode.gu3_1 ++ w3_gu_head ++ decode.rk_line ++ gu3g_2 ++ gu3b_w;
/// the arrival, then fin7cbg with the unit's block and row-slot (`w3_gu_unit`) for its grid lines (`decode.fin7_1b`)
pub const gu_finish = decode.fin7_1a ++ w3_gu_unit ++ skip_rk ++ fin7b_expert ++ decode.fin7_2 ++ decode.fin7_planes ++ decode.fin7_3 ++ decode.fin7_sums ++ decode.fin7_4 ++ decode.fin7_swiglu_clamp ++ decode.fin7_5;
const gu_tail = w3_gu_arrive ++ gu_finish ++ w3_close;
/// the grouped down GEMV to its body: down3bg's pieces with the coherent planes and dfin3's exchange after its constants,
/// and a skipped row-slot's zero y row after its rk line
const dn_head = decode.down3_1 ++ w3_dn_head ++ decode.rk_line ++ w3_dn_skip ++ down3g_2 ++ down3b_w;
/// the arrival, then dfin3bg with the unit's block and row (`w3_dn_unit`) for its grid lines and exchange
/// (`decode.dfin3_1b`)
pub const dn_finish = decode.dfin3_1a ++ w3_dn_unit ++ decode.dfin3_1c ++ skip_rk_next ++ dfin3b_expert ++ decode.dfin3_2;
const dn_tail = w3_dn_arrive ++ dn_finish ++ w3_close;

/// wave3's pieces
pub const w3_zero = @embedFile("kernels/banked/w3_zero.metal");
pub const w3_gu_head = @embedFile("kernels/banked/w3_gu_head.metal");
pub const w3_gu_arrive = @embedFile("kernels/banked/w3_gu_arrive.metal");
pub const w3_gu_unit = @embedFile("kernels/banked/w3_gu_unit.metal");
pub const w3_dn_head = @embedFile("kernels/banked/w3_dn_head.metal");
pub const w3_dn_skip = @embedFile("kernels/banked/w3_dn_skip.metal");
pub const w3_dn_arrive = @embedFile("kernels/banked/w3_dn_arrive.metal");
pub const w3_dn_unit = @embedFile("kernels/banked/w3_dn_unit.metal");
pub const w3_close = @embedFile("kernels/banked/w3_close.metal");
/// banked.zig's grouped pieces these texts run as they are
const skip_rk = @embedFile("kernels/banked/skip_rk.metal");
const skip_rk_next = @embedFile("kernels/banked/skip_rk_next.metal");
const prep2b_expert = @embedFile("kernels/banked/prep2b_expert.metal");
const fin7b_expert = @embedFile("kernels/banked/fin7b_expert.metal");
const dfin3b_expert = @embedFile("kernels/banked/dfin3b_expert.metal");
const gu3b_w = @embedFile("kernels/banked/gu3b_w.metal");
const down3b_w = @embedFile("kernels/banked/down3b_w.metal");
const gu3g_2 = @embedFile("kernels/banked/gu3g_2.metal");
const gu3g_3 = @embedFile("kernels/banked/gu3g_3.metal");
const gu3g_3m3 = @embedFile("kernels/banked/gu3g_3m3.metal");
const down3g_2 = @embedFile("kernels/banked/down3g_2.metal");
const down3g_3 = @embedFile("kernels/banked/down3g_3.metal");
const down3g_3m3 = @embedFile("kernels/banked/down3g_3m3.metal");
/// The fused launches' header: fin7's (`decode.header5`: its H128 in registers, the coherent pointers) with the grouped
/// GEMVs' members and core after it (banked.zig's `header_group` past `decode.header`), a variant's defines before it
const gemv_group = @embedFile("kernels/banked/gemv_group.metal");
pub const header_w3: [:0]const u8 = decode.header5 ++ gemv_group;

/// The launches' operands: prep2bg's with the counters; gu3bg's inputs, then fin7cbg's but its planes and slots, then the
/// counters; fin7cbg's outputs, then the planes (scratch); down3bg's inputs, then dfin3bg's but its planes and slots,
/// then the counters; dfin3bg's output, then the planes (scratch)
pub const prep_outputs = decode.prep2_outputs ++ [_][*:0]const u8{"cnt"};
pub const gu_inputs = banked.gu3b_inputs ++ banked.fin7b_inputs[3..].* ++ [_][*:0]const u8{"cnt"};
pub const gu_outputs = decode.fin3_outputs ++ [_][*:0]const u8{ "zgp", "zup" };
pub const down_inputs = banked.down3b_inputs ++ banked.dfin3b_inputs[2..].* ++ [_][*:0]const u8{"cnt"};
pub const down_outputs = decode.down_outputs ++ [_][*:0]const u8{"zdp"};

/// Which grouped texts a fused set runs: today's (g4) or hw4b's (g4m3)
pub const Texts = enum(u1) { g4, g4m3 };

/// Each variant's header (its defines before `header_w3`) and the fused kernels' names per texts and variant (variant
/// 0's the stock names, each other `_<variant>`: MLX keys a custom kernel's library by its name)
const headers: [group_variants.len][:0]const u8 = blk: {
    var h: [group_variants.len][:0]const u8 = undefined;
    for (group_variants, 0..) |v, i| h[i] = v.defines ++ header_w3;
    break :blk h;
};
const gu_names: [group_variants.len][:0]const u8 = blk: {
    var n: [group_variants.len][:0]const u8 = undefined;
    n[0] = "turboexl3_gu3bgw3";
    for (group_variants[1..], 1..) |v, i| n[i] = "turboexl3_gu3bgw3_" ++ v.name;
    break :blk n;
};
const gu_m3_names: [group_variants.len][:0]const u8 = blk: {
    var n: [group_variants.len][:0]const u8 = undefined;
    n[0] = "turboexl3_gu3bgm3w3";
    for (group_variants[1..], 1..) |v, i| n[i] = "turboexl3_gu3bgm3w3_" ++ v.name;
    break :blk n;
};
const down_names: [group_variants.len][:0]const u8 = blk: {
    var n: [group_variants.len][:0]const u8 = undefined;
    n[0] = "turboexl3_down3bgw3";
    for (group_variants[1..], 1..) |v, i| n[i] = "turboexl3_down3bgw3_" ++ v.name;
    break :blk n;
};
const down_m3_names: [group_variants.len][:0]const u8 = blk: {
    var n: [group_variants.len][:0]const u8 = undefined;
    n[0] = "turboexl3_down3bgm3w3";
    for (group_variants[1..], 1..) |v, i| n[i] = "turboexl3_down3bgm3w3_" ++ v.name;
    break :blk n;
};
pub const prep_name = "turboexl3_prep2bgw3";
/// Every kernel name wave3 makes, for the tests: distinct, and none of banked.zig's
pub const kernel_names = [_][:0]const u8{prep_name} ++ gu_names ++ gu_m3_names ++ down_names ++ down_m3_names;

/// The three launches' configs at one row-slot count
pub const Cfg = struct { prep: mlx.mlx_fast_metal_kernel_config, gu: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config };

/// The template constants each fused text reads (`launchCfgs` passes exactly these; the CPU test holds each text to them)
pub const prep_consts = [_][]const u8{ "DIM", "TOPK", "NCNT", "NTH" };
pub const gu_consts = [_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "RKC", "KS", "SPT", "BT", "GROUP", "DEFER", "NBH", "LIMIT" };
pub const down_consts = [_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "RKC", "KS", "SPT", "BT", "GROUP", "DEFER", "TOPK", "OUT_T", "NBI" };

/// The launch shape at `rows` row-slots: the grouped texts' own (`Banked.cfgsBankedG`'s and `cfgsBankedGM3`'s, through
/// `Kernels3.buildCfgsGroup` at 4 row-slots a lead: form 9's bt1 at one row-slot and bt9 above, kdiv9's K threadgroups,
/// GROUP 4 from two row-slots), and the counters it takes: a finish unit's arrivals are its block's column threadgroups
/// (`gc`) times its row-slot's K threadgroups (`z_gu`, `z_dn`)
pub const Shape = struct {
    bt: u32,
    group: u32,
    z_gu: u32,
    z_dn: u32,
    spt_gu: u32,
    spt_dn: u32,
    /// column threadgroups over a 128-wide block (8 / BT)
    gc: u32,
    /// the counters: the gate|up units' (rows x inter / 128), then the down units' (rows x hidden / 128), at least 8 (MLX
    /// binds an input of fewer than 8 elements in the constant address space, where no atomic is)
    ncnt: u32,
};

/// The launch shape at `rows` row-slots of `k3`'s geometry; error.Exl3EngineNotImplemented for a geometry the banked call
/// does not take (top-1 and clamped) or a launch whose threadgroup memory passes Metal's 32 KB (the GEMV's reduction
/// buffers, the arrival verdicts and the down's exchange), error.BadExl3Shape for a tile count the grouped texts do not
/// take.
pub fn shapeFor(k3: *const Kernels3, rows: usize) !Shape {
    const g = k3.geo;
    if (g.topk != 1 or g.limit == 0) return error.Exl3EngineNotImplemented;
    if (rows == 0 or rows > banked.banked_rows_max) return error.BadExl3Shape;
    const bt: u32 = if (rows == 1) k3.bt1 else k3.bt9;
    if (bt != 1 and bt != 2 and bt != 4 and bt != 8) return error.BadExl3Shape;
    const z_gu = k3.kThreadgroups(k3.ks_gu, rows, k3.kdiv9);
    const z_dn = k3.kThreadgroups(k3.ks_dn, rows, k3.kdiv9d orelse k3.kdiv9);
    const group: u32 = if (rows >= 2) 4 else 1;
    const sh: Shape = .{
        .bt = bt,
        .group = group,
        .z_gu = z_gu,
        .z_dn = z_dn,
        .spt_gu = k3.ks_gu / z_gu,
        .spt_dn = k3.ks_dn / z_dn,
        .gc = 8 / bt,
        .ncnt = @max(@as(u32, @intCast(rows)) * (g.inter / 128 + g.hidden / 128), 8),
    };
    const red_gu = 4 * sh.spt_gu * 4 * group * 16 * bt;
    const red_dn = 4 * sh.spt_dn * 4 * group * 16 * bt;
    if (red_gu + 16 > 32768 or red_dn + 16 + 512 > 32768) return error.Exl3EngineNotImplemented;
    return sh;
}

/// The three launches' configs at `rows` row-slots (`shapeFor`'s shape): prep2bg's grid with the counters as its fifth
/// output; the GEMVs' grids (`buildCfgsGroup`'s) with fin7cbg's and dfin3bg's outputs before the planes; the caller
/// frees them.
pub fn launchCfgs(k3: *const Kernels3, rows: usize) !Cfg {
    const sh = try shapeFor(k3, rows);
    const g = k3.geo;
    const rk: u32 = @intCast(rows); // top-1: a row-slot a row
    const rkc: c_int = @intCast(rk);
    const H: c_int = @intCast(g.hidden);
    const I: c_int = @intCast(g.inter);
    const nbh: c_int = @intCast(g.hidden / 128);
    const nbi: c_int = @intCast(g.inter / 128);
    const nhw = ks.TemplateArg{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } };
    const cb = ks.TemplateArg{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
    const wb = ks.TemplateArg{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
    const hid = ks.TemplateArg{ .name = "HID", .value = .{ .int = H } };
    const inter_t = ks.TemplateArg{ .name = "INTER", .value = .{ .int = I } };
    const rkt = ks.TemplateArg{ .name = "RKC", .value = .{ .int = rkc } };
    const topk = ks.TemplateArg{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } };
    const ks_gu_t = ks.TemplateArg{ .name = "KS", .value = .{ .int = @intCast(k3.ks_gu) } };
    const ks_dn_t = ks.TemplateArg{ .name = "KS", .value = .{ .int = @intCast(k3.ks_dn) } };
    const spt_gu_t = ks.TemplateArg{ .name = "SPT", .value = .{ .int = @intCast(sh.spt_gu) } };
    const spt_dn_t = ks.TemplateArg{ .name = "SPT", .value = .{ .int = @intCast(sh.spt_dn) } };
    const bt_t = ks.TemplateArg{ .name = "BT", .value = .{ .int = @intCast(sh.bt) } };
    const grp_t = ks.TemplateArg{ .name = "GROUP", .value = .{ .int = @intCast(sh.group) } };
    const def_t = ks.TemplateArg{ .name = "DEFER", .value = .{ .int = if (k3.defer_affine) 1 else 0 } };
    const nbh_t = ks.TemplateArg{ .name = "NBH", .value = .{ .int = nbh } };
    const nbi_t = ks.TemplateArg{ .name = "NBI", .value = .{ .int = nbi } };
    const lim_t = ks.TemplateArg{ .name = "LIMIT", .value = .{ .int = @intCast(g.limit) } };
    const out_t = ks.TemplateArg{ .name = "OUT_T", .value = .{ .dtype = k3.out_dtype } };
    const ncnt_t = ks.TemplateArg{ .name = "NCNT", .value = .{ .int = @intCast(sh.ncnt) } };
    const nth_t = ks.TemplateArg{ .name = "NTH", .value = .{ .int = @intCast((g.hidden / 128) * 32 * rk) } };
    const dim_t = ks.TemplateArg{ .name = "DIM", .value = .{ .int = H } };
    const bw: u32 = 16 * sh.bt;
    const prep = try ks.buildConfig(.{
        .grid = .{ (g.hidden / 128) * 32, rk, 1 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ rkc, H }, &[_]c_int{ rkc, H }, &[_]c_int{ rkc, nbh }, &[_]c_int{ rkc, nbh }, &[_]c_int{@intCast(sh.ncnt)} },
        .out_dtypes = &.{ .float16, .float16, .float32, .float32, .uint32 },
        .template = &.{ dim_t, topk, ncnt_t, nth_t },
    });
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(prep);
    const gu = try ks.buildConfig(.{
        .grid = .{ (g.inter / bw) * 128, rk, sh.z_gu },
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, nbi }, &[_]c_int{ @intCast(k3.ks_gu), rkc, I }, &[_]c_int{ @intCast(k3.ks_gu), rkc, I } },
        .out_dtypes = &.{ .float16, .float32, .float32, .float32 },
        .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t, nbh_t, lim_t },
    });
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gu);
    const down = try ks.buildConfig(.{
        .grid = .{ (g.hidden / bw) * 128, rk, sh.z_dn },
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ rkc, H }, &[_]c_int{ @intCast(k3.ks_dn), rkc, H } },
        .out_dtypes = &.{ k3.out_dtype, .float32 },
        .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_dn_t, spt_dn_t, bt_t, grp_t, def_t, topk, out_t, nbi_t },
    });
    return .{ .prep = prep, .gu = gu, .down = down };
}

/// A geometry's wave3 state beside its banked state (`Banked.wave3`): the fused kernels and configs, made on first use.
pub const Wave3 = struct {
    /// `Config.banked_wave3` as configured at the geometry's first whole banked call (null until then): kept for the
    /// geometry's life, as `Banked.init` keeps `banked_group` and `banked_m3`
    on: ?bool = null,
    prep: ?mlx.mlx_fast_metal_kernel = null,
    /// the fused gate|up and down kernels per texts (`Texts`) and grouped GEMV variant
    gu: [2][group_variants.len]?mlx.mlx_fast_metal_kernel = @splat(@splat(null)),
    down: [2][group_variants.len]?mlx.mlx_fast_metal_kernel = @splat(@splat(null)),
    /// the configs per row-slot count (every texts' and variant's: their launch shapes are the same)
    cfg: [banked.banked_rows_max + 1]?Cfg = @splat(null),
    /// the calls run (`moe`), for the tests: a configured call ran wave3
    calls: u64 = 0,

    pub fn deinit(self: *Wave3) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep, cc.gu, cc.down }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        if (self.prep) |k| freeKernel(k);
        self.prep = null;
        for (&self.gu, &self.down) |*gr, *dr| for (gr, dr) |*gk, *dk| {
            if (gk.*) |k| freeKernel(k);
            if (dk.*) |k| freeKernel(k);
            gk.* = null;
            dk.* = null;
        };
    }

    /// `Config.banked_wave3` as configured at this geometry's first whole banked call
    pub fn configured(self: *Wave3) bool {
        if (self.on == null) self.on = config.get().banked_wave3;
        return self.on.?;
    }

    fn prepKernel(self: *Wave3) !mlx.mlx_fast_metal_kernel {
        if (self.prep) |k| return k;
        const k = ks.newKernel(prep_name, &banked.prep2b_inputs, &prep_outputs, text_prep2bgw3, decode.header, true) orelse return error.KernelCreateFailed;
        self.prep = k;
        return k;
    }

    /// The fused gate|up kernel of `texts` at variant `v`, made on first use
    fn guKernel(self: *Wave3, texts: Texts, v: u8) !mlx.mlx_fast_metal_kernel {
        const slot = &self.gu[@intFromEnum(texts)][v];
        if (slot.*) |k| return k;
        const k = (if (texts == .g4m3)
            ks.newKernel(gu_m3_names[v].ptr, &gu_inputs, &gu_outputs, text_gu3bgm3w3, headers[v].ptr, true)
        else
            ks.newKernel(gu_names[v].ptr, &gu_inputs, &gu_outputs, text_gu3bgw3, headers[v].ptr, true)) orelse return error.KernelCreateFailed;
        slot.* = k;
        return k;
    }

    /// The fused down kernel of `texts` at variant `v`, made on first use
    fn downKernel(self: *Wave3, texts: Texts, v: u8) !mlx.mlx_fast_metal_kernel {
        const slot = &self.down[@intFromEnum(texts)][v];
        if (slot.*) |k| return k;
        const k = (if (texts == .g4m3)
            ks.newKernel(down_m3_names[v].ptr, &down_inputs, &down_outputs, text_down3bgm3w3, headers[v].ptr, true)
        else
            ks.newKernel(down_names[v].ptr, &down_inputs, &down_outputs, text_down3bgw3, headers[v].ptr, true)) orelse return error.KernelCreateFailed;
        slot.* = k;
        return k;
    }

    /// The configs at `rows` row-slots (`launchCfgs`), in a table of their own
    fn cfgs(self: *Wave3, k3: *const Kernels3, rows: usize) !Cfg {
        if (rows == 0 or rows > banked.banked_rows_max) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const c = try launchCfgs(k3, rows);
        self.cfg[rows] = c;
        return c;
    }

    /// The whole call's three launches on `texts` with the grouped GEMVs of `vgu` (gate|up) and `vdn` (down): x, ids,
    /// banks and rows as `Banked.bankedGateUp` takes them, `ones` dfin3's unit scores -> hd f16 [rows, inter], hs f32
    /// [rows, inter / 128] and y [rows, hidden] in `k3`'s output dtype, the five launches' words (a skipped row-slot's y
    /// row 0, its hd and hs rows unwritten). Every refusal is made before a launch. The caller frees all three.
    pub fn moe(self: *Wave3, k3: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [3]Kernels.Bank, rows: usize, texts: Texts, vgu: u8, vdn: u8, ones: mlx.mlx_array) ![3]mlx.mlx_array {
        if (vgu >= group_variants.len or vdn >= group_variants.len) return error.BadExl3Shape;
        const c = try self.cfgs(k3, rows);
        const kp = try self.prepKernel();
        const kg = try self.guKernel(texts, vgu);
        const kd = try self.downKernel(texts, vdn);
        const b = banks;
        var xp: [5]mlx.mlx_array = undefined; // xg, xu, their per-block input sums, the zeroed counters
        try ks.applyConfig(kp, c.prep, &.{ x, ids, b[0].gate_suh, b[1].gate_suh, b[2].gate_suh, b[0].up_suh, b[1].up_suh, b[2].up_suh }, &xp, s);
        defer for (xp) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var h: [4]mlx.mlx_array = undefined; // hd, hs, then the gate|up planes (scratch)
        try ks.applyConfig(kg, c.gu, &.{ xp[0], xp[1], ids, b[0].gate_t, b[1].gate_t, b[2].gate_t, b[0].up_t, b[1].up_t, b[2].up_t, b[0].gate_svh, b[1].gate_svh, b[2].gate_svh, b[0].up_svh, b[1].up_svh, b[2].up_svh, b[0].down_suh, b[1].down_suh, b[2].down_suh, xp[2], xp[3], xp[4] }, &h, s);
        _ = mlx.mlx_array_free(h[2]);
        _ = mlx.mlx_array_free(h[3]);
        errdefer {
            _ = mlx.mlx_array_free(h[0]);
            _ = mlx.mlx_array_free(h[1]);
        }
        var o: [2]mlx.mlx_array = undefined; // y, then the down planes (scratch)
        try ks.applyConfig(kd, c.down, &.{ h[0], ids, b[0].down_t, b[1].down_t, b[2].down_t, ones, b[0].down_svh, b[1].down_svh, b[2].down_svh, h[1], xp[4] }, &o, s);
        _ = mlx.mlx_array_free(o[1]);
        self.calls += 1;
        return .{ h[0], h[1], o[0] };
    }
};

const testing = std.testing;

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

test "turboexl3 banked wave3 texts: the five launches' pieces in order, fin7cbg's and dfin3bg's grid lines swapped for a finish unit's and wave3's pieces added; every operand read, every template constant passed" {
    // the cuts keep fin7_1 and dfin3_1 whole
    try testing.expectEqualStrings(decode.fin7_1, decode.fin7_1a ++ decode.fin7_1b);
    try testing.expectEqualStrings(decode.dfin3_1, decode.dfin3_1a ++ decode.dfin3_1b ++ decode.dfin3_1c);
    // prep2bg with the zeroing before its skip line, nothing else
    try testing.expect(std.mem.startsWith(u8, banked.text_prep2bg, decode.prep2_1 ++ skip_rk));
    try testing.expectEqualStrings(decode.prep2_1 ++ w3_zero ++ banked.text_prep2bg[decode.prep2_1.len..], text_prep2bgw3);
    // each GEMV: the grouped text whole, the coherent planes after its constants (and the down's skip after its rk line),
    // then the arrival and the finish: fin7cbg / dfin3bg with the unit's lines in place of their grid lines
    try testing.expectEqualStrings(decode.fin7_1 ++ skip_rk, banked.text_fin7cbg[0 .. decode.fin7_1.len + skip_rk.len]);
    try testing.expectEqualStrings(decode.fin7_1a ++ w3_gu_unit ++ banked.text_fin7cbg[decode.fin7_1.len..], gu_finish);
    try testing.expectEqualStrings(decode.dfin3_1a ++ w3_dn_unit ++ banked.text_dfin3bg[decode.dfin3_1a.len + decode.dfin3_1b.len ..], dn_finish);
    try testing.expect(std.mem.startsWith(u8, banked.text_dfin3bg, decode.dfin3_1 ++ skip_rk_next));
    const Pair = struct { base: []const u8, fused: []const u8 };
    for ([_]Pair{ .{ .base = banked.text_gu3bg, .fused = text_gu3bgw3 }, .{ .base = banked.text_gu3bgm3, .fused = text_gu3bgm3w3 } }) |p| {
        const n1 = decode.gu3_1.len;
        try testing.expectEqualStrings(decode.gu3_1 ++ w3_gu_head, p.fused[0 .. n1 + w3_gu_head.len]);
        try testing.expectEqualStrings(p.base[n1..], p.fused[n1 + w3_gu_head.len ..][0 .. p.base.len - n1]);
        try testing.expectEqualStrings(w3_gu_arrive ++ gu_finish ++ w3_close, p.fused[w3_gu_head.len + p.base.len ..]);
    }
    for ([_]Pair{ .{ .base = banked.text_down3bg, .fused = text_down3bgw3 }, .{ .base = banked.text_down3bgm3, .fused = text_down3bgm3w3 } }) |p| {
        const n1 = decode.down3_1.len;
        const n2 = n1 + decode.rk_line.len;
        try testing.expectEqualStrings(decode.down3_1 ++ w3_dn_head ++ decode.rk_line ++ w3_dn_skip, p.fused[0 .. n2 + w3_dn_head.len + w3_dn_skip.len]);
        const at = n2 + w3_dn_head.len + w3_dn_skip.len;
        try testing.expectEqualStrings(p.base[n2..], p.fused[at..][0 .. p.base.len - n2]);
        try testing.expectEqualStrings(w3_dn_arrive ++ dn_finish ++ w3_close, p.fused[at + p.base.len - n2 ..]);
    }
    // the pin file: [0] the expected arrivals (the unit's column threadgroups times its K threadgroups), in both arrivals;
    // [1] the gate|up unit's counter, [2] the down unit's, after the gate|up units'; [3] the zeroing, before prep2bg's skip
    // line; [4], [5], [6] the coherent planes; [7] a skipped row-slot's zero store; [8] fin7_1b's line of the block and
    // [9] dfin3_1b's line of the row, which no fused text holds; [10] the exchange, once in a fused down text, at kernel
    // scope (never in its finish); [11] the gate|up finisher's choice (the member's simdgroup) and [12] the down's (each
    // unit the threadgroup completed, in member order)
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/banked_wave3.metal"));
    try testing.expectEqual(@as(usize, 1), count(w3_gu_arrive, pin[0]));
    try testing.expectEqual(@as(usize, 1), count(w3_dn_arrive, pin[0]));
    try testing.expectEqual(@as(usize, 1), count(w3_gu_arrive, pin[1]));
    try testing.expectEqual(@as(usize, 1), count(w3_dn_arrive, pin[2]));
    try testing.expectEqual(@as(usize, 1), count(w3_zero, pin[3]));
    try testing.expect(std.mem.indexOf(u8, text_prep2bgw3, pin[3]).? < std.mem.indexOf(u8, text_prep2bgw3, skip_rk).?);
    try testing.expectEqual(@as(usize, 1), count(w3_gu_head, pin[4]));
    try testing.expectEqual(@as(usize, 1), count(w3_gu_head, pin[5]));
    try testing.expectEqual(@as(usize, 1), count(w3_dn_head, pin[6]));
    try testing.expectEqual(@as(usize, 1), count(w3_dn_skip, pin[7]));
    try testing.expectEqual(@as(usize, 1), count(decode.fin7_1b, pin[8]));
    try testing.expectEqual(@as(usize, 1), count(decode.dfin3_1b, pin[9]));
    try testing.expectEqual(@as(usize, 1), count(decode.dfin3_1b, pin[10]));
    try testing.expectEqual(@as(usize, 0), count(dn_finish, pin[10]));
    for ([_][]const u8{ text_gu3bgw3, text_gu3bgm3w3 }) |t| {
        try testing.expectEqual(@as(usize, 0), count(t, pin[8]));
        try testing.expectEqual(@as(usize, 1), count(t, pin[11]));
    }
    for ([_][]const u8{ text_down3bgw3, text_down3bgm3w3 }) |t| {
        try testing.expectEqual(@as(usize, 0), count(t, pin[9]));
        try testing.expectEqual(@as(usize, 1), count(t, pin[10]));
        try testing.expectEqual(@as(usize, 1), count(t, pin[12]));
    }
    // every operand named in its text, none asking MLX for a shape buffer; within Metal's 31 buffers
    const Ops = struct { text: []const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, consts: []const []const u8 };
    for ([_]Ops{
        .{ .text = text_prep2bgw3, .ins = &banked.prep2b_inputs, .outs = &prep_outputs, .consts = &prep_consts },
        .{ .text = text_gu3bgw3, .ins = &gu_inputs, .outs = &gu_outputs, .consts = &gu_consts },
        .{ .text = text_gu3bgm3w3, .ins = &gu_inputs, .outs = &gu_outputs, .consts = &gu_consts },
        .{ .text = text_down3bgw3, .ins = &down_inputs, .outs = &down_outputs, .consts = &down_consts },
        .{ .text = text_down3bgm3w3, .ins = &down_inputs, .outs = &down_outputs, .consts = &down_consts },
    }) |o| {
        for (o.ins) |n| try testing.expect(std.mem.indexOf(u8, o.text, std.mem.span(n)) != null);
        for (o.outs) |n| try testing.expect(std.mem.indexOf(u8, o.text, std.mem.span(n)) != null);
        for (o.ins) |n| for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| {
            var buf: [32]u8 = undefined;
            try testing.expect(std.mem.indexOf(u8, o.text, try std.fmt.bufPrint(&buf, "{s}{s}", .{ std.mem.span(n), suf })) == null);
        };
        try testing.expect(o.ins.len + o.outs.len <= 31);
        for (o.consts) |n| try testing.expect(std.mem.indexOf(u8, o.text, n) != null);
    }
    // the kernel names: distinct, and none of the grouped texts' own
    for (kernel_names, 0..) |a, i| {
        for (kernel_names[i + 1 ..]) |b| try testing.expect(!std.mem.eql(u8, a, b));
        for (banked.group_kernel_names) |b| try testing.expect(!std.mem.eql(u8, a, b));
    }
}
