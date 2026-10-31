//! The routed decode kernels (rows <= DECODE_ROWS_MAX). `moe` (root.zig) runs the form `Config.decode_form` names, each
//! with form 3's bytes (config.zig describes them): form 3's split-K chain and its variants 7, 8 and 9 (the default) in
//! `Kernels3`, forms 5 and 6 in `Kernels5`. Form 1 (`Kernels`), described here, is the plainest and is not served: two
//! launches per MoE layer, any admitted rate, codebook, window and 128-aligned geometry, every constant a template int of
//! one Metal text per kernel; the GPU tests hold it to the host reference.
//!
//! Launch 1, `turboexl3_gu`, one 128-thread threadgroup per (128-wide intermediate block, slot, row): the four simdgroups
//! rotate the row's input for the slot's expert (f16 round, suh, H128 per block as a 5-shuffle + 2-register butterfly,
//! 1/sqrt(128), f16 round: `format.prepareInput`) into threadgroup memory once, then split the input tiles four ways and
//! stream the gate and up trellis tiles of the block's 8 output tiles through the lane plan (`lane.zig`), decode in
//! registers and accumulate f32; the partial sums meet in threadgroup memory, simdgroup 0 rounds to f16 (`innerGemv`),
//! rotates the block's 128 outputs (H128), scales by svh and rounds (`finishOutput`), and writes h = silu(gate) * up as f32.
//!
//! Launch 2, `turboexl3_down`, one threadgroup per (128-wide hidden block, row): for each of the row's slots in order it
//! rotates that slot's h (f16 round, down suh, H128, f16 round), streams the down tiles of its 8 output tiles, reduces,
//! rounds, rotates, scales by svh, rounds, and accumulates score_k * y_k in registers; one f32 write per output at the end.
//! No atomics, no cross-threadgroup traffic: the sum over slots is in slot order, so the bytes are deterministic.
//!
//! Rounding points are Sushi's host reference's (`ref.zig`); the reductions run in a different order, so parity is a
//! tolerance, not bit identity (the GPU test states it).
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const format = @import("format.zig");
const lane = @import("lane.zig");
const sources = @import("sources.zig");

pub const Geometry = struct {
    hidden: u32,
    inter: u32,
    topk: u32,
    rate: format.Rate,
    dec: format.Decode,
    /// timing probes only (never a served kernel): 1 skips the codebook hash, 2 skips the funnel too
    probe: u8 = 0,
    /// the SwiGLU's clamp (`root.moeClamped`: DeepSeek-V4's and GLM's `swiglu_limit`; 0 none): each row-slot's gate clamped
    /// from above at `limit` and its up to [-limit, limit] before the SiLU. `Kernels3` carries it in fin7's variant
    /// (`text_fin7c`, forms 7, 8 and 9); `Kernels` and `Kernels5` refuse a clamped geometry.
    limit: u32 = 0,

    pub fn admitted(g: Geometry) bool {
        return g.hidden % 128 == 0 and g.inter % 128 == 0 and g.hidden > 0 and g.inter > 0 and g.topk > 0 and g.topk <= 32;
    }
};

/// The Metal every decode kernel shares after the helpers: `turboexl3_row_of`, the tile row of each of a lane's eight
/// weights (quantize.py's tensor-core layout).
const metal_common = @embedFile("kernels/decode/common.metal");

/// Launch 1's text.
pub const text_gu = @embedFile("kernels/decode/gu.metal");

/// Launch 2's text.
pub const text_down = @embedFile("kernels/decode/down.metal");

/// The GEMV core every form-3 kernel inlines: lane L of a simdgroup accumulates its 8 tiles x 2 columns over input
/// tiles [tk0, tk1) of expert `wbase` (the stacked bank's expert base, in uint words), x in lane order.
const metal_gemv_core = @embedFile("kernels/decode/gemv_core.metal");

pub const header: [:0]const u8 = lane.metal_helpers ++ metal_common ++ metal_gemv_core;

const gu_inputs = [_][*:0]const u8{ "x", "slots", "gate_t", "gate_suh", "gate_svh", "up_t", "up_suh", "up_svh" };
const gu_outputs = [_][*:0]const u8{"h"};
const down_inputs = [_][*:0]const u8{ "h", "slots", "scores", "down_t", "down_suh", "down_svh" };
const down_outputs = [_][*:0]const u8{"y"};

/// Drops a kernel `ks.newKernel` made, by the consumer's kernel API (README.md's Wiring). A `kernel_launch` that declares
/// `mlx_fast_metal_kernel_free` hands each kernel to its caller and frees it there (mlx-stream's: the kernel's launch-cache
/// entry, then mlx-c's free); one that does not keeps every kernel itself (the plugin's: the server's kernel cache, one
/// kernel per name and text for the process), so nothing is freed. Every kernel turboexl3 drops goes through here.
pub fn freeKernel(k: mlx.mlx_fast_metal_kernel) void {
    if (@hasDecl(ks, "mlx_fast_metal_kernel_free")) _ = ks.mlx_fast_metal_kernel_free(k);
}

/// The two kernels of one geometry (compiled once per (rate, codebook, window, hidden, inter, topk)); configs per row count.
pub const Kernels = struct {
    geo: Geometry,
    gu: mlx.mlx_fast_metal_kernel,
    down: mlx.mlx_fast_metal_kernel,
    cfg_gu: [max_rows + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null),
    cfg_down: [max_rows + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null),

    pub const max_rows: usize = 16;

    pub fn init(geo: Geometry) !Kernels {
        if (!geo.admitted()) return error.BadExl3Shape;
        if (geo.limit != 0) return error.Exl3EngineNotImplemented; // the clamped SwiGLU is Kernels3's
        const gu = ks.newKernel("turboexl3_gu", &gu_inputs, &gu_outputs, text_gu, header, true) orelse return error.KernelCreateFailed;
        const down = ks.newKernel("turboexl3_down", &down_inputs, &down_outputs, text_down, header, true) orelse return error.KernelCreateFailed;
        return .{ .geo = geo, .gu = gu, .down = down };
    }

    pub fn deinit(self: *Kernels) void {
        for (&self.cfg_gu) |*c| if (c.*) |cc| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cc);
            c.* = null;
        };
        for (&self.cfg_down) |*c| if (c.*) |cc| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cc);
            c.* = null;
        };
        freeKernel(self.gu);
        freeKernel(self.down);
    }

    fn template(self: *const Kernels, buf: *[7]ks.TemplateArg) []const ks.TemplateArg {
        buf.* = .{
            .{ .name = "NHW", .value = .{ .int = @intCast(self.geo.rate.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(self.geo.dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(self.geo.dec.window.bits()) } },
            .{ .name = "HID", .value = .{ .int = @intCast(self.geo.hidden) } },
            .{ .name = "INTER", .value = .{ .int = @intCast(self.geo.inter) } },
            .{ .name = "TOPK", .value = .{ .int = @intCast(self.geo.topk) } },
            .{ .name = "PROBE", .value = .{ .int = self.geo.probe } },
        };
        return buf;
    }

    fn configs(self: *Kernels, rows: usize) !struct { gu: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config } {
        if (rows == 0 or rows > max_rows) return error.BadExl3Shape;
        if (self.cfg_gu[rows] == null) {
            var tbuf: [7]ks.TemplateArg = undefined;
            const t = self.template(&tbuf);
            const g = self.geo;
            const r: c_int = @intCast(rows);
            const h_shape = [_]c_int{ r * @as(c_int, @intCast(g.topk)), @intCast(g.inter) };
            self.cfg_gu[rows] = try ks.buildConfig(.{
                .grid = .{ (g.inter / 128) * 128, g.topk, @intCast(rows) },
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &.{&h_shape},
                .out_dtypes = &.{.float32},
                .template = t,
            });
            const y_shape = [_]c_int{ r, @intCast(g.hidden) };
            self.cfg_down[rows] = try ks.buildConfig(.{
                .grid = .{ (g.hidden / 128) * 128, @intCast(rows), 1 },
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &.{&y_shape},
                .out_dtypes = &.{.float32},
                .template = t,
            });
        }
        return .{ .gu = self.cfg_gu[rows].?, .down = self.cfg_down[rows].? };
    }

    pub const Bank = struct {
        gate_t: mlx.mlx_array,
        gate_suh: mlx.mlx_array,
        gate_svh: mlx.mlx_array,
        up_t: mlx.mlx_array,
        up_suh: mlx.mlx_array,
        up_svh: mlx.mlx_array,
        down_t: mlx.mlx_array,
        down_suh: mlx.mlx_array,
        down_svh: mlx.mlx_array,
    };

    /// Launch 1: h f32 [rows * topk, inter] from x f32 [rows, hidden] and slots u32 [rows * topk].
    pub fn gateUp(self: *Kernels, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, bank: Bank, rows: usize) !mlx.mlx_array {
        const c = try self.configs(rows);
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(self.gu, c.gu, &.{ x, slots, bank.gate_t, bank.gate_suh, bank.gate_svh, bank.up_t, bank.up_suh, bank.up_svh }, &outs, s);
        return outs[0];
    }

    /// Launch 2: y f32 [rows, hidden] from h, slots and scores f32 [rows * topk].
    pub fn downReduce(self: *Kernels, s: mlx.mlx_stream, h: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Bank, rows: usize) !mlx.mlx_array {
        const c = try self.configs(rows);
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(self.down, c.down, &.{ h, slots, scores, bank.down_t, bank.down_suh, bank.down_svh }, &outs, s);
        return outs[0];
    }

    /// The layer: both launches; the caller frees h and y.
    pub fn moe(self: *Kernels, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Bank, rows: usize) !struct { h: mlx.mlx_array, y: mlx.mlx_array } {
        const h = try self.gateUp(s, x, slots, bank, rows);
        errdefer _ = mlx.mlx_array_free(h);
        const y = try self.downReduce(s, h, slots, scores, bank, rows);
        return .{ .h = h, .y = y };
    }
};

test "turboexl3 decode texts: every template constant the texts read is one the launch passes" {
    const pin = comptime @import("test_support.zig").pins(@embedFile("kernels/test/pins/decode_texts.metal"));
    const t = std.testing;
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "TOPK", "PROBE" }) |name| {
        try t.expect(std.mem.indexOf(u8, header ++ text_gu, name) != null);
        try t.expect(std.mem.indexOf(u8, header ++ text_down, name) != null);
    }
    // every input the signatures name is read by its text, and the outputs are written
    for (gu_inputs) |n| try t.expect(std.mem.indexOf(u8, text_gu, std.mem.span(n)) != null);
    for (down_inputs) |n| try t.expect(std.mem.indexOf(u8, text_down, std.mem.span(n)) != null);
    try t.expect(std.mem.indexOf(u8, text_gu, pin[0]) != null);
    try t.expect(std.mem.indexOf(u8, text_down, pin[1]) != null);
}

test "turboexl3 decode geometry admission: 128-aligned widths, top-k within the reduce bank" {
    const t = std.testing;
    const ok: Geometry = .{ .hidden = 2560, .inter = 640, .topk = 10, .rate = format.Rate.fromK(5), .dec = .mul1 };
    try t.expect(ok.admitted());
    try t.expect(!(Geometry{ .hidden = 2560, .inter = 600, .topk = 10, .rate = format.Rate.fromK(5), .dec = .mul1 }).admitted());
    try t.expect(!(Geometry{ .hidden = 2560, .inter = 640, .topk = 33, .rate = format.Rate.fromK(5), .dec = .mul1 }).admitted());
}

/// Form 3 (split-K planes). Launch 2: one threadgroup per (128-wide intermediate block, row-slot, group of SPT K slices);
/// the K partition into KS slices is the geometry's alone (a row's summation order never depends on how many rows share
/// the call), the row count only sets how many threadgroups share a row-slot's slices (KS / SPT). The four simdgroups are
/// projection x half of a slice; each slice's partial sums go to its plane of zg / zu f32 [KS, RK, inter] (written once
/// each; no atomics).
pub const text_gu3 = gu3_1 ++ rk_line ++ gu3_body;
/// gu3's pieces after its rk line, cut where gu3sf differs: after its reduction buffer (its declarations) and at the
/// planes' stores
const gu3_1 = @embedFile("kernels/decode/gu3_1.metal");
const gu3_2 = @embedFile("kernels/decode/gu3_2.metal");
const gu3_3 = @embedFile("kernels/decode/gu3_3.metal");
const gu3_stores = @embedFile("kernels/decode/gu3_stores.metal");
const gu3_4 = @embedFile("kernels/decode/gu3_4.metal");
const gu3_body = gu3_2 ++ gu3_3 ++ gu3_stores ++ gu3_4;

/// Form 3, launch 3: per (128-wide intermediate block, row-slot): sum the planes, f16 round, H128, svh, f16 round,
/// SwiGLU, then the down's input rotation (f16 round, down suh, H128, f16 round) into hd in lane order.
pub const text_fin3 = @embedFile("kernels/decode/fin3.metal");

/// Form 3, launch 4: the down GEMV as split-K planes zd f32 [KS2, RK, hidden], the partition fixed per geometry and SPT
/// slices per threadgroup as in launch 2.
pub const text_down3 = @embedFile("kernels/decode/down3_1.metal") ++ rk_line ++ @embedFile("kernels/decode/down3_2.metal");

/// Form 3, launch 5: per (128-wide hidden block, row): for each slot in order sum the planes, f16 round, H128, svh,
/// f16 round, times the score; y f32 [rows, hidden].
pub const text_dfin3 = @embedFile("kernels/decode/dfin3.metal");

/// Form 3's prepare for gate and up in one launch: xg, xu half [RK, DIM] in lane order from x [rows, DIM] and the two suh.
pub const text_prep2 = @embedFile("kernels/decode/prep2_1.metal") ++ prep2_body;
const prep2_inputs = [_][*:0]const u8{ "x", "slots", "gate_suh", "up_suh" };
const prep2_outputs = [_][*:0]const u8{ "xg", "xu", "sg", "su" };

const gu3_inputs = [_][*:0]const u8{ "xgp", "xup", "slots", "gate_t", "up_t" };
const gu3_outputs = [_][*:0]const u8{ "zg", "zu" };
const fin3_inputs = [_][*:0]const u8{ "zg", "zu", "slots", "gate_svh", "up_svh", "down_suh", "xsg", "xsu" };
const fin3_outputs = [_][*:0]const u8{ "hd", "hs" };
const down3_inputs = [_][*:0]const u8{ "hdp", "slots", "down_t" };
const down3_outputs = [_][*:0]const u8{"zd"};
const dfin3_inputs = [_][*:0]const u8{ "zd", "slots", "scores", "down_svh", "hsx" };

/// Form 3: five launches (prepare gate and up, split-K gate/up planes, finish + SwiGLU + down prepare, split-K down
/// planes, finish + reduce); KS slices fill the GPU at one row.
pub const Kernels3 = struct {
    geo: Geometry,
    ks_gu: u32,
    ks_dn: u32,
    /// output tiles per GEMV threadgroup (8, 4, 2 or 1): 2 bt f32 accumulators per lane, 128 / (16 bt) x more threadgroups
    bt: u32,
    /// row-slots per GEMV lead threadgroup from two rows up (2: slots that chose the same expert share one decode)
    group: u32,
    /// how many threadgroups share a row-slot's K slices: `adaptive` (all KS at one row, half at 2-3, a quarter from 4: the
    /// row-slots fill the GPU by themselves), `full` (one threadgroup per slice at every row count); either way the K
    /// partition, so every row's bytes, is the geometry's alone
    kpar: KPar = .adaptive,
    /// `adaptive`'s divisors of the K slices for 1, 2, 3 and 4+ rows (1, 2, 4, 8 each): z = max(1, KS / kdiv) threadgroups
    /// along K per row-slot; set before the first call
    kdiv: [4]u32 = .{ 1, 2, 2, 4 },
    /// y's dtype (the x the host passes: bf16 served; f32 in the tests and the bench); set before the first call
    out_dtype: mlx.mlx_dtype = .float32,
    /// MUL1: the GEMVs sum x times the codeword's byte sum and the finishes apply the affine map once per output
    /// (false decodes every weight to its f16 value, the reference's rounding)
    defer_affine: bool = false,
    prep: mlx.mlx_fast_metal_kernel,
    gu: mlx.mlx_fast_metal_kernel,
    fin: mlx.mlx_fast_metal_kernel,
    down: mlx.mlx_fast_metal_kernel,
    dfin: mlx.mlx_fast_metal_kernel,
    /// form 7's launches 3 and 5 (fin3 and dfin3 with the H128s in registers, dfin's slots side by side): the same bytes
    fin7: mlx.mlx_fast_metal_kernel,
    dfin7: mlx.mlx_fast_metal_kernel,
    /// form 8's launch 1 (prep2 inside gu3, each threadgroup rotating its own K span): the same bytes, four launches
    gu8: mlx.mlx_fast_metal_kernel,
    /// form 9's launch shape, its own (forms 3/7/8 keep bt and kdiv): output tiles per GEMV threadgroup at one row (bt1) and
    /// from two rows (bt9), and the K-slice divisors (kdiv9); 1, 2, 4, 8 each. The K partition, so the bytes, do not depend
    /// on any of them. Set before the first call.
    bt1: u32 = 2,
    bt9: u32 = 2,
    kdiv9: [4]u32 = .{ 1, 1, 1, 1 },
    /// form 9's down-GEMV K-slice divisors when they differ from kdiv9 (null: kdiv9)
    kdiv9d: ?[4]u32 = null,
    /// form 9 runs the prepare inside the gate|up launch (gu8) up to this many rows (where `gu8Admitted`), prep2 + gu3 above
    gu8_rows: usize = 1,
    /// the shared expert's K slices per GEMV threadgroup, gate|up (sspt, a divisor of its 20 slices) and down (ssptd, of its 5);
    /// launch shape only (every slice computed as a threadgroup of its own computes it). Set before the first call.
    sspt: u32 = 1,
    ssptd: u32 = 1,
    cfg: [Kernels.max_rows + 1]?Cfg = @splat(null),
    /// form 9's configs
    cfg9: [Kernels.max_rows + 1]?Cfg = @splat(null),
    /// the shared expert inside form 9's launches (`moeShared`): the heads on gu3 / gu8 / fin7 / down3 and dfin7s
    gu3s: mlx.mlx_fast_metal_kernel,
    gu8s: mlx.mlx_fast_metal_kernel,
    fin7s: mlx.mlx_fast_metal_kernel,
    down3s: mlx.mlx_fast_metal_kernel,
    dfin7s: mlx.mlx_fast_metal_kernel,
    scfg: [shared_rows_max + 1]?SCfg = @splat(null),
    scfg_key: [shared_rows_max + 1]SKey = undefined,
    /// at the fused shared expert's rows past gu8_rows (the verify widths), gu3s and fin7s run as one launch (gu3sf: the
    /// threadgroup that completes a routed (row-slot, 128-block) runs fin7's statements for it, the one that completes a
    /// shared 128-block runs the shared fin head's for every row), after a counter zeroing that depends on nothing:
    /// fin7s's bytes, one dependent launch fewer (`Config.fused_fin_fold`). The kernels are built on first use.
    fused_fin_fold: bool = true,
    gu3sf: ?mlx.mlx_fast_metal_kernel = null,
    vzero: ?mlx.mlx_fast_metal_kernel = null,
    /// the shared expert in its own launches beside form 9's (`moeSharedSide`): the three heads standalone (RKC 0, so every
    /// grid row is the shared expert's) and the routed kernels untouched, each with its own threadgroup memory and registers
    sgu: mlx.mlx_fast_metal_kernel,
    sfin: mlx.mlx_fast_metal_kernel,
    sdown: mlx.mlx_fast_metal_kernel,
    sscfg: [shared_rows_max + 1]?SSCfg = @splat(null),
    sscfg_key: [shared_rows_max + 1]SKey = undefined,
    /// the v2 shared heads (`moeShared2` / `moeSharedSide2`, `sheads` 2; 1 runs `moeShared` / `moeSharedSide`). Launch shape
    /// only, the dense planes' bytes under every option: `sots` / `sotsd` the gate|up / down output tiles per GEMV simdgroup
    /// at 1, 2, 3 and 4+ rows (`sharedOts` clamps them); `sprep` the gate|up input prepared once in the routed prepare launch
    /// where form 9 runs one (else rotated in every threadgroup); `slm` lane-major planes (a float4 per plane and lane in the
    /// finishes); `spre` (per row count) a GEMV simdgroup's weight prefetch distance in input tiles, in whole parts of the
    /// slice, at most 3 parts and 16 tiles in flight; `sfirst` the shared threadgroups ahead of the routed ones in the gate|up
    /// and down grids. Set before the first call.
    sheads: u8 = 1,
    sots: [4]u32 = .{ 2, 2, 1, 1 },
    sotsd: [4]u32 = .{ 2, 2, 1, 1 },
    sprep: bool = true,
    slm: bool = true,
    spre: [4]u32 = .{ 2, 2, 2, 2 },
    sfirst: bool = false,
    /// the v2 heads on form 9's launches: gu8t (rotating head + gu8), gu3t (rotating head + gu3), gu3p (prepared-input head +
    /// gu3, after prep2t = prep2 + the shared prepare), fin7t, down3t, dfin7t; beside them: sgu2, sgu2p, sprep2, sfin2, sdown2
    gu8t: mlx.mlx_fast_metal_kernel,
    gu3t: mlx.mlx_fast_metal_kernel,
    gu3p: mlx.mlx_fast_metal_kernel,
    prep2t: mlx.mlx_fast_metal_kernel,
    fin7t: mlx.mlx_fast_metal_kernel,
    down3t: mlx.mlx_fast_metal_kernel,
    dfin7t: mlx.mlx_fast_metal_kernel,
    sgu2: mlx.mlx_fast_metal_kernel,
    sgu2p: mlx.mlx_fast_metal_kernel,
    sprep2: mlx.mlx_fast_metal_kernel,
    sfin2: mlx.mlx_fast_metal_kernel,
    sdown2: mlx.mlx_fast_metal_kernel,
    tcfg: [shared_rows_max + 1]?TCfg = @splat(null),
    tcfg_key: [shared_rows_max + 1]TKey = undefined,
    tscfg: [shared_rows_max + 1]?TSCfg = @splat(null),
    tscfg_key: [shared_rows_max + 1]TKey = undefined,
    /// The host's softmax router folded into form 9's routed prepare launch (`routeLaunch`, `moeSharedRouted`): the
    /// plain kernel and the one that also computes the shared gate's logit, built on first use, and one config per row count
    /// keyed on the launch's experts, scores dtype and gate width (`RouteKey`).
    route_prep: ?mlx.mlx_fast_metal_kernel = null,
    route_prepg: ?mlx.mlx_fast_metal_kernel = null,
    route_cfg: [Kernels.max_rows + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null),
    route_key: [Kernels.max_rows + 1]RouteKey = undefined,

    /// The shared expert `moeShared` folds into the routed launches: three dense EXL3 linears (gate, up: hidden -> inter;
    /// down: inter -> hidden; trellis U16 [in/16, out/16, n], suh [in], svh [out]) as dense.zig serves them (`n_*` their
    /// rates, `ks_*` `Dense.ksFor`'s K splits, `defer_affine` its MUL1 deferral), the host's SwiGLU table (OUT_T [65536]:
    /// MLX's sigmoid of every 16-bit pattern) and the shared gate's logit per row (OUT_T [rows]).
    pub const SharedBank = struct {
        gate_t: mlx.mlx_array,
        gate_suh: mlx.mlx_array,
        gate_svh: mlx.mlx_array,
        up_t: mlx.mlx_array,
        up_suh: mlx.mlx_array,
        up_svh: mlx.mlx_array,
        down_t: mlx.mlx_array,
        down_suh: mlx.mlx_array,
        down_svh: mlx.mlx_array,
        n_gu: u32,
        n_dn: u32,
        ks_gu: u32,
        ks_dn: u32,
        defer_affine: bool,
        sigtab: mlx.mlx_array,
        logit: mlx.mlx_array,
    };
    /// verify widths the shared heads carry (their GEMVs keep every row's accumulators in one threadgroup, as dgemv's RB)
    pub const shared_rows_max: usize = 4;
    const SKey = struct { n_gu: u32, n_dn: u32, ks_gu: u32, ks_dn: u32, def: bool, sp_gu: u32, sp_dn: u32 };
    /// `want` slices per threadgroup when it divides the shared K split, else 1
    fn sharedSpt(ks_: u32, want: u32) u32 {
        return if (want >= 1 and ks_ % want == 0) want else 1;
    }
    /// `vf`: the configs of gu3sf and of its counter zeroing (rows past gu8_rows)
    const SCfg = struct { gu: mlx.mlx_fast_metal_kernel_config, fin: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config, dfin: mlx.mlx_fast_metal_kernel_config, gu8: bool, vf: ?VFold = null };
    const VFold = struct { gu: mlx.mlx_fast_metal_kernel_config, zero: mlx.mlx_fast_metal_kernel_config };
    fn freeSCfg(c: SCfg) void {
        inline for (.{ c.gu, c.fin, c.down, c.dfin }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
        if (c.vf) |vf| inline for (.{ vf.gu, vf.zero }) |k| {
            _ = mlx.mlx_fast_metal_kernel_config_free(k);
        };
    }
    const SSCfg = struct { gu: mlx.mlx_fast_metal_kernel_config, fin: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config };
    /// the v2 configs' key: the shared bank's shape and every v2 knob the launches read (`sots` / `sotsd` as resolved per row)
    const TKey = struct { n_gu: u32, n_dn: u32, ks_gu: u32, ks_dn: u32, def: bool, ots: u32, otsd: u32, prep: bool, lm: bool, pre: u32, first: bool, gu8_rows: usize };
    /// inside: the optional prepare (prep2t), gate|up (gu8t / gu3t / gu3p), fin7t, down3t, dfin7t; `gu8`: launch 1 is gu8t
    const TCfg = struct { prep: ?mlx.mlx_fast_metal_kernel_config, gu: mlx.mlx_fast_metal_kernel_config, fin: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config, dfin: mlx.mlx_fast_metal_kernel_config, gu8: bool };
    /// beside: the optional standalone prepare (sprep2, then sgu2p; else sgu2), sfin2, sdown2
    const TSCfg = struct { prep: ?mlx.mlx_fast_metal_kernel_config, gu: mlx.mlx_fast_metal_kernel_config, fin: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config };
    fn freeTCfg(c: TCfg) void {
        if (c.prep) |p| _ = mlx.mlx_fast_metal_kernel_config_free(p);
        inline for (.{ c.gu, c.fin, c.down, c.dfin }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
    }
    fn freeTSCfg(c: TSCfg) void {
        if (c.prep) |p| _ = mlx.mlx_fast_metal_kernel_config_free(p);
        inline for (.{ c.gu, c.fin, c.down }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
    }
    /// `want` output tiles per v2 GEMV simdgroup at `rows` rows: a value whose simdgroup tiles would leave one projection
    /// (`per_proj` tiles each) or whose groups of four would not tile the `nproj` projections runs 1 (always admitted: the
    /// widths are 128-aligned); an admitted value whose accumulators would pass 64 floats per lane (the part's sums and the
    /// slice's, rows x tiles x 2 columns each: rows x tiles > 16) runs the largest admitted value within that budget, so no
    /// setting hands the back-end compiler a huge live state
    pub fn sharedOts(per_proj: u32, nproj: u32, want: u32, rows: u32) u32 {
        const tiling = struct {
            fn ok(pp: u32, np: u32, c: u32) bool {
                return c >= 1 and pp % c == 0 and (np * pp) % (4 * c) == 0;
            }
        };
        if (!tiling.ok(per_proj, nproj, want)) return 1;
        var c: u32 = want;
        while (c > 1 and (rows * c > 16 or !tiling.ok(per_proj, nproj, c))) c -= 1;
        return c;
    }
    /// the v2 heads' resolved output tiles per simdgroup at `rows` rows: gate|up, down
    pub fn sharedTiles(self: *const Kernels3, rows: usize) [2]u32 {
        const ri: usize = @min(rows, 4) - 1;
        const r: u32 = @intCast(rows);
        return .{ sharedOts(self.geo.inter / 16, 2, self.sots[ri], r), sharedOts(self.geo.hidden / 16, 1, self.sotsd[ri], r) };
    }

    const Cfg = struct { prep_h: mlx.mlx_fast_metal_kernel_config, gu: mlx.mlx_fast_metal_kernel_config, fin: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config, dfin: mlx.mlx_fast_metal_kernel_config, fin7: mlx.mlx_fast_metal_kernel_config, dfin7: mlx.mlx_fast_metal_kernel_config, gu8: mlx.mlx_fast_metal_kernel_config };
    const prep_in = [_][*:0]const u8{ "x", "slots", "suh" };
    const prep_out = [_][*:0]const u8{"xp"};

    pub const KPar = enum { adaptive, full };

    /// K slices, fixed per geometry: gate/up over hidden/16 tiles in KS x 2 parts, down over inter/16 tiles in KS2 x 4
    /// parts; each part at least 4 tiles.
    pub fn init(geo: Geometry, bt: u32, ks_max: u32) !Kernels3 {
        return initWith(geo, bt, ks_max, 2);
    }

    /// `group` 2 pairs the row-slots that chose the same expert under one threadgroup from two rows up (the weights
    /// read and decoded once, fed to both members' accumulators; Sushi's grouped kernels do the same); 1 keeps every
    /// row-slot its own threadgroup.
    pub fn initWith(geo: Geometry, bt: u32, ks_max: u32, group: u32) !Kernels3 {
        if (!geo.admitted() or (bt != 8 and bt != 4 and bt != 2 and bt != 1) or (ks_max != 1 and ks_max != 2 and ks_max != 4 and ks_max != 8) or (group != 1 and group != 2)) return error.BadExl3Shape;
        var ks_gu: u32 = ks_max;
        while (ks_gu > 1 and (geo.hidden / 16) % (ks_gu * 2) != 0) ks_gu /= 2;
        var ks_dn: u32 = ks_max;
        while (ks_dn > 1 and ((geo.inter / 16) % (ks_dn * 4) != 0 or (geo.inter / 16) / (ks_dn * 4) < 2)) ks_dn /= 2;
        return .{
            .geo = geo,
            .ks_gu = ks_gu,
            .ks_dn = ks_dn,
            .bt = bt,
            .group = group,
            .defer_affine = geo.dec.codebook == .mul1,
            .prep = ks.newKernel("turboexl3_prep2", &prep2_inputs, &prep2_outputs, text_prep2, header, true) orelse return error.KernelCreateFailed,
            .gu = ks.newKernel("turboexl3_gu3", &gu3_inputs, &gu3_outputs, text_gu3, header, true) orelse return error.KernelCreateFailed,
            .fin = ks.newKernel("turboexl3_fin3", &fin3_inputs, &fin3_outputs, text_fin3, header, true) orelse return error.KernelCreateFailed,
            .down = ks.newKernel("turboexl3_down3", &down3_inputs, &down3_outputs, text_down3, header, true) orelse return error.KernelCreateFailed,
            .dfin = ks.newKernel("turboexl3_dfin3", &dfin3_inputs, &down_outputs, text_dfin3, header, true) orelse return error.KernelCreateFailed,
            // a clamped geometry's fin7 is the clamped variant, so every launch of fin7 (forms 7, 8 and 9) clamps
            .fin7 = (if (geo.limit == 0)
                ks.newKernel("turboexl3_fin7", &fin3_inputs, &fin3_outputs, text_fin7, header5, true)
            else
                ks.newKernel("turboexl3_fin7c", &fin3_inputs, &fin3_outputs, text_fin7c, header5, true)) orelse return error.KernelCreateFailed,
            .dfin7 = ks.newKernel("turboexl3_dfin7", &dfin3_inputs, &down_outputs, text_dfin7, header5, true) orelse return error.KernelCreateFailed,
            .gu8 = ks.newKernel("turboexl3_gu8", &gu8_inputs, &gu8_outputs, text_gu8, header5, true) orelse return error.KernelCreateFailed,
            .gu3s = ks.newKernel("turboexl3_gu3s", &gu3s_inputs, &gu3s_outputs, text_gu3s, header_s, true) orelse return error.KernelCreateFailed,
            .gu8s = ks.newKernel("turboexl3_gu8s", &gu8s_inputs, &gu8s_outputs, text_gu8s, header_s, true) orelse return error.KernelCreateFailed,
            .fin7s = ks.newKernel("turboexl3_fin7s", &fin7s_inputs, &fin7s_outputs, text_fin7s, header_s, true) orelse return error.KernelCreateFailed,
            .down3s = ks.newKernel("turboexl3_down3s", &down3s_inputs, &down3s_outputs, text_down3s, header_s, true) orelse return error.KernelCreateFailed,
            .dfin7s = ks.newKernel("turboexl3_dfin7s", &dfin7s_inputs, &down_outputs, text_dfin7s, header_s, true) orelse return error.KernelCreateFailed,
            .sgu = ks.newKernel("turboexl3_sgu", &shared_gu_inputs, &shared_gu_outputs, shared_gu_head, header_s, true) orelse return error.KernelCreateFailed,
            .sfin = ks.newKernel("turboexl3_sfin", &sfin_inputs, &sfin_outputs, shared_fin_head, header_s, true) orelse return error.KernelCreateFailed,
            .sdown = ks.newKernel("turboexl3_sdown", &sdown_inputs, &sdown_outputs, shared_down_head, header_s, true) orelse return error.KernelCreateFailed,
            .gu8t = ks.newKernel("turboexl3_gu8t", &gu8s_inputs, &gu8s_outputs, text_gu8t, header_t, true) orelse return error.KernelCreateFailed,
            .gu3t = ks.newKernel("turboexl3_gu3t", &gu3s_inputs, &gu3s_outputs, text_gu3t, header_t, true) orelse return error.KernelCreateFailed,
            .gu3p = ks.newKernel("turboexl3_gu3p", &gu3p_inputs, &gu3p_outputs, text_gu3p, header_t, true) orelse return error.KernelCreateFailed,
            .prep2t = ks.newKernel("turboexl3_prep2t", &prep2t_inputs, &prep2t_outputs, text_prep2t, header_t, true) orelse return error.KernelCreateFailed,
            .fin7t = ks.newKernel("turboexl3_fin7t", &fin7s_inputs, &fin7s_outputs, text_fin7t, header_t, true) orelse return error.KernelCreateFailed,
            .down3t = ks.newKernel("turboexl3_down3t", &down3s_inputs, &down3s_outputs, text_down3t, header_t, true) orelse return error.KernelCreateFailed,
            .dfin7t = ks.newKernel("turboexl3_dfin7t", &dfin7s_inputs, &down_outputs, text_dfin7t, header_t, true) orelse return error.KernelCreateFailed,
            .sgu2 = ks.newKernel("turboexl3_sgu2", &shared_gu_inputs, &shared_gu_outputs, shared_gu2_head, header_t, true) orelse return error.KernelCreateFailed,
            .sgu2p = ks.newKernel("turboexl3_sgu2p", &shared_gu2p_inputs, &shared_gu2p_outputs, shared_gu2p_head, header_t, true) orelse return error.KernelCreateFailed,
            .sprep2 = ks.newKernel("turboexl3_sprep2", &shared_prep_inputs, &shared_prep_outputs, shared_prep_head, header_t, true) orelse return error.KernelCreateFailed,
            .sfin2 = ks.newKernel("turboexl3_sfin2", &sfin_inputs, &sfin_outputs, shared_fin2_head, header_t, true) orelse return error.KernelCreateFailed,
            .sdown2 = ks.newKernel("turboexl3_sdown2", &sdown_inputs, &sdown_outputs, shared_down2_head, header_t, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *Kernels3) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.gu, cc.fin, cc.down, cc.dfin, cc.fin7, cc.dfin7, cc.gu8 }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        for (&self.cfg9) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.gu, cc.fin, cc.down, cc.dfin, cc.fin7, cc.dfin7, cc.gu8 }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        for (&self.scfg) |*c| if (c.*) |cc| {
            freeSCfg(cc);
            c.* = null;
        };
        inline for (.{ self.gu3sf, self.vzero }) |k| if (k) |kk| {
            freeKernel(kk);
        };
        for (&self.sscfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.gu, cc.fin, cc.down }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        for (&self.tcfg) |*c| if (c.*) |cc| {
            freeTCfg(cc);
            c.* = null;
        };
        for (&self.tscfg) |*c| if (c.*) |cc| {
            freeTSCfg(cc);
            c.* = null;
        };
        for (&self.route_cfg) |*c| if (c.*) |cc| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cc);
            c.* = null;
        };
        inline for (.{ self.route_prep, self.route_prepg }) |k| if (k) |kk| {
            freeKernel(kk);
        };
        inline for (.{ self.prep, self.gu, self.fin, self.down, self.dfin, self.fin7, self.dfin7, self.gu8, self.gu3s, self.gu8s, self.fin7s, self.down3s, self.dfin7s, self.sgu, self.sfin, self.sdown }) |k| freeKernel(k);
        inline for (.{ self.gu8t, self.gu3t, self.gu3p, self.prep2t, self.fin7t, self.down3t, self.dfin7t, self.sgu2, self.sgu2p, self.sprep2, self.sfin2, self.sdown2 }) |k| freeKernel(k);
    }

    /// Form 8's gate|up launch rotates whole 128-blocks: a threadgroup's span (its SPT slices of hidden / KS values) must be
    /// 128-aligned; otherwise form 8 runs form 7's launches.
    pub fn gu8Admitted(self: *const Kernels3, rows: usize, kdiv: [4]u32) bool {
        const spt = self.ks_gu / self.kThreadgroups(self.ks_gu, rows, kdiv);
        return (self.geo.hidden / self.ks_gu * spt) % 128 == 0 and self.geo.hidden % self.ks_gu == 0;
    }

    /// Threadgroups along K per row-slot for a row count (`adaptive`, `kdiv`'s defaults): all ks slices apart at one row, two
    /// slices per threadgroup at 2-3 rows, four from 4 rows (the row-slots fill the GPU by themselves). The slices themselves,
    /// and so the summation order, do not change: a threadgroup loops over its ks / z slices and writes each slice's plane.
    fn kThreadgroups(self: *const Kernels3, ks_fixed: u32, rows: usize, kdiv: [4]u32) u32 {
        if (self.kpar == .full) return ks_fixed;
        const div: u32 = kdiv[@min(rows, 4) - 1];
        return @max(1, ks_fixed / @max(1, div));
    }

    fn cfgs(self: *Kernels3, rows: usize) !Cfg {
        if (rows == 0 or rows > Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const c = try self.buildCfgs(rows, self.bt, self.kdiv);
        self.cfg[rows] = c;
        return c;
    }

    /// Form 9's configs: bt1 tiles per GEMV threadgroup at one row, bt9 from two, kdiv9's K threadgroups.
    fn cfgs9(self: *Kernels3, rows: usize) !Cfg {
        if (rows == 0 or rows > Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg9[rows]) |c| return c;
        const bt = if (rows == 1) self.bt1 else self.bt9;
        if (bt != 1 and bt != 2 and bt != 4 and bt != 8) return error.BadExl3Shape;
        const c = try self.buildCfgsSplit(rows, bt, self.kdiv9, self.kdiv9d orelse self.kdiv9);
        self.cfg9[rows] = c;
        return c;
    }

    fn buildCfgs(self: *Kernels3, rows: usize, bt: u32, kdiv: [4]u32) !Cfg {
        return self.buildCfgsSplit(rows, bt, kdiv, kdiv);
    }

    fn buildCfgsSplit(self: *Kernels3, rows: usize, bt: u32, kdiv: [4]u32, kdiv_dn: [4]u32) !Cfg {
        const g = self.geo;
        const z_gu = self.kThreadgroups(self.ks_gu, rows, kdiv);
        const z_dn = self.kThreadgroups(self.ks_dn, rows, kdiv_dn);
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nhw = ks.TemplateArg{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } };
        const cb = ks.TemplateArg{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = ks.TemplateArg{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = ks.TemplateArg{ .name = "HID", .value = .{ .int = H } };
        const inter_t = ks.TemplateArg{ .name = "INTER", .value = .{ .int = I } };
        const rkt = ks.TemplateArg{ .name = "RKC", .value = .{ .int = rkc } };
        const topk = ks.TemplateArg{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } };
        const ks_gu_t = ks.TemplateArg{ .name = "KS", .value = .{ .int = @intCast(self.ks_gu) } };
        const ks_dn_t = ks.TemplateArg{ .name = "KS", .value = .{ .int = @intCast(self.ks_dn) } };
        const spt_gu_t = ks.TemplateArg{ .name = "SPT", .value = .{ .int = @intCast(self.ks_gu / z_gu) } };
        const spt_dn_t = ks.TemplateArg{ .name = "SPT", .value = .{ .int = @intCast(self.ks_dn / z_dn) } };
        const bt_t = ks.TemplateArg{ .name = "BT", .value = .{ .int = @intCast(bt) } };
        const grp_t = ks.TemplateArg{ .name = "GROUP", .value = .{ .int = @intCast(if (rows >= 2) self.group else 1) } };
        const out_t = ks.TemplateArg{ .name = "OUT_T", .value = .{ .dtype = self.out_dtype } };
        const def_t = ks.TemplateArg{ .name = "DEFER", .value = .{ .int = if (self.defer_affine) 1 else 0 } };
        const nbh_t = ks.TemplateArg{ .name = "NBH", .value = .{ .int = @intCast(g.hidden / 128) } };
        const nbi_t = ks.TemplateArg{ .name = "NBI", .value = .{ .int = @intCast(g.inter / 128) } };
        // fin7's constants, the clamp last where the geometry clamps (an unclamped fin7 takes exactly its five)
        const fin7_t = [_]ks.TemplateArg{ inter_t, rkt, ks_gu_t, def_t, nbh_t, .{ .name = "LIMIT", .value = .{ .int = @intCast(g.limit) } } };
        const fin7_n: usize = if (g.limit != 0) fin7_t.len else fin7_t.len - 1;
        const bw: u32 = 16 * bt;
        return .{
            .prep_h = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{ &[_]c_int{ rkc, H }, &[_]c_int{ rkc, H }, &[_]c_int{ rkc, @intCast(g.hidden / 128) }, &[_]c_int{ rkc, @intCast(g.hidden / 128) } }, .out_dtypes = &.{ .float16, .float16, .float32, .float32 }, .template = &.{ .{ .name = "DIM", .value = .{ .int = H } }, topk } }),
            .gu = try ks.buildConfig(.{ .grid = .{ (g.inter / bw) * 128, rk, z_gu }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(self.ks_gu), rkc, I } }, .out_dtypes = &.{ .float32, .float32 }, .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t } }),
            .fin = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 128, rk, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, @intCast(g.inter / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = &.{ inter_t, rkt, ks_gu_t, def_t, nbh_t } }),
            .down = try ks.buildConfig(.{ .grid = .{ (g.hidden / bw) * 128, rk, z_dn }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(self.ks_dn), rkc, H }}, .out_dtypes = &.{.float32}, .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_dn_t, spt_dn_t, bt_t, grp_t, def_t } }),
            .dfin = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 128, @intCast(rows), 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{self.out_dtype}, .template = &.{ hid, topk, rkt, ks_dn_t, out_t, def_t, nbi_t } }),
            .fin7 = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, @intCast(g.inter / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = fin7_t[0..fin7_n] }),
            .dfin7 = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32 * g.topk, @intCast(rows), 1 }, .threadgroup = .{ 32 * g.topk, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{self.out_dtype}, .template = &.{ hid, topk, rkt, ks_dn_t, out_t, def_t, nbi_t } }),
            .gu8 = try ks.buildConfig(.{ .grid = .{ (g.inter / bw) * 128, rk, z_gu }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ rkc, @intCast(g.hidden / 128) }, &[_]c_int{ rkc, @intCast(g.hidden / 128) } }, .out_dtypes = &.{ .float32, .float32, .float32, .float32 }, .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t, topk } }),
        };
    }

    fn one(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, c, ins, &outs, s);
        return outs[0];
    }

    /// The five launches (one prepare for both projections); the caller frees y.
    pub fn moe(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize) !mlx.mlx_array {
        return self.moeForm(s, x, slots, scores, bank, rows, 3);
    }

    /// `form` 3: the five launches above; 7: the same GEMVs with fin7 / dfin7 as launches 3 and 5; 8: form 7 with the
    /// prepare inside the gate|up launch (gu8, four launches) where `gu8Admitted`; 9: form 8 at one row, form 7 from two
    /// rows (gu8 re-rotates its whole span in every output-tile threadgroup, a cost that grows once a threadgroup carries
    /// several slices), in its own launch shape (bt1, bt9, kdiv9). Every form, form 3's bytes. A clamped geometry runs forms
    /// 7, 8 and 9 (fin7's clamped variant) and refuses form 3 (error.Exl3EngineNotImplemented: fin3 has no clamp).
    pub fn moeForm(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, form: u8) !mlx.mlx_array {
        if (self.geo.limit != 0 and form < 7) return error.Exl3EngineNotImplemented;
        const one_row9 = form == 9 and rows <= self.gu8_rows;
        const c = if (form == 9) try self.cfgs9(rows) else try self.cfgs(rows);
        const f7 = form >= 7;
        // dfin7 runs one simdgroup per routed slot and writes each block's 128 outputs from its first 128 threads, so it needs
        // a top-k of 4 or more; below that form 3's dfin (the same bytes) finishes the down projection.
        const f7d = f7 and self.geo.topk >= 4;
        // planes[0..2] = zg, zu (the split-K gate|up planes), planes[2..4] = the inputs' per-block sums
        var planes: [4]mlx.mlx_array = undefined;
        if ((form == 8 and self.gu8Admitted(rows, self.kdiv)) or (one_row9 and self.gu8Admitted(rows, self.kdiv9))) {
            try ks.applyConfig(self.gu8, c.gu8, &.{ x, slots, bank.gate_suh, bank.up_suh, bank.gate_t, bank.up_t }, &planes, s);
        } else {
            var xp: [4]mlx.mlx_array = undefined; // xg, xu, and their per-block input sums
            try ks.applyConfig(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh, bank.up_suh }, &xp, s);
            defer _ = mlx.mlx_array_free(xp[0]);
            defer _ = mlx.mlx_array_free(xp[1]);
            errdefer _ = mlx.mlx_array_free(xp[2]);
            errdefer _ = mlx.mlx_array_free(xp[3]);
            try ks.applyConfig(self.gu, c.gu, &.{ xp[0], xp[1], slots, bank.gate_t, bank.up_t }, planes[0..2], s);
            planes[2] = xp[2];
            planes[3] = xp[3];
        }
        defer for (planes) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var hd: [2]mlx.mlx_array = undefined; // the down's prepared input and its per-block sums
        try ks.applyConfig(if (f7) self.fin7 else self.fin, if (f7) c.fin7 else c.fin, &.{ planes[0], planes[1], slots, bank.gate_svh, bank.up_svh, bank.down_suh, planes[2], planes[3] }, &hd, s);
        defer for (hd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const zd = try one(self.down, c.down, &.{ hd[0], slots, bank.down_t }, s);
        defer _ = mlx.mlx_array_free(zd);
        return one(if (f7d) self.dfin7 else self.dfin, if (f7d) c.dfin7 else c.dfin, &.{ zd, slots, scores, bank.down_svh, hd[1] }, s);
    }

    /// Whether `moeShared` carries this shared expert at `rows`: up to `shared_rows_max` rows, the routed width as its
    /// intermediate, gate|up K slices of exactly one 128-block of hidden (the heads rotate their block in-kernel: dense prep_s's
    /// span), a down split of whole 4-part slices, and the rows' bf16 or f16 output (the SwiGLU table's dtype). Never on a
    /// clamped geometry: the shared heads' table SwiGLU has no clamp (and the routed side launches run fin7's variant).
    pub fn sharedAdmitted(self: *const Kernels3, rows: usize, sb: SharedBank) bool {
        const g = self.geo;
        if (g.limit != 0) return false;
        if (rows == 0 or rows > shared_rows_max) return false;
        if (g.topk < 4) return false; // dfin7s and dfin7t write each block from their first 128 threads, as dfin7 does
        if (self.out_dtype != .bfloat16 and self.out_dtype != .float16) return false;
        const gs = mlx.getShape(sb.gate_t);
        const us = mlx.getShape(sb.up_t);
        const ds = mlx.getShape(sb.down_t);
        if (gs.len != 3 or us.len != 3 or ds.len != 3) return false;
        const ht: c_int = @intCast(g.hidden / 16);
        const it: c_int = @intCast(g.inter / 16);
        if (gs[0] != ht or gs[1] != it or us[0] != ht or us[1] != it or ds[0] != it or ds[1] != ht) return false;
        if (gs[2] != @as(c_int, @intCast(sb.n_gu)) or us[2] != gs[2] or ds[2] != @as(c_int, @intCast(sb.n_dn))) return false;
        if (sb.ks_gu == 0 or sb.ks_gu * 128 != g.hidden) return false;
        if (sb.ks_dn == 0 or (g.inter / 16) % (sb.ks_dn * 4) != 0) return false;
        return format.Rate.fromN(sb.n_gu) != null and format.Rate.fromN(sb.n_dn) != null;
    }

    /// The four launch configs (form 9's shape: bt1 / bt9, kdiv9) with the shared heads' extra grid rows and constants.
    fn sharedCfgs(self: *Kernels3, rows: usize, sb: SharedBank) !SCfg {
        const key: SKey = .{ .n_gu = sb.n_gu, .n_dn = sb.n_dn, .ks_gu = sb.ks_gu, .ks_dn = sb.ks_dn, .def = sb.defer_affine, .sp_gu = sharedSpt(sb.ks_gu, self.sspt), .sp_dn = sharedSpt(sb.ks_dn, self.ssptd) };
        if (self.scfg[rows]) |c| {
            if (std.meta.eql(self.scfg_key[rows], key)) return c;
            freeSCfg(c);
            self.scfg[rows] = null;
        }
        const g = self.geo;
        const bt: u32 = if (rows == 1) self.bt1 else self.bt9;
        const z_gu = self.kThreadgroups(self.ks_gu, rows, self.kdiv9);
        const z_dn = self.kThreadgroups(self.ks_dn, rows, self.kdiv9d orelse self.kdiv9);
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const R: c_int = @intCast(rows);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nbh: c_int = @intCast(g.hidden / 128);
        const nbi: c_int = @intCast(g.inter / 128);
        const ys_gu: u32 = (2 * (sb.ks_gu / key.sp_gu) + z_gu - 1) / z_gu;
        const ys_dn: u32 = (sb.ks_dn / key.sp_dn + z_dn - 1) / z_dn;
        const T = ks.TemplateArg;
        const nhw = T{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } };
        const cb = T{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = T{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = T{ .name = "HID", .value = .{ .int = H } };
        const inter_t = T{ .name = "INTER", .value = .{ .int = I } };
        const rkt = T{ .name = "RKC", .value = .{ .int = rkc } };
        const topk = T{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } };
        const ks_gu_t = T{ .name = "KS", .value = .{ .int = @intCast(self.ks_gu) } };
        const ks_dn_t = T{ .name = "KS", .value = .{ .int = @intCast(self.ks_dn) } };
        const spt_gu_t = T{ .name = "SPT", .value = .{ .int = @intCast(self.ks_gu / z_gu) } };
        const spt_dn_t = T{ .name = "SPT", .value = .{ .int = @intCast(self.ks_dn / z_dn) } };
        const bt_t = T{ .name = "BT", .value = .{ .int = @intCast(bt) } };
        const grp_t = T{ .name = "GROUP", .value = .{ .int = @intCast(if (rows >= 2) self.group else 1) } };
        const out_t = T{ .name = "OUT_T", .value = .{ .dtype = self.out_dtype } };
        const def_t = T{ .name = "DEFER", .value = .{ .int = if (self.defer_affine) 1 else 0 } };
        const nbh_t = T{ .name = "NBH", .value = .{ .int = nbh } };
        const nbi_t = T{ .name = "NBI", .value = .{ .int = nbi } };
        const sng = T{ .name = "SNG", .value = .{ .int = @intCast(sb.n_gu) } };
        const snd = T{ .name = "SND", .value = .{ .int = @intCast(sb.n_dn) } };
        const sksg = T{ .name = "SKSG", .value = .{ .int = @intCast(sb.ks_gu) } };
        const sksd = T{ .name = "SKSD", .value = .{ .int = @intCast(sb.ks_dn) } };
        const srs = T{ .name = "SRS", .value = .{ .int = R } };
        const sdef = T{ .name = "SDEF", .value = .{ .int = if (sb.defer_affine) 1 else 0 } };
        const sz_gu = T{ .name = "SZ", .value = .{ .int = @intCast(z_gu) } };
        const sz_dn = T{ .name = "SZ", .value = .{ .int = @intCast(z_dn) } };
        const ssp = T{ .name = "SSPT", .value = .{ .int = @intCast(key.sp_gu) } };
        const sspd = T{ .name = "SSPTD", .value = .{ .int = @intCast(key.sp_dn) } };
        const bw: u32 = 16 * bt;
        const use_gu8 = rows <= self.gu8_rows and self.gu8Admitted(rows, self.kdiv9);
        const sgu_shapes = [_][]const c_int{ &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ R, nbh }, &[_]c_int{ R, nbh } };
        const gu = if (use_gu8)
            try ks.buildConfig(.{
                .grid = .{ (g.inter / bw) * 128, rk + ys_gu, z_gu },
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &([_][]const c_int{ &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ rkc, nbh }, &[_]c_int{ rkc, nbh } } ++ sgu_shapes),
                .out_dtypes = &.{ .float32, .float32, .float32, .float32, .float32, .float32, .float32, .float32 },
                .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t, topk, sng, sksg, srs, sdef, sz_gu, ssp },
            })
        else
            try ks.buildConfig(.{
                .grid = .{ (g.inter / bw) * 128, rk + ys_gu, z_gu },
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &([_][]const c_int{ &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(self.ks_gu), rkc, I } } ++ sgu_shapes),
                .out_dtypes = &.{ .float32, .float32, .float32, .float32, .float32, .float32 },
                .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t, sng, sksg, srs, sdef, sz_gu, ssp },
            });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gu);
        const fin = try ks.buildConfig(.{
            .grid = .{ (g.inter / 128) * 32, rk + @as(u32, @intCast(rows)), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, nbi }, &[_]c_int{ R, I }, &[_]c_int{ R, nbi } },
            .out_dtypes = &.{ .float16, .float32, .float16, .float32 },
            .template = &.{ inter_t, rkt, ks_gu_t, def_t, nbh_t, out_t, sksg, srs, sdef },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(fin);
        const down = try ks.buildConfig(.{
            .grid = .{ (g.hidden / bw) * 128, rk + ys_dn, z_dn },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ @intCast(self.ks_dn), rkc, H }, &[_]c_int{ @intCast(sb.ks_dn), R, H } },
            .out_dtypes = &.{ .float32, .float32 },
            .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_dn_t, spt_dn_t, bt_t, grp_t, def_t, snd, sksd, srs, sdef, sz_dn, sspd },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(down);
        const dfin = try ks.buildConfig(.{
            .grid = .{ (g.hidden / 128) * 32 * (g.topk + 1), @intCast(rows), 1 },
            .threadgroup = .{ 32 * (g.topk + 1), 1, 1 },
            .out_shapes = &.{&[_]c_int{ R, H }},
            .out_dtypes = &.{self.out_dtype},
            .template = &.{ hid, topk, rkt, ks_dn_t, out_t, def_t, nbi_t, sksd, srs, sdef },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(dfin);
        // past gu8_rows: gu3s's grid and template (and fin7s's output dtype) for gu3sf, with fin7s's four outputs, then
        // the routed and shared planes and the shared input sums' block copies; one counter per routed (row-slot,
        // intermediate block), then one per shared block (at least 8: MLX binds an input of fewer than 8 elements in the
        // constant address space, no atomics there)
        const vf: ?VFold = if (use_gu8) null else blk: {
            if (self.gu3sf == null) self.gu3sf = ks.newKernel("turboexl3_gu3sf", &gu3sf_inputs, &gu3sf_outputs, text_gu3sf, header_s, true) orelse return error.KernelCreateFailed;
            if (self.vzero == null) self.vzero = ks.newKernel("turboexl3_vfold_zero", &vfold_zero_inputs, &vfold_zero_outputs, text_vfold_zero, header, true) orelse return error.KernelCreateFailed;
            const nbi_u: u32 = g.inter / 128;
            const ncnt: u32 = @max(rk * nbi_u + nbi_u, 8);
            const nsx: u32 = if (sb.defer_affine) @max(nbi_u * @as(u32, @intCast(rows)) * 2 * (g.hidden / 128), 8) else 8;
            const gcfg = try ks.buildConfig(.{
                .grid = .{ (g.inter / bw) * 128, rk + ys_gu, z_gu },
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, nbi }, &[_]c_int{ R, I }, &[_]c_int{ R, nbi }, &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{@intCast(nsx)} },
                .out_dtypes = &.{ .float16, .float32, .float16, .float32, .float32, .float32, .float32, .float32, .float32 },
                .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t, sng, sksg, srs, sdef, sz_gu, ssp, out_t },
            });
            errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gcfg);
            const zcfg = try ks.buildConfig(.{ .grid = .{ ((ncnt + 127) / 128) * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{@intCast(ncnt)}}, .out_dtypes = &.{.uint32}, .template = &.{.{ .name = "NCNT", .value = .{ .int = @intCast(ncnt) } }} });
            break :blk .{ .gu = gcfg, .zero = zcfg };
        };
        const c: SCfg = .{ .gu = gu, .fin = fin, .down = down, .dfin = dfin, .gu8 = use_gu8, .vf = vf };
        self.scfg[rows] = c;
        self.scfg_key[rows] = key;
        return c;
    }

    /// Form 9 with the shared expert folded in: y = the routed sum in the output dtype plus sigmoid(logit) * shared(x), in the
    /// bytes of the host's chain (form 3's routed y, the three dense linears, the table SwiGLU, MLX's sigmoid, Multiply, Add)
    /// with no launch of its own: the shared GEMVs ride the gate|up and down launches as extra grid rows, its finishes ride
    /// launches 3 and 5. Past gu8_rows with `fused_fin_fold` launches 2 and 3 are one (`sharedFold`). The caller frees
    /// y; `sharedAdmitted` first.
    pub fn moeShared(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, sb: SharedBank) !mlx.mlx_array {
        if (!self.sharedAdmitted(rows, sb)) return error.BadExl3Shape;
        const sc = try self.sharedCfgs(rows, sb);
        // routed: zg, zu, xsg, xsu; shared: szg, szu, sxsg, sxsu
        var pl: [8]mlx.mlx_array = undefined;
        if (sc.gu8) {
            try ks.applyConfig(self.gu8s, sc.gu, &.{ x, slots, bank.gate_suh, bank.up_suh, bank.gate_t, bank.up_t, x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &pl, s);
        } else {
            const c = try self.cfgs9(rows);
            var xp: [4]mlx.mlx_array = undefined; // xg, xu, and their per-block input sums
            try ks.applyConfig(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh, bank.up_suh }, &xp, s);
            if (self.fused_fin_fold and sc.vf != null) {
                defer for (xp) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                return self.sharedFold(s, sc, xp, x, slots, scores, bank, sb, sb.logit);
            }
            defer _ = mlx.mlx_array_free(xp[0]);
            defer _ = mlx.mlx_array_free(xp[1]);
            errdefer _ = mlx.mlx_array_free(xp[2]);
            errdefer _ = mlx.mlx_array_free(xp[3]);
            var o: [6]mlx.mlx_array = undefined;
            try ks.applyConfig(self.gu3s, sc.gu, &.{ xp[0], xp[1], slots, bank.gate_t, bank.up_t, x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &o, s);
            pl = .{ o[0], o[1], xp[2], xp[3], o[2], o[3], o[4], o[5] };
        }
        defer for (pl) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var hd: [4]mlx.mlx_array = undefined; // hd, hs; the shared down's prepared input sxpd and its block sums sxsd
        try ks.applyConfig(self.fin7s, sc.fin, &.{ pl[0], pl[1], slots, bank.gate_svh, bank.up_svh, bank.down_suh, pl[2], pl[3], pl[4], pl[5], pl[6], pl[7], sb.gate_svh, sb.up_svh, sb.down_suh, sb.sigtab }, &hd, s);
        defer for (hd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var zd: [2]mlx.mlx_array = undefined; // the routed down planes, the shared down planes
        try ks.applyConfig(self.down3s, sc.down, &.{ hd[0], slots, bank.down_t, hd[2], sb.down_t }, &zd, s);
        defer for (zd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        return one(self.dfin7s, sc.dfin, &.{ zd[0], slots, scores, bank.down_svh, hd[1], zd[1], hd[3], sb.down_svh, sb.sigtab, sb.logit }, s);
    }

    /// `moeShared`'s launches past its prepare with gu3s and fin7s as one (gu3sf, after the counters' zeroing launch),
    /// then down3s and dfin7s unchanged; `xp` the routed prepare's xg, xu and block sums (the caller's), `logit` the shared
    /// gate's. The caller frees y.
    fn sharedFold(self: *Kernels3, s: mlx.mlx_stream, sc: SCfg, xp: [4]mlx.mlx_array, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, sb: SharedBank, logit: mlx.mlx_array) !mlx.mlx_array {
        const vf = sc.vf orelse return error.BadExl3Shape;
        const cnt = try one(self.vzero.?, vf.zero, &.{bank.gate_svh}, s);
        defer _ = mlx.mlx_array_free(cnt);
        var o: [9]mlx.mlx_array = undefined; // hd, hs, sxpd, sxsd; zg, zu, szg, szu, sxsx (scratch)
        try ks.applyConfig(self.gu3sf.?, vf.gu, &.{ xp[0], xp[1], slots, bank.gate_t, bank.up_t, x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh, bank.gate_svh, bank.up_svh, bank.down_suh, xp[2], xp[3], sb.gate_svh, sb.up_svh, sb.down_suh, sb.sigtab, cnt }, &o, s);
        defer for (o) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var zd: [2]mlx.mlx_array = undefined; // the routed down planes, the shared down planes
        try ks.applyConfig(self.down3s, sc.down, &.{ o[0], slots, bank.down_t, o[2], sb.down_t }, &zd, s);
        defer for (zd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        return one(self.dfin7s, sc.dfin, &.{ zd[0], slots, scores, bank.down_svh, o[1], zd[1], o[3], sb.down_svh, sb.sigtab, logit }, s);
    }

    /// `routeLaunch`'s config key: the experts, the scores' dtype, the gate's width (0: none).
    pub const RouteKey = struct { ne: u32, tout: mlx.mlx_dtype, gk: u32 };
    /// The merged launch's outputs, each the caller's: prep2's (the gate|up inputs prepared, their per-block sums), the
    /// router's (the slots, the renormalized weights) and the shared gate's logit per row (an empty handle without a gate).
    pub const RoutedPrep = struct { xg: mlx.mlx_array, xu: mlx.mlx_array, xsg: mlx.mlx_array, xsu: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, glogit: mlx.mlx_array };

    /// Simdgroups per merged threadgroup: one per row-slot (phase 2's prepares), one more for the router's softmax and weights,
    /// and with the gate at least nine below that one (simdgroup 0 selects while 1..8 take the gate's dot).
    pub fn routeSimdgroups(topk: u32, gate: bool) u32 {
        return @max(topk, if (gate) @as(u32, 9) else 1) + 1;
    }

    /// The router's top-k and weights and form 3's prepare in one launch (`text_route_prep`) from the router gemv's
    /// logits [rows, E] (the scores in their dtype) and x [rows, hidden]; with `gate_vec` (the shared gate's weight
    /// vector, bf16 [hidden], x bf16) also the shared gate's logit per row, in MLX's gemv order. The bytes of the
    /// router kernel (`text_router_softmax`) and prep2 (`routeChainPrep`), which the GPU test pins.
    pub fn routeLaunch(self: *Kernels3, s: mlx.mlx_stream, logits: mlx.mlx_array, x: mlx.mlx_array, bank: Kernels.Bank, rows: usize, gate_vec: ?mlx.mlx_array) !RoutedPrep {
        const g = self.geo;
        if (rows == 0 or rows > Kernels.max_rows or g.topk > 31) return error.Exl3EngineNotImplemented;
        const lsh = mlx.getShape(logits);
        if (lsh.len != 2 or lsh[0] != @as(c_int, @intCast(rows)) or lsh[1] < @as(c_int, @intCast(g.topk)) or lsh[1] > 2048) return error.Exl3EngineNotImplemented;
        const tout = mlx.mlx_array_dtype(logits);
        if (tout != .bfloat16 and tout != .float16 and tout != .float32) return error.Exl3EngineNotImplemented;
        const gate = gate_vec != null;
        if (gate_vec) |gv| {
            if (mlx.mlx_array_dtype(gv) != .bfloat16 or mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_size(gv) != g.hidden) return error.Exl3EngineNotImplemented;
        }
        const key: RouteKey = .{ .ne = @intCast(lsh[1]), .tout = tout, .gk = if (gate) g.hidden else 0 };
        if (gate) {
            if (self.route_prepg == null) self.route_prepg = ks.newKernel("turboexl3_route_prepg", &route_prepg_inputs, &route_prepg_outputs, text_route_prepg, "", true) orelse return error.KernelCreateFailed;
        } else if (self.route_prep == null) {
            self.route_prep = ks.newKernel("turboexl3_route_prep", &route_prep_inputs, &route_prep_outputs, text_route_prep, "", true) orelse return error.KernelCreateFailed;
        }
        if (self.route_cfg[rows]) |c| {
            if (!std.meta.eql(self.route_key[rows], key)) {
                _ = mlx.mlx_fast_metal_kernel_config_free(c);
                self.route_cfg[rows] = null;
            }
        }
        if (self.route_cfg[rows] == null) {
            const tg: u32 = 32 * routeSimdgroups(g.topk, gate);
            const rk: c_int = @intCast(rows * g.topk);
            const R: c_int = @intCast(rows);
            const H: c_int = @intCast(g.hidden);
            const nbh: c_int = @intCast(g.hidden / 128);
            const tmpl = [_]ks.TemplateArg{
                .{ .name = "NE", .value = .{ .int = @intCast(key.ne) } },
                .{ .name = "NK", .value = .{ .int = @intCast(g.topk) } },
                .{ .name = "TG", .value = .{ .int = @intCast(tg) } },
                .{ .name = "TOUT", .value = .{ .dtype = tout } },
                .{ .name = "DIM", .value = .{ .int = H } },
                .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
                .{ .name = "GK", .value = .{ .int = H } },
            };
            const shapes = [_][]const c_int{ &[_]c_int{ rk, H }, &[_]c_int{ rk, H }, &[_]c_int{ rk, nbh }, &[_]c_int{ rk, nbh }, &[_]c_int{rk}, &[_]c_int{rk}, &[_]c_int{R} };
            const dtypes = [_]mlx.mlx_dtype{ .float16, .float16, .float32, .float32, .uint32, tout, .bfloat16 };
            const n: usize = if (gate) 7 else 6;
            self.route_cfg[rows] = try ks.buildConfig(.{
                .grid = .{ (g.hidden / 128) * tg, @intCast(rows), 1 },
                .threadgroup = .{ tg, 1, 1 },
                .out_shapes = shapes[0..n],
                .out_dtypes = dtypes[0..n],
                .template = tmpl[0..n],
            });
            self.route_key[rows] = key;
        }
        var o: [7]mlx.mlx_array = undefined;
        if (gate_vec) |gv| {
            try ks.applyConfig(self.route_prepg.?, self.route_cfg[rows].?, &.{ logits, x, bank.gate_suh, bank.up_suh, x, gv }, o[0..7], s);
        } else {
            try ks.applyConfig(self.route_prep.?, self.route_cfg[rows].?, &.{ logits, x, bank.gate_suh, bank.up_suh }, o[0..6], s);
            o[6] = .{ .ctx = null };
        }
        return .{ .xg = o[0], .xu = o[1], .xsg = o[2], .xsu = o[3], .slots = o[4], .scores = o[5], .glogit = o[6] };
    }

    /// The chain the GPU test pins `routeLaunch` to, prep2's half: form 9's prepare launch (prep2 at `cfgs9`'s config,
    /// as `moeShared` runs it) over the slots the router kernel wrote. Outputs xg, xu, xsg, xsu (the caller's).
    pub fn routeChainPrep(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, bank: Kernels.Bank, rows: usize) ![4]mlx.mlx_array {
        const c = try self.cfgs9(rows);
        var xp: [4]mlx.mlx_array = undefined;
        try ks.applyConfig(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh, bank.up_suh }, &xp, s);
        return xp;
    }

    /// `moeShared` with the routing folded into its prepare launch: `routeLaunch` in prep2's place, then `moeShared`'s
    /// gu3s, fin7s, down3s and dfin7s (with `fused_fin_fold`: gu3sf, down3s and dfin7s) unchanged on the launch's slots
    /// and scores and, with `gate_vec`, its gate logit in `sb.logit`'s place. The router kernel's and `moeShared`'s
    /// bytes, one launch and one dependent level fewer. error.Exl3EngineNotImplemented at the rows whose prepare rides the
    /// gate|up launch (gu8s: no prepare launch to fold into); `sharedAdmitted` first. The caller frees y.
    pub fn moeSharedRouted(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, logits: mlx.mlx_array, bank: Kernels.Bank, rows: usize, sb: SharedBank, gate_vec: ?mlx.mlx_array) !mlx.mlx_array {
        if (!self.sharedAdmitted(rows, sb)) return error.BadExl3Shape;
        const sc = try self.sharedCfgs(rows, sb);
        if (sc.gu8) return error.Exl3EngineNotImplemented;
        const r = try self.routeLaunch(s, logits, x, bank, rows, gate_vec);
        defer {
            _ = mlx.mlx_array_free(r.xg);
            _ = mlx.mlx_array_free(r.xu);
            _ = mlx.mlx_array_free(r.slots);
            _ = mlx.mlx_array_free(r.scores);
            if (r.glogit.ctx != null) _ = mlx.mlx_array_free(r.glogit);
        }
        if (self.fused_fin_fold and sc.vf != null) {
            defer _ = mlx.mlx_array_free(r.xsg);
            defer _ = mlx.mlx_array_free(r.xsu);
            return self.sharedFold(s, sc, .{ r.xg, r.xu, r.xsg, r.xsu }, x, r.slots, r.scores, bank, sb, if (gate_vec != null) r.glogit else sb.logit);
        }
        // moeShared's launches from here on, slots / scores / logit from the merged launch
        var o: [6]mlx.mlx_array = undefined;
        ks.applyConfig(self.gu3s, sc.gu, &.{ r.xg, r.xu, r.slots, bank.gate_t, bank.up_t, x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &o, s) catch |e| {
            _ = mlx.mlx_array_free(r.xsg);
            _ = mlx.mlx_array_free(r.xsu);
            return e;
        };
        const pl = [8]mlx.mlx_array{ o[0], o[1], r.xsg, r.xsu, o[2], o[3], o[4], o[5] };
        defer for (pl) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var hd: [4]mlx.mlx_array = undefined; // hd, hs; the shared down's prepared input sxpd and its block sums sxsd
        try ks.applyConfig(self.fin7s, sc.fin, &.{ pl[0], pl[1], r.slots, bank.gate_svh, bank.up_svh, bank.down_suh, pl[2], pl[3], pl[4], pl[5], pl[6], pl[7], sb.gate_svh, sb.up_svh, sb.down_suh, sb.sigtab }, &hd, s);
        defer for (hd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var zd: [2]mlx.mlx_array = undefined; // the routed down planes, the shared down planes
        try ks.applyConfig(self.down3s, sc.down, &.{ hd[0], r.slots, bank.down_t, hd[2], sb.down_t }, &zd, s);
        defer for (zd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        return one(self.dfin7s, sc.dfin, &.{ zd[0], r.slots, r.scores, bank.down_svh, hd[1], zd[1], hd[3], sb.down_svh, sb.sigtab, if (gate_vec != null) r.glogit else sb.logit }, s);
    }

    /// The standalone heads' configs for `moeSharedSide` (form 9's tiles per threadgroup): gate|up with every (K slice,
    /// projection) its own threadgroup along z, fin per (128-block, row), down with every K slice its own threadgroup.
    fn sideCfgs(self: *Kernels3, rows: usize, sb: SharedBank) !SSCfg {
        const key: SKey = .{ .n_gu = sb.n_gu, .n_dn = sb.n_dn, .ks_gu = sb.ks_gu, .ks_dn = sb.ks_dn, .def = sb.defer_affine, .sp_gu = sharedSpt(sb.ks_gu, self.sspt), .sp_dn = sharedSpt(sb.ks_dn, self.ssptd) };
        if (self.sscfg[rows]) |c| {
            if (std.meta.eql(self.sscfg_key[rows], key)) return c;
            inline for (.{ c.gu, c.fin, c.down }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            self.sscfg[rows] = null;
        }
        const g = self.geo;
        const bt: u32 = if (rows == 1) self.bt1 else self.bt9;
        const bw: u32 = 16 * bt;
        const R: c_int = @intCast(rows);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nbh: c_int = @intCast(g.hidden / 128);
        const nbi: c_int = @intCast(g.inter / 128);
        const T = ks.TemplateArg;
        const cb = T{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = T{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = T{ .name = "HID", .value = .{ .int = H } };
        const inter_t = T{ .name = "INTER", .value = .{ .int = I } };
        const rk0 = T{ .name = "RKC", .value = .{ .int = 0 } };
        const bt_t = T{ .name = "BT", .value = .{ .int = @intCast(bt) } };
        const out_t = T{ .name = "OUT_T", .value = .{ .dtype = self.out_dtype } };
        const nbh_t = T{ .name = "NBH", .value = .{ .int = nbh } };
        const sng = T{ .name = "SNG", .value = .{ .int = @intCast(sb.n_gu) } };
        const snd = T{ .name = "SND", .value = .{ .int = @intCast(sb.n_dn) } };
        const sksg = T{ .name = "SKSG", .value = .{ .int = @intCast(sb.ks_gu) } };
        const sksd = T{ .name = "SKSD", .value = .{ .int = @intCast(sb.ks_dn) } };
        const srs = T{ .name = "SRS", .value = .{ .int = R } };
        const sdef = T{ .name = "SDEF", .value = .{ .int = if (sb.defer_affine) 1 else 0 } };
        const gu = try ks.buildConfig(.{
            .grid = .{ (g.inter / bw) * 128, 1, 2 * (sb.ks_gu / key.sp_gu) },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ R, nbh }, &[_]c_int{ R, nbh } },
            .out_dtypes = &.{ .float32, .float32, .float32, .float32 },
            .template = &.{ srs, bt_t, rk0, wb, sksg, T{ .name = "SZ", .value = .{ .int = @intCast(2 * (sb.ks_gu / key.sp_gu)) } }, hid, sng, cb, sdef, inter_t, T{ .name = "SSPT", .value = .{ .int = @intCast(key.sp_gu) } } },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gu);
        const fin = try ks.buildConfig(.{
            .grid = .{ (g.inter / 128) * 32, @intCast(rows), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ R, I }, &[_]c_int{ R, nbi } },
            .out_dtypes = &.{ .float16, .float32 },
            .template = &.{ inter_t, rk0, nbh_t, sdef, sksg, srs, out_t },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(fin);
        const down = try ks.buildConfig(.{
            .grid = .{ (g.hidden / bw) * 128, 1, sb.ks_dn / key.sp_dn },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{&[_]c_int{ @intCast(sb.ks_dn), R, H }},
            .out_dtypes = &.{.float32},
            .template = &.{ srs, bt_t, rk0, wb, sksd, inter_t, T{ .name = "SZ", .value = .{ .int = @intCast(sb.ks_dn / key.sp_dn) } }, snd, cb, hid, sdef, T{ .name = "SSPTD", .value = .{ .int = @intCast(key.sp_dn) } } },
        });
        const c: SSCfg = .{ .gu = gu, .fin = fin, .down = down };
        self.sscfg[rows] = c;
        self.sscfg_key[rows] = key;
        return c;
    }

    /// Form 9's launches 1-4 (`moeForm`'s, less the final reduce): the routed down planes and the down inputs' block sums;
    /// the caller frees both.
    fn routedPlanes9(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, bank: Kernels.Bank, rows: usize) !struct { zd: mlx.mlx_array, hs: mlx.mlx_array } {
        const c = try self.cfgs9(rows);
        var planes: [4]mlx.mlx_array = undefined;
        if (rows <= self.gu8_rows and self.gu8Admitted(rows, self.kdiv9)) {
            try ks.applyConfig(self.gu8, c.gu8, &.{ x, slots, bank.gate_suh, bank.up_suh, bank.gate_t, bank.up_t }, &planes, s);
        } else {
            var xp: [4]mlx.mlx_array = undefined; // xg, xu, and their per-block input sums
            try ks.applyConfig(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh, bank.up_suh }, &xp, s);
            defer _ = mlx.mlx_array_free(xp[0]);
            defer _ = mlx.mlx_array_free(xp[1]);
            errdefer _ = mlx.mlx_array_free(xp[2]);
            errdefer _ = mlx.mlx_array_free(xp[3]);
            try ks.applyConfig(self.gu, c.gu, &.{ xp[0], xp[1], slots, bank.gate_t, bank.up_t }, planes[0..2], s);
            planes[2] = xp[2];
            planes[3] = xp[3];
        }
        defer for (planes) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var hd: [2]mlx.mlx_array = undefined; // the down's prepared input and its per-block sums
        try ks.applyConfig(self.fin7, c.fin7, &.{ planes[0], planes[1], slots, bank.gate_svh, bank.up_svh, bank.down_suh, planes[2], planes[3] }, &hd, s);
        defer _ = mlx.mlx_array_free(hd[0]);
        errdefer _ = mlx.mlx_array_free(hd[1]);
        const zd = try one(self.down, c.down, &.{ hd[0], slots, bank.down_t }, s);
        return .{ .zd = zd, .hs = hd[1] };
    }

    /// `moeShared`'s bytes with the shared expert in launches of its own beside form 9's (the routed kernels untouched; the
    /// fused heads enlarge every routed threadgroup's memory): gate|up (rotating in-kernel) beside launch 1, fin beside fin7,
    /// down beside down3, the reduce in dfin7s. The shared chain is as deep as the routed one, so MLX's tape (Kahn order from
    /// the outputs) puts each shared launch between the routed chain's own barriers. `sharedAdmitted` first; the caller frees y.
    pub fn moeSharedSide(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, sb: SharedBank) !mlx.mlx_array {
        if (!self.sharedAdmitted(rows, sb)) return error.BadExl3Shape;
        const sc = try self.sharedCfgs(rows, sb);
        const sd = try self.sideCfgs(rows, sb);
        const rp = try self.routedPlanes9(s, x, slots, bank, rows);
        defer _ = mlx.mlx_array_free(rp.zd);
        defer _ = mlx.mlx_array_free(rp.hs);
        var sp: [4]mlx.mlx_array = undefined; // szg, szu, sxsg, sxsu
        try ks.applyConfig(self.sgu, sd.gu, &.{ x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &sp, s);
        defer for (sp) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var sf: [2]mlx.mlx_array = undefined; // sxpd, sxsd
        try ks.applyConfig(self.sfin, sd.fin, &.{ sp[0], sp[1], sp[2], sp[3], sb.gate_svh, sb.up_svh, sb.down_suh, sb.sigtab }, &sf, s);
        defer for (sf) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const szd = try one(self.sdown, sd.down, &.{ sf[0], sb.down_t }, s);
        defer _ = mlx.mlx_array_free(szd);
        return one(self.dfin7s, sc.dfin, &.{ rp.zd, slots, scores, bank.down_svh, rp.hs, szd, sf[1], sb.down_svh, sb.sigtab, sb.logit }, s);
    }

    /// The v2 configs' key at `rows`: the shared bank's shape and the v2 knobs, the tiles per simdgroup resolved for the row
    /// count (`sharedOts`).
    fn sharedKey2(self: *const Kernels3, rows: usize, sb: SharedBank) TKey {
        const ri: usize = @min(rows, 4) - 1;
        const tiles = self.sharedTiles(rows);
        return .{
            .n_gu = sb.n_gu,
            .n_dn = sb.n_dn,
            .ks_gu = sb.ks_gu,
            .ks_dn = sb.ks_dn,
            .def = sb.defer_affine,
            .ots = tiles[0],
            .otsd = tiles[1],
            .prep = self.sprep,
            .lm = self.slm,
            .pre = self.spre[ri],
            .first = self.sfirst,
            .gu8_rows = self.gu8_rows,
        };
    }

    /// The v2 heads' shared threadgroup counts: gate|up K slices x groups of 4 ots tiles over both projections' inter / 16,
    /// down K slices x groups of 4 otsd of the hidden / 16 tiles.
    pub fn sharedTgs2(self: *const Kernels3, ks_gu: u32, ks_dn: u32, ots: u32, otsd: u32) struct { gu: u32, dn: u32 } {
        const g = self.geo;
        return .{ .gu = ks_gu * ((2 * (g.inter / 16)) / (4 * ots)), .dn = ks_dn * ((g.hidden / 16) / (4 * otsd)) };
    }

    /// The v2 inside configs (form 9's routed shape: bt1 / bt9, kdiv9) with the v2 heads' extra grid rows and constants: the
    /// gate|up grid's SX x SZ x SYS slots hold its shared threadgroups (the rows after the routed ones, or before them with
    /// `sfirst`), the same for the down; prep2t where form 9 runs its prepare launch and `sprep`.
    fn sharedCfgs2(self: *Kernels3, rows: usize, sb: SharedBank) !TCfg {
        const key = self.sharedKey2(rows, sb);
        if (self.tcfg[rows]) |c| {
            if (std.meta.eql(self.tcfg_key[rows], key)) return c;
            freeTCfg(c);
            self.tcfg[rows] = null;
        }
        const g = self.geo;
        const bt: u32 = if (rows == 1) self.bt1 else self.bt9;
        const bw: u32 = 16 * bt;
        const z_gu = self.kThreadgroups(self.ks_gu, rows, self.kdiv9);
        const z_dn = self.kThreadgroups(self.ks_dn, rows, self.kdiv9d orelse self.kdiv9);
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const R: c_int = @intCast(rows);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nbh: c_int = @intCast(g.hidden / 128);
        const nbi: c_int = @intCast(g.inter / 128);
        const nt = self.sharedTgs2(key.ks_gu, key.ks_dn, key.ots, key.otsd);
        const x_gu: u32 = g.inter / bw;
        const x_dn: u32 = g.hidden / bw;
        const ys_gu: u32 = (nt.gu + x_gu * z_gu - 1) / (x_gu * z_gu);
        const ys_dn: u32 = (nt.dn + x_dn * z_dn - 1) / (x_dn * z_dn);
        const T = ks.TemplateArg;
        const nhw = T{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } };
        const cb = T{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = T{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = T{ .name = "HID", .value = .{ .int = H } };
        const inter_t = T{ .name = "INTER", .value = .{ .int = I } };
        const rkt = T{ .name = "RKC", .value = .{ .int = rkc } };
        const topk = T{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } };
        const ks_gu_t = T{ .name = "KS", .value = .{ .int = @intCast(self.ks_gu) } };
        const ks_dn_t = T{ .name = "KS", .value = .{ .int = @intCast(self.ks_dn) } };
        const spt_gu_t = T{ .name = "SPT", .value = .{ .int = @intCast(self.ks_gu / z_gu) } };
        const spt_dn_t = T{ .name = "SPT", .value = .{ .int = @intCast(self.ks_dn / z_dn) } };
        const bt_t = T{ .name = "BT", .value = .{ .int = @intCast(bt) } };
        const grp_t = T{ .name = "GROUP", .value = .{ .int = @intCast(if (rows >= 2) self.group else 1) } };
        const out_t = T{ .name = "OUT_T", .value = .{ .dtype = self.out_dtype } };
        const def_t = T{ .name = "DEFER", .value = .{ .int = if (self.defer_affine) 1 else 0 } };
        const nbh_t = T{ .name = "NBH", .value = .{ .int = nbh } };
        const nbi_t = T{ .name = "NBI", .value = .{ .int = nbi } };
        const sng = T{ .name = "SNG", .value = .{ .int = @intCast(sb.n_gu) } };
        const snd = T{ .name = "SND", .value = .{ .int = @intCast(sb.n_dn) } };
        const sksg = T{ .name = "SKSG", .value = .{ .int = @intCast(sb.ks_gu) } };
        const sksd = T{ .name = "SKSD", .value = .{ .int = @intCast(sb.ks_dn) } };
        const srs = T{ .name = "SRS", .value = .{ .int = R } };
        const sdef = T{ .name = "SDEF", .value = .{ .int = if (sb.defer_affine) 1 else 0 } };
        const slm = T{ .name = "SLM", .value = .{ .int = if (key.lm) 1 else 0 } };
        const spre = T{ .name = "SPRE", .value = .{ .int = @intCast(key.pre) } };
        const snp = T{ .name = "SNP", .value = .{ .int = if ((g.inter / 16) % (4 * key.ots) == 0) 1 else 2 } };
        const h_gu = [_]T{
            sng, sksg, srs, sdef, slm, spre, snp,
            T{ .name = "SOTS", .value = .{ .int = @intCast(key.ots) } },
            T{ .name = "SX", .value = .{ .int = @intCast(x_gu) } },
            T{ .name = "SYS", .value = .{ .int = @intCast(ys_gu) } },
            T{ .name = "SYOFF", .value = .{ .int = if (key.first) @as(i32, @intCast(ys_gu)) else 0 } },
        };
        const h_dn = [_]T{
            snd, sksd, srs, sdef, slm, spre,
            T{ .name = "SOTS", .value = .{ .int = @intCast(key.otsd) } },
            T{ .name = "SX", .value = .{ .int = @intCast(x_dn) } },
            T{ .name = "SYS", .value = .{ .int = @intCast(ys_dn) } },
            T{ .name = "SYOFF", .value = .{ .int = if (key.first) @as(i32, @intCast(ys_dn)) else 0 } },
        };
        const use_gu8 = rows <= self.gu8_rows and self.gu8Admitted(rows, self.kdiv9);
        const use_prep = !use_gu8 and key.prep;
        const prep: ?mlx.mlx_fast_metal_kernel_config = if (use_prep) try ks.buildConfig(.{
            .grid = .{ (g.hidden / 128) * 32, rk + @as(u32, @intCast(rows)), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ rkc, H }, &[_]c_int{ rkc, H }, &[_]c_int{ rkc, nbh }, &[_]c_int{ rkc, nbh }, &[_]c_int{ R, H }, &[_]c_int{ R, H }, &[_]c_int{ R, nbh }, &[_]c_int{ R, nbh } },
            .out_dtypes = &.{ .float16, .float16, .float32, .float32, .float16, .float16, .float32, .float32 },
            .template = &.{ T{ .name = "DIM", .value = .{ .int = H } }, topk, rkt },
        }) else null;
        errdefer {
            if (prep) |p| _ = mlx.mlx_fast_metal_kernel_config_free(p);
        }
        const zgu = [_][]const c_int{ &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(self.ks_gu), rkc, I } };
        const xsgu = [_][]const c_int{ &[_]c_int{ rkc, nbh }, &[_]c_int{ rkc, nbh } };
        const szgu = [_][]const c_int{ &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ @intCast(sb.ks_gu), R, I } };
        const sxsgu = [_][]const c_int{ &[_]c_int{ R, nbh }, &[_]c_int{ R, nbh } };
        const gu_grid = [3]u32{ (g.inter / bw) * 128, rk + ys_gu, z_gu };
        const gu = if (use_gu8)
            try ks.buildConfig(.{
                .grid = gu_grid,
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &(zgu ++ xsgu ++ szgu ++ sxsgu),
                .out_dtypes = &.{ .float32, .float32, .float32, .float32, .float32, .float32, .float32, .float32 },
                .template = &([_]T{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t, topk } ++ h_gu),
            })
        else if (use_prep)
            try ks.buildConfig(.{
                .grid = gu_grid,
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &(zgu ++ szgu),
                .out_dtypes = &.{ .float32, .float32, .float32, .float32 },
                .template = &([_]T{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t } ++ h_gu),
            })
        else
            try ks.buildConfig(.{
                .grid = gu_grid,
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &(zgu ++ szgu ++ sxsgu),
                .out_dtypes = &.{ .float32, .float32, .float32, .float32, .float32, .float32 },
                .template = &([_]T{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t } ++ h_gu),
            });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gu);
        const fin = try ks.buildConfig(.{
            .grid = .{ (g.inter / 128) * 32, rk + @as(u32, @intCast(rows)), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, nbi }, &[_]c_int{ R, I }, &[_]c_int{ R, nbi } },
            .out_dtypes = &.{ .float16, .float32, .float16, .float32 },
            .template = &.{ inter_t, rkt, ks_gu_t, def_t, nbh_t, out_t, sksg, srs, sdef, slm },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(fin);
        const down = try ks.buildConfig(.{
            .grid = .{ (g.hidden / bw) * 128, rk + ys_dn, z_dn },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ @intCast(self.ks_dn), rkc, H }, &[_]c_int{ @intCast(sb.ks_dn), R, H } },
            .out_dtypes = &.{ .float32, .float32 },
            .template = &([_]T{ nhw, cb, wb, hid, inter_t, rkt, ks_dn_t, spt_dn_t, bt_t, grp_t, def_t } ++ h_dn),
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(down);
        const dfin = try ks.buildConfig(.{
            .grid = .{ (g.hidden / 128) * 32 * (g.topk + 1), @intCast(rows), 1 },
            .threadgroup = .{ 32 * (g.topk + 1), 1, 1 },
            .out_shapes = &.{&[_]c_int{ R, H }},
            .out_dtypes = &.{self.out_dtype},
            .template = &.{ hid, topk, rkt, ks_dn_t, out_t, def_t, nbi_t, sksd, srs, sdef, slm },
        });
        const c: TCfg = .{ .prep = prep, .gu = gu, .fin = fin, .down = down, .dfin = dfin, .gu8 = use_gu8 };
        self.tcfg[rows] = c;
        self.tcfg_key[rows] = key;
        return c;
    }

    /// The v2 beside configs: the standalone heads (RKC 0: every grid row is the shared expert's) over one-dimensional grids
    /// of their shared threadgroups (SX = the count, SYS = 1, SYOFF = 0); sprep2 then sgu2p where the routed chain runs its
    /// prepare launch (rows above gu8_rows, so both chains are as deep) and `sprep`, else sgu2 rotating in-kernel.
    fn sideCfgs2(self: *Kernels3, rows: usize, sb: SharedBank) !TSCfg {
        const key = self.sharedKey2(rows, sb);
        if (self.tscfg[rows]) |c| {
            if (std.meta.eql(self.tscfg_key[rows], key)) return c;
            freeTSCfg(c);
            self.tscfg[rows] = null;
        }
        const g = self.geo;
        const R: c_int = @intCast(rows);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nbh: c_int = @intCast(g.hidden / 128);
        const nbi: c_int = @intCast(g.inter / 128);
        const nt = self.sharedTgs2(key.ks_gu, key.ks_dn, key.ots, key.otsd);
        const T = ks.TemplateArg;
        const cb = T{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = T{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = T{ .name = "HID", .value = .{ .int = H } };
        const inter_t = T{ .name = "INTER", .value = .{ .int = I } };
        const rk0 = T{ .name = "RKC", .value = .{ .int = 0 } };
        const out_t = T{ .name = "OUT_T", .value = .{ .dtype = self.out_dtype } };
        const nbh_t = T{ .name = "NBH", .value = .{ .int = nbh } };
        const sng = T{ .name = "SNG", .value = .{ .int = @intCast(sb.n_gu) } };
        const snd = T{ .name = "SND", .value = .{ .int = @intCast(sb.n_dn) } };
        const sksg = T{ .name = "SKSG", .value = .{ .int = @intCast(sb.ks_gu) } };
        const sksd = T{ .name = "SKSD", .value = .{ .int = @intCast(sb.ks_dn) } };
        const srs = T{ .name = "SRS", .value = .{ .int = R } };
        const sdef = T{ .name = "SDEF", .value = .{ .int = if (sb.defer_affine) 1 else 0 } };
        const slm = T{ .name = "SLM", .value = .{ .int = if (key.lm) 1 else 0 } };
        const spre = T{ .name = "SPRE", .value = .{ .int = @intCast(key.pre) } };
        const snp = T{ .name = "SNP", .value = .{ .int = if ((g.inter / 16) % (4 * key.ots) == 0) 1 else 2 } };
        const one_d = [_]T{ T{ .name = "SYS", .value = .{ .int = 1 } }, T{ .name = "SYOFF", .value = .{ .int = 0 } } };
        const use_gu8 = rows <= self.gu8_rows and self.gu8Admitted(rows, self.kdiv9);
        const use_prep = !use_gu8 and key.prep;
        const prep: ?mlx.mlx_fast_metal_kernel_config = if (use_prep) try ks.buildConfig(.{
            .grid = .{ (g.hidden / 128) * 32, @intCast(rows), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ R, H }, &[_]c_int{ R, H }, &[_]c_int{ R, nbh }, &[_]c_int{ R, nbh } },
            .out_dtypes = &.{ .float16, .float16, .float32, .float32 },
            .template = &.{ T{ .name = "DIM", .value = .{ .int = H } }, rk0 },
        }) else null;
        errdefer {
            if (prep) |p| _ = mlx.mlx_fast_metal_kernel_config_free(p);
        }
        const h_gu = [_]T{
            wb, hid, inter_t, cb, rk0, sng, sksg, srs, sdef, slm, spre, snp,
            T{ .name = "SOTS", .value = .{ .int = @intCast(key.ots) } },
            T{ .name = "SX", .value = .{ .int = @intCast(nt.gu) } },
        } ++ one_d;
        const szgu = [_][]const c_int{ &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ @intCast(sb.ks_gu), R, I } };
        const sxsgu = [_][]const c_int{ &[_]c_int{ R, nbh }, &[_]c_int{ R, nbh } };
        const gu = if (use_prep)
            try ks.buildConfig(.{ .grid = .{ nt.gu * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &szgu, .out_dtypes = &.{ .float32, .float32 }, .template = &h_gu })
        else
            try ks.buildConfig(.{ .grid = .{ nt.gu * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &(szgu ++ sxsgu), .out_dtypes = &.{ .float32, .float32, .float32, .float32 }, .template = &h_gu });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gu);
        const fin = try ks.buildConfig(.{
            .grid = .{ (g.inter / 128) * 32, @intCast(rows), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ R, I }, &[_]c_int{ R, nbi } },
            .out_dtypes = &.{ .float16, .float32 },
            .template = &.{ inter_t, rk0, nbh_t, sdef, sksg, srs, out_t, slm },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(fin);
        const h_dn = [_]T{
            wb, hid, inter_t, cb, rk0, snd, sksd, srs, sdef, slm, spre,
            T{ .name = "SOTS", .value = .{ .int = @intCast(key.otsd) } },
            T{ .name = "SX", .value = .{ .int = @intCast(nt.dn) } },
        } ++ one_d;
        const down = try ks.buildConfig(.{
            .grid = .{ nt.dn * 128, 1, 1 },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{&[_]c_int{ @intCast(sb.ks_dn), R, H }},
            .out_dtypes = &.{.float32},
            .template = &h_dn,
        });
        const c: TSCfg = .{ .prep = prep, .gu = gu, .fin = fin, .down = down };
        self.tscfg[rows] = c;
        self.tscfg_key[rows] = key;
        return c;
    }

    /// `moeShared`'s bytes with the v2 heads (one simdgroup per (K slice, sots output tiles) writing dgemv's plane values with
    /// no threadgroup reduction; the gate|up input prepared once in prep2t where form 9 runs a prepare launch and `sprep`, else
    /// rotated per threadgroup; planes in layout `slm`). `sharedAdmitted` first; the caller frees y.
    pub fn moeShared2(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, sb: SharedBank) !mlx.mlx_array {
        if (!self.sharedAdmitted(rows, sb)) return error.BadExl3Shape;
        const tc = try self.sharedCfgs2(rows, sb);
        // routed: zg, zu, xsg, xsu; shared: szg, szu, sxsg, sxsu
        var pl: [8]mlx.mlx_array = undefined;
        if (tc.gu8) {
            try ks.applyConfig(self.gu8t, tc.gu, &.{ x, slots, bank.gate_suh, bank.up_suh, bank.gate_t, bank.up_t, x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &pl, s);
        } else if (tc.prep) |pc| {
            var xp: [8]mlx.mlx_array = undefined; // xg, xu and their block sums; the shared sxg, sxu and theirs
            try ks.applyConfig(self.prep2t, pc, &.{ x, slots, bank.gate_suh, bank.up_suh, sb.gate_suh, sb.up_suh }, &xp, s);
            defer _ = mlx.mlx_array_free(xp[0]);
            defer _ = mlx.mlx_array_free(xp[1]);
            defer _ = mlx.mlx_array_free(xp[4]);
            defer _ = mlx.mlx_array_free(xp[5]);
            errdefer _ = mlx.mlx_array_free(xp[2]);
            errdefer _ = mlx.mlx_array_free(xp[3]);
            errdefer _ = mlx.mlx_array_free(xp[6]);
            errdefer _ = mlx.mlx_array_free(xp[7]);
            var o: [4]mlx.mlx_array = undefined; // zg, zu, szg, szu
            try ks.applyConfig(self.gu3p, tc.gu, &.{ xp[0], xp[1], slots, bank.gate_t, bank.up_t, xp[4], xp[5], sb.gate_t, sb.up_t }, &o, s);
            pl = .{ o[0], o[1], xp[2], xp[3], o[2], o[3], xp[6], xp[7] };
        } else {
            const c = try self.cfgs9(rows);
            var xp: [4]mlx.mlx_array = undefined; // xg, xu, and their per-block input sums
            try ks.applyConfig(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh, bank.up_suh }, &xp, s);
            defer _ = mlx.mlx_array_free(xp[0]);
            defer _ = mlx.mlx_array_free(xp[1]);
            errdefer _ = mlx.mlx_array_free(xp[2]);
            errdefer _ = mlx.mlx_array_free(xp[3]);
            var o: [6]mlx.mlx_array = undefined; // zg, zu, szg, szu, sxsg, sxsu
            try ks.applyConfig(self.gu3t, tc.gu, &.{ xp[0], xp[1], slots, bank.gate_t, bank.up_t, x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &o, s);
            pl = .{ o[0], o[1], xp[2], xp[3], o[2], o[3], o[4], o[5] };
        }
        defer for (pl) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var hd: [4]mlx.mlx_array = undefined; // hd, hs; the shared down's prepared input sxpd and its block sums sxsd
        try ks.applyConfig(self.fin7t, tc.fin, &.{ pl[0], pl[1], slots, bank.gate_svh, bank.up_svh, bank.down_suh, pl[2], pl[3], pl[4], pl[5], pl[6], pl[7], sb.gate_svh, sb.up_svh, sb.down_suh, sb.sigtab }, &hd, s);
        defer for (hd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var zd: [2]mlx.mlx_array = undefined; // the routed down planes, the shared down planes
        try ks.applyConfig(self.down3t, tc.down, &.{ hd[0], slots, bank.down_t, hd[2], sb.down_t }, &zd, s);
        defer for (zd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        return one(self.dfin7t, tc.dfin, &.{ zd[0], slots, scores, bank.down_svh, hd[1], zd[1], hd[3], sb.down_svh, sb.sigtab, sb.logit }, s);
    }

    /// `moeShared2`'s bytes with the v2 heads in launches of their own beside form 9's untouched routed launches (sprep2 then
    /// sgu2p, or sgu2; sfin2; sdown2), the reduce in dfin7t. `sharedAdmitted` first; the caller frees y.
    pub fn moeSharedSide2(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, sb: SharedBank) !mlx.mlx_array {
        if (!self.sharedAdmitted(rows, sb)) return error.BadExl3Shape;
        const tc = try self.sharedCfgs2(rows, sb);
        const sd = try self.sideCfgs2(rows, sb);
        const rp = try self.routedPlanes9(s, x, slots, bank, rows);
        defer _ = mlx.mlx_array_free(rp.zd);
        defer _ = mlx.mlx_array_free(rp.hs);
        var sp: [4]mlx.mlx_array = undefined; // szg, szu, sxsg, sxsu
        if (sd.prep) |pc| {
            var xp: [4]mlx.mlx_array = undefined; // sxg, sxu, sxsg, sxsu
            try ks.applyConfig(self.sprep2, pc, &.{ x, sb.gate_suh, sb.up_suh }, &xp, s);
            defer _ = mlx.mlx_array_free(xp[0]);
            defer _ = mlx.mlx_array_free(xp[1]);
            errdefer _ = mlx.mlx_array_free(xp[2]);
            errdefer _ = mlx.mlx_array_free(xp[3]);
            var o: [2]mlx.mlx_array = undefined; // szg, szu
            try ks.applyConfig(self.sgu2p, sd.gu, &.{ xp[0], xp[1], sb.gate_t, sb.up_t }, &o, s);
            sp = .{ o[0], o[1], xp[2], xp[3] };
        } else {
            try ks.applyConfig(self.sgu2, sd.gu, &.{ x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &sp, s);
        }
        defer for (sp) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var sf: [2]mlx.mlx_array = undefined; // sxpd, sxsd
        try ks.applyConfig(self.sfin2, sd.fin, &.{ sp[0], sp[1], sp[2], sp[3], sb.gate_svh, sb.up_svh, sb.down_suh, sb.sigtab }, &sf, s);
        defer for (sf) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const szd = try one(self.sdown2, sd.down, &.{ sf[0], sb.down_t }, s);
        defer _ = mlx.mlx_array_free(szd);
        return one(self.dfin7t, tc.dfin, &.{ rp.zd, slots, scores, bank.down_svh, rp.hs, szd, sf[1], sb.down_svh, sb.sigtab, sb.logit }, s);
    }
};

/// Form 5's helpers, appended to `header` for its three kernels (`header5`): expert groups of up to GM row-slots, the GEMV over
/// prepared inputs held in threadgroup memory, and the H128 of a simdgroup's 128 values held 4 per lane.
const metal_form5 = @embedFile("kernels/decode/form5.metal");

pub const header5: [:0]const u8 = lane.metal_helpers ++ metal_common ++ metal_gemv_core ++ metal_form5;

/// The shared expert inside the routed launches (`Kernels3.moeShared`): dense.zig's decode-row arithmetic (prep_s, dgemv,
/// dfinish), so the shared expert's bytes are the dense linears'.
const metal_shared = @embedFile("kernels/decode/shared.metal");

pub const header_s: [:0]const u8 = header5 ++ metal_shared;

/// Form 5, launch 1: per (BT output tiles, lead row-slot, K slice): the expert group's members' rotated inputs over the slice
/// (prep2's arithmetic, into threadgroup memory: no prepare launch, no device round trip), then gu3's GEMV for every member
/// at once (each weight decoded once per group) into the same split-K planes zg / zu f32 [KS, RK, inter], and each 128-block's
/// input sum (xsg / xsu [RK, hidden / 128], written by the blk 0 threadgroups) for the deferred MUL1 map.
pub const text_gu5 = @embedFile("kernels/decode/gu5.metal");

/// Form 5, launch 2: per (128-wide hidden block, lead row-slot), 8 simdgroups: every member's mid over the whole intermediate
/// (fin3's arithmetic: the planes' sum, the deferred map, f16 round, H128, svh, f16 round, SwiGLU, the down's suh, H128, f16
/// round, lane order, and each 128-block's sum) into threadgroup memory, then the down GEMV of the block's 8 output tiles
/// (one per simdgroup) for every member in down3's K partition, then dfin3's per-slot finish up to the score: the deferred
/// map, f16 round, H128, svh, f16 round -> wv half [RK, hidden].
const dn5_head = @embedFile("kernels/decode/dn5_head.metal");
/// Form 5's down tail: the finished values to wv for launch 3.
const dn5_tail = @embedFile("kernels/decode/dn5_tail.metal");
/// Form 6's down tail: the finished values stored through coherent pointers, then member j's row counts this 128-block's
/// arrival (device-scope seq_cst fences around a relaxed fetch_add; no waiting) and the simdgroup that brings
/// the row's TOPK-th slot reduces the block in slot order (red5's fma chain, so red5's bytes) straight into y.
const dn6_tail = @embedFile("kernels/decode/dn6_tail.metal");

pub const text_dn5: [:0]const u8 = dn5_head ++ dn5_tail;
pub const text_dn6: [:0]const u8 = dn5_head ++ dn6_tail;

/// Form 5, launch 3: y [rows, hidden] = the score-weighted sum of a row's slots in slot order (dfin3's fma chain).
pub const text_red5 = @embedFile("kernels/decode/red5.metal");

const gu5_inputs = [_][*:0]const u8{ "x", "slots", "gate_suh", "up_suh", "gate_t", "up_t" };
const gu5_outputs = [_][*:0]const u8{ "zg", "zu", "xsg", "xsu", "cnt" };
const dn5_inputs = [_][*:0]const u8{ "zg", "zu", "xsg", "xsu", "slots", "gate_svh", "up_svh", "down_suh", "down_t", "down_svh" };
const dn5_outputs = [_][*:0]const u8{"wv"};
const dn6_inputs = [_][*:0]const u8{ "zg", "zu", "xsg", "xsu", "slots", "gate_svh", "up_svh", "down_suh", "down_t", "down_svh", "cnt", "scores" };
const dn6_outputs = [_][*:0]const u8{ "wv", "y" };
const red5_inputs = [_][*:0]const u8{ "wv", "scores" };
const red5_outputs = [_][*:0]const u8{"y"};

/// Form 7, launch 3 (fin3 with the H128s in registers): one simdgroup per (128-wide intermediate block, row-slot), lane L
/// holding elements L + 32 q; fin3's arithmetic in fin3's order (`turboexl3_h128r` is `turboexl3_h128`'s stages, the block sum
/// the same four simd sums), so fin3's bytes without its twelve threadgroup barriers.
pub const text_fin7 = fin7_1 ++ fin7_2 ++ fin7_planes ++ fin7_3 ++ fin7_sums ++ fin7_4 ++ fin7_swiglu ++ fin7_5;
/// fin7's pieces: its constants and grid lines; its body from the expert line, cut at the planes' read (gu3sf reads the
/// planes through coherent pointers), at the input sums' read and at the SwiGLU's line (`text_fin7c` clamps there)
const fin7_1 = @embedFile("kernels/decode/fin7_1.metal");
const fin7_2 = @embedFile("kernels/decode/fin7_2.metal");
const fin7_planes = @embedFile("kernels/decode/fin7_planes.metal");
const fin7_3 = @embedFile("kernels/decode/fin7_3.metal");
const fin7_sums = @embedFile("kernels/decode/fin7_sums.metal");
const fin7_4 = @embedFile("kernels/decode/fin7_4.metal");
const fin7_swiglu = @embedFile("kernels/decode/fin7_swiglu.metal");
const fin7_5 = @embedFile("kernels/decode/fin7_5.metal");

/// Form 7's launch 3 with the clamped SwiGLU (`Geometry.limit`, the template int LIMIT; `root.moeClamped`): fin7 with its
/// SwiGLU line replaced by the clamp (the gate clamped from above at LIMIT and up to [-LIMIT, LIMIT], then fin7's SwiGLU
/// on them), every other statement fin7's, so where no value passes the limit it gives fin7's bytes.
pub const text_fin7c = fin7_1 ++ fin7_2 ++ fin7_planes ++ fin7_3 ++ fin7_sums ++ fin7_4 ++ fin7_swiglu_clamp ++ fin7_5;
const fin7_swiglu_clamp = @embedFile("kernels/decode/fin7_swiglu_clamp.metal");

/// Form 7, launch 5 (dfin3 with the slots side by side): per (128-wide hidden block, row), simdgroup k finishes slot k (the
/// planes' sum, the deferred map, f16 round, H128 in registers, svh, f16 round) into threadgroup memory, then 128 threads
/// run dfin3's score fma chain in slot order: dfin3's bytes, one H128 deep where dfin3 runs TOPK of them in a row.
pub const text_dfin7 = @embedFile("kernels/decode/dfin7.metal");

/// Form 8, launch 1 (prep2 + gu3 in one launch): gu3's threadgroup over its SPT slices, whose span (SPT hidden / KS values,
/// 128-aligned: the host admits it only then) the four simdgroups first rotate for every member and both projections in
/// prep2's arithmetic into threadgroup memory; then gu3's GEMV reads the input from there. The blk 0 threadgroups write
/// each block's input sum (xsg / xsu [RK, hidden / 128]) for fin's deferred map. gu3's planes, so form 3's bytes.
pub const text_gu8 = @embedFile("kernels/decode/gu8_1.metal") ++ rk_line ++ @embedFile("kernels/decode/gu8_2.metal");
const gu8_inputs = [_][*:0]const u8{ "x", "slots", "gate_suh", "up_suh", "gate_t", "up_t" };
const gu8_outputs = [_][*:0]const u8{ "zg", "zu", "xsg", "xsu" };

/// The shared expert in the routed launches (`Kernels3.moeShared`), launch 1's head: grid rows from RKC on are the shared
/// gate|up GEMV's threadgroups, j = (y - RKC) * SZ + z -> projection j / (SKSG / SSPT) and the SSPT consecutive K slices from
/// (j % (SKSG / SSPT)) SSPT (a slice = one 128-block of hidden: SKSG = hidden / 128); each slice is rotated in-kernel (dense
/// prep_s) and run as dense dgemv's slice for the grid's BT output tiles, exactly as a threadgroup of its own computes it, so
/// SSPT is launch shape only. Planes szg / szu f32 [SKSG, SRS, inter], block sums sxsg / sxsu [SRS, hidden / 128].
const shared_gu_head = sgh_1 ++ sgh_2 ++ sgh_3 ++ sgh_z ++ sgh_4 ++ sgh_sums ++ sgh_5 ++ sgh_end;
/// the head's pieces, cut where gu3sf differs: after the threadgroup arrays and the thread lines (its declarations), at
/// the planes' pointer, at the input sums' store and before the end
const sgh_1 = @embedFile("kernels/decode/shared_gu_head_1.metal");
const sgh_2 = @embedFile("kernels/decode/shared_gu_head_2.metal");
const sgh_3 = @embedFile("kernels/decode/shared_gu_head_3.metal");
const sgh_z = @embedFile("kernels/decode/shared_gu_head_z.metal");
const sgh_4 = @embedFile("kernels/decode/shared_gu_head_4.metal");
const sgh_sums = @embedFile("kernels/decode/shared_gu_head_sums.metal");
const sgh_5 = @embedFile("kernels/decode/shared_gu_head_5.metal");
const sgh_end = @embedFile("kernels/decode/shared_gu_head_end.metal");
pub const text_gu3s: [:0]const u8 = shared_gu_head ++ text_gu3;
pub const text_gu8s: [:0]const u8 = shared_gu_head ++ text_gu8;
const shared_gu_inputs = [_][*:0]const u8{ "xr", "sh_gate_t", "sh_up_t", "sh_gate_suh", "sh_up_suh" };
const shared_gu_outputs = [_][*:0]const u8{ "szg", "szu", "sxsg", "sxsu" };
const gu3s_inputs = gu3_inputs ++ shared_gu_inputs;
const gu3s_outputs = gu3_outputs ++ shared_gu_outputs;
const gu8s_inputs = gu8_inputs ++ shared_gu_inputs;
const gu8s_outputs = gu8_outputs ++ shared_gu_outputs;

/// Launch 3's head: grid rows RKC.. RKC + SRS are the shared expert's rows: per 128-block of the intermediate, dense dfinish of
/// the gate and up planes (to OUT_T), the host's table SwiGLU (`T act = g * sigtab[g]; act * up`, sigtab = MLX's sigmoid of
/// every 16-bit pattern), then dense prep_s of the product for the down: sxpd half [SRS, inter] lane order, sxsd [SRS, NBI].
const shared_fin_head = sfh_1 ++ sfh_2 ++ sfh_xg ++ sfh_xu ++ sfh_3 ++ sfh_g ++ sfh_u ++ sfh_4 ++ sfh_end;
/// the head's pieces: its constants and grid lines; its body from the input sums, cut at the two sums' reads and the two
/// finishes (the lines gu3sf maps); its end
const sfh_1 = @embedFile("kernels/decode/shared_fin_head_1.metal");
const sfh_2 = @embedFile("kernels/decode/shared_fin_head_2.metal");
const sfh_xg = @embedFile("kernels/decode/shared_fin_head_xg.metal");
const sfh_xu = @embedFile("kernels/decode/shared_fin_head_xu.metal");
const sfh_3 = @embedFile("kernels/decode/shared_fin_head_3.metal");
const sfh_g = @embedFile("kernels/decode/shared_fin_head_g.metal");
const sfh_u = @embedFile("kernels/decode/shared_fin_head_u.metal");
const sfh_4 = @embedFile("kernels/decode/shared_fin_head_4.metal");
const sfh_end = @embedFile("kernels/decode/shared_fin_head_end.metal");
pub const text_fin7s: [:0]const u8 = shared_fin_head ++ text_fin7;
const sfin_inputs = [_][*:0]const u8{ "szg", "szu", "sxsg", "sxsu", "sh_gate_svh", "sh_up_svh", "sh_down_suh", "sigtab" };
const sfin_outputs = [_][*:0]const u8{ "sxpd", "sxsd" };
const fin7s_inputs = fin3_inputs ++ sfin_inputs;
const fin7s_outputs = fin3_outputs ++ sfin_outputs;

// ── fin7s folded into gu3s at the fused verify widths ──
//
// gu3sf is gu3s (the shared gate|up head, then gu3) with fin7s's work done by the threadgroups that complete it, so the fused
// shared expert's rows past gu8_rows run gu3sf, down3s, dfin7s after a counter zeroing that depends on nothing.
// - Routed: every threadgroup stores its planes through coherent pointers and counts itself at each member's (row-slot,
//   128-block of the intermediate) counter; the one that brings it to ARR (the block's GC column groups x the slot's K
//   threadgroups) runs fin7's statements for that pair in simdgroup j for member j (the input sums are the routed
//   prepare's, read as fin7 reads them).
// - Shared: every head threadgroup stores its planes through coherent pointers, publishes its slices' input sums into its
//   block's own copy (sxsx [NBI, SRS, 2, NBH], written where s_blk % S_GC == 0: one writer per value, and every writer is
//   one of the block's arrivals) and counts itself at its 128-block's counter; the one that brings it to S_ARR (the block's
//   column groups x both projections' slice groups) runs the shared fin head's statements for the block, every row
//   (simdgroup s_sg: rows s_sg, s_sg + 4, ...), with `turboexl3_sfinish`'s statements inline over the coherent planes
//   (vf_sfin_g / vf_sfin_u: its body with its parameters bound by name).
// DEC-ONE's arrival idiom: device-scope seq_cst fences around one relaxed fetch_add per threadgroup, nothing waits. The
// counters (routed [RK, NBI], then shared [NBI]) come zeroed from text_vfold_zero. fin7s's bytes, one dependent launch fewer.
pub const text_gu3sf = sgh_1 ++ vf_head_tg ++ sgh_2 ++ vf_head_decls ++ sgh_3 ++ vf_head_zc ++ sgh_4 ++ vf_head_sumsc ++ sgh_5 ++
    vf_head_arrive ++ sfh_2 ++ vf_sfin_xgc ++ vf_sfin_xuc ++ sfh_3 ++ vf_sfin_g ++ vf_sfin_u ++ sfh_4 ++ vf_head_close ++ sgh_end ++
    gu3_1 ++ rk_line ++ gu3_2 ++ vf_gu3_decls ++ gu3_3 ++ vf_gu3_stores ++ gu3_4 ++ vf_arrive ++
    fin7_2 ++ vf_fin7_planes ++ fin7_3 ++ fin7_sums ++ fin7_4 ++ fin7_swiglu ++ fin7_5 ++ vf_fin7_close;
const vf_head_tg = @embedFile("kernels/decode/vf_head_tg.metal");
const vf_head_decls = @embedFile("kernels/decode/vf_head_decls.metal");
const vf_head_zc = @embedFile("kernels/decode/vf_head_zc.metal");
const vf_head_sumsc = @embedFile("kernels/decode/vf_head_sumsc.metal");
const vf_head_arrive = @embedFile("kernels/decode/vf_head_arrive.metal");
const vf_sfin_xgc = @embedFile("kernels/decode/vf_sfin_xgc.metal");
const vf_sfin_xuc = @embedFile("kernels/decode/vf_sfin_xuc.metal");
const vf_sfin_g = @embedFile("kernels/decode/vf_sfin_g.metal");
const vf_sfin_u = @embedFile("kernels/decode/vf_sfin_u.metal");
const vf_head_close = @embedFile("kernels/decode/vf_head_close.metal");
const vf_gu3_decls = @embedFile("kernels/decode/vf_gu3_decls.metal");
const vf_gu3_stores = @embedFile("kernels/decode/vf_gu3_stores.metal");
const vf_arrive = @embedFile("kernels/decode/vf_arrive.metal");
const vf_fin7_planes = @embedFile("kernels/decode/vf_fin7_planes.metal");
const vf_fin7_close = @embedFile("kernels/decode/vf_fin7_close.metal");
/// gu3s's inputs, then fin7s's (less the planes and the shared sums gu3sf keeps) and the counters
const gu3sf_inputs = gu3s_inputs ++ [_][*:0]const u8{ "gate_svh", "up_svh", "down_suh", "xsg", "xsu", "sh_gate_svh", "sh_up_svh", "sh_down_suh", "sigtab", "cnt" };
/// fin7s's outputs (hd, hs, sxpd, sxsd), then the routed and shared planes and the shared input sums' block copies (scratch)
const gu3sf_outputs = fin7s_outputs ++ [_][*:0]const u8{ "zg", "zu", "szg", "szu", "sxsx" };
/// gu3sf's arrival counters, zeroed by a launch nothing it reads depends on (gate_svh is a weight, taken only for
/// ordering), so MLX needs no barrier before it: dense's DEC-ONE zeroing text
pub const text_vfold_zero = @embedFile("kernels/dense/dec_zero.metal");
const vfold_zero_inputs = [_][*:0]const u8{"gate_svh"};
const vfold_zero_outputs = [_][*:0]const u8{"cnt"};

/// Launch 4's head: grid rows from RKC on are the shared down GEMV's threadgroups, each over the SSPTD consecutive slices from
/// ((y - RKC) * SZ + z) SSPTD of SKSD (dense dgemv over sxpd, every slice as a threadgroup of its own computes it), planes
/// szd f32 [SKSD, SRS, hidden].
const shared_down_head = @embedFile("kernels/decode/shared_down_head.metal");
pub const text_down3s: [:0]const u8 = shared_down_head ++ text_down3;
const sdown_inputs = [_][*:0]const u8{ "sxpd", "sh_down_t" };
const sdown_outputs = [_][*:0]const u8{"szd"};
const down3s_inputs = down3_inputs ++ sdown_inputs;
const down3s_outputs = [_][*:0]const u8{ "zd", "szd" };

/// Launch 5 with the shared expert: dfin7's slots in simdgroups 0..TOPK-1 and the shared down's dense dfinish in simdgroup TOPK,
/// then per output the routed sum in OUT_T plus the host's gate tail in OUT_T: sigmoid(logit) (MLX's, by table) * down, then
/// the sum (MLX's Multiply and Add), so y = moeSharedGateTail(routed, logit, down)'s bytes.
pub const text_dfin7s = @embedFile("kernels/decode/dfin7s.metal");
const dfin7s_inputs = dfin3_inputs ++ [_][*:0]const u8{ "szd", "sxsd", "sh_down_svh", "sigtab", "slogit" };

/// The v2 shared heads' GEMV helpers (`Kernels3.moeShared2` / `moeSharedSide2`), appended to `header_s` (`header_t`). One
/// simdgroup per (K slice, OTS consecutive output tiles) computes dense dgemv's plane values of its tiles by itself: the
/// slice's four parts (dgemv's simdgroups: PER input tiles each) in dgemv's order (tile order, weights m = 0..7), each part
/// from zero and folded over the quad lanes (dgemv's `turboexl3_fold`), the slice the left sum of its parts (dgemv's plane
/// store ((part 0 + part 1) + part 2) + part 3). The planes are dgemv's bit for bit with no threadgroup reduction and no
/// second barrier; the finishes keep dense dfinish's order (plane 0, then + plane k in slice order) and read the planes in
/// the natural layout or lane-major (LM: one float4 per plane and lane). The parts take every codeword from the window's
/// 32-bit words (`turboexl3_scw`, exl3_tile_cw's 96-bit rules) for both window readers: no 64-bit window (`Exl3BitsM<1>`),
/// which the Metal back-end compiler failed to build in the down head at 2+ rows. Same bits.
const metal_shared2 = @embedFile("kernels/decode/shared2.metal");

pub const header_t: [:0]const u8 = header_s ++ metal_shared2;

/// The v2 heads' shared-threadgroup test and id (`moeShared2`): with SYOFF 0 the grid rows from RKC on are the shared
/// expert's (the routed rows first, as v1); SYOFF > 0 puts the SYS shared rows first (the routed body's rk = y - SYOFF,
/// `text_*_y`). The shared threadgroups are numbered z-major (z * SYS + row) * SX + x, the dispatch order of the grid's slabs.
const shared2_prologue = @embedFile("kernels/decode/shared2_prologue.metal");

/// Launch 1's v2 head, the gate|up GEMV with the input rotated in-kernel (form 9's gu8 launch, gu3 without `sprep`, and the
/// standalone beside variant): shared threadgroup s_id = (K slice, group of 4 SOTS output tiles of the 2 inter / 16 tiles,
/// gate's then up's); simdgroup sg owns the group's SOTS tiles from (4 group + sg) SOTS. It walks the slice part by part
/// (dgemv's four simdgroup shares), its weights SPRE input tiles ahead of the decode in whole parts (at most 3 parts, at
/// most 16 tiles of raw words in flight); the first ones are issued before the threadgroup rotates the slice's 128-block
/// (dense prep_s: one simdgroup per (projection, row) the threadgroup reads, the block sums written by each projection's
/// first threadgroup) and its barrier. The plane values are stored straight from the quads. Planes szg / szu f32 [SKSG,
/// SRS, inter] (layout SLM), block sums sxsg / sxsu [SRS, hidden / 128].
const shared_gu2_head = @embedFile("kernels/decode/shared_gu2_head_1.metal") ++
    shared2_prologue ++
    @embedFile("kernels/decode/shared_gu2_head_2.metal");

/// Launch 1's v2 head with the input prepared once by the routed prepare launch (`shared_prep_head`, form 9 from two rows,
/// `sprep`): `shared_gu2_head` reading the lane-order rows sxg / sxu half [SRS, hidden] from device memory, so no rotation,
/// no threadgroup memory and no barrier. Planes szg / szu.
const shared_gu2p_head = shared2_prologue ++ @embedFile("kernels/decode/shared_gu2p_head.metal");

/// Launch 0's head for `shared_gu2p_head` (prep2 from two rows; standalone beside): grid rows from RKC on are the shared
/// expert's rows, one simdgroup per (128-block of hidden, row) running dense prep_s for gate and up: sxg / sxu half [SRS,
/// hidden] lane order, the block sums sxsg / sxsu [SRS, hidden / 128].
const shared_prep_head = @embedFile("kernels/decode/shared_prep_head.metal");

/// Launch 3's v2 head: `shared_fin_head` with the planes read in layout SLM (`turboexl3_sfinish2`).
const shared_fin2_head = @embedFile("kernels/decode/shared_fin2_head.metal");

/// Launch 4's v2 head: shared threadgroup s_id = (K slice of SKSD, group of 4 SOTS of the hidden / 16 output tiles), simdgroup
/// sg owning SOTS tiles from (4 group + sg) SOTS over the slice's four parts of sxpd (lane order, device memory): no
/// threadgroup memory, no barrier. Planes szd f32 [SKSD, SRS, hidden] (layout SLM).
const shared_down2_head = shared2_prologue ++ @embedFile("kernels/decode/shared_down2_head.metal");

/// Launch 5 with the v2 planes: `text_dfin7s` with the shared down's finish reading layout SLM.
pub const text_dfin7t = @embedFile("kernels/decode/dfin7t.metal");

/// The routed bodies under the v2 heads: rk = y - SYOFF (SYOFF 0: v1's rows; > 0: the shared rows first).
const rk_line = @embedFile("kernels/decode/rk_line.metal");
const rk_line_y = @embedFile("kernels/decode/rk_line_y.metal");
const text_gu8_y: [:0]const u8 = @embedFile("kernels/decode/gu8_1.metal") ++ rk_line_y ++ @embedFile("kernels/decode/gu8_2.metal");
const text_gu3_y: [:0]const u8 = gu3_1 ++ rk_line_y ++ gu3_body;
const text_down3_y: [:0]const u8 = @embedFile("kernels/decode/down3_1.metal") ++ rk_line_y ++ @embedFile("kernels/decode/down3_2.metal");
pub const text_gu8t: [:0]const u8 = shared_gu2_head ++ text_gu8_y;
pub const text_gu3t: [:0]const u8 = shared_gu2_head ++ text_gu3_y;
pub const text_gu3p: [:0]const u8 = shared_gu2p_head ++ text_gu3_y;
pub const text_prep2t: [:0]const u8 = shared_prep_head ++ text_prep2;
pub const text_fin7t: [:0]const u8 = shared_fin2_head ++ text_fin7;
pub const text_down3t: [:0]const u8 = shared_down2_head ++ text_down3_y;
const shared_gu2p_inputs = [_][*:0]const u8{ "sxg", "sxu", "sh_gate_t", "sh_up_t" };
const shared_gu2p_outputs = [_][*:0]const u8{ "szg", "szu" };
const shared_prep_inputs = [_][*:0]const u8{ "x", "sh_gate_suh", "sh_up_suh" };
const shared_prep_outputs = [_][*:0]const u8{ "sxg", "sxu", "sxsg", "sxsu" };
const gu3p_inputs = gu3_inputs ++ shared_gu2p_inputs;
const gu3p_outputs = gu3_outputs ++ shared_gu2p_outputs;
const prep2t_inputs = prep2_inputs ++ [_][*:0]const u8{ "sh_gate_suh", "sh_up_suh" };
const prep2t_outputs = prep2_outputs ++ shared_prep_outputs;

// ── The host's softmax router folded into form 9's routed prepare launch ──

/// The host's fused softmax router (transformer.zig `moeRouterSource(.softmax)`, the MLX custom kernel
/// "mlxserve_moe_router_softmax"), byte for byte: the GPU test runs it as the chain's router, the merged launch below takes
/// its statements from it (its pieces), and the host's test pins this copy to its own source
/// (`exl3_text_router_softmax`). Inputs logits; outputs inds, scores; template TOUT, NE, NK, TG, NORM (the host's
/// config: TG = 256 at 512 experts).
pub const text_router_softmax: [:0]const u8 = @embedFile("kernels/decode/router_softmax_1.metal") ++ router_softmax ++ router_sel ++
    @embedFile("kernels/decode/router_softmax_4.metal") ++ router_weights ++ @embedFile("kernels/decode/router_softmax_6.metal");
pub const router_inputs = [_][*:0]const u8{"logits"};
pub const router_outputs = [_][*:0]const u8{ "inds", "scores" };

/// `parts` joined and 0-terminated (a kernel source).
fn textJoin(comptime parts: []const []const u8) [:0]const u8 {
    comptime {
        var out: []const u8 = "";
        for (parts) |p| out = out ++ p;
        var buf: [out.len:0]u8 = undefined;
        @memcpy(buf[0..out.len], out);
        const final = buf;
        return &final;
    }
}

/// The router's selection (simdgroup 0's K rounds of max-then-mask), its softmax tree and its weights (renormalized, then
/// written with the slots), and prep2's body after its index lines: verbatim slices of their kernels' texts.
const router_sel = @embedFile("kernels/decode/router_softmax_3.metal");
const router_softmax = @embedFile("kernels/decode/router_softmax_2.metal");
const router_weights = @embedFile("kernels/decode/router_softmax_5.metal");
const prep2_body = @embedFile("kernels/decode/prep2_2.metal");

/// The shared gate's logit for one row exactly as MLX computes it at the verify widths, as one simdgroup's part
/// (`gx` x [R, GK] bf16, `gw` the gate's weight vector [GK] bf16, `gp` the 8 partials): MLX 0.32's `matmul` of x and the gate's
/// [GK, 1] at R = 2..4 (GK > 64, GK >= 16 R) runs gemv_bfloat16_bm1_bn8_sm1_sn32_tm{1,4}_tn4 (gemv.h GEMVKernel::run), whose
/// simdgroup gs of 8 and lane sum the columns (32 gs + lane) * 4 + 1024 i + tn in that order, the leftover block zero-filled
/// past GK (load_safe), then fold a shuffle-down tree; this simdgroup plays the gemv's simdgroup gs = tid / 32 - 1. The host's
/// router mode `softmax_gate_gemv` splices the same text (its test pins it to this one).
pub const text_gate_gemv = sources.gate_gemv;
/// The gate's sum of the 8 partials as the gemv's simdgroup 0 lane 0 folds them (its own first, then 1..7 in order), and its
/// bf16 store.
pub const text_gate_sum = sources.gate_sum;

/// The router kernel and form 3's prepare (`text_prep2`) as one launch, one threadgroup per (128-wide hidden block b, row)
/// of NSG = TG / 32 simdgroups (`Kernels3.routeSimdgroups`). The keys are staged as the router stages them; then phase
/// 1: simdgroup 0 runs the router's selection on them (its K rounds, verbatim; every block of the row repeats it, so no
/// block waits on another), and in block 0 simdgroup NSG - 1 runs the router's softmax tree (verbatim) beside it,
/// handing smax / snorm on through threadgroup memory, and with the gate simdgroups 1..8 take the shared gate's dot.
/// Phase 2: simdgroup j < K prepares row-slot (row, j) of block b with prep2's statements, its expert the selection's
/// j-th pick; in block 0 simdgroup NSG - 1 runs the router's weights and writes the slots and scores (verbatim), and
/// with the gate sums the gate's partials. Every value a statement of the router or of prep2 computes, on the same
/// lanes: their bytes.
fn routePrepText(comptime gate: bool) [:0]const u8 {
    return textJoin(&.{
        @embedFile("kernels/decode/route_prep_1.metal"),
        if (gate) @embedFile("kernels/decode/route_prep_gp.metal") else "",
        @embedFile("kernels/decode/route_prep_2.metal"),
        router_sel,
        @embedFile("kernels/decode/route_prep_3.metal"),
        router_softmax,
        @embedFile("kernels/decode/route_prep_4.metal"),
        if (gate) @embedFile("kernels/decode/route_prep_close_gate.metal") ++ text_gate_gemv else @embedFile("kernels/decode/route_prep_close.metal"),
        @embedFile("kernels/decode/route_prep_5.metal"),
        prep2_body,
        @embedFile("kernels/decode/route_prep_6.metal"),
        router_weights,
        if (gate) @embedFile("kernels/decode/route_prep_gsum_open.metal") ++ text_gate_sum ++ @embedFile("kernels/decode/route_prep_gsum_close.metal") else "",
        @embedFile("kernels/decode/route_prep_7.metal"),
    });
}
pub const text_route_prep: [:0]const u8 = routePrepText(false);
pub const text_route_prepg: [:0]const u8 = routePrepText(true);
const route_prep_inputs = [_][*:0]const u8{ "logits", "x", "gate_suh", "up_suh" };
const route_prep_outputs = [_][*:0]const u8{ "xg", "xu", "sg", "su", "inds", "scores" };
const route_prepg_inputs = route_prep_inputs ++ [_][*:0]const u8{ "gx", "gw" };
const route_prepg_outputs = route_prep_outputs ++ [_][*:0]const u8{"glogit"};

/// Form 5: three launches (prepare + gate|up GEMV, mid + down GEMV + per-slot finish, the slot reduce) where form 3 has five,
/// in form 3's K partitions and rounding points, so a row's bytes are form 3's; the row-slots that chose one expert (up to
/// `gmax`, members from other rows) share one threadgroup, so that expert's weights are read and decoded once per group.
pub const Kernels5 = struct {
    geo: Geometry,
    ks_gu: u32,
    ks_dn: u32,
    /// output tiles per gate|up threadgroup (1, 2, 4, 8): the inline prepare is shared by bt tiles
    bt: u32,
    /// row-slots per expert group (1..4); a call of r rows groups up to min(gmax, r)
    gmax: u32,
    /// the integer-rate reader (n % 16 == 0; the same codewords as the general reader)
    intr: bool,
    /// y's dtype (the x the host passes: bf16 served; f32 in the tests and the bench); set before the first call
    out_dtype: mlx.mlx_dtype = .float32,
    /// MUL1's deferred affine map (false decodes every weight to its f16 value)
    defer_affine: bool = false,
    gu: mlx.mlx_fast_metal_kernel,
    dn: mlx.mlx_fast_metal_kernel,
    red: mlx.mlx_fast_metal_kernel,
    /// form 6's down (the slot reduce fused by arrival counting): two launches
    dn6: mlx.mlx_fast_metal_kernel,
    cfg: [Kernels.max_rows + 1]?Cfg = @splat(null),

    const Cfg = struct { gu: mlx.mlx_fast_metal_kernel_config, dn: mlx.mlx_fast_metal_kernel_config, red: mlx.mlx_fast_metal_kernel_config, dn6: mlx.mlx_fast_metal_kernel_config };

    /// The K partitions are form 3's (`Kernels3.initWith` at the same ks_max): gate|up hidden / 16 tiles in KS x 2 parts,
    /// down inter / 16 tiles in KS2 x 4 parts of at least 2 tiles.
    pub fn init(geo: Geometry, bt: u32, ks_max: u32, gmax: u32) !Kernels5 {
        if (!geo.admitted() or (bt != 1 and bt != 2 and bt != 4 and bt != 8) or (ks_max != 1 and ks_max != 2 and ks_max != 4 and ks_max != 8) or gmax < 1 or gmax > 4) return error.BadExl3Shape;
        if ((geo.inter / 16) % bt != 0) return error.BadExl3Shape;
        if (geo.limit != 0) return error.Exl3EngineNotImplemented; // the clamped SwiGLU is Kernels3's (dn5 has no clamp)
        var ks_gu: u32 = ks_max;
        while (ks_gu > 1 and (geo.hidden / 16) % (ks_gu * 2) != 0) ks_gu /= 2;
        var ks_dn: u32 = ks_max;
        while (ks_dn > 1 and ((geo.inter / 16) % (ks_dn * 4) != 0 or (geo.inter / 16) / (ks_dn * 4) < 2)) ks_dn /= 2;
        return .{
            .geo = geo,
            .ks_gu = ks_gu,
            .ks_dn = ks_dn,
            .bt = bt,
            .gmax = gmax,
            .intr = geo.rate.n % 16 == 0,
            .defer_affine = geo.dec.codebook == .mul1,
            .gu = ks.newKernel("turboexl3_gu5", &gu5_inputs, &gu5_outputs, text_gu5, header5, true) orelse return error.KernelCreateFailed,
            .dn = ks.newKernel("turboexl3_dn5", &dn5_inputs, &dn5_outputs, text_dn5, header5, true) orelse return error.KernelCreateFailed,
            .red = ks.newKernel("turboexl3_red5", &red5_inputs, &red5_outputs, text_red5, header5, true) orelse return error.KernelCreateFailed,
            .dn6 = ks.newKernel("turboexl3_dn6", &dn6_inputs, &dn6_outputs, text_dn6, header5, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *Kernels5) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.gu, cc.dn, cc.red, cc.dn6 }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        inline for (.{ self.gu, self.dn, self.red, self.dn6 }) |k| freeKernel(k);
    }

    fn cfgs(self: *Kernels5, rows: usize) !Cfg {
        if (rows == 0 or rows > Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const g = self.geo;
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nbh: c_int = @intCast(g.hidden / 128);
        const gm: u32 = @min(self.gmax, @as(u32, @intCast(rows)));
        const nhw = ks.TemplateArg{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } };
        const cb = ks.TemplateArg{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = ks.TemplateArg{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = ks.TemplateArg{ .name = "HID", .value = .{ .int = H } };
        const inter_t = ks.TemplateArg{ .name = "INTER", .value = .{ .int = I } };
        const rkt = ks.TemplateArg{ .name = "RKC", .value = .{ .int = rkc } };
        const topk = ks.TemplateArg{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } };
        const ks_t = ks.TemplateArg{ .name = "KS", .value = .{ .int = @intCast(self.ks_gu) } };
        const ks2_t = ks.TemplateArg{ .name = "KS2", .value = .{ .int = @intCast(self.ks_dn) } };
        const bt_t = ks.TemplateArg{ .name = "BT", .value = .{ .int = @intCast(self.bt) } };
        const gm_t = ks.TemplateArg{ .name = "GMAX", .value = .{ .int = @intCast(gm) } };
        const def_t = ks.TemplateArg{ .name = "DEFER", .value = .{ .int = if (self.defer_affine) 1 else 0 } };
        const ir_t = ks.TemplateArg{ .name = "IR", .value = .{ .int = if (self.intr) 1 else 0 } };
        const out_t = ks.TemplateArg{ .name = "OUT_T", .value = .{ .dtype = self.out_dtype } };
        const gu = try ks.buildConfig(.{
            .grid = .{ (g.inter / (16 * self.bt)) * 128, rk, self.ks_gu },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ rkc, nbh }, &[_]c_int{ rkc, nbh }, &[_]c_int{ @as(c_int, @intCast(rows)) * nbh } },
            .out_dtypes = &.{ .float32, .float32, .float32, .float32, .uint32 },
            .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_t, bt_t, gm_t, topk, def_t, ir_t },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gu);
        const dn = try ks.buildConfig(.{
            .grid = .{ (g.hidden / 128) * 256, rk, 1 },
            .threadgroup = .{ 256, 1, 1 },
            .out_shapes = &.{&[_]c_int{ rkc, H }},
            .out_dtypes = &.{.float16},
            .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_t, ks2_t, gm_t, def_t, ir_t },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(dn);
        const red = try ks.buildConfig(.{
            .grid = .{ g.hidden, @intCast(rows), 1 },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{&[_]c_int{ @intCast(rows), H }},
            .out_dtypes = &.{self.out_dtype},
            .template = &.{ hid, topk, out_t },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(red);
        const dn6 = try ks.buildConfig(.{
            .grid = .{ (g.hidden / 128) * 256, rk, 1 },
            .threadgroup = .{ 256, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ rkc, H }, &[_]c_int{ @intCast(rows), H } },
            .out_dtypes = &.{ .float16, self.out_dtype },
            .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_t, ks2_t, gm_t, def_t, ir_t, topk, out_t },
        });
        const c: Cfg = .{ .gu = gu, .dn = dn, .red = red, .dn6 = dn6 };
        self.cfg[rows] = c;
        return c;
    }

    /// The three launches (`fused`: form 6, the two with the slot reduce in the down launch); the caller frees y. x [rows,
    /// hidden], slots and scores [rows * topk], as `Kernels3.moe`.
    pub fn moe(self: *Kernels5, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, fused: bool) !mlx.mlx_array {
        const c = try self.cfgs(rows);
        var z: [5]mlx.mlx_array = undefined; // zg, zu, the inputs' per-block sums, form 6's zeroed counters
        try ks.applyConfig(self.gu, c.gu, &.{ x, slots, bank.gate_suh, bank.up_suh, bank.gate_t, bank.up_t }, &z, s);
        defer for (z) |a| {
            _ = mlx.mlx_array_free(a);
        };
        if (fused) {
            var o6: [2]mlx.mlx_array = undefined; // wv (the exchange), y
            try ks.applyConfig(self.dn6, c.dn6, &.{ z[0], z[1], z[2], z[3], slots, bank.gate_svh, bank.up_svh, bank.down_suh, bank.down_t, bank.down_svh, z[4], scores }, &o6, s);
            _ = mlx.mlx_array_free(o6[0]);
            return o6[1];
        }
        var wv: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(self.dn, c.dn, &.{ z[0], z[1], z[2], z[3], slots, bank.gate_svh, bank.up_svh, bank.down_suh, bank.down_t, bank.down_svh }, &wv, s);
        defer _ = mlx.mlx_array_free(wv[0]);
        var y: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(self.red, c.red, &.{ wv[0], scores }, &y, s);
        return y[0];
    }
};

test "turboexl3 decode forms 7 and 8 texts: every template constant read is passed, every signature name is used" {
    const t = std.testing;
    for ([_][]const u8{ "INTER", "RKC", "KS", "DEFER", "NBH" }) |name| try t.expect(std.mem.indexOf(u8, text_fin7, name) != null);
    for ([_][]const u8{ "HID", "TOPK", "RKC", "KS", "OUT_T", "DEFER", "NBI" }) |name| try t.expect(std.mem.indexOf(u8, text_dfin7, name) != null);
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "RKC", "KS", "SPT", "BT", "GROUP", "DEFER", "TOPK" }) |name| try t.expect(std.mem.indexOf(u8, text_gu8, name) != null);
    for (fin3_inputs ++ fin3_outputs) |n| try t.expect(std.mem.indexOf(u8, text_fin7, std.mem.span(n)) != null);
    for (dfin3_inputs ++ down_outputs) |n| try t.expect(std.mem.indexOf(u8, text_dfin7, std.mem.span(n)) != null);
    for (gu8_inputs ++ gu8_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu8, std.mem.span(n)) != null);
}

/// The lines of `text` (no empty ones), for the fold test's line-for-line checks
fn nonEmptyLines(a: std.mem.Allocator, text: []const u8) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(a);
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |l| if (l.len > 0) try out.append(a, l);
    return out.toOwnedSlice(a);
}

test "turboexl3 decode fused fin fold texts: gu3sf is gu3s but its plane and sums stores, then fin7's and the shared fin head's statements" {
    const t = std.testing;
    const a = t.allocator;
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "RKC", "KS", "SPT", "BT", "GROUP", "DEFER", "SNG", "SKSG", "SRS", "SDEF", "SZ", "SSPT", "OUT_T" }) |name| try t.expect(std.mem.indexOf(u8, text_gu3sf, name) != null);
    for (gu3sf_inputs ++ gu3sf_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu3sf, std.mem.span(n)) != null);
    try t.expect(std.mem.indexOf(u8, text_vfold_zero, "NCNT") != null);
    // gu3s and fin7s are their pieces; every piece gu3sf keeps is in it line for line, every piece it replaces is gone and
    // its replacement there once
    try t.expect(std.mem.eql(u8, text_gu3s, sgh_1 ++ sgh_2 ++ sgh_3 ++ sgh_z ++ sgh_4 ++ sgh_sums ++ sgh_5 ++ sgh_end ++ text_gu3));
    try t.expect(std.mem.eql(u8, text_fin7s, sfh_1 ++ sfh_2 ++ sfh_xg ++ sfh_xu ++ sfh_3 ++ sfh_g ++ sfh_u ++ sfh_4 ++ sfh_end ++ text_fin7));
    const kept = [_][]const u8{ sgh_1, sgh_2, sgh_3, sgh_4, sgh_5, sgh_end, sfh_2, sfh_3, sfh_4, gu3_1, rk_line, gu3_2, gu3_3, gu3_4, fin7_2, fin7_3, fin7_sums, fin7_4 };
    for (kept) |p| {
        const ls = try nonEmptyLines(a, p);
        defer a.free(ls);
        for (ls) |l| try t.expect(std.mem.indexOf(u8, text_gu3sf, l) != null);
    }
    for ([_][]const u8{ sgh_z, sgh_sums, sfh_xg, sfh_xu, sfh_g, sfh_u, gu3_stores, fin7_planes }) |p| try t.expect(std.mem.indexOf(u8, text_gu3sf, p) == null);
    for ([_][]const u8{ vf_head_zc, vf_head_sumsc, vf_sfin_xgc, vf_sfin_xuc, vf_sfin_g, vf_sfin_u, vf_gu3_stores, vf_fin7_planes }) |p| try t.expectEqual(@as(usize, 1), std.mem.count(u8, text_gu3sf, p));
    // the inline finishes: turboexl3_sfinish's 12 statements each, its parameters bound by name, in a block of their own
    for ([_][]const u8{ vf_sfin_g, vf_sfin_u }) |p| {
        const ls = try nonEmptyLines(a, p);
        defer a.free(ls);
        try t.expectEqual(@as(usize, 14), ls.len);
    }
}

test "turboexl3 decode shared-expert texts: every template constant read is passed, every signature name is used" {
    const t = std.testing;
    for ([_][]const u8{ "SRS", "BT", "RKC", "WBITS", "HID", "SKSG", "SZ", "INTER", "SNG", "CODEBOOK", "SDEF", "SSPT" }) |name| {
        try t.expect(std.mem.indexOf(u8, shared_gu_head, name) != null);
    }
    for ([_][]const u8{ "RKC", "INTER", "NBH", "SDEF", "SKSG", "SRS", "OUT_T" }) |name| try t.expect(std.mem.indexOf(u8, shared_fin_head, name) != null);
    for ([_][]const u8{ "SRS", "BT", "RKC", "WBITS", "SKSD", "INTER", "SZ", "SND", "CODEBOOK", "HID", "SDEF", "SSPTD" }) |name| try t.expect(std.mem.indexOf(u8, shared_down_head, name) != null);
    for ([_][]const u8{ "HID", "TOPK", "KS", "DEFER", "NBI", "RKC", "OUT_T", "SKSD", "SRS", "SDEF" }) |name| try t.expect(std.mem.indexOf(u8, text_dfin7s, name) != null);
    for (gu3s_inputs ++ gu3s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu3s, std.mem.span(n)) != null);
    for (gu8s_inputs ++ gu8s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu8s, std.mem.span(n)) != null);
    for (fin7s_inputs ++ fin7s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_fin7s, std.mem.span(n)) != null);
    for (down3s_inputs ++ down3s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_down3s, std.mem.span(n)) != null);
    for (dfin7s_inputs ++ down_outputs) |n| try t.expect(std.mem.indexOf(u8, text_dfin7s, std.mem.span(n)) != null);
}

test "turboexl3 decode v2 shared-head texts: every template constant read is passed, every signature name is used" {
    const pin = comptime @import("test_support.zig").pins(@embedFile("kernels/test/pins/decode_shared2.metal"));
    const t = std.testing;
    const gu_names = [_][]const u8{ "SRS", "SYOFF", "RKC", "SYS", "SX", "WBITS", "SNG", "HID", "SKSG", "INTER", "SOTS", "SPRE", "CODEBOOK", "SDEF", "SLM" };
    for (gu_names ++ [_][]const u8{"SNP"}) |name| try t.expect(std.mem.indexOf(u8, shared_gu2_head, name) != null);
    for (gu_names) |name| try t.expect(std.mem.indexOf(u8, shared_gu2p_head, name) != null);
    for ([_][]const u8{ "RKC", "DIM" }) |name| try t.expect(std.mem.indexOf(u8, shared_prep_head, name) != null);
    for ([_][]const u8{ "RKC", "INTER", "NBH", "SDEF", "SKSG", "SRS", "OUT_T", "SLM" }) |name| try t.expect(std.mem.indexOf(u8, shared_fin2_head, name) != null);
    for ([_][]const u8{ "SRS", "SYOFF", "RKC", "SYS", "SX", "WBITS", "SND", "HID", "SKSD", "INTER", "SOTS", "SPRE", "CODEBOOK", "SDEF", "SLM" }) |name| try t.expect(std.mem.indexOf(u8, shared_down2_head, name) != null);
    for ([_][]const u8{ "HID", "TOPK", "KS", "DEFER", "NBI", "RKC", "OUT_T", "SKSD", "SRS", "SDEF", "SLM" }) |name| try t.expect(std.mem.indexOf(u8, text_dfin7t, name) != null);
    // the routed bodies under the v2 heads take rk = y - SYOFF, their one rk line spliced
    for ([_][]const u8{ text_gu8_y, text_gu3_y, text_down3_y }) |txt| {
        try t.expect(std.mem.indexOf(u8, txt, pin[0]) != null);
        try t.expect(std.mem.indexOf(u8, txt, pin[1]) == null);
    }
    for (gu8s_inputs ++ gu8s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu8t, std.mem.span(n)) != null);
    for (gu3s_inputs ++ gu3s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu3t, std.mem.span(n)) != null);
    for (gu3p_inputs ++ gu3p_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu3p, std.mem.span(n)) != null);
    for (prep2t_inputs ++ prep2t_outputs) |n| try t.expect(std.mem.indexOf(u8, text_prep2t, std.mem.span(n)) != null);
    for (fin7s_inputs ++ fin7s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_fin7t, std.mem.span(n)) != null);
    for (down3s_inputs ++ down3s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_down3t, std.mem.span(n)) != null);
    for (dfin7s_inputs ++ down_outputs) |n| try t.expect(std.mem.indexOf(u8, text_dfin7t, std.mem.span(n)) != null);
    for (shared_gu_inputs ++ shared_gu_outputs) |n| try t.expect(std.mem.indexOf(u8, shared_gu2_head, std.mem.span(n)) != null);
    for (shared_gu2p_inputs ++ shared_gu2p_outputs) |n| try t.expect(std.mem.indexOf(u8, shared_gu2p_head, std.mem.span(n)) != null);
    for (shared_prep_inputs ++ shared_prep_outputs) |n| try t.expect(std.mem.indexOf(u8, shared_prep_head, std.mem.span(n)) != null);
    for (sfin_inputs ++ sfin_outputs) |n| try t.expect(std.mem.indexOf(u8, shared_fin2_head, std.mem.span(n)) != null);
    for (sdown_inputs ++ sdown_outputs) |n| try t.expect(std.mem.indexOf(u8, shared_down2_head, std.mem.span(n)) != null);
    // the parts take every codeword from the window's 32-bit words (no 64-bit window anywhere in the v2 helpers)
    try t.expect(std.mem.indexOf(u8, metal_shared2, pin[2]) != null);
    try t.expect(std.mem.indexOf(u8, metal_shared2, pin[3]) == null);
    try t.expect(std.mem.indexOf(u8, metal_shared2, "Exl3BitsM<exl3_rmode<N>()> ww;") == null);
    // each launch's text (the v2 kernels, the routed down body under the rk splice, the standalone heads) uses every name of
    // its signature
    const Sig = struct { text: []const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8 };
    const sigs = [_]Sig{
        .{ .text = text_down3t, .inputs = &down3s_inputs, .outputs = &down3s_outputs },
        .{ .text = shared_down2_head, .inputs = &sdown_inputs, .outputs = &sdown_outputs },
        .{ .text = text_down3_y, .inputs = &down3_inputs, .outputs = &down3_outputs },
        .{ .text = text_down3s, .inputs = &down3s_inputs, .outputs = &down3s_outputs },
        .{ .text = shared_down_head, .inputs = &sdown_inputs, .outputs = &sdown_outputs },
        .{ .text = text_gu3p, .inputs = &gu3p_inputs, .outputs = &gu3p_outputs },
        .{ .text = text_gu3t, .inputs = &gu3s_inputs, .outputs = &gu3s_outputs },
        .{ .text = text_gu8t, .inputs = &gu8s_inputs, .outputs = &gu8s_outputs },
        .{ .text = shared_gu2_head, .inputs = &shared_gu_inputs, .outputs = &shared_gu_outputs },
        .{ .text = shared_gu2p_head, .inputs = &shared_gu2p_inputs, .outputs = &shared_gu2p_outputs },
        .{ .text = text_dfin7t, .inputs = &dfin7s_inputs, .outputs = &down_outputs },
    };
    for (sigs) |p| {
        for (p.inputs) |n| try t.expect(std.mem.indexOf(u8, p.text, std.mem.span(n)) != null);
        for (p.outputs) |n| try t.expect(std.mem.indexOf(u8, p.text, std.mem.span(n)) != null);
    }
    // tiles per simdgroup: whole groups of four simdgroups within a projection, else 1; within 64 accumulators per lane
    try t.expectEqual(@as(u32, 2), Kernels3.sharedOts(40, 2, 2, 1));
    try t.expectEqual(@as(u32, 4), Kernels3.sharedOts(40, 2, 4, 1));
    try t.expectEqual(@as(u32, 1), Kernels3.sharedOts(40, 2, 8, 1));
    try t.expectEqual(@as(u32, 8), Kernels3.sharedOts(160, 1, 8, 1));
    try t.expectEqual(@as(u32, 1), Kernels3.sharedOts(160, 1, 3, 1));
    try t.expectEqual(@as(u32, 10), Kernels3.sharedOts(40, 2, 10, 1));
    try t.expectEqual(@as(u32, 5), Kernels3.sharedOts(40, 2, 10, 2));
    try t.expectEqual(@as(u32, 5), Kernels3.sharedOts(40, 2, 5, 3));
    try t.expectEqual(@as(u32, 4), Kernels3.sharedOts(40, 2, 10, 4));
    try t.expectEqual(@as(u32, 4), Kernels3.sharedOts(160, 1, 8, 4));
    try t.expectEqual(@as(u32, 2), Kernels3.sharedOts(160, 1, 8, 8));
}

test "turboexl3 decode form 5 texts: every template constant read is passed, every signature name is used" {
    const t = std.testing;
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "RKC", "KS", "BT", "GMAX", "TOPK", "DEFER", "IR" }) |name| try t.expect(std.mem.indexOf(u8, text_gu5, name) != null);
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "RKC", "KS2", "GMAX", "DEFER", "IR" }) |name| try t.expect(std.mem.indexOf(u8, text_dn5, name) != null);
    for (gu5_inputs ++ gu5_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu5, std.mem.span(n)) != null);
    for (dn5_inputs ++ dn5_outputs) |n| try t.expect(std.mem.indexOf(u8, text_dn5, std.mem.span(n)) != null);
    for (red5_inputs ++ red5_outputs) |n| try t.expect(std.mem.indexOf(u8, text_red5, std.mem.span(n)) != null);
    for (dn6_inputs ++ dn6_outputs) |n| try t.expect(std.mem.indexOf(u8, text_dn6, std.mem.span(n)) != null);
}

test "turboexl3 decode router-folded prepare texts: the merged launch carries the router's and prep2's statements verbatim, every name is used" {
    const pin = comptime @import("test_support.zig").pins(@embedFile("kernels/test/pins/decode_route_prep.metal"));
    const t = std.testing;
    const texts = [_][]const u8{ text_route_prep, text_route_prepg };
    // the pieces are pieces of the router copy and of prep2, and each lands once in each merged text
    for ([_][]const u8{ router_sel, router_softmax, router_weights }) |piece| {
        try t.expect(std.mem.indexOf(u8, text_router_softmax, piece) != null);
        for (texts) |txt| try t.expectEqual(@as(usize, 1), std.mem.count(u8, txt, piece));
    }
    try t.expect(std.mem.indexOf(u8, text_prep2, prep2_body) != null);
    for (texts) |txt| try t.expectEqual(@as(usize, 1), std.mem.count(u8, txt, prep2_body));
    // the value-setting lines of the router's rounds, softmax and weights, and of prep2's rotation
    const lines = [_][]const u8{
        pin[0],
        pin[1],
        pin[2],
        pin[3],
        pin[4],
        pin[5],
        pin[6],
        pin[7],
        pin[8],
        pin[9],
        pin[10],
        pin[11],
        pin[12],
        pin[13],
        pin[14],
    };
    for (lines) |line| {
        try t.expect(std.mem.indexOf(u8, text_router_softmax ++ text_prep2, line) != null);
        for (texts) |txt| try t.expect(std.mem.indexOf(u8, txt, line) != null);
    }
    // the hand-offs: smax / snorm through threadgroup memory, the expert from the selection, the row-slot of the simdgroup
    for (texts) |txt| {
        try t.expect(std.mem.indexOf(u8, txt, pin[15]) != null);
        try t.expect(std.mem.indexOf(u8, txt, @embedFile("kernels/test/pins/decode_route_prep_1.metal")) != null);
        try t.expect(std.mem.indexOf(u8, txt, pin[16]) != null);
        try t.expect(std.mem.indexOf(u8, txt, pin[17]) != null);
        try t.expectEqual(@as(usize, 2), std.mem.count(u8, txt, pin[18]));
    }
    // the shared gate's dot and sum only in the gate kernel
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, text_route_prepg, text_gate_gemv));
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, text_route_prepg, text_gate_sum));
    try t.expect(std.mem.indexOf(u8, text_route_prep, pin[19]) == null);
    // every input / output name and template constant is read or written
    for (route_prep_inputs ++ route_prep_outputs) |n| try t.expect(std.mem.indexOf(u8, text_route_prep, std.mem.span(n)) != null);
    for (route_prepg_inputs ++ route_prepg_outputs) |n| try t.expect(std.mem.indexOf(u8, text_route_prepg, std.mem.span(n)) != null);
    for ([_][]const u8{ "NE", "NK", "TG", "TOUT", "DIM", "TOPK" }) |name| for (texts) |txt| try t.expect(std.mem.indexOf(u8, txt, name) != null);
    try t.expect(std.mem.indexOf(u8, text_route_prepg, "GK") != null);
    for (router_inputs ++ router_outputs) |n| try t.expect(std.mem.indexOf(u8, text_router_softmax, std.mem.span(n)) != null);
    for ([_][]const u8{ "gx", "gw", "GK", pin[20], pin[21] }) |name| try t.expect(std.mem.indexOf(u8, text_gate_gemv, name) != null);
    // one simdgroup per slot plus the weights', with the gate's eight beside the selection
    try t.expectEqual(@as(u32, 11), Kernels3.routeSimdgroups(10, false));
    try t.expectEqual(@as(u32, 11), Kernels3.routeSimdgroups(10, true));
    try t.expectEqual(@as(u32, 5), Kernels3.routeSimdgroups(4, false));
    try t.expectEqual(@as(u32, 10), Kernels3.routeSimdgroups(4, true));
}
