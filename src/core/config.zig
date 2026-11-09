//! The choices a consumer makes once, before its first call (`configure`). The defaults are the served configuration;
//! every other launch shape is a constant of the file that reads it. A kernel cache keeps what it built, so a change
//! after the first launch reaches only the geometries built after it.
const std = @import("std");
const Global = @import("global.zig").Global;

pub const Config = struct {
    /// The routed decode chain's form: 3, 5, 6, 7, 8 or 9, each with form 3's bytes (the GPU tests hold forms 5 and 6, and
    /// 7, 8 and 9, to form 3 byte for byte). 5 and 6 share one decode among the row-slots that chose an expert (three and
    /// two launches); 7 keeps fin / dfin's H128s in registers, 8 also moves the prepare into the gate|up launch, and 9 runs
    /// 8 at one row and 7 from two.
    decode_form: u8 = 9,
    /// Gate|up K slices per GEMV threadgroup of the fused shared expert (of its 20): 1, 2, 4, 5, 10 or 20. Launch shape
    /// only.
    shared_spt: u32 = 4,
    /// The fused shared expert from two rows with fin7s folded into the gate|up launch (gu3sf: one dependent launch
    /// fewer); false keeps gu3s + fin7s. The same bytes.
    fused_fin_fold: bool = true,
    /// The prefill GEMMs' A operands in fragment order per 16 (2) or 32 (1) elements, or natural (0). The same bytes.
    pf_alay: u8 = 2,
    /// Each expert's full 128-row runs in 128-row prefill tiles. The same bytes.
    pf_t128: bool = true,
    /// The 128-row tiles' launch over tiles of up to 256 rows on the bm-256 body (`prefill.Options.t256`; with
    /// `pf_t128`): each expert's full 256-row runs and a remainder of more than `pf_cut` rows, its weights decoded once a
    /// tile. The same bytes.
    pf_t256: bool = false,
    /// The big tiles' remainder rule (`prefill.Options.cut`): past an expert's full 128-row (or 256-row) tiles, a remainder
    /// of more than min(pf_cut, 127) rows takes one part-full big tile, the rest the mixed launch. A launch shape: the same
    /// bytes.
    pf_cut: u8 = 64,
    /// 64-wide output blocks per gate|up GEMM threadgroup (`prefill.Options.gu_ncb`): 1, 2 or 4. The same bytes.
    pf_gu_ncb: u8 = 1,
    /// The routed prefill's finish as the down GEMM's epilogue (`prefill.Options.dfin`): every top-1 routed prefill call
    /// (an expert streamer's prompt calls, `moeSwigluRows`, or `moe` at top-1), the down launch writing y. The same bytes.
    pf_dfin: bool = false,
    /// The fused finish's tile rows per staging round (`prefill.Options.dfr`): 32 (staged in the weight stage's freed
    /// buffer, no threadgroup memory of its own) or 64 (16 KB of its own). The same bytes.
    pf_dfr: u16 = 32,
    /// With `pf_dfin`, the fused finish only for an expert of at least this many rows in the call (`prefill.Options.dfin_min`);
    /// the others keep the served path; 0 every expert. Needs `pf_t128`. The same bytes.
    pf_dfin_min: u16 = 0,
    /// The routed prefill's M tile rows (`prefill.Options.bm`): 64 (served), 128 or 256; anything but 64 runs without the
    /// 128-row tiles (`pf_t128` needs bm 64). The same bytes.
    pf_bm: u16 = 64,
    /// The register path for each expert's <= 32-row remainder in the GEMMs' launch (small `.mixed`, served); false puts
    /// every tile on the shared-decode body (small `.off`). The same bytes.
    pf_mixed: bool = true,
    /// Routed prefill rows on the fused GEMM chain; false refuses them (`error.Exl3EngineNotImplemented`).
    prefill: bool = true,
    /// The device passes MLX's NAX gate (an M5-class GPU and macOS 26.2 or later), the consumer's to say. False keeps
    /// every kernel built on MLX's NAX tiles off: routed prefill rows are refused (`error.Exl3EngineNotImplemented`), and
    /// a dense linear's prefill rows dequantize + run MLX's matmul instead of the register-path GEMM. Below the M5 those
    /// GEMMs compute wrong.
    nax: bool = true,
    /// Dense decode linears in one launch (DEC-ONE); false the three-launch chain. The same bytes.
    dec_one: bool = true,
    /// A one-row dense decode call on its trunk shape's entry of `dense.one_row_table` (Ling-3.1-flash's shapes on the M5
    /// Max: output tiles per GEMV threadgroup, and DEC-ONE's prepare inside the launch or in its own); a shape the table
    /// does not name keeps the rest of this configuration's. Launch shape only: the same bytes.
    dec_one_row_table: bool = false,
    /// DEC-ONE's GEMV over prepared rows on a variant (`dense.dec_variants`, `dec_variants` of them; 0 the served GEMV):
    /// the dense dec0 launch (a decode call whose prepare runs in its own launch, and `Dense.decLinearPrepared`) and
    /// Ling-3.1-flash's KDA projections' launch with b_proj (`ling.kdaProjB`). 1 t4 (the MUL1 codeword's t as one half2,
    /// banked's group variant t4), 2 pf (each input tile's words loaded a pass ahead of their decode), 3 both. Another
    /// schedule of the same reads, decodes and sums: the same bytes.
    dec_variant: u8 = 0,
    /// The row chunks of the dense decode GEMV chain: at most 8 rows, a chunk of 6 to 8 rows on the lean GEMV text (its
    /// codeword loop by input element); false keeps chunks of at most 5 rows, all on the plain GEMV text. The same bytes.
    dense_lean: bool = true,
    /// Chunks of 5 rows in a GEMV launch of `dense.Dense.lean_bt2_threadgroups` or more at BT 1 on the lean text at BT 2
    /// (`dense.Dense.wideLean`: DeepSeek-V4.1's LM head, wq_b and wo_a at 5, 9 and 10 rows); false keeps them on
    /// text_dgemv at BT 1. The same bytes.
    dense_lean5: bool = true,
    /// The banked decode's row-slots of one packed id per GEMV lead threadgroup (`moeClampedBanked`): 2 pairs them, 4 runs
    /// the grouped texts, which take up to four and skip a row-slot whose id is `skip_id` (no decode, its y row 0). The
    /// same bytes for every other row-slot.
    banked_group: u32 = 2,
    /// hw4b: the grouped texts with an M = 3 member path (`decode.text_gu3bgm3`, `text_down3bgm3`) for a lead with three
    /// members (today's run the M = 4 path, a fourth member's FMAs padding). The same bytes. Needs `banked_group` 4.
    banked_m3: bool = false,
    /// hw2: the grouped banked gate|up and down GEMVs' variants (`decode.group_variants`, `banked_variants` of them; 0
    /// today's): other schedules of the same reads, decodes and sums, so the same bytes, on the grouped texts the call
    /// runs (today's, or hw4b's with `banked_m3`). Any but 0 needs `banked_group` 4.
    banked_variant_gu: u8 = 0,
    banked_variant_dn: u8 = 0,
    /// Ling-3.1-flash's MLA attention in the latent (`ling.mlaLatentDecode`): launch 2 of a call of 2 rows or more on one
    /// threadgroup for this many of its rows (2 or 4; `mla_latent.Attn.rows2`, `rows4`), each key group's latents and rope
    /// keys read once for them; 0 keeps a threadgroup a row. The same bytes.
    mla_latent_rows: u8 = 0,
    /// The same launch with each sub-chunk's latents and rope keys staged in f32 in threadgroup memory
    /// (`mla_latent.Attn.stage`), at the calls `mla_latent_rows` does not take. The same bytes.
    mla_latent_stage: bool = false,
    /// The same launch with 2 x 2 score tiles and 4 x 4 output tiles a simdgroup (`mla_latent.Attn.block`), at the calls
    /// `mla_latent_rows` does not take; not with `mla_latent_stage`. The same bytes.
    mla_latent_block: bool = false,
    /// A diagnostic: each routed decode call's routing read back (a sync per call), its expert multiplicities printed
    /// every 480 calls.
    slot_stats: bool = false,
    /// A diagnostic: the first 512 routed prefill calls' routing written under this directory (a sync per call).
    pf_capture_dir: ?[]const u8 = null,

    /// Whether every field holds a value the library serves.
    pub fn valid(c: Config) bool {
        const forms = [_]u8{ 3, 5, 6, 7, 8, 9 };
        const spts = [_]u32{ 1, 2, 4, 5, 10, 20 };
        return std.mem.indexOfScalar(u8, &forms, c.decode_form) != null and
            std.mem.indexOfScalar(u32, &spts, c.shared_spt) != null and c.pf_alay <= 2 and
            (c.pf_gu_ncb == 1 or c.pf_gu_ncb == 2 or c.pf_gu_ncb == 4) and (c.pf_dfr == 32 or c.pf_dfr == 64) and
            (c.pf_bm == 64 or c.pf_bm == 128 or c.pf_bm == 256) and
            (c.banked_group == 2 or c.banked_group == 4) and (!c.banked_m3 or c.banked_group == 4) and
            c.banked_variant_gu < banked_variants and c.banked_variant_dn < banked_variants and c.dec_variant < dec_variants and
            ((c.banked_variant_gu == 0 and c.banked_variant_dn == 0) or c.banked_group == 4) and
            (c.mla_latent_rows == 0 or c.mla_latent_rows == 2 or c.mla_latent_rows == 4) and
            !(c.mla_latent_stage and c.mla_latent_block);
    }
};

/// The grouped banked GEMVs' variant count (`decode.group_variants.len`, checked there).
pub const banked_variants: u8 = 12;

/// DEC-ONE's GEMV variants' count (`dense.dec_variants.len`, checked there).
pub const dec_variants: u8 = 4;

const current = Global(Config, "config_current", .{});

/// Sets the process's configuration; refused (nothing changed) when a field holds a value the library does not serve.
pub fn configure(c: Config) error{InvalidTurboexl3Config}!void {
    if (!c.valid()) return error.InvalidTurboexl3Config;
    current.get().* = c;
}

/// The configuration in force.
pub fn get() *const Config {
    return &current.get().*;
}

test "turboexl3 config: the defaults are valid and served; an unserved form, slice count or layout is refused" {
    const t = std.testing;
    const d: Config = .{};
    try t.expect(d.valid());
    try t.expectEqual(@as(u8, 9), d.decode_form);
    try t.expect(d.prefill and d.nax and d.dec_one and d.pf_t128 and !d.slot_stats and d.pf_capture_dir == null);
    for ([_]u8{ 3, 5, 6, 7, 8, 9 }) |f| try t.expect((Config{ .decode_form = f }).valid());
    for ([_]u8{ 0, 4, 10 }) |f| try t.expect(!(Config{ .decode_form = f }).valid());
    try t.expect(!(Config{ .shared_spt = 3 }).valid());
    try t.expect(!(Config{ .pf_alay = 3 }).valid());
    // the routed prefill's launch shapes: off, or the served shape, by default; each a value the chain serves
    try t.expect(!d.pf_t256 and d.pf_cut == 64 and d.pf_gu_ncb == 1 and !d.pf_dfin and d.pf_dfr == 32 and d.pf_dfin_min == 0 and d.pf_bm == 64 and d.pf_mixed);
    for ([_]u8{ 2, 4 }) |n| try t.expect((Config{ .pf_gu_ncb = n }).valid());
    for ([_]u8{ 0, 3, 8 }) |n| try t.expect(!(Config{ .pf_gu_ncb = n }).valid());
    try t.expect((Config{ .pf_dfr = 64 }).valid() and !(Config{ .pf_dfr = 128 }).valid());
    for ([_]u16{ 128, 256 }) |bm| try t.expect((Config{ .pf_bm = bm }).valid());
    try t.expect(!(Config{ .pf_bm = 32 }).valid() and !(Config{ .pf_bm = 512 }).valid());
    try t.expect((Config{ .pf_t256 = true, .pf_dfin = true, .pf_cut = 0, .pf_mixed = false }).valid());
    try t.expectEqual(@as(u32, 2), d.banked_group);
    try t.expect((Config{ .banked_group = 4 }).valid());
    for ([_]u32{ 0, 1, 3, 8 }) |g| try t.expect(!(Config{ .banked_group = g }).valid());
    try t.expect(!d.banked_m3);
    try t.expect((Config{ .banked_group = 4, .banked_m3 = true }).valid());
    try t.expect(!(Config{ .banked_group = 2, .banked_m3 = true }).valid());
    try t.expect(d.banked_variant_gu == 0 and d.banked_variant_dn == 0);
    for (1..banked_variants) |v| {
        try t.expect((Config{ .banked_group = 4, .banked_variant_gu = @intCast(v) }).valid());
        try t.expect((Config{ .banked_group = 4, .banked_variant_dn = @intCast(v) }).valid());
        try t.expect((Config{ .banked_group = 4, .banked_m3 = true, .banked_variant_gu = @intCast(v), .banked_variant_dn = @intCast(v) }).valid());
        try t.expect(!(Config{ .banked_variant_gu = @intCast(v) }).valid());
        try t.expect(!(Config{ .banked_variant_dn = @intCast(v) }).valid());
    }
    try t.expect(!(Config{ .banked_group = 4, .banked_variant_gu = banked_variants }).valid());
    try t.expect(!(Config{ .banked_group = 4, .banked_variant_dn = banked_variants }).valid());
    try t.expect(d.dense_lean);
    try t.expect(d.fused_fin_fold);
    try t.expect(!d.dec_one_row_table);
    try t.expect((Config{ .dec_one_row_table = true }).valid());
    try t.expectEqual(@as(u8, 0), d.dec_variant);
    for (1..dec_variants) |v| try t.expect((Config{ .dec_variant = @intCast(v) }).valid());
    try t.expect(!(Config{ .dec_variant = dec_variants }).valid());
    try t.expect(d.mla_latent_rows == 0 and !d.mla_latent_stage and !d.mla_latent_block);
    for ([_]u8{ 2, 4 }) |r| try t.expect((Config{ .mla_latent_rows = r }).valid());
    for ([_]u8{ 1, 3, 8 }) |r| try t.expect(!(Config{ .mla_latent_rows = r }).valid());
    try t.expect((Config{ .mla_latent_rows = 4, .mla_latent_block = true }).valid());
    try t.expect(!(Config{ .mla_latent_stage = true, .mla_latent_block = true }).valid());
    const saved = current.get().*;
    defer current.get().* = saved;
    try t.expectError(error.InvalidTurboexl3Config, configure(.{ .decode_form = 4 }));
    try t.expectEqual(saved.decode_form, get().decode_form);
    try configure(.{ .decode_form = 3, .dec_one = false });
    try t.expectEqual(@as(u8, 3), get().decode_form);
    try t.expect(!get().dec_one);
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{current};
