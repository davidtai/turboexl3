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
    /// The routed prefill's down GEMM in 128-deep K stages (`prefill.Options.bk128`: two 64-deep sub-stages per buffer
    /// between barriers). The same bytes.
    pf_bk128: bool = false,
    /// A routed prefill simdgroup whose rows end within its first 16 multiplies only that fragment row
    /// (`prefill.Options.half_rows`). The same bytes.
    pf_half_rows: bool = false,
    /// Routed prefill rows on the fused GEMM chain; false refuses them (`error.Exl3EngineNotImplemented`).
    prefill: bool = true,
    /// The device passes MLX's NAX gate (an M5-class GPU and macOS 26.2 or later), the consumer's to say. False keeps
    /// every kernel built on MLX's NAX tiles off: routed prefill rows are refused (`error.Exl3EngineNotImplemented`), and
    /// a dense linear's prefill rows dequantize + run MLX's matmul instead of the register-path GEMM. Below the M5 those
    /// GEMMs compute wrong.
    nax: bool = true,
    /// Dense decode linears in one launch (DEC-ONE); false the three-launch chain. The same bytes.
    dec_one: bool = true,
    /// DEC-ONE's GEMV variant (`dense.dec_variants`, `dense_variants` of them; 0 today's): another schedule of the same
    /// reads, decodes and sums, so the same bytes. A dense shape's engine reads it when it is made.
    dense_variant: u8 = 0,
    /// DEC-ONE's output tiles per GEMV threadgroup at each decode row count (index: the call's rows, 1 to 4; 0 the
    /// default, `dense.Dense.btFor`'s): 1, 2, 4 or 8. Launch shape only (a row's sums do not depend on it), so the same
    /// bytes. A dense shape's engine reads it when it is made.
    dec_bt: [5]u8 = @splat(0),
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
            (c.banked_group == 2 or c.banked_group == 4) and (!c.banked_m3 or c.banked_group == 4) and
            c.banked_variant_gu < banked_variants and c.banked_variant_dn < banked_variants and
            ((c.banked_variant_gu == 0 and c.banked_variant_dn == 0) or c.banked_group == 4) and
            c.dense_variant < dense_variants and decBtValid(c.dec_bt);
    }

    fn decBtValid(bt: [5]u8) bool {
        if (bt[0] != 0) return false;
        for (bt[1..]) |b| if (b != 0 and b != 1 and b != 2 and b != 4 and b != 8) return false;
        return true;
    }
};

/// The grouped banked GEMVs' variant count (`decode.group_variants.len`, checked there).
pub const banked_variants: u8 = 12;
/// DEC-ONE's GEMV variant count (`dense.dec_variants.len`, checked there).
pub const dense_variants: u8 = 6;

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
    // DEC-ONE's variants and output tiles: today's by default, every variant and tile count served, others refused
    try t.expectEqual(@as(u8, 0), d.dense_variant);
    try t.expectEqualSlices(u8, &.{ 0, 0, 0, 0, 0 }, &d.dec_bt);
    for (0..dense_variants) |v| try t.expect((Config{ .dense_variant = @intCast(v) }).valid());
    try t.expect(!(Config{ .dense_variant = dense_variants }).valid());
    for ([_]u8{ 1, 2, 4, 8 }) |b| try t.expect((Config{ .dec_bt = .{ 0, b, b, b, b } }).valid());
    for ([_]u8{ 3, 5, 16 }) |b| try t.expect(!(Config{ .dec_bt = .{ 0, 0, 0, b, 0 } }).valid());
    try t.expect(!(Config{ .dec_bt = .{ 2, 0, 0, 0, 0 } }).valid());
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
