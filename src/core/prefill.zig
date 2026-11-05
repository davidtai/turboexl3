//! The prefill chain (rows > DECODE_ROWS_MAX), M3: the routed slots sorted by expert, each expert's rows in M tiles of
//! `Options.bm` rows, and the EXL3 trellis decoded inside the GEMMs' weight loader (DIG-X's shape: one 64 x 64 weight stage
//! decoded once into threadgroup memory feeds 128 rows on the NAX matrix units, f32 accumulation). `Options{}` (128-row
//! tiles) runs seven launches per MoE layer; the served set (`root.pfOptions`: 64-row tiles with the register path for
//! each expert's <= 32-row remainder, `froute`, `t128`, below) runs eight, the route in one and each GEMM in two:
//!
//!   argsort   MLX's argsort of the flat expert ids: sorted position p -> slot order[p] = row * K + k
//!   table     one threadgroup: each expert's run [lo, hi) of sorted positions (a binary search per expert), its M tiles,
//!             an exclusive scan over the experts; tiles[t] = (expert, first position, rows); unused tiles rows = 0
//!             (`Options.froute`: one launch instead of the two: the table kernel counts each expert's slots itself,
//!             threadgroup atomics, scans the counts into each run's start and scatters the slots there)
//!   prep      per (sorted position, 128-block of hidden): the gate and up inputs (f16 round, suh, H128, f16 round) as
//!             half [S, hidden] in sorted order, and inv[order[p]] = p
//!   gemm      gate|up: per (M tile, projection, 64-wide output tile), the K loop of 64-deep stages: decode the stage's 16
//!             trellis tiles into B^T (threadgroup memory), barrier, NAX MMA; the inner product rounded to f16
//!   mid       per (sorted position, 128-block of inter): finish gate and up (H128, svh, f16), SwiGLU in f32 (a clamped
//!             geometry clamps gate and up first: `Geometry.limit`), the down input (f16 round, suh, H128, f16 round)
//!   gemm      down: the same kernel over the inter dim (A/B options: 128-deep stages, several output blocks per threadgroup)
//!   finish    per (row, 128-block of hidden): each slot's down inner through inv, H128, svh, f16, times its score,
//!             summed in slot order
//!
//! Rounding points are `format.project`'s; the GEMM sums in the NAX's order, so parity with the reference is a tolerance.
//! A row's bytes do not depend on where its slots land in the sorted order or the M tiles, so `froute`'s atomic scatter
//! (slots of one expert in arrival order, not slot order) changes no output; no atomics anywhere else.
//!
//! `Options.alay` stores the GEMMs' A operands (xg, xu, xd) in the A fragments' slot order (`alayPos`: per 32-element
//! block, a fragment row of a 32-deep step is one 16-byte load; `alayPos16`: per 16-element block, one 8-byte load per
//! 16-deep slice); `Options.half_rows` lets a simdgroup whose rows end within its first 16 skip its second fragment row.
//! Both move loads and MMAs, not arithmetic: identical outputs.
//!
//! `Options.t128` runs `moeT128`: the route also cuts each expert's rows into 128-row tiles (its full 128-row runs, and a
//! remainder of more than `t128_cut` rows), which the bm-128 shared-decode body multiplies in a launch of their own beside
//! the mixed launch over the rest; each launch writes its own output, and mid and finish read each row from the launch that
//! computed it. bm 128 and bm 64 run the same per-simdgroup K loop, so every output sums the same 16-deep MMAs in the same
//! order: `moe`'s bytes.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const format = @import("format.zig");
const decode = @import("decode.zig");
const lane = @import("lane.zig");

/// MLX steel's NAX tiles (NAXTile, tile_matmad_nax over mpp::tensor_ops::matmul2d; `steel.zig`: the consumer's MLX
/// headers, else the vendored kernels/steel_nax.metal), then turboexl3's decode helpers.
const steel_nax = @import("steel.zig").nax;
const using_steel = @embedFile("kernels/prefill/using_steel.metal");
pub const header_nax: [:0]const u8 = sentinel(steel_nax ++ using_steel ++ decode.header ++ metal_small_helpers ++ metal_gemm_helpers);
/// The register path's header: the NAX tiles, the decoders and its fragment fills.
pub const header_nax_small: [:0]const u8 = sentinel(steel_nax ++ using_steel ++ decode.header ++ metal_small_helpers);
/// The mixed GEMM's header: both bodies' helpers.
pub const header_nax_mixed: [:0]const u8 = header_nax;

/// M tile rows, output tile columns, K stage depth of the GEMM (DIG-X's m128 text).
pub const BM: u32 = 128;
pub const BN: u32 = 64;
pub const BK: u32 = 64;

/// The tile table: one threadgroup of 1024 threads, thread e owns expert e (EXP <= 1024). tiles int32 [TMAX, 4] =
/// (expert, first sorted position, rows, 0) for the tiles in expert order, then rows = 0 to the end.
pub const text_table = @embedFile("kernels/prefill/table_1.metal") ++ table_bsearch ++ @embedFile("kernels/prefill/table_2.metal");

/// The mixed tile table (`Options.small = .mixed`): each expert's rows in BMT-row tiles, except a remainder of <= 32 * SPL rows
/// (or an expert of that few rows), which becomes SPL or fewer register-path tiles of <= 32 rows. tiles int32
/// [TBIG + SPL * EXP, 4]: the BMT-row tiles in expert order from 0, the small tiles in expert order from
/// TBIG = ceil(S / BMT) + EXP; unused entries rows = 0.
pub const text_table_mixed = @embedFile("kernels/prefill/table_mixed_1.metal") ++ table_bsearch ++ @embedFile("kernels/prefill/table_mixed_2.metal");

/// The tables' per-expert binary search over argsort's order (both tables carry it verbatim).
const table_bsearch = @embedFile("kernels/prefill/table_bsearch.metal");

/// `Options.froute`'s replacement for it: count each expert's slots (threadgroup atomics), an exclusive scan of the counts is
/// each expert's first sorted position, then every slot takes the next position of its expert's run: `order` is written
/// here (an output), so the launch replaces argsort + table. Within an expert the slots land in arrival order.
const froute_count = @embedFile("kernels/prefill/froute_count.metal");

/// The fused route kernels (`Options.froute`): the tables with `froute_count` for the binary search; inputs inds, outputs
/// order u32 [S] and tiles.
pub const text_froute: [:0]const u8 = sentinel(@embedFile("kernels/prefill/table_1.metal") ++ froute_count ++ @embedFile("kernels/prefill/table_2.metal"));
pub const text_froute_m: [:0]const u8 = sentinel(@embedFile("kernels/prefill/table_mixed_1.metal") ++ froute_count ++ @embedFile("kernels/prefill/table_mixed_2.metal"));

/// The gate and up inputs in sorted order: xg, xu half [S, HID] = H128(f16(x[row]) * suh_e) * 2^-3.5 rounded, per 128-block;
/// one simdgroup per (block, sorted position), BPT blocks per threadgroup; inv u32 [S]: inv[order[p]] = p. ALAY
/// (`Options.alay`): each 32-block's element j at slot `turboexl3_alay(j)` (1) or `turboexl3_alay16(j)` (2) of its block; the
/// block sums are the same.
pub const text_prep = @embedFile("kernels/prefill/prep.metal");

/// gate|up finish, SwiGLU and the down input, per (128-block of inter, sorted position): xd half [S, INTER] (ALAY: in
/// `text_prep`'s A layout).
pub const text_mid = mid_1 ++ mid_2 ++ mid_3 ++ mid_4 ++ mid_swiglu ++ mid_5;
/// `text_mid` in the pieces the 128-row tiles' mid (`text_mid_t128`) and the clamped mids compose: mid_1 ends at the expert
/// line, mid_3 is the gate and up loads, mid_swiglu the SwiGLU's line (the vector-layout mid's too)
const mid_1 = @embedFile("kernels/prefill/mid_1.metal");
const mid_2 = @embedFile("kernels/prefill/mid_2.metal");
const mid_3 = @embedFile("kernels/prefill/mid_3.metal");
const mid_4 = @embedFile("kernels/prefill/mid_4.metal");
const mid_swiglu = @embedFile("kernels/prefill/mid_swiglu.metal");
const mid_5 = @embedFile("kernels/prefill/mid_5.metal");

/// The clamped SwiGLU (`Geometry.limit`, the template int LIMIT; `root.moeClamped`): each mid with its SwiGLU line replaced
/// by the clamp (the gate clamped from above at LIMIT and up to [-LIMIT, LIMIT], then the mid's SwiGLU on them), every
/// other statement the mid's, so where no value passes the limit each gives its mid's bytes: `text_midc` `text_mid`'s,
/// `text_mid_vc` `text_mid_v`'s, `text_mid_t128c` `text_mid_t128`'s.
const mid_swiglu_clamp = @embedFile("kernels/prefill/mid_swiglu_clamp.metal");
pub const text_midc = mid_1 ++ mid_2 ++ mid_3 ++ mid_4 ++ mid_swiglu_clamp ++ mid_5;
pub const text_mid_vc = mid_v_1 ++ mid_swiglu_clamp ++ mid_v_2;
pub const text_mid_t128c: [:0]const u8 = sentinel(mid_1 ++ mid_t128_pick ++ mid_2 ++ mid_t128_loads ++ mid_4 ++ mid_swiglu_clamp ++ mid_5);

/// The down finish and the slot reduce, per (128-block of hidden, row): y OUT_T [R, HID] = sum over the row's slots in slot
/// order of score * f16(H128(zd[inv[slot]]) * 2^-3.5 * svh_e).
pub const text_finish = finish_1 ++ finish_2 ++ finish_3 ++ finish_4;
/// `text_finish` in the pieces the 128-row tiles' finish (`text_finish_t128`) composes: finish_1 ends at the expert line,
/// finish_3 is the down load
const finish_1 = @embedFile("kernels/prefill/finish_1.metal");
const finish_2 = @embedFile("kernels/prefill/finish_2.metal");
const finish_3 = @embedFile("kernels/prefill/finish_3.metal");
const finish_4 = @embedFile("kernels/prefill/finish_4.metal");

/// `Options.aux`: prep, mid and finish with each lane holding 4 consecutive elements of its 128-block (8-byte loads and
/// stores; `turboexl3_h128v4`, `turboexl3_sum_x4`: the same values, so identical bytes) and ASG simdgroups per threadgroup over
/// the flattened (position or row, block) items instead of one small threadgroup per position.
pub const text_prep_v = @embedFile("kernels/prefill/prep_v.metal");

pub const text_mid_v = mid_v_1 ++ mid_swiglu ++ mid_v_2;
/// `text_mid_v` cut at its SwiGLU line (`mid_swiglu`, the same line as `text_mid`'s)
const mid_v_1 = @embedFile("kernels/prefill/mid_v_1.metal");
const mid_v_2 = @embedFile("kernels/prefill/mid_v_2.metal");

pub const text_finish_v = @embedFile("kernels/prefill/finish_v.metal");

/// The fused GEMM: z OUT [S, ND] = f16(x [S, KD] (sorted rows) x W_e [KD, ND]) per projection op (NOPS 1: down; 2: gate|up,
/// op 0 reads x0/w0 and writes z0, op 1 x1/w1/z1). One threadgroup of 2 BMR / 32 simdgroups (BMR / 32 x 2, 32 x 32 outputs
/// each) per (M tile, op, NCB consecutive 64-wide output blocks), the blocks one after another. Per 64-deep K sub-stage
/// simdgroup sg decodes trellis tiles sg, sg + NSG, ... of its 4 x 4 (lane L: its eight codewords at code positions 8L..8L+7,
/// the tensor-core layout's rows 2q, 2q+1, 2q+8, 2q+9 of columns c, c+8, stored as four half2 into B^T [n][k], conflict-free
/// at LDB 72, or with FRAGB as the lane's NAX fragment), and each simdgroup runs two 32-deep NAX MMAs over it (A from device,
/// B^T from threadgroup memory). A K stage, the span between barriers, is BKS / 64 sub-stages: 1, or 2 for the down GEMM
/// with `Options.bk128` (16 KB per fragment-order buffer). Double-buffered (DBUF), stage f + 1 decodes while stage f
/// multiplies, sub-stage by sub-stage; the stage sequence runs on across a threadgroup's column blocks, so block j + 1's first
/// stage decodes during block j's last. Each block's accumulator starts clear and takes its sub-stages in K order: every
/// output sums the same 16-deep MMAs in the same order at any BKS and NCB. ALAY (`Options.alay`, fragb): A rows hold each
/// 32-block (1) or 16-block (2) in the fragments' slot order, so lane q's elements start at 8q (one 16-byte load per fragment
/// row and 32-deep step, `exl3pf_afrag2`) or 4q (one 8-byte load per fragment row and 16-deep slice, `exl3pf_afrag4`)
/// instead of 2q; H16 (`Options.half_rows`, fragb): a simdgroup with at most 16 rows multiplies only its first fragment row
/// (`exl3pf_mma_top`). `two`: the two lines that name op 1's operands read x1 / w1 / xs1 and write z1 (gate|up), else op 0's
/// again (down, NOPS 1); `mixed`: the mixed launch's big body, its weight stage `mixed_bs` declared at kernel scope.
fn gemmText(comptime two: bool, comptime mixed: bool) []const u8 {
    const stage = if (mixed) @embedFile("kernels/prefill/gemm_mixed_bs.metal") else @embedFile("kernels/prefill/gemm_2.metal");
    const wx = if (two) @embedFile("kernels/prefill/gemm_op2_wx.metal") else @embedFile("kernels/prefill/gemm_op1_wx.metal");
    const xsz = if (two) @embedFile("kernels/prefill/gemm_op2_xsz.metal") else @embedFile("kernels/prefill/gemm_op1_xsz.metal");
    return @embedFile("kernels/prefill/gemm_1.metal") ++ stage ++ @embedFile("kernels/prefill/gemm_3.metal") ++ wx ++ @embedFile("kernels/prefill/gemm_4.metal") ++ xsz ++
        @embedFile("kernels/prefill/gemm_5.metal");
}

/// The GEMM's two halves of a K stage. `exl3pf_stage`: simdgroup sg decodes trellis tiles sg, sg + NSG, ... of the stage's
/// 4 x 4 (lane L: its eight codewords at code positions 8L..8L+7, the tensor-core layout's rows 2q, 2q+1, 2q+8, 2q+9 of
/// columns c, c+8) into B^T [n][k] as four half2, conflict-free at LDB 72. `exl3pf_mma`: two 32-deep NAX MMAs of the
/// simdgroup's 32 x 32 outputs (A from device, B^T from threadgroup memory).
const metal_gemm_helpers = @embedFile("kernels/prefill/gemm_helpers.metal");

/// The register-path GEMM for tiles of <= 32 rows (`Options.small`): every simdgroup is an independent work item computing a
/// 32-row x 32-column output block of one tile, no threadgroup memory and no barriers. Per 32-deep K step a lane loads its NAX
/// A fragment elements and decodes its four trellis tiles straight into its B^T fragment registers. NAX lane L (bits b4..b0)
/// holds fragment rows fm = 4 b4 + 2 b2 + b1 and fm + 8 at k-slots 4q .. 4q + 3, q = 2 b3 + b0 (BaseNAXFrag::get_coord); the
/// trellis lane plan of lane L' = (b4, b2, b1, b3, b0) decodes exactly the weights of n = fm, fm + 8 at true k = 2q + {0, 1,
/// 8, 9}, in fragment element order. So slot 4q + j stands for true k 2q + {0, 1, 8, 9}[j] in BOTH operands: the A elements
/// are read from the natural-order inputs as two half2 per row (k 2q, 2q + 1 and 2q + 8, 2q + 9), and the contraction over
/// slots is the contraction over k. Rows past the tile's count read its last row (clamped pointers, no branches); their
/// products are never stored. Threadgroups hold SGT work items. SINTR (`Options.sintr`, n % 16 == 0): the lane plan's
/// codewords through the integer-rate reader instead of the general one, set up from the same permuted lane L' (both
/// readers place that lane's first window at the same start bit, so the same words, word selects, extract offsets and
/// codewords). ALAY (`Options.alay`): the inputs hold each 32-block (1: one 16-byte load per fragment row and 32-deep step, at
/// 8q) or each 16-block (2: one 8-byte load per fragment row and 16-deep slice, at 4q) in that slot order; H16
/// (`Options.half_rows`): an item of at most 16 rows multiplies only its first fragment row. `two` as `gemmText`'s; `mixed`:
/// the mixed launch's small body, its work items from GBIGM on over the table's small tiles.
fn gemmSmallText(comptime two: bool, comptime mixed: bool) []const u8 {
    const items = if (mixed) @embedFile("kernels/prefill/gemm_mixed_index.metal") else @embedFile("kernels/prefill/gemm_small_2.metal");
    const wx = if (two) @embedFile("kernels/prefill/gemm_small_op2_wx.metal") else @embedFile("kernels/prefill/gemm_small_op1_wx.metal");
    const xsz = if (two) @embedFile("kernels/prefill/gemm_small_op2_xsz.metal") else @embedFile("kernels/prefill/gemm_small_op1_xsz.metal");
    return @embedFile("kernels/prefill/gemm_small_1.metal") ++ items ++ @embedFile("kernels/prefill/gemm_small_3.metal") ++ wx ++ @embedFile("kernels/prefill/gemm_small_4.metal") ++
        xsz ++ @embedFile("kernels/prefill/gemm_small_5.metal");
}

/// The register path's fragment fills: A from two natural-order rows (k 2q, 2q + 1, 2q + 8, 2q + 9 of each) or, with ALAY,
/// from two rows in that slot order (`exl3pf_afrag2`: both 16-deep halves of a 32-deep step; `exl3pf_afrag4`: one slice),
/// B^T from one trellis tile decoded with the permuted lane plan; the first-fragment-row MMA (`exl3pf_mma_top`, H16); and the
/// codeword reader both GEMM bodies decode through (`exl3pf_cw`).
const metal_small_helpers = @embedFile("kernels/prefill/small_helpers.metal");

/// The NAX lane L's trellis lane for the register path: L' = (b4, b2, b1, b3, b0).
pub fn smallLanePlan(l: u32) u32 {
    return (l & 16) | (((l >> 2) & 1) << 3) | (((l >> 1) & 1) << 2) | (((l >> 3) & 1) << 1) | (l & 1);
}

/// H128 of a simdgroup's 128 values, lane L holding elements L + 32 j: five shuffle stages, then the two register stages.
const metal_h128 = @embedFile("kernels/prefill/h128.metal");
pub const header_small: [:0]const u8 = sentinel(metal_h128 ++ decode.header);

/// `Options.alay` 1's slot of element j (0..31) of a 32-element block of an A row: the fragment lane q = (j >> 1) & 3 reads k
/// 2q, 2q + 1, 2q + 8, 2q + 9 of each 16-deep slice (the trellis lane plan's order), so those of slice j >> 4 sit at
/// 8q + 4 (j >> 4) + {0, 1, 2, 3}: a lane's 8 elements of a 32-deep step are one 16-byte run. The Metal `turboexl3_alay`.
pub fn alayPos(j: u32) u32 {
    return 8 * ((j >> 1) & 3) + 4 * (j >> 4) + ((j & 1) | (((j >> 3) & 1) << 1));
}

/// `Options.alay` 2's slot: the same order per 16-element block (16 (j >> 4) + 4q + {0, 1, 2, 3}, dense.zig's LANE_ORDER): a
/// lane's 4 elements of a 16-deep slice are one 8-byte run. The Metal `turboexl3_alay16`.
pub fn alayPos16(j: u32) u32 {
    return (j & 16) + 4 * ((j >> 1) & 3) + ((j & 1) | (((j >> 3) & 1) << 1));
}

/// A comptime string as a NUL-terminated slice (the C API's text arguments).
fn sentinel(comptime s: []const u8) [:0]const u8 {
    comptime {
        var buf: [s.len:0]u8 = undefined;
        @memcpy(buf[0..s.len], s);
        const final = buf;
        return &final;
    }
}

fn replace(comptime s: []const u8, comptime needle: []const u8, comptime with: []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(10_000_000);
        var out: []const u8 = "";
        var i: usize = 0;
        while (std.mem.indexOfPos(u8, s, i, needle)) |j| {
            out = out ++ s[i..j] ++ with;
            i = j + needle.len;
        }
        return out ++ s[i..];
    }
}

const gemm2_text = sentinel(gemmText(true, false));
pub const gemm1_text = sentinel(gemmText(false, false));
/// The mixed GEMM (`Options.small = .mixed`): one launch, threadgroups below GBIGM take a BMR-row tile (NCB column blocks of it)
/// through the shared-decode body (`gemmText`), the rest run NSG independent register-path work items each
/// (`gemmSmallText`) over the table's small tiles. Both bounds come from the tile table's shape, so the row count compiles
/// nothing. The weight stage is declared at kernel scope (the big body's `bs_mem`, same size).
fn gemmMixedText(comptime two: bool) []const u8 {
    return @embedFile("kernels/prefill/gemm_mixed_1.metal") ++ gemmText(two, true) ++ @embedFile("kernels/prefill/gemm_mixed_2.metal") ++ gemmSmallText(two, true) ++
        @embedFile("kernels/prefill/gemm_mixed_3.metal");
}
const gemm2m_text = sentinel(gemmMixedText(true));
const gemm1m_text = sentinel(gemmMixedText(false));
const gemm2s_text = sentinel(gemmSmallText(true, false));
pub const gemm1s_text = sentinel(gemmSmallText(false, false));

// ── The 128-row tiles (`Options.t128`, `Prefill.moeT128`) ──
//
// Every expert's full 128-row runs (and, past its last one, a remainder of more than `t128_cut` rows) become 128-row tiles
// on the bm-128 shared-decode body, where one decoded 64 x 64 weight stage feeds 128 rows instead of 64; the rest of each
// expert (at most `t128_cut` rows) takes the mixed launch (64-row tiles, register-path tiles of <= 32 rows). A launch has
// one threadgroup size, so the two run as two launches, each writing its own z (an MLX output [S, N] at sorted positions,
// never written or read at the other launch's rows); the route's `xbase` [E] (each expert's first remainder position)
// tells mid and finish which z holds a row: position p of expert e is the 128-row tiles' when p < xbase[e]. Every write is
// to its own kernel's output (a launch writing another's input binding is undefined in Metal), so MLX orders the readers
// after both launches, and nothing passes between the two, so the second starts as the first drains. Both tables are
// sized by routing-independent bounds (`t128MixedRows`, `t128TileRows`) that keep the early-exit threadgroups near the
// mixed table's count. Every output sums the same 16-deep MMAs in the same K order (bm 128 and bm 64 run the same
// per-simdgroup K loop): `moe`'s bytes, for a second [S, N] output per GEMM.

/// Past an expert's full 128-row tiles, a remainder of more than this many rows takes one part-full 128-row tile (one
/// decode pass over its weights where the mixed tiles of the same rows take two, at the cost of idle simdgroups).
pub const t128_cut: u8 = 64;

/// The 128-row tiles' route (`Options.froute`'s one launch: `froute_count` verbatim, so `order` is froute's): from each
/// expert's first sorted position, 128-row tiles while 128 rows are left, then, when more than CUT remain (CU = min(CUT,
/// 127)), one part-full 128-row tile for them; whatever is left takes `text_table_mixed`'s rule (BMT-row tiles, a <= 32
/// SPL-row remainder as register-path tiles). T128 0: no 128-row tiles (the mixed table of every row). Outputs order u32
/// [S], tiles int32 [TBIG + SPL EXP, 4] (`text_table_mixed`'s layout, the mixed launch's table), tiles128 int32 [TQ, 4] and
/// xbase int32 [EXP] (each expert's first remainder position: its rows before it are the 128-row tiles', `text_mid_t128` /
/// `text_finish_t128`); TBIG and TQ are `t128MixedRows`' and `t128TileRows`' expressions (bounds that hold for any routing);
/// unused entries rows = 0.
pub const text_froute_t128: [:0]const u8 = sentinel(@embedFile("kernels/prefill/froute_t128_1.metal") ++ froute_count ++
    @embedFile("kernels/prefill/froute_t128_2.metal"));

/// The 128-row tiles' mid: `text_mid` with each position's gate and up rows from the launch that computed them (zgt / zut,
/// the 128-row tiles' outputs, for p < xbase[e]; zg / zu, the mixed launch's, otherwise); one simdgroup holds one position,
/// so the choice is uniform per simdgroup. Nothing else changes, so the same bytes from the same rows.
const mid_t128_pick = @embedFile("kernels/prefill/mid_t128_pick.metal");
const mid_t128_loads = @embedFile("kernels/prefill/mid_t128_loads.metal");
pub const text_mid_t128: [:0]const u8 = sentinel(mid_1 ++ mid_t128_pick ++ mid_2 ++ mid_t128_loads ++ mid_4 ++ mid_swiglu ++ mid_5);
/// The 128-row tiles' finish: `text_finish` with each slot's down row from the launch that computed it (zdt for p < xbase[e],
/// else zd)
const finish_t128_pick = @embedFile("kernels/prefill/finish_t128_pick.metal");
pub const text_finish_t128: [:0]const u8 = sentinel(finish_1 ++ finish_t128_pick ++ finish_2 ++ @embedFile("kernels/prefill/finish_t128_load.metal") ++
    finish_4);

const table_in = [_][*:0]const u8{ "inds", "order" };
const table_out = [_][*:0]const u8{"tiles"};
const froute_in = [_][*:0]const u8{"inds"};
const froute_out = [_][*:0]const u8{ "order", "tiles" };
const prep_in = [_][*:0]const u8{ "x", "inds", "order", "suh_g", "suh_u" };
const prep_out = [_][*:0]const u8{ "xg", "xu", "inv", "xsg", "xsu" };
const gemm2_in = [_][*:0]const u8{ "tiles", "x0", "x1", "w0", "w1", "xs0", "xs1" };
const gemm2_out = [_][*:0]const u8{ "z0", "z1" };
pub const gemm1_in = [_][*:0]const u8{ "tiles", "x0", "w0", "xs0" };
pub const gemm1_out = [_][*:0]const u8{"z0"};
const mid_in = [_][*:0]const u8{ "zg", "zu", "inds", "order", "svh_g", "svh_u", "suh_d" };
const mid_out = [_][*:0]const u8{ "xd", "xsd" };
const fin_in = [_][*:0]const u8{ "zd", "inds", "inv", "scores", "svh_d" };
const fin_out = [_][*:0]const u8{"y"};
/// The 128-row tiles' route outputs; mid and finish take the 128-row tiles' outputs and xbase as more inputs (both GEMMs'
/// launches keep the mixed chain's kernels, inputs and outputs)
const froute_t128_out = [_][*:0]const u8{ "order", "tiles", "tiles128", "xbase" };
const mid_t128_in = mid_in ++ [_][*:0]const u8{ "zgt", "zut", "xbase" };
const fin_t128_in = fin_in ++ [_][*:0]const u8{ "zdt", "xbase" };

pub const Geometry = struct {
    hidden: u32,
    inter: u32,
    topk: u32,
    experts: u32,
    rate: format.Rate,
    dec: format.Decode,
    /// the SwiGLU's clamp (`root.moeClamped`: DeepSeek-V4's and GLM's `swiglu_limit`; 0 none): every mid is its clamped
    /// variant (`text_midc`, `text_mid_vc`, `text_mid_t128c`) with LIMIT = limit; nothing else changes
    limit: u32 = 0,

    pub fn admitted(g: Geometry) bool {
        return g.hidden % 128 == 0 and g.inter % 128 == 0 and g.hidden % BK == 0 and g.inter % BK == 0 and
            g.hidden % BN == 0 and g.inter % BN == 0 and g.experts >= 1 and g.experts <= 1024 and g.topk >= 1 and g.topk <= 32;
    }
};

/// The largest count of 128-blocks per threadgroup (at most 8 simdgroups) that divides `blocks`.
fn bptFor(blocks: u32) u32 {
    var b: u32 = 8;
    while (b > 1) : (b -= 1) if (blocks % b == 0) return b;
    return 1;
}

pub const Operands = struct {
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

/// Build-time variants of the GEMM (the benchmark's arms): M tile rows (64, 128 or 256: 2, 4 or 8 simdgroup rows of 32, two
/// simdgroups per row covering the 64-wide output tile) and a double-buffered weight stage (one barrier per K stage, the next
/// stage decoded while this one multiplies).
pub const Options = struct {
    bm: u32 = 128,
    dbuf: bool = false,
    /// .all: every M tile is <= 32 rows and runs the register-path GEMM (`gemmSmallText`), sgt work items per threadgroup
    small: Small = .off,
    sgt: u32 = 4,
    /// timing probes only (outputs are wrong), a bit mask: 1 skips the trellis read and decode (constant weights), 2 skips the
    /// MMA step entirely (the compiler then drops the unread decode too), 4 replaces the A loads by zeros, 8 the B loads;
    /// 16 (outputs still right) keeps the fragment-order stage at the padded 64 x 72 size it had before it was trimmed to 64 x 64;
    /// 32 (fragb) adds a 128-point Hadamard's butterflies to every decoded weight (C14's cost floor)
    probe: u8 = 0,
    /// producer simdgroups that only decode (0 = every simdgroup decodes and multiplies); needs dbuf
    wsp: u8 = 0,
    /// issue the stage's MMA before decoding the next stage (dbuf)
    mma_first: bool = false,
    /// both 32-deep sub-steps' operands loaded up front, no compiler barrier between them
    unroll: bool = false,
    /// the weight stage in NAX fragment order (one 16-byte store per decoded tile per lane, one 16-byte load per fragment)
    fragb: bool = false,
    /// the integer-rate codeword reader (n % 16 == 0)
    intr: bool = false,
    /// the register path (small = .all, or .mixed's <= 32-row tiles) reads its codewords with intr's integer-rate reader too
    /// (needs intr); off, it keeps the general reader, so an A/B of the two register-path readers runs in one binary. Same
    /// codewords either way: outputs identical
    sintr: bool = false,
    /// MUL1: the GEMMs multiply the codeword byte sums t and the epilogue applies w = A t + C as A acc + C sum(x)
    defer_affine: bool = false,
    /// .mixed: remainders of up to 32 * split rows take the register path as split tiles of <= 32 rows (1 or 2)
    split: u8 = 1,
    /// the down GEMM's K stages 128 deep: two 64-deep sub-stages per buffer between barriers (each decoded and multiplied by
    /// the same helpers as a 64-deep stage), so half the barriers and stage-loop turns of its K loop; 16 KB per fragment-order
    /// buffer, 32 KB double-buffered (the padded layout's 36 KB does not fit Apple's 32 KB: fragb, or single-buffered).
    /// gate|up keeps 64
    bk128: bool = false,
    /// 64-wide output blocks per down-GEMM threadgroup (1, 2 or 4; the hidden width's blocks must divide by it): each
    /// threadgroup walks ncb consecutive blocks of its M tile, the double buffer carried across them (block j + 1's first
    /// stage decoded while block j's last multiplies); gate|up keeps 1
    ncb: u8 = 1,
    /// route in one launch (the table kernel counts, scans and scatters the slots itself) instead of MLX's argsort + the
    /// table; outputs identical (see the module doc)
    froute: bool = false,
    /// prep, mid and finish in the vector layout (`text_prep_v`, ...) with aux simdgroups per threadgroup (4, 8, 16 or 32;
    /// 0 = the strided kernels); outputs identical
    aux: u8 = 0,
    /// with aux: each block's sums (the deferred map's C term) in the lane's own order instead of the strided kernels' (no
    /// shuffles; f32 sums in a different order, so outputs may differ in the last bits)
    aux_lane_sums: bool = false,
    /// the GEMMs' A operands (prep's xg / xu, mid's xd) stored in the A fragments' slot order: the lane that multiplies k 2q,
    /// 2q + 1, 2q + 8, 2q + 9 of a 16-deep slice (fragb's and the register path's rule) finds them contiguous. 1: each
    /// 32-element block in that order (`alayPos`), both slices of a 32-deep step at halfs 8q .. 8q + 7, one 16-byte load per
    /// fragment row and step (8 per 64-deep stage instead of 32 half2 loads; 4 per register-path step instead of 16); 2: each
    /// 16-element block (`alayPos16`, dense.zig's LANE_ORDER rule), one 8-byte load straight into each fragment row half (16
    /// per stage, 8 per step: no splitting of a loaded vector across two fragments). The same values in the same slots, so
    /// identical outputs. Where the shared-decode body runs it needs fragb (the padded bodies load A in natural order); its
    /// writers are the strided prep and mid (aux 0)
    alay: u8 = 0,
    /// a fragment-order shared-decode simdgroup (fragb) or a register-path item whose rows end within its first 16 loads and
    /// multiplies only that 16-row fragment row: half its 16x32x16 MMAs and A loads; the rows it skips are past the tile and
    /// never stored, the rows it keeps take the same MMAs in the same order, so identical outputs
    half_rows: bool = false,
    /// each expert's full 128-row runs (and a remainder of more than `t128_cut` rows) in 128-row tiles on the bm-128 body, a
    /// launch of their own beside the mixed launch over the rest (`Prefill.moeT128`, the notes above `t128_cut`): `moe`'s
    /// bytes. Needs bm 64, the mixed table at split 1, froute's route and the strided mid and finish (aux 0)
    t128: bool = false,

    pub const Small = enum { off, all, mixed };

    pub fn valid(o: Options) bool {
        if (o.small == .mixed and o.wsp != 0) return false;
        if (o.split != 1 and (o.split != 2 or o.small != .mixed)) return false;
        if (o.wsp != 0 and (!o.dbuf or (o.wsp != 2 and o.wsp != 4 and o.wsp != 8 and o.wsp != 16) or 2 * o.bm / 32 + o.wsp > 32)) return false;
        // both reshape the shared-decode body: nothing to do for the all-register path
        if ((o.bk128 or o.ncb != 1) and o.small == .all) return false;
        // the register path's reader: intr's (and its n % 16 == 0 check), and nothing to do where no register path runs
        if (o.sintr and (!o.intr or o.small == .off)) return false;
        if (o.ncb != 1 and o.ncb != 2 and o.ncb != 4) return false;
        if (o.aux != 0 and o.aux != 4 and o.aux != 8 and o.aux != 16 and o.aux != 32) return false;
        // the A layout and the first-fragment-row MMA live in the fragment-order body and the register path; the layout's
        // writers are the strided prep and mid
        if (o.alay > 2) return false;
        if ((o.alay != 0 or o.half_rows) and !o.fragb and o.small != .all) return false;
        if (o.alay != 0 and o.aux != 0) return false;
        // the remainder tiles ride the mixed launch (froute's route at split 1, the strided mid and finish), the 128-row tiles
        // the bm-128 shared-decode body beside bm 64's
        if (o.t128 and (o.small != .mixed or o.split != 1 or !o.froute or o.aux != 0 or o.bm != 64)) return false;
        if (o.stageBytes(o.downStage()) > 32768) return false;
        return (o.bm == 64 or o.bm == 128 or o.bm == 256) and (o.sgt == 1 or o.sgt == 2 or o.sgt == 4 or o.sgt == 8);
    }

    /// rows per M tile of the table
    pub fn tileRows(o: Options) u32 {
        return if (o.small == .all) 32 else o.bm;
    }

    /// the down GEMM's K stage depth (gate|up's is BK)
    pub fn downStage(o: Options) u32 {
        return if (o.bk128) 128 else BK;
    }

    /// threadgroup memory of the shared-decode GEMM's weight stage at K stage depth `bk` (bk / BK sub-stages per buffer),
    /// bytes, both buffers with dbuf
    pub fn stageBytes(o: Options, bk: u32) u32 {
        const sub: u32 = if (o.fragb and (o.probe & 16) == 0) BN * BK else BN * (BK + 8);
        return (if (o.dbuf) @as(u32, 2) else 1) * (bk / BK) * sub * 2;
    }
};

/// The routing of one call: sorted position -> slot (`order`), the M tiles (`tiles` int32 [tmax, 4]).
pub const Route = struct {
    order: mlx.mlx_array,
    tiles: mlx.mlx_array,
    tmax: u32,

    pub fn deinit(r: Route) void {
        _ = mlx.mlx_array_free(r.order);
        _ = mlx.mlx_array_free(r.tiles);
    }
};

/// The 128-row tiles' routing (`Prefill.routeT128`): `Route`'s order and mixed table (`tiles` over every expert's remainder
/// past its 128-row tiles, `tmax` its BMT-row tile rows, `t128MixedRows`), the 128-row tiles (`tiles128` int32 [tq, 4], tq =
/// `t128TileRows`) and `xbase` int32 [experts] (each expert's first remainder position, `ZT128`).
pub const RouteT128 = struct {
    order: mlx.mlx_array,
    tiles: mlx.mlx_array,
    tiles128: mlx.mlx_array,
    xbase: mlx.mlx_array,
    tmax: u32,
    tq: u32,

    pub fn deinit(r: RouteT128) void {
        _ = mlx.mlx_array_free(r.order);
        _ = mlx.mlx_array_free(r.tiles);
        _ = mlx.mlx_array_free(r.tiles128);
        _ = mlx.mlx_array_free(r.xbase);
    }
};

/// One GEMM's f16 inner products as its two launches wrote them, each its own MLX output at sorted positions: `m` the mixed
/// launch's (the remainder tiles' rows), `t` the 128-row tiles' launch's. Row p of expert e is `t`'s when p < xbase[e],
/// else `m`'s; neither array is written (or read) at the other's rows.
pub const ZT128 = struct {
    m: [2]mlx.mlx_array,
    t: [2]mlx.mlx_array,
    n: u8,

    pub fn deinit(x: ZT128) void {
        for (x.m[0..x.n]) |a| _ = mlx.mlx_array_free(a);
        for (x.t[0..x.n]) |a| _ = mlx.mlx_array_free(a);
    }
};

/// The 128-row tile table's rows for S slots (at least 1; 1 with the tiles off). A full tile holds 128 rows, a part-full one
/// more than cu = min(cut, 127) and an expert has at most one, so np part-full tiles leave at most (S - (cu + 1) np) / 128
/// full ones; the count is largest at np = min(experts, S / (cu + 1)). Every routing fits, and the launch's grid stays near
/// its tiles (each spare table row is a row of early-exit threadgroups). `text_froute_t128`'s TQ is the same expression.
pub fn t128TileRows(S: u32, experts: u32, cut: u8, t128: bool) u32 {
    if (!t128) return 1;
    const cu1: u32 = @as(u32, @min(cut, 127)) + 1;
    const np = @min(experts, S / cu1);
    return @max(1, np + (S - cu1 * np) / 128);
}

/// The mixed table's `bm`-row tile rows for S slots (its register-path rows follow them): ceil(S / bm) + experts without the
/// 128-row tiles; with them an expert's remainder is at most cu = min(cut, 127) rows, so at most cu / bm + (cu % bm > 32
/// split) such tiles each, every one of at least min(bm, 32 split + 1) rows (at least 1). The tables of every routing fit.
/// `text_froute_t128`'s TBIG is the same expression.
pub fn t128MixedRows(S: u32, experts: u32, bm: u32, split: u32, cut: u8, t128: bool) u32 {
    const all = (S + bm - 1) / bm + experts;
    if (!t128) return all;
    const cu: u32 = @min(cut, 127);
    const sr = 32 * split;
    const per = cu / bm + @intFromBool(cu % bm > sr);
    return @max(1, @min(all, @min(experts * per, S / @min(bm, sr + 1))));
}

/// The 128-row tiles' tables from per-expert slot counts, `text_froute_t128`'s rule on the CPU (experts in order, each run
/// from its first sorted position): the 128-row tiles, the remainder's `bm`-row tiles and its register-path tiles (split
/// 1), each as (expert, first sorted position, rows). Tests check the route kernel's tables and their cover against it.
pub const T128Tables = struct {
    t128: std.ArrayList([3]u32) = .empty,
    big: std.ArrayList([3]u32) = .empty,
    small: std.ArrayList([3]u32) = .empty,

    pub fn deinit(t: *T128Tables, a: std.mem.Allocator) void {
        t.t128.deinit(a);
        t.big.deinit(a);
        t.small.deinit(a);
    }
};

pub fn t128Tables(a: std.mem.Allocator, counts: []const u32, bm: u32, cut: u8, t128: bool) !T128Tables {
    var t: T128Tables = .{};
    errdefer t.deinit(a);
    const cu: u32 = @min(cut, 127);
    var lo: u32 = 0;
    for (counts, 0..) |c0, ei| {
        const e: u32 = @intCast(ei);
        var nq: u32 = if (t128) c0 / 128 else 0;
        var cnt = c0 - 128 * nq;
        if (t128 and cnt > cu) {
            nq += 1;
            cnt = 0;
        }
        var q: u32 = 0;
        while (q < nq) : (q += 1) try t.t128.append(a, .{ e, lo + 128 * q, @min(128, c0 - 128 * q) });
        const base = lo + c0 - cnt;
        const nfull = cnt / bm;
        const r = cnt - nfull * bm;
        const nb = nfull + @intFromBool(r > 32);
        var i: u32 = 0;
        while (i < nb) : (i += 1) try t.big.append(a, .{ e, base + bm * i, @min(bm, cnt - bm * i) });
        if (r > 0 and r <= 32) try t.small.append(a, .{ e, base + nfull * bm, r });
        lo += c0;
    }
    return t;
}

/// The 128-row tiles' kernels (`Options.t128`): the route, both GEMMs' launches (over the remainder table: the mixed launch;
/// over the 128-row tiles: the shared-decode body at bm 128; the mixed chain's texts, header, inputs and outputs), and mid
/// and finish taking each row from the launch that computed it (mid's clamped variant where the geometry clamps).
pub const T128Kernels = struct {
    route: mlx.mlx_fast_metal_kernel,
    gu_mixed: mlx.mlx_fast_metal_kernel,
    gu_t128: mlx.mlx_fast_metal_kernel,
    dn_mixed: mlx.mlx_fast_metal_kernel,
    dn_t128: mlx.mlx_fast_metal_kernel,
    mid: mlx.mlx_fast_metal_kernel,
    fin: mlx.mlx_fast_metal_kernel,

    fn init(limit: u32) !T128Kernels {
        return .{
            .route = ks.newKernel("turboexl3_pf_t128_route", &froute_in, &froute_t128_out, text_froute_t128, header_small, true) orelse return error.KernelCreateFailed,
            .gu_mixed = ks.newKernel("turboexl3_pf_t128_gum", &gemm2_in, &gemm2_out, gemm2m_text, header_nax, true) orelse return error.KernelCreateFailed,
            .gu_t128 = ks.newKernel("turboexl3_pf_t128_gut", &gemm2_in, &gemm2_out, gemm2_text, header_nax, true) orelse return error.KernelCreateFailed,
            .dn_mixed = ks.newKernel("turboexl3_pf_t128_dnm", &gemm1_in, &gemm1_out, gemm1m_text, header_nax, true) orelse return error.KernelCreateFailed,
            .dn_t128 = ks.newKernel("turboexl3_pf_t128_dnt", &gemm1_in, &gemm1_out, gemm1_text, header_nax, true) orelse return error.KernelCreateFailed,
            .mid = (if (limit == 0)
                ks.newKernel("turboexl3_pf_t128_mid", &mid_t128_in, &mid_out, text_mid_t128, header_small, true)
            else
                ks.newKernel("turboexl3_pf_t128_midc", &mid_t128_in, &mid_out, text_mid_t128c, header_small, true)) orelse return error.KernelCreateFailed,
            .fin = ks.newKernel("turboexl3_pf_t128_fin", &fin_t128_in, &fin_out, text_finish_t128, header_small, true) orelse return error.KernelCreateFailed,
        };
    }

    fn deinit(k: T128Kernels) void {
        inline for (.{ k.route, k.gu_mixed, k.gu_t128, k.dn_mixed, k.dn_t128, k.mid, k.fin }) |x| decode.freeKernel(x);
    }
};

/// The compiled kernels of one prefill geometry and option set; launch configs are built per call (their sizes follow the
/// row count, the templates do not, so a new prompt length compiles nothing). The stages are public so the benchmark can
/// time each alone; `moe` runs them in order.
pub const Prefill = struct {
    geo: Geometry,
    opts: Options,
    table: mlx.mlx_fast_metal_kernel,
    prep_k: mlx.mlx_fast_metal_kernel,
    gemm2: mlx.mlx_fast_metal_kernel,
    gemm1: mlx.mlx_fast_metal_kernel,
    mid_k: mlx.mlx_fast_metal_kernel,
    fin: mlx.mlx_fast_metal_kernel,
    gemm2s: mlx.mlx_fast_metal_kernel,
    gemm1s: mlx.mlx_fast_metal_kernel,
    table_m: mlx.mlx_fast_metal_kernel,
    gemm2m: mlx.mlx_fast_metal_kernel,
    gemm1m: mlx.mlx_fast_metal_kernel,
    froute: mlx.mlx_fast_metal_kernel,
    froute_m: mlx.mlx_fast_metal_kernel,
    prep_v: mlx.mlx_fast_metal_kernel,
    mid_v: mlx.mlx_fast_metal_kernel,
    fin_v: mlx.mlx_fast_metal_kernel,
    /// the 128-row tiles' kernels (`moeT128`); made only with `opts.t128`
    t128_k: ?T128Kernels = null,

    pub fn init(geo: Geometry, opts: Options) !Prefill {
        if (!geo.admitted() or !opts.valid() or (opts.intr and geo.rate.n % 16 != 0) or (opts.defer_affine and geo.dec.codebook != .mul1) or
            (geo.hidden / BN) % opts.ncb != 0) return error.BadExl3Shape;
        return .{
            .geo = geo,
            .opts = opts,
            .table = ks.newKernel("turboexl3_pf_table", &table_in, &table_out, text_table, header_small, true) orelse return error.KernelCreateFailed,
            .prep_k = ks.newKernel("turboexl3_pf_prep", &prep_in, &prep_out, text_prep, header_small, true) orelse return error.KernelCreateFailed,
            .gemm2 = ks.newKernel("turboexl3_pf_gemm2", &gemm2_in, &gemm2_out, gemm2_text, header_nax, true) orelse return error.KernelCreateFailed,
            .gemm1 = ks.newKernel("turboexl3_pf_gemm1", &gemm1_in, &gemm1_out, gemm1_text, header_nax, true) orelse return error.KernelCreateFailed,
            // a clamped geometry's mids are the clamped variants (`Geometry.limit`)
            .mid_k = (if (geo.limit == 0)
                ks.newKernel("turboexl3_pf_mid", &mid_in, &mid_out, text_mid, header_small, true)
            else
                ks.newKernel("turboexl3_pf_midc", &mid_in, &mid_out, text_midc, header_small, true)) orelse return error.KernelCreateFailed,
            .fin = ks.newKernel("turboexl3_pf_fin", &fin_in, &fin_out, text_finish, header_small, true) orelse return error.KernelCreateFailed,
            .gemm2s = ks.newKernel("turboexl3_pf_gemm2s", &gemm2_in, &gemm2_out, gemm2s_text, header_nax_small, true) orelse return error.KernelCreateFailed,
            .gemm1s = ks.newKernel("turboexl3_pf_gemm1s", &gemm1_in, &gemm1_out, gemm1s_text, header_nax_small, true) orelse return error.KernelCreateFailed,
            .table_m = ks.newKernel("turboexl3_pf_table_m", &table_in, &table_out, text_table_mixed, header_small, true) orelse return error.KernelCreateFailed,
            .gemm2m = ks.newKernel("turboexl3_pf_gemm2m", &gemm2_in, &gemm2_out, gemm2m_text, header_nax_mixed, true) orelse return error.KernelCreateFailed,
            .gemm1m = ks.newKernel("turboexl3_pf_gemm1m", &gemm1_in, &gemm1_out, gemm1m_text, header_nax_mixed, true) orelse return error.KernelCreateFailed,
            .froute = ks.newKernel("turboexl3_pf_froute", &froute_in, &froute_out, text_froute, header_small, true) orelse return error.KernelCreateFailed,
            .froute_m = ks.newKernel("turboexl3_pf_froute_m", &froute_in, &froute_out, text_froute_m, header_small, true) orelse return error.KernelCreateFailed,
            .prep_v = ks.newKernel("turboexl3_pf_prep_v", &prep_in, &prep_out, text_prep_v, header_small, true) orelse return error.KernelCreateFailed,
            .mid_v = (if (geo.limit == 0)
                ks.newKernel("turboexl3_pf_mid_v", &mid_in, &mid_out, text_mid_v, header_small, true)
            else
                ks.newKernel("turboexl3_pf_mid_vc", &mid_in, &mid_out, text_mid_vc, header_small, true)) orelse return error.KernelCreateFailed,
            .fin_v = ks.newKernel("turboexl3_pf_fin_v", &fin_in, &fin_out, text_finish_v, header_small, true) orelse return error.KernelCreateFailed,
            .t128_k = if (!opts.t128) null else try T128Kernels.init(geo.limit),
        };
    }

    pub fn deinit(self: *Prefill) void {
        inline for (.{ self.table, self.prep_k, self.gemm2, self.gemm1, self.mid_k, self.fin, self.gemm2s, self.gemm1s, self.table_m, self.gemm2m, self.gemm1m, self.froute, self.froute_m, self.prep_v, self.mid_v, self.fin_v }) |k| decode.freeKernel(k);
        if (self.t128_k) |k| k.deinit();
    }

    fn decTempl(self: *const Prefill) [3]ks.TemplateArg {
        return .{
            .{ .name = "NHW", .value = .{ .int = @intCast(self.geo.rate.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(self.geo.dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(self.geo.dec.window.bits()) } },
        };
    }

    fn run(k: mlx.mlx_fast_metal_kernel, l: ks.Launch, ins: []const mlx.mlx_array, outs: []mlx.mlx_array, s: mlx.mlx_stream) !void {
        const c = try ks.buildConfig(l);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try ks.applyConfig(k, c, ins, outs, s);
    }

    /// argsort of the flat expert ids, then the M tile table (one threadgroup); `froute`: the one-launch route kernel.
    pub fn route(self: *Prefill, s: mlx.mlx_stream, inds: mlx.mlx_array, S: u32) !Route {
        const g = self.geo;
        const tr = self.opts.tileRows();
        const tmax: u32 = (S + tr - 1) / tr + g.experts;
        const mixed = self.opts.small == .mixed;
        const rows_t: u32 = if (mixed) tmax + self.opts.split * g.experts else tmax;
        const tmpl = [_]ks.TemplateArg{
            .{ .name = "EXP", .value = .{ .int = @intCast(g.experts) } },
            .{ .name = "BMT", .value = .{ .int = @intCast(tr) } },
            .{ .name = "SPL", .value = .{ .int = self.opts.split } },
        };
        if (self.opts.froute) {
            var outs: [2]mlx.mlx_array = undefined;
            try run(if (mixed) self.froute_m else self.froute, .{ .grid = .{ 1024, 1, 1 }, .threadgroup = .{ 1024, 1, 1 }, .out_shapes = &.{ &[_]c_int{@intCast(S)}, &[_]c_int{ @intCast(rows_t), 4 } }, .out_dtypes = &.{ .uint32, .int32 }, .template = &tmpl }, &.{inds}, &outs, s);
            return .{ .order = outs[0], .tiles = outs[1], .tmax = tmax };
        }
        var order = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(order);
        try mlx.check(mlx.mlx_argsort_axis(&order, inds, 0, s));
        var tiles: [1]mlx.mlx_array = undefined;
        try run(if (mixed) self.table_m else self.table, .{ .grid = .{ 1024, 1, 1 }, .threadgroup = .{ 1024, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows_t), 4 }}, .out_dtypes = &.{.int32}, .template = &tmpl }, &.{ inds, order }, &tiles, s);
        return .{ .order = order, .tiles = tiles[0], .tmax = tmax };
    }

    /// xg, xu half [S, hidden] in sorted order, inv u32 [S], and xsg, xsu f32 [S, hidden / 128]: each 128-block's sum of the
    /// rounded inputs (the deferred MUL1 map's C term).
    pub fn prep(self: *Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, inds: mlx.mlx_array, order: mlx.mlx_array, b: Operands, S: u32) ![5]mlx.mlx_array {
        const g = self.geo;
        const H: c_int = @intCast(g.hidden);
        const bpt = bptFor(g.hidden / 128);
        var po: [5]mlx.mlx_array = undefined;
        const sums = [_]c_int{ @intCast(S), @intCast(g.hidden / 128) };
        if (self.opts.aux != 0) {
            const asg: u32 = self.opts.aux;
            const tgs = (S * (g.hidden / 128) + asg - 1) / asg;
            try run(self.prep_v, .{ .grid = .{ tgs * 32 * asg, 1, 1 }, .threadgroup = .{ 32 * asg, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(S), H }, &[_]c_int{ @intCast(S), H }, &[_]c_int{@intCast(S)}, &sums, &sums }, .out_dtypes = &.{ .float16, .float16, .uint32, .float32, .float32 }, .template = &.{
                .{ .name = "HID", .value = .{ .int = H } },
                .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
                .{ .name = "ASG", .value = .{ .int = @intCast(asg) } },
                .{ .name = "XSO", .value = .{ .int = @intFromBool(self.opts.aux_lane_sums) } },
            } }, &.{ x, inds, order, b.gate_suh, b.up_suh }, &po, s);
            return po;
        }
        try run(self.prep_k, .{ .grid = .{ (g.hidden / 128 / bpt) * 32 * bpt, S, 1 }, .threadgroup = .{ 32 * bpt, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(S), H }, &[_]c_int{ @intCast(S), H }, &[_]c_int{@intCast(S)}, &sums, &sums }, .out_dtypes = &.{ .float16, .float16, .uint32, .float32, .float32 }, .template = &.{
            .{ .name = "HID", .value = .{ .int = H } },
            .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
            .{ .name = "BPT", .value = .{ .int = @intCast(bpt) } },
            .{ .name = "ALAY", .value = .{ .int = self.opts.alay } },
        } }, &.{ x, inds, order, b.gate_suh, b.up_suh }, &po, s);
        return po;
    }

    /// A GEMM launch over `tmax` M tiles, K stages `bks` deep and `ncb` output blocks per threadgroup; the template arguments
    /// live in the caller's `tmpl` (the launch only points at them).
    fn gemmCfg(self: *const Prefill, tmpl: *[23]ks.TemplateArg, tmax: u32, kd: u32, nd: u32, nops: u32, bks: u32, ncb: u32) ks.Launch {
        const t = self.decTempl();
        // both bodies: the A layout (`Options.alay`) and the first-fragment-row MMA (`Options.half_rows`)
        const alay_t = ks.TemplateArg{ .name = "ALAY", .value = .{ .int = self.opts.alay } };
        const h16_t = ks.TemplateArg{ .name = "H16", .value = .{ .int = @intFromBool(self.opts.half_rows) } };
        if (self.opts.small == .all) {
            const sgt = self.opts.sgt;
            tmpl.* = .{
                t[0], t[1], t[2],
                .{ .name = "KD", .value = .{ .int = @intCast(kd) } },
                .{ .name = "ND", .value = .{ .int = @intCast(nd) } },
                .{ .name = "NOPS", .value = .{ .int = @intCast(nops) } },
                .{ .name = "SGT", .value = .{ .int = @intCast(sgt) } },
                .{ .name = "DEFER", .value = .{ .int = if (self.opts.defer_affine) 1 else 0 } },
                .{ .name = "NBX", .value = .{ .int = @intCast(kd / 128) } },
                .{ .name = "SINTR", .value = .{ .int = if (self.opts.sintr) 1 else 0 } },
                alay_t, h16_t,
                undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined,
            };
            const items = tmax * nops * (nd / 32);
            return .{ .grid = .{ ((items + sgt - 1) / sgt) * 32 * sgt, 1, 1 }, .threadgroup = .{ 32 * sgt, 1, 1 }, .out_shapes = &.{}, .out_dtypes = &.{}, .template = tmpl[0..12] };
        }
        const threads: u32 = 2 * self.opts.bm + 32 * @as(u32, self.opts.wsp);
        tmpl.* = .{
            t[0], t[1], t[2],
            .{ .name = "KD", .value = .{ .int = @intCast(kd) } },
            .{ .name = "ND", .value = .{ .int = @intCast(nd) } },
            .{ .name = "NOPS", .value = .{ .int = @intCast(nops) } },
            .{ .name = "BMR", .value = .{ .int = @intCast(self.opts.bm) } },
            .{ .name = "DBUF", .value = .{ .int = if (self.opts.dbuf) 1 else 0 } },
            .{ .name = "PROBE", .value = .{ .int = self.opts.probe } },
            .{ .name = "WSP", .value = .{ .int = self.opts.wsp } },
            .{ .name = "MFIRST", .value = .{ .int = if (self.opts.mma_first) 1 else 0 } },
            .{ .name = "UNR", .value = .{ .int = if (self.opts.unroll) 1 else 0 } },
            .{ .name = "SGT", .value = .{ .int = @intCast(threads / 32) } },
            .{ .name = "TMAXS", .value = .{ .int = @intCast(self.opts.split * self.geo.experts) } },
            .{ .name = "FRAGB", .value = .{ .int = if (self.opts.fragb) 1 else 0 } },
            .{ .name = "INTR", .value = .{ .int = if (self.opts.intr) 1 else 0 } },
            .{ .name = "DEFER", .value = .{ .int = if (self.opts.defer_affine) 1 else 0 } },
            .{ .name = "NBX", .value = .{ .int = @intCast(kd / 128) } },
            .{ .name = "BKS", .value = .{ .int = @intCast(bks) } },
            .{ .name = "NCB", .value = .{ .int = @intCast(ncb) } },
            // the mixed launch's register-path half (the shared-decode body does not read it)
            .{ .name = "SINTR", .value = .{ .int = if (self.opts.sintr) 1 else 0 } },
            alay_t,
            h16_t,
        };
        // shared-decode threadgroups: ncb output blocks of one (M tile, op) each
        const big_tgs = tmax * nops * (nd / BN / ncb);
        if (self.opts.small == .mixed) {
            const nsg = threads / 32;
            const small_tgs = (self.opts.split * self.geo.experts * nops * (nd / 32) + nsg - 1) / nsg;
            return .{ .grid = .{ (big_tgs + small_tgs) * threads, 1, 1 }, .threadgroup = .{ threads, 1, 1 }, .out_shapes = &.{}, .out_dtypes = &.{}, .template = tmpl };
        }
        return .{ .grid = .{ big_tgs * threads, 1, 1 }, .threadgroup = .{ threads, 1, 1 }, .out_shapes = &.{}, .out_dtypes = &.{}, .template = tmpl };
    }

    /// the gate and up inner products, half [S, inter] each (sorted order)
    pub fn gemmGU(self: *Prefill, s: mlx.mlx_stream, r: Route, xg: mlx.mlx_array, xu: mlx.mlx_array, xsg: mlx.mlx_array, xsu: mlx.mlx_array, b: Operands, S: u32) ![2]mlx.mlx_array {
        const g = self.geo;
        var tmpl: [23]ks.TemplateArg = undefined;
        var l = self.gemmCfg(&tmpl, r.tmax, g.hidden, g.inter, 2, BK, 1);
        const I: c_int = @intCast(g.inter);
        const shape = [_]c_int{ @intCast(S), I };
        l.out_shapes = &.{ &shape, &shape };
        l.out_dtypes = &.{ .float16, .float16 };
        var z: [2]mlx.mlx_array = undefined;
        try run(switch (self.opts.small) { .all => self.gemm2s, .mixed => self.gemm2m, .off => self.gemm2 }, l, &.{ r.tiles, xg, xu, b.gate_t, b.up_t, xsg, xsu }, &z, s);
        return z;
    }

    /// gate|up finish, SwiGLU and the down input: xd half [S, inter] and xsd f32 [S, inter / 128] (its per-block sums)
    pub fn mid(self: *Prefill, s: mlx.mlx_stream, zg: mlx.mlx_array, zu: mlx.mlx_array, inds: mlx.mlx_array, order: mlx.mlx_array, b: Operands, S: u32) ![2]mlx.mlx_array {
        const g = self.geo;
        const I: c_int = @intCast(g.inter);
        const bpt = bptFor(g.inter / 128);
        var xd: [2]mlx.mlx_array = undefined;
        if (self.opts.aux != 0) {
            const asg: u32 = self.opts.aux;
            const tgs = (S * (g.inter / 128) + asg - 1) / asg;
            var tv: [4]ks.TemplateArg = undefined;
            try run(self.mid_v, .{ .grid = .{ tgs * 32 * asg, 1, 1 }, .threadgroup = .{ 32 * asg, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(S), I }, &[_]c_int{ @intCast(S), @intCast(g.inter / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = self.midTempl(&tv, .{
                .{ .name = "INTER", .value = .{ .int = I } },
                .{ .name = "ASG", .value = .{ .int = @intCast(asg) } },
                .{ .name = "XSO", .value = .{ .int = @intFromBool(self.opts.aux_lane_sums) } },
            }) }, &.{ zg, zu, inds, order, b.gate_svh, b.up_svh, b.down_suh }, &xd, s);
            return xd;
        }
        var tm: [4]ks.TemplateArg = undefined;
        try run(self.mid_k, .{ .grid = .{ (g.inter / 128 / bpt) * 32 * bpt, S, 1 }, .threadgroup = .{ 32 * bpt, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(S), I }, &[_]c_int{ @intCast(S), @intCast(g.inter / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = self.midTempl(&tm, .{
            .{ .name = "INTER", .value = .{ .int = I } },
            .{ .name = "BPT", .value = .{ .int = @intCast(bpt) } },
            .{ .name = "ALAY", .value = .{ .int = self.opts.alay } },
        }) }, &.{ zg, zu, inds, order, b.gate_svh, b.up_svh, b.down_suh }, &xd, s);
        return xd;
    }

    /// A mid launch's template constants: its three, then the clamp's LIMIT where the geometry clamps (an unclamped mid takes
    /// exactly its three). They live in the caller's `buf` (the launch only points at them).
    fn midTempl(self: *const Prefill, buf: *[4]ks.TemplateArg, own: [3]ks.TemplateArg) []const ks.TemplateArg {
        buf.* = .{ own[0], own[1], own[2], .{ .name = "LIMIT", .value = .{ .int = @intCast(self.geo.limit) } } };
        const n: usize = if (self.geo.limit != 0) 4 else 3;
        return buf[0..n];
    }

    /// the down inner product, half [S, hidden] (sorted order); `Options.bk128` / `ncb` shape its shared-decode body
    pub fn gemmD(self: *Prefill, s: mlx.mlx_stream, r: Route, xd: mlx.mlx_array, xsd: mlx.mlx_array, b: Operands, S: u32) !mlx.mlx_array {
        const g = self.geo;
        var tmpl: [23]ks.TemplateArg = undefined;
        var l = self.gemmCfg(&tmpl, r.tmax, g.inter, g.hidden, 1, self.opts.downStage(), self.opts.ncb);
        const shape = [_]c_int{ @intCast(S), @intCast(g.hidden) };
        l.out_shapes = &.{&shape};
        l.out_dtypes = &.{.float16};
        var z: [1]mlx.mlx_array = undefined;
        try run(switch (self.opts.small) { .all => self.gemm1s, .mixed => self.gemm1m, .off => self.gemm1 }, l, &.{ r.tiles, xd, b.down_t, xsd }, &z, s);
        return z[0];
    }

    /// the down finish and the slot reduce: y [rows, hidden] in `out_dtype`
    pub fn finish(self: *Prefill, s: mlx.mlx_stream, zd: mlx.mlx_array, inds: mlx.mlx_array, inv: mlx.mlx_array, scores: mlx.mlx_array, b: Operands, rows: usize, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
        const g = self.geo;
        const H: c_int = @intCast(g.hidden);
        const bpt = bptFor(g.hidden / 128);
        var y: [1]mlx.mlx_array = undefined;
        if (self.opts.aux != 0) {
            const asg: u32 = self.opts.aux;
            const items: u32 = @intCast(rows * (g.hidden / 128));
            const tgs = (items + asg - 1) / asg;
            try run(self.fin_v, .{ .grid = .{ tgs * 32 * asg, 1, 1 }, .threadgroup = .{ 32 * asg, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{out_dtype}, .template = &.{
                .{ .name = "HID", .value = .{ .int = H } },
                .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
                .{ .name = "ASG", .value = .{ .int = @intCast(asg) } },
                .{ .name = "OUT_T", .value = .{ .dtype = out_dtype } },
            } }, &.{ zd, inds, inv, scores, b.down_svh }, &y, s);
            return y[0];
        }
        try run(self.fin, .{ .grid = .{ (g.hidden / 128 / bpt) * 32 * bpt, @intCast(rows), 1 }, .threadgroup = .{ 32 * bpt, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{out_dtype}, .template = &.{
            .{ .name = "HID", .value = .{ .int = H } },
            .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
            .{ .name = "BPT", .value = .{ .int = @intCast(bpt) } },
            .{ .name = "OUT_T", .value = .{ .dtype = out_dtype } },
        } }, &.{ zd, inds, inv, scores, b.down_svh }, &y, s);
        return y[0];
    }

    /// y [rows, hidden] in `out_dtype` for x [rows, hidden], inds / scores [rows * topk] (the host's dtypes).
    pub fn moe(self: *Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, b: Operands, rows: usize, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
        const S: u32 = @intCast(rows * self.geo.topk);
        const r = try self.route(s, inds, S);
        defer r.deinit();
        const po = try self.prep(s, x, inds, r.order, b, S);
        defer for (po) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const zgu = try self.gemmGU(s, r, po[0], po[1], po[3], po[4], b, S);
        defer for (zgu) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const xd = try self.mid(s, zgu[0], zgu[1], inds, r.order, b, S);
        defer for (xd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const zd = try self.gemmD(s, r, xd[0], xd[1], b, S);
        defer _ = mlx.mlx_array_free(zd);
        return self.finish(s, zd, inds, po[2], scores, b, rows, out_dtype);
    }

    /// The 128-row tiles' route: `froute`'s count and scatter (the same `order`), then the 128-row tiles, the mixed table over
    /// the remainders and each expert's first remainder position (`text_froute_t128`), one launch.
    pub fn routeT128(self: *Prefill, s: mlx.mlx_stream, inds: mlx.mlx_array, S: u32) !RouteT128 {
        const k = self.t128_k orelse return error.Exl3EngineNotImplemented;
        const g = self.geo;
        const tr = self.opts.tileRows();
        const tmax = t128MixedRows(S, g.experts, tr, self.opts.split, t128_cut, true);
        const rows_t: u32 = tmax + self.opts.split * g.experts;
        const tq = t128TileRows(S, g.experts, t128_cut, true);
        const tmpl = [_]ks.TemplateArg{
            .{ .name = "EXP", .value = .{ .int = @intCast(g.experts) } },
            .{ .name = "BMT", .value = .{ .int = @intCast(tr) } },
            .{ .name = "SPL", .value = .{ .int = self.opts.split } },
            .{ .name = "CUT", .value = .{ .int = t128_cut } },
            .{ .name = "T128", .value = .{ .int = 1 } },
        };
        var outs: [4]mlx.mlx_array = undefined;
        try run(k.route, .{ .grid = .{ 1024, 1, 1 }, .threadgroup = .{ 1024, 1, 1 }, .out_shapes = &.{ &[_]c_int{@intCast(S)}, &[_]c_int{ @intCast(rows_t), 4 }, &[_]c_int{ @intCast(tq), 4 }, &[_]c_int{@intCast(g.experts)} }, .out_dtypes = &.{ .uint32, .int32, .int32, .int32 }, .template = &tmpl }, &.{inds}, &outs, s);
        return .{ .order = outs[0], .tiles = outs[1], .tiles128 = outs[2], .xbase = outs[3], .tmax = tmax, .tq = tq };
    }

    /// The 128-row tiles' launch: `gemmCfg`'s shared-decode body at bm 128 (8 simdgroups, the same stage and K loop), over
    /// `tq` table rows
    fn t128Cfg(self: *const Prefill, tmpl: *[23]ks.TemplateArg, tq: u32, kd: u32, nd: u32, nops: u32, bks: u32, ncb: u32) ks.Launch {
        var p = self.*;
        p.opts.bm = 128;
        p.opts.small = .off;
        return p.gemmCfg(tmpl, tq, kd, nd, nops, bks, ncb);
    }

    /// A GEMM's outputs ([S, nd] f16 per op, `ZT128`): the mixed launch's over the remainder table, then the bm-128 launch's
    /// over the 128-row tiles (so the register-path tail at the mixed launch's end overlaps the 128-row tiles' start), each
    /// an output of its own launch: no data between the two, so no barrier between them, and MLX orders every reader after
    /// both. `ins_m` / `ins_t` are the launches' inputs.
    fn t128Gemm(self: *Prefill, s: mlx.mlx_stream, rt: RouteT128, k_m: mlx.mlx_fast_metal_kernel, k_t: mlx.mlx_fast_metal_kernel, nops: u32, kd: u32, nd: u32, bks: u32, ncb: u32, ins_m: []const mlx.mlx_array, ins_t: []const mlx.mlx_array, S: u32) !ZT128 {
        const shape = [_]c_int{ @intCast(S), @intCast(nd) };
        const shapes = [_][]const c_int{ &shape, &shape };
        const dtypes = [_]mlx.mlx_dtype{ .float16, .float16 };
        var x: ZT128 = .{ .m = undefined, .t = undefined, .n = @intCast(nops) };
        var tmpl: [23]ks.TemplateArg = undefined;
        var lm = self.gemmCfg(&tmpl, rt.tmax, kd, nd, nops, bks, ncb);
        lm.out_shapes = shapes[0..nops];
        lm.out_dtypes = dtypes[0..nops];
        try run(k_m, lm, ins_m, x.m[0..nops], s);
        errdefer for (x.m[0..nops]) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var tmpl_t: [23]ks.TemplateArg = undefined;
        var lt = self.t128Cfg(&tmpl_t, rt.tq, kd, nd, nops, bks, ncb);
        lt.out_shapes = shapes[0..nops];
        lt.out_dtypes = dtypes[0..nops];
        try run(k_t, lt, ins_t, x.t[0..nops], s);
        return x;
    }

    /// The gate and up inner products, half [S, inter] each (sorted order, `ZT128`): `gemmGU`'s per-output MMAs from the
    /// 128-row tiles and the mixed tiles.
    pub fn gemmGUT128(self: *Prefill, s: mlx.mlx_stream, rt: RouteT128, xg: mlx.mlx_array, xu: mlx.mlx_array, xsg: mlx.mlx_array, xsu: mlx.mlx_array, b: Operands, S: u32) !ZT128 {
        const k = self.t128_k orelse return error.Exl3EngineNotImplemented;
        const g = self.geo;
        return self.t128Gemm(s, rt, k.gu_mixed, k.gu_t128, 2, g.hidden, g.inter, BK, 1, &.{ rt.tiles, xg, xu, b.gate_t, b.up_t, xsg, xsu }, &.{ rt.tiles128, xg, xu, b.gate_t, b.up_t, xsg, xsu }, S);
    }

    /// `mid`'s strided launch with `text_mid_t128` (each position's rows from the launch that computed them)
    pub fn midT128(self: *Prefill, s: mlx.mlx_stream, zt: ZT128, rt: RouteT128, inds: mlx.mlx_array, b: Operands, S: u32) ![2]mlx.mlx_array {
        const k = self.t128_k orelse return error.Exl3EngineNotImplemented;
        const g = self.geo;
        const I: c_int = @intCast(g.inter);
        const bpt = bptFor(g.inter / 128);
        var xd: [2]mlx.mlx_array = undefined;
        var tm: [4]ks.TemplateArg = undefined;
        try run(k.mid, .{ .grid = .{ (g.inter / 128 / bpt) * 32 * bpt, S, 1 }, .threadgroup = .{ 32 * bpt, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(S), I }, &[_]c_int{ @intCast(S), @intCast(g.inter / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = self.midTempl(&tm, .{
            .{ .name = "INTER", .value = .{ .int = I } },
            .{ .name = "BPT", .value = .{ .int = @intCast(bpt) } },
            .{ .name = "ALAY", .value = .{ .int = self.opts.alay } },
        }) }, &.{ zt.m[0], zt.m[1], inds, rt.order, b.gate_svh, b.up_svh, b.down_suh, zt.t[0], zt.t[1], rt.xbase }, &xd, s);
        return xd;
    }

    /// The down inner product, half [S, hidden] (sorted order, `ZT128` with one op), as `gemmGUT128`
    pub fn gemmDT128(self: *Prefill, s: mlx.mlx_stream, rt: RouteT128, xd: mlx.mlx_array, xsd: mlx.mlx_array, b: Operands, S: u32) !ZT128 {
        const k = self.t128_k orelse return error.Exl3EngineNotImplemented;
        const g = self.geo;
        return self.t128Gemm(s, rt, k.dn_mixed, k.dn_t128, 1, g.inter, g.hidden, self.opts.downStage(), self.opts.ncb, &.{ rt.tiles, xd, b.down_t, xsd }, &.{ rt.tiles128, xd, b.down_t, xsd }, S);
    }

    /// `finish`'s strided launch with `text_finish_t128` (each slot's row from the launch that computed it)
    pub fn finishT128(self: *Prefill, s: mlx.mlx_stream, zt: ZT128, xbase: mlx.mlx_array, inds: mlx.mlx_array, inv: mlx.mlx_array, scores: mlx.mlx_array, b: Operands, rows: usize, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
        const k = self.t128_k orelse return error.Exl3EngineNotImplemented;
        const g = self.geo;
        const H: c_int = @intCast(g.hidden);
        const bpt = bptFor(g.hidden / 128);
        var y: [1]mlx.mlx_array = undefined;
        try run(k.fin, .{ .grid = .{ (g.hidden / 128 / bpt) * 32 * bpt, @intCast(rows), 1 }, .threadgroup = .{ 32 * bpt, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{out_dtype}, .template = &.{
            .{ .name = "HID", .value = .{ .int = H } },
            .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
            .{ .name = "BPT", .value = .{ .int = @intCast(bpt) } },
            .{ .name = "OUT_T", .value = .{ .dtype = out_dtype } },
        } }, &.{ zt.m[0], inds, inv, scores, b.down_svh, zt.t[0], xbase }, &y, s);
        return y[0];
    }

    /// `moe`'s y through the 128-row tiles (`routeT128`, `prep`, `gemmGUT128`, `midT128`, `gemmDT128`, `finishT128`): its
    /// bytes. Refused (error.Exl3EngineNotImplemented, nothing launched) without `opts.t128`.
    pub fn moeT128(self: *Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, b: Operands, rows: usize, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
        if (self.t128_k == null) return error.Exl3EngineNotImplemented;
        const S: u32 = @intCast(rows * self.geo.topk);
        const rt = try self.routeT128(s, inds, S);
        defer rt.deinit();
        const po = try self.prep(s, x, inds, rt.order, b, S);
        defer for (po) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const zgu = try self.gemmGUT128(s, rt, po[0], po[1], po[3], po[4], b, S);
        defer zgu.deinit();
        const xd = try self.midT128(s, zgu, rt, inds, b, S);
        defer for (xd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const zd = try self.gemmDT128(s, rt, xd[0], xd[1], b, S);
        defer zd.deinit();
        return self.finishT128(s, zd, rt.xbase, inds, po[2], scores, b, rows, out_dtype);
    }
};

test "turboexl3 prefill texts: the NAX header carries steel's tiles and turboexl3's decoder; the GEMM's second-op names resolve" {
    const pin = comptime @import("test_support.zig").pins(@embedFile("kernels/test/pins/prefill_texts.metal"));
    const t = std.testing;
    try t.expect(std.mem.indexOf(u8, header_nax, pin[0]) != null);
    try t.expect(std.mem.indexOf(u8, header_nax, "exl3_gen_cw") != null);
    try t.expect(std.mem.indexOf(u8, gemm2_text, "XSECOND") == null);
    try t.expect(std.mem.indexOf(u8, gemm2_text, pin[1]) != null);
    try t.expect(std.mem.indexOf(u8, gemm1_text, pin[2]) != null);
    try t.expect(bptFor(20) == 5 and bptFor(5) == 5 and bptFor(7) == 7 and bptFor(16) == 8);
}

test "turboexl3 prefill down options: bk128 doubles the stage (32 KB double-buffered in fragment order); ncb is 1, 2 or 4" {
    const t = std.testing;
    const served: Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .defer_affine = true };
    try t.expect(served.stageBytes(served.downStage()) == 16384);
    var o = served;
    o.bk128 = true;
    try t.expect(o.valid() and o.stageBytes(o.downStage()) == 32768 and o.stageBytes(BK) == 16384);
    o.ncb = 4;
    try t.expect(o.valid());
    o.ncb = 3;
    try t.expect(!o.valid());
    try t.expect(!(Options{ .dbuf = true, .bk128 = true }).valid()); // the padded B^T sub-stages: 36 KB
    try t.expect(!(Options{ .dbuf = true, .fragb = true, .bk128 = true, .probe = 16 }).valid());
    try t.expect((Options{ .bk128 = true }).valid() and (Options{ .bk128 = true }).stageBytes(128) == 18432); // single-buffered
    try t.expect((Options{ .fragb = true, .bk128 = true, .ncb = 2 }).valid());
    try t.expect(!(Options{ .small = .all, .ncb = 2 }).valid() and !(Options{ .small = .all, .fragb = true, .bk128 = true }).valid());
}

test "turboexl3 prefill alay: a lane's A fragment elements are one aligned 16-byte (32-deep step) or 8-byte (16-deep slice) run; the writers' slot is alayPos / alayPos16" {
    const pin = comptime @import("test_support.zig").pins(@embedFile("kernels/test/pins/prefill_alay.metal"));
    const t = std.testing;
    // each a permutation of a 32-block; alayPos16 keeps each 16-block's elements in it
    var seen: [32]bool = @splat(false);
    var seen16: [32]bool = @splat(false);
    for (0..32) |j_| {
        const j: u32 = @intCast(j_);
        const p = alayPos(j);
        const p16 = alayPos16(j);
        try t.expect(p < 32 and !seen[p] and p16 < 32 and !seen16[p16] and p16 / 16 == j / 16);
        seen[p] = true;
        seen16[p16] = true;
    }
    // the fragment rule both GEMM bodies share (exl3pf_afrag; the register path's lane-plan test above): NAX lane L, q = 2 b3 +
    // b0, holds at element e of its fragment row for 16-deep slice kk the input k 16 kk + 2q + {0, 1, 8, 9}[e]. Stored at
    // alayPos that element is half 8q + 4 kk + e of the 32-block: the lane's eight elements of both slices are halfs 8q .. 8q + 7,
    // one vec<half, 8> at the row pointer's lane offset 8q (exl3pf_afrag2; 16q bytes). Stored at alayPos16 it is half
    // 16 kk + 4q + e: the slice's four are one half4 at 16 kk + 4q (exl3pf_afrag4; 8q bytes). Rows of an admitted width (a
    // multiple of 128 halfs) and the blocks (64 / 32 bytes) keep both aligned.
    const kof = [_]u32{ 0, 1, 8, 9 };
    for (0..32) |l_| {
        const l: u32 = @intCast(l_);
        const q = ((l >> 2) & 2) | (l & 1);
        for (0..2) |kk_| for (0..4) |e_| {
            const kk: u32 = @intCast(kk_);
            const e: u32 = @intCast(e_);
            const k = 16 * kk + 2 * q + kof[e];
            try t.expectEqual(8 * q + 4 * kk + e, alayPos(k));
            try t.expectEqual(16 * kk + 4 * q + e, alayPos16(k));
        };
    }
    // prep and mid write lane L's element L of each 32-block at turboexl3_alay(L) / turboexl3_alay16(L): the expressions of
    // alayPos / alayPos16
    try t.expect(std.mem.indexOf(u8, header_small, pin[0]) != null);
    try t.expect(std.mem.indexOf(u8, header_small, pin[1]) != null);
    for ([_][]const u8{ text_prep, text_mid }) |txt| try t.expect(std.mem.indexOf(u8, txt, pin[2]) != null);
    // both GEMM bodies (and so the mixed launch) read the layout and the first-fragment-row switch
    for ([_][]const u8{ gemm2m_text, gemm1m_text, gemm2s_text, gemm1s_text, gemm2_text, gemm1_text }) |txt| {
        try t.expect(std.mem.indexOf(u8, txt, pin[3]) != null);
        try t.expect(std.mem.indexOf(u8, txt, "H16") != null);
    }
}

test "turboexl3 prefill alay / half_rows options: fragment order where the shared-decode body runs, the strided prep and mid" {
    const t = std.testing;
    const served: Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .sintr = true, .defer_affine = true, .ncb = 4, .froute = true };
    try t.expect(served.valid());
    for ([_]u8{ 1, 2 }) |mode| {
        var o = served;
        o.alay = mode;
        o.half_rows = true;
        try t.expect(o.valid());
        o.bm = 128;
        try t.expect(o.valid());
        // the padded bodies load A in natural order (and multiply whole fragments)
        try t.expect(!(Options{ .small = .mixed, .dbuf = true, .alay = mode }).valid());
        // the all-register path needs no stage layout
        try t.expect((Options{ .small = .all, .alay = mode, .half_rows = true }).valid());
        // the vector-layout prep / mid write natural order
        var v = served;
        v.alay = mode;
        v.aux = 8;
        try t.expect(!v.valid());
    }
    try t.expect(!(Options{ .dbuf = true, .unroll = true, .half_rows = true }).valid());
    var o3 = served;
    o3.alay = 3;
    try t.expect(!o3.valid());
    var w = served;
    w.aux = 8;
    w.half_rows = true;
    try t.expect(w.valid());
}

test "turboexl3 prefill stage plan: a sub-stage fills its own BSZH halfs; the carried double buffer feeds every block its sub-stages in K order" {
    const t = std.testing;
    // one 64-deep sub-stage (the text's plan): decoding simdgroup d of DEC takes tiles j = d + DEC jj (kt = j >> 2, ntl = j & 3);
    // fragment order: lane L writes (ntl 4 + kt) 256 + 8 lnx(L), and the MMA's lane L reads tile (tn / 16 + nn, 2 i + kk) at
    // ((tn / 16 + nn) 4 + 2 i + kk) 256 + 8 L; padded: four half2 at d, d + 8, d + 8 LDB, d + 8 LDB + 8, d = (ntl 16 + c) LDB +
    // kt 16 + 2 q. Either way the sub-stage stays inside BSZH halfs, so a 128-deep buffer holds two side by side.
    for ([_]u32{ 2, 4, 8, 16 }) |dec| {
        var writer: [512]?[3]u32 = @splat(null); // fragment slot (8 halfs) -> (ntl, kt, lnx) of its writer
        var padded_end: u32 = 0;
        for (0..dec) |d| for (0..16 / dec) |jj| {
            const j: u32 = @intCast(d + dec * jj);
            const kt = j >> 2;
            const ntl = j & 3;
            for (0..32) |l_| {
                const l: u32 = @intCast(l_);
                const lnx = (((l >> 4) & 1) << 4) | (((l >> 1) & 1) << 3) | (((l >> 3) & 1) << 2) | (((l >> 2) & 1) << 1) | (l & 1);
                const slot = ((ntl * 4 + kt) * 256 + lnx * 8) / 8;
                try t.expect(writer[slot] == null);
                writer[slot] = .{ ntl, kt, lnx };
                padded_end = @max(padded_end, (ntl * 16 + (l >> 2)) * (BK + 8) + kt * 16 + 2 * (l & 3) + 8 * (BK + 8) + 8 + 2);
            }
        };
        for (writer) |w| try t.expect(w != null);
        try t.expect(padded_end <= BN * (BK + 8));
        for ([_]u32{ 0, 32 }) |tn| for (0..2) |nn| for (0..2) |i| for (0..2) |kk| for (0..32) |l| {
            const ntl: u32 = @intCast(tn / 16 + nn);
            const kt: u32 = @intCast(2 * i + kk);
            try t.expectEqual([3]u32{ ntl, kt, @intCast(l) }, writer[((ntl * 4 + kt) * 256 + 8 * l) / 8].?);
        };
    }
    // the double-buffered sequence: NCB blocks of STAGES stages of SUBS sub-stages; the prologue decodes block 0's sub-stages
    // 0 .. SUBS - 1 into buffer 0; step f = (cb, s) multiplies buffer f & 1's sub-stages SUBS s + h and decodes the next stage's
    // (s + 1 of cb, or 0 of cb + 1) into buffer (f + 1) & 1
    for ([_]u32{ 1, 2 }) |subs| for ([_]u32{ 1, 2, 4 }) |ncb| for ([_]u32{ 640, 2560 }) |kd| {
        const stages = kd / BK / subs;
        var buf: [2][2]?[2]u32 = .{ .{ null, null }, .{ null, null } };
        for (0..subs) |h| buf[0][h] = .{ 0, @intCast(h) };
        var next_k: [4]u32 = @splat(0); // per block: the next 64-deep sub-stage its accumulator must take
        for (0..ncb) |cb_| for (0..stages) |s_| {
            const cb: u32 = @intCast(cb_);
            const s: u32 = @intCast(s_);
            const f = cb * stages + s;
            const last = s + 1 == stages;
            for (0..subs) |h_| {
                const h: u32 = @intCast(h_);
                if (!last or cb + 1 < ncb) buf[(f + 1) & 1][h] = .{ if (last) cb + 1 else cb, subs * (if (last) 0 else s + 1) + h };
                const got = buf[f & 1][h].?;
                try t.expectEqual([2]u32{ cb, subs * s + h }, got);
                try t.expectEqual(next_k[cb], got[1]);
                next_k[cb] += 1;
            }
        };
        for (next_k[0..ncb]) |k| try t.expectEqual(kd / BK, k);
    };
}

test "turboexl3 prefill small: lane L decoding the trellis plan of L' = (b4,b2,b1,b3,b0) holds its NAX B^T fragment elements in order" {
    var perm: [format.TILE_VALUES]usize = undefined;
    format.tensorCorePerm(&perm); // code position p -> row-major tile index (row = input k, column = output n)
    const kof = [_]usize{ 0, 1, 8, 9 };
    for (0..32) |l| {
        const lp = smallLanePlan(@intCast(l));
        // BaseNAXFrag::get_coord(idx): row fm + (idx >> 2) * 8, column fn + idx % 4
        const qid = l >> 2;
        const fm = (qid & 4) | ((l >> 1) & 3);
        const fnc = ((qid & 2) | (l & 1)) * 4;
        for (0..8) |idx| {
            const n_frag = fm + (idx >> 2) * 8; // B^T fragment row = output n
            const slot = fnc + idx % 4; // fragment column = k-slot
            const k_true = 2 * (slot >> 2) + kof[slot & 3];
            const rm = perm[8 * lp + idx]; // weight idx of trellis lane lp
            try std.testing.expectEqual(k_true, rm / 16);
            try std.testing.expectEqual(n_frag, rm % 16);
        }
    }
    // L -> L' is a permutation of the 32 lanes
    var seen: [32]bool = @splat(false);
    for (0..32) |l| seen[smallLanePlan(@intCast(l))] = true;
    for (seen) |b| try std.testing.expect(b);
}

test "turboexl3 prefill small sintr: at every register-path lane the integer-rate reader picks the general reader's words, halfword and extract offset (n % 16 == 0)" {
    // lane.zig's Metal readers at lane L' = smallLanePlan(L), K = 1..8: the same start bit gives the same word qb and halfword a
    // (so the same IntWords) and lp = 8 b + K mod 8; window m then selects the same h[j] and extracts at the same offset
    var prng = std.Random.DefaultPrng.init(0x5147);
    const rnd = prng.random();
    var words: [format.Rate.max_n]u16 = undefined;
    var want: [format.TILE_VALUES]u16 = undefined;
    var gi: [8]u16 = undefined;
    var gg: [8]u16 = undefined;
    var k: u32 = 1;
    while (k <= 8) : (k += 1) {
        const rate = format.Rate.fromK(k);
        const n = rate.n;
        for (0..32) |l| {
            const lpl = smallLanePlan(@intCast(l));
            const ip = lane.intPlan(rate, lpl);
            const gp = lane.genPlan(rate, lpl);
            try std.testing.expectEqual(gp.qb, ip.qb);
            try std.testing.expectEqual(gp.a, ip.a);
            try std.testing.expectEqual(gp.lp, 8 * ip.b + k % 8);
            for (0..8) |m_| {
                const m: u32 = @intCast(m_);
                // exl3_int_cw: CM = K mod 8 + K m, h[(CM >> 4) + carry] with carry = b and (CM & 15) >= 8, offset (8 b + CM) & 15
                const cm = k % 8 + k * m;
                const ji = (cm >> 4) + @intFromBool(ip.b != 0 and (cm & 15) >= 8);
                const ri = (8 * ip.b + cm) & 15;
                // exl3_gen_cw: C_m = ((m + 1) N >> 4) - (N >> 4), pos = lp + C_m, h[pos >> 4] (J or J + 1), offset pos & 15
                const pos = gp.lp + ((((m + 1) * n) >> 4) - (n >> 4));
                try std.testing.expectEqual(pos >> 4, ji);
                try std.testing.expectEqual(pos & 15, ri);
                try std.testing.expect(ji <= 5); // an IntWords index: h[i] holds halfwords a + i, a + i + 1 of the four words
            }
        }
        for (0..4) |_| {
            for (words[0..n]) |*w| w.* = rnd.int(u16);
            format.unpackTile(words[0..n], rate, &want);
            for (0..32) |l| {
                const lpl = smallLanePlan(@intCast(l));
                lane.laneCodewordsInt(words[0..n], rate, lpl, &gi);
                lane.laneCodewordsGen(words[0..n], rate, lpl, &gg);
                try std.testing.expectEqualSlices(u16, &gg, &gi);
                try std.testing.expectEqualSlices(u16, want[8 * lpl ..][0..8], &gi);
            }
        }
    }
}

/// One kernel as a standalone Metal translation unit (the CPU-side compile check, TURBOEXL3_METAL_DUMP=1): the header, the
/// template ints and dtypes as file-scope constants, and a [[kernel]] whose arguments mirror what MLX binds (inputs in order,
/// `<name>_shape` where the source reads it, then outputs, then the thread attributes the sources use; `params`: the list up to
/// those, a line of the dump test's data file). Printed between @@METAL_BEGIN / @@METAL_END markers; the box splits them
/// and runs `xcrun metal` on each without touching the GPU.
pub fn printMetal(name: []const u8, header: []const u8, consts: []const u8, params: []const u8, body: []const u8) void {
    std.debug.print("@@METAL_BEGIN {s}\n{s}\n{s}\n{s}{s}{s}{s}\n{s}\n{s}\n@@METAL_END\n", .{ name, header, consts, @embedFile("kernels/test/dump/kernel.metal"), name, params, @embedFile("kernels/test/dump/attributes.metal"), body, @embedFile("kernels/test/dump/close.metal") });
}

test "turboexl3 prefill metal sources: every kernel as a standalone translation unit (TURBOEXL3_METAL_DUMP=1)" {
    const pin = comptime @import("test_support.zig").pins(@embedFile("kernels/test/dump/prefill.metal"));
    _ = std.c.getenv("TURBOEXL3_METAL_DUMP") orelse return error.SkipZigTest;
    const gemm_c = pin[0];
    // the register path through the integer-rate reader (j = Options.sintr), with the deferred MUL1 map and without it
    const gemm_j = pin[1];
    const gemm_j0 = pin[2];
    // the A layouts (l = Options.alay 1, l2 = alay 2) and the first-fragment-row MMA (e = Options.half_rows), each alone and
    // with e, on the served gate|up launch (bm 64, 4 simdgroups) and at bm 128; with the A-load probe (p4: the layout's zero
    // fill)
    const gemm_jl = pin[3];
    const gemm_je = pin[4];
    const gemm_jle = pin[5];
    const gemm_jle128 = pin[6];
    const gemm_jle_p4 = pin[7];
    const gemm_jl2e = pin[8];
    const g2 = pin[9];
    const g1 = pin[10];
    printMetal("turboexl3_pf_gemm2", header_nax, gemm_c, g2, gemm2_text);
    printMetal("turboexl3_pf_gemm1", header_nax, gemm_c, g1, gemm1_text);
    printMetal("turboexl3_pf_gemm2s", header_nax_small, gemm_c, g2, gemm2s_text);
    printMetal("turboexl3_pf_gemm2m", header_nax_mixed, gemm_c, g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm1m", header_nax_mixed, gemm_c, g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm2s_j", header_nax_small, gemm_j, g2, gemm2s_text);
    printMetal("turboexl3_pf_gemm2s_j0", header_nax_small, gemm_j0, g2, gemm2s_text);
    printMetal("turboexl3_pf_gemm2m_j", header_nax_mixed, gemm_j, g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm2m_jl", header_nax_mixed, gemm_jl, g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm2m_je", header_nax_mixed, gemm_je, g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm2m_jle", header_nax_mixed, gemm_jle, g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm2m_jle_p4", header_nax_mixed, gemm_jle_p4, g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm2_le128", header_nax, gemm_jle128, g2, gemm2_text);
    printMetal("turboexl3_pf_gemm2s_jle", header_nax_small, gemm_jle, g2, gemm2s_text);
    printMetal("turboexl3_pf_gemm2m_jl2e", header_nax_mixed, gemm_jl2e, g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm2s_jl2e", header_nax_small, gemm_jl2e, g2, gemm2s_text);
    // the down GEMM at Qwen's geometry (K 640, N 2560): the served stage (bm 64, fragment order), 128-deep stages (k), several
    // column blocks per threadgroup (n<c>), and the single-buffered padded body at bm 128 (both sub-stage and block loops there);
    // the all-register down GEMM and the served launch (n4) with the register path's integer-rate reader (j)
    const down_c = pin[11];
    const served = down_c ++ pin[12];
    const served_j = pin[13];
    const served_jle = pin[14];
    printMetal("turboexl3_pf_gemm1m_n4_jle", header_nax_mixed, served_jle ++ pin[15], g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1s_jle", header_nax_small, served_jle ++ pin[16], g1, gemm1s_text);
    const served_jl2e = pin[17];
    printMetal("turboexl3_pf_gemm1m_n4_jl2e", header_nax_mixed, served_jl2e ++ pin[15], g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1s", header_nax_small, served ++ pin[16], g1, gemm1s_text);
    printMetal("turboexl3_pf_gemm1s_j", header_nax_small, served_j ++ pin[16], g1, gemm1s_text);
    printMetal("turboexl3_pf_gemm1m_n4_j", header_nax_mixed, served_j ++ pin[15], g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1m_down", header_nax_mixed, served ++ pin[16], g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1m_k", header_nax_mixed, served ++ pin[18], g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1m_n2", header_nax_mixed, served ++ pin[19], g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1m_n4", header_nax_mixed, served ++ pin[15], g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1m_kn2", header_nax_mixed, served ++ pin[20], g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1m_kn4", header_nax_mixed, served ++ pin[21], g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1_kn2_sb", header_nax, down_c ++ pin[22], g1, gemm1_text);
    const tbl = pin[23];
    printMetal("turboexl3_pf_table", header_small, pin[24], tbl, text_table);
    printMetal("turboexl3_pf_table_m", header_small, pin[25], tbl, text_table_mixed);
    const frt = pin[26];
    printMetal("turboexl3_pf_froute", header_small, pin[27], frt, text_froute);
    printMetal("turboexl3_pf_froute_m", header_small, pin[28], frt, text_froute_m);
    const prep = pin[29];
    printMetal("turboexl3_pf_prep", header_small, pin[30], prep, text_prep);
    printMetal("turboexl3_pf_prep_l", header_small, pin[31], prep, text_prep);
    printMetal("turboexl3_pf_prep_l2", header_small, pin[32], prep, text_prep);
    const prep_v = pin[33];
    printMetal("turboexl3_pf_prep_v", header_small, pin[34], prep_v, text_prep_v);
    const mid = pin[35];
    printMetal("turboexl3_pf_mid", header_small, pin[36], mid, text_mid);
    printMetal("turboexl3_pf_mid_l", header_small, pin[37], mid, text_mid);
    printMetal("turboexl3_pf_mid_l2", header_small, pin[38], mid, text_mid);
    const mid_v = pin[39];
    printMetal("turboexl3_pf_mid_v", header_small, pin[40], mid_v, text_mid_v);
    const fin = pin[41];
    printMetal("turboexl3_pf_fin", header_small, pin[42], fin, text_finish);
    const fin_v = pin[43];
    printMetal("turboexl3_pf_fin_v", header_small, pin[44], fin_v, text_finish_v);
}

test "turboexl3 prefill 128-row tiles texts: both GEMMs' launches are the mixed chain's kernels; mid and finish change only where they read a row; the route is froute's count and scatter with the 128-row rule" {
    const pin = comptime @import("test_support.zig").pins(@embedFile("kernels/test/pins/prefill_t128.metal"));
    const t = std.testing;
    const has = struct {
        fn f(h: []const u8, n: []const u8) bool {
            return std.mem.indexOf(u8, h, n) != null;
        }
    }.f;
    const count = struct {
        fn f(h: []const u8, n: []const u8) usize {
            return std.mem.count(u8, h, n);
        }
    }.f;
    // nothing in the chain writes through an input binding: every launch writes only its own outputs
    for ([_][]const u8{ text_froute_t128, text_mid_t128, text_finish_t128, gemm2m_text, gemm2_text, gemm1m_text, gemm1_text }) |x| try t.expect(!has(x, pin[0]));
    // mid: text_mid with the row choice after the expert and the two loads through it, nothing else
    try t.expect(count(text_mid_t128, pin[1]) == 1 and count(text_mid_t128, pin[2]) == 1 and count(text_mid_t128, pin[3]) == 1);
    try t.expect(has(text_mid_t128, pin[4]) and has(text_mid_t128, pin[5]) and !has(text_mid_t128, pin[6]) and !has(text_mid_t128, pin[7]));
    {
        const back = comptime replace(replace(replace(text_mid_t128, mid_t128_pick, ""), pin[8], pin[6]), pin[9], pin[7]);
        try t.expectEqualStrings(text_mid, back);
    }
    // finish: text_finish with the slot's row choice and its load through it
    try t.expect(count(text_finish_t128, pin[10]) == 1 and has(text_finish_t128, pin[11]) and !has(text_finish_t128, pin[12]));
    {
        const back = comptime replace(replace(text_finish_t128, finish_t128_pick, ""), pin[13], pin[12]);
        try t.expectEqualStrings(text_finish, back);
    }
    // the choice reads the rows before an expert's first remainder position from the 128-row tiles' outputs
    try t.expect(mid_t128_in.len == mid_in.len + 3 and fin_t128_in.len == fin_in.len + 2);
    for (mid_in, mid_t128_in[0..mid_in.len]) |p, q| try t.expect(std.mem.eql(u8, std.mem.span(p), std.mem.span(q)));
    for (fin_in, fin_t128_in[0..fin_in.len]) |p, q| try t.expect(std.mem.eql(u8, std.mem.span(p), std.mem.span(q)));
    try t.expect(std.mem.eql(u8, std.mem.span(mid_t128_in[mid_in.len + 2]), "xbase") and std.mem.eql(u8, std.mem.span(fin_t128_in[fin_in.len + 1]), "xbase"));
    // the route: froute's count and scatter verbatim, the 128-row table's bound as t128TileRows computes it, the mixed rule,
    // each expert's first remainder position
    try t.expect(has(text_froute_t128, froute_count));
    try t.expect(has(text_froute_t128, pin[14]) and has(text_froute_t128, pin[15]));
    try t.expect(has(text_froute_t128, pin[16]) and has(text_froute_t128, pin[17]));
    try t.expect(has(text_froute_t128, pin[18]));
    try t.expect(has(text_froute_t128, pin[19]));
    try t.expect(has(text_froute_t128, pin[20]));
    try t.expect(has(text_froute_t128, pin[21]) and has(text_froute_t128, pin[22]) and has(text_froute_t128, pin[23]));
    try t.expect(has(text_froute_t128, @embedFile("kernels/test/pins/prefill_t128_1.metal")));
    try t.expect(froute_t128_out.len == 4 and std.mem.eql(u8, std.mem.span(froute_t128_out[3]), "xbase"));
}

test "turboexl3 prefill 128-row tiles tables: every sorted position in exactly one tile; 128-row tiles hold full runs or one remainder past the cut; the mixed tiles hold what is left; the route's bounds hold" {
    const t = std.testing;
    const a = t.allocator;
    var prng = std.Random.DefaultPrng.init(0xE96E);
    const rnd = prng.random();
    var counts: [512]u32 = undefined;
    for (0..24) |ci| {
        for (&counts, 0..) |*c, e| c.* = switch (ci % 6) {
            0 => rnd.uintLessThan(u32, 200),
            // skewed: a few hot experts, many cold ones
            1 => if (rnd.uintLessThan(u32, 8) == 0) rnd.uintLessThan(u32, 3000) else rnd.uintLessThan(u32, 40),
            2 => @intCast(e % 301),
            // every count on a 128-row tile's cut edges, and on the 64- and 32-row ones
            3 => @intCast((e % 4) * 128 + 63 + (e % 3)),
            4 => @intCast((e % 3) * 64 + 31 + (e % 4)),
            else => if (e % 7 == 0) 0 else rnd.uintLessThan(u32, 600),
        };
        var S: u32 = 0;
        for (counts) |c| S += c;
        for ([_]u8{ 0, 1, 32, 63, 64, 65, 96, 127, 200 }) |cut| for ([_]bool{ true, false }) |on| {
            var tb = try t128Tables(a, &counts, 64, cut, on);
            defer tb.deinit(a);
            const cover = try a.alloc(u8, S);
            defer a.free(cover);
            @memset(cover, 0);
            const cu: u32 = @min(cut, 127);
            for (tb.t128.items, 0..) |q, i| {
                try t.expect(q[2] >= 1 and q[2] <= 128 and (q[2] == 128 or q[2] > cu));
                // a part-full one is its expert's last 128-row tile
                if (q[2] < 128 and i + 1 < tb.t128.items.len) try t.expect(tb.t128.items[i + 1][0] != q[0]);
                for (q[1]..q[1] + q[2]) |p| cover[p] += 1;
            }
            for (tb.big.items) |q| {
                try t.expect(q[2] >= 1 and q[2] <= 64 and (q[2] == 64 or q[2] > 32));
                for (q[1]..q[1] + q[2]) |p| cover[p] += 1;
            }
            for (tb.small.items) |q| {
                try t.expect(q[2] >= 1 and q[2] <= 32);
                for (q[1]..q[1] + q[2]) |p| cover[p] += 1;
            }
            for (cover) |v| try t.expectEqual(@as(u8, 1), v);
            // the route's output bounds: the 128-row table (`t128TileRows`, `text_froute_t128`'s TQ) and the mixed one's
            // (`t128MixedRows`, TBIG; never above the mixed table without the 128-row tiles)
            try t.expect(tb.t128.items.len <= t128TileRows(S, 512, cut, on));
            try t.expect(tb.big.items.len <= t128MixedRows(S, 512, 64, 1, cut, on) and tb.small.items.len <= 512);
            try t.expect(t128MixedRows(S, 512, 64, 1, cut, on) <= (S + 63) / 64 + 512);
            if (!on) try t.expectEqual(@as(usize, 0), tb.t128.items.len);
            // with the 128-row tiles on, the mixed tiles take each expert's rows past its full 128-row tiles when no more
            // than the cut (an expert with a part-full 128-row tile has none), and nothing else
            if (on) {
                var mixed: [512]u32 = @splat(0);
                for (tb.big.items) |m| mixed[m[0]] += m[2];
                for (tb.small.items) |m| mixed[m[0]] += m[2];
                for (counts, mixed) |c, mx| try t.expect(mx <= cu and mx == (if (c % 128 > cu) 0 else c % 128));
            }
        };
    }
}

test "turboexl3 prefill 128-row tiles options: t128 needs bm 64, the mixed table at split 1, froute and aux 0; the tables' row bounds" {
    const t = std.testing;
    const served: Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .defer_affine = true, .ncb = 4, .sintr = true, .froute = true, .alay = 2 };
    var o = served;
    o.t128 = true;
    try t.expect(o.valid());
    var v = o;
    v.bm = 128;
    try t.expect(!v.valid());
    v = o;
    v.froute = false;
    try t.expect(!v.valid());
    v = o;
    v.split = 2;
    try t.expect(!v.valid());
    v = o;
    v.small = .off;
    try t.expect(!v.valid());
    v = o;
    v.alay = 0;
    v.aux = 8;
    try t.expect(!v.valid());
    try t.expectEqual(@as(u32, 1), t128TileRows(13280, 512, 64, false));
    try t.expectEqual(@as(u32, 81920 / 128), t128TileRows(81920, 512, 127, true));
    // 512 part-full tiles of 65 rows, then full ones: 512 + (40960 - 33280) / 128
    try t.expectEqual(@as(u32, 572), t128TileRows(40960, 512, 64, true));
    try t.expectEqual(@as(u32, 892), t128TileRows(81920, 512, 64, true));
    try t.expectEqual(@as(u32, 204), t128TileRows(13280, 512, 64, true));
    try t.expectEqual(@as(u32, 828), t128TileRows(40960, 512, 0, true));
    try t.expectEqual(@as(u32, 1), t128TileRows(10, 512, 64, true));
    // the mixed table: all of it without the 128-row tiles; with them one tile per expert at cut 64 (two at 127, none at
    // 32), at most S / 33 of them
    try t.expectEqual(@as(u32, 1152), t128MixedRows(40960, 512, 64, 1, 64, false));
    try t.expectEqual(@as(u32, 512), t128MixedRows(81920, 512, 64, 1, 64, true));
    try t.expectEqual(@as(u32, 402), t128MixedRows(13280, 512, 64, 1, 64, true));
    try t.expectEqual(@as(u32, 1024), t128MixedRows(40960, 512, 64, 1, 127, true));
    try t.expectEqual(@as(u32, 1), t128MixedRows(40960, 512, 64, 1, 32, true));
    try t.expectEqual(@as(u32, 9), t128MixedRows(310, 512, 64, 1, 64, true));
}

test "turboexl3 prefill 128-row tiles metal sources: the route, both GEMMs' launches (both bodies), mid and finish as standalone translation units (TURBOEXL3_METAL_DUMP=1)" {
    const pin = comptime @import("test_support.zig").pins(@embedFile("kernels/test/dump/prefill_t128.metal"));
    _ = std.c.getenv("TURBOEXL3_METAL_DUMP") orelse return error.SkipZigTest;
    const frt = pin[0];
    printMetal("turboexl3_pf_t128_route", header_small, pin[1], frt, text_froute_t128);
    // the served gate|up launch (bm 64, fragment order, A layout 2, the register path's integer-rate reader) and its bm-128
    // body; the served down launch (ncb 4) and its bm-128 body; the mixed chain's kernels' arguments
    const gu = pin[2];
    const dn = pin[3];
    const m64 = pin[4];
    const m128 = pin[5];
    const g2 = pin[6];
    const g1 = pin[7];
    printMetal("turboexl3_pf_t128_gum", header_nax, gu ++ m64, g2, gemm2m_text);
    printMetal("turboexl3_pf_t128_gut", header_nax, gu ++ m128, g2, gemm2_text);
    printMetal("turboexl3_pf_t128_dnm", header_nax, dn ++ m64, g1, gemm1m_text);
    printMetal("turboexl3_pf_t128_dnt", header_nax, dn ++ m128, g1, gemm1_text);
    const mid = pin[8];
    printMetal("turboexl3_pf_t128_mid", header_small, pin[9], mid, text_mid_t128);
    const fin = pin[10];
    printMetal("turboexl3_pf_t128_fin", header_small, pin[11], fin, text_finish_t128);
}
