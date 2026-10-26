//! The decode chain (rows <= DECODE_ROWS_MAX), form 1: two launches per MoE layer, any admitted rate, codebook, window
//! and 128-aligned geometry, every constant a template int of one Metal text per kernel.
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

pub const Geometry = struct {
    hidden: u32,
    inter: u32,
    topk: u32,
    rate: format.Rate,
    dec: format.Decode,
    /// timing probes only (never a served kernel): 1 skips the codebook hash, 2 skips the funnel too
    probe: u8 = 0,

    pub fn admitted(g: Geometry) bool {
        return g.hidden % 128 == 0 and g.inter % 128 == 0 and g.hidden > 0 and g.inter > 0 and g.topk > 0 and g.topk <= 32;
    }
};

/// The Metal body shared by both kernels after the helpers: the lane plan's tile walk. `turboexl3_tile_fma` decodes a
/// lane's eight weights of one tile and accumulates them against the rotated input held in threadgroup memory (f16).
const metal_common =
    \\// lane L's eight weights of a tile: code positions 8L + m, m < 4 at column L / 4, m >= 4 at column L / 4 + 8,
    \\// rows 2 (L & 3) + {0, 1, 8, 9}[m & 3] (quantize.py's tensor-core layout)
    \\inline uint turboexl3_row_of(uint L, uint m) { const uint q = m & 3u; return 2u * (L & 3u) + ((q & 1u) ? 1u : 0u) + ((q & 2u) ? 8u : 0u); }
    \\
;

/// Launch 1's text.
pub const text_gu =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr int PRB = int(PROBE);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint IN_TILES = HIDC / 16u;
    \\    constexpr uint OUT_TILES = INTC / 16u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;      // 128-wide block of the intermediate
    \\    const uint k = threadgroup_position_in_grid.y;        // slot
    \\    const uint row = threadgroup_position_in_grid.z;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint e = uint(slots[row * TOPKC + k]);
    \\    threadgroup half xg[HIDC];
    \\    threadgroup half xu[HIDC];
    \\    threadgroup float red_g[4][128];
    \\    threadgroup float red_u[4][128];
    \\    // 1. the rotated inputs for gate and up (their suh differ): simdgroup sg owns blocks sg, sg + 4, ...
    \\    for (uint b = sg; b < HIDC / 128u; b += 4u) {
    \\        float vg[4]; float vu[4];
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint i = b * 128u + L + 32u * j;
    \\            const float xr = float(half(x[row * HIDC + i]));
    \\            vg[j] = xr * float(gate_suh[e * HIDC + i]);
    \\            vu[j] = xr * float(up_suh[e * HIDC + i]);
    \\        }
    \\        for (uint s = 0; s < 5u; ++s) {
    \\            const uint bit = 1u << s;
    \\            for (uint j = 0; j < 4u; ++j) {
    \\                const float pg = simd_shuffle_xor(vg[j], ushort(bit));
    \\                const float pu = simd_shuffle_xor(vu[j], ushort(bit));
    \\                vg[j] = (L & bit) ? (pg - vg[j]) : (vg[j] + pg);
    \\                vu[j] = (L & bit) ? (pu - vu[j]) : (vu[j] + pu);
    \\            }
    \\        }
    \\        {   // stages 5 and 6 pair the registers (positions L + 32 j: bit 5 is j & 1, bit 6 is j & 2)
    \\            float tg0 = vg[0] + vg[1], tg1 = vg[0] - vg[1], tg2 = vg[2] + vg[3], tg3 = vg[2] - vg[3];
    \\            float tu0 = vu[0] + vu[1], tu1 = vu[0] - vu[1], tu2 = vu[2] + vu[3], tu3 = vu[2] - vu[3];
    \\            vg[0] = tg0 + tg2; vg[1] = tg1 + tg3; vg[2] = tg0 - tg2; vg[3] = tg1 - tg3;
    \\            vu[0] = tu0 + tu2; vu[1] = tu1 + tu3; vu[2] = tu0 - tu2; vu[3] = tu1 - tu3;
    \\        }
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint i = b * 128u + L + 32u * j;
    \\            xg[i] = half(vg[j] * HAD_SCALE);
    \\            xu[i] = half(vu[j] * HAD_SCALE);
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // 2. the GEMV over this simdgroup's quarter of the input tiles, the block's 8 output tiles, gate and up
    \\    constexpr bool INTK = (N % 16u) == 0u;
    \\    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    \\    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    \\    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    \\    uint off[8];
    \\    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    \\    uint rows_of[4];
    \\    for (uint m = 0; m < 4u; ++m) rows_of[m] = turboexl3_row_of(L, m);
    \\    float ag[8][2]; float au[8][2];
    \\    for (uint t = 0; t < 8u; ++t) { ag[t][0] = 0.0f; ag[t][1] = 0.0f; au[t][0] = 0.0f; au[t][1] = 0.0f; }
    \\    const uint tk_per = IN_TILES / 4u;
    \\    const device uint* gbase = (const device uint*)(gate_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\    const device uint* ubase = (const device uint*)(up_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\    for (uint tk = sg * tk_per; tk < (sg + 1u) * tk_per; ++tk) {
    \\        float xr_g[4]; float xr_u[4];
    \\        for (uint m = 0; m < 4u; ++m) { xr_g[m] = float(xg[tk * 16u + rows_of[m]]); xr_u[m] = float(xu[tk * 16u + rows_of[m]]); }
    \\        for (uint t = 0; t < 8u; ++t) {
    \\            const uint tn = blk * 8u + t;
    \\            const ulong toff = (ulong(tk) * OUT_TILES + tn) * ulong(NW);
    \\            if (INTK) {
    \\                const IntWords wg = exl3_int_words<N>(gbase + toff, base, ia);
    \\                const IntWords wu = exl3_int_words<N>(ubase + toff, base, ia);
    \\                uint cg[8], cu[8];
    \\                cg[0] = exl3_int_cw<N, WMASK, 0>(wg, ib); cg[1] = exl3_int_cw<N, WMASK, 1>(wg, ib); cg[2] = exl3_int_cw<N, WMASK, 2>(wg, ib); cg[3] = exl3_int_cw<N, WMASK, 3>(wg, ib);
    \\                cg[4] = exl3_int_cw<N, WMASK, 4>(wg, ib); cg[5] = exl3_int_cw<N, WMASK, 5>(wg, ib); cg[6] = exl3_int_cw<N, WMASK, 6>(wg, ib); cg[7] = exl3_int_cw<N, WMASK, 7>(wg, ib);
    \\                cu[0] = exl3_int_cw<N, WMASK, 0>(wu, ib); cu[1] = exl3_int_cw<N, WMASK, 1>(wu, ib); cu[2] = exl3_int_cw<N, WMASK, 2>(wu, ib); cu[3] = exl3_int_cw<N, WMASK, 3>(wu, ib);
    \\                cu[4] = exl3_int_cw<N, WMASK, 4>(wu, ib); cu[5] = exl3_int_cw<N, WMASK, 5>(wu, ib); cu[6] = exl3_int_cw<N, WMASK, 6>(wu, ib); cu[7] = exl3_int_cw<N, WMASK, 7>(wu, ib);
    \\                for (uint m = 0; m < 8u; ++m) {
    \\                    const float wgv = (PRB == 0) ? float(exl3_decode<CB>(cg[m])) : ((PRB == 1) ? float(as_type<half>(ushort(cg[m] & 0x3fffu))) : float(as_type<half>(ushort((wg.h[0] >> (m * 2u)) & 0x3fffu))));
    \\                    const float wuv = (PRB == 0) ? float(exl3_decode<CB>(cu[m])) : ((PRB == 1) ? float(as_type<half>(ushort(cu[m] & 0x3fffu))) : float(as_type<half>(ushort((wu.h[0] >> (m * 2u)) & 0x3fffu))));
    \\                    const uint c = (m >> 2);
    \\                    ag[t][c] = fma(xr_g[m & 3u], wgv, ag[t][c]);
    \\                    au[t][c] = fma(xr_u[m & 3u], wuv, au[t][c]);
    \\                }
    \\            } else {
    \\                const LaneWords wg = exl3_lane_words<N>(gbase + toff, base);
    \\                const LaneWords wu = exl3_lane_words<N>(ubase + toff, base);
    \\                for (uint m = 0; m < 8u; ++m) {
    \\                    const float wgv = float(exl3_decode<CB>(exl3_lane_cw<WMASK>(wg, off[m])));
    \\                    const float wuv = float(exl3_decode<CB>(exl3_lane_cw<WMASK>(wu, off[m])));
    \\                    const uint c = (m >> 2);
    \\                    ag[t][c] = fma(xr_g[m & 3u], wgv, ag[t][c]);
    \\                    au[t][c] = fma(xr_u[m & 3u], wuv, au[t][c]);
    \\                }
    \\            }
    \\        }
    \\    }
    \\    // 3. the column sums: lanes 4c..4c+3 hold column c (and c + 8); fold them, then the four simdgroups
    \\    for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) {
    \\        ag[t][c] += simd_shuffle_xor(ag[t][c], ushort(1)); ag[t][c] += simd_shuffle_xor(ag[t][c], ushort(2));
    \\        au[t][c] += simd_shuffle_xor(au[t][c], ushort(1)); au[t][c] += simd_shuffle_xor(au[t][c], ushort(2));
    \\    }
    \\    if ((L & 3u) == 0u) {
    \\        for (uint t = 0; t < 8u; ++t) {
    \\            red_g[sg][t * 16u + (L >> 2)] = ag[t][0]; red_g[sg][t * 16u + (L >> 2) + 8u] = ag[t][1];
    \\            red_u[sg][t * 16u + (L >> 2)] = au[t][0]; red_u[sg][t * 16u + (L >> 2) + 8u] = au[t][1];
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // 4. simdgroup 0: the block's 128 outputs (positions L + 32 j): f16 round, H128, svh, f16 round, SwiGLU
    \\    if (sg == 0u) {
    \\        float vg[4]; float vu[4];
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint i = L + 32u * j;
    \\            vg[j] = float(half(((red_g[0][i] + red_g[1][i]) + red_g[2][i]) + red_g[3][i]));
    \\            vu[j] = float(half(((red_u[0][i] + red_u[1][i]) + red_u[2][i]) + red_u[3][i]));
    \\        }
    \\        for (uint s = 0; s < 5u; ++s) {
    \\            const uint bit = 1u << s;
    \\            for (uint j = 0; j < 4u; ++j) {
    \\                const float pg = simd_shuffle_xor(vg[j], ushort(bit));
    \\                const float pu = simd_shuffle_xor(vu[j], ushort(bit));
    \\                vg[j] = (L & bit) ? (pg - vg[j]) : (vg[j] + pg);
    \\                vu[j] = (L & bit) ? (pu - vu[j]) : (vu[j] + pu);
    \\            }
    \\        }
    \\        {
    \\            float tg0 = vg[0] + vg[1], tg1 = vg[0] - vg[1], tg2 = vg[2] + vg[3], tg3 = vg[2] - vg[3];
    \\            float tu0 = vu[0] + vu[1], tu1 = vu[0] - vu[1], tu2 = vu[2] + vu[3], tu3 = vu[2] - vu[3];
    \\            vg[0] = tg0 + tg2; vg[1] = tg1 + tg3; vg[2] = tg0 - tg2; vg[3] = tg1 - tg3;
    \\            vu[0] = tu0 + tu2; vu[1] = tu1 + tu3; vu[2] = tu0 - tu2; vu[3] = tu1 - tu3;
    \\        }
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint o = blk * 128u + L + 32u * j;
    \\            const float g = float(half(vg[j] * HAD_SCALE * float(gate_svh[e * INTC + o])));
    \\            const float u = float(half(vu[j] * HAD_SCALE * float(up_svh[e * INTC + o])));
    \\            h[(row * TOPKC + k) * INTC + o] = (g / (1.0f + metal::exp(-g))) * u;
    \\        }
    \\    }
    \\
;

/// Launch 1 of form 1b: `text_gu` with the input rotation done once per (row, slot) by `unfused.text_prep`.
pub const text_gu4 =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr int PRB = int(PROBE);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint IN_TILES = HIDC / 16u;
    \\    constexpr uint OUT_TILES = INTC / 16u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;      // 128-wide block of the intermediate
    \\    const uint k = threadgroup_position_in_grid.y;        // slot
    \\    const uint row = threadgroup_position_in_grid.z;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint e = uint(slots[row * TOPKC + k]);
    \\    threadgroup float red_g[4][128];
    \\    threadgroup float red_u[4][128];
    \\    const device half* xg = xgp + ulong(row * TOPKC + k) * ulong(HIDC);
    \\    const device half* xu = xup + ulong(row * TOPKC + k) * ulong(HIDC);
    \\    // 2. the GEMV over this simdgroup's quarter of the input tiles, the block's 8 output tiles, gate and up
    \\    constexpr bool INTK = (N % 16u) == 0u;
    \\    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    \\    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    \\    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    \\    uint off[8];
    \\    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    \\    uint rows_of[4];
    \\    for (uint m = 0; m < 4u; ++m) rows_of[m] = turboexl3_row_of(L, m);
    \\    float ag[8][2]; float au[8][2];
    \\    for (uint t = 0; t < 8u; ++t) { ag[t][0] = 0.0f; ag[t][1] = 0.0f; au[t][0] = 0.0f; au[t][1] = 0.0f; }
    \\    const uint tk_per = IN_TILES / 4u;
    \\    const device uint* gbase = (const device uint*)(gate_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\    const device uint* ubase = (const device uint*)(up_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\    for (uint tk = sg * tk_per; tk < (sg + 1u) * tk_per; ++tk) {
    \\        float xr_g[4]; float xr_u[4];
    \\        for (uint m = 0; m < 4u; ++m) { xr_g[m] = float(xg[tk * 16u + rows_of[m]]); xr_u[m] = float(xu[tk * 16u + rows_of[m]]); }
    \\        for (uint t = 0; t < 8u; ++t) {
    \\            const uint tn = blk * 8u + t;
    \\            const ulong toff = (ulong(tk) * OUT_TILES + tn) * ulong(NW);
    \\            if (INTK) {
    \\                const IntWords wg = exl3_int_words<N>(gbase + toff, base, ia);
    \\                const IntWords wu = exl3_int_words<N>(ubase + toff, base, ia);
    \\                uint cg[8], cu[8];
    \\                cg[0] = exl3_int_cw<N, WMASK, 0>(wg, ib); cg[1] = exl3_int_cw<N, WMASK, 1>(wg, ib); cg[2] = exl3_int_cw<N, WMASK, 2>(wg, ib); cg[3] = exl3_int_cw<N, WMASK, 3>(wg, ib);
    \\                cg[4] = exl3_int_cw<N, WMASK, 4>(wg, ib); cg[5] = exl3_int_cw<N, WMASK, 5>(wg, ib); cg[6] = exl3_int_cw<N, WMASK, 6>(wg, ib); cg[7] = exl3_int_cw<N, WMASK, 7>(wg, ib);
    \\                cu[0] = exl3_int_cw<N, WMASK, 0>(wu, ib); cu[1] = exl3_int_cw<N, WMASK, 1>(wu, ib); cu[2] = exl3_int_cw<N, WMASK, 2>(wu, ib); cu[3] = exl3_int_cw<N, WMASK, 3>(wu, ib);
    \\                cu[4] = exl3_int_cw<N, WMASK, 4>(wu, ib); cu[5] = exl3_int_cw<N, WMASK, 5>(wu, ib); cu[6] = exl3_int_cw<N, WMASK, 6>(wu, ib); cu[7] = exl3_int_cw<N, WMASK, 7>(wu, ib);
    \\                for (uint m = 0; m < 8u; ++m) {
    \\                    const float wgv = (PRB == 0) ? float(exl3_decode<CB>(cg[m])) : ((PRB == 1) ? float(as_type<half>(ushort(cg[m] & 0x3fffu))) : float(as_type<half>(ushort((wg.h[0] >> (m * 2u)) & 0x3fffu))));
    \\                    const float wuv = (PRB == 0) ? float(exl3_decode<CB>(cu[m])) : ((PRB == 1) ? float(as_type<half>(ushort(cu[m] & 0x3fffu))) : float(as_type<half>(ushort((wu.h[0] >> (m * 2u)) & 0x3fffu))));
    \\                    const uint c = (m >> 2);
    \\                    ag[t][c] = fma(xr_g[m & 3u], wgv, ag[t][c]);
    \\                    au[t][c] = fma(xr_u[m & 3u], wuv, au[t][c]);
    \\                }
    \\            } else {
    \\                const LaneWords wg = exl3_lane_words<N>(gbase + toff, base);
    \\                const LaneWords wu = exl3_lane_words<N>(ubase + toff, base);
    \\                for (uint m = 0; m < 8u; ++m) {
    \\                    const float wgv = float(exl3_decode<CB>(exl3_lane_cw<WMASK>(wg, off[m])));
    \\                    const float wuv = float(exl3_decode<CB>(exl3_lane_cw<WMASK>(wu, off[m])));
    \\                    const uint c = (m >> 2);
    \\                    ag[t][c] = fma(xr_g[m & 3u], wgv, ag[t][c]);
    \\                    au[t][c] = fma(xr_u[m & 3u], wuv, au[t][c]);
    \\                }
    \\            }
    \\        }
    \\    }
    \\    // 3. the column sums: lanes 4c..4c+3 hold column c (and c + 8); fold them, then the four simdgroups
    \\    for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) {
    \\        ag[t][c] += simd_shuffle_xor(ag[t][c], ushort(1)); ag[t][c] += simd_shuffle_xor(ag[t][c], ushort(2));
    \\        au[t][c] += simd_shuffle_xor(au[t][c], ushort(1)); au[t][c] += simd_shuffle_xor(au[t][c], ushort(2));
    \\    }
    \\    if ((L & 3u) == 0u) {
    \\        for (uint t = 0; t < 8u; ++t) {
    \\            red_g[sg][t * 16u + (L >> 2)] = ag[t][0]; red_g[sg][t * 16u + (L >> 2) + 8u] = ag[t][1];
    \\            red_u[sg][t * 16u + (L >> 2)] = au[t][0]; red_u[sg][t * 16u + (L >> 2) + 8u] = au[t][1];
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // 4. simdgroup 0: the block's 128 outputs (positions L + 32 j): f16 round, H128, svh, f16 round, SwiGLU
    \\    if (sg == 0u) {
    \\        float vg[4]; float vu[4];
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint i = L + 32u * j;
    \\            vg[j] = float(half(((red_g[0][i] + red_g[1][i]) + red_g[2][i]) + red_g[3][i]));
    \\            vu[j] = float(half(((red_u[0][i] + red_u[1][i]) + red_u[2][i]) + red_u[3][i]));
    \\        }
    \\        for (uint s = 0; s < 5u; ++s) {
    \\            const uint bit = 1u << s;
    \\            for (uint j = 0; j < 4u; ++j) {
    \\                const float pg = simd_shuffle_xor(vg[j], ushort(bit));
    \\                const float pu = simd_shuffle_xor(vu[j], ushort(bit));
    \\                vg[j] = (L & bit) ? (pg - vg[j]) : (vg[j] + pg);
    \\                vu[j] = (L & bit) ? (pu - vu[j]) : (vu[j] + pu);
    \\            }
    \\        }
    \\        {
    \\            float tg0 = vg[0] + vg[1], tg1 = vg[0] - vg[1], tg2 = vg[2] + vg[3], tg3 = vg[2] - vg[3];
    \\            float tu0 = vu[0] + vu[1], tu1 = vu[0] - vu[1], tu2 = vu[2] + vu[3], tu3 = vu[2] - vu[3];
    \\            vg[0] = tg0 + tg2; vg[1] = tg1 + tg3; vg[2] = tg0 - tg2; vg[3] = tg1 - tg3;
    \\            vu[0] = tu0 + tu2; vu[1] = tu1 + tu3; vu[2] = tu0 - tu2; vu[3] = tu1 - tu3;
    \\        }
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint o = blk * 128u + L + 32u * j;
    \\            const float g = float(half(vg[j] * HAD_SCALE * float(gate_svh[e * INTC + o])));
    \\            const float u = float(half(vu[j] * HAD_SCALE * float(up_svh[e * INTC + o])));
    \\            h[(row * TOPKC + k) * INTC + o] = (g / (1.0f + metal::exp(-g))) * u;
    \\        }
    \\    }
    \\
;

/// Launch 2's text.
pub const text_down =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr int PRB = int(PROBE);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint IN_TILES = INTC / 16u;
    \\    constexpr uint OUT_TILES = HIDC / 16u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;      // 128-wide block of the hidden output
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    threadgroup half hd[INTC];
    \\    threadgroup float red[4][128];
    \\    constexpr bool INTK = (N % 16u) == 0u;
    \\    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    \\    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    \\    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    \\    uint off[8];
    \\    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    \\    uint rows_of[4];
    \\    for (uint m = 0; m < 4u; ++m) rows_of[m] = turboexl3_row_of(L, m);
    \\    const uint tk_per = IN_TILES / 4u;
    \\    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};              // simdgroup 0's outputs L + 32 j, summed over the slots
    \\    for (uint k = 0; k < TOPKC; ++k) {
    \\        const uint e = uint(slots[row * TOPKC + k]);
    \\        const float score = float(scores[row * TOPKC + k]);
    \\        // 1. this slot's rotated h: f16 round, down suh, H128, 1/sqrt(128), f16 round
    \\        for (uint b = sg; b < INTC / 128u; b += 4u) {
    \\            float v[4];
    \\            for (uint j = 0; j < 4u; ++j) {
    \\                const uint i = b * 128u + L + 32u * j;
    \\                v[j] = float(half(h[(row * TOPKC + k) * INTC + i])) * float(down_suh[e * INTC + i]);
    \\            }
    \\            for (uint s = 0; s < 5u; ++s) {
    \\                const uint bit = 1u << s;
    \\                for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); }
    \\            }
    \\            { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\            for (uint j = 0; j < 4u; ++j) hd[b * 128u + L + 32u * j] = half(v[j] * HAD_SCALE);
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        // 2. the GEMV over this simdgroup's quarter of the input tiles, the block's 8 output tiles
    \\        float a[8][2];
    \\        for (uint t = 0; t < 8u; ++t) { a[t][0] = 0.0f; a[t][1] = 0.0f; }
    \\        const device uint* dbase = (const device uint*)(down_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\        for (uint tk = sg * tk_per; tk < (sg + 1u) * tk_per; ++tk) {
    \\            float xr[4];
    \\            for (uint m = 0; m < 4u; ++m) xr[m] = float(hd[tk * 16u + rows_of[m]]);
    \\            for (uint t = 0; t < 8u; ++t) {
    \\                const uint tn = blk * 8u + t;
    \\                if (INTK) {
    \\                    const IntWords w = exl3_int_words<N>(dbase + (ulong(tk) * OUT_TILES + tn) * ulong(NW), base, ia);
    \\                    uint cw[8];
    \\                    cw[0] = exl3_int_cw<N, WMASK, 0>(w, ib); cw[1] = exl3_int_cw<N, WMASK, 1>(w, ib); cw[2] = exl3_int_cw<N, WMASK, 2>(w, ib); cw[3] = exl3_int_cw<N, WMASK, 3>(w, ib);
    \\                    cw[4] = exl3_int_cw<N, WMASK, 4>(w, ib); cw[5] = exl3_int_cw<N, WMASK, 5>(w, ib); cw[6] = exl3_int_cw<N, WMASK, 6>(w, ib); cw[7] = exl3_int_cw<N, WMASK, 7>(w, ib);
    \\                    for (uint m = 0; m < 8u; ++m) {
    \\                        const float wv = (PRB == 0) ? float(exl3_decode<CB>(cw[m])) : ((PRB == 1) ? float(as_type<half>(ushort(cw[m] & 0x3fffu))) : float(as_type<half>(ushort((w.h[0] >> (m * 2u)) & 0x3fffu))));
    \\                        a[t][m >> 2] = fma(xr[m & 3u], wv, a[t][m >> 2]);
    \\                    }
    \\                } else {
    \\                    const LaneWords w = exl3_lane_words<N>(dbase + (ulong(tk) * OUT_TILES + tn) * ulong(NW), base);
    \\                    for (uint m = 0; m < 8u; ++m) {
    \\                        const float wv = float(exl3_decode<CB>(exl3_lane_cw<WMASK>(w, off[m])));
    \\                        a[t][m >> 2] = fma(xr[m & 3u], wv, a[t][m >> 2]);
    \\                    }
    \\                }
    \\            }
    \\        }
    \\        for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) { a[t][c] += simd_shuffle_xor(a[t][c], ushort(1)); a[t][c] += simd_shuffle_xor(a[t][c], ushort(2)); }
    \\        if ((L & 3u) == 0u) for (uint t = 0; t < 8u; ++t) { red[sg][t * 16u + (L >> 2)] = a[t][0]; red[sg][t * 16u + (L >> 2) + 8u] = a[t][1]; }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        // 3. simdgroup 0: f16 round, H128, svh, f16 round, score-weighted accumulate
    \\        if (sg == 0u) {
    \\            float v[4];
    \\            for (uint j = 0; j < 4u; ++j) { const uint i = L + 32u * j; v[j] = float(half(((red[0][i] + red[1][i]) + red[2][i]) + red[3][i])); }
    \\            for (uint s = 0; s < 5u; ++s) {
    \\                const uint bit = 1u << s;
    \\                for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); }
    \\            }
    \\            { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\            for (uint j = 0; j < 4u; ++j) {
    \\                const uint o = blk * 128u + L + 32u * j;
    \\                acc[j] = fma(score, float(half(v[j] * HAD_SCALE * float(down_svh[e * HIDC + o]))), acc[j]);
    \\            }
    \\        }
    \\        // the next slot's hd and red overwrite what every simdgroup just read
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (sg == 0u) for (uint j = 0; j < 4u; ++j) y[row * HIDC + blk * 128u + L + 32u * j] = acc[j];
    \\
;

/// Launch 2 of form 1b: `text_down` with each slot's h rotation done once by `unfused.text_prep` (over inter).
pub const text_down4 =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr int PRB = int(PROBE);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint IN_TILES = INTC / 16u;
    \\    constexpr uint OUT_TILES = HIDC / 16u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;      // 128-wide block of the hidden output
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    threadgroup float red[4][128];
    \\    constexpr bool INTK = (N % 16u) == 0u;
    \\    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    \\    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    \\    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    \\    uint off[8];
    \\    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    \\    uint rows_of[4];
    \\    for (uint m = 0; m < 4u; ++m) rows_of[m] = turboexl3_row_of(L, m);
    \\    const uint tk_per = IN_TILES / 4u;
    \\    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};              // simdgroup 0's outputs L + 32 j, summed over the slots
    \\    for (uint k = 0; k < TOPKC; ++k) {
    \\        const uint e = uint(slots[row * TOPKC + k]);
    \\        const float score = float(scores[row * TOPKC + k]);
    \\        const device half* hd = hdp + ulong(row * TOPKC + k) * ulong(INTC);
    \\        // 2. the GEMV over this simdgroup's quarter of the input tiles, the block's 8 output tiles
    \\        float a[8][2];
    \\        for (uint t = 0; t < 8u; ++t) { a[t][0] = 0.0f; a[t][1] = 0.0f; }
    \\        const device uint* dbase = (const device uint*)(down_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\        for (uint tk = sg * tk_per; tk < (sg + 1u) * tk_per; ++tk) {
    \\            float xr[4];
    \\            for (uint m = 0; m < 4u; ++m) xr[m] = float(hd[tk * 16u + rows_of[m]]);
    \\            for (uint t = 0; t < 8u; ++t) {
    \\                const uint tn = blk * 8u + t;
    \\                if (INTK) {
    \\                    const IntWords w = exl3_int_words<N>(dbase + (ulong(tk) * OUT_TILES + tn) * ulong(NW), base, ia);
    \\                    uint cw[8];
    \\                    cw[0] = exl3_int_cw<N, WMASK, 0>(w, ib); cw[1] = exl3_int_cw<N, WMASK, 1>(w, ib); cw[2] = exl3_int_cw<N, WMASK, 2>(w, ib); cw[3] = exl3_int_cw<N, WMASK, 3>(w, ib);
    \\                    cw[4] = exl3_int_cw<N, WMASK, 4>(w, ib); cw[5] = exl3_int_cw<N, WMASK, 5>(w, ib); cw[6] = exl3_int_cw<N, WMASK, 6>(w, ib); cw[7] = exl3_int_cw<N, WMASK, 7>(w, ib);
    \\                    for (uint m = 0; m < 8u; ++m) {
    \\                        const float wv = (PRB == 0) ? float(exl3_decode<CB>(cw[m])) : ((PRB == 1) ? float(as_type<half>(ushort(cw[m] & 0x3fffu))) : float(as_type<half>(ushort((w.h[0] >> (m * 2u)) & 0x3fffu))));
    \\                        a[t][m >> 2] = fma(xr[m & 3u], wv, a[t][m >> 2]);
    \\                    }
    \\                } else {
    \\                    const LaneWords w = exl3_lane_words<N>(dbase + (ulong(tk) * OUT_TILES + tn) * ulong(NW), base);
    \\                    for (uint m = 0; m < 8u; ++m) {
    \\                        const float wv = float(exl3_decode<CB>(exl3_lane_cw<WMASK>(w, off[m])));
    \\                        a[t][m >> 2] = fma(xr[m & 3u], wv, a[t][m >> 2]);
    \\                    }
    \\                }
    \\            }
    \\        }
    \\        for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) { a[t][c] += simd_shuffle_xor(a[t][c], ushort(1)); a[t][c] += simd_shuffle_xor(a[t][c], ushort(2)); }
    \\        if ((L & 3u) == 0u) for (uint t = 0; t < 8u; ++t) { red[sg][t * 16u + (L >> 2)] = a[t][0]; red[sg][t * 16u + (L >> 2) + 8u] = a[t][1]; }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        // 3. simdgroup 0: f16 round, H128, svh, f16 round, score-weighted accumulate
    \\        if (sg == 0u) {
    \\            float v[4];
    \\            for (uint j = 0; j < 4u; ++j) { const uint i = L + 32u * j; v[j] = float(half(((red[0][i] + red[1][i]) + red[2][i]) + red[3][i])); }
    \\            for (uint s = 0; s < 5u; ++s) {
    \\                const uint bit = 1u << s;
    \\                for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); }
    \\            }
    \\            { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\            for (uint j = 0; j < 4u; ++j) {
    \\                const uint o = blk * 128u + L + 32u * j;
    \\                acc[j] = fma(score, float(half(v[j] * HAD_SCALE * float(down_svh[e * HIDC + o]))), acc[j]);
    \\            }
    \\        }
    \\        // the next slot's hd and red overwrite what every simdgroup just read
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (sg == 0u) for (uint j = 0; j < 4u; ++j) y[row * HIDC + blk * 128u + L + 32u * j] = acc[j];
    \\
;

/// The GEMV core every form-3 kernel inlines: lane L of a simdgroup accumulates its 8 tiles x 2 columns over input
/// tiles [tk0, tk1) of expert `wbase` (the stacked bank's expert base, in uint words), x in lane order.
const metal_gemv_core =
    \\// the row-slots that chose expert e, in slot order: slot rk leads when its occurrence index is a multiple of GRP and
    \\// takes the next GRP - 1 occurrences as its members (GRP is 1 or 2); false for a slot some lead covers. `slots` is any
    \\// pointer: MLX binds an input of fewer than 8 elements (a 1-slot call at under 8 rows) in the constant address space
    \\template <uint GRP, uint RKC, typename SP>
    \\inline bool turboexl3_members(SP slots, uint rk, uint e, uint L, thread uint (&members)[2], thread uint& m) {
    \\  members[0] = rk; members[1] = rk; m = 1u;
    \\  if (GRP == 1u) return true;
    \\  uint before = 0u; uint next = 0xffffffffu;
    \\  for (uint b = 0u; b < RKC; b += 32u) {
    \\    const uint i = b + L;
    \\    const bool hit = (i < RKC) && (i != rk) && (uint(slots[i]) == e);
    \\    const uint mask = uint(static_cast<simd_vote::vote_t>(simd_ballot(hit)));
    \\    const uint below = (b + 32u <= rk) ? 0xffffffffu : ((b >= rk) ? 0u : ((1u << (rk - b)) - 1u));
    \\    before += popcount(mask & below);
    \\    const uint above = mask & ~below;
    \\    if (next == 0xffffffffu && above != 0u) next = b + ctz(above);
    \\  }
    \\  if ((before % GRP) != 0u) return false;
    \\  if (next != 0xffffffffu) { members[1] = next; m = 2u; }
    \\  return true;
    \\}
    \\template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT>
    \\inline void turboexl3_gemv_core(const device uint* wbase, const device half* xrow, uint tk0, uint tk1, uint blk, uint L, thread float (&a)[BT][2]) {
    \\  constexpr uint NW = N / 2u;
    \\  // the general fast path (every even N): one select, one shift, one mask per weight
    \\  const uint qb = exl3_gen_qb<N>(L);
    \\  const uint ga = exl3_gen_a<N>(L);
    \\  const uint lp = exl3_gen_lp<N>(L);
    \\  for (uint tk = tk0; tk < tk1; ++tk) {
    \\    const half4 x4 = *reinterpret_cast<const device half4*>(xrow + tk * 16u);
    \\    const float xr[4] = { float(x4.x), float(x4.y), float(x4.z), float(x4.w) };
    \\    for (uint t = 0; t < BT; ++t) {
    \\      const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * BT + t) * ulong(NW);
    \\      const auto ww = exl3_tile_load<N, 0>(tile, qb, ga, lp);
    \\      uint cw[8];
    \\      cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
    \\      cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
    \\      for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(cw[m])), a[t][m >> 2]);
    \\    }
    \\  }
    \\}
    \\// the simdgroup's BT x 2 accumulators folded over the quad lanes into red[sg][16 BT] (output = tile * 16 + column)
    \\template <uint BT>
    \\inline void turboexl3_fold(thread float (&a)[BT][2], threadgroup float* red_sg, uint L) {
    \\  for (uint t = 0; t < BT; ++t) for (uint c = 0; c < 2u; ++c) { a[t][c] += simd_shuffle_xor(a[t][c], ushort(1)); a[t][c] += simd_shuffle_xor(a[t][c], ushort(2)); }
    \\  if ((L & 3u) == 0u) for (uint t = 0; t < BT; ++t) { red_sg[t * 16u + (L >> 2)] = a[t][0]; red_sg[t * 16u + (L >> 2) + 8u] = a[t][1]; }
    \\}
    \\// the input's half4 at p: device memory (a prepare launch's rows) or threadgroup memory (the prepare inside the launch)
    \\inline half4 turboexl3_x4(const device half* p) { return *reinterpret_cast<const device half4*>(p); }
    \\inline half4 turboexl3_x4(const threadgroup half* p) { return *reinterpret_cast<const threadgroup half4*>(p); }
    \\// the core over M members (row-slots that chose this expert): every weight is read and decoded once and fed to each
    \\// member's accumulators in the one-member order, so a member's bytes equal the one-member kernel's; folds into
    \\// red_sg[j * 16 BT + output]
    \\template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT, uint M, uint DEF = 0u, typename XP>
    \\inline void turboexl3_gemv_members(const device uint* wbase, XP xrow0, XP xrow1, uint tk0, uint tk1, uint blk, uint L, threadgroup float* red_sg) {
    \\  constexpr uint NW = N / 2u;
    \\  const uint qb = exl3_gen_qb<N>(L);
    \\  const uint ga = exl3_gen_a<N>(L);
    \\  const uint lp = exl3_gen_lp<N>(L);
    \\  float a[M][BT][2];
    \\  for (uint j = 0; j < M; ++j) for (uint t = 0; t < BT; ++t) { a[j][t][0] = 0.0f; a[j][t][1] = 0.0f; }
    \\  // EXL3_KU 1: input tiles in pairs, both tiles' words loaded before either decodes (2 BT tiles in flight); each tile's FMAs
    \\  // still in tile order, so the same sums
    \\  constexpr uint KU = (EXL3_KU != 0) ? 2u : 1u;
    \\  for (uint tk2 = tk0; tk2 < tk1; tk2 += KU) {
    \\    Exl3BitsM<exl3_rmode<N>()> wpre[KU][BT];
    \\    for (uint u = 0; u < KU; ++u) if (tk2 + u < tk1) for (uint t = 0; t < BT; ++t) wpre[u][t] = exl3_tile_load<N, 0>(wbase + (ulong(tk2 + u) * OUT_TILES + blk * BT + t) * ulong(NW), qb, ga, lp);
    \\  for (uint u = 0; u < KU; ++u) {
    \\    const uint tk = tk2 + u;
    \\    if (tk >= tk1) break;
    \\    const half4 x0 = turboexl3_x4(xrow0 + tk * 16u);
    \\    const float xr0[4] = { float(x0.x), float(x0.y), float(x0.z), float(x0.w) };
    \\    float xr1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    \\    if (M > 1u) { const half4 x1 = turboexl3_x4(xrow1 + tk * 16u); xr1[0] = float(x1.x); xr1[1] = float(x1.y); xr1[2] = float(x1.z); xr1[3] = float(x1.w); }
    \\    for (uint t = 0; t < BT; ++t) {
    \\      const auto ww = wpre[u][t];
    \\      uint cw[8];
    \\      cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
    \\      cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
    \\      for (uint m = 0; m < 8u; ++m) {
    \\        // DEF: MUL1's byte sum t (the finish applies A and C): no half FMA or half->float convert per weight
    \\        const float w = (DEF != 0u) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
    \\        a[0][t][m >> 2] = fma(xr0[m & 3u], w, a[0][t][m >> 2]);
    \\        if (M > 1u) a[M - 1u][t][m >> 2] = fma(xr1[m & 3u], w, a[M - 1u][t][m >> 2]);
    \\      }
    \\    }
    \\  }
    \\  }
    \\  for (uint j = 0; j < M; ++j) turboexl3_fold<BT>(a[j], red_sg + j * 16u * BT, L);
    \\}
    \\// H128 of 128 values held one per thread (tid): 5 shuffle stages in the simdgroup, 2 exchange stages through xch
    \\inline float turboexl3_h128(float v, threadgroup float* xch, uint tid, uint L) {
    \\  for (uint st = 0; st < 5u; ++st) { const uint bit = 1u << st; const float p = simd_shuffle_xor(v, ushort(bit)); v = (L & bit) ? (p - v) : (v + p); }
    \\  for (uint st = 5; st < 7u; ++st) {
    \\    const uint bit = 1u << st;
    \\    xch[tid] = v; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const float p = xch[tid ^ bit]; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    v = (tid & bit) ? (p - v) : (v + p);
    \\  }
    \\  return v;
    \\}
    \\
;

pub const header: [:0]const u8 = lane.metal_helpers ++ metal_common ++ metal_gemv_core;

const gu_inputs = [_][*:0]const u8{ "x", "slots", "gate_t", "gate_suh", "gate_svh", "up_t", "up_suh", "up_svh" };
const gu_outputs = [_][*:0]const u8{"h"};
const down_inputs = [_][*:0]const u8{ "h", "slots", "scores", "down_t", "down_suh", "down_svh" };
const down_outputs = [_][*:0]const u8{"y"};
const gu4_inputs = [_][*:0]const u8{ "xgp", "xup", "slots", "gate_t", "gate_svh", "up_t", "up_svh" };
const down4_inputs = [_][*:0]const u8{ "hdp", "slots", "scores", "down_t", "down_svh" };

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
        _ = mlx.mlx_fast_metal_kernel_free(self.gu);
        _ = mlx.mlx_fast_metal_kernel_free(self.down);
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
    const t = std.testing;
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "TOPK", "PROBE" }) |name| {
        try t.expect(std.mem.indexOf(u8, header ++ text_gu, name) != null);
        try t.expect(std.mem.indexOf(u8, header ++ text_down, name) != null);
    }
    // every input the signatures name is read by its text, and the outputs are written
    for (gu_inputs) |n| try t.expect(std.mem.indexOf(u8, text_gu, std.mem.span(n)) != null);
    for (down_inputs) |n| try t.expect(std.mem.indexOf(u8, text_down, std.mem.span(n)) != null);
    try t.expect(std.mem.indexOf(u8, text_gu, "h[(row * TOPKC + k) * INTC + o] =") != null);
    try t.expect(std.mem.indexOf(u8, text_down, "y[row * HIDC + blk * 128u + L + 32u * j] = acc[j]") != null);
}

test "turboexl3 decode geometry admission: 128-aligned widths, top-k within the reduce bank" {
    const t = std.testing;
    const ok: Geometry = .{ .hidden = 2560, .inter = 640, .topk = 10, .rate = format.Rate.fromK(5), .dec = .mul1 };
    try t.expect(ok.admitted());
    try t.expect(!(Geometry{ .hidden = 2560, .inter = 600, .topk = 10, .rate = format.Rate.fromK(5), .dec = .mul1 }).admitted());
    try t.expect(!(Geometry{ .hidden = 2560, .inter = 640, .topk = 33, .rate = format.Rate.fromK(5), .dec = .mul1 }).admitted());
}

/// Form 1b: four launches per layer. The two prepares are `unfused.text_prep` (one simdgroup per 128-block and row-slot:
/// each rotation computed once), then `text_gu4` (GEMV + finish + SwiGLU) and `text_down4` (GEMV + finish + reduce).
pub const Kernels4 = struct {
    geo: Geometry,
    prep: mlx.mlx_fast_metal_kernel,
    gu: mlx.mlx_fast_metal_kernel,
    down: mlx.mlx_fast_metal_kernel,
    cfg: [Kernels.max_rows + 1]?Cfg = @splat(null),

    const Cfg = struct { prep_h: mlx.mlx_fast_metal_kernel_config, prep_i: mlx.mlx_fast_metal_kernel_config, gu: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config };
    const prep_in = [_][*:0]const u8{ "x", "slots", "suh" };
    const prep_out = [_][*:0]const u8{"xp"};

    pub fn init(geo: Geometry) !Kernels4 {
        if (!geo.admitted()) return error.BadExl3Shape;
        const unf = @import("unfused.zig");
        return .{
            .geo = geo,
            .prep = ks.newKernel("turboexl3_prep", &prep_in, &prep_out, unf.text_prep, header, true) orelse return error.KernelCreateFailed,
            .gu = ks.newKernel("turboexl3_gu4", &gu4_inputs, &gu_outputs, text_gu4, header, true) orelse return error.KernelCreateFailed,
            .down = ks.newKernel("turboexl3_down4", &down4_inputs, &down_outputs, text_down4, header, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *Kernels4) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.prep_i, cc.gu, cc.down }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        inline for (.{ self.prep, self.gu, self.down }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
    }

    fn cfgs(self: *Kernels4, rows: usize) !Cfg {
        if (rows == 0 or rows > Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const g = self.geo;
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const t = [_]ks.TemplateArg{
            .{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } },
            .{ .name = "HID", .value = .{ .int = H } },
            .{ .name = "INTER", .value = .{ .int = I } },
            .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
            .{ .name = "PROBE", .value = .{ .int = 0 } },
        };
        const c: Cfg = .{
            .prep_h = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float16}, .template = &.{ .{ .name = "DIM", .value = .{ .int = H } }, .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } } } }),
            .prep_i = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float16}, .template = &.{ .{ .name = "DIM", .value = .{ .int = I } }, .{ .name = "TOPK", .value = .{ .int = 1 } } } }),
            .gu = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 128, g.topk, @intCast(rows) }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float32}, .template = &t }),
            .down = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 128, @intCast(rows), 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{.float32}, .template = &t }),
        };
        self.cfg[rows] = c;
        return c;
    }

    fn one(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, c, ins, &outs, s);
        return outs[0];
    }

    /// The four launches; the caller frees y.
    pub fn moe(self: *Kernels4, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize) !mlx.mlx_array {
        const c = try self.cfgs(rows);
        const xg = try one(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh }, s);
        defer _ = mlx.mlx_array_free(xg);
        const xu = try one(self.prep, c.prep_h, &.{ x, slots, bank.up_suh }, s);
        defer _ = mlx.mlx_array_free(xu);
        const h = try one(self.gu, c.gu, &.{ xg, xu, slots, bank.gate_t, bank.gate_svh, bank.up_t, bank.up_svh }, s);
        defer _ = mlx.mlx_array_free(h);
        const hd = try one(self.prep, c.prep_i, &.{ h, slots, bank.down_suh }, s);
        defer _ = mlx.mlx_array_free(hd);
        return one(self.down, c.down, &.{ hd, slots, scores, bank.down_t, bank.down_svh }, s);
    }
};

/// Form 2, launch 1: one 128-thread threadgroup per (128-wide intermediate block, row-slot); simdgroup sg handles
/// projection sg >> 1 (gate, up) over input half sg & 1. Each lane holds one projection's 8 tiles x 2 columns and loads
/// its quad's four x rows per tile as one half4 (lane-ordered xg / xu from the prepare). The two halves meet in
/// threadgroup memory; all 128 threads then round, rotate the block's outputs (5 shuffle stages + 2 threadgroup
/// exchanges), scale by svh, round and write h = silu(gate) * up.
pub const text_gu2 =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint IN_TILES = HIDC / 16u;
    \\    constexpr uint OUT_TILES = INTC / 16u;
    \\    constexpr bool INTK = (N % 16u) == 0u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint e = uint(slots[rk]);
    \\    const uint proj = sg >> 1u;          // 0 gate, 1 up
    \\    const uint khalf = sg & 1u;
    \\    threadgroup float red[4][128];       // [sg][output]
    \\    threadgroup float xch[128];          // the rotation's cross-simdgroup exchange
    \\    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    \\    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    \\    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    \\    uint off[8];
    \\    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    \\    float a[8][2];
    \\    for (uint t = 0; t < 8u; ++t) { a[t][0] = 0.0f; a[t][1] = 0.0f; }
    \\    const device uint* wbase = (const device uint*)(proj == 0u ? gate_t : up_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\    const device half* xrow = (proj == 0u ? xgp : xup) + ulong(rk) * ulong(HIDC) + 4u * (L & 3u);
    \\    const uint tk_per = IN_TILES / 2u;
    \\    for (uint tk = khalf * tk_per; tk < (khalf + 1u) * tk_per; ++tk) {
    \\        const half4 x4 = *reinterpret_cast<const device half4*>(xrow + tk * 16u);
    \\        const float xr[4] = { float(x4.x), float(x4.y), float(x4.z), float(x4.w) };
    \\        for (uint t = 0; t < 8u; ++t) {
    \\            const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * 8u + t) * ulong(NW);
    \\            if (INTK) {
    \\                const IntWords ww = exl3_int_words<N>(tile, base, ia);
    \\                uint cw[8];
    \\                cw[0] = exl3_int_cw<N, WMASK, 0>(ww, ib); cw[1] = exl3_int_cw<N, WMASK, 1>(ww, ib); cw[2] = exl3_int_cw<N, WMASK, 2>(ww, ib); cw[3] = exl3_int_cw<N, WMASK, 3>(ww, ib);
    \\                cw[4] = exl3_int_cw<N, WMASK, 4>(ww, ib); cw[5] = exl3_int_cw<N, WMASK, 5>(ww, ib); cw[6] = exl3_int_cw<N, WMASK, 6>(ww, ib); cw[7] = exl3_int_cw<N, WMASK, 7>(ww, ib);
    \\                for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(cw[m])), a[t][m >> 2]);
    \\            } else {
    \\                const LaneWords ww = exl3_lane_words<N>(tile, base);
    \\                for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(exl3_lane_cw<WMASK>(ww, off[m]))), a[t][m >> 2]);
    \\            }
    \\        }
    \\    }
    \\    for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) { a[t][c] += simd_shuffle_xor(a[t][c], ushort(1)); a[t][c] += simd_shuffle_xor(a[t][c], ushort(2)); }
    \\    if ((L & 3u) == 0u) for (uint t = 0; t < 8u; ++t) { red[sg][t * 16u + (L >> 2)] = a[t][0]; red[sg][t * 16u + (L >> 2) + 8u] = a[t][1]; }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // every thread owns output o = tid for the finish: gate (red 0 + 1) and up (red 2 + 3), f16 round, H128
    \\    float g = float(half(red[0][tid] + red[1][tid]));
    \\    float u = float(half(red[2][tid] + red[3][tid]));
    \\    for (uint st = 0; st < 5u; ++st) {
    \\        const uint bit = 1u << st;
    \\        const float pg = simd_shuffle_xor(g, ushort(bit)); const float pu = simd_shuffle_xor(u, ushort(bit));
    \\        g = (L & bit) ? (pg - g) : (g + pg); u = (L & bit) ? (pu - u) : (u + pu);
    \\    }
    \\    // stages 5 and 6 cross simdgroups (bit 5 = sg & 1, bit 6 = sg >> 1) through threadgroup memory
    \\    for (uint st = 5; st < 7u; ++st) {
    \\        const uint bit = 1u << st;
    \\        xch[tid] = g; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        const float pg = xch[tid ^ bit]; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        g = (tid & bit) ? (pg - g) : (g + pg);
    \\        xch[tid] = u; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        const float pu = xch[tid ^ bit]; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        u = (tid & bit) ? (pu - u) : (u + pu);
    \\    }
    \\    const uint o = blk * 128u + tid;
    \\    const float gv = float(half(g * HAD_SCALE * float(gate_svh[e * INTC + o])));
    \\    const float uv = float(half(u * HAD_SCALE * float(up_svh[e * INTC + o])));
    \\    h[ulong(rk) * ulong(INTC) + o] = (gv / (1.0f + metal::exp(-gv))) * uv;
    \\
;

/// Form 2, launch 2: one threadgroup per (128-wide hidden block, row-slot); the four simdgroups split the input tiles;
/// all 128 threads finish (round, rotate, svh, round) and write yk f32 [RK, hidden] scaled by the slot's score, for the
/// reduce launch (slot-order sums: deterministic bytes).
pub const text_down2 =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint IN_TILES = INTC / 16u;
    \\    constexpr uint OUT_TILES = HIDC / 16u;
    \\    constexpr bool INTK = (N % 16u) == 0u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint e = uint(slots[rk]);
    \\    threadgroup float red[4][128];
    \\    threadgroup float xch[128];
    \\    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    \\    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    \\    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    \\    uint off[8];
    \\    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    \\    float a[8][2];
    \\    for (uint t = 0; t < 8u; ++t) { a[t][0] = 0.0f; a[t][1] = 0.0f; }
    \\    const device uint* wbase = (const device uint*)(down_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\    const device half* xrow = hdp + ulong(rk) * ulong(INTC) + 4u * (L & 3u);
    \\    const uint tk_per = IN_TILES / 4u;
    \\    for (uint tk = sg * tk_per; tk < (sg + 1u) * tk_per; ++tk) {
    \\        const half4 x4 = *reinterpret_cast<const device half4*>(xrow + tk * 16u);
    \\        const float xr[4] = { float(x4.x), float(x4.y), float(x4.z), float(x4.w) };
    \\        for (uint t = 0; t < 8u; ++t) {
    \\            const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * 8u + t) * ulong(NW);
    \\            if (INTK) {
    \\                const IntWords ww = exl3_int_words<N>(tile, base, ia);
    \\                uint cw[8];
    \\                cw[0] = exl3_int_cw<N, WMASK, 0>(ww, ib); cw[1] = exl3_int_cw<N, WMASK, 1>(ww, ib); cw[2] = exl3_int_cw<N, WMASK, 2>(ww, ib); cw[3] = exl3_int_cw<N, WMASK, 3>(ww, ib);
    \\                cw[4] = exl3_int_cw<N, WMASK, 4>(ww, ib); cw[5] = exl3_int_cw<N, WMASK, 5>(ww, ib); cw[6] = exl3_int_cw<N, WMASK, 6>(ww, ib); cw[7] = exl3_int_cw<N, WMASK, 7>(ww, ib);
    \\                for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(cw[m])), a[t][m >> 2]);
    \\            } else {
    \\                const LaneWords ww = exl3_lane_words<N>(tile, base);
    \\                for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(exl3_lane_cw<WMASK>(ww, off[m]))), a[t][m >> 2]);
    \\            }
    \\        }
    \\    }
    \\    for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) { a[t][c] += simd_shuffle_xor(a[t][c], ushort(1)); a[t][c] += simd_shuffle_xor(a[t][c], ushort(2)); }
    \\    if ((L & 3u) == 0u) for (uint t = 0; t < 8u; ++t) { red[sg][t * 16u + (L >> 2)] = a[t][0]; red[sg][t * 16u + (L >> 2) + 8u] = a[t][1]; }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    float v = float(half(((red[0][tid] + red[1][tid]) + red[2][tid]) + red[3][tid]));
    \\    for (uint st = 0; st < 5u; ++st) { const uint bit = 1u << st; const float p = simd_shuffle_xor(v, ushort(bit)); v = (L & bit) ? (p - v) : (v + p); }
    \\    for (uint st = 5; st < 7u; ++st) {
    \\        const uint bit = 1u << st;
    \\        xch[tid] = v; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        const float p = xch[tid ^ bit]; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        v = (tid & bit) ? (p - v) : (v + p);
    \\    }
    \\    const uint o = blk * 128u + tid;
    \\    yk[ulong(rk) * ulong(HIDC) + o] = float(scores[rk]) * float(half(v * HAD_SCALE * float(down_svh[e * HIDC + o])));
    \\
;

/// reduce over the slots of a row (the scores already applied): y f32 [rows, hidden] = sum_k yk[r k], slot order
pub const text_reduce2 =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    const uint d = thread_position_in_grid.x;
    \\    const uint r = thread_position_in_grid.y;
    \\    float acc = 0.0f;
    \\    for (uint k = 0; k < TOPKC; ++k) acc += yk[(r * TOPKC + k) * DIMC + d];
    \\    y[r * DIMC + d] = acc;
    \\
;

const gu2_inputs = [_][*:0]const u8{ "xgp", "xup", "slots", "gate_t", "gate_svh", "up_t", "up_svh" };
const down2_inputs = [_][*:0]const u8{ "hdp", "slots", "scores", "down_t", "down_svh" };
const down2_outputs = [_][*:0]const u8{"yk"};
const reduce2_inputs = [_][*:0]const u8{"yk"};

/// Form 2: five launches per layer (prepare gate, prepare up, gate/up GEMV+finish+SwiGLU, prepare h, down GEMV+finish,
/// reduce), every threadgroup one (block, row-slot), lane-ordered prepared inputs.
pub const Kernels2 = struct {
    geo: Geometry,
    prep: mlx.mlx_fast_metal_kernel,
    gu: mlx.mlx_fast_metal_kernel,
    down: mlx.mlx_fast_metal_kernel,
    reduce: mlx.mlx_fast_metal_kernel,
    cfg: [Kernels.max_rows + 1]?Cfg = @splat(null),

    const Cfg = struct { prep_h: mlx.mlx_fast_metal_kernel_config, prep_i: mlx.mlx_fast_metal_kernel_config, gu: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config, reduce: mlx.mlx_fast_metal_kernel_config };
    const prep_in = [_][*:0]const u8{ "x", "slots", "suh" };
    const prep_out = [_][*:0]const u8{"xp"};

    pub fn init(geo: Geometry) !Kernels2 {
        if (!geo.admitted()) return error.BadExl3Shape;
        const unf = @import("unfused.zig");
        return .{
            .geo = geo,
            .prep = ks.newKernel("turboexl3_prep", &prep_in, &prep_out, unf.text_prep, header, true) orelse return error.KernelCreateFailed,
            .gu = ks.newKernel("turboexl3_gu2", &gu2_inputs, &gu_outputs, text_gu2, header, true) orelse return error.KernelCreateFailed,
            .down = ks.newKernel("turboexl3_down2", &down2_inputs, &down2_outputs, text_down2, header, true) orelse return error.KernelCreateFailed,
            .reduce = ks.newKernel("turboexl3_reduce2", &reduce2_inputs, &down_outputs, text_reduce2, header, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *Kernels2) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.prep_i, cc.gu, cc.down, cc.reduce }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        inline for (.{ self.prep, self.gu, self.down, self.reduce }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
    }

    fn cfgs(self: *Kernels2, rows: usize) !Cfg {
        if (rows == 0 or rows > Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const g = self.geo;
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const t = [_]ks.TemplateArg{
            .{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } },
            .{ .name = "HID", .value = .{ .int = H } },
            .{ .name = "INTER", .value = .{ .int = I } },
        };
        const c: Cfg = .{
            .prep_h = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float16}, .template = &.{ .{ .name = "DIM", .value = .{ .int = H } }, .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } } } }),
            .prep_i = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float16}, .template = &.{ .{ .name = "DIM", .value = .{ .int = I } }, .{ .name = "TOPK", .value = .{ .int = 1 } } } }),
            .gu = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 128, rk, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float32}, .template = &t }),
            .down = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 128, rk, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float32}, .template = &t }),
            .reduce = try ks.buildConfig(.{ .grid = .{ g.hidden, @intCast(rows), 1 }, .threadgroup = .{ 256, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{.float32}, .template = &.{ .{ .name = "DIM", .value = .{ .int = H } }, .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } } } }),
        };
        self.cfg[rows] = c;
        return c;
    }

    fn one(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, c, ins, &outs, s);
        return outs[0];
    }

    /// The five launches; the caller frees y.
    pub fn moe(self: *Kernels2, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize) !mlx.mlx_array {
        const c = try self.cfgs(rows);
        const xg = try one(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh }, s);
        defer _ = mlx.mlx_array_free(xg);
        const xu = try one(self.prep, c.prep_h, &.{ x, slots, bank.up_suh }, s);
        defer _ = mlx.mlx_array_free(xu);
        const h = try one(self.gu, c.gu, &.{ xg, xu, slots, bank.gate_t, bank.gate_svh, bank.up_t, bank.up_svh }, s);
        defer _ = mlx.mlx_array_free(h);
        const hd = try one(self.prep, c.prep_i, &.{ h, slots, bank.down_suh }, s);
        defer _ = mlx.mlx_array_free(hd);
        const yk = try one(self.down, c.down, &.{ hd, slots, scores, bank.down_t, bank.down_svh }, s);
        defer _ = mlx.mlx_array_free(yk);
        return one(self.reduce, c.reduce, &.{yk}, s);
    }
};

/// Form 3 (split-K planes). Launch 2: one threadgroup per (128-wide intermediate block, row-slot, group of SPT K slices);
/// the K partition into KS slices is the geometry's alone (a row's summation order never depends on how many rows share
/// the call), the row count only sets how many threadgroups share a row-slot's slices (KS / SPT). The four simdgroups are
/// projection x half of a slice; each slice's partial sums go to its plane of zg / zu f32 [KS, RK, inter] (written once
/// each; no atomics).
pub const text_gu3 =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr uint SPTC = uint(SPT);   // K slices per threadgroup (of the KS the partition fixes)
    \\    constexpr uint BTC = uint(BT);     // output tiles per threadgroup: 2 BT f32 accumulators per lane
    \\    constexpr uint IN_TILES = HIDC / 16u;
    \\    constexpr uint OUT_TILES = INTC / 16u;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint kz = threadgroup_position_in_grid.z;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    constexpr uint GRP = uint(GROUP);   // row-slots per lead threadgroup (1 or 2)
    \\    const uint e = uint(slots[rk]);
    \\    const uint proj = sg >> 1u;
    \\    const uint khalf = sg & 1u;
    \\    uint members[2]; uint m;
    \\    if (!turboexl3_members<GRP, uint(RKC)>(slots, rk, e, L, members, m)) return;
    \\    threadgroup float red[SPTC][4][2u * 16u * BTC];   // one reduction buffer per slice: a single barrier for all
    \\    const device uint* wbase = (const device uint*)(proj == 0u ? gate_t : up_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(N / 2u);
    \\    const device half* xbase = (proj == 0u ? xgp : xup) + 4u * (L & 3u);
    \\    const device half* xrow0 = xbase + ulong(members[0]) * ulong(HIDC);
    \\    const device half* xrow1 = xbase + ulong(members[1]) * ulong(HIDC);
    \\    const uint per = IN_TILES / (KSC * 2u);
    \\    for (uint si = 0; si < SPTC; ++si) {
    \\        const uint tk0 = ((kz * SPTC + si) * 2u + khalf) * per;
    \\        if (m == 1u) turboexl3_gemv_members<N, WMASK, CB, OUT_TILES, BTC, 1u, uint(DEFER)>(wbase, xrow0, xrow0, tk0, tk0 + per, blk, L, red[si][sg]);
    \\        else turboexl3_gemv_members<N, WMASK, CB, OUT_TILES, BTC, 2u, uint(DEFER)>(wbase, xrow0, xrow1, tk0, tk0 + per, blk, L, red[si][sg]);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (tid < 16u * BTC * m) {
    \\        const uint j = tid / (16u * BTC);
    \\        for (uint si = 0; si < SPTC; ++si) {
    \\            const ulong o = (ulong(kz * SPTC + si) * ulong(RKC) + members[j]) * ulong(INTC) + blk * (16u * BTC) + (tid - j * 16u * BTC);
    \\            zg[o] = red[si][0][tid] + red[si][1][tid];
    \\            zu[o] = red[si][2][tid] + red[si][3][tid];
    \\        }
    \\    }
    \\
;

/// Form 3, launch 3: per (128-wide intermediate block, row-slot): sum the planes, f16 round, H128, svh, f16 round,
/// SwiGLU, then the down's input rotation (f16 round, down suh, H128, f16 round) into hd in lane order.
pub const text_fin3 =
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint e = uint(slots[rk]);
    \\    threadgroup float xch[128];
    \\    const uint o = blk * 128u + tid;
    \\    float g = 0.0f, u = 0.0f;
    \\    for (uint ks = 0; ks < KSC; ++ks) { const ulong i = (ulong(ks) * ulong(RKC) + rk) * ulong(INTC) + o; g += zg[i]; u += zu[i]; }
    \\    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    \\    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    \\    if (DEFER != 0) {
    \\        float sgx = 0.0f, sux = 0.0f;
    \\        for (uint q = 0; q < uint(NBH); ++q) { sgx += xsg[ulong(rk) * uint(NBH) + q]; sux += xsu[ulong(rk) * uint(NBH) + q]; }
    \\        g = fma(g, MUL1_A, MUL1_C * sgx);
    \\        u = fma(u, MUL1_A, MUL1_C * sux);
    \\    }
    \\    g = turboexl3_h128(float(half(g)), xch, tid, L);
    \\    u = turboexl3_h128(float(half(u)), xch, tid, L);
    \\    const float gv = float(half(g * HAD_SCALE * float(gate_svh[e * INTC + o])));
    \\    const float uv = float(half(u * HAD_SCALE * float(up_svh[e * INTC + o])));
    \\    const float hv = (gv / (1.0f + metal::exp(-gv))) * uv;
    \\    // the down's prepare: f16 round, down suh, H128, f16 round, stored in lane order within the 16-wide tile
    \\    float d = float(half(hv)) * float(down_suh[e * INTC + o]);
    \\    d = turboexl3_h128(d, xch, tid, L);
    \\    const uint r = o & 15u;
    \\    const uint q = (r & 7u) >> 1u;
    \\    const uint m = (r & 1u) | ((r >> 3u) << 1u);
    \\    const half hh = half(d * HAD_SCALE);
    \\    hd[ulong(rk) * ulong(INTC) + (o & ~15u) + 4u * q + m] = hh;
    \\    // this 128-block's sum of the down's rounded input (the down finish's deferred MUL1 map): 4 simdgroup sums via xch
    \\    const float hsum = simd_sum(float(hh));
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (L == 0u) xch[simdgroup_index_in_threadgroup] = hsum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (tid == 0u) hs[ulong(rk) * ulong(INTC / 128u) + blk] = ((xch[0] + xch[1]) + xch[2]) + xch[3];
    \\
;

/// Form 3, launch 4: the down GEMV as split-K planes zd f32 [KS2, RK, hidden], the partition fixed per geometry and SPT
/// slices per threadgroup as in launch 2.
pub const text_down3 =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr uint SPTC = uint(SPT);   // K slices per threadgroup (of the KS the partition fixes)
    \\    constexpr uint BTC = uint(BT);     // output tiles per threadgroup: 2 BT f32 accumulators per lane
    \\    constexpr uint IN_TILES = INTC / 16u;
    \\    constexpr uint OUT_TILES = HIDC / 16u;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint kz = threadgroup_position_in_grid.z;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    constexpr uint GRP = uint(GROUP);   // row-slots per lead threadgroup (1 or 2)
    \\    const uint e = uint(slots[rk]);
    \\    uint members[2]; uint m;
    \\    if (!turboexl3_members<GRP, uint(RKC)>(slots, rk, e, L, members, m)) return;
    \\    threadgroup float red[SPTC][4][2u * 16u * BTC];   // one reduction buffer per slice: a single barrier for all
    \\    const device uint* wbase = (const device uint*)(down_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(N / 2u);
    \\    const device half* xbase = hdp + 4u * (L & 3u);
    \\    const device half* xrow0 = xbase + ulong(members[0]) * ulong(INTC);
    \\    const device half* xrow1 = xbase + ulong(members[1]) * ulong(INTC);
    \\    const uint per = IN_TILES / (KSC * 4u);
    \\    for (uint si = 0; si < SPTC; ++si) {
    \\        const uint tk0 = ((kz * SPTC + si) * 4u + sg) * per;
    \\        if (m == 1u) turboexl3_gemv_members<N, WMASK, CB, OUT_TILES, BTC, 1u, uint(DEFER)>(wbase, xrow0, xrow0, tk0, tk0 + per, blk, L, red[si][sg]);
    \\        else turboexl3_gemv_members<N, WMASK, CB, OUT_TILES, BTC, 2u, uint(DEFER)>(wbase, xrow0, xrow1, tk0, tk0 + per, blk, L, red[si][sg]);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (tid < 16u * BTC * m) {
    \\        const uint j = tid / (16u * BTC);
    \\        for (uint si = 0; si < SPTC; ++si)
    \\            zd[(ulong(kz * SPTC + si) * ulong(RKC) + members[j]) * ulong(HIDC) + blk * (16u * BTC) + (tid - j * 16u * BTC)] = ((red[si][0][tid] + red[si][1][tid]) + red[si][2][tid]) + red[si][3][tid];
    \\    }
    \\
;

/// Form 3, launch 5: per (128-wide hidden block, row): for each slot in order sum the planes, f16 round, H128, svh,
/// f16 round, times the score; y f32 [rows, hidden].
pub const text_dfin3 =
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    threadgroup float xch[128];
    \\    const uint o = blk * 128u + tid;
    \\    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    \\    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    \\    float acc = 0.0f;
    \\    for (uint k = 0; k < TOPKC; ++k) {
    \\        const uint rk = row * TOPKC + k;
    \\        const uint e = uint(slots[rk]);
    \\        float v = 0.0f;
    \\        for (uint ks = 0; ks < KSC; ++ks) v += zd[(ulong(ks) * ulong(RKC) + rk) * ulong(HIDC) + o];
    \\        if (DEFER != 0) {
    \\            float hsum = 0.0f;
    \\            for (uint q = 0; q < uint(NBI); ++q) hsum += hsx[ulong(rk) * uint(NBI) + q];
    \\            v = fma(v, MUL1_A, MUL1_C * hsum);
    \\        }
    \\        v = turboexl3_h128(float(half(v)), xch, tid, L);
    \\        acc = fma(float(scores[rk]), float(half(v * HAD_SCALE * float(down_svh[e * HIDC + o]))), acc);
    \\    }
    \\    y[ulong(row) * ulong(HIDC) + o] = OUT_T(acc);   // the caller's dtype (bf16 served, f32 in the tests)
    \\
;

/// Form 3's prepare for gate and up in one launch: xg, xu half [RK, DIM] in lane order from x [rows, DIM] and the two suh.
pub const text_prep2 =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint e = uint(slots[rk]);
    \\    const uint row = rk / TOPKC;
    \\    float vg[4]; float vu[4];
    \\    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; const float xr = float(half(float(x[row * DIMC + i]))); vg[j] = xr * float(gate_suh[e * DIMC + i]); vu[j] = xr * float(up_suh[e * DIMC + i]); }
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float pg = simd_shuffle_xor(vg[j], ushort(bit)); const float pu = simd_shuffle_xor(vu[j], ushort(bit)); vg[j] = (L & bit) ? (pg - vg[j]) : (vg[j] + pg); vu[j] = (L & bit) ? (pu - vu[j]) : (vu[j] + pu); } }
    \\    { float t0 = vg[0] + vg[1], t1 = vg[0] - vg[1], t2 = vg[2] + vg[3], t3 = vg[2] - vg[3]; vg[0] = t0 + t2; vg[1] = t1 + t3; vg[2] = t0 - t2; vg[3] = t1 - t3; }
    \\    { float t0 = vu[0] + vu[1], t1 = vu[0] - vu[1], t2 = vu[2] + vu[3], t3 = vu[2] - vu[3]; vu[0] = t0 + t2; vu[1] = t1 + t3; vu[2] = t0 - t2; vu[3] = t1 - t3; }
    \\    float sgv = 0.0f, suv = 0.0f;
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint i = b * 128u + L + 32u * j;
    \\        const uint r = i & 15u; const uint q = (r & 7u) >> 1u; const uint m = (r & 1u) | ((r >> 3u) << 1u);
    \\        const ulong o = ulong(rk) * ulong(DIMC) + (i & ~15u) + 4u * q + m;
    \\        const half hg = half(vg[j] * HAD_SCALE);
    \\        const half hu = half(vu[j] * HAD_SCALE);
    \\        xg[o] = hg;
    \\        xu[o] = hu;
    \\        sgv += float(hg);
    \\        suv += float(hu);
    \\    }
    \\    // each 128-block's sum of the rounded inputs (the deferred MUL1 map needs the row-slot's input sum)
    \\    sgv = simd_sum(sgv);
    \\    suv = simd_sum(suv);
    \\    if (L == 0u) { sg[ulong(rk) * ulong(DIMC / 128u) + b] = sgv; su[ulong(rk) * ulong(DIMC / 128u) + b] = suv; }
    \\
;
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
    /// (MLX_SERVE_TURBOEXL3_DEFER=0 decodes every weight to its f16 value, the reference's rounding)
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
    const SCfg = struct { gu: mlx.mlx_fast_metal_kernel_config, fin: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config, dfin: mlx.mlx_fast_metal_kernel_config, gu8: bool };
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
            .defer_affine = geo.dec.codebook == .mul1 and blk: {
                const v = std.c.getenv("MLX_SERVE_TURBOEXL3_DEFER") orelse break :blk true;
                break :blk v[0] != '0';
            },
            .prep = ks.newKernel("turboexl3_prep2", &prep2_inputs, &prep2_outputs, text_prep2, header, true) orelse return error.KernelCreateFailed,
            .gu = ks.newKernel("turboexl3_gu3", &gu3_inputs, &gu3_outputs, text_gu3, header, true) orelse return error.KernelCreateFailed,
            .fin = ks.newKernel("turboexl3_fin3", &fin3_inputs, &fin3_outputs, text_fin3, header, true) orelse return error.KernelCreateFailed,
            .down = ks.newKernel("turboexl3_down3", &down3_inputs, &down3_outputs, text_down3, header, true) orelse return error.KernelCreateFailed,
            .dfin = ks.newKernel("turboexl3_dfin3", &dfin3_inputs, &down_outputs, text_dfin3, header, true) orelse return error.KernelCreateFailed,
            .fin7 = ks.newKernel("turboexl3_fin7", &fin3_inputs, &fin3_outputs, text_fin7, header5, true) orelse return error.KernelCreateFailed,
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
            inline for (.{ cc.gu, cc.fin, cc.down, cc.dfin }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
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
            _ = mlx.mlx_fast_metal_kernel_free(kk);
        };
        inline for (.{ self.prep, self.gu, self.fin, self.down, self.dfin, self.fin7, self.dfin7, self.gu8, self.gu3s, self.gu8s, self.fin7s, self.down3s, self.dfin7s, self.sgu, self.sfin, self.sdown }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
        inline for (.{ self.gu8t, self.gu3t, self.gu3p, self.prep2t, self.fin7t, self.down3t, self.dfin7t, self.sgu2, self.sgu2p, self.sprep2, self.sfin2, self.sdown2 }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
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
        const bw: u32 = 16 * bt;
        return .{
            .prep_h = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{ &[_]c_int{ rkc, H }, &[_]c_int{ rkc, H }, &[_]c_int{ rkc, @intCast(g.hidden / 128) }, &[_]c_int{ rkc, @intCast(g.hidden / 128) } }, .out_dtypes = &.{ .float16, .float16, .float32, .float32 }, .template = &.{ .{ .name = "DIM", .value = .{ .int = H } }, topk } }),
            .gu = try ks.buildConfig(.{ .grid = .{ (g.inter / bw) * 128, rk, z_gu }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(self.ks_gu), rkc, I } }, .out_dtypes = &.{ .float32, .float32 }, .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t } }),
            .fin = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 128, rk, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, @intCast(g.inter / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = &.{ inter_t, rkt, ks_gu_t, def_t, nbh_t } }),
            .down = try ks.buildConfig(.{ .grid = .{ (g.hidden / bw) * 128, rk, z_dn }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(self.ks_dn), rkc, H }}, .out_dtypes = &.{.float32}, .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_dn_t, spt_dn_t, bt_t, grp_t, def_t } }),
            .dfin = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 128, @intCast(rows), 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{self.out_dtype}, .template = &.{ hid, topk, rkt, ks_dn_t, out_t, def_t, nbi_t } }),
            .fin7 = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, @intCast(g.inter / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = &.{ inter_t, rkt, ks_gu_t, def_t, nbh_t } }),
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
    /// several slices), in its own launch shape (bt1, bt9, kdiv9). Every form, form 3's bytes.
    pub fn moeForm(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, form: u8) !mlx.mlx_array {
        const one_row9 = form == 9 and rows <= self.gu8_rows;
        const c = if (form == 9) try self.cfgs9(rows) else try self.cfgs(rows);
        const f7 = form >= 7;
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
        return one(if (f7) self.dfin7 else self.dfin, if (f7) c.dfin7 else c.dfin, &.{ zd, slots, scores, bank.down_svh, hd[1] }, s);
    }

    /// Whether `moeShared` carries this shared expert at `rows`: up to `shared_rows_max` rows, the routed width as its
    /// intermediate, gate|up K slices of exactly one 128-block of hidden (the heads rotate their block in-kernel: dense prep_s's
    /// span), a down split of whole 4-part slices, and the rows' bf16 or f16 output (the SwiGLU table's dtype).
    pub fn sharedAdmitted(self: *const Kernels3, rows: usize, sb: SharedBank) bool {
        const g = self.geo;
        if (rows == 0 or rows > shared_rows_max) return false;
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
            inline for (.{ c.gu, c.fin, c.down, c.dfin }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
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
        const c: SCfg = .{ .gu = gu, .fin = fin, .down = down, .dfin = dfin, .gu8 = use_gu8 };
        self.scfg[rows] = c;
        self.scfg_key[rows] = key;
        return c;
    }

    /// Form 9 with the shared expert folded in: y = the routed sum in the output dtype plus sigmoid(logit) * shared(x), in the
    /// bytes of the host's chain (form 3's routed y, the three dense linears, the table SwiGLU, MLX's sigmoid, Multiply, Add)
    /// with no launch of its own: the shared GEMVs ride the gate|up and down launches as extra grid rows, its finishes ride
    /// launches 3 and 5. The caller frees y; `sharedAdmitted` first.
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
    /// gu3s, fin7s, down3s and dfin7s unchanged on the launch's slots and scores and, with `gate_vec`, its gate logit
    /// in `sb.logit`'s place. The router kernel's and `moeShared`'s bytes, one launch and one dependent level fewer.
    /// error.Exl3EngineNotImplemented at the rows whose prepare rides the gate|up launch (gu8s: no prepare launch to fold
    /// into); `sharedAdmitted` first. The caller frees y.
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
const metal_form5 =
    \\// coherent(device) (Metal 3.2+): another threadgroup's device stores are only guaranteed visible through it (form 6's
    \\// arrival reduce; C33's idiom)
    \\#if defined(__HAVE_COHERENT__)
    \\#define TURBOEXL3_COH coherent(device)
    \\#else
    \\#define TURBOEXL3_COH
    \\#endif
    \\// the row-slots that chose expert e, in slot order: slot rk leads when its occurrence index is a multiple of GM (1..4) and
    \\// takes the next GM - 1 occurrences as its members (top-k picks distinct experts per row, so members are other rows'
    \\// slots); false for a slot some lead covers
    \\template <uint GM, uint RKC, typename SP>
    \\inline bool turboexl3_group(SP slots, uint rk, uint e, uint L, thread uint (&mem)[4], thread uint& m) {
    \\  mem[0] = rk; mem[1] = rk; mem[2] = rk; mem[3] = rk; m = 1u;
    \\  if (GM == 1u) return true;
    \\  uint before = 0u;
    \\  for (uint b = 0u; b < RKC; b += 32u) {
    \\    const uint i = b + L;
    \\    const bool hit = (i < RKC) && (i != rk) && (uint(slots[i]) == e);
    \\    const uint mask = uint(static_cast<simd_vote::vote_t>(simd_ballot(hit)));
    \\    const uint below = (b + 32u <= rk) ? 0xffffffffu : ((b >= rk) ? 0u : ((1u << (rk - b)) - 1u));
    \\    before += popcount(mask & below);
    \\    uint above = mask & ~below;
    \\    while (above != 0u && m < GM) { mem[m] = b + ctz(above); m += 1u; above &= above - 1u; }
    \\  }
    \\  return (before % GM) == 0u;
    \\}
    \\template <uint N, uint WMASK, uint MM, int IR> inline uint turboexl3_cw5(IntWords w, uint lp) {
    \\  if (IR != 0) return exl3_int_cw<N, WMASK, MM>(w, lp);
    \\  return exl3_gen_cw<N, WMASK, MM>(w, lp);
    \\}
    \\// lane L's BT output tiles (ot0..) x 2 columns over input tiles [tk0, tk1) of expert `wbase` for M members, member j's
    \\// prepared input in threadgroup memory at xj (lane order, offset 4 (L & 3), element 0 = input tile xtk0): every weight is
    \\// read and decoded once and fed to each member in the one-member order, so a member's sums are the one-member kernel's
    \\template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT, uint M, uint DEF, int IR>
    \\inline void turboexl3_gemv_tg(const device uint* wbase, const threadgroup half* x0, const threadgroup half* x1, const threadgroup half* x2, const threadgroup half* x3, uint tk0, uint tk1, uint xtk0, uint ot0, uint L, thread float (&a)[M][BT][2]) {
    \\  constexpr uint NW = N / 2u;
    \\  const uint qb = (IR != 0) ? exl3_int_qb<N>(L) : exl3_gen_qb<N>(L);
    \\  const uint ga = (IR != 0) ? exl3_int_a<N>(L) : exl3_gen_a<N>(L);
    \\  const uint lp = (IR != 0) ? exl3_int_b<N>(L) : exl3_gen_lp<N>(L);
    \\  for (uint tk = tk0; tk < tk1; ++tk) {
    \\    float xv[M][4];
    \\    for (uint j = 0; j < M; ++j) {
    \\      const threadgroup half* xp = (j == 0u) ? x0 : ((j == 1u) ? x1 : ((j == 2u) ? x2 : x3));
    \\      const half4 h = *reinterpret_cast<const threadgroup half4*>(xp + (tk - xtk0) * 16u);
    \\      xv[j][0] = float(h.x); xv[j][1] = float(h.y); xv[j][2] = float(h.z); xv[j][3] = float(h.w);
    \\    }
    \\    for (uint t = 0; t < BT; ++t) {
    \\      const device uint* tile = wbase + (ulong(tk) * OUT_TILES + ot0 + t) * ulong(NW);
    \\      const auto ww = exl3_tile_load<N, IR>(tile, qb, ga, lp);
    \\      uint cw[8];
    \\      cw[0] = exl3_tile_cw<N, WMASK, 0, IR>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, IR>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, IR>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, IR>(ww, lp);
    \\      cw[4] = exl3_tile_cw<N, WMASK, 4, IR>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, IR>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, IR>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, IR>(ww, lp);
    \\      for (uint mm = 0; mm < 8u; ++mm) {
    \\        const float w = (DEF != 0u) ? exl3_mul1_t(cw[mm]) : float(exl3_decode<CB>(cw[mm]));
    \\        for (uint j = 0; j < M; ++j) a[j][t][mm >> 2] = fma(xv[j][mm & 3u], w, a[j][t][mm >> 2]);
    \\      }
    \\    }
    \\  }
    \\}
    \\// H128 of a simdgroup's 128 values, lane L holding elements L + 32 q: turboexl3_h128's stages in its order (5 shuffle stages,
    \\// then bits 5 and 6 between the registers), so the same f32 values
    \\inline void turboexl3_h128r(thread float (&v)[4], uint L) {
    \\  for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint q = 0; q < 4u; ++q) { const float p = simd_shuffle_xor(v[q], ushort(bit)); v[q] = (L & bit) ? (p - v[q]) : (v[q] + p); } }
    \\  const float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3];
    \\  v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3;
    \\}
    \\// the gate|up GEMV of one simdgroup for M members, folded over the quad lanes into red_sg[j * 16 BT + output]
    \\template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT, uint M, uint DEF, int IR>
    \\inline void turboexl3_gu5_sg(const device uint* wbase, const threadgroup half* xs0, uint stride, uint tk0, uint tk1, uint xtk0, uint ot0, uint L, threadgroup float* red_sg) {
    \\  float a[M][BT][2];
    \\  for (uint j = 0; j < M; ++j) for (uint t = 0; t < BT; ++t) { a[j][t][0] = 0.0f; a[j][t][1] = 0.0f; }
    \\  turboexl3_gemv_tg<N, WMASK, CB, OUT_TILES, BT, M, DEF, IR>(wbase, xs0, xs0 + stride, xs0 + 2u * stride, xs0 + 3u * stride, tk0, tk1, xtk0, ot0, L, a);
    \\  for (uint j = 0; j < M; ++j) turboexl3_fold<BT>(a[j], red_sg + j * 16u * BT, L);
    \\}
    \\// one output tile's GEMV over NT input tiles from tk0 for M members, input tiles taken in pairs: both tiles' words load
    \\// before either decodes (two tiles in flight per simdgroup), each tile's FMAs in turboexl3_gemv_tg's order (tile tk before
    \\// tk + 1, weights m = 0..7), so the same sums
    \\template <uint N, uint WMASK, int CB, uint OUT_TILES, uint M, uint DEF, int IR, uint NT>
    \\inline void turboexl3_gemv_pair(const device uint* wbase, const threadgroup half* x0, const threadgroup half* x1, const threadgroup half* x2, const threadgroup half* x3, uint tk0, uint ot, uint L, thread float (&a)[M][2]) {
    \\  constexpr uint NW = N / 2u;
    \\  const uint qb = (IR != 0) ? exl3_int_qb<N>(L) : exl3_gen_qb<N>(L);
    \\  const uint ga = (IR != 0) ? exl3_int_a<N>(L) : exl3_gen_a<N>(L);
    \\  const uint lp = (IR != 0) ? exl3_int_b<N>(L) : exl3_gen_lp<N>(L);
    \\  for (uint i = 0; i < NT; i += 2u) {
    \\    const uint nt = (i + 1u < NT) ? 2u : 1u;
    \\    Exl3BitsM<exl3_rmode<N>()> ww[2];
    \\    for (uint u = 0; u < 2u; ++u) if (u < nt) ww[u] = exl3_tile_load<N, IR>(wbase + (ulong(tk0 + i + u) * OUT_TILES + ot) * ulong(NW), qb, ga, lp);
    \\    for (uint u = 0; u < 2u; ++u) if (u < nt) {
    \\      const uint tk = tk0 + i + u;
    \\      float xv[M][4];
    \\      for (uint j = 0; j < M; ++j) {
    \\        const threadgroup half* xp = (j == 0u) ? x0 : ((j == 1u) ? x1 : ((j == 2u) ? x2 : x3));
    \\        const half4 h = *reinterpret_cast<const threadgroup half4*>(xp + tk * 16u);
    \\        xv[j][0] = float(h.x); xv[j][1] = float(h.y); xv[j][2] = float(h.z); xv[j][3] = float(h.w);
    \\      }
    \\      uint cw[8];
    \\      cw[0] = exl3_tile_cw<N, WMASK, 0, IR>(ww[u], lp); cw[1] = exl3_tile_cw<N, WMASK, 1, IR>(ww[u], lp); cw[2] = exl3_tile_cw<N, WMASK, 2, IR>(ww[u], lp); cw[3] = exl3_tile_cw<N, WMASK, 3, IR>(ww[u], lp);
    \\      cw[4] = exl3_tile_cw<N, WMASK, 4, IR>(ww[u], lp); cw[5] = exl3_tile_cw<N, WMASK, 5, IR>(ww[u], lp); cw[6] = exl3_tile_cw<N, WMASK, 6, IR>(ww[u], lp); cw[7] = exl3_tile_cw<N, WMASK, 7, IR>(ww[u], lp);
    \\      for (uint mm = 0; mm < 8u; ++mm) {
    \\        const float w = (DEF != 0u) ? exl3_mul1_t(cw[mm]) : float(exl3_decode<CB>(cw[mm]));
    \\        for (uint j = 0; j < M; ++j) a[j][mm >> 2] = fma(xv[j][mm & 3u], w, a[j][mm >> 2]);
    \\      }
    \\    }
    \\  }
    \\}
    \\// the down GEMV of one simdgroup (one output tile ot) for M members over every input tile in down3's K partition: KS2
    \\// slices of four parts of PER2 tiles, each part from zero and folded over the quad lanes, a slice the left sum of its
    \\// parts, the total the left sum of the slices from zero (down3 then dfin3's order); every lane ends with the sums
    \\template <uint N, uint WMASK, int CB, uint OUT_TILES, uint M, uint DEF, int IR, uint KS2C, uint PER2>
    \\inline void turboexl3_dn5_sg(const device uint* wbase, const threadgroup half* xs0, uint stride, uint ot, uint L, thread float (&tot)[4][2]) {
    \\  for (uint j = 0; j < 4u; ++j) { tot[j][0] = 0.0f; tot[j][1] = 0.0f; }
    \\  for (uint si = 0; si < KS2C; ++si) {
    \\    float sl[M][2];
    \\    for (uint p = 0; p < 4u; ++p) {
    \\      float a[M][2];
    \\      for (uint j = 0; j < M; ++j) { a[j][0] = 0.0f; a[j][1] = 0.0f; }
    \\      const uint tk0 = (si * 4u + p) * PER2;
    \\      turboexl3_gemv_pair<N, WMASK, CB, OUT_TILES, M, DEF, IR, PER2>(wbase, xs0, xs0 + stride, xs0 + 2u * stride, xs0 + 3u * stride, tk0, ot, L, a);
    \\      for (uint j = 0; j < M; ++j) for (uint c = 0; c < 2u; ++c) {
    \\        float v = a[j][c];
    \\        v += simd_shuffle_xor(v, ushort(1)); v += simd_shuffle_xor(v, ushort(2));
    \\        sl[j][c] = (p == 0u) ? v : (sl[j][c] + v);
    \\      }
    \\    }
    \\    for (uint j = 0; j < M; ++j) for (uint c = 0; c < 2u; ++c) tot[j][c] += sl[j][c];
    \\  }
    \\}
    \\
;

pub const header5: [:0]const u8 = lane.metal_helpers ++ metal_common ++ metal_gemv_core ++ metal_form5;

/// The shared expert inside the routed launches (`Kernels3.moeShared`): dense.zig's decode-row arithmetic (prep_s, dgemv,
/// dfinish), so the shared expert's bytes are the dense linears'.
const metal_shared =
    \\// one K slice of dense.zig's text_dgemv for RS rows: input tiles [tk0, tk0 + PER) of trellis wbase (OUT_TILES output
    \\// tiles), BT output tiles from blk * BT, row r's prepared input at x0 + r * XSTRIDE (lane order, offset 4 (L & 3), element
    \\// 0 = tile tk0); every weight decoded once and applied to each row in tile order, folded into red_sg[r * 16 BT + output]
    \\template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT, uint RS, uint DEF, uint PER, uint XSTRIDE, typename XP>
    \\inline void turboexl3_sdgemv(const device uint* wbase, XP x0, uint tk0, uint blk, uint L, threadgroup float* red_sg) {
    \\  constexpr uint NW = N / 2u;
    \\  const uint qb = exl3_gen_qb<N>(L);
    \\  const uint ga = exl3_gen_a<N>(L);
    \\  const uint lp = exl3_gen_lp<N>(L);
    \\  float a[RS][BT][2];
    \\  for (uint r = 0; r < RS; ++r) for (uint t = 0; t < BT; ++t) { a[r][t][0] = 0.0f; a[r][t][1] = 0.0f; }
    \\  for (uint i = 0; i < PER; ++i) {
    \\    const uint tk = tk0 + i;
    \\    float xr[RS][4];
    \\    for (uint r = 0; r < RS; ++r) {
    \\      const half4 x4 = turboexl3_x4(x0 + r * XSTRIDE + i * 16u);
    \\      xr[r][0] = float(x4.x); xr[r][1] = float(x4.y); xr[r][2] = float(x4.z); xr[r][3] = float(x4.w);
    \\    }
    \\    for (uint t = 0; t < BT; ++t) {
    \\      const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * BT + t) * ulong(NW);
    \\      const auto ww = exl3_tile_load<N, 0>(tile, qb, ga, lp);
    \\      uint cw[8];
    \\      cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
    \\      cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
    \\      for (uint m = 0; m < 8u; ++m) {
    \\        const float wv = (DEF != 0u) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
    \\        for (uint r = 0; r < RS; ++r) a[r][t][m >> 2] = fma(xr[r][m & 3u], wv, a[r][t][m >> 2]);
    \\      }
    \\    }
    \\  }
    \\  for (uint r = 0; r < RS; ++r) turboexl3_fold<BT>(a[r], red_sg + r * 16u * BT, L);
    \\}
    \\// dense.zig's text_prep_s for one 128-block b of one row, v[q] = float(half(x[b 128 + L + 32 q])) on entry: * suh, H128,
    \\// f16; writes the lane-order block to dst (device or threadgroup) and returns the simdgroup's sum of the rounded values
    \\template <typename SU, typename DP>
    \\inline float turboexl3_sprep(thread float (&v)[4], SU suh, uint b, uint L, DP dst) {
    \\  constexpr float HAD_SCALE = 0.08838834764831845f;
    \\  for (uint q = 0; q < 4u; ++q) v[q] = v[q] * float(suh[b * 128u + L + 32u * q]);
    \\  turboexl3_h128r(v, L);
    \\  float sum = 0.0f;
    \\  for (uint q = 0; q < 4u; ++q) {
    \\    const uint i = L + 32u * q;
    \\    const uint r = i & 15u; const uint qq = (r & 7u) >> 1u; const uint mm = (r & 1u) | ((r >> 3u) << 1u);
    \\    const half hv = half(v[q] * HAD_SCALE);
    \\    dst[(i & ~15u) + 4u * qq + mm] = hv;
    \\    sum += float(hv);
    \\  }
    \\  return simd_sum(sum);
    \\}
    \\// dense.zig's text_dfinish for one 128-block of one row: the KS planes z[(k RS + row) DIM + o] summed in slice order, the
    \\// deferred MUL1 map with the row's input sum xsum, f16, H128, svh, f16 (the caller converts to the output dtype)
    \\template <uint KS, uint RS, uint DIM, uint DEF, typename SV>
    \\inline void turboexl3_sfinish(const device float* z, SV svh, float xsum, uint row, uint b, uint L, thread float (&v)[4]) {
    \\  constexpr float HAD_SCALE = 0.08838834764831845f;
    \\  const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    \\  const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    \\  for (uint q = 0; q < 4u; ++q) {
    \\    const uint o = b * 128u + L + 32u * q;
    \\    float acc = z[ulong(row) * ulong(DIM) + o];
    \\    for (uint k = 1u; k < KS; ++k) acc += z[(ulong(k) * ulong(RS) + row) * ulong(DIM) + o];
    \\    if (DEF != 0u) acc = fma(acc, MUL1_A, MUL1_C * xsum);
    \\    v[q] = float(half(acc));
    \\  }
    \\  turboexl3_h128r(v, L);
    \\  for (uint q = 0; q < 4u; ++q) v[q] = float(half(v[q] * HAD_SCALE * float(svh[b * 128u + L + 32u * q])));
    \\}
    \\
;

pub const header_s: [:0]const u8 = header5 ++ metal_shared;

/// Form 5, launch 1: per (BT output tiles, lead row-slot, K slice): the expert group's members' rotated inputs over the slice
/// (prep2's arithmetic, into threadgroup memory: no prepare launch, no device round trip), then gu3's GEMV for every member
/// at once (each weight decoded once per group) into the same split-K planes zg / zu f32 [KS, RK, inter], and each 128-block's
/// input sum (xsg / xsu [RK, hidden / 128], written by the blk 0 threadgroups) for the deferred MUL1 map.
pub const text_gu5 =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr uint BTC = uint(BT);
    \\    constexpr uint GM = uint(GMAX);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint NBHC = HIDC / 128u;
    \\    constexpr uint IN_TILES = HIDC / 16u;
    \\    constexpr uint OUT_TILES = INTC / 16u;
    \\    constexpr uint PER = IN_TILES / (KSC * 2u);   // input tiles per half slice
    \\    constexpr uint SPAN = 32u * PER;               // a slice's input values
    \\    constexpr uint NBS = (SPAN % 128u == 0u) ? SPAN / 128u : SPAN / 128u + 2u;   // 128-blocks a slice can touch
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint kz = threadgroup_position_in_grid.z;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint e = uint(slots[rk]);
    \\    uint mem[4]; uint m;
    \\    if (!turboexl3_group<GM, uint(RKC)>(slots, rk, e, L, mem, m)) return;
    \\    // form 6's arrival counters, one per (row, 128-block of hidden), zeroed here (rk 0 always leads)
    \\    if (blk == 0u && rk == 0u && kz == 0u) for (uint i = tid; i < (uint(RKC) / TOPKC) * NBHC; i += 128u) cnt[i] = 0u;
    \\    threadgroup half xs[GM * 2u * NBS * 128u];   // [member][projection][block][128], lane order within each 16-tile
    \\    threadgroup float red[4][GM * 16u * BTC];
    \\    // 1. the members' rotated gate and up inputs over the slice's blocks
    \\    const uint b0 = (kz * SPAN) / 128u;
    \\    const uint nb = ((kz + 1u) * SPAN + 127u) / 128u - b0;
    \\    for (uint w = sg; w < m * 2u * nb; w += 4u) {
    \\        const uint j = w / (2u * nb);
    \\        const uint pj = (w / nb) & 1u;
    \\        const uint b = b0 + (w % nb);
    \\        const uint row = mem[j] / TOPKC;
    \\        const device half* suh = (pj == 0u) ? gate_suh : up_suh;
    \\        float v[4];
    \\        for (uint q = 0; q < 4u; ++q) { const uint i = b * 128u + L + 32u * q; const float xr = float(half(float(x[row * HIDC + i]))); v[q] = xr * float(suh[e * HIDC + i]); }
    \\        turboexl3_h128r(v, L);
    \\        threadgroup half* dst = xs + ((j * 2u + pj) * NBS + (b - b0)) * 128u;
    \\        float sv = 0.0f;
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const uint i = L + 32u * q;
    \\            const uint r = i & 15u; const uint qq = (r & 7u) >> 1u; const uint mm = (r & 1u) | ((r >> 3u) << 1u);
    \\            const half h = half(v[q] * HAD_SCALE);
    \\            dst[(i & ~15u) + 4u * qq + mm] = h;
    \\            sv += float(h);
    \\        }
    \\        sv = simd_sum(sv);
    \\        if (blk == 0u && L == 0u) { if (pj == 0u) xsg[mem[j] * NBHC + b] = sv; else xsu[mem[j] * NBHC + b] = sv; }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // 2. gu3's GEMV: simdgroup sg = projection (sg >> 1) x half (sg & 1) of the slice, BT output tiles, every member
    \\    const uint pj = sg >> 1u;
    \\    const device uint* wbase = (const device uint*)(pj == 0u ? gate_t : up_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(N / 2u);
    \\    const uint tk0 = (kz * 2u + (sg & 1u)) * PER;
    \\    const threadgroup half* xs0 = xs + pj * NBS * 128u + 4u * (L & 3u);
    \\    constexpr uint XST = 2u * NBS * 128u;   // one member to the next
    \\    if (m == 1u) turboexl3_gu5_sg<N, WMASK, CB, OUT_TILES, BTC, 1u, uint(DEFER), int(IR)>(wbase, xs0, XST, tk0, tk0 + PER, b0 * 8u, blk * BTC, L, red[sg]);
    \\    else if (GM >= 2u && m == 2u) turboexl3_gu5_sg<N, WMASK, CB, OUT_TILES, BTC, 2u, uint(DEFER), int(IR)>(wbase, xs0, XST, tk0, tk0 + PER, b0 * 8u, blk * BTC, L, red[sg]);
    \\    else if (GM >= 3u && m == 3u) turboexl3_gu5_sg<N, WMASK, CB, OUT_TILES, BTC, 3u, uint(DEFER), int(IR)>(wbase, xs0, XST, tk0, tk0 + PER, b0 * 8u, blk * BTC, L, red[sg]);
    \\    else if (GM >= 4u) turboexl3_gu5_sg<N, WMASK, CB, OUT_TILES, BTC, 4u, uint(DEFER), int(IR)>(wbase, xs0, XST, tk0, tk0 + PER, b0 * 8u, blk * BTC, L, red[sg]);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // 3. the slice's planes: half 0 + half 1 (gu3's sum)
    \\    for (uint i = tid; i < 16u * BTC * m; i += 128u) {
    \\        const uint j = i / (16u * BTC);
    \\        const ulong o = (ulong(kz) * ulong(RKC) + mem[j]) * ulong(INTC) + blk * (16u * BTC) + (i - j * 16u * BTC);
    \\        zg[o] = red[0][i] + red[1][i];
    \\        zu[o] = red[2][i] + red[3][i];
    \\    }
    \\
;

/// Form 5, launch 2: per (128-wide hidden block, lead row-slot), 8 simdgroups: every member's mid over the whole intermediate
/// (fin3's arithmetic: the planes' sum, the deferred map, f16 round, H128, svh, f16 round, SwiGLU, the down's suh, H128, f16
/// round, lane order, and each 128-block's sum) into threadgroup memory, then the down GEMV of the block's 8 output tiles
/// (one per simdgroup) for every member in down3's K partition, then dfin3's per-slot finish up to the score: the deferred
/// map, f16 round, H128, svh, f16 round -> wv half [RK, hidden].
const dn5_head =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint KSC = uint(KS);      // the gate|up planes
    \\    constexpr uint KS2C = uint(KS2);    // the down's K slices
    \\    constexpr uint GM = uint(GMAX);
    \\    constexpr uint NBHC = HIDC / 128u;
    \\    constexpr uint NBIC = INTC / 128u;
    \\    constexpr uint IN_TILES = INTC / 16u;
    \\    constexpr uint OUT_TILES = HIDC / 16u;
    \\    constexpr uint PER2 = IN_TILES / (KS2C * 4u);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint hb = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint e = uint(slots[rk]);
    \\    uint mem[4]; uint m;
    \\    if (!turboexl3_group<GM, uint(RKC)>(slots, rk, e, L, mem, m)) return;
    \\    threadgroup half hd[GM * INTC];
    \\    threadgroup float hsb[GM * NBIC];
    \\    threadgroup float fin[GM * 128u];
    \\    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    \\    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    \\    // 1. the members' mid, a 128-block per simdgroup pass (lane L holds elements L + 32 q)
    \\    for (uint w = sg; w < m * NBIC; w += 8u) {
    \\        const uint j = w / NBIC;
    \\        const uint b = w % NBIC;
    \\        const uint sl = mem[j];
    \\        float g[4]; float u[4];
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const uint o = b * 128u + L + 32u * q;
    \\            float gg = 0.0f, uu = 0.0f;
    \\            for (uint k = 0; k < KSC; ++k) { const ulong i = (ulong(k) * ulong(RKC) + sl) * ulong(INTC) + o; gg += zg[i]; uu += zu[i]; }
    \\            g[q] = gg; u[q] = uu;
    \\        }
    \\        if (DEFER != 0) {
    \\            float sgx = 0.0f, sux = 0.0f;
    \\            for (uint q = 0; q < NBHC; ++q) { sgx += xsg[sl * NBHC + q]; sux += xsu[sl * NBHC + q]; }
    \\            for (uint q = 0; q < 4u; ++q) { g[q] = fma(g[q], MUL1_A, MUL1_C * sgx); u[q] = fma(u[q], MUL1_A, MUL1_C * sux); }
    \\        }
    \\        for (uint q = 0; q < 4u; ++q) { g[q] = float(half(g[q])); u[q] = float(half(u[q])); }
    \\        turboexl3_h128r(g, L);
    \\        turboexl3_h128r(u, L);
    \\        float d[4];
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const uint o = b * 128u + L + 32u * q;
    \\            const float gv = float(half(g[q] * HAD_SCALE * float(gate_svh[e * INTC + o])));
    \\            const float uv = float(half(u[q] * HAD_SCALE * float(up_svh[e * INTC + o])));
    \\            const float hv = (gv / (1.0f + metal::exp(-gv))) * uv;
    \\            d[q] = float(half(hv)) * float(down_suh[e * INTC + o]);
    \\        }
    \\        turboexl3_h128r(d, L);
    \\        float s[4];
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const uint i = L + 32u * q;
    \\            const uint r = i & 15u; const uint qq = (r & 7u) >> 1u; const uint mm = (r & 1u) | ((r >> 3u) << 1u);
    \\            const half hh = half(d[q] * HAD_SCALE);
    \\            hd[j * INTC + b * 128u + (i & ~15u) + 4u * qq + mm] = hh;
    \\            s[q] = simd_sum(float(hh));
    \\        }
    \\        if (L == 0u) hsb[j * NBIC + b] = ((s[0] + s[1]) + s[2]) + s[3];
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // 2. the down GEMV: simdgroup sg = output tile 8 hb + sg, every member
    \\    const device uint* wbase = (const device uint*)(down_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(N / 2u);
    \\    const uint ot = hb * 8u + sg;
    \\    const threadgroup half* xs0 = hd + 4u * (L & 3u);
    \\    float tot[4][2];
    \\    if (m == 1u) turboexl3_dn5_sg<N, WMASK, CB, OUT_TILES, 1u, uint(DEFER), int(IR), KS2C, PER2>(wbase, xs0, INTC, ot, L, tot);
    \\    else if (GM >= 2u && m == 2u) turboexl3_dn5_sg<N, WMASK, CB, OUT_TILES, 2u, uint(DEFER), int(IR), KS2C, PER2>(wbase, xs0, INTC, ot, L, tot);
    \\    else if (GM >= 3u && m == 3u) turboexl3_dn5_sg<N, WMASK, CB, OUT_TILES, 3u, uint(DEFER), int(IR), KS2C, PER2>(wbase, xs0, INTC, ot, L, tot);
    \\    else if (GM >= 4u) turboexl3_dn5_sg<N, WMASK, CB, OUT_TILES, 4u, uint(DEFER), int(IR), KS2C, PER2>(wbase, xs0, INTC, ot, L, tot);
    \\    if ((L & 3u) == 0u) for (uint j = 0; j < m; ++j) { fin[j * 128u + sg * 16u + (L >> 2u)] = tot[j][0]; fin[j * 128u + sg * 16u + (L >> 2u) + 8u] = tot[j][1]; }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // 3. dfin3's per-slot finish up to the score, simdgroup j for member j
    \\    if (sg < m) {
    \\        const uint j = sg;
    \\        float hsum = 0.0f;
    \\        if (DEFER != 0) for (uint q = 0; q < NBIC; ++q) hsum += hsb[j * NBIC + q];
    \\        float v[4];
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            float t = fin[j * 128u + L + 32u * q];
    \\            if (DEFER != 0) t = fma(t, MUL1_A, MUL1_C * hsum);
    \\            v[q] = float(half(t));
    \\        }
    \\        turboexl3_h128r(v, L);
    \\
;
/// Form 5's down tail: the finished values to wv for launch 3.
const dn5_tail =
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const uint o = hb * 128u + L + 32u * q;
    \\            wv[ulong(mem[j]) * ulong(HIDC) + o] = half(v[q] * HAD_SCALE * float(down_svh[e * HIDC + o]));
    \\        }
    \\    }
    \\
;
/// Form 6's down tail: the finished values stored through coherent pointers, then member j's row counts this 128-block's
/// arrival (device-scope seq_cst fences around a relaxed fetch_add, C33's idiom; no waiting) and the simdgroup that brings
/// the row's TOPK-th slot reduces the block in slot order (red5's fma chain, so red5's bytes) straight into y.
const dn6_tail =
    \\        device TURBOEXL3_COH half* wvc = (device TURBOEXL3_COH half*)wv;
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const uint o = hb * 128u + L + 32u * q;
    \\            wvc[ulong(mem[j]) * ulong(HIDC) + o] = half(v[q] * HAD_SCALE * float(down_svh[e * HIDC + o]));
    \\        }
    \\        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\        simdgroup_barrier(mem_flags::mem_device);
    \\        const uint row = mem[j] / uint(TOPK);
    \\        uint old = 0u;
    \\        if (L == 0u) {
    \\            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\            old = atomic_fetch_add_explicit((device atomic_uint*)cnt + row * NBHC + hb, 1u, memory_order_relaxed);
    \\            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\        }
    \\        old = simd_broadcast(old, ushort(0));
    \\        if (old == uint(TOPK) - 1u) {
    \\            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\            const device TURBOEXL3_COH half* wr = (const device TURBOEXL3_COH half*)wv;
    \\            for (uint q = 0; q < 4u; ++q) {
    \\                const uint o = hb * 128u + L + 32u * q;
    \\                float acc = 0.0f;
    \\                for (uint k = 0; k < uint(TOPK); ++k) { const uint r2 = row * uint(TOPK) + k; acc = fma(float(scores[r2]), float(wr[ulong(r2) * ulong(HIDC) + o]), acc); }
    \\                y[ulong(row) * ulong(HIDC) + o] = OUT_T(acc);
    \\            }
    \\        }
    \\    }
    \\
;

pub const text_dn5: [:0]const u8 = dn5_head ++ dn5_tail;
pub const text_dn6: [:0]const u8 = dn5_head ++ dn6_tail;

/// Form 5, launch 3: y [rows, hidden] = the score-weighted sum of a row's slots in slot order (dfin3's fma chain).
pub const text_red5 =
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    const uint o = thread_position_in_grid.x;
    \\    const uint row = thread_position_in_grid.y;
    \\    float acc = 0.0f;
    \\    for (uint k = 0; k < TOPKC; ++k) { const uint rk = row * TOPKC + k; acc = fma(float(scores[rk]), float(wv[ulong(rk) * ulong(HIDC) + o]), acc); }
    \\    y[ulong(row) * ulong(HIDC) + o] = OUT_T(acc);
    \\
;

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
pub const text_fin7 =
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr uint NBHC = uint(NBH);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint e = uint(slots[rk]);
    \\    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    \\    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    \\    float g[4]; float u[4];
    \\    for (uint q = 0; q < 4u; ++q) {
    \\        const uint o = b * 128u + L + 32u * q;
    \\        float gg = 0.0f, uu = 0.0f;
    \\        for (uint ks = 0; ks < KSC; ++ks) { const ulong i = (ulong(ks) * ulong(RKC) + rk) * ulong(INTC) + o; gg += zg[i]; uu += zu[i]; }
    \\        g[q] = gg; u[q] = uu;
    \\    }
    \\    if (DEFER != 0) {
    \\        float sgx = 0.0f, sux = 0.0f;
    \\        for (uint q = 0; q < NBHC; ++q) { sgx += xsg[ulong(rk) * NBHC + q]; sux += xsu[ulong(rk) * NBHC + q]; }
    \\        for (uint q = 0; q < 4u; ++q) { g[q] = fma(g[q], MUL1_A, MUL1_C * sgx); u[q] = fma(u[q], MUL1_A, MUL1_C * sux); }
    \\    }
    \\    for (uint q = 0; q < 4u; ++q) { g[q] = float(half(g[q])); u[q] = float(half(u[q])); }
    \\    turboexl3_h128r(g, L);
    \\    turboexl3_h128r(u, L);
    \\    float d[4];
    \\    for (uint q = 0; q < 4u; ++q) {
    \\        const uint o = b * 128u + L + 32u * q;
    \\        const float gv = float(half(g[q] * HAD_SCALE * float(gate_svh[e * INTC + o])));
    \\        const float uv = float(half(u[q] * HAD_SCALE * float(up_svh[e * INTC + o])));
    \\        const float hv = (gv / (1.0f + metal::exp(-gv))) * uv;
    \\        d[q] = float(half(hv)) * float(down_suh[e * INTC + o]);
    \\    }
    \\    turboexl3_h128r(d, L);
    \\    float s[4];
    \\    for (uint q = 0; q < 4u; ++q) {
    \\        const uint i = L + 32u * q;
    \\        const uint r = i & 15u; const uint qq = (r & 7u) >> 1u; const uint mm = (r & 1u) | ((r >> 3u) << 1u);
    \\        const half hh = half(d[q] * HAD_SCALE);
    \\        hd[ulong(rk) * ulong(INTC) + b * 128u + (i & ~15u) + 4u * qq + mm] = hh;
    \\        s[q] = simd_sum(float(hh));
    \\    }
    \\    if (L == 0u) hs[ulong(rk) * ulong(INTC / 128u) + b] = ((s[0] + s[1]) + s[2]) + s[3];
    \\
;

/// Form 7, launch 5 (dfin3 with the slots side by side): per (128-wide hidden block, row), simdgroup k finishes slot k (the
/// planes' sum, the deferred map, f16 round, H128 in registers, svh, f16 round) into threadgroup memory, then 128 threads
/// run dfin3's score fma chain in slot order: dfin3's bytes, one H128 deep where dfin3 runs TOPK of them in a row.
pub const text_dfin7 =
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint k = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    threadgroup float term[TOPKC * 128u];
    \\    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    \\    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    \\    const uint rk = row * TOPKC + k;
    \\    const uint e = uint(slots[rk]);
    \\    float v[4];
    \\    for (uint q = 0; q < 4u; ++q) {
    \\        const uint o = blk * 128u + L + 32u * q;
    \\        float t = 0.0f;
    \\        for (uint ks = 0; ks < KSC; ++ks) t += zd[(ulong(ks) * ulong(RKC) + rk) * ulong(HIDC) + o];
    \\        v[q] = t;
    \\    }
    \\    if (DEFER != 0) {
    \\        float hsum = 0.0f;
    \\        for (uint q = 0; q < uint(NBI); ++q) hsum += hsx[ulong(rk) * uint(NBI) + q];
    \\        for (uint q = 0; q < 4u; ++q) v[q] = fma(v[q], MUL1_A, MUL1_C * hsum);
    \\    }
    \\    for (uint q = 0; q < 4u; ++q) v[q] = float(half(v[q]));
    \\    turboexl3_h128r(v, L);
    \\    for (uint q = 0; q < 4u; ++q) {
    \\        const uint o = blk * 128u + L + 32u * q;
    \\        term[k * 128u + L + 32u * q] = float(half(v[q] * HAD_SCALE * float(down_svh[e * HIDC + o])));
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (tid < 128u) {
    \\        float acc = 0.0f;
    \\        for (uint j = 0; j < TOPKC; ++j) acc = fma(float(scores[row * TOPKC + j]), term[j * 128u + tid], acc);
    \\        y[ulong(row) * ulong(HIDC) + blk * 128u + tid] = OUT_T(acc);   // the caller's dtype (bf16 served, f32 in the tests)
    \\    }
    \\
;

/// Form 8, launch 1 (prep2 + gu3 in one launch): gu3's threadgroup over its SPT slices, whose span (SPT hidden / KS values,
/// 128-aligned: the host admits it only then) the four simdgroups first rotate for every member and both projections in
/// prep2's arithmetic into threadgroup memory; then gu3's GEMV reads the input from there. The blk 0 threadgroups write
/// each block's input sum (xsg / xsu [RK, hidden / 128]) for fin's deferred map. gu3's planes, so form 3's bytes.
pub const text_gu8 =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr uint SPTC = uint(SPT);
    \\    constexpr uint BTC = uint(BT);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint IN_TILES = HIDC / 16u;
    \\    constexpr uint OUT_TILES = INTC / 16u;
    \\    constexpr uint NBHC = HIDC / 128u;
    \\    constexpr uint PER = IN_TILES / (KSC * 2u);           // input tiles per half slice
    \\    constexpr uint NBT = (SPTC * 2u * PER * 16u) / 128u;  // the threadgroup span's 128-blocks
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint kz = threadgroup_position_in_grid.z;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    constexpr uint GRP = uint(GROUP);
    \\    const uint e = uint(slots[rk]);
    \\    const uint proj = sg >> 1u;
    \\    const uint khalf = sg & 1u;
    \\    uint members[2]; uint m;
    \\    if (!turboexl3_members<GRP, uint(RKC)>(slots, rk, e, L, members, m)) return;
    \\    threadgroup half xs[GRP * 2u * NBT * 128u];   // [member][projection][block][128], lane order within each 16-tile
    \\    threadgroup float red[SPTC][4][2u * 16u * BTC];
    \\    // 1. prep2 over the span: x f16 round, suh, H128, 1/sqrt(128), f16 round, lane order; the block's sum of the rounded inputs
    \\    const uint b0 = kz * NBT;
    \\    for (uint w = sg; w < m * 2u * NBT; w += 4u) {
    \\        const uint j = w / (2u * NBT);
    \\        const uint pj = (w / NBT) & 1u;
    \\        const uint bl = w % NBT;
    \\        const uint b = b0 + bl;
    \\        const uint row = members[j] / TOPKC;
    \\        const device half* suh = (pj == 0u) ? gate_suh : up_suh;
    \\        float v[4];
    \\        for (uint q = 0; q < 4u; ++q) { const uint i = b * 128u + L + 32u * q; const float xr = float(half(float(x[row * HIDC + i]))); v[q] = xr * float(suh[e * HIDC + i]); }
    \\        turboexl3_h128r(v, L);
    \\        threadgroup half* dst = xs + ((j * 2u + pj) * NBT + bl) * 128u;
    \\        float sv = 0.0f;
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const uint i = L + 32u * q;
    \\            const uint r = i & 15u; const uint qq = (r & 7u) >> 1u; const uint mm = (r & 1u) | ((r >> 3u) << 1u);
    \\            const half h = half(v[q] * HAD_SCALE);
    \\            dst[(i & ~15u) + 4u * qq + mm] = h;
    \\            sv += float(h);
    \\        }
    \\        sv = simd_sum(sv);
    \\        if (blk == 0u && L == 0u) { if (pj == 0u) xsg[ulong(members[j]) * NBHC + b] = sv; else xsu[ulong(members[j]) * NBHC + b] = sv; }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // 2. gu3's GEMV, tile indices relative to the span (the weight base moved to its first tile)
    \\    const uint tkb = kz * SPTC * 2u * PER;
    \\    const device uint* wbase = (const device uint*)(proj == 0u ? gate_t : up_t) + (ulong(e) * ulong(IN_TILES) + ulong(tkb)) * ulong(OUT_TILES) * ulong(N / 2u);
    \\    const threadgroup half* xrow0 = xs + proj * NBT * 128u + 4u * (L & 3u);
    \\    const threadgroup half* xrow1 = xrow0 + 2u * NBT * 128u;
    \\    for (uint si = 0; si < SPTC; ++si) {
    \\        const uint tk0 = (si * 2u + khalf) * PER;
    \\        if (m == 1u) turboexl3_gemv_members<N, WMASK, CB, OUT_TILES, BTC, 1u, uint(DEFER)>(wbase, xrow0, xrow0, tk0, tk0 + PER, blk, L, red[si][sg]);
    \\        else turboexl3_gemv_members<N, WMASK, CB, OUT_TILES, BTC, 2u, uint(DEFER)>(wbase, xrow0, xrow1, tk0, tk0 + PER, blk, L, red[si][sg]);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (tid < 16u * BTC * m) {
    \\        const uint j = tid / (16u * BTC);
    \\        for (uint si = 0; si < SPTC; ++si) {
    \\            const ulong o = (ulong(kz * SPTC + si) * ulong(RKC) + members[j]) * ulong(INTC) + blk * (16u * BTC) + (tid - j * 16u * BTC);
    \\            zg[o] = red[si][0][tid] + red[si][1][tid];
    \\            zu[o] = red[si][2][tid] + red[si][3][tid];
    \\        }
    \\    }
    \\
;
const gu8_inputs = [_][*:0]const u8{ "x", "slots", "gate_suh", "up_suh", "gate_t", "up_t" };
const gu8_outputs = [_][*:0]const u8{ "zg", "zu", "xsg", "xsu" };

/// The shared expert in the routed launches (`Kernels3.moeShared`), launch 1's head: grid rows from RKC on are the shared
/// gate|up GEMV's threadgroups, j = (y - RKC) * SZ + z -> projection j / (SKSG / SSPT) and the SSPT consecutive K slices from
/// (j % (SKSG / SSPT)) SSPT (a slice = one 128-block of hidden: SKSG = hidden / 128); each slice is rotated in-kernel (dense
/// prep_s) and run as dense dgemv's slice for the grid's BT output tiles, exactly as a threadgroup of its own computes it, so
/// SSPT is launch shape only. Planes szg / szu f32 [SKSG, SRS, inter], block sums sxsg / sxsu [SRS, hidden / 128].
const shared_gu_head =
    \\    threadgroup half s_xs[uint(SRS) * 128u];
    \\    threadgroup float s_red[4][uint(SRS) * 16u * uint(BT)];
    \\    if (threadgroup_position_in_grid.y >= uint(RKC)) {
    \\        constexpr uint S_WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\        constexpr uint S_HID = uint(HID);
    \\        constexpr uint S_KS = uint(SKSG);
    \\        constexpr uint S_RS = uint(SRS);
    \\        constexpr uint S_NBH = S_HID / 128u;
    \\        constexpr uint S_BW = 16u * uint(BT);
    \\        constexpr uint S_SP = uint(SSPT);
    \\        constexpr uint S_NG = S_KS / S_SP;
    \\        const uint s_j = (threadgroup_position_in_grid.y - uint(RKC)) * uint(SZ) + threadgroup_position_in_grid.z;
    \\        if (s_j >= 2u * S_NG) return;
    \\        const uint s_pj = s_j / S_NG;
    \\        const uint s_blk = threadgroup_position_in_grid.x;
    \\        const uint s_sg = simdgroup_index_in_threadgroup;
    \\        const uint s_L = thread_index_in_simdgroup;
    \\        const uint s_tid = thread_index_in_threadgroup;
    \\        const device uint* s_w = (const device uint*)(s_pj == 0u ? sh_gate_t : sh_up_t);
    \\        device float* s_z = (s_pj == 0u) ? szg : szu;
    \\        // the next slice's prep writes s_xs only after this slice's GEMV (barrier 2), and its GEMV writes s_red only after
    \\        // every plane store of this one (its barrier 1)
    \\        for (uint s_q = 0; s_q < S_SP; ++s_q) {
    \\            const uint s_ksl = (s_j % S_NG) * S_SP + s_q;
    \\            for (uint r = s_sg; r < S_RS; r += 4u) {
    \\                float v[4];
    \\                for (uint q = 0; q < 4u; ++q) v[q] = float(half(float(xr[ulong(r) * S_HID + s_ksl * 128u + s_L + 32u * q])));
    \\                const float sum = (s_pj == 0u) ? turboexl3_sprep(v, sh_gate_suh, s_ksl, s_L, (threadgroup half*)(s_xs + r * 128u)) : turboexl3_sprep(v, sh_up_suh, s_ksl, s_L, (threadgroup half*)(s_xs + r * 128u));
    \\                if (s_blk == 0u && s_L == 0u) { if (s_pj == 0u) sxsg[r * S_NBH + s_ksl] = sum; else sxsu[r * S_NBH + s_ksl] = sum; }
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            const threadgroup half* s_x0 = s_xs + s_sg * 32u + 4u * (s_L & 3u);
    \\            turboexl3_sdgemv<uint(SNG), S_WMASK, int(CODEBOOK), uint(INTER) / 16u, uint(BT), S_RS, uint(SDEF), 2u, 128u>(s_w, s_x0, s_ksl * 8u + s_sg * 2u, s_blk, s_L, s_red[s_sg]);
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            for (uint i = s_tid; i < S_RS * S_BW; i += 128u) {
    \\                const uint r = i / S_BW;
    \\                s_z[(ulong(s_ksl) * ulong(S_RS) + r) * ulong(uint(INTER)) + s_blk * S_BW + (i - r * S_BW)] = ((s_red[0][i] + s_red[1][i]) + s_red[2][i]) + s_red[3][i];
    \\            }
    \\        }
    \\        return;
    \\    }
    \\
;
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
const shared_fin_head =
    \\    if (threadgroup_position_in_grid.y >= uint(RKC)) {
    \\        constexpr uint S_INT = uint(INTER);
    \\        constexpr uint S_NBH = uint(NBH);
    \\        constexpr uint S_NBI = uint(INTER) / 128u;
    \\        const uint s_b = threadgroup_position_in_grid.x;
    \\        const uint s_row = threadgroup_position_in_grid.y - uint(RKC);
    \\        const uint s_L = thread_index_in_simdgroup;
    \\        float s_xg = 0.0f, s_xu = 0.0f;
    \\        if (SDEF != 0) for (uint q = 0; q < S_NBH; ++q) s_xg += sxsg[s_row * S_NBH + q];
    \\        if (SDEF != 0) for (uint q = 0; q < S_NBH; ++q) s_xu += sxsu[s_row * S_NBH + q];
    \\        float g[4], u[4];
    \\        turboexl3_sfinish<uint(SKSG), uint(SRS), S_INT, uint(SDEF)>(szg, sh_gate_svh, s_xg, s_row, s_b, s_L, g);
    \\        turboexl3_sfinish<uint(SKSG), uint(SRS), S_INT, uint(SDEF)>(szu, sh_up_svh, s_xu, s_row, s_b, s_L, u);
    \\        float v[4];
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const OUT_T gt = OUT_T(g[q]);
    \\            const OUT_T ut = OUT_T(u[q]);
    \\            const OUT_T sgt = sigtab[as_type<ushort>(gt)];
    \\            const OUT_T act = gt * sgt;
    \\            const OUT_T h = act * ut;
    \\            v[q] = float(half(float(h)));
    \\        }
    \\        const float s_sum = turboexl3_sprep(v, sh_down_suh, s_b, s_L, (device half*)(sxpd + ulong(s_row) * ulong(S_INT) + s_b * 128u));
    \\        if (s_L == 0u) sxsd[s_row * S_NBI + s_b] = s_sum;
    \\        return;
    \\    }
    \\
;
pub const text_fin7s: [:0]const u8 = shared_fin_head ++ text_fin7;
const sfin_inputs = [_][*:0]const u8{ "szg", "szu", "sxsg", "sxsu", "sh_gate_svh", "sh_up_svh", "sh_down_suh", "sigtab" };
const sfin_outputs = [_][*:0]const u8{ "sxpd", "sxsd" };
const fin7s_inputs = fin3_inputs ++ sfin_inputs;
const fin7s_outputs = fin3_outputs ++ sfin_outputs;

/// Launch 4's head: grid rows from RKC on are the shared down GEMV's threadgroups, each over the SSPTD consecutive slices from
/// ((y - RKC) * SZ + z) SSPTD of SKSD (dense dgemv over sxpd, every slice as a threadgroup of its own computes it), planes
/// szd f32 [SKSD, SRS, hidden].
const shared_down_head =
    \\    threadgroup float s_red[4][uint(SRS) * 16u * uint(BT)];
    \\    if (threadgroup_position_in_grid.y >= uint(RKC)) {
    \\        constexpr uint S_WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\        constexpr uint S_KS = uint(SKSD);
    \\        constexpr uint S_RS = uint(SRS);
    \\        constexpr uint S_BW = 16u * uint(BT);
    \\        constexpr uint S_PER = (uint(INTER) / 16u) / (S_KS * 4u);
    \\        constexpr uint S_SP = uint(SSPTD);
    \\        const uint s_g = (threadgroup_position_in_grid.y - uint(RKC)) * uint(SZ) + threadgroup_position_in_grid.z;
    \\        if (s_g * S_SP >= S_KS) return;
    \\        const uint s_blk = threadgroup_position_in_grid.x;
    \\        const uint s_sg = simdgroup_index_in_threadgroup;
    \\        const uint s_L = thread_index_in_simdgroup;
    \\        const uint s_tid = thread_index_in_threadgroup;
    \\        for (uint s_q = 0; s_q < S_SP; ++s_q) {
    \\            const uint s_ksl = s_g * S_SP + s_q;
    \\            if (s_q != 0u) threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous slice's plane stores read s_red
    \\            const uint s_tk0 = (s_ksl * 4u + s_sg) * S_PER;
    \\            turboexl3_sdgemv<uint(SND), S_WMASK, int(CODEBOOK), uint(HID) / 16u, uint(BT), S_RS, uint(SDEF), S_PER, uint(INTER)>((const device uint*)(sh_down_t), sxpd + s_tk0 * 16u + 4u * (s_L & 3u), s_tk0, s_blk, s_L, s_red[s_sg]);
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            for (uint i = s_tid; i < S_RS * S_BW; i += 128u) {
    \\                const uint r = i / S_BW;
    \\                szd[(ulong(s_ksl) * ulong(S_RS) + r) * ulong(uint(HID)) + s_blk * S_BW + (i - r * S_BW)] = ((s_red[0][i] + s_red[1][i]) + s_red[2][i]) + s_red[3][i];
    \\            }
    \\        }
    \\        return;
    \\    }
    \\
;
pub const text_down3s: [:0]const u8 = shared_down_head ++ text_down3;
const sdown_inputs = [_][*:0]const u8{ "sxpd", "sh_down_t" };
const sdown_outputs = [_][*:0]const u8{"szd"};
const down3s_inputs = down3_inputs ++ sdown_inputs;
const down3s_outputs = [_][*:0]const u8{ "zd", "szd" };

/// Launch 5 with the shared expert: dfin7's slots in simdgroups 0..TOPK-1 and the shared down's dense dfinish in simdgroup TOPK,
/// then per output the routed sum in OUT_T plus the host's gate tail in OUT_T: sigmoid(logit) (MLX's, by table) * down, then
/// the sum (MLX's Multiply and Add), so y = moeSharedGateTail(routed, logit, down)'s bytes.
pub const text_dfin7s =
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint k = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    threadgroup float term[TOPKC * 128u];
    \\    threadgroup OUT_T sterm[128];
    \\    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    \\    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    \\    if (k < TOPKC) {
    \\        const uint rk = row * TOPKC + k;
    \\        const uint e = uint(slots[rk]);
    \\        float v[4];
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const uint o = blk * 128u + L + 32u * q;
    \\            float t = 0.0f;
    \\            for (uint ks = 0; ks < KSC; ++ks) t += zd[(ulong(ks) * ulong(RKC) + rk) * ulong(HIDC) + o];
    \\            v[q] = t;
    \\        }
    \\        if (DEFER != 0) {
    \\            float hsum = 0.0f;
    \\            for (uint q = 0; q < uint(NBI); ++q) hsum += hsx[ulong(rk) * uint(NBI) + q];
    \\            for (uint q = 0; q < 4u; ++q) v[q] = fma(v[q], MUL1_A, MUL1_C * hsum);
    \\        }
    \\        for (uint q = 0; q < 4u; ++q) v[q] = float(half(v[q]));
    \\        turboexl3_h128r(v, L);
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const uint o = blk * 128u + L + 32u * q;
    \\            term[k * 128u + L + 32u * q] = float(half(v[q] * HAD_SCALE * float(down_svh[e * HIDC + o])));
    \\        }
    \\    } else {
    \\        float s_xsum = 0.0f;
    \\        if (SDEF != 0) for (uint q = 0; q < uint(NBI); ++q) s_xsum += sxsd[row * uint(NBI) + q];
    \\        float v[4];
    \\        turboexl3_sfinish<uint(SKSD), uint(SRS), HIDC, uint(SDEF)>(szd, sh_down_svh, s_xsum, row, blk, L, v);
    \\        for (uint q = 0; q < 4u; ++q) sterm[L + 32u * q] = OUT_T(v[q]);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (tid < 128u) {
    \\        float acc = 0.0f;
    \\        for (uint j = 0; j < TOPKC; ++j) acc = fma(float(scores[row * TOPKC + j]), term[j * 128u + tid], acc);
    \\        const OUT_T yb = OUT_T(acc);
    \\        const OUT_T sig = sigtab[as_type<ushort>(slogit[row])];
    \\        const OUT_T gated = sig * sterm[tid];
    \\        y[ulong(row) * ulong(HIDC) + blk * 128u + tid] = yb + gated;
    \\    }
    \\
;
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
const metal_shared2 =
    \\// the v2 GEMV jobs' tile reader, in the loop shapes of dgemv / turboexl3_gemv_pair / turboexl3_dn5_sg (no forced unrolling, every
    \\// thread array indexed by small constant-trip loops): turboexl3_spload issues lane L's raw words (exl3_tile_load<N, 0>'s loads,
    \\// dgemv's general reader) of one part, the OTS output tiles from ot0 for each of its PER input tiles from tk0, so a
    \\// simdgroup can have parts in flight ahead of its decode (and across a barrier); turboexl3_spart funnels them with
    \\// exl3_tile_load's integer arithmetic (the same bits) into the window's 32-bit words and takes each codeword from them
    \\// (turboexl3_scw). The IntWords reader (rmode 0, EXL3_R64 0) loads at the funnel instead.
    \\struct Exl3Raw { uint a, b, c, d; };
    \\template <uint N, uint OUT_TILES, uint PER, uint OTS>
    \\inline void turboexl3_spload(const device uint* wbase, uint tk0, uint ot0, uint L, thread Exl3Raw (&r)[PER][OTS]) {
    \\  constexpr uint NW = N / 2u;
    \\  constexpr uint CM7 = ((8u * N) >> 4u) - (N >> 4u);
    \\  const uint e = exl3_gen_qb<N>(L) * 32u + exl3_gen_a<N>(L) * 16u + exl3_gen_lp<N>(L) + CM7 + 16u;
    \\  const uint last = ((e - 1u) >> 5u) % NW;
    \\  const uint w1 = (last + NW - 1u) % NW;
    \\  const uint w2 = (w1 + NW - 1u) % NW;
    \\  const uint w3 = (w2 + NW - 1u) % NW;
    \\  for (uint i = 0; i < PER; ++i) {
    \\    for (uint t = 0; t < OTS; ++t) {
    \\      const device uint* tile = wbase + (ulong(tk0 + i) * OUT_TILES + ot0 + t) * ulong(NW);
    \\      if (exl3_rmode<N>() != 0) {
    \\        r[i][t].a = tile[last]; r[i][t].b = tile[w1]; r[i][t].c = tile[w2];
    \\        r[i][t].d = (exl3_rmode<N>() == 2) ? tile[w3] : 0u;
    \\      } else {
    \\        r[i][t].a = 0u; r[i][t].b = 0u; r[i][t].c = 0u; r[i][t].d = 0u;
    \\      }
    \\    }
    \\  }
    \\}
    \\// codeword M of a lane's window held as 32-bit words: lo, mid, hi = its bits 0-31, 32-63 and 64-95 counted from the window's
    \\// end (the 96-bit reader's words; a 64-bit window never reaches hi), taken by exl3_tile_cw's 96-bit rules at its
    \\// compile-time offset SH (one bit-field extract, or a two-word funnel): the 16 bits exl3_tile_cw takes from either window
    \\// reader, with no 64-bit window value (rmode 1's ulong)
    \\template <uint N, uint WMASK, uint M>
    \\inline uint turboexl3_scw(uint lo, uint mid, uint hi) {
    \\  constexpr uint SH = (((8u * N) >> 4u) - (N >> 4u)) - ((((M + 1u) * N) >> 4u) - (N >> 4u));
    \\  uint cw;
    \\  if constexpr (SH + 16u <= 32u) cw = metal::extract_bits(lo, SH, 16u);
    \\  else if constexpr (SH < 32u) cw = ((lo >> SH) | (mid << (32u - SH))) & 0xffffu;
    \\  else if constexpr (SH + 16u <= 64u) cw = metal::extract_bits(mid, SH - 32u, 16u);
    \\  else if constexpr (SH < 64u) cw = ((mid >> (SH - 32u)) | (hi << (64u - SH))) & 0xffffu;
    \\  else cw = metal::extract_bits(hi, SH - 64u, 16u);
    \\  return (WMASK == 0xffffu) ? cw : (cw & WMASK);
    \\}
    \\// one part of a K slice (dgemv's simdgroup share: PER input tiles from tk0) for OTS output tiles and RS rows, row j's
    \\// lane-order input at x0 + j XSTRIDE (element 0 = the part's first input tile): from zero in dgemv's order (tile order,
    \\// weights m = 0..7, a[j][t][m / 4] = fma(x[j][m % 4], w, a[j][t][m / 4])), folded over the quad lanes (dgemv's
    \\// turboexl3_fold) and added into the slice's left sum s (dgemv's plane store ((part 0 + part 1) + part 2) + part 3)
    \\template <uint N, uint WMASK, int CB, uint OUT_TILES, uint RS, uint OTS, uint PER, uint DEF, uint XSTRIDE, typename XP>
    \\inline void turboexl3_spart(thread const Exl3Raw (&r)[PER][OTS], const device uint* wbase, uint tk0, uint ot0, XP x0, uint L, bool first, thread float (&s)[RS][OTS][2]) {
    \\  constexpr uint NW = N / 2u;
    \\  constexpr uint CM7 = ((8u * N) >> 4u) - (N >> 4u);
    \\  const uint qb = exl3_gen_qb<N>(L);
    \\  const uint ga = exl3_gen_a<N>(L);
    \\  const uint lp = exl3_gen_lp<N>(L);
    \\  const uint sh = (0u - (qb * 32u + ga * 16u + lp + CM7 + 16u)) & 31u;
    \\  float a[RS][OTS][2];
    \\  for (uint j = 0; j < RS; ++j) for (uint t = 0; t < OTS; ++t) { a[j][t][0] = 0.0f; a[j][t][1] = 0.0f; }
    \\  for (uint i = 0; i < PER; ++i) {
    \\    float xr[RS][4];
    \\    for (uint j = 0; j < RS; ++j) {
    \\      const half4 x4 = turboexl3_x4(x0 + j * XSTRIDE + i * 16u);
    \\      xr[j][0] = float(x4.x); xr[j][1] = float(x4.y); xr[j][2] = float(x4.z); xr[j][3] = float(x4.w);
    \\    }
    \\    for (uint t = 0; t < OTS; ++t) {
    \\      uint cw[8];
    \\      if constexpr (exl3_rmode<N>() != 0) {
    \\        // the window's 32-bit words from the raw words (exl3_tile_load's funnels), hi only for the 96-bit window
    \\        const uint lo = uint(((ulong(r[i][t].b) << 32u) | ulong(r[i][t].a)) >> sh);
    \\        const uint mid = uint(((ulong(r[i][t].c) << 32u) | ulong(r[i][t].b)) >> sh);
    \\        const uint hi = (exl3_rmode<N>() == 2) ? uint(((ulong(r[i][t].d) << 32u) | ulong(r[i][t].c)) >> sh) : 0u;
    \\        cw[0] = turboexl3_scw<N, WMASK, 0>(lo, mid, hi); cw[1] = turboexl3_scw<N, WMASK, 1>(lo, mid, hi); cw[2] = turboexl3_scw<N, WMASK, 2>(lo, mid, hi); cw[3] = turboexl3_scw<N, WMASK, 3>(lo, mid, hi);
    \\        cw[4] = turboexl3_scw<N, WMASK, 4>(lo, mid, hi); cw[5] = turboexl3_scw<N, WMASK, 5>(lo, mid, hi); cw[6] = turboexl3_scw<N, WMASK, 6>(lo, mid, hi); cw[7] = turboexl3_scw<N, WMASK, 7>(lo, mid, hi);
    \\      } else {
    \\        const auto ww = exl3_tile_load<N, 0>(wbase + (ulong(tk0 + i) * OUT_TILES + ot0 + t) * ulong(NW), qb, ga, lp);
    \\        cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
    \\        cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
    \\      }
    \\      for (uint m = 0; m < 8u; ++m) {
    \\        const float wv = (DEF != 0u) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
    \\        for (uint j = 0; j < RS; ++j) a[j][t][m >> 2] = fma(xr[j][m & 3u], wv, a[j][t][m >> 2]);
    \\      }
    \\    }
    \\  }
    \\  for (uint j = 0; j < RS; ++j) {
    \\    for (uint t = 0; t < OTS; ++t) {
    \\      for (uint c = 0; c < 2u; ++c) {
    \\        float v = a[j][t][c];
    \\        v += simd_shuffle_xor(v, ushort(1));
    \\        v += simd_shuffle_xor(v, ushort(2));
    \\        s[j][t][c] = first ? v : (s[j][t][c] + v);
    \\      }
    \\    }
    \\  }
    \\}
    \\// output o of row r in K slice k of planes [KS][RS][DIM]: the natural layout, or lane-major (LM) within each 128-block,
    \\// offset 4 (i % 32) + i / 32 of block o / 128 (i = o % 128), so the finish's lane L reads its values L + 32 q as one float4
    \\template <uint LM, uint RS, uint DIM>
    \\inline ulong turboexl3_sidx(uint k, uint r, uint o) {
    \\  if (LM != 0u) return (ulong(k * RS + r) * ulong(DIM / 128u) + ulong(o >> 7u)) * ulong(128u) + ulong(4u * (o & 31u) + ((o >> 5u) & 3u));
    \\  return ulong(k * RS + r) * ulong(DIM) + ulong(o);
    \\}
    \\// the slice's values to the planes: every lane of a quad holds the quad's 2 RS OTS values (columns L / 4 + 8 c of the OTS
    \\// tiles from ot0); value v = (r OTS + t) 2 + c is stored by the quad's lane v % 4
    \\template <uint LM, uint RS, uint OTS, uint DIM>
    \\inline void turboexl3_sstore(device float* z, thread const float (&s)[RS][OTS][2], uint k, uint ot0, uint L) {
  \\  for (uint r = 0; r < RS; ++r) {
    \\    for (uint t = 0; t < OTS; ++t) {
    \\      for (uint c = 0; c < 2u; ++c) {
    \\        if ((((r * OTS + t) * 2u + c) & 3u) == (L & 3u)) z[turboexl3_sidx<LM, RS, DIM>(k, r, (ot0 + t) * 16u + (L >> 2u) + 8u * c)] = s[r][t][c];
    \\      }
    \\    }
    \\  }
    \\}
    \\// turboexl3_sfinish over planes in either layout: plane 0, then + plane k in slice order (LM: as float4, the same four sums),
    \\// the deferred MUL1 map with the row's input sum, f16, H128, svh, f16
    \\template <uint KS, uint RS, uint DIM, uint DEF, uint LM, typename SV>
    \\inline void turboexl3_sfinish2(const device float* z, SV svh, float xsum, uint row, uint b, uint L, thread float (&v)[4]) {
    \\  constexpr float HAD_SCALE = 0.08838834764831845f;
    \\  const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    \\  const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    \\  float acc[4];
    \\  if (LM != 0u) {
    \\    const device float4* z4 = reinterpret_cast<const device float4*>(z);
    \\    float4 t4 = z4[(ulong(row) * ulong(DIM / 128u) + ulong(b)) * ulong(32u) + ulong(L)];
    \\    for (uint k = 1u; k < KS; ++k) t4 += z4[((ulong(k) * ulong(RS) + ulong(row)) * ulong(DIM / 128u) + ulong(b)) * ulong(32u) + ulong(L)];
    \\    acc[0] = t4.x; acc[1] = t4.y; acc[2] = t4.z; acc[3] = t4.w;
    \\  } else {
    \\    for (uint q = 0; q < 4u; ++q) {
    \\      const uint o = b * 128u + L + 32u * q;
    \\      float a = z[ulong(row) * ulong(DIM) + o];
    \\      for (uint k = 1u; k < KS; ++k) a += z[(ulong(k) * ulong(RS) + row) * ulong(DIM) + o];
    \\      acc[q] = a;
    \\    }
    \\  }
    \\  for (uint q = 0; q < 4u; ++q) {
    \\    float a = acc[q];
    \\    if (DEF != 0u) a = fma(a, MUL1_A, MUL1_C * xsum);
    \\    v[q] = float(half(a));
    \\  }
    \\  turboexl3_h128r(v, L);
    \\  for (uint q = 0; q < 4u; ++q) v[q] = float(half(v[q] * HAD_SCALE * float(svh[b * 128u + L + 32u * q])));
    \\}
    \\
;

pub const header_t: [:0]const u8 = header_s ++ metal_shared2;

/// The v2 heads' shared-threadgroup test and id (`moeShared2`): with SYOFF 0 the grid rows from RKC on are the shared
/// expert's (the routed rows first, as v1); SYOFF > 0 puts the SYS shared rows first (the routed body's rk = y - SYOFF,
/// `text_*_y`). The shared threadgroups are numbered z-major (z * SYS + row) * SX + x, the dispatch order of the grid's slabs.
const shared2_prologue =
    \\    if ((uint(SYOFF) != 0u) ? (threadgroup_position_in_grid.y < uint(SYOFF)) : (threadgroup_position_in_grid.y >= uint(RKC))) {
    \\        const uint s_yy = (uint(SYOFF) != 0u) ? threadgroup_position_in_grid.y : (threadgroup_position_in_grid.y - uint(RKC));
    \\        const uint s_id = (threadgroup_position_in_grid.z * uint(SYS) + s_yy) * uint(SX) + threadgroup_position_in_grid.x;
    \\
;

/// Launch 1's v2 head, the gate|up GEMV with the input rotated in-kernel (form 9's gu8 launch, gu3 without `sprep`, and the
/// standalone beside variant): shared threadgroup s_id = (K slice, group of 4 SOTS output tiles of the 2 inter / 16 tiles,
/// gate's then up's); simdgroup sg owns the group's SOTS tiles from (4 group + sg) SOTS. It walks the slice part by part
/// (dgemv's four simdgroup shares), its weights SPRE input tiles ahead of the decode in whole parts (at most 3 parts, at
/// most 16 tiles of raw words in flight); the first ones are issued before the threadgroup rotates the slice's 128-block
/// (dense prep_s: one simdgroup per (projection, row) the threadgroup reads, the block sums written by each projection's
/// first threadgroup) and its barrier. The plane values are stored straight from the quads. Planes szg / szu f32 [SKSG,
/// SRS, inter] (layout SLM), block sums sxsg / sxsu [SRS, hidden / 128].
const shared_gu2_head =
    \\    threadgroup half s_xs2[uint(SNP) * uint(SRS) * 128u];
    \\
    ++ shared2_prologue ++
    \\        constexpr uint S_WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\        constexpr uint S_N = uint(SNG);
    \\        constexpr uint S_HID = uint(HID);
    \\        constexpr uint S_RS = uint(SRS);
    \\        constexpr uint S_KS = uint(SKSG);
    \\        constexpr uint S_NBH = S_HID / 128u;
    \\        constexpr uint S_OTG = uint(INTER) / 16u;              // output tiles per projection
    \\        constexpr uint S_OTS = uint(SOTS);                     // output tiles per simdgroup
    \\        constexpr uint S_TGS = (2u * S_OTG) / (4u * S_OTS);    // threadgroups per K slice
    \\        constexpr uint S_PER = (S_HID / 16u) / (S_KS * 4u);    // input tiles per part (dgemv's PER)
    \\        constexpr uint S_NK = 4u * S_PER;                      // input tiles per slice: one 128-block
    \\        constexpr uint S_DW = ((uint(SPRE) / S_PER) < 3u) ? (uint(SPRE) / S_PER) : 3u;   // SPRE input tiles, whole parts, <= 3 ahead
    \\        constexpr uint S_DC = 16u / (S_PER * S_OTS);                                     // <= 16 tiles of raw words in flight
    \\        constexpr uint S_D = (S_DW < S_DC) ? S_DW : S_DC;                                // parts in flight ahead of the decode
    \\        if (s_id >= S_KS * S_TGS) return;
    \\        const uint s_ksl = s_id / S_TGS;
    \\        const uint s_g = s_id - s_ksl * S_TGS;
    \\        const uint s_sg = simdgroup_index_in_threadgroup;
    \\        const uint s_L = thread_index_in_simdgroup;
    \\        const uint s_gt0 = (s_g * 4u + s_sg) * S_OTS;          // first output tile, gate's 0..OTG then up's
    \\        const uint s_pj = s_gt0 / S_OTG;
    \\        const uint s_ot0 = s_gt0 - s_pj * S_OTG;
    \\        const uint s_pj0 = (s_g * 4u * S_OTS) / S_OTG;          // the threadgroup's projections
    \\        const uint s_npj = ((s_g * 4u + 3u) * S_OTS) / S_OTG - s_pj0 + 1u;
    \\        const device uint* s_wb = (const device uint*)((s_pj == 0u) ? sh_gate_t : sh_up_t);
    \\        const uint s_tk0 = s_ksl * S_NK;
    \\        Exl3Raw s_q[S_D + 1u][S_PER][S_OTS];   // the parts in flight ahead of the decode, s_q[0] the next one
    \\        for (uint s_p = 0; s_p < S_D; ++s_p) turboexl3_spload<S_N, S_OTG, S_PER, S_OTS>(s_wb, s_tk0 + s_p * S_PER, s_ot0, s_L, s_q[s_p]);
    \\        for (uint s_wi = s_sg; s_wi < s_npj * S_RS; s_wi += 4u) {
    \\            const uint s_pw = s_pj0 + s_wi / S_RS;
    \\            const uint s_r = s_wi - (s_wi / S_RS) * S_RS;
    \\            float s_v[4];
    \\            for (uint q = 0; q < 4u; ++q) s_v[q] = float(half(float(xr[ulong(s_r) * S_HID + s_ksl * 128u + s_L + 32u * q])));
    \\            threadgroup half* s_dst = s_xs2 + s_wi * 128u;
    \\            const float s_sum = (s_pw == 0u) ? turboexl3_sprep(s_v, sh_gate_suh, s_ksl, s_L, s_dst) : turboexl3_sprep(s_v, sh_up_suh, s_ksl, s_L, s_dst);
    \\            if (s_L == 0u && s_g == (s_pw * S_OTG) / (4u * S_OTS)) { if (s_pw == 0u) sxsg[s_r * S_NBH + s_ksl] = s_sum; else sxsu[s_r * S_NBH + s_ksl] = s_sum; }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        const threadgroup half* s_x0 = s_xs2 + (s_pj - s_pj0) * S_RS * 128u + 4u * (s_L & 3u);
    \\        float s_s[S_RS][S_OTS][2];
    \\        for (uint j = 0; j < S_RS; ++j) for (uint t = 0; t < S_OTS; ++t) { s_s[j][t][0] = 0.0f; s_s[j][t][1] = 0.0f; }
    \\        // the slice's four parts in order, each issuing the part S_D ahead before it decodes, then the queue moves up one
    \\        for (uint s_p = 0; s_p < 4u; ++s_p) {
    \\            if (s_p + S_D < 4u) turboexl3_spload<S_N, S_OTG, S_PER, S_OTS>(s_wb, s_tk0 + (s_p + S_D) * S_PER, s_ot0, s_L, s_q[S_D]);
    \\            turboexl3_spart<S_N, S_WMASK, int(CODEBOOK), S_OTG, S_RS, S_OTS, S_PER, uint(SDEF), 128u>(s_q[0], s_wb, s_tk0 + s_p * S_PER, s_ot0, s_x0 + s_p * S_PER * 16u, s_L, s_p == 0u, s_s);
    \\            for (uint s_j = 0; s_j < S_D; ++s_j) for (uint i = 0; i < S_PER; ++i) for (uint t = 0; t < S_OTS; ++t) s_q[s_j][i][t] = s_q[s_j + 1u][i][t];
    \\        }
    \\        turboexl3_sstore<uint(SLM), S_RS, S_OTS, uint(INTER)>((s_pj == 0u) ? szg : szu, s_s, s_ksl, s_ot0, s_L);
    \\        return;
    \\    }
    \\
;

/// Launch 1's v2 head with the input prepared once by the routed prepare launch (`shared_prep_head`, form 9 from two rows,
/// `sprep`): `shared_gu2_head` reading the lane-order rows sxg / sxu half [SRS, hidden] from device memory, so no rotation,
/// no threadgroup memory and no barrier. Planes szg / szu.
const shared_gu2p_head = shared2_prologue ++
    \\        constexpr uint S_WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\        constexpr uint S_N = uint(SNG);
    \\        constexpr uint S_HID = uint(HID);
    \\        constexpr uint S_RS = uint(SRS);
    \\        constexpr uint S_KS = uint(SKSG);
    \\        constexpr uint S_OTG = uint(INTER) / 16u;
    \\        constexpr uint S_OTS = uint(SOTS);
    \\        constexpr uint S_TGS = (2u * S_OTG) / (4u * S_OTS);
    \\        constexpr uint S_PER = (S_HID / 16u) / (S_KS * 4u);
    \\        constexpr uint S_NK = 4u * S_PER;
    \\        constexpr uint S_DW = ((uint(SPRE) / S_PER) < 3u) ? (uint(SPRE) / S_PER) : 3u;   // SPRE input tiles, whole parts, <= 3 ahead
    \\        constexpr uint S_DC = 16u / (S_PER * S_OTS);                                     // <= 16 tiles of raw words in flight
    \\        constexpr uint S_D = (S_DW < S_DC) ? S_DW : S_DC;                                // parts in flight ahead of the decode
    \\        if (s_id >= S_KS * S_TGS) return;
    \\        const uint s_ksl = s_id / S_TGS;
    \\        const uint s_g = s_id - s_ksl * S_TGS;
    \\        const uint s_L = thread_index_in_simdgroup;
    \\        const uint s_gt0 = (s_g * 4u + simdgroup_index_in_threadgroup) * S_OTS;
    \\        const uint s_pj = s_gt0 / S_OTG;
    \\        const uint s_ot0 = s_gt0 - s_pj * S_OTG;
    \\        const device uint* s_wb = (const device uint*)((s_pj == 0u) ? sh_gate_t : sh_up_t);
    \\        const uint s_tk0 = s_ksl * S_NK;
    \\        const device half* s_x0 = ((s_pj == 0u) ? sxg : sxu) + s_tk0 * 16u + 4u * (s_L & 3u);
    \\        Exl3Raw s_q[S_D + 1u][S_PER][S_OTS];   // the parts in flight ahead of the decode, s_q[0] the next one
    \\        for (uint s_p = 0; s_p < S_D; ++s_p) turboexl3_spload<S_N, S_OTG, S_PER, S_OTS>(s_wb, s_tk0 + s_p * S_PER, s_ot0, s_L, s_q[s_p]);
    \\        float s_s[S_RS][S_OTS][2];
    \\        for (uint j = 0; j < S_RS; ++j) for (uint t = 0; t < S_OTS; ++t) { s_s[j][t][0] = 0.0f; s_s[j][t][1] = 0.0f; }
    \\        // the slice's four parts in order, each issuing the part S_D ahead before it decodes, then the queue moves up one
    \\        for (uint s_p = 0; s_p < 4u; ++s_p) {
    \\            if (s_p + S_D < 4u) turboexl3_spload<S_N, S_OTG, S_PER, S_OTS>(s_wb, s_tk0 + (s_p + S_D) * S_PER, s_ot0, s_L, s_q[S_D]);
    \\            turboexl3_spart<S_N, S_WMASK, int(CODEBOOK), S_OTG, S_RS, S_OTS, S_PER, uint(SDEF), S_HID>(s_q[0], s_wb, s_tk0 + s_p * S_PER, s_ot0, s_x0 + s_p * S_PER * 16u, s_L, s_p == 0u, s_s);
    \\            for (uint s_j = 0; s_j < S_D; ++s_j) for (uint i = 0; i < S_PER; ++i) for (uint t = 0; t < S_OTS; ++t) s_q[s_j][i][t] = s_q[s_j + 1u][i][t];
    \\        }
    \\        turboexl3_sstore<uint(SLM), S_RS, S_OTS, uint(INTER)>((s_pj == 0u) ? szg : szu, s_s, s_ksl, s_ot0, s_L);
    \\        return;
    \\    }
    \\
;

/// Launch 0's head for `shared_gu2p_head` (prep2 from two rows; standalone beside): grid rows from RKC on are the shared
/// expert's rows, one simdgroup per (128-block of hidden, row) running dense prep_s for gate and up: sxg / sxu half [SRS,
/// hidden] lane order, the block sums sxsg / sxsu [SRS, hidden / 128].
const shared_prep_head =
    \\    if (threadgroup_position_in_grid.y >= uint(RKC)) {
    \\        constexpr uint S_DIM = uint(DIM);
    \\        constexpr uint S_NB = S_DIM / 128u;
    \\        const uint s_b = threadgroup_position_in_grid.x;
    \\        const uint s_r = threadgroup_position_in_grid.y - uint(RKC);
    \\        const uint s_L = thread_index_in_simdgroup;
    \\        for (uint s_p = 0; s_p < 2u; ++s_p) {
    \\            float s_v[4];
    \\            for (uint q = 0; q < 4u; ++q) s_v[q] = float(half(float(x[ulong(s_r) * S_DIM + s_b * 128u + s_L + 32u * q])));
    \\            device half* s_dst = ((s_p == 0u) ? sxg : sxu) + ulong(s_r) * S_DIM + s_b * 128u;
    \\            const float s_sum = (s_p == 0u) ? turboexl3_sprep(s_v, sh_gate_suh, s_b, s_L, s_dst) : turboexl3_sprep(s_v, sh_up_suh, s_b, s_L, s_dst);
    \\            if (s_L == 0u) { if (s_p == 0u) sxsg[s_r * S_NB + s_b] = s_sum; else sxsu[s_r * S_NB + s_b] = s_sum; }
    \\        }
    \\        return;
    \\    }
    \\
;

/// Launch 3's v2 head: `shared_fin_head` with the planes read in layout SLM (`turboexl3_sfinish2`).
const shared_fin2_head =
    \\    if (threadgroup_position_in_grid.y >= uint(RKC)) {
    \\        constexpr uint S_INT = uint(INTER);
    \\        constexpr uint S_NBH = uint(NBH);
    \\        constexpr uint S_NBI = uint(INTER) / 128u;
    \\        const uint s_b = threadgroup_position_in_grid.x;
    \\        const uint s_row = threadgroup_position_in_grid.y - uint(RKC);
    \\        const uint s_L = thread_index_in_simdgroup;
    \\        float s_xg = 0.0f, s_xu = 0.0f;
    \\        if (SDEF != 0) for (uint q = 0; q < S_NBH; ++q) s_xg += sxsg[s_row * S_NBH + q];
    \\        if (SDEF != 0) for (uint q = 0; q < S_NBH; ++q) s_xu += sxsu[s_row * S_NBH + q];
    \\        float g[4], u[4];
    \\        turboexl3_sfinish2<uint(SKSG), uint(SRS), S_INT, uint(SDEF), uint(SLM)>(szg, sh_gate_svh, s_xg, s_row, s_b, s_L, g);
    \\        turboexl3_sfinish2<uint(SKSG), uint(SRS), S_INT, uint(SDEF), uint(SLM)>(szu, sh_up_svh, s_xu, s_row, s_b, s_L, u);
    \\        float v[4];
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const OUT_T gt = OUT_T(g[q]);
    \\            const OUT_T ut = OUT_T(u[q]);
    \\            const OUT_T sgt = sigtab[as_type<ushort>(gt)];
    \\            const OUT_T act = gt * sgt;
    \\            const OUT_T h = act * ut;
    \\            v[q] = float(half(float(h)));
    \\        }
    \\        const float s_sum = turboexl3_sprep(v, sh_down_suh, s_b, s_L, (device half*)(sxpd + ulong(s_row) * ulong(S_INT) + s_b * 128u));
    \\        if (s_L == 0u) sxsd[s_row * S_NBI + s_b] = s_sum;
    \\        return;
    \\    }
    \\
;

/// Launch 4's v2 head: shared threadgroup s_id = (K slice of SKSD, group of 4 SOTS of the hidden / 16 output tiles), simdgroup
/// sg owning SOTS tiles from (4 group + sg) SOTS over the slice's four parts of sxpd (lane order, device memory): no
/// threadgroup memory, no barrier. Planes szd f32 [SKSD, SRS, hidden] (layout SLM).
const shared_down2_head = shared2_prologue ++
    \\        constexpr uint S_WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\        constexpr uint S_N = uint(SND);
    \\        constexpr uint S_RS = uint(SRS);
    \\        constexpr uint S_KS = uint(SKSD);
    \\        constexpr uint S_INT = uint(INTER);
    \\        constexpr uint S_OTD = uint(HID) / 16u;
    \\        constexpr uint S_OTS = uint(SOTS);
    \\        constexpr uint S_TGS = S_OTD / (4u * S_OTS);
    \\        constexpr uint S_PER = (S_INT / 16u) / (S_KS * 4u);
    \\        constexpr uint S_NK = 4u * S_PER;
    \\        constexpr uint S_DW = ((uint(SPRE) / S_PER) < 3u) ? (uint(SPRE) / S_PER) : 3u;   // SPRE input tiles, whole parts, <= 3 ahead
    \\        constexpr uint S_DC = 16u / (S_PER * S_OTS);                                     // <= 16 tiles of raw words in flight
    \\        constexpr uint S_D = (S_DW < S_DC) ? S_DW : S_DC;                                // parts in flight ahead of the decode
    \\        if (s_id >= S_KS * S_TGS) return;
    \\        const uint s_ksl = s_id / S_TGS;
    \\        const uint s_g = s_id - s_ksl * S_TGS;
    \\        const uint s_L = thread_index_in_simdgroup;
    \\        const uint s_ot0 = (s_g * 4u + simdgroup_index_in_threadgroup) * S_OTS;
    \\        const uint s_tk0 = s_ksl * S_NK;
    \\        const device uint* s_wb = (const device uint*)(sh_down_t);
    \\        const device half* s_x0 = sxpd + s_tk0 * 16u + 4u * (s_L & 3u);
    \\        Exl3Raw s_q[S_D + 1u][S_PER][S_OTS];   // the parts in flight ahead of the decode, s_q[0] the next one
    \\        for (uint s_p = 0; s_p < S_D; ++s_p) turboexl3_spload<S_N, S_OTD, S_PER, S_OTS>(s_wb, s_tk0 + s_p * S_PER, s_ot0, s_L, s_q[s_p]);
    \\        float s_s[S_RS][S_OTS][2];
    \\        for (uint j = 0; j < S_RS; ++j) for (uint t = 0; t < S_OTS; ++t) { s_s[j][t][0] = 0.0f; s_s[j][t][1] = 0.0f; }
    \\        // the slice's four parts in order, each issuing the part S_D ahead before it decodes, then the queue moves up one
    \\        for (uint s_p = 0; s_p < 4u; ++s_p) {
    \\            if (s_p + S_D < 4u) turboexl3_spload<S_N, S_OTD, S_PER, S_OTS>(s_wb, s_tk0 + (s_p + S_D) * S_PER, s_ot0, s_L, s_q[S_D]);
    \\            turboexl3_spart<S_N, S_WMASK, int(CODEBOOK), S_OTD, S_RS, S_OTS, S_PER, uint(SDEF), S_INT>(s_q[0], s_wb, s_tk0 + s_p * S_PER, s_ot0, s_x0 + s_p * S_PER * 16u, s_L, s_p == 0u, s_s);
    \\            for (uint s_j = 0; s_j < S_D; ++s_j) for (uint i = 0; i < S_PER; ++i) for (uint t = 0; t < S_OTS; ++t) s_q[s_j][i][t] = s_q[s_j + 1u][i][t];
    \\        }
    \\        turboexl3_sstore<uint(SLM), S_RS, S_OTS, uint(HID)>(szd, s_s, s_ksl, s_ot0, s_L);
    \\        return;
    \\    }
    \\
;

/// Launch 5 with the v2 planes: `text_dfin7s` with the shared down's finish reading layout SLM.
pub const text_dfin7t =
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint k = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    threadgroup float term[TOPKC * 128u];
    \\    threadgroup OUT_T sterm[128];
    \\    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    \\    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    \\    if (k < TOPKC) {
    \\        const uint rk = row * TOPKC + k;
    \\        const uint e = uint(slots[rk]);
    \\        float v[4];
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const uint o = blk * 128u + L + 32u * q;
    \\            float t = 0.0f;
    \\            for (uint ks = 0; ks < KSC; ++ks) t += zd[(ulong(ks) * ulong(RKC) + rk) * ulong(HIDC) + o];
    \\            v[q] = t;
    \\        }
    \\        if (DEFER != 0) {
    \\            float hsum = 0.0f;
    \\            for (uint q = 0; q < uint(NBI); ++q) hsum += hsx[ulong(rk) * uint(NBI) + q];
    \\            for (uint q = 0; q < 4u; ++q) v[q] = fma(v[q], MUL1_A, MUL1_C * hsum);
    \\        }
    \\        for (uint q = 0; q < 4u; ++q) v[q] = float(half(v[q]));
    \\        turboexl3_h128r(v, L);
    \\        for (uint q = 0; q < 4u; ++q) {
    \\            const uint o = blk * 128u + L + 32u * q;
    \\            term[k * 128u + L + 32u * q] = float(half(v[q] * HAD_SCALE * float(down_svh[e * HIDC + o])));
    \\        }
    \\    } else {
    \\        float s_xsum = 0.0f;
    \\        if (SDEF != 0) for (uint q = 0; q < uint(NBI); ++q) s_xsum += sxsd[row * uint(NBI) + q];
    \\        float v[4];
    \\        turboexl3_sfinish2<uint(SKSD), uint(SRS), HIDC, uint(SDEF), uint(SLM)>(szd, sh_down_svh, s_xsum, row, blk, L, v);
    \\        for (uint q = 0; q < 4u; ++q) sterm[L + 32u * q] = OUT_T(v[q]);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (tid < 128u) {
    \\        float acc = 0.0f;
    \\        for (uint j = 0; j < TOPKC; ++j) acc = fma(float(scores[row * TOPKC + j]), term[j * 128u + tid], acc);
    \\        const OUT_T yb = OUT_T(acc);
    \\        const OUT_T sig = sigtab[as_type<ushort>(slogit[row])];
    \\        const OUT_T gated = sig * sterm[tid];
    \\        y[ulong(row) * ulong(HIDC) + blk * 128u + tid] = yb + gated;
    \\    }
    \\
;

/// prefill.zig's `replaceOnce` + `sentinel` for the v2 routed texts: exactly one occurrence, or the build fails.
fn spliceOnce(comptime s: []const u8, comptime needle: []const u8, comptime with: []const u8) [:0]const u8 {
    comptime {
        @setEvalBranchQuota(10_000_000);
        const first = std.mem.indexOf(u8, s, needle) orelse @compileError("splice not found: " ++ needle);
        if (std.mem.indexOfPos(u8, s, first + needle.len, needle) != null) @compileError("splice found twice: " ++ needle);
        const out: []const u8 = s[0..first] ++ with ++ s[first + needle.len ..];
        var buf: [out.len:0]u8 = undefined;
        @memcpy(buf[0..out.len], out);
        const final = buf;
        return &final;
    }
}

/// The routed bodies under the v2 heads: rk = y - SYOFF (SYOFF 0: v1's rows; > 0: the shared rows first).
const rk_line = "const uint rk = threadgroup_position_in_grid.y;";
const rk_line_y = "const uint rk = threadgroup_position_in_grid.y - uint(SYOFF);";
const text_gu8_y = spliceOnce(text_gu8, rk_line, rk_line_y);
const text_gu3_y = spliceOnce(text_gu3, rk_line, rk_line_y);
const text_down3_y = spliceOnce(text_down3, rk_line, rk_line_y);
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
/// its statements from it (`textSlice`), and the host's test pins this copy to its own source
/// (`exl3_text_router_softmax`). Inputs logits; outputs inds, scores; template TOUT, NE, NK, TG, NORM (the host's
/// config: TG = 256 at 512 experts).
pub const text_router_softmax: [:0]const u8 =
    \\uint row = thread_position_in_grid.y;
    \\uint tid = thread_position_in_threadgroup.x;
    \\uint lane = thread_index_in_simdgroup;
    \\
    \\threadgroup float rk[NE];
    \\
    \\threadgroup uint sel[NK];
    \\
    \\size_t rbase = (size_t)row * (size_t)NE;
    \\for (uint e = tid; e < NE; e += TG) {
    \\  rk[e] = float(logits[rbase + e]);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\// Selection runs on simdgroup 0 only: K rounds of "max, then mask".
    \\// simd_min over the candidate indices makes the tie-break the LOWEST
    \\// expert id, so the pick is a pure function of the keys.
    \\if (tid < 32) {
    \\
    \\  constexpr int VT = (NE + 3) / 4;
    \\  constexpr int SIMDS = (VT + 31) / 32;
    \\  float lane_max = -INFINITY;
    \\  for (int g = 0; g < SIMDS; ++g) {
    \\    int vt = g * 32 + int(lane);
    \\    float m = -INFINITY;
    \\    for (int i = 0; i < 4; ++i) {
    \\      int e = vt * 4 + i;
    \\      if (e < NE) m = metal::max(m, rk[e]);
    \\    }
    \\    float mg = simd_max(m);
    \\    if (int(lane) == g) lane_max = mg;
    \\  }
    \\  float smax = simd_max(lane_max);
    \\  float lane_sum = 0.0f;
    \\  for (int g = 0; g < SIMDS; ++g) {
    \\    int vt = g * 32 + int(lane);
    \\    float acc = 0.0f;
    \\    for (int i = 0; i < 4; ++i) {
    \\      int e = vt * 4 + i;
    \\      acc += (e < NE) ? fast::exp(rk[e] - smax) : 0.0f;
    \\    }
    \\    float pg = simd_sum(acc);
    \\    if (int(lane) == g) lane_sum = pg;
    \\  }
    \\  float snorm = 1.0f / simd_sum(lane_sum);
    \\  // The lane's strided slice of the keys lives in registers for the K rounds: a round is
    \\  // a register max, two simd reductions and a register mask, no threadgroup traffic.
    \\  constexpr uint EPL = (NE + 31) / 32;
    \\  float mine[EPL];
    \\  for (uint i = 0; i < EPL; ++i) {
    \\    uint e = lane + 32 * i;
    \\    mine[i] = (e < NE) ? rk[e] : -INFINITY;
    \\  }
    \\  for (uint r = 0; r < NK; ++r) {
    \\    float best = -INFINITY;
    \\    uint bslot = 0xFFFFFFFFu;
    \\    for (uint i = 0; i < EPL; ++i) {
    \\      if (mine[i] > best) { best = mine[i]; bslot = i; }
    \\    }
    \\    float gmax = simd_max(best);
    \\    uint cand = (best == gmax) ? lane + 32 * bslot : 0xFFFFFFFFu;
    \\    uint gidx = metal::min(simd_min(cand), uint(NE - 1));
    \\    if (gidx % 32 == lane) mine[gidx / 32] = -INFINITY;
    \\    if (lane == 0) sel[r] = gidx;
    \\  }
    \\  simdgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\  float w = 0.0f;
    \\  uint myidx = 0;
    \\  if (lane < NK) {
    \\    myidx = sel[lane];
    \\    w = float(TOUT(fast::exp(float(logits[rbase + myidx]) - smax) * snorm));
    \\  }
    \\  // MLX's `row_reduce_small` owns an 8-wide row in ONE thread and
    \\  // folds it in ASCENDING order in the reduce's accumulator type —
    \\  // which for a bf16 input is bf16, rounding after every add. A simd
    \\  // butterfly over the same values is a different number.
    \\  float tot = 0.0f;
    \\  for (int j = 0; j < NK; ++j) tot = float(TOUT(simd_shuffle(w, ushort(j)) + tot));
    \\  w = w / float(TOUT(tot));
    \\  if (lane < NK) {
    \\    inds[(size_t)row * (size_t)NK + lane] = myidx;
    \\    scores[(size_t)row * (size_t)NK + lane] = TOUT(w);
    \\  }
    \\}
    \\
;
pub const router_inputs = [_][*:0]const u8{"logits"};
pub const router_outputs = [_][*:0]const u8{ "inds", "scores" };

/// The verbatim slice of `src` from the first `from` through the first `to` after it (the build fails without both).
fn textSlice(comptime src: []const u8, comptime from: []const u8, comptime to: []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(10_000_000);
        const a = std.mem.indexOf(u8, src, from) orelse @compileError("router fold: slice start not found: " ++ from);
        const b = std.mem.indexOfPos(u8, src, a, to) orelse @compileError("router fold: slice end not found: " ++ to);
        return src[a .. b + to.len];
    }
}

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
const router_sel = textSlice(text_router_softmax, "  // The lane's strided slice", "  simdgroup_barrier(mem_flags::mem_threadgroup);\n");
const router_softmax = textSlice(text_router_softmax, "  constexpr int VT = (NE + 3) / 4;", "  float snorm = 1.0f / simd_sum(lane_sum);\n");
const router_weights = textSlice(text_router_softmax, "  float w = 0.0f;", "    scores[(size_t)row * (size_t)NK + lane] = TOUT(w);\n  }\n");
const prep2_body = textSlice(text_prep2, "    float vg[4]; float vu[4];", "su[ulong(rk) * ulong(DIMC / 128u) + b] = suv; }\n");

/// The shared gate's logit for one row exactly as MLX computes it at the verify widths, as one simdgroup's part
/// (`gx` x [R, GK] bf16, `gw` the gate's weight vector [GK] bf16, `gp` the 8 partials): MLX 0.32's `matmul` of x and the gate's
/// [GK, 1] at R = 2..4 (GK > 64, GK >= 16 R) runs gemv_bfloat16_bm1_bn8_sm1_sn32_tm{1,4}_tn4 (gemv.h GEMVKernel::run), whose
/// simdgroup gs of 8 and lane sum the columns (32 gs + lane) * 4 + 1024 i + tn in that order, the leftover block zero-filled
/// past GK (load_safe), then fold a shuffle-down tree; this simdgroup plays the gemv's simdgroup gs = tid / 32 - 1. The host's
/// router mode `softmax_gate_gemv` splices the same text (its test pins it to this one).
pub const text_gate_gemv =
    \\  {
    \\    // MLX's gemv (gemv.h GEMVKernel::run, BM 1 BN 8 SM 1 SN 32 TN 4) for one output row: simdgroup gs, its lanes
    \\    const int gs = int(tid / 32u) - 1;
    \\    float gres = 0.0f;
    \\    int gbn = (32 * gs + int(lane)) * 4;
    \\    const device bfloat16_t* gmat = gx + (size_t)row * (size_t)GK;
    \\    float gcoef[4];
    \\    bfloat16_t ginter[4];
    \\    for (int i = 0; i < GK / 1024; ++i) {
    \\      for (int tn = 0; tn < 4; tn++) gcoef[tn] = static_cast<float>(gw[gbn + tn]);
    \\      for (int tn = 0; tn < 4; tn++) ginter[tn] = static_cast<bfloat16_t>(gmat[gbn + tn]);
    \\      for (int tn = 0; tn < 4; tn++) gres += ginter[tn] * gcoef[tn];
    \\      gbn += 1024;
    \\    }
    \\    if (GK % 1024 > 0) {
    \\      if (gbn + 4 <= GK) {
    \\        for (int tn = 0; tn < 4; tn++) gcoef[tn] = static_cast<float>(gw[gbn + tn]);
    \\      } else {
    \\        for (int tn = 0; tn < 4; tn++) gcoef[tn] = gbn + tn < GK ? static_cast<float>(gw[gbn + tn]) : float(0);
    \\      }
    \\      if (gbn + 4 <= GK) {
    \\        for (int tn = 0; tn < 4; tn++) ginter[tn] = static_cast<bfloat16_t>(gmat[gbn + tn]);
    \\      } else {
    \\        for (int tn = 0; tn < 4; tn++) ginter[tn] = gbn + tn < GK ? static_cast<bfloat16_t>(gmat[gbn + tn]) : bfloat16_t(0);
    \\      }
    \\      for (int tn = 0; tn < 4; tn++) gres += ginter[tn] * gcoef[tn];
    \\    }
    \\    for (ushort sn = 16; sn >= 1; sn >>= 1) gres += simd_shuffle_down(gres, sn);
    \\    if (lane == 0) gp[gs] = gres;
    \\  }
    \\
;
/// The gate's sum of the 8 partials as the gemv's simdgroup 0 lane 0 folds them (its own first, then 1..7 in order), and its
/// bf16 store.
pub const text_gate_sum =
    \\    float gt = gp[0];
    \\    for (int sgn = 1; sgn < 8; sgn++) gt += gp[sgn];
    \\    glogit[row] = static_cast<bfloat16_t>(gt);
    \\
;

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
        \\uint row = thread_position_in_grid.y;
        \\uint tid = thread_position_in_threadgroup.x;
        \\uint lane = thread_index_in_simdgroup;
        \\const uint b = threadgroup_position_in_grid.x;
        \\const uint sgi = simdgroup_index_in_threadgroup;
        \\constexpr uint NSG = uint(TG) / 32u;
        \\
        \\threadgroup float rk[NE];
        \\threadgroup uint sel[NK];
        \\threadgroup float tsm[2];
        \\
        ,
        if (gate) "threadgroup float gp[8];\n" else "",
        \\size_t rbase = (size_t)row * (size_t)NE;
        \\for (uint e = tid; e < NE; e += TG) {
        \\  rk[e] = float(logits[rbase + e]);
        \\}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\// phase 1: the selection (simdgroup 0), the softmax tree (block 0's last simdgroup) [, the shared gate's dot (block 0's 1..8)]
        \\if (tid < 32) {
        \\
        ,
        router_sel,
        \\} else if (b == 0u && sgi == NSG - 1u) {
        \\
        ,
        router_softmax,
        \\  if (lane == 0) { tsm[0] = smax; tsm[1] = snorm; }
        \\}
        ,
        if (gate) " else if (b == 0u && sgi >= 1u && sgi <= 8u)\n" ++ text_gate_gemv else "\n",
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\// phase 2: prep2 for row-slot (row, sgi) and block b; block 0's last simdgroup: the router's weights, slots and scores
        \\if (sgi < uint(NK)) {
        \\  constexpr uint DIMC = uint(DIM);
        \\  constexpr uint TOPKC = uint(TOPK);
        \\  constexpr float HAD_SCALE = 0.08838834764831845f;
        \\  const uint rk = row * TOPKC + sgi;
        \\  const uint L = lane;
        \\  const uint e = sel[sgi];
        \\
        ,
        prep2_body,
        \\} else if (b == 0u && sgi == NSG - 1u) {
        \\  const float smax = tsm[0];
        \\  const float snorm = tsm[1];
        \\
        ,
        router_weights,
        if (gate) "  if (lane == 0) {\n" ++ text_gate_sum ++ "  }\n" else "",
        \\}
        \\
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
    /// MUL1's deferred affine map (MLX_SERVE_TURBOEXL3_DEFER=0 decodes every weight to its f16 value)
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
            .defer_affine = geo.dec.codebook == .mul1 and blk: {
                const v = std.c.getenv("MLX_SERVE_TURBOEXL3_DEFER") orelse break :blk true;
                break :blk v[0] != '0';
            },
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
        inline for (.{ self.gu, self.dn, self.red, self.dn6 }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
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
        try t.expect(std.mem.indexOf(u8, txt, rk_line_y) != null);
        try t.expect(std.mem.indexOf(u8, txt, rk_line) == null);
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
    try t.expect(std.mem.indexOf(u8, metal_shared2, "turboexl3_scw<N, WMASK, 7>(lo, mid, hi)") != null);
    try t.expect(std.mem.indexOf(u8, metal_shared2, "as_type<ulong>") == null);
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
    const t = std.testing;
    const texts = [_][]const u8{ text_route_prep, text_route_prepg };
    // the pieces are slices of the router copy and of prep2 (textSlice), and each lands once in each merged text
    for ([_][]const u8{ router_sel, router_softmax, router_weights }) |piece| {
        try t.expect(std.mem.indexOf(u8, text_router_softmax, piece) != null);
        for (texts) |txt| try t.expectEqual(@as(usize, 1), std.mem.count(u8, txt, piece));
    }
    try t.expect(std.mem.indexOf(u8, text_prep2, prep2_body) != null);
    for (texts) |txt| try t.expectEqual(@as(usize, 1), std.mem.count(u8, txt, prep2_body));
    // the value-setting lines of the router's rounds, softmax and weights, and of prep2's rotation
    const lines = [_][]const u8{
        "  rk[e] = float(logits[rbase + e]);",
        "      if (mine[i] > best) { best = mine[i]; bslot = i; }",
        "    uint gidx = metal::min(simd_min(cand), uint(NE - 1));",
        "    if (gidx % 32 == lane) mine[gidx / 32] = -INFINITY;",
        "      if (e < NE) m = metal::max(m, rk[e]);",
        "  float smax = simd_max(lane_max);",
        "      acc += (e < NE) ? fast::exp(rk[e] - smax) : 0.0f;",
        "  float snorm = 1.0f / simd_sum(lane_sum);",
        "    w = float(TOUT(fast::exp(float(logits[rbase + myidx]) - smax) * snorm));",
        "  for (int j = 0; j < NK; ++j) tot = float(TOUT(simd_shuffle(w, ushort(j)) + tot));",
        "  w = w / float(TOUT(tot));",
        "    scores[(size_t)row * (size_t)NK + lane] = TOUT(w);",
        "const float xr = float(half(float(x[row * DIMC + i]))); vg[j] = xr * float(gate_suh[e * DIMC + i]); vu[j] = xr * float(up_suh[e * DIMC + i]);",
        "        const half hg = half(vg[j] * HAD_SCALE);",
        "    sgv = simd_sum(sgv);",
    };
    for (lines) |line| {
        try t.expect(std.mem.indexOf(u8, text_router_softmax ++ text_prep2, line) != null);
        for (texts) |txt| try t.expect(std.mem.indexOf(u8, txt, line) != null);
    }
    // the hand-offs: smax / snorm through threadgroup memory, the expert from the selection, the row-slot of the simdgroup
    for (texts) |txt| {
        try t.expect(std.mem.indexOf(u8, txt, "if (lane == 0) { tsm[0] = smax; tsm[1] = snorm; }") != null);
        try t.expect(std.mem.indexOf(u8, txt, "  const float smax = tsm[0];\n  const float snorm = tsm[1];\n") != null);
        try t.expect(std.mem.indexOf(u8, txt, "  const uint e = sel[sgi];") != null);
        try t.expect(std.mem.indexOf(u8, txt, "  const uint rk = row * TOPKC + sgi;") != null);
        try t.expectEqual(@as(usize, 2), std.mem.count(u8, txt, "threadgroup_barrier(mem_flags::mem_threadgroup);"));
    }
    // the shared gate's dot and sum only in the gate kernel
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, text_route_prepg, text_gate_gemv));
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, text_route_prepg, text_gate_sum));
    try t.expect(std.mem.indexOf(u8, text_route_prep, "gp[") == null);
    // every input / output name and template constant is read or written
    for (route_prep_inputs ++ route_prep_outputs) |n| try t.expect(std.mem.indexOf(u8, text_route_prep, std.mem.span(n)) != null);
    for (route_prepg_inputs ++ route_prepg_outputs) |n| try t.expect(std.mem.indexOf(u8, text_route_prepg, std.mem.span(n)) != null);
    for ([_][]const u8{ "NE", "NK", "TG", "TOUT", "DIM", "TOPK" }) |name| for (texts) |txt| try t.expect(std.mem.indexOf(u8, txt, name) != null);
    try t.expect(std.mem.indexOf(u8, text_route_prepg, "GK") != null);
    for (router_inputs ++ router_outputs) |n| try t.expect(std.mem.indexOf(u8, text_router_softmax, std.mem.span(n)) != null);
    for ([_][]const u8{ "gx", "gw", "GK", "gp[gs] = gres", "simd_shuffle_down(gres, sn)" }) |name| try t.expect(std.mem.indexOf(u8, text_gate_gemv, name) != null);
    // one simdgroup per slot plus the weights', with the gate's eight beside the selection
    try t.expectEqual(@as(u32, 11), Kernels3.routeSimdgroups(10, false));
    try t.expectEqual(@as(u32, 11), Kernels3.routeSimdgroups(10, true));
    try t.expectEqual(@as(u32, 5), Kernels3.routeSimdgroups(4, false));
    try t.expectEqual(@as(u32, 10), Kernels3.routeSimdgroups(4, true));
}
