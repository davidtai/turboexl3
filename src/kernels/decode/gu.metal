    constexpr uint N = uint(NHW);
    constexpr uint NW = N / 2u;
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr int PRB = int(PROBE);
    constexpr uint HIDC = uint(HID);
    constexpr uint INTC = uint(INTER);
    constexpr uint TOPKC = uint(TOPK);
    constexpr uint IN_TILES = HIDC / 16u;
    constexpr uint OUT_TILES = INTC / 16u;
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint blk = threadgroup_position_in_grid.x;      // 128-wide block of the intermediate
    const uint k = threadgroup_position_in_grid.y;        // slot
    const uint row = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint e = uint(slots[row * TOPKC + k]);
    threadgroup half xg[HIDC];
    threadgroup half xu[HIDC];
    threadgroup float red_g[4][128];
    threadgroup float red_u[4][128];
    // 1. the rotated inputs for gate and up (their suh differ): simdgroup sg owns blocks sg, sg + 4, ...
    for (uint b = sg; b < HIDC / 128u; b += 4u) {
        float vg[4]; float vu[4];
        for (uint j = 0; j < 4u; ++j) {
            const uint i = b * 128u + L + 32u * j;
            const float xr = float(half(x[row * HIDC + i]));
            vg[j] = xr * float(gate_suh[e * HIDC + i]);
            vu[j] = xr * float(up_suh[e * HIDC + i]);
        }
        for (uint s = 0; s < 5u; ++s) {
            const uint bit = 1u << s;
            for (uint j = 0; j < 4u; ++j) {
                const float pg = simd_shuffle_xor(vg[j], ushort(bit));
                const float pu = simd_shuffle_xor(vu[j], ushort(bit));
                vg[j] = (L & bit) ? (pg - vg[j]) : (vg[j] + pg);
                vu[j] = (L & bit) ? (pu - vu[j]) : (vu[j] + pu);
            }
        }
        {   // stages 5 and 6 pair the registers (positions L + 32 j: bit 5 is j & 1, bit 6 is j & 2)
            float tg0 = vg[0] + vg[1], tg1 = vg[0] - vg[1], tg2 = vg[2] + vg[3], tg3 = vg[2] - vg[3];
            float tu0 = vu[0] + vu[1], tu1 = vu[0] - vu[1], tu2 = vu[2] + vu[3], tu3 = vu[2] - vu[3];
            vg[0] = tg0 + tg2; vg[1] = tg1 + tg3; vg[2] = tg0 - tg2; vg[3] = tg1 - tg3;
            vu[0] = tu0 + tu2; vu[1] = tu1 + tu3; vu[2] = tu0 - tu2; vu[3] = tu1 - tu3;
        }
        for (uint j = 0; j < 4u; ++j) {
            const uint i = b * 128u + L + 32u * j;
            xg[i] = half(vg[j] * HAD_SCALE);
            xu[i] = half(vu[j] * HAD_SCALE);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2. the GEMV over this simdgroup's quarter of the input tiles, the block's 8 output tiles, gate and up
    constexpr bool INTK = (N % 16u) == 0u;
    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    uint off[8];
    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    uint rows_of[4];
    for (uint m = 0; m < 4u; ++m) rows_of[m] = turboexl3_row_of(L, m);
    float ag[8][2]; float au[8][2];
    for (uint t = 0; t < 8u; ++t) { ag[t][0] = 0.0f; ag[t][1] = 0.0f; au[t][0] = 0.0f; au[t][1] = 0.0f; }
    const uint tk_per = IN_TILES / 4u;
    const device uint* gbase = (const device uint*)(gate_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    const device uint* ubase = (const device uint*)(up_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    for (uint tk = sg * tk_per; tk < (sg + 1u) * tk_per; ++tk) {
        float xr_g[4]; float xr_u[4];
        for (uint m = 0; m < 4u; ++m) { xr_g[m] = float(xg[tk * 16u + rows_of[m]]); xr_u[m] = float(xu[tk * 16u + rows_of[m]]); }
        for (uint t = 0; t < 8u; ++t) {
            const uint tn = blk * 8u + t;
            const ulong toff = (ulong(tk) * OUT_TILES + tn) * ulong(NW);
            if (INTK) {
                const IntWords wg = exl3_int_words<N>(gbase + toff, base, ia);
                const IntWords wu = exl3_int_words<N>(ubase + toff, base, ia);
                uint cg[8], cu[8];
                cg[0] = exl3_int_cw<N, WMASK, 0>(wg, ib); cg[1] = exl3_int_cw<N, WMASK, 1>(wg, ib); cg[2] = exl3_int_cw<N, WMASK, 2>(wg, ib); cg[3] = exl3_int_cw<N, WMASK, 3>(wg, ib);
                cg[4] = exl3_int_cw<N, WMASK, 4>(wg, ib); cg[5] = exl3_int_cw<N, WMASK, 5>(wg, ib); cg[6] = exl3_int_cw<N, WMASK, 6>(wg, ib); cg[7] = exl3_int_cw<N, WMASK, 7>(wg, ib);
                cu[0] = exl3_int_cw<N, WMASK, 0>(wu, ib); cu[1] = exl3_int_cw<N, WMASK, 1>(wu, ib); cu[2] = exl3_int_cw<N, WMASK, 2>(wu, ib); cu[3] = exl3_int_cw<N, WMASK, 3>(wu, ib);
                cu[4] = exl3_int_cw<N, WMASK, 4>(wu, ib); cu[5] = exl3_int_cw<N, WMASK, 5>(wu, ib); cu[6] = exl3_int_cw<N, WMASK, 6>(wu, ib); cu[7] = exl3_int_cw<N, WMASK, 7>(wu, ib);
                for (uint m = 0; m < 8u; ++m) {
                    const float wgv = (PRB == 0) ? float(exl3_decode<CB>(cg[m])) : ((PRB == 1) ? float(as_type<half>(ushort(cg[m] & 0x3fffu))) : float(as_type<half>(ushort((wg.h[0] >> (m * 2u)) & 0x3fffu))));
                    const float wuv = (PRB == 0) ? float(exl3_decode<CB>(cu[m])) : ((PRB == 1) ? float(as_type<half>(ushort(cu[m] & 0x3fffu))) : float(as_type<half>(ushort((wu.h[0] >> (m * 2u)) & 0x3fffu))));
                    const uint c = (m >> 2);
                    ag[t][c] = fma(xr_g[m & 3u], wgv, ag[t][c]);
                    au[t][c] = fma(xr_u[m & 3u], wuv, au[t][c]);
                }
            } else {
                const LaneWords wg = exl3_lane_words<N>(gbase + toff, base);
                const LaneWords wu = exl3_lane_words<N>(ubase + toff, base);
                for (uint m = 0; m < 8u; ++m) {
                    const float wgv = float(exl3_decode<CB>(exl3_lane_cw<WMASK>(wg, off[m])));
                    const float wuv = float(exl3_decode<CB>(exl3_lane_cw<WMASK>(wu, off[m])));
                    const uint c = (m >> 2);
                    ag[t][c] = fma(xr_g[m & 3u], wgv, ag[t][c]);
                    au[t][c] = fma(xr_u[m & 3u], wuv, au[t][c]);
                }
            }
        }
    }
    // 3. the column sums: lanes 4c..4c+3 hold column c (and c + 8); fold them, then the four simdgroups
    for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) {
        ag[t][c] += simd_shuffle_xor(ag[t][c], ushort(1)); ag[t][c] += simd_shuffle_xor(ag[t][c], ushort(2));
        au[t][c] += simd_shuffle_xor(au[t][c], ushort(1)); au[t][c] += simd_shuffle_xor(au[t][c], ushort(2));
    }
    if ((L & 3u) == 0u) {
        for (uint t = 0; t < 8u; ++t) {
            red_g[sg][t * 16u + (L >> 2)] = ag[t][0]; red_g[sg][t * 16u + (L >> 2) + 8u] = ag[t][1];
            red_u[sg][t * 16u + (L >> 2)] = au[t][0]; red_u[sg][t * 16u + (L >> 2) + 8u] = au[t][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 4. simdgroup 0: the block's 128 outputs (positions L + 32 j): f16 round, H128, svh, f16 round, SwiGLU
    if (sg == 0u) {
        float vg[4]; float vu[4];
        for (uint j = 0; j < 4u; ++j) {
            const uint i = L + 32u * j;
            vg[j] = float(half(((red_g[0][i] + red_g[1][i]) + red_g[2][i]) + red_g[3][i]));
            vu[j] = float(half(((red_u[0][i] + red_u[1][i]) + red_u[2][i]) + red_u[3][i]));
        }
        for (uint s = 0; s < 5u; ++s) {
            const uint bit = 1u << s;
            for (uint j = 0; j < 4u; ++j) {
                const float pg = simd_shuffle_xor(vg[j], ushort(bit));
                const float pu = simd_shuffle_xor(vu[j], ushort(bit));
                vg[j] = (L & bit) ? (pg - vg[j]) : (vg[j] + pg);
                vu[j] = (L & bit) ? (pu - vu[j]) : (vu[j] + pu);
            }
        }
        {
            float tg0 = vg[0] + vg[1], tg1 = vg[0] - vg[1], tg2 = vg[2] + vg[3], tg3 = vg[2] - vg[3];
            float tu0 = vu[0] + vu[1], tu1 = vu[0] - vu[1], tu2 = vu[2] + vu[3], tu3 = vu[2] - vu[3];
            vg[0] = tg0 + tg2; vg[1] = tg1 + tg3; vg[2] = tg0 - tg2; vg[3] = tg1 - tg3;
            vu[0] = tu0 + tu2; vu[1] = tu1 + tu3; vu[2] = tu0 - tu2; vu[3] = tu1 - tu3;
        }
        for (uint j = 0; j < 4u; ++j) {
            const uint o = blk * 128u + L + 32u * j;
            const float g = float(half(vg[j] * HAD_SCALE * float(gate_svh[e * INTC + o])));
            const float u = float(half(vu[j] * HAD_SCALE * float(up_svh[e * INTC + o])));
            h[(row * TOPKC + k) * INTC + o] = (g / (1.0f + metal::exp(-g))) * u;
        }
    }
