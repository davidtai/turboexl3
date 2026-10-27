    constexpr uint N = uint(NHW);
    constexpr uint NW = N / 2u;
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr int PRB = int(PROBE);
    constexpr uint HIDC = uint(HID);
    constexpr uint INTC = uint(INTER);
    constexpr uint TOPKC = uint(TOPK);
    constexpr uint IN_TILES = INTC / 16u;
    constexpr uint OUT_TILES = HIDC / 16u;
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint blk = threadgroup_position_in_grid.x;      // 128-wide block of the hidden output
    const uint row = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    threadgroup half hd[INTC];
    threadgroup float red[4][128];
    constexpr bool INTK = (N % 16u) == 0u;
    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    uint off[8];
    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    uint rows_of[4];
    for (uint m = 0; m < 4u; ++m) rows_of[m] = turboexl3_row_of(L, m);
    const uint tk_per = IN_TILES / 4u;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};              // simdgroup 0's outputs L + 32 j, summed over the slots
    for (uint k = 0; k < TOPKC; ++k) {
        const uint e = uint(slots[row * TOPKC + k]);
        const float score = float(scores[row * TOPKC + k]);
        // 1. this slot's rotated h: f16 round, down suh, H128, 1/sqrt(128), f16 round
        for (uint b = sg; b < INTC / 128u; b += 4u) {
            float v[4];
            for (uint j = 0; j < 4u; ++j) {
                const uint i = b * 128u + L + 32u * j;
                v[j] = float(half(h[(row * TOPKC + k) * INTC + i])) * float(down_suh[e * INTC + i]);
            }
            for (uint s = 0; s < 5u; ++s) {
                const uint bit = 1u << s;
                for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); }
            }
            { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
            for (uint j = 0; j < 4u; ++j) hd[b * 128u + L + 32u * j] = half(v[j] * HAD_SCALE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 2. the GEMV over this simdgroup's quarter of the input tiles, the block's 8 output tiles
        float a[8][2];
        for (uint t = 0; t < 8u; ++t) { a[t][0] = 0.0f; a[t][1] = 0.0f; }
        const device uint* dbase = (const device uint*)(down_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
        for (uint tk = sg * tk_per; tk < (sg + 1u) * tk_per; ++tk) {
            float xr[4];
            for (uint m = 0; m < 4u; ++m) xr[m] = float(hd[tk * 16u + rows_of[m]]);
            for (uint t = 0; t < 8u; ++t) {
                const uint tn = blk * 8u + t;
                if (INTK) {
                    const IntWords w = exl3_int_words<N>(dbase + (ulong(tk) * OUT_TILES + tn) * ulong(NW), base, ia);
                    uint cw[8];
                    cw[0] = exl3_int_cw<N, WMASK, 0>(w, ib); cw[1] = exl3_int_cw<N, WMASK, 1>(w, ib); cw[2] = exl3_int_cw<N, WMASK, 2>(w, ib); cw[3] = exl3_int_cw<N, WMASK, 3>(w, ib);
                    cw[4] = exl3_int_cw<N, WMASK, 4>(w, ib); cw[5] = exl3_int_cw<N, WMASK, 5>(w, ib); cw[6] = exl3_int_cw<N, WMASK, 6>(w, ib); cw[7] = exl3_int_cw<N, WMASK, 7>(w, ib);
                    for (uint m = 0; m < 8u; ++m) {
                        const float wv = (PRB == 0) ? float(exl3_decode<CB>(cw[m])) : ((PRB == 1) ? float(as_type<half>(ushort(cw[m] & 0x3fffu))) : float(as_type<half>(ushort((w.h[0] >> (m * 2u)) & 0x3fffu))));
                        a[t][m >> 2] = fma(xr[m & 3u], wv, a[t][m >> 2]);
                    }
                } else {
                    const LaneWords w = exl3_lane_words<N>(dbase + (ulong(tk) * OUT_TILES + tn) * ulong(NW), base);
                    for (uint m = 0; m < 8u; ++m) {
                        const float wv = float(exl3_decode<CB>(exl3_lane_cw<WMASK>(w, off[m])));
                        a[t][m >> 2] = fma(xr[m & 3u], wv, a[t][m >> 2]);
                    }
                }
            }
        }
        for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) { a[t][c] += simd_shuffle_xor(a[t][c], ushort(1)); a[t][c] += simd_shuffle_xor(a[t][c], ushort(2)); }
        if ((L & 3u) == 0u) for (uint t = 0; t < 8u; ++t) { red[sg][t * 16u + (L >> 2)] = a[t][0]; red[sg][t * 16u + (L >> 2) + 8u] = a[t][1]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 3. simdgroup 0: f16 round, H128, svh, f16 round, score-weighted accumulate
        if (sg == 0u) {
            float v[4];
            for (uint j = 0; j < 4u; ++j) { const uint i = L + 32u * j; v[j] = float(half(((red[0][i] + red[1][i]) + red[2][i]) + red[3][i])); }
            for (uint s = 0; s < 5u; ++s) {
                const uint bit = 1u << s;
                for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); }
            }
            { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
            for (uint j = 0; j < 4u; ++j) {
                const uint o = blk * 128u + L + 32u * j;
                acc[j] = fma(score, float(half(v[j] * HAD_SCALE * float(down_svh[e * HIDC + o]))), acc[j]);
            }
        }
        // the next slot's hd and red overwrite what every simdgroup just read
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0u) for (uint j = 0; j < 4u; ++j) y[row * HIDC + blk * 128u + L + 32u * j] = acc[j];
