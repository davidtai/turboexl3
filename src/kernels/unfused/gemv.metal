    constexpr uint N = uint(NHW);
    constexpr uint NW = N / 2u;
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr uint INC = uint(IN);
    constexpr uint OUTC = uint(OUT);
    constexpr uint IN_TILES = INC / 16u;
    constexpr uint OUT_TILES = OUTC / 16u;
    constexpr bool INTK = (N % 16u) == 0u;
    const uint blk = threadgroup_position_in_grid.x;
    const uint rk = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint e = slots[rk];
    threadgroup float red[4][128];
    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    uint off[8];
    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    float a[8][2];
    for (uint t = 0; t < 8u; ++t) { a[t][0] = 0.0f; a[t][1] = 0.0f; }
    const uint tk_per = IN_TILES / 4u;
    const device uint* wbase = (const device uint*)(w) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    const device half* xrow = xp + ulong(rk) * ulong(INC) + 4u * (L & 3u);   // lane order: the quad's four rows
    for (uint tk = sg * tk_per; tk < (sg + 1u) * tk_per; ++tk) {
        const half4 x4 = *reinterpret_cast<const device half4*>(xrow + tk * 16u);
        const float xr[4] = { float(x4.x), float(x4.y), float(x4.z), float(x4.w) };
        for (uint t = 0; t < 8u; ++t) {
            const uint tn = blk * 8u + t;
            const device uint* tile = wbase + (ulong(tk) * OUT_TILES + tn) * ulong(NW);
            if (INTK) {
                const IntWords ww = exl3_int_words<N>(tile, base, ia);
                uint cw[8];
                cw[0] = exl3_int_cw<N, WMASK, 0>(ww, ib); cw[1] = exl3_int_cw<N, WMASK, 1>(ww, ib); cw[2] = exl3_int_cw<N, WMASK, 2>(ww, ib); cw[3] = exl3_int_cw<N, WMASK, 3>(ww, ib);
                cw[4] = exl3_int_cw<N, WMASK, 4>(ww, ib); cw[5] = exl3_int_cw<N, WMASK, 5>(ww, ib); cw[6] = exl3_int_cw<N, WMASK, 6>(ww, ib); cw[7] = exl3_int_cw<N, WMASK, 7>(ww, ib);
                for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(cw[m])), a[t][m >> 2]);
            } else {
                const LaneWords ww = exl3_lane_words<N>(tile, base);
                for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(exl3_lane_cw<WMASK>(ww, off[m]))), a[t][m >> 2]);
            }
        }
    }
    for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) { a[t][c] += simd_shuffle_xor(a[t][c], ushort(1)); a[t][c] += simd_shuffle_xor(a[t][c], ushort(2)); }
    if ((L & 3u) == 0u) for (uint t = 0; t < 8u; ++t) { red[sg][t * 16u + (L >> 2)] = a[t][0]; red[sg][t * 16u + (L >> 2) + 8u] = a[t][1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    z[ulong(rk) * ulong(OUTC) + blk * 128u + tid] = ((red[0][tid] + red[1][tid]) + red[2][tid]) + red[3][tid];
