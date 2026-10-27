    constexpr uint N = uint(NHW);
    constexpr uint NW = N / 2u;
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr uint INC = uint(IN);
    constexpr uint OUTC = uint(OUT);
    constexpr uint OUT_TILES = OUTC / 16u;
    const uint tile_id = threadgroup_position_in_grid.x * 4u + simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    if (tile_id >= (INC / 16u) * OUT_TILES) return;
    const uint tk = tile_id / OUT_TILES;
    const uint tn = tile_id - tk * OUT_TILES;
    const device uint* tile = (const device uint*)(w) + ulong(tile_id) * ulong(NW);
    const uint qb = exl3_gen_qb<N>(L);
    const uint ga = exl3_gen_a<N>(L);
    const uint lp = exl3_gen_lp<N>(L);
    const auto ww = exl3_tile_load<N, 0>(tile, qb, ga, lp);
    uint cw[8];
    cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
    cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
    for (uint m = 0; m < 8u; ++m) {
        const uint r = turboexl3_row_of(L, m & 3u);
        const uint c = (L >> 2u) + 8u * (m >> 2u);
        W[ulong(tk * 16u + r) * ulong(OUTC) + tn * 16u + c] = exl3_decode<CB>(cw[m]);
    }
