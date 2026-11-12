    uint cw[8];
    cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
    cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
    for (uint m = 0; m < 8u; ++m) {
        const uint r = turboexl3_row_of(L, m & 3u);
        const uint c = (L >> 2u) + 8u * (m >> 2u);
        W[ulong(tk * 16u + r) * ulong(OUTC) + tn * 16u + c] = exl3_decode<CB>(cw[m]);
    }
