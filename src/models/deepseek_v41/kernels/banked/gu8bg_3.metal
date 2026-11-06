    for (uint si = 0; si < SPTC; ++si) {
        const uint tk0 = (si * 2u + khalf) * PER;
        if (m == 1u) turboexl3_gemv_group<N, WMASK, CB, OUT_TILES, BTC, 1u, uint(DEFER)>(wbase, xbase, ld, lm, m, tk0, tk0 + PER, blk, L, red[si][sg]);
        else if (m == 2u) turboexl3_gemv_group<N, WMASK, CB, OUT_TILES, BTC, 2u, uint(DEFER)>(wbase, xbase, ld, lm, m, tk0, tk0 + PER, blk, L, red[si][sg]);
        else turboexl3_gemv_group<N, WMASK, CB, OUT_TILES, BTC, 4u, uint(DEFER)>(wbase, xbase, ld, lm, m, tk0, tk0 + PER, blk, L, red[si][sg]);
    }
