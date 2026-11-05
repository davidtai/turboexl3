    const device half* xbase = (proj == 0u ? xgp : xup) + 4u * (L & 3u);
    const uint per = IN_TILES / (KSC * 2u);
    for (uint si = 0; si < SPTC; ++si) {
        const uint tk0 = ((kz * SPTC + si) * 2u + khalf) * per;
        if (m == 1u) turboexl3_gemv_group<N, WMASK, CB, OUT_TILES, BTC, 1u, uint(DEFER)>(wbase, xbase, ulong(HIDC), members, m, tk0, tk0 + per, blk, L, red[si][sg]);
        else if (m == 2u) turboexl3_gemv_group<N, WMASK, CB, OUT_TILES, BTC, 2u, uint(DEFER)>(wbase, xbase, ulong(HIDC), members, m, tk0, tk0 + per, blk, L, red[si][sg]);
        else if (m == 3u) turboexl3_gemv_group<N, WMASK, CB, OUT_TILES, BTC, 3u, uint(DEFER)>(wbase, xbase, ulong(HIDC), members, m, tk0, tk0 + per, blk, L, red[si][sg]);
        else turboexl3_gemv_group<N, WMASK, CB, OUT_TILES, BTC, 4u, uint(DEFER)>(wbase, xbase, ulong(HIDC), members, m, tk0, tk0 + per, blk, L, red[si][sg]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = tid; q < 16u * BTC * m; q += 128u) {
        const uint j = q / (16u * BTC);
        for (uint si = 0; si < SPTC; ++si) {
            const ulong o = (ulong(kz * SPTC + si) * ulong(RKC) + members[j]) * ulong(INTC) + blk * (16u * BTC) + (q - j * 16u * BTC);
            zg[o] = red[si][0][q] + red[si][1][q];
            zu[o] = red[si][2][q] + red[si][3][q];
        }
    }
