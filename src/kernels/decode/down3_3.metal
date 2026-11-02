    const device half* xbase = hdp + 4u * (L & 3u);
    const device half* xrow0 = xbase + ulong(members[0]) * ulong(INTC);
    const device half* xrow1 = xbase + ulong(members[1]) * ulong(INTC);
    const uint per = IN_TILES / (KSC * 4u);
    for (uint si = 0; si < SPTC; ++si) {
        const uint tk0 = ((kz * SPTC + si) * 4u + sg) * per;
        if (m == 1u) turboexl3_gemv_members<N, WMASK, CB, OUT_TILES, BTC, 1u, uint(DEFER)>(wbase, xrow0, xrow0, tk0, tk0 + per, blk, L, red[si][sg]);
        else turboexl3_gemv_members<N, WMASK, CB, OUT_TILES, BTC, 2u, uint(DEFER)>(wbase, xrow0, xrow1, tk0, tk0 + per, blk, L, red[si][sg]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16u * BTC * m) {
        const uint j = tid / (16u * BTC);
        for (uint si = 0; si < SPTC; ++si)
            zd[(ulong(kz * SPTC + si) * ulong(RKC) + members[j]) * ulong(HIDC) + blk * (16u * BTC) + (tid - j * 16u * BTC)] = ((red[si][0][tid] + red[si][1][tid]) + red[si][2][tid]) + red[si][3][tid];
    }
