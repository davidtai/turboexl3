    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = tid; q < 16u * BTC * m; q += 128u) {
        const uint j = q / (16u * BTC);
        for (uint si = 0; si < SPTC; ++si) {
            const ulong o = (ulong(kz * SPTC + si) * ulong(RKC) + members[j]) * ulong(INTC) + blk * (16u * BTC) + (q - j * 16u * BTC);
            zg[o] = red[si][0][q] + red[si][1][q];
            zu[o] = red[si][2][q] + red[si][3][q];
        }
    }
