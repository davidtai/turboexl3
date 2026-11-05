    for (uint i = tid; i < RBC * 16u * BTC; i += 128u) {
        const uint r = i / (16u * BTC);
        const uint row = r0 + r;
        // row-slot row G + g of the routed down finish's planes z [KS, ROWS G, OUT]
        if (row < ROWSC) z[(ulong(ksl) * ulong(ROWSC * GC) + row * GC + g) * ulong(OUTC) + blk * (16u * BTC) + (i - r * 16u * BTC)] = ((red[0][i] + red[1][i]) + red[2][i]) + red[3][i];
    }
