        }
    }
    for (uint r = 0; r < RBC; ++r) turboexl3_fold<BTC>(a[r], &red[sg][r * 16u * BTC], L);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < RBC * 16u * BTC; i += 128u) {
        const uint r = i / (16u * BTC);
        const uint row = r0 + r;
        if (row < ROWSC) z[(ulong(ksl) * ulong(ROWSC) + row) * ulong(OUTC) + blk * (16u * BTC) + (i - r * 16u * BTC)] = ((red[0][i] + red[1][i]) + red[2][i]) + red[3][i];
    }
