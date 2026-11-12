    // the weight stage, and past it with DFR 64 the fused finish's own staging (64 rows of a 128-column block, 16 KB);
    // DFR 32 stages each round in the stage buffer the pair's last stage was read from
    threadgroup uint4 bs_mem[NBUF * BSZ * 2 / 16 + ((DFR == 64) ? 64 * 128 * 2 / 16 : 0)];
    threadgroup half* Bs = reinterpret_cast<threadgroup half*>(bs_mem);
