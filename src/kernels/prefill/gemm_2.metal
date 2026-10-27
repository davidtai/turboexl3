    threadgroup uint4 bs_mem[NBUF * BSZ * 2 / 16];
    threadgroup half* Bs = reinterpret_cast<threadgroup half*>(bs_mem);
