    // The epilogue. The tile's halves (MLX's store rounding) in threadgroup memory, row stride BN.
    threadgroup half zt[BM * BN];
    Dtile.template store<half, BN, 1>(&zt[tm * BN + tn]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
