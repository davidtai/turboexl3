    lo_s[tid] = lo;
    if (tid == 0u) lo_s[1024] = SC;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint c0 = 0u, cnt = 0u, nq = 0u, nb = 0u, ns = 0u, nfull = 0u, r = 0u;
    if (tid < EC) {
        c0 = lo_s[tid + 1u] - lo;
