    lo_s[tid] = lo;
    if (tid == 0u) lo_s[1024] = SC;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint cnt = 0u, nt = 0u;
    if (tid < EC) { cnt = lo_s[tid + 1u] - lo; nt = (cnt + BMC - 1u) / BMC; }
    const uint incl = simd_prefix_inclusive_sum(nt);
    if (L == 31u) pref[sg] = incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) pref[L] = simd_prefix_exclusive_sum(pref[L]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint start = pref[sg] + incl - nt;
    for (uint t = 0u; t < nt; ++t) {
        const uint i = start + t;
        tiles[4u * i] = int(tid); tiles[4u * i + 1u] = int(lo + t * BMC); tiles[4u * i + 2u] = int(min(BMC, cnt - t * BMC)); tiles[4u * i + 3u] = 0;
    }
    if (tid == 1023u) total_s = start + nt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = total_s + tid; i < TMAXC; i += 1024u) { tiles[4u * i] = 0; tiles[4u * i + 1u] = 0; tiles[4u * i + 2u] = 0; tiles[4u * i + 3u] = 0; }
