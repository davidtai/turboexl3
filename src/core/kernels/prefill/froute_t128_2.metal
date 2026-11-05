    lo_s[tid] = lo;
    if (tid == 0u) lo_s[1024] = SC;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint c0 = 0u, cnt = 0u, nq = 0u, nb = 0u, ns = 0u, nfull = 0u, r = 0u;
    if (tid < EC) {
        c0 = lo_s[tid + 1u] - lo;
        nq = (T128 != 0) ? c0 / 128u : 0u;
        cnt = c0 - 128u * nq;
        if (T128 != 0 && cnt > CU) { nq += 1u; cnt = 0u; }
        nfull = cnt / BMC;
        r = cnt - nfull * BMC;
        if (r == 0u) nb = nfull;
        else if (r > SR) nb = nfull + 1u;
        else { nb = nfull; ns = (r + 31u) / 32u; }
    }
    // the remainder's first sorted position
    const uint base = lo + c0 - cnt;
    if (tid < EC) xbase[tid] = int(base);
    const uint iq = simd_prefix_inclusive_sum(nq);
    const uint ib = simd_prefix_inclusive_sum(nb);
    const uint is = simd_prefix_inclusive_sum(ns);
    if (L == 31u) { prefq[sg] = iq; prefb[sg] = ib; prefs[sg] = is; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) { prefq[L] = simd_prefix_exclusive_sum(prefq[L]); prefb[L] = simd_prefix_exclusive_sum(prefb[L]); prefs[L] = simd_prefix_exclusive_sum(prefs[L]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sq = prefq[sg] + iq - nq;
    const uint sb = prefb[sg] + ib - nb;
    const uint ss = prefs[sg] + is - ns;
    for (uint t = 0u; t < nq; ++t) {
        const uint i = sq + t;
        tiles128[4u * i] = int(tid); tiles128[4u * i + 1u] = int(lo + 128u * t); tiles128[4u * i + 2u] = int(min(128u, c0 - 128u * t)); tiles128[4u * i + 3u] = 0;
    }
    for (uint t = 0u; t < nb; ++t) {
        const uint i = sb + t;
        tiles[4u * i] = int(tid); tiles[4u * i + 1u] = int(base + t * BMC); tiles[4u * i + 2u] = int(min(BMC, cnt - t * BMC)); tiles[4u * i + 3u] = 0;
    }
    for (uint t = 0u; t < ns; ++t) {
        const uint i = TBIG + ss + t;
        tiles[4u * i] = int(tid); tiles[4u * i + 1u] = int(base + nfull * BMC + 32u * t); tiles[4u * i + 2u] = int(min(32u, r - 32u * t)); tiles[4u * i + 3u] = 0;
    }
    if (tid == 1023u) { totq_s = sq + nq; totb_s = sb + nb; tots_s = ss + ns; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = totq_s + tid; i < TQ; i += 1024u) { tiles128[4u * i] = 0; tiles128[4u * i + 1u] = 0; tiles128[4u * i + 2u] = 0; tiles128[4u * i + 3u] = 0; }
    for (uint i = totb_s + tid; i < TBIG; i += 1024u) { tiles[4u * i] = 0; tiles[4u * i + 1u] = 0; tiles[4u * i + 2u] = 0; tiles[4u * i + 3u] = 0; }
    for (uint i = TBIG + tots_s + tid; i < TALL; i += 1024u) { tiles[4u * i] = 0; tiles[4u * i + 1u] = 0; tiles[4u * i + 2u] = 0; tiles[4u * i + 3u] = 0; }
