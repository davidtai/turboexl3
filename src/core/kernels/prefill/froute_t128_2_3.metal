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
