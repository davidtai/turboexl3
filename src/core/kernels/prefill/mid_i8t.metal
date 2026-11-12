    // int8 down input (prefill.Options.int8 2, one scale a row): the row's amax over every 128-block (one threadgroup a row:
    // BPT = INTER / 128), its scale amax / 127 in every block's slot of sxd, int8 in natural order; the block sums as mid_i8's
    half hd[4];
    float sd = 0.0f, ad = 0.0f;
    for (uint j = 0; j < 4u; ++j) {
        hd[j] = half(d[j] * HAD_SCALE);
        sd += float(hd[j]);
        ad = max(ad, abs(float(hd[j])));
    }
    sd = simd_sum(sd);
    ad = simd_max(ad);
    threadgroup float amx[BPTC];
    if (L == 0u) amx[simdgroup_index_in_threadgroup] = ad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ar = 0.0f;
    for (uint k = 0; k < BPTC; ++k) ar = max(ar, amx[k]);
    const float id = (ar > 0.0f) ? 127.0f / ar : 0.0f;
    for (uint j = 0; j < 4u; ++j) xdq[ulong(p) * IC + b * 128u + L + 32u * j] = int8_t(rint(float(hd[j]) * id));
    if (L == 0u) {
        xsd[ulong(p) * (IC / 128u) + b] = sd;
        sxd[ulong(p) * (IC / 128u) + b] = ar / 127.0f;
    }
