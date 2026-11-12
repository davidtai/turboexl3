    // int8 (prefill.Options.int8 2, one scale a row): the row's amax over every 128-block for gate and up each (one
    // threadgroup a row: BPT = HID / 128), its scale amax / 127 in every block's slot of sxg / sxu, int8 in natural order;
    // the f16 values' block sums as prep_i8's
    half hg[4], hu[4];
    float sg = 0.0f, su = 0.0f, ag = 0.0f, au = 0.0f;
    for (uint j = 0; j < 4u; ++j) {
        hg[j] = half(vg[j] * HAD_SCALE);
        hu[j] = half(vu[j] * HAD_SCALE);
        sg += float(hg[j]);
        su += float(hu[j]);
        ag = max(ag, abs(float(hg[j])));
        au = max(au, abs(float(hu[j])));
    }
    sg = simd_sum(sg);
    su = simd_sum(su);
    ag = simd_max(ag);
    au = simd_max(au);
    threadgroup float amx[2u * BPTC];
    if (L == 0u) {
        amx[simdgroup_index_in_threadgroup] = ag;
        amx[BPTC + simdgroup_index_in_threadgroup] = au;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rg = 0.0f, ru = 0.0f;
    for (uint k = 0; k < BPTC; ++k) {
        rg = max(rg, amx[k]);
        ru = max(ru, amx[BPTC + k]);
    }
    const float ig = (rg > 0.0f) ? 127.0f / rg : 0.0f;
    const float iu = (ru > 0.0f) ? 127.0f / ru : 0.0f;
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        xqg[ulong(p) * HC + i] = int8_t(rint(float(hg[j]) * ig));
        xqu[ulong(p) * HC + i] = int8_t(rint(float(hu[j]) * iu));
    }
    if (L == 0u) {
        xsg[ulong(p) * (HC / 128u) + b] = sg;
        xsu[ulong(p) * (HC / 128u) + b] = su;
        sxg[ulong(p) * (HC / 128u) + b] = rg / 127.0f;
        sxu[ulong(p) * (HC / 128u) + b] = ru / 127.0f;
    }
