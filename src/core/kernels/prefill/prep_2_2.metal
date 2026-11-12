    float sg = 0.0f, su = 0.0f;
    const uint lo = (ALAY == 1) ? turboexl3_alay(L) : ((ALAY == 2) ? turboexl3_alay16(L) : L);
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + 32u * j + lo;
        const half hg = half(vg[j] * HAD_SCALE);
        const half hu = half(vu[j] * HAD_SCALE);
        xg[ulong(p) * HC + i] = hg;
        xu[ulong(p) * HC + i] = hu;
        sg += float(hg);
        su += float(hu);
    }
    sg = simd_sum(sg);
    su = simd_sum(su);
    if (L == 0u) { xsg[ulong(p) * (HC / 128u) + b] = sg; xsu[ulong(p) * (HC / 128u) + b] = su; }
