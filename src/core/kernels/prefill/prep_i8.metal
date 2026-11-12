    // int8 (prefill.Options.int8): each 128-block of the row at one scale, its amax / 127 over the f16 values prep_2_2
    // stores, int8 in natural order; the f16 values' block sums for the deferred map's C term, summed as prep_2_2 sums them
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
    const float ig = (ag > 0.0f) ? 127.0f / ag : 0.0f;
    const float iu = (au > 0.0f) ? 127.0f / au : 0.0f;
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        xqg[ulong(p) * HC + i] = int8_t(rint(float(hg[j]) * ig));
        xqu[ulong(p) * HC + i] = int8_t(rint(float(hu[j]) * iu));
    }
    if (L == 0u) {
        xsg[ulong(p) * (HC / 128u) + b] = sg;
        xsu[ulong(p) * (HC / 128u) + b] = su;
        sxg[ulong(p) * (HC / 128u) + b] = ag / 127.0f;
        sxu[ulong(p) * (HC / 128u) + b] = au / 127.0f;
    }
