    // int8 down input (prefill.Options.int8 1): each 128-block of the row at its own scale, amax / 127 over the f16 values
    // mid_5_2 stores, int8 in natural order (xdq [S + 128, INTER]: 128 spare rows a GEMM tile reads past S, never stored);
    // the f16 values' block sums as mid_5_2 sums them (xsd), the block scales (sxd [S + 128, INTER / 128])
    half hd[4];
    float sd = 0.0f, ad = 0.0f;
    for (uint j = 0; j < 4u; ++j) {
        hd[j] = half(d[j] * HAD_SCALE);
        sd += float(hd[j]);
        ad = max(ad, abs(float(hd[j])));
    }
    sd = simd_sum(sd);
    ad = simd_max(ad);
    const float id = (ad > 0.0f) ? 127.0f / ad : 0.0f;
    for (uint j = 0; j < 4u; ++j) xdq[ulong(p) * IC + b * 128u + L + 32u * j] = int8_t(rint(float(hd[j]) * id));
    if (L == 0u) {
        xsd[ulong(p) * (IC / 128u) + b] = sd;
        sxd[ulong(p) * (IC / 128u) + b] = ad / 127.0f;
    }
