    float sum = 0.0f;
    for (uint j = 0; j < 4u; ++j) {
        h[k][j] = half(v[j] * HAD_SCALE);
        sum += float(h[k][j]);
        am = max(am, abs(float(h[k][j])));
    }
    sum = simd_sum(sum);
    if (L == 0u) bsum[b] = sum;
    }
    // the row's amax over every block and its scale amax / 127, int8 rint(h 127 / amax) in natural order (prep_i8t's
    // statements); xsum the f16 values' sum, the blocks' sums in block order
    am = simd_max(am);
    if (L == 0u) amx[sg] = am;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    for (uint k = 0; k < 8u; ++k) r = max(r, amx[k]);
    const float ig = (r > 0.0f) ? 127.0f / r : 0.0f;
    for (uint k = 0; k < BPS; ++k) {
        const uint b = sg + 8u * k;
        if (b >= NB) break;
        for (uint j = 0; j < 4u; ++j) xq[ulong(row) * DIMC + b * 128u + L + 32u * j] = int8_t(rint(float(h[k][j]) * ig));
    }
    if (sg == 0u && L == 0u) {
        float t = 0.0f;
        for (uint b = 0; b < NB; ++b) t += bsum[b];
        xsum[row] = t;
        sx[row] = r / 127.0f;
    }
