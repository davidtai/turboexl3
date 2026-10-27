        turboexl3_h128x4(v, L);
        for (uint j = 0; j < 4u; ++j) {
            const uint i = b * 128u + L + 32u * j;
            acc[j] += w * float(half(v[j] * HAD_SCALE * float(svh_d[ulong(e) * HC + i])));
        }
    }
    for (uint j = 0; j < 4u; ++j) y[ulong(r) * HC + b * 128u + L + 32u * j] = OUT_T(acc[j]);
