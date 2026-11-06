        for (uint j = 0; j < 4u; ++j) { const uint o = ob * 128u + L + 32u * j; yc[row * OUTC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svhm[o - mo0 * 16u])))); }
    }
