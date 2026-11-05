        for (uint j = 0; j < 4u; ++j) { const uint o = ob * 128u + L + 32u * j; y[row * OUTC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
    }
