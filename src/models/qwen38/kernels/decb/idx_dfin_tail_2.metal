    }
    if (POOL != 0 && b == uint(NQB) && row == 0u) {
        for (uint rq = 1u; rq < uint(RATIO) - uint(ln); ++rq) {
            float xsum = 0.0f;
            if (DEFER != 0) for (uint q = 0; q < INB; ++q) xsum += xs[rq * INB + q];
            float v[4];
            for (uint j = 0; j < 4u; ++j) {
                const uint o = b * 128u + L + 32u * j;
                float acc = z[ulong(rq) * ulong(DIMC) + o];
                for (uint k = 1u; k < KSC; ++k) acc += z[(ulong(k) * ulong(ROWSC) + rq) * ulong(DIMC) + o];
                if (DEFER != 0) acc = fma(acc, MUL1_A, MUL1_C * xsum);
                v[j] = float(half(acc));
            }
