        float v = 0.0f;
        for (uint ks = 0; ks < KSC; ++ks) v += zd[(ulong(ks) * ulong(RKC) + rk) * ulong(HIDC) + o];
        if (DEFER != 0) {
            float hsum = 0.0f;
            for (uint q = 0; q < uint(NBI); ++q) hsum += hsx[ulong(rk) * uint(NBI) + q];
            v = fma(v, MUL1_A, MUL1_C * hsum);
        }
        v = turboexl3_h128(float(half(v)), xch, tid, L);
        acc = fma(float(scores[rk]), float(half(v * HAD_SCALE * float(down_svh[e * HIDC + o]))), acc);
    }
    y[ulong(row) * ulong(HIDC) + o] = OUT_T(acc);   // the caller's dtype (bf16 served, f32 in the tests)
