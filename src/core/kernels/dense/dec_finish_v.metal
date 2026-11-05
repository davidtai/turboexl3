        float v[4];
        for (uint j = 0; j < 4u; ++j) {
            const uint o = ob * 128u + L + 32u * j;
            float acc = zc[ulong(row) * ulong(OUTC) + o];
            for (uint k = 1u; k < KSC; ++k) acc += zc[(ulong(k) * ulong(ROWSC) + row) * ulong(OUTC) + o];
            if (DEFER != 0) acc = fma(acc, MUL1_A, MUL1_C * xsum);
            v[j] = float(half(acc));
        }
        for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
        { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
