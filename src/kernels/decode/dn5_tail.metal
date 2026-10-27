        for (uint q = 0; q < 4u; ++q) {
            const uint o = hb * 128u + L + 32u * q;
            wv[ulong(mem[j]) * ulong(HIDC) + o] = half(v[q] * HAD_SCALE * float(down_svh[e * HIDC + o]));
        }
    }
