            // text_prep_s on the down's input element o: the activation read as the down's linear reads it
            v[j] = float(half(float(h))) * float(suh2[o]);
        }
        for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
        { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
        float sum = 0.0f;
        for (uint j = 0; j < 4u; ++j) {
            const uint i = pb * 128u + L + 32u * j;
            const uint r = i & 15u;
            const uint q = (r & 7u) >> 1u;
            const uint m = (r & 1u) | ((r >> 3u) << 1u);
            const half hv = half(v[j] * HAD_SCALE);
            xp[row * DIMC + ((i & ~15u) + 4u * q + m)] = hv;
            sum += float(hv);
        }
        sum = simd_sum(sum);
        if (L == 0u) xs[row * DB + pb] = sum;
    }
