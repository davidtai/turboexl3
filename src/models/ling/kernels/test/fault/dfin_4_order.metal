        // the planted fault (a red check, never served): tile 0's H128 with its two in-lane stages before its five shuffle
        // stages: the same transform in exact arithmetic, its f32 sums in another order
        if (tile == 0) {
            const float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3];
            v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3;
            for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
        } else turboexl3_h128x4(v, L);
        for (uint j = 0; j < 4u; ++j) {
            const uint i = b * 128u + L + 32u * j;
            acc[j] += w * float(half(v[j] * HAD_SCALE * float(svh_d[ulong(e) * HC + i])));
        }
    }
    for (uint j = 0; j < 4u; ++j) y[ulong(r) * HC + b * 128u + L + 32u * j] = OUT_T(acc[j]);
