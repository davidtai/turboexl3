    float sum = 0.0f;
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const uint r = i & 15u;
        const uint q = (r & 7u) >> 1u;
        const uint m = (r & 1u) | ((r >> 3u) << 1u);
        const half hv = half(v[j] * HAD_SCALE);
        xp[row * DIMC + ((LANE_ORDER != 0) ? ((i & ~15u) + 4u * q + m) : i)] = hv;
        sum += float(hv);
    }
    sum = simd_sum(sum);
    if (L == 0u) xs[row * NB + b] = sum;
    }
    // DEC-ONE: the arrival counters of the GEMV launch that follows, one per (output block, row chunk)
    for (uint g = (row * HQN + uint(hq)) * 256u + uint(d); g < uint(NCNT); g += uint(NTH)) cnt[g] = 0u;
}
