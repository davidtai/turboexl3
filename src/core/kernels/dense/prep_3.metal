    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const uint r = i & 15u;
        const uint q = (r & 7u) >> 1u;
        const uint m = (r & 1u) | ((r >> 3u) << 1u);
        const uint dst = (LANE_ORDER != 0) ? ((i & ~15u) + 4u * q + m) : i;
        xp[row * DIMC + dst] = half(v[j] * HAD_SCALE);
    }
