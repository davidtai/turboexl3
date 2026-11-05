    constexpr uint DIMC = uint(DIM);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint b = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    float v[4];
    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]); }
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const uint r = i & 15u;
        const uint q = (r & 7u) >> 1u;
        const uint m = (r & 1u) | ((r >> 3u) << 1u);
        const uint dst = (LANE_ORDER != 0) ? ((i & ~15u) + 4u * q + m) : i;
        xp[row * DIMC + dst] = half(v[j] * HAD_SCALE);
    }
