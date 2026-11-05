    constexpr uint DIMC = uint(DIM);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint b = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    float xh[4];
    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; xh[j] = float(half(float(x[row * DIMC + i]))); }
    for (uint m = 0; m < 3u; ++m) {
        const device half* suh = (m == 0u) ? suh0 : ((m == 1u) ? suh1 : suh2);
        device half* xp = (m == 0u) ? xp0 : ((m == 1u) ? xp1 : xp2);
        float v[4];
        for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = xh[j] * float(suh[i]); }
        for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
        { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
        for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; xp[row * DIMC + i] = half(v[j] * HAD_SCALE); }
    }
