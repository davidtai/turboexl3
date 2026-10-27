    constexpr uint DIMC = uint(DIM);
    constexpr uint TOPKC = uint(TOPK);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint b = threadgroup_position_in_grid.x;
    const uint rk = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    const uint e = slots[rk];
    const uint row = rk / TOPKC;
    float v[4];
    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(x[row * DIMC + i])) * float(suh[e * DIMC + i]); }
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    // lane order: within each 16-wide tile, position r = 2 q + {0, 1, 8, 9}[m] is stored at 4 q + m, so a GEMV lane
    // (its quad q = L & 3) loads its four rows of a tile as one half4
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const uint r = i & 15u;
        const uint q = (r & 7u) >> 1u;
        const uint m = (r & 1u) | ((r >> 3u) << 1u);
        xp[rk * DIMC + (i & ~15u) + 4u * q + m] = half(v[j] * HAD_SCALE);
    }
