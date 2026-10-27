    constexpr uint DIMC = uint(DIM);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint b = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    const uint r1 = uint(R1), r2 = uint(R2), r3 = uint(R3);
    const device half* z = (row >= r3) ? zp3 : ((row >= r2) ? zp2 : ((row >= r1) ? zp1 : zp0));
    const uint pr = row - ((row >= r3) ? r3 : ((row >= r2) ? r2 : ((row >= r1) ? r1 : 0u)));
    float v[4];
    for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(z[pr * DIMC + b * 128u + L + 32u * j])));
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    for (uint j = 0; j < 4u; ++j) { const uint o = b * 128u + L + 32u * j; y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
