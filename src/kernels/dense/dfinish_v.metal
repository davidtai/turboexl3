    constexpr uint DIMC = uint(DIM);
    constexpr uint KSC = uint(KS);
    constexpr uint ROWSC = uint(ROWS);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint b = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    constexpr uint INB = uint(INBLK);
    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    float xsum = 0.0f;
    if (DEFER != 0) for (uint q = 0; q < INB; ++q) xsum += xs[row * INB + q];
    float v[4];
    for (uint j = 0; j < 4u; ++j) {
        const uint o = b * 128u + L + 32u * j;
        float acc = z[ulong(row) * ulong(DIMC) + o];
        for (uint k = 1u; k < KSC; ++k) acc += z[(ulong(k) * ulong(ROWSC) + row) * ulong(DIMC) + o];
        if (DEFER != 0) acc = fma(acc, MUL1_A, MUL1_C * xsum);
        v[j] = float(half(acc));
    }
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
