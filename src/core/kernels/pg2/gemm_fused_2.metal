    // turboexl3's dense finish per (row, 128-column block) of the tile: its statements, z's halves read from zt
    const uint DIMC = uint(N);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    constexpr uint NB = uint(BN) / 128u;
    for (uint it = simd_group_id; it < uint(BM) * NB; it += uint(WM * WN)) {
      const uint rt = it / NB;
      const uint bt = it % NB;
      // a row past M (the last row tile's): no output (the whole simdgroup skips it)
      if (c_row + int(rt) >= M) continue;
      const uint b = uint(c_col) / 128u + bt;
      const uint row = uint(c_row) + rt;
    const uint L = thread_index_in_simdgroup;
    float v[4];
    for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(zt[rt * uint(BN) + bt * 128u + L + 32u * j])));
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    for (uint j = 0; j < 4u; ++j) { const uint o = b * 128u + L + 32u * j; y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
    }
