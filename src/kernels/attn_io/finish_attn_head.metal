    constexpr uint HDC = uint(HD);
    constexpr uint QDIM = 2u * uint(HQ) * HDC;
    constexpr uint KDIM = uint(HK) * HDC;
    constexpr uint half_rd = uint(RD) / 2u;
    constexpr float HAD_SCALE = 0.08838834764831845f;
    threadgroup OUT_T ys[HD];
    threadgroup OUT_T nrm[HD];
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
    const uint u = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint S = threadgroups_per_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const bool on_q = u < 2u * uint(HQ);
    const bool is_g = on_q && u >= uint(HQ);
    const bool is_k = !on_q && u < 2u * uint(HQ) + uint(HK);
    const bool is_v = !on_q && !is_k;
    const uint h = on_q ? (is_g ? u - uint(HQ) : u) : (is_k ? u - 2u * uint(HQ) : u - 2u * uint(HQ) - uint(HK));
    const device half* z = on_q ? zq : (is_k ? zk : zv);
    const device half* svh = on_q ? svq : (is_k ? svk : svv);
    const uint dimc = on_q ? QDIM : KDIM;
    const uint blk = on_q ? (4u * h + (is_g ? 2u : 0u) + sg) : (2u * h + sg);
    float v[4];
    for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(z[row * dimc + blk * 128u + L + 32u * j])));
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    OUT_T y[4];
    for (uint j = 0; j < 4u; ++j) { const uint o = blk * 128u + L + 32u * j; y[j] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
