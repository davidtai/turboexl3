    constexpr uint HC = uint(HID);
    constexpr uint TOPKC = uint(TOPK);
    constexpr uint BPTC = uint(BPT);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint b = threadgroup_position_in_grid.x * BPTC + simdgroup_index_in_threadgroup;
    const uint p = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    const uint src = order[p];
    const uint r = src / TOPKC;
    const uint e = uint(inds[src]);
    if (b == 0u && L == 0u) inv[src] = p;
    float vg[4], vu[4];
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const float xv = float(half(float(x[ulong(r) * HC + i])));
        vg[j] = xv * float(suh_g[ulong(e) * HC + i]);
        vu[j] = xv * float(suh_u[ulong(e) * HC + i]);
    }
    turboexl3_h128x4(vg, L);
    turboexl3_h128x4(vu, L);
    float sg = 0.0f, su = 0.0f;
    const uint lo = (ALAY == 1) ? turboexl3_alay(L) : ((ALAY == 2) ? turboexl3_alay16(L) : L);
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + 32u * j + lo;
        const half hg = half(vg[j] * HAD_SCALE);
        const half hu = half(vu[j] * HAD_SCALE);
        xg[ulong(p) * HC + i] = hg;
        xu[ulong(p) * HC + i] = hu;
        sg += float(hg);
        su += float(hu);
    }
    sg = simd_sum(sg);
    su = simd_sum(su);
    if (L == 0u) { xsg[ulong(p) * (HC / 128u) + b] = sg; xsu[ulong(p) * (HC / 128u) + b] = su; }
