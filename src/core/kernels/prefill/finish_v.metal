    constexpr uint HC = uint(HID);
    constexpr uint NBC = HC / 128u;
    constexpr uint TOPKC = uint(TOPK);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint L = thread_index_in_simdgroup;
    const uint w = threadgroup_position_in_grid.x * uint(ASG) + simdgroup_index_in_threadgroup;
    if (w >= (uint(inv_shape[0]) / TOPKC) * NBC) return;
    const uint r = w / NBC;
    const uint b = w - r * NBC;
    const uint i0 = b * 128u + 4u * L;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0; k < TOPKC; ++k) {
        const uint src = r * TOPKC + k;
        const uint p = inv[src];
        const uint e = uint(inds[src]);
        const float wt = float(scores[src]);
        const half4 z4 = *reinterpret_cast<const device half4*>(zd + ulong(p) * HC + i0);
        float v[4] = {float(z4[0]), float(z4[1]), float(z4[2]), float(z4[3])};
        turboexl3_h128v4(v, L);
        const half4 sv4 = *reinterpret_cast<const device half4*>(svh_d + ulong(e) * HC + i0);
        for (uint j = 0; j < 4u; ++j) acc[j] += wt * float(half(v[j] * HAD_SCALE * float(sv4[j])));
    }
    for (uint j = 0; j < 4u; ++j) y[ulong(r) * HC + i0 + j] = OUT_T(acc[j]);
