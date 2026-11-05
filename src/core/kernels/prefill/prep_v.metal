    constexpr uint HC = uint(HID);
    constexpr uint NBC = HC / 128u;
    constexpr uint TOPKC = uint(TOPK);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint L = thread_index_in_simdgroup;
    const uint w = threadgroup_position_in_grid.x * uint(ASG) + simdgroup_index_in_threadgroup;
    if (w >= uint(order_shape[0]) * NBC) return;
    const uint p = w / NBC;
    const uint b = w - p * NBC;
    const uint src = order[p];
    const uint r = src / TOPKC;
    const uint e = uint(inds[src]);
    if (b == 0u && L == 0u) inv[src] = p;
    const uint i0 = b * 128u + 4u * L;
    typedef metal::remove_cv_t<metal::remove_addrspace_t<metal::remove_reference_t<decltype(x[0])>>> XT;
    const vec<XT, 4> x4 = *reinterpret_cast<const device vec<XT, 4>*>(x + ulong(r) * HC + i0);
    const half4 sg4 = *reinterpret_cast<const device half4*>(suh_g + ulong(e) * HC + i0);
    const half4 su4 = *reinterpret_cast<const device half4*>(suh_u + ulong(e) * HC + i0);
    float vg[4], vu[4];
    for (uint j = 0; j < 4u; ++j) {
        const float xv = float(half(float(x4[j])));
        vg[j] = xv * float(sg4[j]);
        vu[j] = xv * float(su4[j]);
    }
    turboexl3_h128v4(vg, L);
    turboexl3_h128v4(vu, L);
    const half4 hg = half4(half(vg[0] * HAD_SCALE), half(vg[1] * HAD_SCALE), half(vg[2] * HAD_SCALE), half(vg[3] * HAD_SCALE));
    const half4 hu = half4(half(vu[0] * HAD_SCALE), half(vu[1] * HAD_SCALE), half(vu[2] * HAD_SCALE), half(vu[3] * HAD_SCALE));
    *reinterpret_cast<device half4*>(xg + ulong(p) * HC + i0) = hg;
    *reinterpret_cast<device half4*>(xu + ulong(p) * HC + i0) = hu;
    const float sg = turboexl3_sum_x4<int(XSO)>(hg, L);
    const float su = turboexl3_sum_x4<int(XSO)>(hu, L);
    if (L == 0u) { xsg[ulong(p) * NBC + b] = sg; xsu[ulong(p) * NBC + b] = su; }
