    const uint qb = exl3_gen_qb<N>(L);
    const uint ga = exl3_gen_a<N>(L);
    const uint lp = exl3_gen_lp<N>(L);
    float a[RBC][BTC][2];
    for (uint r = 0; r < RBC; ++r) for (uint t = 0; t < BTC; ++t) { a[r][t][0] = 0.0f; a[r][t][1] = 0.0f; }
    const uint tk0 = (ksl * 4u + sg) * PER;
    const device uint* wbase = (const device uint*)(w);
    const uint tks = ksl * 4u * PER;   // the slice's first input tile
    for (uint tk = tk0; tk < tk0 + PER; ++tk) {
        float xr[RBC][4];
        for (uint r = 0; r < RBC; ++r) {
            const half4 x4 = *reinterpret_cast<const threadgroup half4*>(xt + r * SPAN + (tk - tks) * 16u + 4u * (L & 3u));
            xr[r][0] = float(x4.x); xr[r][1] = float(x4.y); xr[r][2] = float(x4.z); xr[r][3] = float(x4.w);
        }
