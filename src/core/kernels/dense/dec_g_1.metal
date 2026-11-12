    static_assert(exl3_rmode<N>() == 1, "dense g: a rate whose eight windows span at most 64 bits");
    const uint qb = exl3_gen_qb<N>(L);
    const uint ga = exl3_gen_a<N>(L);
    const uint lp = exl3_gen_lp<N>(L);
    float a[RBC][BTC][2];
    for (uint r = 0; r < RBC; ++r) for (uint t = 0; t < BTC; ++t) { a[r][t][0] = 0.0f; a[r][t][1] = 0.0f; }
    const uint tk0 = (ksl * 4u + sg) * PER;
    // exl3_tile_load's word offsets and shift for this lane (the rate's 64-bit window), the same for every tile
    constexpr uint G_CM7 = ((8u * N) >> 4u) - (N >> 4u);
    const uint g_e = qb * 32u + ga * 16u + lp + G_CM7 + 16u;
    const uint g_last = ((g_e - 1u) >> 5u) % NW;
    const uint g_w1 = (g_last + NW - 1u) % NW;
    const uint g_w2 = (g_w1 + NW - 1u) % NW;
    const uint g_sh = (0u - g_e) & 31u;
    // the slice's first tile of this threadgroup's output tiles, stepped one input tile at a time
    const device uint* g_tp = (const device uint*)(w) + (ulong(tk0) * OUT_TILES + blk * BTC) * ulong(NW);
