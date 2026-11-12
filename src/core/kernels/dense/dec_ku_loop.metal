    // ku (`dense.dec_variants`): input tiles in pairs, both tiles' words of the block's BT output tiles loaded before
    // either decodes (2 BT tiles in flight a lane); each tile's FMAs still in tile order, so the same sums
    for (uint tk2 = tk0; tk2 < tk0 + PER; tk2 += 2u) {
        Exl3BitsM<exl3_rmode<N>()> wpre[2][BTC];
        for (uint u = 0; u < 2u; ++u) if (tk2 + u < tk0 + PER) for (uint t = 0; t < BTC; ++t) wpre[u][t] = exl3_tile_load<N, 0>(wbase + (ulong(tk2 + u) * OUT_TILES + blk * BTC + t) * ulong(NW), qb, ga, lp);
        for (uint u = 0; u < 2u; ++u) {
        const uint tk = tk2 + u;
        if (tk >= tk0 + PER) break;
