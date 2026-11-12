    for (uint tk = tk0; tk < tk0 + PER; ++tk, g_tp += OUT_TILES * NW, g_xb += 16u) {
        float xr[RBC][4];
        for (uint r = 0; r < RBC; ++r) {
            const uint row = min(r0 + r, ROWSC - 1u);
