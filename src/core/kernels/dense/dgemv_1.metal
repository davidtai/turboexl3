    constexpr uint N = uint(NHW);
    constexpr uint NW = N / 2u;
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr uint INC = uint(IN);
    constexpr uint OUTC = uint(OUT);
    constexpr uint IN_TILES = INC / 16u;
    constexpr uint OUT_TILES = OUTC / 16u;
    constexpr uint KSC = uint(KS);
    constexpr uint BTC = uint(BT);
    constexpr uint RBC = uint(RB);
    constexpr uint ROWSC = uint(ROWS);
    const uint blk = threadgroup_position_in_grid.x;
    const uint r0 = threadgroup_position_in_grid.y * RBC;
    const uint ksl = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    threadgroup float red[4][RBC * 16u * BTC];
    const uint qb = exl3_gen_qb<N>(L);
    const uint ga = exl3_gen_a<N>(L);
    const uint lp = exl3_gen_lp<N>(L);
    float a[RBC][BTC][2];
    for (uint r = 0; r < RBC; ++r) for (uint t = 0; t < BTC; ++t) { a[r][t][0] = 0.0f; a[r][t][1] = 0.0f; }
    constexpr uint PER = IN_TILES / (KSC * 4u);
    const uint tk0 = (ksl * 4u + sg) * PER;
    const device uint* wbase = (const device uint*)(w);
    const device half* xbase = xp + 4u * (L & 3u);   // lane order: the quad's four rows of each tile
    for (uint tk = tk0; tk < tk0 + PER; ++tk) {
