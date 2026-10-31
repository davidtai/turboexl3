    constexpr uint N = uint(NHW);
    constexpr uint NW = N / 2u;
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr uint GC = uint(GROUPS);
    constexpr uint SIN = uint(IN);                    // a slice's inputs
    constexpr uint INC = GC * SIN;                    // a row of xp: its G slices side by side (the input loads' row stride)
    constexpr uint OUTC = uint(OUT);
    constexpr uint IN_TILES = SIN / 16u;
    constexpr uint OUT_TILES = OUTC / 16u;
    constexpr uint KSC = uint(KS);
    constexpr uint BTC = uint(BT);
    constexpr uint RBC = uint(RB);
    constexpr uint ROWSC = uint(ROWS);
    const uint blk = threadgroup_position_in_grid.x;
    const uint g = threadgroup_position_in_grid.y % GC;            // the slice
    const uint r0 = (threadgroup_position_in_grid.y / GC) * RBC;   // the first row of its row chunk
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
    const device uint* wbase = (const device uint*)(w) + ulong(g) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);   // slice g's trellis
    const device half* xbase = xp + g * SIN + 4u * (L & 3u);   // lane order: the quad's four rows of each tile, of slice g's inputs
    for (uint tk = tk0; tk < tk0 + PER; ++tk) {
