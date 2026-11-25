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
    // the output tiles in pairs (one 16 x 32 x 16 MMA a pair and input tile); a chunk's rows are fragment rows 0..RB-1
    static_assert(BTC % 2u == 0u && RBC >= 1u && RBC <= 8u, "turboexl3_dense_dgemv_mma: BT even, RB 1 to 8");
    constexpr uint NP = BTC / 2u;
    const uint blk = threadgroup_position_in_grid.x;
    const uint r0 = threadgroup_position_in_grid.y * RBC;
    const uint ksl = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    threadgroup float red[4][RBC * 16u * BTC];
    // lane L's NAX fragment slot (BaseNAXFrag::get_coord): rows fm and fm + 8, k slots 4q .. 4q + 3
    const uint fm = ((L >> 2u) & 4u) | ((L >> 1u) & 3u);
    const uint q = ((L >> 2u) & 2u) | (L & 1u);
