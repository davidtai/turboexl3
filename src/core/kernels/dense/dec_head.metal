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
    constexpr uint NCHC = (ROWSC + RBC - 1u) / RBC;   // row chunks
    constexpr uint NB = INC / 128u;                   // input 128-blocks
    constexpr uint GC = 8u / BTC;                     // column groups per 128-wide output block
    constexpr uint ARR = GC * KSC;                    // arrivals per (output block, row chunk)
    constexpr uint PER = IN_TILES / (KSC * 4u);
    constexpr uint SPAN = 64u * PER;                  // a K slice's inputs
    constexpr uint SPB = SPAN / 128u;                 // its 128-blocks (the prepare inside takes SPAN % 128 == 0)
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint blk = threadgroup_position_in_grid.x;
    const uint chunk = threadgroup_position_in_grid.y;
    const uint r0 = chunk * RBC;
    const uint ksl = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint ob = blk / GC;
    threadgroup float red[4][RBC * 16u * BTC];
    threadgroup uint arrived_last;
    device TURBOEXL3_COH float* zc = (device TURBOEXL3_COH float*)z;
