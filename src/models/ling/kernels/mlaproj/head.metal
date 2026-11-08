    // dense's DEC-ONE head (dense/dec_head) for two GEMVs in one launch (an MLA layer's q_a and kv_a, ling/mlaproj.zig):
    // the grid's threadgroups are member 0's (K slice, block), then member 1's, and each threadgroup reads its member's
    // widths, K split, arrivals and simdgroup span as that member's own launch has them, its block and K slice counted in
    // its member (KS0 member 0, KS1 member 1; outputs OUT0, OUT1)
    constexpr uint N = uint(NHW);
    constexpr uint NW = N / 2u;
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr uint INC = uint(IN);
    constexpr uint IN_TILES = INC / 16u;
    constexpr uint BTC = uint(BT);
    constexpr uint RBC = uint(RB);
    constexpr uint ROWSC = uint(ROWS);
    constexpr uint NCHC = (ROWSC + RBC - 1u) / RBC;   // row chunks
    constexpr uint NB = INC / 128u;                   // input 128-blocks
    constexpr uint GC = 8u / BTC;                     // column groups per 128-wide output block
    constexpr float HAD_SCALE = 0.08838834764831845f;
    // each member's threadgroups per K slice, and where member 1's threadgroups start in the grid
    constexpr uint NT0 = uint(OUT0) / 128u * GC;
    constexpr uint NT1 = uint(OUT1) / 128u * GC;
    constexpr uint F1 = NT0 * uint(KS0);
    const uint flat = threadgroup_position_in_grid.x;
    const uint jm = (flat < F1) ? 0u : 1u;
    const uint mflat = flat - ((jm == 0u) ? 0u : F1);
    const uint mnt = (jm == 0u) ? NT0 : NT1;
    const uint OUTC = (jm == 0u) ? uint(OUT0) : uint(OUT1);
    const uint OUT_TILES = OUTC / 16u;
    const uint KSC = (jm == 0u) ? uint(KS0) : uint(KS1);
    const uint ARR = GC * KSC;                        // arrivals per (output block, row chunk)
    const uint PER = IN_TILES / (KSC * 4u);
    const uint blk = mflat % mnt;
    const uint chunk = threadgroup_position_in_grid.y;
    const uint r0 = chunk * RBC;
    const uint ksl = mflat / mnt;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint ob = blk / GC;
    threadgroup float red[4][RBC * 16u * BTC];
    threadgroup uint arrived_last;
