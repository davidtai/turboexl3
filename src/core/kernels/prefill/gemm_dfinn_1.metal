    // the fused finish over the experts of DFMIN rows or more in the call (`Options.dfin_min`), one launch writing all of
    // y: the GEMM half's threadgroups first (both tables' tiles, NCB column blocks each; a tile of an expert of fewer rows
    // returns), then the finish half's (an item a simdgroup: row r and 128-block b of y, where r's expert has fewer than
    // DFMIN rows). The weight stage, and with DFR 64 the staging past it, is declared at kernel scope, as the mixed launch
    // declares its stage
    threadgroup uint4 dfn_bs[((DBUF != 0) ? 2 : 1) * (int(BKS) / 64) * 64 * ((FRAGB != 0 && (PROBE & 16) == 0) ? 64 : 72) * 2 / 16 + ((DFR == 64) ? 64 * 128 * 2 / 16 : 0)];
    const int GDF = (int(tiles_shape[0]) + ((T2 != 0) ? int(tiles2_shape[0]) : 0)) * int(NOPS) * (int(ND) / 64 / int(NCB));
    if (int(threadgroup_position_in_grid.x) < GDF) {
