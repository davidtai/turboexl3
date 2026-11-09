    // the down's finish as this GEMM's epilogue (`Options.dfin`): the finish's constants (HC its width, the GEMM's N), DFR
    // tile rows a staging round, the round's f16 outputs (dfz: a 128-column block, 128 halves a row) and a simdgroup's
    // even column block rounded to f16 as the store rounds it (dfk)
    constexpr uint HC = uint(ND);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    constexpr int DFRC = int(DFR);
    threadgroup half dfz[DFRC * 128];
    NAXTile<half, 2, 2> dfk;
