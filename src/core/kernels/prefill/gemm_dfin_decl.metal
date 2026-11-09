    // the down's finish as this GEMM's epilogue (`Options.dfin`): the finish's constants (HC its width, the GEMM's N), DFR
    // tile rows a staging round and a simdgroup's even column block rounded to f16 as the store rounds it (dfk)
    constexpr uint HC = uint(ND);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    constexpr int DFRC = int(DFR);
    NAXTile<half, 2, 2> dfk;
