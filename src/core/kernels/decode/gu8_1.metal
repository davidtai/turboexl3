    constexpr uint N = uint(NHW);
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr uint HIDC = uint(HID);
    constexpr uint INTC = uint(INTER);
    constexpr uint KSC = uint(KS);
    constexpr uint SPTC = uint(SPT);
    constexpr uint BTC = uint(BT);
    constexpr uint TOPKC = uint(TOPK);
    constexpr uint IN_TILES = HIDC / 16u;
    constexpr uint OUT_TILES = INTC / 16u;
    constexpr uint NBHC = HIDC / 128u;
    constexpr uint PER = IN_TILES / (KSC * 2u);           // input tiles per half slice
    constexpr uint NBT = (SPTC * 2u * PER * 16u) / 128u;  // the threadgroup span's 128-blocks
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint blk = threadgroup_position_in_grid.x;
