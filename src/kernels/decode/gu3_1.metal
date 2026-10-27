    constexpr uint N = uint(NHW);
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr uint HIDC = uint(HID);
    constexpr uint INTC = uint(INTER);
    constexpr uint KSC = uint(KS);
    constexpr uint SPTC = uint(SPT);   // K slices per threadgroup (of the KS the partition fixes)
    constexpr uint BTC = uint(BT);     // output tiles per threadgroup: 2 BT f32 accumulators per lane
    constexpr uint IN_TILES = HIDC / 16u;
    constexpr uint OUT_TILES = INTC / 16u;
    const uint blk = threadgroup_position_in_grid.x;
