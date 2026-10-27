    constexpr uint NBHC = HIDC / 128u;
    constexpr float HAD_SCALE = 0.08838834764831845f;
    constexpr uint GC = 8u / BTC;             // column groups per 128-wide intermediate block
    constexpr uint NBIC = INTC / 128u;
    constexpr uint ARR = GC * (KSC / SPTC);   // arrivals per (row-slot, intermediate block)
    const uint bi = blk / GC;
    threadgroup uint arrived[2];
    device TURBOEXL3_COH float* zgc = (device TURBOEXL3_COH float*)zg;
    device TURBOEXL3_COH float* zuc = (device TURBOEXL3_COH float*)zu;
