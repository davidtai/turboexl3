        constexpr uint S_GC = 8u / uint(BT);      // column groups per 128-wide block of the shared intermediate
        constexpr uint S_INT = uint(INTER);
        constexpr uint S_NBI = S_INT / 128u;
        constexpr uint S_ARR = S_GC * 2u * S_NG;  // arrivals per shared block: its column groups x both projections' slice groups
        const uint s_bi = s_blk / S_GC;
        device TURBOEXL3_COH float* s_xsc = (device TURBOEXL3_COH float*)sxsx;
