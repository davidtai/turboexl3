        }
        for (uint t = 0; t < BTC; ++t) {
            const device uint* tile = g_tp + t * NW;
            const uint g_a = tile[g_last], g_b = tile[g_w1], g_c = tile[g_w2];
            Exl3BitsM<1> ww;
            ww.v = as_type<ulong>(uint2(uint(((ulong(g_b) << 32u) | ulong(g_a)) >> g_sh), uint(((ulong(g_c) << 32u) | ulong(g_b)) >> g_sh)));
