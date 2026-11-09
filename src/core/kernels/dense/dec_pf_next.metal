        // pf: the next input tile's words, loaded before this tile's decode (dec_tile's address at tk + 1); the loop's
        // first pass loads its own first
        Exl3BitsM<exl3_rmode<N>()> wn[BTC];
        for (uint t = 0; t < BTC; ++t) {
            if (tk == tk0) wc[t] = exl3_tile_load<N, 0>(wbase + (ulong(tk) * OUT_TILES + blk * BTC + t) * ulong(NW), qb, ga, lp);
            if (tk + 1u < tk0 + PER) wn[t] = exl3_tile_load<N, 0>(wbase + (ulong(tk + 1u) * OUT_TILES + blk * BTC + t) * ulong(NW), qb, ga, lp);
        }
