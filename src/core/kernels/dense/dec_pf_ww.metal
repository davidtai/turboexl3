            // pf: this tile's words, loaded a pass ahead (dec_gemv_arrive_2_1's load); the next tile's take their place
            const auto ww = wc[t];
            if (tk + 1u < tk0 + PER) wc[t] = wn[t];
