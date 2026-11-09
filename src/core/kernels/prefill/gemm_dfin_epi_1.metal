            // an even column block: its outputs rounded to f16 as the store rounds them (NAXFrag::store's static_cast), kept
            // until the odd block of its 128-column block is done
            if ((cb & 1) == 0) {
                STEEL_PRAGMA_UNROLL
                for (short f = 0; f < 4; ++f) {
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < 8; ++i) dfk.val_frags[f][i] = static_cast<half>(Dt.val_frags[f][i]);
                }
            }
        }
        if ((cb & 1) != 0) {
            // the 128-column block b of blocks cb - 1 and cb, DFR tile rows a round: the round's simdgroups store their two
            // 32 x 32 halves at their columns of dfz (the even block's at 0..63, the odd's at 64..127), then each of the
            // round's (row, block) items runs the finish's statements (finish_2, finish_4) with the row's halves read from
            // dfz where the finish reads zd; at top-1 the slot at a sorted position is its row of y (r)
            const uint b = uint(n0g + (cb - 1) * BNC) / 128u;
            for (int r0 = 0; r0 < rows; r0 += DFRC) {
                if (sgp > 0 && tm >= r0 && tm < r0 + DFRC) {
                    dfk.template store<half, 128, 1>(dfz + (tm - r0) * 128 + tn);
                    Dt.template store<half, 128, 1>(dfz + (tm - r0) * 128 + 64 + tn);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int it = int(sg); it < DFRC && r0 + it < rows; it += NSGC) {
                    const uint r = order[uint(row0 + r0 + it)];
                    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    {
                        const uint src = r;
