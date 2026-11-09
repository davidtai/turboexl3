        // JOINLESS: pair (row, k)'s output row read in place, at its (source, row) in loc
        const ulong lk = 2 * (ulong(row) * ulong(K) + ulong(k));
        const device half* pk = ling_combine_pick_h(loc[lk], s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23);
        const ulong ok = ulong(loc[lk + 1]) * ulong(D) + ulong(col);
