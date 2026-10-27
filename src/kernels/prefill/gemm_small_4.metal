    const device half* ar0 = xa + ulong(min(int(fm), rows - 1)) * ulong(KDC);
    const device half* ar1 = xa + ulong(min(int(fm) + 8, rows - 1)) * ulong(KDC);
    const device half* ar2 = xa + ulong(min(int(fm) + 16, rows - 1)) * ulong(KDC);
    const device half* ar3 = xa + ulong(min(int(fm) + 24, rows - 1)) * ulong(KDC);
    // H16: an item of at most 16 rows multiplies only its first fragment row (rows 16..31 are past the tile, never stored)
    const bool top = H16 != 0 && rows <= 16;
    NAXTile<float, 2, 2> Dt;
    Dt.clear();
    for (int st = 0; st < STEPS; ++st) {
        NAXTile<half, 2, 2> At;
        NAXTile<half, 2, 2> Bt;
        if (ALAY == 1) {
            exl3pf_afrag2(At.frag_at(0, 0), At.frag_at(0, 1), ar0 + st * 32, ar1 + st * 32);
            if (!top) exl3pf_afrag2(At.frag_at(1, 0), At.frag_at(1, 1), ar2 + st * 32, ar3 + st * 32);
        }
        STEEL_PRAGMA_UNROLL
        for (int kk = 0; kk < 2; ++kk) {
            const int k0 = st * 32 + kk * 16;
            if (ALAY == 2) {
                exl3pf_afrag4(At.frag_at(0, kk), ar0 + k0, ar1 + k0);
                if (!top) exl3pf_afrag4(At.frag_at(1, kk), ar2 + k0, ar3 + k0);
            } else if (ALAY != 1) {
                exl3pf_afrag(At.frag_at(0, kk), ar0 + k0, ar1 + k0);
                if (!top) exl3pf_afrag(At.frag_at(1, kk), ar2 + k0, ar3 + k0);
            }
            STEEL_PRAGMA_UNROLL
            for (int nn = 0; nn < 2; ++nn) {
                const device uint* tw = wb + (ulong(st * 2 + kk) * ulong(N16) + ulong(nn)) * ulong(NW);
                exl3pf_bfrag<N, WMASK, CB, int(DEFER), int(SINTR)>(Bt.frag_at(nn, kk), tw, qb, ga, lp);
            }
        }
        if (top) exl3pf_mma_top(Dt, At, Bt);
        else tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<true>{});
    }
