#if (EXL3_PF_OCC & 1)
    const uint ao0 = uint(min(int(fm), rows - 1)) * uint(KDC);
    const uint ao1 = uint(min(int(fm) + 8, rows - 1)) * uint(KDC);
    const uint ao2 = uint(min(int(fm) + 16, rows - 1)) * uint(KDC);
    const uint ao3 = uint(min(int(fm) + 24, rows - 1)) * uint(KDC);
#define EXL3PF_SA(i, k) (xa + (ao##i + uint(k)))
#else
    const device half* ar0 = xa + ulong(min(int(fm), rows - 1)) * ulong(KDC);
    const device half* ar1 = xa + ulong(min(int(fm) + 8, rows - 1)) * ulong(KDC);
    const device half* ar2 = xa + ulong(min(int(fm) + 16, rows - 1)) * ulong(KDC);
    const device half* ar3 = xa + ulong(min(int(fm) + 24, rows - 1)) * ulong(KDC);
#define EXL3PF_SA(i, k) (ar##i + (k))
#endif
    // H16: an item of at most 16 rows multiplies only its first fragment row (rows 16..31 are past the tile, never stored)
    const bool top = H16 != 0 && rows <= 16;
    NAXTile<float, 2, 2> Dt;
    Dt.clear();
#if (EXL3_PF_OCC & 4)
    for (int st = 0; st < STEPS - 1; ++st) {
#else
    for (int st = 0; st < STEPS; ++st) {
#endif
        NAXTile<half, 2, 2> At;
        NAXTile<half, 2, 2> Bt;
        if (ALAY == 1) {
            exl3pf_afrag2(At.frag_at(0, 0), At.frag_at(0, 1), EXL3PF_SA(0, st * 32), EXL3PF_SA(1, st * 32));
            if (!top) exl3pf_afrag2(At.frag_at(1, 0), At.frag_at(1, 1), EXL3PF_SA(2, st * 32), EXL3PF_SA(3, st * 32));
        }
        STEEL_PRAGMA_UNROLL
        for (int kk = 0; kk < 2; ++kk) {
            const int k0 = st * 32 + kk * 16;
            if (ALAY == 2) {
                exl3pf_afrag4(At.frag_at(0, kk), EXL3PF_SA(0, k0), EXL3PF_SA(1, k0));
                if (!top) exl3pf_afrag4(At.frag_at(1, kk), EXL3PF_SA(2, k0), EXL3PF_SA(3, k0));
            } else if (ALAY != 1) {
                exl3pf_afrag(At.frag_at(0, kk), EXL3PF_SA(0, k0), EXL3PF_SA(1, k0));
                if (!top) exl3pf_afrag(At.frag_at(1, kk), EXL3PF_SA(2, k0), EXL3PF_SA(3, k0));
            }
            STEEL_PRAGMA_UNROLL
            for (int nn = 0; nn < 2; ++nn) {
#if (EXL3_PF_OCC & 1)
                const device uint* tw = wb + (uint(st * 2 + kk) * uint(N16) + uint(nn)) * uint(NW);
#else
                const device uint* tw = wb + (ulong(st * 2 + kk) * ulong(N16) + ulong(nn)) * ulong(NW);
#endif
                exl3pf_bfrag<N, WMASK, CB, int(DEFER), int(SINTR)>(Bt.frag_at(nn, kk), tw, qb, ga, lp);
            }
        }
        if (top) exl3pf_mma_top(Dt, At, Bt);
        else tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<true>{});
    }
#undef EXL3PF_SA
