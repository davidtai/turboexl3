    const uint q = L & 3u;
    const uint c = L >> 2u;
    const uint qb = (INTR != 0) ? exl3_int_qb<N>(L) : exl3_gen_qb<N>(L);
    const uint ga = (INTR != 0) ? exl3_int_a<N>(L) : exl3_gen_a<N>(L);
    const uint lp = (INTR != 0) ? exl3_int_b<N>(L) : exl3_gen_lp<N>(L);
    const uint lnx = (((L >> 4u) & 1u) << 4u) | (((L >> 1u) & 1u) << 3u) | (((L >> 3u) & 1u) << 2u) | (((L >> 2u) & 1u) << 1u) | (L & 1u);
    Arows arw;
    {
        const uint fmx = ((L >> 2u) & 4u) | ((L >> 1u) & 3u);
        const uint qnx = ((L >> 2u) & 2u) | (L & 1u);
        const device half* xs = xa + ((ALAY == 1) ? 8u : ((ALAY == 2) ? 4u : 2u)) * qnx;
        const int lastr = max(sgp - 1, 0);
        arw.r[0] = xs + ulong(min(int(fmx), lastr)) * ulong(KDC);
        arw.r[1] = xs + ulong(min(int(fmx) + 8, lastr)) * ulong(KDC);
        arw.r[2] = xs + ulong(min(int(fmx) + 16, lastr)) * ulong(KDC);
        arw.r[3] = xs + ulong(min(int(fmx) + 24, lastr)) * ulong(KDC);
    }
    int boff[JOBS];
    ulong woff[JOBS];
    for (int jj = 0; jj < JOBS; ++jj) {
        const int j = max(dsg, 0) + DEC * jj;
        const int kt = j >> 2;
        const int ntl = j & 3;
        boff[jj] = (FRAGB != 0) ? (ntl * 4 + kt) * 256 : (ntl * 16 + int(c)) * LDB + kt * 16 + 2 * int(q);
        woff[jj] = (ulong(kt) * ulong(N16) + ulong(ntl)) * ulong(N / 2u);
    }
    const bool full = sgp == SM;
    NAXTile<float, 2, 2> Dt;
    Dt.clear();
    if (DBUF != 0) {
        STEEL_PRAGMA_UNROLL
        for (int h = 0; h < SUBS; ++h) if (producer) exl3pf_stage<N, WMASK, CB, JOBS, N16, LDB, int(PROBE), int(FRAGB), int(INTR), int(DEFER)>(wb, h, woff, boff, Bs + h * BSZH, qb, ga, lp, lnx);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int cb = 0; cb < NCBC; ++cb) {
        if (DBUF != 0) {
            for (int s = 0; s < STAGES; ++s) {
                // f: the stage's place in the threadgroup's sequence (its buffer); the next stage is s + 1 of this block or 0
                // of the next
                const int f = cb * STAGES + s;
                const bool last = s + 1 == STAGES;
                const bool more = !last || cb + 1 < NCBC;
                const device uint* wn = wb + ulong(4 * (last ? cb + 1 : cb)) * ulong(N / 2u);
                const int sn = last ? 0 : s + 1;
                STEEL_PRAGMA_UNROLL
                for (int h = 0; h < SUBS; ++h) {
                    const threadgroup half* Bc = Bs + (f & 1) * BSZ + h * BSZH;
                    if (MFIRST != 0 && sgp > 0 && (PROBE & 2) == 0) exl3pf_mma<LDB, KDC, int(PROBE), int(UNR), int(FRAGB), int(ALAY), int(H16)>(Dt, xa, arw, SUBS * s + h, Bc, tn, full, sgp, L);
                    if (producer && more) exl3pf_stage<N, WMASK, CB, JOBS, N16, LDB, int(PROBE), int(FRAGB), int(INTR), int(DEFER)>(wn, SUBS * sn + h, woff, boff, Bs + ((f + 1) & 1) * BSZ + h * BSZH, qb, ga, lp, lnx);
                    if (MFIRST == 0 && sgp > 0 && (PROBE & 2) == 0) exl3pf_mma<LDB, KDC, int(PROBE), int(UNR), int(FRAGB), int(ALAY), int(H16)>(Dt, xa, arw, SUBS * s + h, Bc, tn, full, sgp, L);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            for (int s = 0; s < STAGES; ++s) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                STEEL_PRAGMA_UNROLL
                for (int h = 0; h < SUBS; ++h) exl3pf_stage<N, WMASK, CB, JOBS, N16, LDB, int(PROBE), int(FRAGB), int(INTR), int(DEFER)>(wb + ulong(4 * cb) * ulong(N / 2u), SUBS * s + h, woff, boff, Bs + h * BSZH, qb, ga, lp, lnx);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                STEEL_PRAGMA_UNROLL
                for (int h = 0; h < SUBS; ++h) if (sgp > 0 && (PROBE & 2) == 0) exl3pf_mma<LDB, KDC, int(PROBE), int(UNR), int(FRAGB), int(ALAY), int(H16)>(Dt, xa, arw, SUBS * s + h, Bs + h * BSZH, tn, full, sgp, L);
            }
        }
        if (sgp > 0) {
