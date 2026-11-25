    const uint qb = exl3_gen_qb<N>(Lp);
    const uint ga = exl3_gen_a<N>(Lp);
    const uint lp = exl3_gen_lp<N>(Lp);
    constexpr uint PER = IN_TILES / (KSC * 4u);
    const uint tk0 = (ksl * 4u + sg) * PER;
    const device uint* wbase = (const device uint*)(w);
    // lane order (the prepare's LANE_ORDER 1): slot 4q + j of a 16-block holds k 2q + {0, 1, 8, 9}[j], the k of B^T's slot
    // 4q + j, so a lane's A elements of an input tile are one half4; a chunk's spare rows (ROWS not a multiple of RB)
    // reread the last row, their sums never written
    const device half* xrow = xp + ulong(min(r0 + fm, ROWSC - 1u)) * ulong(INC) + 4u * q;
