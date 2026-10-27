    threadgroup uint4 mixed_bs[((DBUF != 0) ? 2 : 1) * (int(BKS) / 64) * 64 * ((FRAGB != 0 && (PROBE & 16) == 0) ? 64 : 72) * 2 / 16];
    const int TBIGM = int(tiles_shape[0]) - int(TMAXS);
    const int GBIGM = TBIGM * int(NOPS) * (int(ND) / 64 / int(NCB));
    if (int(threadgroup_position_in_grid.x) < GBIGM) {
