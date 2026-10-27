    const int wi = (int(threadgroup_position_in_grid.x) - GBIGM) * SGS + int(simdgroup_index_in_threadgroup);
    const int stile = wi / (NOPSC * NB);
    const int tile = TBIGM + stile;
    if (tile >= int(tiles_shape[0])) return;
    const int rem = wi - stile * (NOPSC * NB);
