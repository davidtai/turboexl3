    // the tile: row `tile` of the first table, or (T2) past its rows a row of the second (the 128-row tiles' mixed table,
    // each of whose tiles this body runs as a part-full tile)
    const int TA = int(tiles_shape[0]);
    const device int* tt = (T2 != 0 && tile >= TA) ? tiles2 + 4 * (tile - TA) : tiles + 4 * tile;
    const int e = tt[0];
    const int row0 = tt[1];
    const int rows = tt[2];
