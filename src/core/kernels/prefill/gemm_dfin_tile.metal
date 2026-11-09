    // the tile: row `tile` of the first table, or (T2) past its rows a row of the second (the 128-row tiles' mixed table,
    // each of whose tiles this body runs as a part-full tile). Each word is read from its own table: MLX binds an input of
    // fewer than 8 words in the constant address space (a one-row table), so the two tables are never one pointer
    const int TA = int(tiles_shape[0]);
    const bool t2t = T2 != 0 && tile >= TA;
    const int ti = 4 * (t2t ? tile - TA : tile);
    const int e = t2t ? int(tiles2[ti]) : int(tiles[ti]);
    const int row0 = t2t ? int(tiles2[ti + 1]) : int(tiles[ti + 1]);
    const int rows = t2t ? int(tiles2[ti + 2]) : int(tiles[ti + 2]);
