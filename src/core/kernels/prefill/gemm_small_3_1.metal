    const int op = rem / NB;
    const int n0 = (rem - op * NB) * 32;
    const int rows = tiles[4 * tile + 2];
    if (rows == 0) return;
    const int e = tiles[4 * tile];
    const int row0 = tiles[4 * tile + 1];
