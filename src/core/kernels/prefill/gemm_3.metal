    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const int g = int(threadgroup_position_in_grid.x);
    const int tile = g / (NOPSC * NTG);
    const int rem = g - tile * (NOPSC * NTG);
    const int op = rem / NTG;
    const int n0g = (rem - op * NTG) * (NCBC * BNC);
    const int e = tiles[4 * tile];
    const int row0 = tiles[4 * tile + 1];
    const int rows = tiles[4 * tile + 2];
    if (rows == 0) return;
    const bool consumer = int(sg) < NSGC;
    const bool producer = NPR == 0 || !consumer;
    const int dsg = (NPR > 0) ? int(sg) - NSGC : int(sg);
    const int tm = SM * int(sg / 2u);
    const int tn = SN * int(sg % 2u);
    const int sgp = consumer ? min(SM, max(0, rows - tm)) : 0;
    // the expert's trellis from the threadgroup's first output column; column block cb starts 4 cb tiles further
