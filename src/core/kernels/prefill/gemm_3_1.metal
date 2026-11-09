    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const int g = int(threadgroup_position_in_grid.x);
    const int tile = g / (NOPSC * NTG);
    const int rem = g - tile * (NOPSC * NTG);
    const int op = rem / NTG;
    const int n0g = (rem - op * NTG) * (NCBC * BNC);
