    const int wi = int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup);
    const int tile = wi / (NOPSC * NB);
    const int rem = wi - tile * (NOPSC * NB);
