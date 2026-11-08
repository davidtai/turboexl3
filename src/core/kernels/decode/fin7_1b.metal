    const uint b = threadgroup_position_in_grid.x;
    const uint rk = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
