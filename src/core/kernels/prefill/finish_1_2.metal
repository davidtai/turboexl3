    const uint b = threadgroup_position_in_grid.x * BPTC + simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.y;
