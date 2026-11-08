    const uint blk = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    threadgroup float xch[128];
