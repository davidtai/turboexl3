    constexpr uint IC = uint(INTER);
    constexpr uint BPTC = uint(BPT);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint b = threadgroup_position_in_grid.x * BPTC + simdgroup_index_in_threadgroup;
    const uint p = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    const uint e = uint(inds[order[p]]);
