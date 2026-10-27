    constexpr uint DIMC = uint(DIM);
    constexpr uint TOPKC = uint(TOPK);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint b = threadgroup_position_in_grid.x;
    const uint rk = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    const uint e = uint(slots[rk]);
    const uint row = rk / TOPKC;
