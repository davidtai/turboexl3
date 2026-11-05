    if (threadgroup_position_in_grid.y >= uint(RKC)) {
        constexpr uint S_INT = uint(INTER);
        constexpr uint S_NBH = uint(NBH);
        constexpr uint S_NBI = uint(INTER) / 128u;
        const uint s_b = threadgroup_position_in_grid.x;
        const uint s_row = threadgroup_position_in_grid.y - uint(RKC);
        const uint s_L = thread_index_in_simdgroup;
