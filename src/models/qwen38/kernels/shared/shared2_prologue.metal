    if ((uint(SYOFF) != 0u) ? (threadgroup_position_in_grid.y < uint(SYOFF)) : (threadgroup_position_in_grid.y >= uint(RKC))) {
        const uint s_yy = (uint(SYOFF) != 0u) ? threadgroup_position_in_grid.y : (threadgroup_position_in_grid.y - uint(RKC));
        const uint s_id = (threadgroup_position_in_grid.z * uint(SYS) + s_yy) * uint(SX) + threadgroup_position_in_grid.x;
