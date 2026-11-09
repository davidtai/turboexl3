    } else {
        // the finish half: item fi of the call's rows times hidden's 128-blocks, a simdgroup each, running the served
        // finish's statements (`text_finish_t128`) with (b, r) taken from fi where the served finish takes them from its grid
        constexpr int NSGF = 2 * int(BMR) / 32;
        const int fi = (int(threadgroup_position_in_grid.x) - GDF) * NSGF + int(simdgroup_index_in_threadgroup);
