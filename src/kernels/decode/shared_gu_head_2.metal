    if (threadgroup_position_in_grid.y >= uint(RKC)) {
        constexpr uint S_WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
        constexpr uint S_HID = uint(HID);
        constexpr uint S_KS = uint(SKSG);
        constexpr uint S_RS = uint(SRS);
        constexpr uint S_NBH = S_HID / 128u;
        constexpr uint S_BW = 16u * uint(BT);
        constexpr uint S_SP = uint(SSPT);
        constexpr uint S_NG = S_KS / S_SP;
        const uint s_j = (threadgroup_position_in_grid.y - uint(RKC)) * uint(SZ) + threadgroup_position_in_grid.z;
        if (s_j >= 2u * S_NG) return;
        const uint s_pj = s_j / S_NG;
        const uint s_blk = threadgroup_position_in_grid.x;
        const uint s_sg = simdgroup_index_in_threadgroup;
        const uint s_L = thread_index_in_simdgroup;
        const uint s_tid = thread_index_in_threadgroup;
