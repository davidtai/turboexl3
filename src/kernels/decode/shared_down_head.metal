    threadgroup float s_red[4][uint(SRS) * 16u * uint(BT)];
    if (threadgroup_position_in_grid.y >= uint(RKC)) {
        constexpr uint S_WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
        constexpr uint S_KS = uint(SKSD);
        constexpr uint S_RS = uint(SRS);
        constexpr uint S_BW = 16u * uint(BT);
        constexpr uint S_PER = (uint(INTER) / 16u) / (S_KS * 4u);
        constexpr uint S_SP = uint(SSPTD);
        const uint s_g = (threadgroup_position_in_grid.y - uint(RKC)) * uint(SZ) + threadgroup_position_in_grid.z;
        if (s_g * S_SP >= S_KS) return;
        const uint s_blk = threadgroup_position_in_grid.x;
        const uint s_sg = simdgroup_index_in_threadgroup;
        const uint s_L = thread_index_in_simdgroup;
        const uint s_tid = thread_index_in_threadgroup;
        for (uint s_q = 0; s_q < S_SP; ++s_q) {
            const uint s_ksl = s_g * S_SP + s_q;
            if (s_q != 0u) threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous slice's plane stores read s_red
            const uint s_tk0 = (s_ksl * 4u + s_sg) * S_PER;
            turboexl3_sdgemv<uint(SND), S_WMASK, int(CODEBOOK), uint(HID) / 16u, uint(BT), S_RS, uint(SDEF), S_PER, uint(INTER)>((const device uint*)(sh_down_t), sxpd + s_tk0 * 16u + 4u * (s_L & 3u), s_tk0, s_blk, s_L, s_red[s_sg]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = s_tid; i < S_RS * S_BW; i += 128u) {
                const uint r = i / S_BW;
                szd[(ulong(s_ksl) * ulong(S_RS) + r) * ulong(uint(HID)) + s_blk * S_BW + (i - r * S_BW)] = ((s_red[0][i] + s_red[1][i]) + s_red[2][i]) + s_red[3][i];
            }
        }
        return;
    }
