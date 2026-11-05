    if (threadgroup_position_in_grid.y >= uint(RKC)) {
        constexpr uint S_DIM = uint(DIM);
        constexpr uint S_NB = S_DIM / 128u;
        const uint s_b = threadgroup_position_in_grid.x;
        const uint s_r = threadgroup_position_in_grid.y - uint(RKC);
        const uint s_L = thread_index_in_simdgroup;
        for (uint s_p = 0; s_p < 2u; ++s_p) {
            float s_v[4];
            for (uint q = 0; q < 4u; ++q) s_v[q] = float(half(float(x[ulong(s_r) * S_DIM + s_b * 128u + s_L + 32u * q])));
            device half* s_dst = ((s_p == 0u) ? sxg : sxu) + ulong(s_r) * S_DIM + s_b * 128u;
            const float s_sum = (s_p == 0u) ? turboexl3_sprep(s_v, sh_gate_suh, s_b, s_L, s_dst) : turboexl3_sprep(s_v, sh_up_suh, s_b, s_L, s_dst);
            if (s_L == 0u) { if (s_p == 0u) sxsg[s_r * S_NB + s_b] = s_sum; else sxsu[s_r * S_NB + s_b] = s_sum; }
        }
        return;
    }
