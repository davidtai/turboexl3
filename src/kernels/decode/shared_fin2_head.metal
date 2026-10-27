    if (threadgroup_position_in_grid.y >= uint(RKC)) {
        constexpr uint S_INT = uint(INTER);
        constexpr uint S_NBH = uint(NBH);
        constexpr uint S_NBI = uint(INTER) / 128u;
        const uint s_b = threadgroup_position_in_grid.x;
        const uint s_row = threadgroup_position_in_grid.y - uint(RKC);
        const uint s_L = thread_index_in_simdgroup;
        float s_xg = 0.0f, s_xu = 0.0f;
        if (SDEF != 0) for (uint q = 0; q < S_NBH; ++q) s_xg += sxsg[s_row * S_NBH + q];
        if (SDEF != 0) for (uint q = 0; q < S_NBH; ++q) s_xu += sxsu[s_row * S_NBH + q];
        float g[4], u[4];
        turboexl3_sfinish2<uint(SKSG), uint(SRS), S_INT, uint(SDEF), uint(SLM)>(szg, sh_gate_svh, s_xg, s_row, s_b, s_L, g);
        turboexl3_sfinish2<uint(SKSG), uint(SRS), S_INT, uint(SDEF), uint(SLM)>(szu, sh_up_svh, s_xu, s_row, s_b, s_L, u);
        float v[4];
        for (uint q = 0; q < 4u; ++q) {
            const OUT_T gt = OUT_T(g[q]);
            const OUT_T ut = OUT_T(u[q]);
            const OUT_T sgt = sigtab[as_type<ushort>(gt)];
            const OUT_T act = gt * sgt;
            const OUT_T h = act * ut;
            v[q] = float(half(float(h)));
        }
        const float s_sum = turboexl3_sprep(v, sh_down_suh, s_b, s_L, (device half*)(sxpd + ulong(s_row) * ulong(S_INT) + s_b * 128u));
        if (s_L == 0u) sxsd[s_row * S_NBI + s_b] = s_sum;
        return;
    }
