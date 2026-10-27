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
