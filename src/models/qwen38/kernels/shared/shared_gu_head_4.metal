        // the next slice's prep writes s_xs only after this slice's GEMV (barrier 2), and its GEMV writes s_red only after
        // every plane store of this one (its barrier 1)
        for (uint s_q = 0; s_q < S_SP; ++s_q) {
            const uint s_ksl = (s_j % S_NG) * S_SP + s_q;
            for (uint r = s_sg; r < S_RS; r += 4u) {
                float v[4];
                for (uint q = 0; q < 4u; ++q) v[q] = float(half(float(xr[ulong(r) * S_HID + s_ksl * 128u + s_L + 32u * q])));
                const float sum = (s_pj == 0u) ? turboexl3_sprep(v, sh_gate_suh, s_ksl, s_L, (threadgroup half*)(s_xs + r * 128u)) : turboexl3_sprep(v, sh_up_suh, s_ksl, s_L, (threadgroup half*)(s_xs + r * 128u));
