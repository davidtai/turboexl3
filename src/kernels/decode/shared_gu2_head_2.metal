        constexpr uint S_WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
        constexpr uint S_N = uint(SNG);
        constexpr uint S_HID = uint(HID);
        constexpr uint S_RS = uint(SRS);
        constexpr uint S_KS = uint(SKSG);
        constexpr uint S_NBH = S_HID / 128u;
        constexpr uint S_OTG = uint(INTER) / 16u;              // output tiles per projection
        constexpr uint S_OTS = uint(SOTS);                     // output tiles per simdgroup
        constexpr uint S_TGS = (2u * S_OTG) / (4u * S_OTS);    // threadgroups per K slice
        constexpr uint S_PER = (S_HID / 16u) / (S_KS * 4u);    // input tiles per part (dgemv's PER)
        constexpr uint S_NK = 4u * S_PER;                      // input tiles per slice: one 128-block
        constexpr uint S_DW = ((uint(SPRE) / S_PER) < 3u) ? (uint(SPRE) / S_PER) : 3u;   // SPRE input tiles, whole parts, <= 3 ahead
        constexpr uint S_DC = 16u / (S_PER * S_OTS);                                     // <= 16 tiles of raw words in flight
        constexpr uint S_D = (S_DW < S_DC) ? S_DW : S_DC;                                // parts in flight ahead of the decode
        if (s_id >= S_KS * S_TGS) return;
        const uint s_ksl = s_id / S_TGS;
        const uint s_g = s_id - s_ksl * S_TGS;
        const uint s_sg = simdgroup_index_in_threadgroup;
        const uint s_L = thread_index_in_simdgroup;
        const uint s_gt0 = (s_g * 4u + s_sg) * S_OTS;          // first output tile, gate's 0..OTG then up's
        const uint s_pj = s_gt0 / S_OTG;
        const uint s_ot0 = s_gt0 - s_pj * S_OTG;
        const uint s_pj0 = (s_g * 4u * S_OTS) / S_OTG;          // the threadgroup's projections
        const uint s_npj = ((s_g * 4u + 3u) * S_OTS) / S_OTG - s_pj0 + 1u;
        const device uint* s_wb = (const device uint*)((s_pj == 0u) ? sh_gate_t : sh_up_t);
        const uint s_tk0 = s_ksl * S_NK;
        Exl3Raw s_q[S_D + 1u][S_PER][S_OTS];   // the parts in flight ahead of the decode, s_q[0] the next one
        for (uint s_p = 0; s_p < S_D; ++s_p) turboexl3_spload<S_N, S_OTG, S_PER, S_OTS>(s_wb, s_tk0 + s_p * S_PER, s_ot0, s_L, s_q[s_p]);
        for (uint s_wi = s_sg; s_wi < s_npj * S_RS; s_wi += 4u) {
            const uint s_pw = s_pj0 + s_wi / S_RS;
            const uint s_r = s_wi - (s_wi / S_RS) * S_RS;
            float s_v[4];
            for (uint q = 0; q < 4u; ++q) s_v[q] = float(half(float(xr[ulong(s_r) * S_HID + s_ksl * 128u + s_L + 32u * q])));
            threadgroup half* s_dst = s_xs2 + s_wi * 128u;
            const float s_sum = (s_pw == 0u) ? turboexl3_sprep(s_v, sh_gate_suh, s_ksl, s_L, s_dst) : turboexl3_sprep(s_v, sh_up_suh, s_ksl, s_L, s_dst);
            if (s_L == 0u && s_g == (s_pw * S_OTG) / (4u * S_OTS)) { if (s_pw == 0u) sxsg[s_r * S_NBH + s_ksl] = s_sum; else sxsu[s_r * S_NBH + s_ksl] = s_sum; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup half* s_x0 = s_xs2 + (s_pj - s_pj0) * S_RS * 128u + 4u * (s_L & 3u);
        float s_s[S_RS][S_OTS][2];
        for (uint j = 0; j < S_RS; ++j) for (uint t = 0; t < S_OTS; ++t) { s_s[j][t][0] = 0.0f; s_s[j][t][1] = 0.0f; }
        // the slice's four parts in order, each issuing the part S_D ahead before it decodes, then the queue moves up one
        for (uint s_p = 0; s_p < 4u; ++s_p) {
            if (s_p + S_D < 4u) turboexl3_spload<S_N, S_OTG, S_PER, S_OTS>(s_wb, s_tk0 + (s_p + S_D) * S_PER, s_ot0, s_L, s_q[S_D]);
            turboexl3_spart<S_N, S_WMASK, int(CODEBOOK), S_OTG, S_RS, S_OTS, S_PER, uint(SDEF), 128u>(s_q[0], s_wb, s_tk0 + s_p * S_PER, s_ot0, s_x0 + s_p * S_PER * 16u, s_L, s_p == 0u, s_s);
            for (uint s_j = 0; s_j < S_D; ++s_j) for (uint i = 0; i < S_PER; ++i) for (uint t = 0; t < S_OTS; ++t) s_q[s_j][i][t] = s_q[s_j + 1u][i][t];
        }
        turboexl3_sstore<uint(SLM), S_RS, S_OTS, uint(INTER)>((s_pj == 0u) ? szg : szu, s_s, s_ksl, s_ot0, s_L);
        return;
    }
