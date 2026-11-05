            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const threadgroup half* s_x0 = s_xs + s_sg * 32u + 4u * (s_L & 3u);
            turboexl3_sdgemv<uint(SNG), S_WMASK, int(CODEBOOK), uint(INTER) / 16u, uint(BT), S_RS, uint(SDEF), 2u, 128u>(s_w, s_x0, s_ksl * 8u + s_sg * 2u, s_blk, s_L, s_red[s_sg]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = s_tid; i < S_RS * S_BW; i += 128u) {
                const uint r = i / S_BW;
                s_z[(ulong(s_ksl) * ulong(S_RS) + r) * ulong(uint(INTER)) + s_blk * S_BW + (i - r * S_BW)] = ((s_red[0][i] + s_red[1][i]) + s_red[2][i]) + s_red[3][i];
            }
        }
