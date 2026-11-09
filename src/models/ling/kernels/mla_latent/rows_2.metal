            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // C. each row's accumulators rescaled, then its P [8 rows, CK keys] times the latents [CK keys, 128 columns]: a
        // latent word read once for the rows, each row's tile with a key the row does not see read as 0
        {
            for (int i = 0; i < RPT; ++i) {
                if (live[i]) {
                    const float a = ash_r[i * 32 + int(mt) * 8 + int(fm)];
                    for (ushort c = 0; c < 16u; ++c) {
                        o[i][c].thread_elements()[0] *= a;
                        o[i][c].thread_elements()[1] *= a;
                    }
                }
            }
            for (int k0 = 0; k0 < CK; k0 += 8) {
                simdgroup_float8x8 pw[RPT];
                for (int i = 0; i < RPT; ++i) {
                    if (live[i]) {
                        threadgroup float* sr = ssh_r + i * (32 * CK);
                        pw[i].thread_elements()[0] = sr[(int(mt) * 8 + int(fm)) * CK + k0 + int(fn)];
                        pw[i].thread_elements()[1] = sr[(int(mt) * 8 + int(fm)) * CK + k0 + int(fn) + 1];
                    }
                }
                const int t = kb + k0 + int(fm);
                const bool ut = t < k_top;
                const device CT* vr = lat + ulong(ut ? t : k_lo) * lt + ulong(int(cs) * 128 + int(fn)) * ld;
                for (ushort c = 0; c < 16u; ++c) {
                    const float x0 = ut ? float(vr[ulong(c * 8u) * ld]) : 0.0f;
                    const float x1 = ut ? float(vr[ulong(c * 8u + 1u) * ld]) : 0.0f;
                    for (int i = 0; i < RPT; ++i) {
                        if (live[i]) {
                            simdgroup_float8x8 vv;
                            vv.thread_elements()[0] = t < k_hi[i] ? x0 : 0.0f;
                            vv.thread_elements()[1] = t < k_hi[i] ? x1 : 0.0f;
                            simdgroup_multiply_accumulate(o[i][c], pw[i], vv, o[i][c]);
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // each row's partials of the group: its 512 sums (this lane's row, 128 columns a simdgroup), then its max and sum
    {
        const int r = int(sgi) * 2 + int(lane / 16u);
        for (int i = 0; i < RPT; ++i) {
            if (i < nr) {
                const int s = s0 + i;
                const ulong slot = (ulong(s) * ulong(H) + ulong(head)) * ulong(NJ) + ulong(j);
                device float* por = po + slot * ulong(DL) + ulong(int(cs) * 128 + int(fn));
                for (ushort c = 0; c < 16u; ++c) {
                    por[c * 8u] = o[i][c].thread_elements()[0];
                    por[c * 8u + 1u] = o[i][c].thread_elements()[1];
                }
                if ((lane % 16u) == 0u) {
                    const ulong sr = (ulong(s) * ulong(H) + ulong(hb + r)) * ulong(NJ) + ulong(j);
                    pm[sr] = msh_r[i * 32 + r];
                    pl[sr] = lsh_r[i * 32 + r];
                }
            }
        }
    }
