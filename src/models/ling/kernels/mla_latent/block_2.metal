            threadgroup_barrier(mem_flags::mem_threadgroup);
            // C. the rows' accumulators rescaled, then P [32 rows, CK keys] times the latents [CK keys, 32 columns]: this
            // simdgroup's 4 x 4 output tiles, a weight word for 4 tiles and a latent word for 4
            {
                float a[4];
                for (ushort m = 0; m < 4u; ++m) a[m] = ash[int(m) * 8 + int(fm)];
                for (ushort m = 0; m < 4u; ++m) {
                    for (ushort c = 0; c < 4u; ++c) {
                        o[m * 4u + c].thread_elements()[0] *= a[m];
                        o[m * 4u + c].thread_elements()[1] *= a[m];
                    }
                }
                for (int k0 = 0; k0 < CK; k0 += 8) {
                    simdgroup_float8x8 pw[4];
                    for (ushort m = 0; m < 4u; ++m) {
                        pw[m].thread_elements()[0] = ssh[(int(m) * 8 + int(fm)) * CK + k0 + int(fn)];
                        pw[m].thread_elements()[1] = ssh[(int(m) * 8 + int(fm)) * CK + k0 + int(fn) + 1];
                    }
                    const int t = kb + k0 + int(fm);
                    const bool vt = t < k_hi;
                    const device CT* vr = lat + ulong(vt ? t : k_lo) * lt + ulong(int(sgi) * 32 + int(fn)) * ld;
                    simdgroup_float8x8 vv[4];
                    for (ushort c = 0; c < 4u; ++c) {
                        vv[c].thread_elements()[0] = vt ? float(vr[ulong(c * 8u) * ld]) : 0.0f;
                        vv[c].thread_elements()[1] = vt ? float(vr[ulong(c * 8u + 1u) * ld]) : 0.0f;
                    }
                    for (ushort m = 0; m < 4u; ++m) {
                        for (ushort c = 0; c < 4u; ++c) {
                            simdgroup_multiply_accumulate(o[m * 4u + c], pw[m], vv[c], o[m * 4u + c]);
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    // the group's partials of each row: its 512 sums (this lane's rows of the 4 M tiles, 32 columns a simdgroup), then
    // its max and sum
    {
        for (ushort m = 0; m < 4u; ++m) {
            const ulong slot = (ulong(s) * ulong(H) + ulong(hb + int(m) * 8 + int(fm))) * ulong(NJ) + ulong(j);
            device float* por = po + slot * ulong(DL) + ulong(int(sgi) * 32 + int(fn));
            for (ushort c = 0; c < 4u; ++c) {
                por[c * 8u] = o[m * 4u + c].thread_elements()[0];
                por[c * 8u + 1u] = o[m * 4u + c].thread_elements()[1];
            }
        }
        const int r = int(sgi) * 2 + int(lane / 16u);
        if ((lane % 16u) == 0u) {
            const ulong sr = (ulong(s) * ulong(H) + ulong(hb + r)) * ulong(NJ) + ulong(j);
            pm[sr] = msh[r];
            pl[sr] = lsh[r];
        }
    }
