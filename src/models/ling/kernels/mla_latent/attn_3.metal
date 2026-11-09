        threadgroup_barrier(mem_flags::mem_threadgroup);
        // C. the rows' accumulators rescaled, then P [8 rows, CK keys] times the latents [CK keys, 128 columns]
        {
            const float a = ash[int(mt) * 8 + int(fm)];
            for (ushort i = 0; i < 16u; ++i) {
                o[i].thread_elements()[0] *= a;
                o[i].thread_elements()[1] *= a;
            }
            for (int k0 = 0; k0 < CK; k0 += 8) {
                simdgroup_float8x8 pw;
                pw.thread_elements()[0] = ssh[(int(mt) * 8 + int(fm)) * CK + k0 + int(fn)];
                pw.thread_elements()[1] = ssh[(int(mt) * 8 + int(fm)) * CK + k0 + int(fn) + 1];
                const int t = kb + k0 + int(fm);
                const bool vt = t < k_hi;
                const device CT* vr = lat + ulong(vt ? t : k_lo) * lt + ulong(int(cs) * 128 + int(fn)) * ld;
                for (ushort i = 0; i < 16u; ++i) {
                    simdgroup_float8x8 vv;
                    vv.thread_elements()[0] = vt ? float(vr[ulong(i * 8u) * ld]) : 0.0f;
                    vv.thread_elements()[1] = vt ? float(vr[ulong(i * 8u + 1u) * ld]) : 0.0f;
                    simdgroup_multiply_accumulate(o[i], pw, vv, o[i]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
