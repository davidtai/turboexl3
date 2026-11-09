        threadgroup_barrier(mem_flags::mem_threadgroup);
        // C. the rows' accumulators rescaled, then P [8 rows, CK keys] times the latents [CK keys, 128 columns], 8 keys at
        // a time staged by the threadgroup (key e, column tid: the cache word converted to f32, or 0 past k_hi)
        {
            const float a = ash[int(mt) * 8 + int(fm)];
            for (ushort i = 0; i < 16u; ++i) {
                o[i].thread_elements()[0] *= a;
                o[i].thread_elements()[1] *= a;
            }
            for (int k0 = 0; k0 < CK; k0 += 8) {
                for (int e = 0; e < 8; ++e) {
                    const int t = kb + k0 + e;
                    const bool v = t < k_hi;
                    stg[e * (DL + 4) + int(tid)] = v ? float(lat[ulong(v ? t : k_lo) * lt + ulong(tid) * ld]) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_float8x8 pw;
                pw.thread_elements()[0] = ssh[(int(mt) * 8 + int(fm)) * CK + k0 + int(fn)];
                pw.thread_elements()[1] = ssh[(int(mt) * 8 + int(fm)) * CK + k0 + int(fn) + 1];
                threadgroup float* vq = stg + int(fm) * (DL + 4) + int(cs) * 128 + int(fn);
                for (ushort i = 0; i < 16u; ++i) {
                    simdgroup_float8x8 vv;
                    vv.thread_elements()[0] = vq[i * 8u];
                    vv.thread_elements()[1] = vq[i * 8u + 1u];
                    simdgroup_multiply_accumulate(o[i], pw, vv, o[i]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
