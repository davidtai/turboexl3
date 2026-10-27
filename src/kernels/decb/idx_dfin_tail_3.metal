            for (uint j = 0; j < 4u; ++j) { const uint o = b * 128u + L + 32u * j; ist[rq][L + 32u * j] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
