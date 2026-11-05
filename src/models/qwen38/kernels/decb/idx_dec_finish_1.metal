        OUT_T yv[4];
        for (uint j = 0; j < 4u; ++j) { const uint o = ob * 128u + L + 32u * j; yv[j] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); y[row * OUTC + o] = yv[j]; }
        for (uint j = 0; j < 4u; ++j) ist[rr][L + 32u * j] = yv[j];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (QEPI != 0 && ob < uint(NQB)) {
            const threadgroup OUT_T* src = ist[rr];
            const device float* arow = angles + ulong(row) * uint(RD);
            device OUT_T* dst = oq + (ulong(ob) * ulong(ROWSC) + row) * IHD;
