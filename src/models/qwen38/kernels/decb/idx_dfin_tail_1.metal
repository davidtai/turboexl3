    threadgroup OUT_T ist[RATIO][128];
    constexpr uint IHD = 128u;
    OUT_T yv[4];
    for (uint j = 0; j < 4u; ++j) { const uint o = b * 128u + L + 32u * j; yv[j] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); y[row * DIMC + o] = yv[j]; }
    for (uint j = 0; j < 4u; ++j) ist[0][L + 32u * j] = yv[j];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (QEPI != 0 && b < uint(NQB)) {
            const threadgroup OUT_T* src = ist[0];
            const device float* arow = angles + ulong(row) * uint(RD);
            device OUT_T* dst = oq + (ulong(b) * ulong(ROWSC) + row) * IHD;
