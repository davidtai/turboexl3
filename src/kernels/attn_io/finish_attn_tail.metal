    const uint c0 = sg * 128u + L;
    if (is_g) {
        for (uint j = 0; j < 4u; ++j) og[(ulong(row) * uint(HQ) + h) * HDC + c0 + 32u * j] = y[j];
    } else if (is_v) {
        for (uint j = 0; j < 4u; ++j) ov[(ulong(h) * S + row) * HDC + c0 + 32u * j] = y[j];
    } else {
        for (uint j = 0; j < 4u; ++j) ys[c0 + 32u * j] = y[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint base = tid * 4u;
        float acc = 0.0f;
        for (uint i = 0; i < 4u; ++i) {
            float xv = float(ys[base + i]);
            acc += xv * xv;
        }
        acc = simd_sum(acc);
        if (sg == 0u) {
            local_sums[L] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (L == 0u) {
            local_sums[sg] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            acc = simd_sum(local_sums[L]);
            if (L == 0u) {
                local_inv[0] = metal::precise::rsqrt(acc / float(HDC) + float(eps));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = local_inv[0];
        const device OUT_T* w = is_k ? kw : qw;
        for (uint i = 0; i < 4u; ++i) {
            nrm[base + i] = OUT_T(w[base + i] * OUT_T(float(ys[base + i]) * inv));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const device float* arow = angles + ulong(row) * uint(RD);
        device OUT_T* dst = is_k ? (ok + (ulong(h) * S + row) * HDC) : (oq + (ulong(h) * S + row) * HDC);
        for (uint i = 0; i < 4u; ++i) {
            const uint j = base + i;
            OUT_T r;
            if (j >= uint(RD)) {
                r = nrm[j];
            } else if (j < half_rd) {
                r = OUT_T(float(nrm[j]) * arow[j] - float(nrm[j + half_rd]) * arow[half_rd + j]);
            } else {
                r = OUT_T(float(nrm[j - half_rd]) * arow[j] + float(nrm[j]) * arow[j - half_rd]);
            }
            dst[j] = r;
        }
    }
