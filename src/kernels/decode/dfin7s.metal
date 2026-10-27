    constexpr uint HIDC = uint(HID);
    constexpr uint TOPKC = uint(TOPK);
    constexpr uint KSC = uint(KS);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint blk = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint k = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    threadgroup float term[TOPKC * 128u];
    threadgroup OUT_T sterm[128];
    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    if (k < TOPKC) {
        const uint rk = row * TOPKC + k;
        const uint e = uint(slots[rk]);
        float v[4];
        for (uint q = 0; q < 4u; ++q) {
            const uint o = blk * 128u + L + 32u * q;
            float t = 0.0f;
            for (uint ks = 0; ks < KSC; ++ks) t += zd[(ulong(ks) * ulong(RKC) + rk) * ulong(HIDC) + o];
            v[q] = t;
        }
        if (DEFER != 0) {
            float hsum = 0.0f;
            for (uint q = 0; q < uint(NBI); ++q) hsum += hsx[ulong(rk) * uint(NBI) + q];
            for (uint q = 0; q < 4u; ++q) v[q] = fma(v[q], MUL1_A, MUL1_C * hsum);
        }
        for (uint q = 0; q < 4u; ++q) v[q] = float(half(v[q]));
        turboexl3_h128r(v, L);
        for (uint q = 0; q < 4u; ++q) {
            const uint o = blk * 128u + L + 32u * q;
            term[k * 128u + L + 32u * q] = float(half(v[q] * HAD_SCALE * float(down_svh[e * HIDC + o])));
        }
    } else {
        float s_xsum = 0.0f;
        if (SDEF != 0) for (uint q = 0; q < uint(NBI); ++q) s_xsum += sxsd[row * uint(NBI) + q];
        float v[4];
        turboexl3_sfinish<uint(SKSD), uint(SRS), HIDC, uint(SDEF)>(szd, sh_down_svh, s_xsum, row, blk, L, v);
        for (uint q = 0; q < 4u; ++q) sterm[L + 32u * q] = OUT_T(v[q]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128u) {
        float acc = 0.0f;
        for (uint j = 0; j < TOPKC; ++j) acc = fma(float(scores[row * TOPKC + j]), term[j * 128u + tid], acc);
        const OUT_T yb = OUT_T(acc);
        const OUT_T sig = sigtab[as_type<ushort>(slogit[row])];
        const OUT_T gated = sig * sterm[tid];
        y[ulong(row) * ulong(HIDC) + blk * 128u + tid] = yb + gated;
    }
