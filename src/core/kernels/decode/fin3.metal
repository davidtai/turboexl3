    constexpr uint INTC = uint(INTER);
    constexpr uint KSC = uint(KS);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint blk = threadgroup_position_in_grid.x;
    const uint rk = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint e = uint(slots[rk]);
    threadgroup float xch[128];
    const uint o = blk * 128u + tid;
    float g = 0.0f, u = 0.0f;
    for (uint ks = 0; ks < KSC; ++ks) { const ulong i = (ulong(ks) * ulong(RKC) + rk) * ulong(INTC) + o; g += zg[i]; u += zu[i]; }
    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    if (DEFER != 0) {
        float sgx = 0.0f, sux = 0.0f;
        for (uint q = 0; q < uint(NBH); ++q) { sgx += xsg[ulong(rk) * uint(NBH) + q]; sux += xsu[ulong(rk) * uint(NBH) + q]; }
        g = fma(g, MUL1_A, MUL1_C * sgx);
        u = fma(u, MUL1_A, MUL1_C * sux);
    }
    g = turboexl3_h128(float(half(g)), xch, tid, L);
    u = turboexl3_h128(float(half(u)), xch, tid, L);
    const float gv = float(half(g * HAD_SCALE * float(gate_svh[e * INTC + o])));
    const float uv = float(half(u * HAD_SCALE * float(up_svh[e * INTC + o])));
    const float hv = (gv / (1.0f + metal::exp(-gv))) * uv;
    // the down's prepare: f16 round, down suh, H128, f16 round, stored in lane order within the 16-wide tile
    float d = float(half(hv)) * float(down_suh[e * INTC + o]);
    d = turboexl3_h128(d, xch, tid, L);
    const uint r = o & 15u;
    const uint q = (r & 7u) >> 1u;
    const uint m = (r & 1u) | ((r >> 3u) << 1u);
    const half hh = half(d * HAD_SCALE);
    hd[ulong(rk) * ulong(INTC) + (o & ~15u) + 4u * q + m] = hh;
    // this 128-block's sum of the down's rounded input (the down finish's deferred MUL1 map): 4 simdgroup sums via xch
    const float hsum = simd_sum(float(hh));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (L == 0u) xch[simdgroup_index_in_threadgroup] = hsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) hs[ulong(rk) * ulong(INTC / 128u) + blk] = ((xch[0] + xch[1]) + xch[2]) + xch[3];
