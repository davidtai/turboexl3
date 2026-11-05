    constexpr uint N = uint(NHW);
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr uint HIDC = uint(HID);
    constexpr uint INTC = uint(INTER);
    constexpr uint KSC = uint(KS);      // the gate|up planes
    constexpr uint KS2C = uint(KS2);    // the down's K slices
    constexpr uint GM = uint(GMAX);
    constexpr uint NBHC = HIDC / 128u;
    constexpr uint NBIC = INTC / 128u;
    constexpr uint IN_TILES = INTC / 16u;
    constexpr uint OUT_TILES = HIDC / 16u;
    constexpr uint PER2 = IN_TILES / (KS2C * 4u);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint hb = threadgroup_position_in_grid.x;
    const uint rk = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint e = uint(slots[rk]);
    uint mem[4]; uint m;
    if (!turboexl3_group<GM, uint(RKC)>(slots, rk, e, L, mem, m)) return;
    threadgroup half hd[GM * INTC];
    threadgroup float hsb[GM * NBIC];
    threadgroup float fin[GM * 128u];
    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    // 1. the members' mid, a 128-block per simdgroup pass (lane L holds elements L + 32 q)
    for (uint w = sg; w < m * NBIC; w += 8u) {
        const uint j = w / NBIC;
        const uint b = w % NBIC;
        const uint sl = mem[j];
        float g[4]; float u[4];
        for (uint q = 0; q < 4u; ++q) {
            const uint o = b * 128u + L + 32u * q;
            float gg = 0.0f, uu = 0.0f;
            for (uint k = 0; k < KSC; ++k) { const ulong i = (ulong(k) * ulong(RKC) + sl) * ulong(INTC) + o; gg += zg[i]; uu += zu[i]; }
            g[q] = gg; u[q] = uu;
        }
        if (DEFER != 0) {
            float sgx = 0.0f, sux = 0.0f;
            for (uint q = 0; q < NBHC; ++q) { sgx += xsg[sl * NBHC + q]; sux += xsu[sl * NBHC + q]; }
            for (uint q = 0; q < 4u; ++q) { g[q] = fma(g[q], MUL1_A, MUL1_C * sgx); u[q] = fma(u[q], MUL1_A, MUL1_C * sux); }
        }
        for (uint q = 0; q < 4u; ++q) { g[q] = float(half(g[q])); u[q] = float(half(u[q])); }
        turboexl3_h128r(g, L);
        turboexl3_h128r(u, L);
        float d[4];
        for (uint q = 0; q < 4u; ++q) {
            const uint o = b * 128u + L + 32u * q;
            const float gv = float(half(g[q] * HAD_SCALE * float(gate_svh[e * INTC + o])));
            const float uv = float(half(u[q] * HAD_SCALE * float(up_svh[e * INTC + o])));
            const float hv = (gv / (1.0f + metal::exp(-gv))) * uv;
            d[q] = float(half(hv)) * float(down_suh[e * INTC + o]);
        }
        turboexl3_h128r(d, L);
        float s[4];
        for (uint q = 0; q < 4u; ++q) {
            const uint i = L + 32u * q;
            const uint r = i & 15u; const uint qq = (r & 7u) >> 1u; const uint mm = (r & 1u) | ((r >> 3u) << 1u);
            const half hh = half(d[q] * HAD_SCALE);
            hd[j * INTC + b * 128u + (i & ~15u) + 4u * qq + mm] = hh;
            s[q] = simd_sum(float(hh));
        }
        if (L == 0u) hsb[j * NBIC + b] = ((s[0] + s[1]) + s[2]) + s[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2. the down GEMV: simdgroup sg = output tile 8 hb + sg, every member
    const device uint* wbase = (const device uint*)(down_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(N / 2u);
    const uint ot = hb * 8u + sg;
    const threadgroup half* xs0 = hd + 4u * (L & 3u);
    float tot[4][2];
    if (m == 1u) turboexl3_dn5_sg<N, WMASK, CB, OUT_TILES, 1u, uint(DEFER), int(IR), KS2C, PER2>(wbase, xs0, INTC, ot, L, tot);
    else if (GM >= 2u && m == 2u) turboexl3_dn5_sg<N, WMASK, CB, OUT_TILES, 2u, uint(DEFER), int(IR), KS2C, PER2>(wbase, xs0, INTC, ot, L, tot);
    else if (GM >= 3u && m == 3u) turboexl3_dn5_sg<N, WMASK, CB, OUT_TILES, 3u, uint(DEFER), int(IR), KS2C, PER2>(wbase, xs0, INTC, ot, L, tot);
    else if (GM >= 4u) turboexl3_dn5_sg<N, WMASK, CB, OUT_TILES, 4u, uint(DEFER), int(IR), KS2C, PER2>(wbase, xs0, INTC, ot, L, tot);
    if ((L & 3u) == 0u) for (uint j = 0; j < m; ++j) { fin[j * 128u + sg * 16u + (L >> 2u)] = tot[j][0]; fin[j * 128u + sg * 16u + (L >> 2u) + 8u] = tot[j][1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 3. dfin3's per-slot finish up to the score, simdgroup j for member j
    if (sg < m) {
        const uint j = sg;
        float hsum = 0.0f;
        if (DEFER != 0) for (uint q = 0; q < NBIC; ++q) hsum += hsb[j * NBIC + q];
        float v[4];
        for (uint q = 0; q < 4u; ++q) {
            float t = fin[j * 128u + L + 32u * q];
            if (DEFER != 0) t = fma(t, MUL1_A, MUL1_C * hsum);
            v[q] = float(half(t));
        }
        turboexl3_h128r(v, L);
