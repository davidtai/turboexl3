    constexpr uint N = uint(NHW);
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr uint HIDC = uint(HID);
    constexpr uint INTC = uint(INTER);
    constexpr uint KSC = uint(KS);
    constexpr uint BTC = uint(BT);
    constexpr uint GM = uint(GMAX);
    constexpr uint TOPKC = uint(TOPK);
    constexpr uint NBHC = HIDC / 128u;
    constexpr uint IN_TILES = HIDC / 16u;
    constexpr uint OUT_TILES = INTC / 16u;
    constexpr uint PER = IN_TILES / (KSC * 2u);   // input tiles per half slice
    constexpr uint SPAN = 32u * PER;               // a slice's input values
    constexpr uint NBS = (SPAN % 128u == 0u) ? SPAN / 128u : SPAN / 128u + 2u;   // 128-blocks a slice can touch
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint blk = threadgroup_position_in_grid.x;
    const uint rk = threadgroup_position_in_grid.y;
    const uint kz = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint e = uint(slots[rk]);
    uint mem[4]; uint m;
    if (!turboexl3_group<GM, uint(RKC)>(slots, rk, e, L, mem, m)) return;
    // form 6's arrival counters, one per (row, 128-block of hidden), zeroed here (rk 0 always leads)
    if (blk == 0u && rk == 0u && kz == 0u) for (uint i = tid; i < (uint(RKC) / TOPKC) * NBHC; i += 128u) cnt[i] = 0u;
    threadgroup half xs[GM * 2u * NBS * 128u];   // [member][projection][block][128], lane order within each 16-tile
    threadgroup float red[4][GM * 16u * BTC];
    // 1. the members' rotated gate and up inputs over the slice's blocks
    const uint b0 = (kz * SPAN) / 128u;
    const uint nb = ((kz + 1u) * SPAN + 127u) / 128u - b0;
    for (uint w = sg; w < m * 2u * nb; w += 4u) {
        const uint j = w / (2u * nb);
        const uint pj = (w / nb) & 1u;
        const uint b = b0 + (w % nb);
        const uint row = mem[j] / TOPKC;
        const device half* suh = (pj == 0u) ? gate_suh : up_suh;
        float v[4];
        for (uint q = 0; q < 4u; ++q) { const uint i = b * 128u + L + 32u * q; const float xr = float(half(float(x[row * HIDC + i]))); v[q] = xr * float(suh[e * HIDC + i]); }
        turboexl3_h128r(v, L);
        threadgroup half* dst = xs + ((j * 2u + pj) * NBS + (b - b0)) * 128u;
        float sv = 0.0f;
        for (uint q = 0; q < 4u; ++q) {
            const uint i = L + 32u * q;
            const uint r = i & 15u; const uint qq = (r & 7u) >> 1u; const uint mm = (r & 1u) | ((r >> 3u) << 1u);
            const half h = half(v[q] * HAD_SCALE);
            dst[(i & ~15u) + 4u * qq + mm] = h;
            sv += float(h);
        }
        sv = simd_sum(sv);
        if (blk == 0u && L == 0u) { if (pj == 0u) xsg[mem[j] * NBHC + b] = sv; else xsu[mem[j] * NBHC + b] = sv; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2. gu3's GEMV: simdgroup sg = projection (sg >> 1) x half (sg & 1) of the slice, BT output tiles, every member
    const uint pj = sg >> 1u;
    const device uint* wbase = (const device uint*)(pj == 0u ? gate_t : up_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(N / 2u);
    const uint tk0 = (kz * 2u + (sg & 1u)) * PER;
    const threadgroup half* xs0 = xs + pj * NBS * 128u + 4u * (L & 3u);
    constexpr uint XST = 2u * NBS * 128u;   // one member to the next
    if (m == 1u) turboexl3_gu5_sg<N, WMASK, CB, OUT_TILES, BTC, 1u, uint(DEFER), int(IR)>(wbase, xs0, XST, tk0, tk0 + PER, b0 * 8u, blk * BTC, L, red[sg]);
    else if (GM >= 2u && m == 2u) turboexl3_gu5_sg<N, WMASK, CB, OUT_TILES, BTC, 2u, uint(DEFER), int(IR)>(wbase, xs0, XST, tk0, tk0 + PER, b0 * 8u, blk * BTC, L, red[sg]);
    else if (GM >= 3u && m == 3u) turboexl3_gu5_sg<N, WMASK, CB, OUT_TILES, BTC, 3u, uint(DEFER), int(IR)>(wbase, xs0, XST, tk0, tk0 + PER, b0 * 8u, blk * BTC, L, red[sg]);
    else if (GM >= 4u) turboexl3_gu5_sg<N, WMASK, CB, OUT_TILES, BTC, 4u, uint(DEFER), int(IR)>(wbase, xs0, XST, tk0, tk0 + PER, b0 * 8u, blk * BTC, L, red[sg]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 3. the slice's planes: half 0 + half 1 (gu3's sum)
    for (uint i = tid; i < 16u * BTC * m; i += 128u) {
        const uint j = i / (16u * BTC);
        const ulong o = (ulong(kz) * ulong(RKC) + mem[j]) * ulong(INTC) + blk * (16u * BTC) + (i - j * 16u * BTC);
        zg[o] = red[0][i] + red[1][i];
        zu[o] = red[2][i] + red[3][i];
    }
