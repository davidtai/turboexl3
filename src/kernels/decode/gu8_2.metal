    const uint kz = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    constexpr uint GRP = uint(GROUP);
    const uint e = uint(slots[rk]);
    const uint proj = sg >> 1u;
    const uint khalf = sg & 1u;
    uint members[2]; uint m;
    if (!turboexl3_members<GRP, uint(RKC)>(slots, rk, e, L, members, m)) return;
    threadgroup half xs[GRP * 2u * NBT * 128u];   // [member][projection][block][128], lane order within each 16-tile
    threadgroup float red[SPTC][4][2u * 16u * BTC];
    // 1. prep2 over the span: x f16 round, suh, H128, 1/sqrt(128), f16 round, lane order; the block's sum of the rounded inputs
    const uint b0 = kz * NBT;
    for (uint w = sg; w < m * 2u * NBT; w += 4u) {
        const uint j = w / (2u * NBT);
        const uint pj = (w / NBT) & 1u;
        const uint bl = w % NBT;
        const uint b = b0 + bl;
        const uint row = members[j] / TOPKC;
        const device half* suh = (pj == 0u) ? gate_suh : up_suh;
        float v[4];
        for (uint q = 0; q < 4u; ++q) { const uint i = b * 128u + L + 32u * q; const float xr = float(half(float(x[row * HIDC + i]))); v[q] = xr * float(suh[e * HIDC + i]); }
        turboexl3_h128r(v, L);
        threadgroup half* dst = xs + ((j * 2u + pj) * NBT + bl) * 128u;
        float sv = 0.0f;
        for (uint q = 0; q < 4u; ++q) {
            const uint i = L + 32u * q;
            const uint r = i & 15u; const uint qq = (r & 7u) >> 1u; const uint mm = (r & 1u) | ((r >> 3u) << 1u);
            const half h = half(v[q] * HAD_SCALE);
            dst[(i & ~15u) + 4u * qq + mm] = h;
            sv += float(h);
        }
        sv = simd_sum(sv);
        if (blk == 0u && L == 0u) { if (pj == 0u) xsg[ulong(members[j]) * NBHC + b] = sv; else xsu[ulong(members[j]) * NBHC + b] = sv; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2. gu3's GEMV, tile indices relative to the span (the weight base moved to its first tile)
    const uint tkb = kz * SPTC * 2u * PER;
    const device uint* wbase = (const device uint*)(proj == 0u ? gate_t : up_t) + (ulong(e) * ulong(IN_TILES) + ulong(tkb)) * ulong(OUT_TILES) * ulong(N / 2u);
    const threadgroup half* xrow0 = xs + proj * NBT * 128u + 4u * (L & 3u);
    const threadgroup half* xrow1 = xrow0 + 2u * NBT * 128u;
    for (uint si = 0; si < SPTC; ++si) {
        const uint tk0 = (si * 2u + khalf) * PER;
        if (m == 1u) turboexl3_gemv_members<N, WMASK, CB, OUT_TILES, BTC, 1u, uint(DEFER)>(wbase, xrow0, xrow0, tk0, tk0 + PER, blk, L, red[si][sg]);
        else turboexl3_gemv_members<N, WMASK, CB, OUT_TILES, BTC, 2u, uint(DEFER)>(wbase, xrow0, xrow1, tk0, tk0 + PER, blk, L, red[si][sg]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16u * BTC * m) {
        const uint j = tid / (16u * BTC);
        for (uint si = 0; si < SPTC; ++si) {
            const ulong o = (ulong(kz * SPTC + si) * ulong(RKC) + members[j]) * ulong(INTC) + blk * (16u * BTC) + (tid - j * 16u * BTC);
            zg[o] = red[si][0][tid] + red[si][1][tid];
            zu[o] = red[si][2][tid] + red[si][3][tid];
        }
    }
