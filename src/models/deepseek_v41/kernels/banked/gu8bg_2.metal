    const uint kz = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    constexpr uint GRP = uint(GROUP);   // row-slots per lead threadgroup (at most 4)
    const uint e = uint(slots[rk]);
    const uint proj = sg >> 1u;
    const uint khalf = sg & 1u;
    uint members[GRP]; uint m;
    if (!turboexl3_members_g<GRP, uint(RKC)>(slots, rk, e, L, members, m)) return;
    // the lead's bank and slot row (e is the row-slot's id, bank << 24 | slot row, as the members compare it): the suh
    // arrays prep2b reads and the trellis gu3b reads, of that bank
    const uint bk = e >> 24u;
    const uint sl = e & 0xFFFFFFu;
    const device half* gate_suh = bk == 0u ? gate_suh0 : (bk == 1u ? gate_suh1 : gate_suh2);
    const device half* up_suh = bk == 0u ? up_suh0 : (bk == 1u ? up_suh1 : up_suh2);
    threadgroup half xs[GRP * 2u * NBT * 128u];   // [member][projection][block][128], lane order within each 16-tile
    threadgroup float red[SPTC][4][GRP * 16u * BTC];
    // 1. prep2b over the threadgroup's span for each member: x f16 round, its bank's suh, H128, 1/sqrt(128), f16 round,
    // lane order; each block's sum of the rounded inputs, published by the first output-tile threadgroup
    const uint b0 = kz * NBT;
    for (uint w = sg; w < m * 2u * NBT; w += 4u) {
        const uint j = w / (2u * NBT);
        const uint pj = (w / NBT) & 1u;
        const uint bl = w % NBT;
        const uint b = b0 + bl;
        const uint row = members[j] / TOPKC;
        const device half* suh = (pj == 0u) ? gate_suh : up_suh;
        float v[4];
        for (uint q = 0; q < 4u; ++q) { const uint i = b * 128u + L + 32u * q; const float xr = float(half(float(x[row * HIDC + i]))); v[q] = xr * float(suh[sl * HIDC + i]); }
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
    // 2. gu3bg's GEMV over the members' prepared spans: the bank's trellis at the slot row with its base moved to the
    // span's first tile (tile indices relative to the span), member j's rows at xs + j 2 NBT 128 (`lm`: the members'
    // places there; `members`: their row-slots, the planes' rows)
    const uint tkb = kz * SPTC * 2u * PER;
    const device uint* wbase = (const device uint*)(proj == 0u ? (bk == 0u ? gate_t0 : (bk == 1u ? gate_t1 : gate_t2)) : (bk == 0u ? up_t0 : (bk == 1u ? up_t1 : up_t2))) + (ulong(sl) * ulong(IN_TILES) + ulong(tkb)) * ulong(OUT_TILES) * ulong(N / 2u);
    const threadgroup half* xbase = xs + proj * NBT * 128u + 4u * (L & 3u);
    const ulong ld = 2u * NBT * 128u;
    const uint lm[4] = { 0u, 1u, 2u, 3u };
