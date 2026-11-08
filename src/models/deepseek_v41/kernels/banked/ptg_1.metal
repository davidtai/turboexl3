    // the persistent grouped GEMV (`Banked.banked_ptg`): NPT threadgroups each run the call's work items in turn, each
    // item what one lead threadgroup of the grouped GEMV runs (its lead, output tile block and K threadgroup); gu3_1's
    // and down3_1's blk is this threadgroup's index among the NPT
    const uint ptg = blk;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    constexpr uint GRP = uint(GROUP);   // row-slots per lead (at most 4)
    constexpr uint RKN = uint(RKC);
    threadgroup float red[SPTC][4][GRP * 16u * BTC];   // one reduction buffer per slice, each work item's in turn
    // the call's leads and their members, once a threadgroup: each row-slot's turboexl3_members_g on one simdgroup (the
    // grouped GEMVs' own scan: the same leads, members and member order), then the leads in row-slot order
    threadgroup uint pt_mem[RKN][GRP];
    threadgroup uint pt_cnt[RKN];       // a lead row-slot's members, 0 for any other row-slot
    threadgroup uint pt_lead[RKN];
    threadgroup uint pt_n;
    for (uint r = sg; r < RKN; r += 4u) {
        uint mr[GRP]; uint mm;
        const bool lead = turboexl3_members_g<GRP, RKN>(slots, r, uint(slots[r]), L, mr, mm);
        if (L == 0u) {
            pt_cnt[r] = lead ? mm : 0u;
            for (uint j = 0; j < GRP; ++j) pt_mem[r][j] = (j < mm) ? mr[j] : r;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint n = 0u;
        for (uint r = 0; r < RKN; ++r) if (pt_cnt[r] != 0u) pt_lead[n++] = r;
        pt_n = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint NBLK = OUT_TILES / BTC;   // the grouped GEMV's grid x: output tile blocks
    constexpr uint NKZ = KSC / SPTC;         // its grid z: K threadgroups
    const uint nwork = pt_n * NBLK * NKZ;
