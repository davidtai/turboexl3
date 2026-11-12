    for (uint w = ptg; w < nwork; w += uint(NPT)) {
        // the work item's K threadgroup, lead and output tile block, in the grouped GEMV's grid order (x fastest)
        const uint kz = w / (pt_n * NBLK);
        const uint wl = w - kz * (pt_n * NBLK);
        const uint li = wl / NBLK;
        const uint blk = wl - li * NBLK;
        const uint rk = pt_lead[li];
        const uint e = uint(slots[rk]);
        uint members[GRP];
        for (uint j = 0; j < GRP; ++j) members[j] = pt_mem[rk][j];
        const uint m = pt_cnt[rk];
