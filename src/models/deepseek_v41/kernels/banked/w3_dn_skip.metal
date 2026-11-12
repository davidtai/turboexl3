    // wave3: a skipped row-slot runs no decode and its y row is 0 (dfin3bg's: its acc past the skip line; at the banked
    // call's top-1 a dfin3 row is one row-slot), stored by the row-slot's first K threadgroup over each column block
    if (uint(slots[rk]) == 0xffffffffu) {
        if (threadgroup_position_in_grid.z == 0u && thread_index_in_threadgroup < 16u * BTC) y[ulong(rk) * ulong(HIDC) + blk * (16u * BTC) + thread_index_in_threadgroup] = OUT_T(0.0f);
        return;
    }
