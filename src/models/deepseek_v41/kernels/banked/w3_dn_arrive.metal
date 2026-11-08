    // wave3's arrival, as the gate|up launch's: one count a member at the member's finish unit, dfin3bg's threadgroup
    // (the 128-wide block of hidden this threadgroup's outputs lie in, the member's row: at the banked call's top-1 one
    // row-slot), the down units' counters after the gate|up units'. This threadgroup runs dfin3bg's statements for each
    // unit its count completed, in member order (dfin3's H128 takes the whole threadgroup). Nothing waits.
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid < m) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint old = atomic_fetch_add_explicit((device atomic_uint*)cnt + (uint(RKC) * (INTC / 128u) + members[tid] * (HIDC / 128u) + blk / (8u / BTC)), 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        arrived_last[tid] = (old == (8u / BTC) * (KSC / SPTC) - 1u) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint w3_ob = blk / (8u / BTC);
    for (uint w3_j = 0; w3_j < m; ++w3_j) {
        if (arrived_last[w3_j] == 0u) continue;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
