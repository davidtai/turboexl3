    // wave3's arrival (DEC-ONE's, dense.zig): the planes' stores fenced device-wide, then one count a member at the
    // member's finish unit, fin7cbg's threadgroup (the 128-wide block of intermediate this threadgroup's outputs lie in,
    // the member's row-slot). A unit's planes come from the 8 / BT column threadgroups over its block at each of the
    // KS / SPT K threadgroups of its row-slot's one lead, so the count that reaches them all is its last: that
    // threadgroup's simdgroup of the member runs fin7cbg's statements for the unit. Nothing waits.
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid < m) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint old = atomic_fetch_add_explicit((device atomic_uint*)cnt + (members[tid] * (INTC / 128u) + blk / (8u / BTC)), 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        arrived_last[tid] = (old == (8u / BTC) * (KSC / SPTC) - 1u) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg >= m || arrived_last[sg] == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    {
