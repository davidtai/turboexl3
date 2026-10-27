    // arrival: the planes fenced device-wide before one count per member; the count that reaches ARR finishes that
    // member's (row-slot, block) here, every other threadgroup is done (nothing waits)
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid < m) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint old = atomic_fetch_add_explicit((device atomic_uint*)cnt + (members[tid] * NBIC + bi), 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        arrived[tid] = (old == ARR - 1u) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg >= m || arrived[sg] == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    {
    // fin7 for (row-slot members[sg], block bi), its statements from its expert line on
    const uint b = bi;
    const uint rk = members[sg];
