    // the pair's arrival: this output block's finish stored (fenced device-wide) and counted once at its (128-block of the
    // gate's outputs, row chunk) pair; the second of the gate's and the up's finishes runs, for the chunk's rows, the
    // host's SwiGLU of the block's gate and up values and the down linear's prepare of that block (text_prep_s's
    // statements, lane order, with its block sum), one simdgroup per row; nothing waits
    const uint pb = m1 ? ob - uint(OB1) : ob;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint old = atomic_fetch_add_explicit((device atomic_uint*)pcnt + (pb * NCHC + chunk), 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        arrived_last = (old == 1u) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (arrived_last == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    constexpr uint DB = uint(OB1);      // the down's input 128-blocks: the gate's (and the up's) output blocks
    constexpr uint DIMC = DB * 128u;    // the down's inputs
    for (uint rr = sg; rr < RBC; rr += 4u) {
        const uint row = r0 + rr;
        if (row >= ROWSC) break;
        float v[4];
        for (uint j = 0; j < 4u; ++j) {
            const uint o = pb * 128u + L + 32u * j;
            const OUT_T gt = yc[row * OUTC + o];
            const OUT_T ut = yc[row * OUTC + DIMC + o];
