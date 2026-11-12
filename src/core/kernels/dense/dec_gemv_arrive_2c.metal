    }
    for (uint r = 0; r < RBC; ++r) turboexl3_fold<BTC>(a[r], &red[sg][r * 16u * BTC], L);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < RBC * 16u * BTC; i += 128u) {
        const uint r = i / (16u * BTC);
        const uint row = r0 + r;
        if (row < ROWSC) zc[(ulong(ksl) * ulong(ROWSC) + row) * ulong(OUTC) + blk * (16u * BTC) + (i - r * 16u * BTC)] = ((red[0][i] + red[1][i]) + red[2][i]) + red[3][i];
    }
    // arrival: this threadgroup's stores fenced device-wide before its one count; the count that reaches ARR finishes
    // the output block here, every other threadgroup is done (nothing waits)
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint old = atomic_fetch_add_explicit((device atomic_uint*)cnt + (ob * NCHC + chunk), 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        arrived_last = (old == ARR - 1u) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (arrived_last == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    // text_dfinish for this output block, simdgroup rr for the chunk's row rr
    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    for (uint rr = sg; rr < RBC; rr += 4u) {
        const uint row = r0 + rr;
        if (row >= ROWSC) break;
        float xsum = 0.0f;
