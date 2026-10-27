        // arrival: the planes and the published sums fenced device-wide before one count at the shared block; the count
        // that reaches S_ARR runs the shared fin head's statements for the block, every row
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        threadgroup_barrier(mem_flags::mem_device);
        if (s_tid == 0u) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            const uint s_old = atomic_fetch_add_explicit((device atomic_uint*)cnt + (uint(RKC) * S_NBI + s_bi), 1u, memory_order_relaxed);
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            s_arrived = (s_old == S_ARR - 1u) ? 1u : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s_arrived == 0u) return;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const device TURBOEXL3_COH float* s_zgc = (const device TURBOEXL3_COH float*)szg;
        const device TURBOEXL3_COH float* s_zuc = (const device TURBOEXL3_COH float*)szu;
        for (uint s_row = s_sg; s_row < S_RS; s_row += 4u) {
        // the shared fin head for (s_row, block s_bi), its statements from its sums on
        const uint s_b = s_bi;
