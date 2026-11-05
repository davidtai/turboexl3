        device TURBOEXL3_COH half* wvc = (device TURBOEXL3_COH half*)wv;
        for (uint q = 0; q < 4u; ++q) {
            const uint o = hb * 128u + L + 32u * q;
            wvc[ulong(mem[j]) * ulong(HIDC) + o] = half(v[q] * HAD_SCALE * float(down_svh[e * HIDC + o]));
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        simdgroup_barrier(mem_flags::mem_device);
        const uint row = mem[j] / uint(TOPK);
        uint old = 0u;
        if (L == 0u) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            old = atomic_fetch_add_explicit((device atomic_uint*)cnt + row * NBHC + hb, 1u, memory_order_relaxed);
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        }
        old = simd_broadcast(old, ushort(0));
        if (old == uint(TOPK) - 1u) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            const device TURBOEXL3_COH half* wr = (const device TURBOEXL3_COH half*)wv;
            for (uint q = 0; q < 4u; ++q) {
                const uint o = hb * 128u + L + 32u * q;
                float acc = 0.0f;
                for (uint k = 0; k < uint(TOPK); ++k) { const uint r2 = row * uint(TOPK) + k; acc = fma(float(scores[r2]), float(wr[ulong(r2) * ulong(HIDC) + o]), acc); }
                y[ulong(row) * ulong(HIDC) + o] = OUT_T(acc);
            }
        }
    }
