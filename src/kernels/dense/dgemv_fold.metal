        }
    }
    for (uint r = 0; r < RBC; ++r) turboexl3_fold<BTC>(a[r], &red[sg][r * 16u * BTC], L);
    threadgroup_barrier(mem_flags::mem_threadgroup);
