        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (POOL != 0 && ob == uint(NQB) && chunk == 0u && sg == 0u) {
