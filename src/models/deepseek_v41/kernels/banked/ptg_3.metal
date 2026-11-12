        threadgroup_barrier(mem_flags::mem_threadgroup);   // every store has read red before the next item folds into it
    }
