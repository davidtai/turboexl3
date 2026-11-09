                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (cb + 1 < NCBC) Dt.clear();
    }
