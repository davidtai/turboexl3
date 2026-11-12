            sum += float(hv);
        }
        sum = simd_sum(sum);
        // the deferred map's input sums: the block's first column group publishes them (it arrives like every other)
        if (DEFER != 0 && (blk % GC) == 0u && L == 0u && r0 + rr < ROWSC) xsc[((ob * NCHC + chunk) * RBC + rr) * NB + b] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
