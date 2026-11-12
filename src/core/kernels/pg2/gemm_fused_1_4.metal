    // Prepare threadgroup memory
    threadgroup_barrier(mem_flags::mem_none);
    // Find block in A, B, C
    const int c_row = tid_y * BM;
