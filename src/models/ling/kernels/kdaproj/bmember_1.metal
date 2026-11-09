    // b_proj as a bf16 member of the GEMV launch (ling/kdaproj.zig `kdaProjB`, the outside plugin turboexl3_ling's
    // MLX_SERVE_LING_KDA_BPROJ): the grid's threadgroups from FB on, OUTB / 4 of them a row chunk, are b's. Each runs
    // MLX's gemv (gemv.h GEMVKernel::run at BM 1, BN 8, SM 1, SN 32, TM 4, TN 4: the kernel MLX's matmul picks for a
    // row x [1, IN] bf16 times b's weight as the host loads it, [IN, OUTB] column-major, OUTB < 4096 and IN >= 16 OUTB)
    // over each row of its chunk, its statements in order. MLX's threadgroup runs its eight simdgroups' K slices side by
    // side and adds their sums in simdgroup order; here simdgroup 0 runs the eight slices one after another and adds them
    // in that order: the same values in the same order. IN % 1024 == 0 and OUTB % 4 == 0 (the host checks both): no
    // leftover block, no tail rows.
    if (threadgroup_position_in_grid.x >= uint(FB)) {
      if (simdgroup_index_in_threadgroup != 0u) return;
      const int thrN = int(thread_index_in_simdgroup);
      const int out_row = int(threadgroup_position_in_grid.x - uint(FB)) * 4;
      const device bfloat16_t* mat = wb + size_t(out_row) * size_t(IN);
      const uint rb0 = threadgroup_position_in_grid.y * uint(RB);
      const uint rb1 = min(rb0 + uint(RB), uint(ROWS));
      for (uint row = rb0; row < rb1; ++row) {
