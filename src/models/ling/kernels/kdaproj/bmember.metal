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
        const device bfloat16_t* in_vec = xb + size_t(row) * size_t(IN);
        float total[4];
        for (int sgN = 0; sgN < 8; ++sgN) {
          float result[4] = {0};
          bfloat16_t inter[4];
          float v_coeff[4];
          const int simdN = 32 * sgN;
          int bn = (simdN + thrN) * 4;
          for (int i = 0; i < IN / 1024; ++i) {
            for (int tn = 0; tn < 4; tn++) {
              v_coeff[tn] = static_cast<float>(in_vec[bn + tn]);
            }
            int mat_offset = 0;
            for (int tm = 0; tm < 4; tm++) {
              for (int tn = 0; tn < 4; tn++) {
                inter[tn] = static_cast<bfloat16_t>(mat[mat_offset + bn + tn]);
              }
              for (int tn = 0; tn < 4; tn++) {
                result[tm] += inter[tn] * v_coeff[tn];
              }
              mat_offset += IN;
            }
            bn += 1024;
          }
          for (int tm = 0; tm < 4; tm++) {
            for (ushort sn = 16; sn >= 1; sn >>= 1) {
              result[tm] += simd_shuffle_down(result[tm], sn);
            }
          }
          for (int tm = 0; tm < 4; tm++) {
            total[tm] = (sgN == 0) ? result[tm] : (total[tm] + result[tm]);
          }
        }
        if (thrN == 0) {
          for (int tm = 0; tm < 4; tm++) {
            yb[size_t(row) * size_t(OUTB) + size_t(out_row + tm)] = static_cast<bfloat16_t>(total[tm]);
          }
        }
      }
      return;
    }
