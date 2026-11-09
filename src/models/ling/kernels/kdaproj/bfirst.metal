    // b_proj in the variant launch (ling/kdaproj.zig `kdaProjB` with a variant on), kdaproj/bmember_1's choice of
    // threadgroups and rows at BF 0. BF 1 (Config.kda_bfirst) puts b's threadgroups first in the grid, OUTB / 4 of them a
    // row chunk (the members' follow from x = OUTB / 4), so they run beside the members' instead of after them, and gives
    // each row of the chunk its own simdgroup (simdgroup s takes rows rb0 + s, rb0 + s + 4, ...) in place of simdgroup 0
    // taking every row in turn. Each row runs kdaproj/bmember_2's statements (MLX's gemv over that row) as before: a
    // row's words depend on neither the simdgroup nor the threadgroup that computes it.
    if ((BF != 0) ? (threadgroup_position_in_grid.x < uint(OUTB) / 4u) : (threadgroup_position_in_grid.x >= uint(FB))) {
      if (BF == 0 && simdgroup_index_in_threadgroup != 0u) return;
      const int thrN = int(thread_index_in_simdgroup);
      const int out_row = int((BF != 0) ? threadgroup_position_in_grid.x : (threadgroup_position_in_grid.x - uint(FB))) * 4;
      const device bfloat16_t* mat = wb + size_t(out_row) * size_t(IN);
      const uint rb0 = threadgroup_position_in_grid.y * uint(RB);
      const uint rb1 = min(rb0 + uint(RB), uint(ROWS));
      for (uint row = rb0 + ((BF != 0) ? simdgroup_index_in_threadgroup : 0u); row < rb1; row += ((BF != 0) ? 4u : 1u)) {
