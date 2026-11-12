    const int tiles_m = (M + BM - 1) / BM;
    const int swizzle_log = SWZ;
    const int gemm_k_iterations_aligned = K / BK;
    constexpr bool align_M = AM != 0;
    constexpr bool align_N = AN != 0;
    constexpr bool align_K = AK != 0;
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    // Find block
    const int tid_y = ((tid.y) << swizzle_log) +
        ((tid.x) & ((1 << swizzle_log) - 1));
    const int tid_x = (tid.x) >> swizzle_log;
    // Exit early if out of bounds
    if (tiles_n <= tid_x || tiles_m <= tid_y) {
      return;
    }
