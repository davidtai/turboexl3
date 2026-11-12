    // pg2: pg1's copy of MLX's fused NAX GEMM kernel (steel/gemm/kernels/steel_gemm_fused_nax.h `gemm`, MLX 0.32.3; its
    // gemm_loop verbatim) at nn, x and w half, with turboexl3's dense finish (kernels/dense/finish.metal) as its
    // epilogue: y OUT_T [M, N] = finish(z), z = x [M, K] w [K, N] never in device memory. The GEMM's statements are pg1's
    // (MLX's params from the operands' shapes; BM BN BK WM WN SWZ and AM AN AK template constants). The epilogue rounds
    // the accumulators to half as MLX's store rounds them (NAXFrag::store's static_cast), stages them in threadgroup
    // memory at the simdgroup's (tm, tn), and runs the finish's statements per (row, 128-column block) of the tile, a
    // simdgroup a block at a time, the block's halves read from threadgroup memory instead of z. N a multiple of BN, BN of
    // 128 (the host's check).
    const int M = int(x_shape[0]);
    const int K = int(x_shape[1]);
    const int N = int(w_shape[1]);
    const int lda = K;
    const int ldb = N;
    const int tiles_n = (N + BN - 1) / BN;
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
    const device T* A = x;
    const device T* B = w;
    // Prepare threadgroup memory
    threadgroup_barrier(mem_flags::mem_none);
    // Find block in A, B, C
    const int c_row = tid_y * BM;
    const int c_col = tid_x * BN;
    const size_t c_row_long = size_t(c_row);
    const size_t c_col_long = size_t(c_col);
    A += c_row_long * lda;
    B += c_col_long;
    constexpr short SM = BM / WM;
    constexpr short SN = BN / WN;
    constexpr short SK = 32;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    const short tm = SM * (simd_group_id / WN);
    const short tn = SN * (simd_group_id % WN);
    const int sgp_sm_int =
        align_M ? int(SM) : min(int(SM), M - (c_row + tm));
    const short sgp_sm = short(sgp_sm_int);
    const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
    const int sgp_sn_int =
        align_N ? int(SN) : min(int(SN), N - (c_col + tn));
    const short sgp_sn = short(sgp_sn_int);
    const bool is_unaligned_sn = align_N ? false : (sgp_sn != SN);
    A += tm * lda;
    B += tn;
    NAXTile<float, TM, TN> Dtile;
    dispatch_bool(align_K, [&](auto kAlignedK) {
      dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
        dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
          Dtile = pg1_gemm_loop<
              T,
              SM,
              SN,
              SK,
              BK,
              false,
              false,
              kAlignedM.value,
              kAlignedN.value,
              kAlignedK.value,
              float>(
              A,
              B,
              lda,
              ldb,
              K,
              gemm_k_iterations_aligned,
              sgp_sm,
              sgp_sn);
        });
      });
    });
    // The epilogue. The tile's halves (MLX's store rounding) in threadgroup memory, row stride BN.
    threadgroup half zt[BM * BN];
    Dtile.template store<half, BN, 1>(&zt[tm * BN + tn]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
