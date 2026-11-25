    // pg2's split K in registers (pgemm.zig `SplitK`), in gemm_fused_1_6_2's place: MLX's NAX split-K
    // (steel/gemm/kernels/steel_gemm_splitk_nax.h `gemm_splitk_nax`) runs partition p of SKS k (the last up to K) from
    // a cleared tile with its own gemm_loop and stores the f32 partial whole; its accumulation
    // (steel/gemm/kernels/steel_gemm_splitk.h `gemm_splitk_accum`) sums the SKP partials in order from 0 in f32, then
    // rounds to the output type once. Here each partition's loop is the same call on the same operands (A and B at
    // k_start, the partition's k count and its full BK steps, its alignment at runtime), its tile added to the sum's in
    // partition order from a cleared tile: the same f32 adds, no partial in device memory. The sum is the tile the
    // epilogue stages (gemm_fused_1_6_3: MLX's store rounding to half).
    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();
    for (int p = 0; p < SKP; ++p) {
      const int k_start = SKS * p;
      const int k_end = min(k_start + SKS, K);
      const int partition_k_size = k_end - k_start;
      const int partition_k_iters = partition_k_size / BK;
      const bool partition_k_aligned = (partition_k_size % BK) == 0;
      NAXTile<float, TM, TN> Ptile;
      dispatch_bool(partition_k_aligned, [&](auto kAlignedK) {
        dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
          dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
            Ptile = pg1_gemm_loop<
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
                A + k_start,
                B + size_t(k_start) * ldb,
                lda,
                ldb,
                partition_k_size,
                partition_k_iters,
                sgp_sm,
                sgp_sn);
          });
        });
      });
      // gemm_splitk_accum's `out += C_split[offset]`, partition p's add, element by element
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < NAXTile<float, TM, TN>::kElemsPerTile; ++i) Dtile.elems()[i] += Ptile.elems()[i];
    }
