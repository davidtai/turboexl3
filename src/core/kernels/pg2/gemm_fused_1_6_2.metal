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
