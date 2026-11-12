
    V += tidl.z * v_strides[0] + // Batch
        kv_head_idx * v_strides[1]; // Head

    O += tidl.z * O_strides[0] + // Batch
        tidl.y * O_strides[1] + // Head
        tidl.x * BQ * O_strides[2]; // Sequence

    const metal::uniform<float> scale2 =
        make_uniform(scale[0]) * make_uniform(1.44269504089f);

    // Prepare MMA tiles
    constexpr short kU = 16;

    constexpr int kNWarps = WM * WN;
    static_assert(
        BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
        "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

    // Q seq frags per warp
    constexpr int TQ = BQ / (kNWarps * kU);
    // HeadDim frags (all warps load the same frags)
    constexpr int TD = BD / kU;
    constexpr int TV = BV / kU;
    static_assert(BD % kU == 0 && BV % (2 * kU) == 0, "Invalid head dimensions");
    // KV seq frags per warp
    constexpr short TK = BK / kU;

    static_assert(TQ == 1, "Check TQ");
    using otile_t = NAXTile<float, TQ, TV>;
    otile_t Otile;

    Otile.clear();

    // Prepare mma tile offsets
    const short tm = kU * TQ * simd_group_id;
    Q += tm * int(q_strides[2]);

    const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
    const short sm = simd_coord.y;
    const short sn = simd_coord.x;

    // Init row reduction variables
    constexpr short kRowsPT = otile_t::kRowsPerThread;

    metal::vec<float, kRowsPT> max_score;
    metal::vec<float, kRowsPT> sum_score{0};

    // Init to -Inf
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = mla_flash_finite_min;
    }

    int kb_lim = NK;
    int kb_min_causal = NK;

    // Query positions covered by this block are [q_min, q_max)
    const int q_max = (tid.x + 1) * BQ + qL_off;
    const int q_min = max(0, int(tid.x) * BQ + qL_off);

    if (do_causal) {
      kb_lim = (q_max + BK - 1) / BK;
      kb_lim = min(NK, kb_lim);

      kb_min_causal = (q_min / BK);
    }

    const bool is_last_bq = int(tid.x) == (NQ_aligned);
    const bool is_last_q = is_last_bq;

    const short lim_rows_q = qL_rem - tm;
    const short lim_rows_k = kL_rem;

    // Loop over KV seq length
    for (int kb = 0; kb < kb_lim; kb++) {
      const int is_last_k = (kb == (NK_aligned));

      // Do S = Q @ K.T
      using stile_t = NAXTile<float, TQ, TK>;
      stile_t Stile;

      Stile.clear();

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik += 2) {
#pragma clang loop unroll_count(4)
          for (short id = 0; id < TD; id++) {
            NAXTile<T, 1, 1> Qtile;
            NAXTile<T, 2, 1> Ktile;

            const int Q_load_off = iq * kU * int(q_strides[2]) + id * kU;
