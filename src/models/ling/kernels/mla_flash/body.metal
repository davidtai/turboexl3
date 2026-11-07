    // turboexl3_ling_mla_flash: MLX 0.32.5's attention_nax (steel/attn/kernels/steel_attention_nax.h, Copyright © 2024-26
    // Apple Inc., MIT), its statements in its order, less its mask, sinks, log-sum-exp and window arms (none taken here).
    // Its AttnParams come from the operands (q [B, H, qL, BD], k [B, Hk, kL, BD], v [B, Hk, kL, BV] at their own strides,
    // o contiguous) and `scale`; its function constants align_Q, align_K and do_causal are the template constants
    // ALIGN_Q, ALIGN_K and CAUSAL. Causal from the bottom right as MLX aligns it (query row i sees keys up to kL - qL + i).
    // Launched as MLX launches it: threadgroups (ceil(qL / BQ), H, B) of (32, WM, WN) threads.
    constexpr bool align_Q = ALIGN_Q != 0;
    constexpr bool align_K = ALIGN_K != 0;
    constexpr bool do_causal = CAUSAL != 0;

    const int qL = int(q_shape[2]);
    const int kL = int(k_shape[2]);
    const int gqa_factor = int(q_shape[1]) / int(k_shape[1]);
    const int NK = (kL + BK - 1) / BK;
    const int NQ_aligned = qL / BQ;
    const int NK_aligned = kL / BK;
    const int qL_rem = qL - NQ_aligned * BQ;
    const int kL_rem = kL - NK_aligned * BK;
    const int qL_off = kL - qL;
    const int64_t O_strides[3] = {int64_t(q_shape[1]) * int64_t(qL) * int64_t(BV), int64_t(qL) * int64_t(BV), int64_t(BV)};

    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;

    // Move to correct block
    ulong3 tidl{tid.x, tid.y, tid.z};

    const device T* Q = q;
    const device T* K = k;
    const device T* V = v;
    device T* O = o;

    Q += tidl.z * q_strides[0] + // Batch
        tidl.y * q_strides[1] + // Head
        tidl.x * BQ * q_strides[2]; // Sequence

    ulong kv_head_idx = int(tid.y) / gqa_factor;
    K += tidl.z * k_strides[0] + // Batch
        kv_head_idx * k_strides[1]; // Head

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
            const int K_load_off = ik * kU * int(k_strides[2]) + id * kU;

            if (!align_Q && is_last_q) {
              Qtile.load_rows(
                  Q + Q_load_off,
                  int(q_strides[2]),
                  lim_rows_q - iq * kU);
            } else {
              Qtile.load(Q + Q_load_off, int(q_strides[2]));
            }

            if (!align_K && is_last_k) {
              Ktile.load_rows(
                  K + K_load_off,
                  int(k_strides[2]),
                  lim_rows_k - ik * kU);
            } else {
              Ktile.load(K + K_load_off, int(k_strides[2]));
            }

            stile_t::NAXFrag_t::mma(
                Stile.frag_at(iq, ik),
                Stile.frag_at(iq, ik + 1),
                Qtile.frag_at(0, 0),
                metal::false_type{},
                Ktile.frag_at(0, 0),
                Ktile.frag_at(1, 0),
                metal::true_type{});
          }
        }
      }

      // Scale S
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
        Stile.elems()[ii] *= float(scale2);
      }

      // Mask out length sequence
      if (!align_K && is_last_k) {
        constexpr auto neg_inf = mla_flash_finite_min;

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const short col_pos = ik * kU + sn;

            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
              STEEL_PRAGMA_UNROLL
              for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
                const auto loc = ii * stile_t::kFragThrCols + jj;
                fg[loc] = ((col_pos + jj) < kL_rem) ? fg[loc] : neg_inf;
              }
            }
          }
        }
      }

      // Mask out if causal
      const bool is_causal_masked = do_causal && (kb >= kb_min_causal);

      if (is_causal_masked) {
        constexpr auto neg_inf = mla_flash_finite_min;

        const int base_row = tid.x * BQ + qL_off + tm;
        const int base_col = kb * BK;

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
              STEEL_PRAGMA_UNROLL
              for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
                const auto r =
                    base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
                const auto c = base_col + ik * kU + jj + sn;
                const auto loc = ii * stile_t::kFragThrCols + jj;

                fg[loc] = (r < c) ? neg_inf : fg[loc];
              }
            }
          }
        }
      }

      // Do softmax

      // Temp variables
      metal::vec<float, kRowsPT> new_max;
      metal::vec<float, kRowsPT> factor;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        new_max[i] = max_score[i];
      }

      // Row max
      Stile.template row_reduce<MaxOp>(new_max);

      // exp(Si - rowmax(Si))
      Stile.template row_bin_op<ExpSubOp>(new_max);

      // Factor exp(rowmax(Si) - rowmax(Si-1))
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        factor[i] = fast::exp2(max_score[i] - new_max[i]);
        max_score[i] = new_max[i];
      }

      // Row Sum
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        sum_score[i] = sum_score[i] * factor[i];
      }

      Stile.template row_reduce<SumOp>(sum_score);

      // Update O
      Otile.template row_bin_op<MulOp>(factor);

      simdgroup_barrier(mem_flags::mem_none);

      // Do O = P @ V
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TV; id += 2) {
          if constexpr (BV == 128) {
            if (id == 4) {
              threadgroup_barrier(mem_flags::mem_none);
            }
          }

          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            NAXTile<T, 1, 2> Vtile;

            const int V_load_off = ik * kU * int(v_strides[2]) + id * kU;

            if (!align_K && is_last_k) {
              Vtile.load_rows(
                  V + V_load_off,
                  int(v_strides[2]),
                  lim_rows_k - ik * kU);
            } else {
              Vtile.load(V + V_load_off, int(v_strides[2]));
            }

            otile_t::NAXFrag_t::mma(
                Otile.frag_at(iq, id),
                Otile.frag_at(iq, id + 1),
                Stile.frag_at(iq, ik),
                metal::false_type{},
                Vtile.frag_at(0, 0),
                Vtile.frag_at(0, 1),
                metal::false_type{});
          }
        }
      }

      // Prepare for next iteration
      K += BK * int(k_strides[2]);
      V += BK * int(v_strides[2]);
    }

    // Normalize output

    threadgroup_barrier(mem_flags::mem_none);

    metal::vec<float, kRowsPT> rcp;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      rcp[i] = 1.f / sum_score[i];
    }

    Otile.template row_bin_op<MulOp>(rcp);

    // Store results
    O += tm * int(O_strides[2]);

    if (!align_Q && is_last_q) {
      if (lim_rows_q <= 0)
        return;

      Otile.store_rows(O, int(O_strides[2]), lim_rows_q);
    } else {
      Otile.store(O, int(O_strides[2]));
    }
