
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
