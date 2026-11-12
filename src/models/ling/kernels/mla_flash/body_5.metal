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
