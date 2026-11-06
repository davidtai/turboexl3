        for (uint i = 0; i < C; ++i) {
            if (col + i < uint(D)) {
                const float p = pk[ok + i] * w;                 // MLX's Multiply(routed, weights): the product in f32
                const float v = p + 0.0f;                       // col_reduce_small's thread k: op(v_k, Sum::init)
                t[i] = (k == 0u) ? v : (v + t[i]);              // thread 0: shared_vals[0], then op(shared_vals[k], totals)
            }
        }
    }
    for (uint i = 0; i < C; ++i) {
        if (col + i < uint(D)) {
            out[obase + i] = t[i];
        }
    }
