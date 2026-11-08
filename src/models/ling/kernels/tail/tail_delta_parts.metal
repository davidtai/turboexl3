        {
            // the delta over the call's unjoined outputs: each pair's row read in place from its source s0..s7 (f32
            // [r_i, D]) at its (source, row) in loc int32 [rows, K, 2], the join's last carrier; then tail_delta's
            // statements: MLX's Multiply rounds each product to f32 on its own, col_reduce_small's fold in k order, the
            // astype to T, then the MoE block's add of the shared expert XS in T
            #pragma METAL fp contract(off)
            float tk = 0.0f;
            for (uint k = 0; k < uint(K); ++k) {
                const float w = float(WT[size_t(r) * K + k]);
                const size_t lk = 2 * (size_t(r) * K + k);
                const device float* yk = ling_tail_pick(loc[lk], s0, s1, s2, s3, s4, s5, s6, s7);
                const float pk = float(yk[size_t(loc[lk + 1]) * D + c]) * w;
                const float v = pk + 0.0f;
                tk = (k == 0u) ? v : (v + tk);
            }
            delta = float(T(tk));
            delta = float(T(delta + float(XS[at])));
        }
