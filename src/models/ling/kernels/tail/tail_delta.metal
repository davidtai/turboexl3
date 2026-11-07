        {
            // the delta: the source's routed sum over the call's joined rows Y [rows, K, D] f32 and the routes' weights WT
            // [rows, K] (turboexl3's combine statements: MLX's Multiply rounds each product to f32 on its own, then
            // col_reduce_small's fold in k order), its astype to T, then the MoE block's add of the shared expert XS in T
            #pragma METAL fp contract(off)
            float tk = 0.0f;
            for (uint k = 0; k < uint(K); ++k) {
                const float w = float(WT[size_t(r) * K + k]);
                const float pk = float(Y[(size_t(r) * K + k) * D + c]) * w;
                const float v = pk + 0.0f;
                tk = (k == 0u) ? v : (v + tk);
            }
            delta = float(T(tk));
            delta = float(T(delta + float(XS[at])));
        }
