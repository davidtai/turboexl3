        // swigluOut's two products in its order and MLX's astype to H_T (swiglu_close's value), then the down linear's
        // prepare of it (dense prep_2_1's statement: x's f16 rounding times suh), kept for the block's butterfly
        hp[j] = float(half(float(H_T((yv * sg) * uv)))) * float(suh[o]);
    }
    for (uint j = 0; j < 4u; ++j) v[j] = hp[j];
