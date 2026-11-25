        // MLX's Multiply (o times the sigmoid, one f32 product) and MLX's astype to G_T (gate_close's value), then
        // o_proj's prepare of it (dense prep_2_1's statement: x's f16 rounding times suh), kept for the block's butterfly
        hp[j] = float(half(float(G_T(ov * sg)))) * float(suh_o[o]);
    }
    for (uint j = 0; j < 4u; ++j) v[j] = hp[j];
