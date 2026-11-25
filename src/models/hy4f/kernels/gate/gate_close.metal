        // MLX's Multiply (o times the sigmoid, one f32 product), then MLX's astype to G_T
        gated[ulong(row) * DIMC + o] = G_T(ov * sg);
    }
