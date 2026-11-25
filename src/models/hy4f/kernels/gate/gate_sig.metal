        // MLX's Sigmoid (mlx/backend/metal/kernels/unary_ops.h) at T = float, as MLX runs it on the gate's f32 copy
        const float ye = 1 / (1 + metal::precise::exp(metal::abs(yv)));
        const float sg = (yv < 0) ? ye : 1 - ye;
