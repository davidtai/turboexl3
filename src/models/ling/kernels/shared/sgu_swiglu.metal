            // the host's SwiGLU, mlx-serve's fusedSwiGLU (kernels/mlp/swiglu.metal) in the activation dtype: the sigmoid
            // read from the host's table of MLX's own results by the gate's pattern, then MLX's Multiply twice
            const OUT_T sig = sigtab[as_type<ushort>(gt)];
            const OUT_T act = gt * sig;
            const OUT_T h = act * ut;
