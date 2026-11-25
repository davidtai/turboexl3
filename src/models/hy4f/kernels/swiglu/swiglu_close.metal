        // swigluOut's MLX Multiply twice in its order (the gate times its sigmoid, that times up, each one f32 product),
        // then MLX's astype to H_T (the down linear's input dtype)
        h[ulong(row) * DIMC + o] = H_T((yv * sg) * uv);
    }
