    // text_finish's store loop with up's value kept (models/hy4f/swiglu.zig): the gate's y at the same element taken to
    // f32 (MLX's astype), MLX's Sigmoid of it (gate's gate_sig), then swigluOut's two products stored in H_T where
    // text_finish stores y (swiglu_close)
    for (uint j = 0; j < 4u; ++j) {
        const uint o = b * 128u + L + 32u * j;
        const float uv = static_cast<float>(OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))));
        const float yv = static_cast<float>(gate[ulong(row) * DIMC + o]);
