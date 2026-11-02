    // F3: the shared expert's gate and up finishes (text_dfinish's statements: the planes in slice order, the deferred
    // MUL1 map, f16, H128, svh, f16, the output dtype) of one 128-wide block of their outputs, then the host's clamped
    // SwiGLU in f32 on them (sdk/blocks/mlp/activations.zig clampedSwiGLU after astype f32: MLX's Minimum, Maximum (clip's
    // lower bound first), Sigmoid and Multiply functors, verbatim below, Safe math as every MLX kernel), the activation
    // dtype's rounding, then the down linear's prepare (text_prep_s, lane order) on it: one simdgroup per (block, row).
    constexpr uint DIMC = uint(DIM);          // the gate's and up's outputs, the down's inputs
    constexpr uint KSG = uint(KS1);
    constexpr uint KSU = uint(KS3);
    constexpr uint ROWSC = uint(ROWS);
    constexpr uint INB = uint(INBLK);         // the gate's and up's input blocks
    constexpr uint NB = DIMC / 128u;
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint b = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    const float hi = lim[0];
    const float lo = -lim[0];
    float xsum1 = 0.0f;
    if (DEFER != 0) for (uint q = 0; q < INB; ++q) xsum1 += xs1[row * INB + q];
    float xsum3 = 0.0f;
    if (DEFER != 0) for (uint q = 0; q < INB; ++q) xsum3 += xs3[row * INB + q];
    float v1[4];
    float v3[4];
    for (uint j = 0; j < 4u; ++j) {
        const uint o = b * 128u + L + 32u * j;
        float acc = z1[ulong(row) * ulong(DIMC) + o];
        for (uint k = 1u; k < KSG; ++k) acc += z1[(ulong(k) * ulong(ROWSC) + row) * ulong(DIMC) + o];
        if (DEFER != 0) acc = fma(acc, MUL1_A, MUL1_C * xsum1);
        v1[j] = float(half(acc));
        float acc3 = z3[ulong(row) * ulong(DIMC) + o];
        for (uint k = 1u; k < KSU; ++k) acc3 += z3[(ulong(k) * ulong(ROWSC) + row) * ulong(DIMC) + o];
        if (DEFER != 0) acc3 = fma(acc3, MUL1_A, MUL1_C * xsum3);
        v3[j] = float(half(acc3));
    }
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v1[j], ushort(bit)); v1[j] = (L & bit) ? (p - v1[j]) : (v1[j] + p); } }
    { float t0 = v1[0] + v1[1], t1 = v1[0] - v1[1], t2 = v1[2] + v1[3], t3 = v1[2] - v1[3]; v1[0] = t0 + t2; v1[1] = t1 + t3; v1[2] = t0 - t2; v1[3] = t1 - t3; }
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v3[j], ushort(bit)); v3[j] = (L & bit) ? (p - v3[j]) : (v3[j] + p); } }
    { float t0 = v3[0] + v3[1], t1 = v3[0] - v3[1], t2 = v3[2] + v3[3], t3 = v3[2] - v3[3]; v3[0] = t0 + t2; v3[1] = t1 + t3; v3[2] = t0 - t2; v3[3] = t1 - t3; }
    float v[4];
    for (uint j = 0; j < 4u; ++j) {
        const uint o = b * 128u + L + 32u * j;
        // the two finishes' outputs (text_dfinish's store) and the host's astype to f32
        const float gate = float(OUT_T(float(half(v1[j] * HAD_SCALE * float(svh1[o])))));
        const float up = float(OUT_T(float(half(v3[j] * HAD_SCALE * float(svh3[o])))));
        // clampedSwiGLU: g = minimum(gate, hi); u = clip(up, lo, hi) = minimum(maximum(up, lo), hi); sig = sigmoid(g);
        // act = g * sig; out = act * u (MLX's functors for float)
        float g = metal::isnan(gate) ? gate : (gate < hi ? gate : hi);
        float um = metal::isnan(up) ? up : (up > lo ? up : lo);
        float u = metal::isnan(um) ? um : (um < hi ? um : hi);
        float sy = 1 / (1 + metal::precise::exp(metal::abs(g)));
        float sig = (g < 0) ? sy : 1 - sy;
        float act = g * sig;
        float outv = act * u;
        // the host's astype to the activation dtype, then text_prep_s on the down's input element o
        v[j] = float(half(float(ACT_T(outv)))) * float(suh2[o]);
    }
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    float sum = 0.0f;
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const uint r = i & 15u;
        const uint q = (r & 7u) >> 1u;
        const uint m = (r & 1u) | ((r >> 3u) << 1u);
        const half hv = half(v[j] * HAD_SCALE);
        xp[row * DIMC + ((i & ~15u) + 4u * q + m)] = hv;
        sum += float(hv);
    }
    sum = simd_sum(sum);
    if (L == 0u) xs[row * NB + b] = sum;
