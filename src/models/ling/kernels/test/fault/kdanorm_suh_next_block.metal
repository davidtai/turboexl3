    //! copied_from: mlx-serve kernels/gdn/gdn_normgate.metal @ bc6c5f2c: its statements at head b, the sigmoid gate on
    //! its table (TAB 1); each element's value then taken by dense's prepare in its load's place (L5, kdanorm.zig)
    const uint lane = L;
    const uint head = b;
    uint base = (row * uint(HV) + head) * uint(DV) + lane * 4;
    float xs[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        xs[i] = float(y[base + i]);
        sumsq += xs[i] * xs[i];
    }
    sumsq = simd_sum(sumsq);
    float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
    // the planted fault (a red check, never served): suh read from the next block
    float v[4];
    for (uint j = 0; j < 4u; ++j) {
        const uint c = L + 32u * j;
        const float xc = float(y[(row * uint(HV) + head) * uint(DV) + c]);
        const T normed = norm_w[c] * T(xc * inv);
        const T zv = z[row * uint(ZSTRIDE) + uint(ZOFF) + head * uint(DV) + c];
        T sig;
        sig = sigtab[as_type<ushort>(zv)];
        const T ov = normed * sig;
        v[j] = float(half(float(ov))) * float(suh[((b + 1u) % uint(HV)) * 128u + c]);
    }
