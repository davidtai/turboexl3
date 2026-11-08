    // text_finish's store loop with up's value kept: then the host's SwiGLU of the gate's finished value at the same
    // element and up's, and h stored where the chain stores up (swiglu.zig)
    for (uint j = 0; j < 4u; ++j) {
        const uint o = b * 128u + L + 32u * j;
        const OUT_T ut = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))));
        const OUT_T gt = gate[row * DIMC + o];
