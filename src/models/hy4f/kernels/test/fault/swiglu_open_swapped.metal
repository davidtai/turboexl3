    // the planted fault (a red check, never served): swiglu_open with the gate and up swapped, so h = silu(up) * gate
    for (uint j = 0; j < 4u; ++j) {
        const uint o = b * 128u + L + 32u * j;
        const float yv = static_cast<float>(OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))));
        const float uv = static_cast<float>(gate[ulong(row) * DIMC + o]);
