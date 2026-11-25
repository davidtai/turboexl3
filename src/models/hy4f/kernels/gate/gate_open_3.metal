    for (uint j = 0; j < 4u; ++j) {
        const uint o = b * 128u + L + 32u * j;
        const float yv = static_cast<float>(OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))));
