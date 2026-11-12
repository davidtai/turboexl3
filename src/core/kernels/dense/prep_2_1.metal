    float v[4];
    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]); }
