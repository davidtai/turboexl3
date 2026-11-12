    const uint o = blk * 128u + tid;
    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    float acc = 0.0f;
    for (uint k = 0; k < TOPKC; ++k) {
        const uint rk = row * TOPKC + k;
