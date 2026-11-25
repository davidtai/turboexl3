    // text_finish's store loop with q's value kept (models/hy4f/qrope.zig): MLX's astype to f32 of linear's OUT_T value
    // (text_finish's store expression); on each head's last ROPE dims MLX's fast::rope of it with its pair's value
    // (qrope_pair, qrope_rot), at the row's position; stored in f32 where text_finish stores y (qrope_close)
    constexpr uint HDC = uint(HD);
    constexpr uint NOPEC = uint(HD) - uint(ROPE);
    constexpr uint RH = uint(ROPE) / 2u;
    static_assert(uint(DIM) % uint(HD) == 0u && uint(ROPE) % 2u == 0u && uint(ROPE) <= uint(HD) && NOPEC % 2u == 0u, "hy4f q_b rope: whole heads, even rope dims, pairs on even dims");
    const float base = as_type<float>(int(LOG2B));
    const int pos0 = off[0];
    for (uint j = 0; j < 4u; ++j) {
        const uint o = b * 128u + L + 32u * j;
        const float qv = static_cast<float>(OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))));
        const uint c = o % HDC;
