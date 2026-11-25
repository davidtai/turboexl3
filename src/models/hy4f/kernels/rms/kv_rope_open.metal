    // k_pe: the projection's values LAT .. LAT + ROPE of the row (the staged y), MLX's fast::rope of each with its
    // traditional pair at the row's position, in the simdgroup holding them (qrope's pair and rope statements, the rope
    // part's dim c - LAT); its f32 values at k_pe's row. Every lane of the simdgroup takes each step (the pair's shuffle),
    // its element inside the row (the block LAT / 128 lies in DIM); only the rope part's are stored (kv_rope_close).
    if (b == uint(LAT) / 128u) {
        constexpr uint NOPEC = uint(LAT);
        constexpr uint RH = uint(ROPE) / 2u;
        static_assert(uint(LAT) % 128u == 0u && uint(ROPE) % 2u == 0u && uint(ROPE) <= 128u && uint(LAT) + uint(ROPE) <= uint(DIM), "hy4f kv: the rope part one simdgroup's, on even dims");
        const float base = as_type<float>(int(LOG2B));
        const int pos0 = off[0];
        for (uint j = 0; j < (uint(ROPE) + 31u) / 32u; ++j) {
            const uint o = b * 128u + L + 32u * j;
            const uint c = o;
            const float qv = ys[o];
