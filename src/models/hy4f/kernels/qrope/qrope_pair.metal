        // MLX's traditional (GPT-J) pairs: the rope part's dims 2 i and 2 i + 1, elements o and o ^ 1, lanes L and L ^ 1
        // (every lane shuffles: the rope part need not start at a lane quad)
        const float pv = simd_shuffle_xor(qv, ushort(1));
        const bool second = (c & 1u) != 0u;
        const uint ip = (c >= NOPEC) ? ((c - NOPEC) >> 1u) : 0u;
