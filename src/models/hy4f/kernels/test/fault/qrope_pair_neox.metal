        // the planted fault (a red check, never served): MLX's non-traditional (NeoX) pairs, dims i and i + ROPE / 2 of
        // the rope part: at ROPE 64 the element o ^ 32, the same lane's value j ^ 1 (text_finish's expression on it)
        static_assert(uint(ROPE) == 64u, "hy4f q_b rope fault: the NeoX partner at ROPE 64");
        const float pv = static_cast<float>(OUT_T(float(half(v[j ^ 1u] * HAD_SCALE * float(svh[o ^ 32u])))));
        const bool second = (c >= NOPEC) && ((c - NOPEC) >= RH);
        const uint ip = (c >= NOPEC) ? ((c - NOPEC) % RH) : 0u;
