        // MLX's rope (mlx/backend/metal/kernels/rope.metal, `rope` / `rope_impl` with forward and traditional, T float),
        // its statements in its order: d the pair index over dims / 2 (grid.x there), inv_freq = exp2(-d log2(base))
        // (log2(base) the host's float, here LOG2B's bits; MLX's exp2 under -fno-fast-math is the precise one), L = scale
        // (row + offset) at scale 1, theta = L inv_freq, the fast cosine and sine, then (x1, x2) the pair's even and odd
        // values rotated
        float d = static_cast<float>(ip) / static_cast<float>(RH);
        float inv_freq = metal::precise::exp2(-d * base);
        const float scale = 1.0f;
        float Lp = scale * static_cast<float>(row + pos0);
        float theta = Lp * inv_freq;
        float costheta = metal::fast::cos(theta);
        float sintheta = metal::fast::sin(theta);
        const float x1 = second ? pv : qv;
        const float x2 = second ? qv : pv;
        float rx1 = x1 * costheta - x2 * sintheta;
        float rx2 = x1 * sintheta + x2 * costheta;
