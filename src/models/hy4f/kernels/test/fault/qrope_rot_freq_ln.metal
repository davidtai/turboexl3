        // the planted fault (a red check, never served): qrope_rot with the frequency by MLX's CPU fallback formula
        // (fast.cpp's default_inv_freqs: exp(-i ln(base) / (dims / 2))) in place of the GPU kernel's exp2(-d log2(base))
        float inv_freq = metal::precise::exp(-static_cast<float>(ip) * ((base * M_LN2_F) / static_cast<float>(RH)));
        const float scale = 1.0f;
        float Lp = scale * static_cast<float>(row + pos0);
        float theta = Lp * inv_freq;
        float costheta = metal::fast::cos(theta);
        float sintheta = metal::fast::sin(theta);
        const float x1 = second ? pv : qv;
        const float x2 = second ? qv : pv;
        float rx1 = x1 * costheta - x2 * sintheta;
        float rx2 = x1 * sintheta + x2 * costheta;
