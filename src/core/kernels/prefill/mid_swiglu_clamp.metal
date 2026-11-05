        // the clamped SwiGLU (DeepSeek-V4's, GLM's): the gate clamped from above at LIMIT and up to [-LIMIT, LIMIT], in f32
        // before the SiLU; a NaN passes through, as torch.clamp passes it
        const float lim = float(LIMIT);
        const float gc = gg > lim ? lim : gg;
        const float uc = uu > lim ? lim : (uu < -lim ? -lim : uu);
        const float h = (gc / (1.0f + metal::precise::exp(-gc))) * uc;
