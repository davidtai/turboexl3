        // the clamped SwiGLU (DeepSeek-V4's, GLM's): the gate clamped from above at LIMIT and up to [-LIMIT, LIMIT], in f32
        // before the SiLU; a NaN passes through, as torch.clamp passes it
        const float lim = float(LIMIT);
        const float gc = gv > lim ? lim : gv;
        const float uc = uv > lim ? lim : (uv < -lim ? -lim : uv);
        const float hv = (gc / (1.0f + metal::exp(-gc))) * uc;
