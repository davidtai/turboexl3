        // the SwiGLU clamped after the SiLU (Ling-3.1's, exllamav3's act_limit with SiLU): the gate's SiLU clamped from
        // above at LIMIT and up to [-LIMIT, LIMIT], in f32; a NaN passes through, as torch.clamp passes it
        const float lim = float(LIMIT);
        const float sg = gg / (1.0f + metal::precise::exp(-gg));
        const float sc = sg > lim ? lim : sg;
        const float uc = uu > lim ? lim : (uu < -lim ? -lim : uu);
        const float h = sc * uc;
