        // the SwiGLU clamped after the SiLU (Ling-3.1's, exllamav3's act_limit with SiLU): the gate's SiLU clamped from
        // above at LIMIT and up to [-LIMIT, LIMIT], in f32; a NaN passes through, as torch.clamp passes it
        const float lim = float(LIMIT);
        const float sg = gv / (1.0f + metal::exp(-gv));
        const float sc = sg > lim ? lim : sg;
        const float uc = uv > lim ? lim : (uv < -lim ? -lim : uv);
        const float hv = sc * uc;
