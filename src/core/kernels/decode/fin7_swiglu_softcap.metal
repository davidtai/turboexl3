        // the softcapped SwiGLU (Inkling 2's): cg tanh(g / cg) sigmoid(g) * cu tanh(u / cu) in f32, the sigmoid on the raw
        // gate; LIMIT packs the two integer caps, cg in bits 0-7 and cu in bits 8-15 (format.softcapLimit)
        const float cg = float(LIMIT & 255);
        const float cu = float((LIMIT >> 8) & 255);
        const float gs = cg * metal::precise::tanh(gv / cg);
        const float us = cu * metal::precise::tanh(uv / cu);
        const float hv = (gs / (1.0f + metal::exp(-gv))) * us;
