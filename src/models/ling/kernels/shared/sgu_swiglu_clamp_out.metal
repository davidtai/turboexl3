            // the host's SwiGLU clamped after the SiLU (the plugin's swigluClampOut): both in f32, MLX's Sigmoid (read from
            // the host's table of MLX's f32 sigmoid of every 16-bit pattern taken to f32, by the gate's pattern) and
            // Multiply, Minimum with L, clip to [-L, L] (Maximum, then Minimum), Multiply, back in the activation dtype
            const float hi = lim[0];
            const float lo = -lim[0];
            const float g = float(gt);
            const float u = float(ut);
            const float sig = sig32[as_type<ushort>(gt)];
            const float act = g * sig;
            const float am = metal::isnan(act) ? act : (act < hi ? act : hi);
            const float um = metal::isnan(u) ? u : (u > lo ? u : lo);
            const float uc = metal::isnan(um) ? um : (um < hi ? um : hi);
            const float outv = am * uc;
            const OUT_T h = OUT_T(outv);
