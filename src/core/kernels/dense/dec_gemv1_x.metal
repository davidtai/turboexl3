        float xr[RBC][4];
        for (uint r = 0; r < RBC; ++r) {
            const half4 x4 = *reinterpret_cast<const threadgroup half4*>(xt + r * SPAN + (tk - tks) * 16u + 4u * (L & 3u));
            xr[r][0] = float(x4.x); xr[r][1] = float(x4.y); xr[r][2] = float(x4.z); xr[r][3] = float(x4.w);
        }
