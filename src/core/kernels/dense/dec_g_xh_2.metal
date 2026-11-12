            const half4 x4 = *reinterpret_cast<const device half4*>(g_xb + ulong(row) * ulong(INC));
            xr[r][0] = float(x4.x); xr[r][1] = float(x4.y); xr[r][2] = float(x4.z); xr[r][3] = float(x4.w);
