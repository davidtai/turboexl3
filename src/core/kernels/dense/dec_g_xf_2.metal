            const float4 x4 = *reinterpret_cast<const device float4*>(g_xb + ulong(row) * ulong(INC));
            xr[r][0] = x4.x; xr[r][1] = x4.y; xr[r][2] = x4.z; xr[r][3] = x4.w;
