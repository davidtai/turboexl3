    }
    turboexl3_h128x4(g, L);
    turboexl3_h128x4(u, L);
    float d[4];
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const float gg = float(half(g[j] * HAD_SCALE * float(svh_g[ulong(e) * IC + i])));
        const float uu = float(half(u[j] * HAD_SCALE * float(svh_u[ulong(e) * IC + i])));
        const float h = (gg / (1.0f + metal::precise::exp(-gg))) * uu;
        d[j] = float(half(h)) * float(suh_d[ulong(e) * IC + i]);
    }
    turboexl3_h128x4(d, L);
    float sd = 0.0f;
    const uint lo = (ALAY == 1) ? turboexl3_alay(L) : ((ALAY == 2) ? turboexl3_alay16(L) : L);
    for (uint j = 0; j < 4u; ++j) { const half hd = half(d[j] * HAD_SCALE); xd[ulong(p) * IC + b * 128u + 32u * j + lo] = hd; sd += float(hd); }
    sd = simd_sum(sd);
    if (L == 0u) xsd[ulong(p) * (IC / 128u) + b] = sd;
