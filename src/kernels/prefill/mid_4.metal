    }
    turboexl3_h128x4(g, L);
    turboexl3_h128x4(u, L);
    float d[4];
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const float gg = float(half(g[j] * HAD_SCALE * float(svh_g[ulong(e) * IC + i])));
        const float uu = float(half(u[j] * HAD_SCALE * float(svh_u[ulong(e) * IC + i])));
