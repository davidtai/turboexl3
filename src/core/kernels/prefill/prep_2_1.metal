    const uint e = uint(inds[src]);
    if (b == 0u && L == 0u) inv[src] = p;
    float vg[4], vu[4];
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const float xv = float(half(float(x[ulong(r) * HC + i])));
        vg[j] = xv * float(suh_g[ulong(e) * HC + i]);
        vu[j] = xv * float(suh_u[ulong(e) * HC + i]);
    }
    turboexl3_h128x4(vg, L);
    turboexl3_h128x4(vu, L);
