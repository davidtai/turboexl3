        d[j] = float(half(h)) * float(sd4[j]);
    }
    turboexl3_h128v4(d, L);
    const half4 hd = half4(half(d[0] * HAD_SCALE), half(d[1] * HAD_SCALE), half(d[2] * HAD_SCALE), half(d[3] * HAD_SCALE));
    *reinterpret_cast<device half4*>(xd + ulong(p) * IC + i0) = hd;
    const float sd = turboexl3_sum_x4<int(XSO)>(hd, L);
    if (L == 0u) xsd[ulong(p) * NBC + b] = sd;
