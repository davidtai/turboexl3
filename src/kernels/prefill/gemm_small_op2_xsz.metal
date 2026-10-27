    if (DEFER != 0) exl3pf_affine(Dt, (op == 0 ? xs0 : xs1) + ulong(row0) * ulong(NBX), int(NBX), rows, L);
    device half* zp = (op == 0 ? z0 : z1) + ulong(row0) * ulong(NDC) + n0;
