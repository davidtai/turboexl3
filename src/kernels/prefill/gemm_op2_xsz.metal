            if (DEFER != 0) exl3pf_affine(Dt, (op == 0 ? xs0 : xs1) + ulong(row0 + tm) * ulong(NBX), int(NBX), sgp, L);
            device half* zp = (op == 0 ? z0 : z1) + ulong(row0 + tm) * ulong(NDC) + (n0g + cb * BNC) + tn;
