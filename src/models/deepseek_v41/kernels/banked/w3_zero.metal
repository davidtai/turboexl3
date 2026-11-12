    // wave3: the arrival counters of the call's two fused launches (one a finish unit: the gate|up units', then the down
    // units'), zeroed before the row-slot's skip line so a skipped row-slot's threadgroups zero their share too
    for (uint g = (rk * (DIMC / 128u) + b) * 32u + L; g < uint(NCNT); g += uint(NTH)) cnt[g] = 0u;
