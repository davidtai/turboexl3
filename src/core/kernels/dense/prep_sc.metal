    // DEC-ONE: the arrival counters of the GEMV launch that follows, one per (output block, row chunk)
    for (uint g = (row * NB + b) * 32u + L; g < uint(NCNT); g += uint(NTH)) cnt[g] = 0u;
