    // DEC-ONE: the members' arrival counters zeroed (prep_sc's loop, over this launch's threads)
    {
        constexpr uint NB = uint(D) / 128u;
        const uint row = r;
        const uint b = sg;
        const uint L = lane;
        for (uint g = (row * NB + b) * 32u + L; g < uint(NCNT); g += uint(NTH)) cnt[g] = 0u;
    }
