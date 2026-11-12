        // dec_finish_store's value, stored in its member's own output (member 0's y0 [ROWS, OB1 128], member 1's y1
        // [ROWS, OUT - OB1 128]), each row-contiguous as the member's `linear` stores it
        for (uint j = 0; j < 4u; ++j) { const uint o = ob * 128u + L + 32u * j; const OUT_T yv = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); if (ob < uint(OB1)) y0[row * (uint(OB1) * 128u) + o] = yv; else y1[row * (OUTC - uint(OB1) * 128u) + (o - uint(OB1) * 128u)] = yv; }
    }
