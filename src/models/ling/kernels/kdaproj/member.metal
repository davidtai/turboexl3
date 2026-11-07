    // the threadgroup's member, each in its own buffers (ling/kdaproj.zig): its prepared rows and their block sums (rows
    // m * ROWS.. of the joined prepare, the rows its own prep_sc writes), its trellis, svh and output, its K slices'
    // partial sums, and its arrival counters (after the members before it: their output blocks times the row chunks)
    const device half* xp = jxp + ulong(jm) * ulong(ROWSC) * ulong(INC);
    const device float* xs = jxs + ulong(jm) * ulong(ROWSC) * ulong(NB);
    const auto w = (jm == 0u) ? w0 : ((jm == 1u) ? w1 : w2);
    const device half* svh = (jm == 0u) ? svh0 : ((jm == 1u) ? svh1 : svh2);
    device OUT_T* y = (jm == 0u) ? y0 : ((jm == 1u) ? y1 : y2);
    const device uint* cnt = jcnt + ((jm == 0u) ? 0u : ((jm == 1u) ? uint(OUT0) / 128u : (uint(OUT0) + uint(OUT1)) / 128u)) * NCHC;
    device TURBOEXL3_COH float* zc = (device TURBOEXL3_COH float*)((jm == 0u) ? z0 : ((jm == 1u) ? z1 : z2));
