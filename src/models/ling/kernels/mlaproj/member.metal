    // the threadgroup's member, each in its own buffers (ling/mlaproj.zig): its prepared rows and their block sums (rows
    // m * ROWS.. of the joined prepare, the rows its own prep_sc writes), its trellis, svh and output, its K slices'
    // partial sums, and its arrival counters (member 1's after member 0's output blocks times the row chunks)
    const device half* xp = jxp + ulong(jm) * ulong(ROWSC) * ulong(INC);
    const device float* xs = jxs + ulong(jm) * ulong(ROWSC) * ulong(NB);
    const auto w = (jm == 0u) ? w0 : w1;
    const device half* svh = (jm == 0u) ? svh0 : svh1;
    device OUT_T* y = (jm == 0u) ? y0 : y1;
    const device uint* cnt = jcnt + ((jm == 0u) ? 0u : uint(OUT0) / 128u) * NCHC;
    device TURBOEXL3_COH float* zc = (device TURBOEXL3_COH float*)((jm == 0u) ? z0 : z1);
