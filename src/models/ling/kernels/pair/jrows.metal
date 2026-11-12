    // two members of one input in one launch, each in its own buffers (Ling's pair, ling/pair.zig): the threadgroup's
    // output block belongs to member 0 below OB1 128-blocks, else member 1; it reads that member's prepared rows (rows
    // m * ROWS.. of the pair's prepare), which are the rows the member's own prep_sc writes
    const uint jm = (ob < uint(OB1)) ? 0u : 1u;
    const device half* xp = jxp + ulong(jm) * ulong(ROWSC) * ulong(INC);
    const device float* xs = jxs + ulong(jm) * ulong(ROWSC) * ulong(NB);
