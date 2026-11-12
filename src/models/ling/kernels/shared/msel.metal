    // Ling's shared expert: its gate (member 0) and its up (member 1), one input x, in one DEC-ONE launch, each member in
    // its own buffers (no joined copy). The threadgroup's output block belongs to the gate below OB1 128-blocks, else to
    // the up: the prepare scales x by that member's suh, the GEMV reads that member's trellis (`mtile.metal`: its own
    // base, its own count of output tiles, the block's tile counted from the member's first) and the finish its svh. y
    // holds both members' outputs side by side (the gate's columns, then the up's), read back by the pair's second finish.
    const bool m1 = ob >= uint(OB1);
    const device half* suh = m1 ? suh1 : suh0;
    const device half* svhm = m1 ? svh1 : svh0;
    const uint mo0 = m1 ? uint(OB1) * 8u : 0u;
    const uint mtiles = m1 ? OUT_TILES - uint(OB1) * 8u : uint(OB1) * 8u;
    const device uint* wm = (const device uint*)(m1 ? w1 : w);
    device TURBOEXL3_COH OUT_T* yc = (device TURBOEXL3_COH OUT_T*)y;
