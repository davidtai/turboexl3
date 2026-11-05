    // two members joined end to end (member 0's whole trellis [IN/16, OB1 8, NHW], then member 1's; their svh joined
    // along OUT, one input x): the threadgroup's output block belongs to member 0 below OB1 128-blocks, else member 1;
    // the prepare scales x by that member's suh, and the GEMV reads that member's trellis (`jtile.metal`): its base, its
    // own count of output tiles, the block's tile counted from the member's first
    const device half* suh = (ob < uint(OB1)) ? suh0 : suh1;
    const uint mo0 = (ob < uint(OB1)) ? 0u : uint(OB1) * 8u;
    const uint mtiles = (ob < uint(OB1)) ? uint(OB1) * 8u : OUT_TILES - uint(OB1) * 8u;
    const ulong mbase = ulong(IN_TILES) * ulong(mo0) * ulong(NW);
