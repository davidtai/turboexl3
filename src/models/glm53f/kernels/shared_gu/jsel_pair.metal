    // two members of one input x as two trellises, each where it lies (member 0's [IN/16, OB1 8, NHW] in w0, member 1's
    // [IN/16, OUT/16 - OB1 8, NHW] in w1; their svh joined along OUT): the threadgroup's output block belongs to member 0
    // below OB1 128-blocks, else member 1; the prepare scales x by that member's suh, and the GEMV reads that member's
    // trellis (`jtile.metal` from the trellis's start): its own count of output tiles, the block's tile counted from the
    // member's first
    const device half* suh = (ob < uint(OB1)) ? suh0 : suh1;
    const auto w = (ob < uint(OB1)) ? w0 : w1;
    const uint mo0 = (ob < uint(OB1)) ? 0u : uint(OB1) * 8u;
    const uint mtiles = (ob < uint(OB1)) ? uint(OB1) * 8u : OUT_TILES - uint(OB1) * 8u;
    const ulong mbase = 0ul;
