    // two members of one input in one launch, each in its own buffers (Ling's pair, ling/pair.zig): the threadgroup's
    // output block belongs to member 0 below OB1 128-blocks, else member 1; the prepare scales x by that member's suh
    const uint jm = (ob < uint(OB1)) ? 0u : 1u;
    const device half* suh = (jm == 0u) ? suh0 : suh1;
