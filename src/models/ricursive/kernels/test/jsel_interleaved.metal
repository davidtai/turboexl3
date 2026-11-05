    // two members joined along OUT (their trellises and svh concatenated, one input x): the threadgroup's output block
    // belongs to member 0 below OB1 128-blocks, else member 1; the prepare scales x by that member's suh
    const device half* suh = (ob < uint(OB1)) ? suh0 : suh1;
