    // a KDA layer's three projections of one input prepared in one launch (Ling's q|k|v, f and g, ling/kdaproj.zig):
    // member threadgroup_position_in_grid.z scales x by its own suh and writes its rows of the launch's outputs (member
    // m's rows m * ROWS..), so each member's prepared rows are the ones its own prep_sc writes
    const uint jm = threadgroup_position_in_grid.z;
    const device half* suh = (jm == 0u) ? suh0 : ((jm == 1u) ? suh1 : suh2);
    device half* xp = jxp + ulong(jm) * ulong(uint(ROWS)) * ulong(uint(DIM));
    device float* xs = jxs + ulong(jm) * ulong(uint(ROWS)) * ulong(uint(DIM) / 128u);
