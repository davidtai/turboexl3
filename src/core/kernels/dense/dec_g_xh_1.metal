    const device half* g_xb = xp + 4u * (L & 3u) + tk0 * 16u;   // lane order: the quad's four rows of each tile
