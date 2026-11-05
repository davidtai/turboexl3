    const device uint* wbase = (const device uint*)(proj == 0u ? gate_t : up_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(N / 2u);
