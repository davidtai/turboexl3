    const uint bk = e >> 24u;   // e is the row-slot's id (bank << 24 | slot row), as the members compare it
    const device uint* wbase = (const device uint*)(bk == 0u ? down_t0 : (bk == 1u ? down_t1 : down_t2)) + ulong(e & 0xFFFFFFu) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(N / 2u);
