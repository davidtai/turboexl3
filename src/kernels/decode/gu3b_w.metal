    const uint bk = e >> 24u;   // e is the row-slot's id (bank << 24 | slot row), as the members compare it
    const device uint* wbase = (const device uint*)(proj == 0u ? (bk == 0u ? gate_t0 : (bk == 1u ? gate_t1 : gate_t2)) : (bk == 0u ? up_t0 : (bk == 1u ? up_t1 : up_t2))) + ulong(e & 0xFFFFFFu) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(N / 2u);
