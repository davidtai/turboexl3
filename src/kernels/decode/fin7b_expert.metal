    const uint e = uint(slots[rk]) & 0xFFFFFFu;   // the row-slot's slot row: its id is (bank << 24 | slot row)
    const uint bk = uint(slots[rk]) >> 24u;       // its bank (0, 1 or 2), whose arrays the row-slot reads
    const device half* gate_svh = bk == 0u ? gate_svh0 : (bk == 1u ? gate_svh1 : gate_svh2);
    const device half* up_svh = bk == 0u ? up_svh0 : (bk == 1u ? up_svh1 : up_svh2);
    const device half* down_suh = bk == 0u ? down_suh0 : (bk == 1u ? down_suh1 : down_suh2);
