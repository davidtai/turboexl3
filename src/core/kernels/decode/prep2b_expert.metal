    const uint e = uint(slots[rk]) & 0xFFFFFFu;   // the row-slot's slot row: its id is (bank << 24 | slot row)
    const uint bk = uint(slots[rk]) >> 24u;       // its bank (0, 1 or 2), whose arrays the row-slot reads
    const device half* gate_suh = bk == 0u ? gate_suh0 : (bk == 1u ? gate_suh1 : gate_suh2);
    const device half* up_suh = bk == 0u ? up_suh0 : (bk == 1u ? up_suh1 : up_suh2);
