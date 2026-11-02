        const uint e = uint(slots[rk]) & 0xFFFFFFu;   // the row-slot's slot row: its id is (bank << 24 | slot row)
        const uint bk = uint(slots[rk]) >> 24u;       // its bank (0, 1 or 2), whose down svh the row-slot reads
        const device half* down_svh = bk == 0u ? down_svh0 : (bk == 1u ? down_svh1 : down_svh2);
