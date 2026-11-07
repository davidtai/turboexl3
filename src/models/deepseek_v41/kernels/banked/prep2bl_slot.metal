    const uint sid = uint(lut[uint(idx[rk])]);   // the row-slot's packed id: its routed expert through the layer's resident LUT
    if (b == 0u && L == 0u) slots[rk] = sid;      // the ids the gate|up's GEMV and finish and the down read, one write a row-slot
