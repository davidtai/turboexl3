    // the gate|up GEMMs' operands, op 0 the gate's and op 1 the up's
    const device uint* wsel = reinterpret_cast<const device uint*>(op == 0 ? w0 : w1);
    const device int8_t* xqsel = op == 0 ? xq0 : xq1;
    const device float* sxsel = op == 0 ? sx0 : sx1;
    const device float* xssel = op == 0 ? xs0 : xs1;
    device half* zsel = op == 0 ? z0 : z1;
