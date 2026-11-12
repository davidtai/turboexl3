    // the down GEMM's operands (one op)
    const device uint* wsel = reinterpret_cast<const device uint*>(w0);
    const device int8_t* xqsel = xq0;
    const device float* sxsel = sx0;
    const device float* xssel = xs0;
    device half* zsel = z0;
