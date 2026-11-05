    const device uint* wb = reinterpret_cast<const device uint*>(op == 0 ? w0 : w1) + (ulong(e) * ulong(K16 * N16) + ulong(n0g / 16)) * ulong(N / 2u);
    const device half* xa = (op == 0 ? x0 : x1) + ulong(row0 + tm) * ulong(KDC);
