    const device uint* wb = reinterpret_cast<const device uint*>(op == 0 ? w0 : w0) + ulong(e) * ulong(K16 * N16) * ulong(NW) + ulong(n0 / 16) * ulong(NW);
    const device half* xa = (op == 0 ? x0 : x0) + ulong(row0) * ulong(KDC) + ((ALAY == 1) ? 8u : ((ALAY == 2) ? 4u : 2u)) * q;
