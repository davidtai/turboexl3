    uint lo = SC;
    if (tid < EC) {
        uint a = 0u, b = SC;
        while (a < b) { const uint mid = (a + b) >> 1u; if (uint(inds[order[mid]]) < tid) a = mid + 1u; else b = mid; }
        lo = a;
    }
