    }
    for (uint t = 0u; t < nb; ++t) {
        const uint i = sb + t;
        tiles[4u * i] = int(tid); tiles[4u * i + 1u] = int(base + t * BMC); tiles[4u * i + 2u] = int(min(BMC, cnt - t * BMC)); tiles[4u * i + 3u] = 0;
    }
    for (uint t = 0u; t < ns; ++t) {
        const uint i = TBIG + ss + t;
        tiles[4u * i] = int(tid); tiles[4u * i + 1u] = int(base + nfull * BMC + 32u * t); tiles[4u * i + 2u] = int(min(32u, r - 32u * t)); tiles[4u * i + 3u] = 0;
    }
