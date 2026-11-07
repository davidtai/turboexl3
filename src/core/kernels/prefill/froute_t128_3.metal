    if (tid == 1023u) { totq_s = sq + nq; totb_s = sb + nb; tots_s = ss + ns; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = totq_s + tid; i < TQ; i += 1024u) { tiles128[4u * i] = 0; tiles128[4u * i + 1u] = 0; tiles128[4u * i + 2u] = 0; tiles128[4u * i + 3u] = 0; }
    for (uint i = totb_s + tid; i < TBIG; i += 1024u) { tiles[4u * i] = 0; tiles[4u * i + 1u] = 0; tiles[4u * i + 2u] = 0; tiles[4u * i + 3u] = 0; }
    for (uint i = TBIG + tots_s + tid; i < TALL; i += 1024u) { tiles[4u * i] = 0; tiles[4u * i + 1u] = 0; tiles[4u * i + 2u] = 0; tiles[4u * i + 3u] = 0; }
