turboexl3_alay(uint j) { return 8u * ((j >> 1u) & 3u) + 4u * (j >> 4u) + ((j & 1u) | (((j >> 3u) & 1u) << 1u)); }
turboexl3_alay16(uint j) { return (j & 16u) + 4u * ((j >> 1u) & 3u) + ((j & 1u) | (((j >> 3u) & 1u) << 1u)); }
(ALAY == 1) ? turboexl3_alay(L) : ((ALAY == 2) ? turboexl3_alay16(L) : L)
((ALAY == 1) ? 8u : ((ALAY == 2) ? 4u : 2u))
