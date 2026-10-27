// lane L's eight weights of a tile: code positions 8L + m, m < 4 at column L / 4, m >= 4 at column L / 4 + 8,
// rows 2 (L & 3) + {0, 1, 8, 9}[m & 3] (quantize.py's tensor-core layout)
inline uint turboexl3_row_of(uint L, uint m) { const uint q = m & 3u; return 2u * (L & 3u) + ((q & 1u) ? 1u : 0u) + ((q & 2u) ? 8u : 0u); }
