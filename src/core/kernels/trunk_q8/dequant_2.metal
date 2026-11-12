    // trunk int8 (trunk_q8.zig): the tile's weights as int8 (exl3pf_q8: round((t - 510) / 4) of a codeword's byte sum t;
    // the GEMM's epilogue applies 4 A and 510 A + C), stored wq int8 [OUT, IN] (k fastest: the GEMM's B, a row of IN bytes
    // an output). Lane L (c = L >> 2, q = L & 3) holds the tile's weights at n c and c + 8, k 2q, 2q + 1 (pair 0) and
    // 2q + 8, 2q + 9 (pair 1): dequant_2.metal's places (turboexl3_row_of), two bytes a store
    const uint c = L >> 2u;
    const uint q = L & 3u;
    const char2 p0 = char2(exl3pf_q8(exl3_tile_cw<N, WMASK, 0, 0>(ww, lp)), exl3pf_q8(exl3_tile_cw<N, WMASK, 1, 0>(ww, lp)));
    const char2 p1 = char2(exl3pf_q8(exl3_tile_cw<N, WMASK, 2, 0>(ww, lp)), exl3pf_q8(exl3_tile_cw<N, WMASK, 3, 0>(ww, lp)));
    const char2 p2 = char2(exl3pf_q8(exl3_tile_cw<N, WMASK, 4, 0>(ww, lp)), exl3pf_q8(exl3_tile_cw<N, WMASK, 5, 0>(ww, lp)));
    const char2 p3 = char2(exl3pf_q8(exl3_tile_cw<N, WMASK, 6, 0>(ww, lp)), exl3pf_q8(exl3_tile_cw<N, WMASK, 7, 0>(ww, lp)));
    device int8_t* d = wq + ulong(tn * 16u + c) * ulong(INC) + ulong(tk * 16u + 2u * q);
    *reinterpret_cast<device char2*>(d) = p0;
    *reinterpret_cast<device char2*>(d + 8u) = p1;
    *reinterpret_cast<device char2*>(d + 8ul * ulong(INC)) = p2;
    *reinterpret_cast<device char2*>(d + 8ul * ulong(INC) + 8u) = p3;
