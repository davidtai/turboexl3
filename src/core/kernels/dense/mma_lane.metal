    // the trellis lane whose eight codewords are lane L's B^T fragment (prefill.smallLanePlan, L' = b4 b2 b1 b3 b0): its
    // weights of columns fm, fm + 8 at k 2q + {0, 1, 8, 9}, in fragment element order
    const uint Lp = (L & 16u) | (((L >> 2u) & 1u) << 3u) | (((L >> 1u) & 1u) << 2u) | (((L >> 3u) & 1u) << 1u) | (L & 1u);
