    // the next linear's prepare (dense prep_2_1's statement) on the staged Q_T row: lane L of simdgroup b takes elements
    // b 128 + L + 32 j, x's f16 rounding times suh; then prep_2_2's butterfly and prep_3's store (xp, natural order)
    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(ys[i])) * float(suh[i]); }
