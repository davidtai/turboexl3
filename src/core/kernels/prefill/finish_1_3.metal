    const uint L = thread_index_in_simdgroup;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0; k < TOPKC; ++k) {
        const uint src = r * TOPKC + k;
        const uint p = inv[src];
        const uint e = uint(inds[src]);
