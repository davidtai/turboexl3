    constexpr uint DIMC = uint(DIM);
    constexpr uint TOPKC = uint(TOPK);
    const uint d = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    float acc = 0.0f;
    for (uint k = 0; k < TOPKC; ++k) acc = fma(scores[r * TOPKC + k], yk[(r * TOPKC + k) * DIMC + d], acc);
    y[r * DIMC + d] = acc;
