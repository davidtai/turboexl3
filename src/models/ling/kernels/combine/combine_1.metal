
    // one thread: row `row`'s C adjacent outputs from column `col`; the K pairs folded in k order
    constexpr uint C = 4;
    constexpr uint DC = (uint(D) + C - 1) / C;
    const uint g = thread_position_in_grid.x;
    const uint row = g / DC;
    const uint col = (g - row * DC) * C;
    const ulong obase = ulong(row) * ulong(D) + ulong(col);
    float t[C] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0; k < uint(K); ++k) {
        const float w = weights[ulong(row) * ulong(K) + ulong(k)];
