    constexpr uint HIDC = uint(HID);
    constexpr uint TOPKC = uint(TOPK);
    const uint o = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    float acc = 0.0f;
    for (uint k = 0; k < TOPKC; ++k) { const uint rk = row * TOPKC + k; acc = fma(float(scores[rk]), float(wv[ulong(rk) * ulong(HIDC) + o]), acc); }
    y[ulong(row) * ulong(HIDC) + o] = OUT_T(acc);
