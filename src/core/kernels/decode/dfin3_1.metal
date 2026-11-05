    constexpr uint HIDC = uint(HID);
    constexpr uint TOPKC = uint(TOPK);
    constexpr uint KSC = uint(KS);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint blk = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    threadgroup float xch[128];
    const uint o = blk * 128u + tid;
    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    float acc = 0.0f;
    for (uint k = 0; k < TOPKC; ++k) {
        const uint rk = row * TOPKC + k;
