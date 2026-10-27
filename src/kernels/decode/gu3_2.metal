    const uint kz = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    constexpr uint GRP = uint(GROUP);   // row-slots per lead threadgroup (1 or 2)
    const uint e = uint(slots[rk]);
    const uint proj = sg >> 1u;
    const uint khalf = sg & 1u;
    uint members[2]; uint m;
    if (!turboexl3_members<GRP, uint(RKC)>(slots, rk, e, L, members, m)) return;
    threadgroup float red[SPTC][4][2u * 16u * BTC];   // one reduction buffer per slice: a single barrier for all
