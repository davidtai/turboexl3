    uint members[2]; uint m;
    if (!turboexl3_members<GRP, uint(RKC)>(slots, rk, e, L, members, m)) return;
    threadgroup float red[SPTC][4][2u * 16u * BTC];   // one reduction buffer per slice: a single barrier for all
