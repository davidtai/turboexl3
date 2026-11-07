// One member's DEC-ONE prepare of a normed row held in threadgroup memory: text_prep_s's statements (f16 round, suh,
// H128, lane order, the 128-block sums) with x read from the row `x` instead of device rows, into row `row` of the
// joined prepared rows `xp` and their block sums `xs`; simdgroup `b` of the row's threadgroup takes 128-block `b`.
template <typename TX, uint DIMC>
inline void turboexl3_ling_an_prep(threadgroup const TX* x, const device half* suh, device half* xp, device float* xs, const uint row, const uint b, const uint L) {
    constexpr uint NB = DIMC / 128u;
    constexpr uint LANE_ORDER = 1u;
    constexpr float HAD_SCALE = 0.08838834764831845f;
    float v[4];
    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(float(x[i]))) * float(suh[i]); }
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    float sum = 0.0f;
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const uint r = i & 15u;
        const uint q = (r & 7u) >> 1u;
        const uint m = (r & 1u) | ((r >> 3u) << 1u);
        const half hv = half(v[j] * HAD_SCALE);
        xp[row * DIMC + ((LANE_ORDER != 0) ? ((i & ~15u) + 4u * q + m) : i)] = hv;
        sum += float(hv);
    }
    sum = simd_sum(sum);
    if (L == 0u) xs[row * NB + b] = sum;
}
