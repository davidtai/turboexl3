const device float*
const device half*
ling_combine_pick(
ling_combine_pick_h(
ling_tail_pick(loc[lk], s0, s1, s2, s3, s4, s5, s6, s7);
ling_combine_pick(loc[lk], s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23);
#pragma METAL fp contract(off)
const device float* W = WE;
const float eps = WE[D];
float tk = 0.0f;
acc = simd_sum(acc);
for (int q = NC * 4 - 1; q >= 0; q--) acc += hv[q] * hv[q];
acc += hv[p * 4 + i] * hv[p * 4 + i];
