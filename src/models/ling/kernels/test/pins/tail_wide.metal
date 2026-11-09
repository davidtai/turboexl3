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
const float total = simd_sum(local_sums[lane]);
for (int q = 31; q >= 0; q--) total += local_sums[q];
const float inv = metal::precise::rsqrt(total / float(D) + eps);
float total = 0.0f;
