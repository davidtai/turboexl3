// mlx-serve's add_norm (the plain arm) of the residual H and the mixer's output XD with the norm's weight NW, for every
// row this threadgroup reads x of: the router's threadgroups their row, the predictor's every row. The add_norm runs
// HK / 4 threads a row (32 simdgroups at HK 4096), four columns a thread: each thread's sum of squares (acc), each
// simdgroup's sum of those (simd_sum), then the sum of the 32-slot table of the simdgroups' sums (simd_sum). Here each
// of the 4 simdgroups runs 8 of the add_norm's simdgroups in turn, lane l as that simdgroup's lane l, so each sum is the
// add_norm's word; a slot past the add_norm's simdgroups holds 0, as its table does. The router's threadgroups store
// their columns of the residual (H_NEW) and the normed rows (HN).
threadgroup float an_part[32];
threadgroup float an_inv[R];
{
  const bool an_router = tg < uint(R * (NE / 16));
  const uint an_lo = an_router ? tg / uint(NE / 16) : 0u;
  const uint an_hi = an_router ? an_lo + 1u : uint(R);
  for (uint ar = an_lo; ar < an_hi; ++ar) {
    for (uint v = 0; v < 8u; ++v) {
      const uint asg = simd_gid * 8u + v;
      const uint t = asg * 32u + simd_lid;
      float acc = 0.0f;
      for (int i = 0; i < 4; i++) {
        const int c = int(t) * 4 + i;
        float hv = 0.0f;
        if (c < HK) {
          const size_t at = size_t(ar) * HK + c;
          const bfloat16_t hn = bfloat16_t(float(H[at]) + float(XD[at]));
          hv = float(hn);
        }
        acc += hv * hv;
      }
      acc = simd_sum(acc);
      if (simd_lid == 0) an_part[asg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = simd_sum(an_part[simd_lid]);
    if (simd_gid == 0 && simd_lid == 0) an_inv[ar] = metal::precise::rsqrt(total / float(HK) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (an_router) {
    const uint ab = tg % uint(NE / 16);
    for (uint c = ab * 128u + simd_gid * 32u + simd_lid; c < uint(HK); c += uint(NE / 16) * 128u) {
      const size_t at = size_t(an_lo) * HK + c;
      const bfloat16_t hn = bfloat16_t(float(H[at]) + float(XD[at]));
      H_NEW[at] = hn;
      HN[at] = bfloat16_t(float(NW[c]) * float(bfloat16_t(float(hn) * an_inv[an_lo])));
    }
  }
}
