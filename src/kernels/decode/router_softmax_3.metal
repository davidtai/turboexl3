  // The lane's strided slice of the keys lives in registers for the K rounds: a round is
  // a register max, two simd reductions and a register mask, no threadgroup traffic.
  constexpr uint EPL = (NE + 31) / 32;
  float mine[EPL];
  for (uint i = 0; i < EPL; ++i) {
    uint e = lane + 32 * i;
    mine[i] = (e < NE) ? rk[e] : -INFINITY;
  }
  for (uint r = 0; r < NK; ++r) {
    float best = -INFINITY;
    uint bslot = 0xFFFFFFFFu;
    for (uint i = 0; i < EPL; ++i) {
      if (mine[i] > best) { best = mine[i]; bslot = i; }
    }
    float gmax = simd_max(best);
    uint cand = (best == gmax) ? lane + 32 * bslot : 0xFFFFFFFFu;
    uint gidx = metal::min(simd_min(cand), uint(NE - 1));
    if (gidx % 32 == lane) mine[gidx / 32] = -INFINITY;
    if (lane == 0) sel[r] = gidx;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
