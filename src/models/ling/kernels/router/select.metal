uint row = thread_position_in_grid.y;
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;

threadgroup float rk[NE];
threadgroup float rw[NE];
threadgroup float gs[NG];
threadgroup uint sel[NK];

size_t rbase = (size_t)row * (size_t)NE;
for (uint e = tid; e < NE; e += TG) {
  float l = float(logits[rbase + e]);
  float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(l)));
  float sg = (l < 0.0f) ? sy : (1.0f - sy);
  // kolibri1 selects on the raw logit + bias; the others on sigmoid + bias.
  rk[e] = sg + float(bias[e]);
  rw[e] = sg;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Selection runs on simdgroup 0 only: K rounds of "max, then mask".
// simd_min over the candidate indices makes the tie-break the LOWEST
// expert id, so the pick is a pure function of the keys.
if (tid < 32) {
  for (uint g = lane; g < NG; g += 32) {
    float m1 = -INFINITY, m2 = -INFINITY;
    for (uint j = 0; j < GSZ; ++j) {
      float v = rk[g * GSZ + j];
      if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) { m2 = v; }
    }
    gs[g] = m1 + m2;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  uint keep_mask = 0u;
  for (uint r = 0; r < TOPKG; ++r) {
    float best = -INFINITY;
    uint bidx = 0xFFFFFFFFu;
    for (uint g = lane; g < NG; g += 32) {
      float v = gs[g];
      if (v > best) { best = v; bidx = g; }
    }
    float gmax = simd_max(best);
    uint cand = (best == gmax) ? bidx : 0xFFFFFFFFu;
    uint gidx = metal::min(simd_min(cand), uint(NG - 1));
    keep_mask |= (1u << gidx);
    if (lane == 0) gs[gidx] = -INFINITY;
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint e = lane; e < NE; e += 32) {
    if ((keep_mask & (1u << (e / GSZ))) == 0u) rk[e] = -1.0e4f;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

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

  float w = 0.0f;
  uint myidx = 0;
  if (lane < NK) {
    myidx = sel[lane];
    w = rw[myidx];
  }
  // MLX's `row_reduce_small` owns an 8-wide row in ONE thread and
  // folds it in ASCENDING order in the reduce's accumulator type —
  // which for a bf16 input is bf16, rounding after every add. A simd
  // butterfly over the same values is a different number.
  float tot = 0.0f;
  for (int j = 0; j < NK; ++j) tot += simd_shuffle(w, ushort(j));
  if (NORM != 0 && NK > 1) {
    w = w / (tot + 1e-20f);
  }
  // `scale` is a 0-dim array, which MLX binds as `const constant
  // float&` — a reference, not a subscriptable pointer.
  w = w * scale;
  if (lane < NK) {
    inds[(size_t)row * (size_t)NK + lane] = myidx;
    scores[(size_t)row * (size_t)NK + lane] = TOUT(w);
  }
}
