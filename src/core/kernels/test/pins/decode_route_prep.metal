  rk[e] = float(logits[rbase + e]);
      if (mine[i] > best) { best = mine[i]; bslot = i; }
    uint gidx = metal::min(simd_min(cand), uint(NE - 1));
    if (gidx % 32 == lane) mine[gidx / 32] = -INFINITY;
      if (e < NE) m = metal::max(m, rk[e]);
  float smax = simd_max(lane_max);
      acc += (e < NE) ? fast::exp(rk[e] - smax) : 0.0f;
  float snorm = 1.0f / simd_sum(lane_sum);
    w = float(TOUT(fast::exp(float(logits[rbase + myidx]) - smax) * snorm));
  for (int j = 0; j < NK; ++j) tot = float(TOUT(simd_shuffle(w, ushort(j)) + tot));
  w = w / float(TOUT(tot));
    scores[(size_t)row * (size_t)NK + lane] = TOUT(w);
const float xr = float(half(float(x[row * DIMC + i]))); vg[j] = xr * float(gate_suh[e * DIMC + i]); vu[j] = xr * float(up_suh[e * DIMC + i]);
        const half hg = half(vg[j] * HAD_SCALE);
    sgv = simd_sum(sgv);
if (lane == 0) { tsm[0] = smax; tsm[1] = snorm; }
  const uint e = sel[sgi];
  const uint rk = row * TOPKC + sgi;
threadgroup_barrier(mem_flags::mem_threadgroup);
gp[
gp[gs] = gres
simd_shuffle_down(gres, sn)
