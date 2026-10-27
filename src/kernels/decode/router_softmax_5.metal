  float w = 0.0f;
  uint myidx = 0;
  if (lane < NK) {
    myidx = sel[lane];
    w = float(TOUT(fast::exp(float(logits[rbase + myidx]) - smax) * snorm));
  }
  // MLX's `row_reduce_small` owns an 8-wide row in ONE thread and
  // folds it in ASCENDING order in the reduce's accumulator type —
  // which for a bf16 input is bf16, rounding after every add. A simd
  // butterfly over the same values is a different number.
  float tot = 0.0f;
  for (int j = 0; j < NK; ++j) tot = float(TOUT(simd_shuffle(w, ushort(j)) + tot));
  w = w / float(TOUT(tot));
  if (lane < NK) {
    inds[(size_t)row * (size_t)NK + lane] = myidx;
    scores[(size_t)row * (size_t)NK + lane] = TOUT(w);
  }
