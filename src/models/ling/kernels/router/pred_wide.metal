  if (R > 1) {
    // 2 to 8 rows: MLX's gemv_wide (gemv.h GemvWide::run: its route on M3 and later for 2 to 15 rows of a bf16 product
    // with b transposed, HK % 4 == 0, offsets and leading dimensions 4-aligned): passes (R + 4) / 5, k_lanes 32 at one
    // pass (R <= 5) and 16 at two (R 6 to 8; NE > 64), unroll 8. Each simdgroup sums its 32 / KL rows of pw against every
    // row of x, its statements in order. A row's sum does not depend on the threadgroup or the chunk of rows that takes
    // it, so one threadgroup here takes all R rows (MLX's passes split them over grid.x). HK % (KL * 32) == 0 (no tail
    // loop) and NE % (4 * 32 / KL) == 0 (no clamped rows): the host checks both.
    constexpr int KL = (R + 4) / 5 == 1 ? 32 : 16;
    constexpr int RPS = 32 / KL;
    const short k_lane = short(simd_lid % uint(KL));
    const short sg_row = short(simd_lid / uint(KL));
    const int out_row = int(ptg) * (RPS * 4) + RPS * int(simd_gid) + sg_row;
    const int prow = min(out_row, NE - 1);
    const device vec<bfloat16_t, 4>* w4 = (const device vec<bfloat16_t, 4>*)(pw + int64_t(prow) * HK);
    const int n_v4 = HK / 4;
    const int n_main = n_v4 - n_v4 % (KL * 8);
    const device vec<bfloat16_t, 4>* x4 = (const device vec<bfloat16_t, 4>*)x;
    const int x_ld4 = HK / 4;
    float result[R] = {0};
    for (int base = 0; base < n_main; base += KL * 8) {
      vec<float, 4> wf[8];
      for (int i = 0; i < 8; i++) {
        wf[i] = vec<float, 4>(w4[base + i * KL + k_lane]);
      }
      for (int v = 0; v < R; v++) {
        float acc = 0;
        for (int i = 0; i < 8; i++) {
          acc += dot(wf[i], vec<float, 4>(x4[v * x_ld4 + base + i * KL + k_lane]));
        }
        result[v] += acc;
      }
    }
    for (int v = 0; v < R; v++) {
      for (ushort off = KL / 2; off >= 1; off >>= 1) {
        result[v] += simd_shuffle_down(result[v], off);
      }
    }
    if (k_lane == 0 && out_row < NE) {
      for (int v = 0; v < R; v++) {
        scores[v * NE + out_row] = turboexl3_ling_score(static_cast<float>(static_cast<bfloat16_t>(result[v])), pb[out_row]);
      }
    }
  }
