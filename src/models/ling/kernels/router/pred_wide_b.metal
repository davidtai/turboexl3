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
