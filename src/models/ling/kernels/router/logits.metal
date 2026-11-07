  // MLX's gemv_t (gemv.h GEMVTKernel::run at BM 1, BN 4, SM 8, SN 4, TM 4, TN 4: its route for x [1, HK] times the f32
  // router table w [HK, NE] row-major, HK < 8192, 512 <= NE < 2048), threadgroup blk of row's NE / 64, its statements in
  // order. MLX's matmul first widens x to f32 (an astype); here x is read as bf16 and widened in registers, the same
  // values. HK % 32 == 0 (no leftover block) and NE % 64 == 0 (no tail tile): the host checks both.
  const device bfloat16_t* in_vec = x + (size_t)row * (size_t)HK;
  float result[4] = {0};
  float inter[4];
  float v_coeff[4];
  const int thrM = int(simd_lid) / 4;
  const int thrN = int(simd_lid) % 4;
  const int sgM = int(simd_gid) / 4;
  const int sgN = int(simd_gid) % 4;
  const int simdM = 8 * sgM;
  const int simdN = 4 * sgN;
  int cm = (simdM + thrM);
  int cn = (simdN + thrN);
  int bm = cm * 4;
  int bn = cn * 4;
  int out_col = int(blk) * 64 + bn;
  if (out_col < NE) {
    out_col = out_col + 4 < NE ? out_col : NE - 4;
    for (int i = 0; i < HK / 32; ++i) {
      threadgroup_barrier(mem_flags::mem_none);
      for (int tm = 0; tm < 4; tm++) {
        v_coeff[tm] = static_cast<float>(in_vec[bm + tm]);
      }
      for (int tm = 0; tm < 4; tm++) {
        auto vc = static_cast<float>(v_coeff[tm]);
        for (int tn = 0; tn < 4; tn++) {
          inter[tn] = w[size_t(bm + tm) * NE + out_col + tn];
        }
        for (int tn = 0; tn < 4; tn++) {
          result[tn] += vc * inter[tn];
        }
      }
      bm += 32;
    }
  }
  for (int tn = 0; tn < 4; tn++) {
    for (ushort sm = 4; sm >= 1; sm >>= 1) {
      result[tn] += simd_shuffle_down(result[tn], 4 * sm);
    }
  }
  if (cm == 0 && out_col < NE) {
    for (int j = 0; j < 4; j++) {
      logits[(size_t)row * NE + out_col + j] = static_cast<float>(result[j]);
    }
  }
