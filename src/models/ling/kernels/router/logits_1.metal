  // MLX's gemv (gemv.h GEMVKernel::run at BM 4, BN 1, SM 1, SN 32, TM 4, TN 4) over the router's table as the host's
  // loader builds it: [HK, NE] column-major (strides 1, HK: the astype and the + 0 of the weight's transposed view keep
  // its layout), which MLX's matmul reads as b transposed, its memory w [NE, HK] row-major (NE < 4096, 64 < HK < 16 NE).
  // Threadgroup blk of row's NE / 16, its statements in order. MLX's matmul first widens x to f32 (an astype); here x
  // is read as bf16 and widened in registers, the same values. HK % 128 == 0 (no leftover block) and NE % 16 == 0 (no
  // tail rows): the host checks both.
  const device bfloat16_t* in_vec = x + (size_t)row * (size_t)HK;
  float result[4] = {0};
  float inter[4];
  float v_coeff[4];
  const int thrM = 0;
  const int thrN = int(simd_lid);
  const int simdM = int(simd_gid);
  const int simdN = 0;
  int bm = (simdM + thrM) * 4;
  int bn = (simdN + thrN) * 4;
  int out_row = int(blk) * 16 + bm;
  if (out_row < NE) {
    out_row = out_row + 4 <= NE ? out_row : NE - 4;
    auto mat = w + size_t(out_row) * HK;
    for (int i = 0; i < HK / 128; ++i) {
      for (int tn = 0; tn < 4; tn++) {
        v_coeff[tn] = static_cast<float>(in_vec[bn + tn]);
      }
      int mat_offset = 0;
      for (int tm = 0; tm < 4; tm++) {
        for (int tn = 0; tn < 4; tn++) {
