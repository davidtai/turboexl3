  // router/logits_1's statements (MLX's gemv, gemv.h GEMVKernel::run at BM 4, BN 1, SM 1, SN 32, TM 4, TN 4, over the
  // table's memory [NE, HK] row-major) for the group's RG rows at once: each row keeps its own coefficients and sums, and
  // each table word read below is taken by every row of the group in the row's own statement. A row's statements and
  // their order are those of its own threadgroup in router/head's launch, so its logits are that launch's words. A group
  // past the call's last row repeats that row and stores nothing for it.
  const int row0 = int(grp) * RG;
  float result[RG][4];
  for (int r = 0; r < RG; ++r) {
    for (int tm = 0; tm < 4; tm++) {
      result[r][tm] = 0;
    }
  }
  float inter[4];
  float v_coeff[RG][4];
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
      for (int r = 0; r < RG; ++r) {
        const device bfloat16_t* in_vec = x + (size_t)min(row0 + r, R - 1) * (size_t)HK;
        for (int tn = 0; tn < 4; tn++) {
          v_coeff[r][tn] = static_cast<float>(in_vec[bn + tn]);
        }
      }
      int mat_offset = 0;
      for (int tm = 0; tm < 4; tm++) {
        for (int tn = 0; tn < 4; tn++) {
