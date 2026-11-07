    float result[4] = {0};
    bfloat16_t inter[4];
    float v_coeff[4];
    const int thrM = 0;
    const int thrN = int(simd_lid);
    const int simdM = int(simd_gid);
    const int simdN = 0;
    int bm = (simdM + thrM) * 4;
    int bn = (simdN + thrN) * 4;
    int out_row = int(ptg) * 16 + bm;
    if (out_row < NE) {
      out_row = out_row + 4 <= NE ? out_row : NE - 4;
      mat += size_t(out_row) * HK;
      for (int i = 0; i < HK / 128; ++i) {
        for (int tn = 0; tn < 4; tn++) {
          v_coeff[tn] = static_cast<float>(in_vec[bn + tn]);
        }
        int mat_offset = 0;
        for (int tm = 0; tm < 4; tm++) {
          for (int tn = 0; tn < 4; tn++) {
            inter[tn] = static_cast<bfloat16_t>(mat[mat_offset + bn + tn]);
          }
          for (int tn = 0; tn < 4; tn++) {
            result[tm] += inter[tn] * v_coeff[tn];
          }
          mat_offset += HK;
        }
        bn += 128;
      }
      for (int tm = 0; tm < 4; tm++) {
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
          result[tm] += simd_shuffle_down(result[tm], sn);
        }
      }
      if (simdN == 0 && thrN == 0) {
        for (int tm = 0; tm < 4; tm++) {
          const int e = out_row + tm;
          scores[e] = turboexl3_ling_score(static_cast<float>(static_cast<bfloat16_t>(result[tm])), pb[e]);
        }
      }
    }
  }
