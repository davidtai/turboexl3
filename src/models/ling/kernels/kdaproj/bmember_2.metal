        const device bfloat16_t* in_vec = xb + size_t(row) * size_t(IN);
        float total[4];
        for (int sgN = 0; sgN < 8; ++sgN) {
          float result[4] = {0};
          bfloat16_t inter[4];
          float v_coeff[4];
          const int simdN = 32 * sgN;
          int bn = (simdN + thrN) * 4;
          for (int i = 0; i < IN / 1024; ++i) {
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
              mat_offset += IN;
            }
            bn += 1024;
          }
          for (int tm = 0; tm < 4; tm++) {
            for (ushort sn = 16; sn >= 1; sn >>= 1) {
              result[tm] += simd_shuffle_down(result[tm], sn);
            }
          }
          for (int tm = 0; tm < 4; tm++) {
            total[tm] = (sgN == 0) ? result[tm] : (total[tm] + result[tm]);
          }
        }
        if (thrN == 0) {
          for (int tm = 0; tm < 4; tm++) {
            yb[size_t(row) * size_t(OUTB) + size_t(out_row + tm)] = static_cast<bfloat16_t>(total[tm]);
          }
        }
      }
      return;
    }
