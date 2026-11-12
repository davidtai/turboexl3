        }
        for (int r = 0; r < RG; ++r) {
          for (int tn = 0; tn < 4; tn++) {
            result[r][tm] += inter[tn] * v_coeff[r][tn];
          }
        }
        mat_offset += HK;
      }
      bn += 128;
    }
    for (int r = 0; r < RG; ++r) {
      for (int tm = 0; tm < 4; tm++) {
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
          result[r][tm] += simd_shuffle_down(result[r][tm], sn);
        }
      }
    }
    if (simdN == 0 && thrN == 0) {
      for (int r = 0; r < RG; ++r) {
        if (row0 + r < R) {
          for (int tm = 0; tm < 4; tm++) {
            logits[(size_t)(row0 + r) * NE + out_row + tm] = static_cast<float>(result[r][tm]);
          }
        }
      }
    }
  }
