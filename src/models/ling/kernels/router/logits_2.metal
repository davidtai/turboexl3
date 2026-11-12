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
        logits[(size_t)row * NE + out_row + tm] = static_cast<float>(result[tm]);
      }
    }
  }
