    for (int sgn = BN - 1; sgn >= 1; sgn--) {
      for (int r = 0; r < RB; r++) {
        for (int tm = 0; tm < TM; tm++) {
          result[r][tm] += tgp_memory[(r * BN + sgn) * (blockM + TM) + bm + tm];
        }
      }
    }
