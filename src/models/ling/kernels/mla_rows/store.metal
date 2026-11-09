  }
}
// MLX's store, each row of the chunk that x has
if (simdN == 0 && thrN == 0) {
  for (int r = 0; r < RB; r++) {
    if (r0 + r < ROWS) {
      for (int tm = 0; tm < TM; tm++) {
        y[size_t(r0 + r) * size_t(OUT) + size_t(out_row + tm)] = static_cast<bfloat16_t>(result[r][tm]);
      }
    }
  }
}
