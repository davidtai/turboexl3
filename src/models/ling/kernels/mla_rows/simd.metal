      }
    }
  }
  bn += blockN;
}
// MLX's simdgroup sum, each row's accumulators on their own
for (int r = 0; r < RB; r++) {
  for (int tm = 0; tm < TM; tm++) {
    for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
      result[r][tm] += simd_shuffle_down(result[r][tm], sn);
    }
  }
}
// MLX's threadgroup sum (BN > 1): each simdgroup's sums into threadgroup memory, then simdgroup 0 adds the others'
if (BN > 1) {
  if (thrN == 0) {
    for (int r = 0; r < RB; r++) {
      for (int tm = 0; tm < TM; tm++) {
        tgp_memory[(r * BN + sgN) * (blockM + TM) + bm + tm] = result[r][tm];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thrN == 0 && sgN == 0) {
