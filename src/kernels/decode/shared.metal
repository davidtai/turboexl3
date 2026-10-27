// one K slice of dense.zig's text_dgemv for RS rows: input tiles [tk0, tk0 + PER) of trellis wbase (OUT_TILES output
// tiles), BT output tiles from blk * BT, row r's prepared input at x0 + r * XSTRIDE (lane order, offset 4 (L & 3), element
// 0 = tile tk0); every weight decoded once and applied to each row in tile order, folded into red_sg[r * 16 BT + output]
template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT, uint RS, uint DEF, uint PER, uint XSTRIDE, typename XP>
inline void turboexl3_sdgemv(const device uint* wbase, XP x0, uint tk0, uint blk, uint L, threadgroup float* red_sg) {
  constexpr uint NW = N / 2u;
  const uint qb = exl3_gen_qb<N>(L);
  const uint ga = exl3_gen_a<N>(L);
  const uint lp = exl3_gen_lp<N>(L);
  float a[RS][BT][2];
  for (uint r = 0; r < RS; ++r) for (uint t = 0; t < BT; ++t) { a[r][t][0] = 0.0f; a[r][t][1] = 0.0f; }
  for (uint i = 0; i < PER; ++i) {
    const uint tk = tk0 + i;
    float xr[RS][4];
    for (uint r = 0; r < RS; ++r) {
      const half4 x4 = turboexl3_x4(x0 + r * XSTRIDE + i * 16u);
      xr[r][0] = float(x4.x); xr[r][1] = float(x4.y); xr[r][2] = float(x4.z); xr[r][3] = float(x4.w);
    }
    for (uint t = 0; t < BT; ++t) {
      const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * BT + t) * ulong(NW);
      const auto ww = exl3_tile_load<N, 0>(tile, qb, ga, lp);
      uint cw[8];
      cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
      cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
      for (uint m = 0; m < 8u; ++m) {
        const float wv = (DEF != 0u) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
        for (uint r = 0; r < RS; ++r) a[r][t][m >> 2] = fma(xr[r][m & 3u], wv, a[r][t][m >> 2]);
      }
    }
  }
  for (uint r = 0; r < RS; ++r) turboexl3_fold<BT>(a[r], red_sg + r * 16u * BT, L);
}
// dense.zig's text_prep_s for one 128-block b of one row, v[q] = float(half(x[b 128 + L + 32 q])) on entry: * suh, H128,
// f16; writes the lane-order block to dst (device or threadgroup) and returns the simdgroup's sum of the rounded values
template <typename SU, typename DP>
inline float turboexl3_sprep(thread float (&v)[4], SU suh, uint b, uint L, DP dst) {
  constexpr float HAD_SCALE = 0.08838834764831845f;
  for (uint q = 0; q < 4u; ++q) v[q] = v[q] * float(suh[b * 128u + L + 32u * q]);
  turboexl3_h128r(v, L);
  float sum = 0.0f;
  for (uint q = 0; q < 4u; ++q) {
    const uint i = L + 32u * q;
    const uint r = i & 15u; const uint qq = (r & 7u) >> 1u; const uint mm = (r & 1u) | ((r >> 3u) << 1u);
    const half hv = half(v[q] * HAD_SCALE);
    dst[(i & ~15u) + 4u * qq + mm] = hv;
    sum += float(hv);
  }
  return simd_sum(sum);
}
// dense.zig's text_dfinish for one 128-block of one row: the KS planes z[(k RS + row) DIM + o] summed in slice order, the
// deferred MUL1 map with the row's input sum xsum, f16, H128, svh, f16 (the caller converts to the output dtype)
template <uint KS, uint RS, uint DIM, uint DEF, typename SV>
inline void turboexl3_sfinish(const device float* z, SV svh, float xsum, uint row, uint b, uint L, thread float (&v)[4]) {
  constexpr float HAD_SCALE = 0.08838834764831845f;
  const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
  const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
  for (uint q = 0; q < 4u; ++q) {
    const uint o = b * 128u + L + 32u * q;
    float acc = z[ulong(row) * ulong(DIM) + o];
    for (uint k = 1u; k < KS; ++k) acc += z[(ulong(k) * ulong(RS) + row) * ulong(DIM) + o];
    if (DEF != 0u) acc = fma(acc, MUL1_A, MUL1_C * xsum);
    v[q] = float(half(acc));
  }
  turboexl3_h128r(v, L);
  for (uint q = 0; q < 4u; ++q) v[q] = float(half(v[q] * HAD_SCALE * float(svh[b * 128u + L + 32u * q])));
}
