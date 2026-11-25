// The x GEMV (`decode.text_gu3x`, `text_down3x`; `Config.decode_iop`): turboexl3_gemv_members' reads, decodes and FMAs
// in its order, so its bytes, with each tile's words indexed from the expert base by 32-bit tile offsets and, at integer
// rates, the 64-bit window landed without the variable funnel (K % 4 == 0: the lane's window ends on a word, so no shift;
// K % 4 == 2: a shift of 0 or 16 bits, one select). EXL3_RED 1 (tests only) plants a defect the byte tests must see: the
// window one weight off.
#ifndef EXL3_RED
#define EXL3_RED 0
#endif
// a tile's 64-bit window (exl3_tile_load's, the general plan: s = qb 32 + ga 16 + lp) from the words at tile offset `off` of
// `wb`, by 32-bit index
template <uint N> inline Exl3BitsM<exl3_rmode<N>()> exl3x_tile_load(const device uint* wb, uint off, uint qb, uint ga, uint lp) {
  if constexpr (exl3_rmode<N>() != 1) {
    return exl3_tile_load<N, 0>(wb + off, qb, ga, lp);
  } else {
    Exl3BitsM<1> r;
    constexpr uint NW = N / 2u;
    constexpr uint CM7 = ((8u * N) >> 4u) - (N >> 4u);
    const uint s = qb * 32u + ga * 16u + lp;
    const uint e = s + CM7 + 16u;
    const uint last = ((e - 1u) >> 5u) % NW;
    const uint w1 = (last + NW - 1u) % NW;
    const uint w2 = (w1 + NW - 1u) % NW;
    const uint a = wb[off + last], b = wb[off + w1], c = wb[off + w2];
    uint lo, hi;
    if constexpr (N % 64u == 0u) {
      // K % 4 == 0: e = 8 K (L + 1) + 256 K, a multiple of 32, so the shift is 0
      lo = a; hi = b;
    } else if constexpr (N % 32u == 0u) {
      // K % 4 == 2: e mod 32 is 0 or 16
      const bool h = ((0u - e) & 31u) != 0u;
      lo = h ? ((a >> 16u) | (b << 16u)) : a;
      hi = h ? ((b >> 16u) | (c << 16u)) : b;
    } else {
      const uint sh = (0u - e) & 31u;
      lo = uint(((ulong(b) << 32u) | ulong(a)) >> sh);
      hi = uint(((ulong(c) << 32u) | ulong(b)) >> sh);
    }
    r.v = as_type<ulong>(uint2(lo, hi));
#if EXL3_RED == 1
    r.v >>= (N / 16u);
#endif
    return r;
  }
}
// output tile t of block blk at input tile tk, 32-bit indexed
template <uint N, uint OUT_TILES, uint BT>
inline Exl3BitsM<exl3_rmode<N>()> turboexl3_x_tile(const device uint* wbase, uint tk, uint blk, uint t, uint qb, uint ga, uint lp) {
  return exl3x_tile_load<N>(wbase, (tk * OUT_TILES + blk * BT + t) * (N / 2u), qb, ga, lp);
}
// one input tile: its BT tiles' words `w` and the members' inputs at tk, decoded and FMA'd as turboexl3_gemv_members does
template <uint N, uint WMASK, int CB, uint BT, uint M, uint DEF>
inline void turboexl3_x_step(thread const Exl3BitsM<exl3_rmode<N>()> (&w)[BT], half4 x0, half4 x1, uint lp, thread float (&a)[M][BT][2]) {
  const float xr0[4] = { float(x0.x), float(x0.y), float(x0.z), float(x0.w) };
  const float xr1[4] = { float(x1.x), float(x1.y), float(x1.z), float(x1.w) };
  for (uint t = 0; t < BT; ++t) {
    const auto ww = w[t];
    uint cw[8];
    cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
    cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
    for (uint m = 0; m < 8u; ++m) {
      const float wv = (DEF != 0u) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
      a[0][t][m >> 2] = fma(xr0[m & 3u], wv, a[0][t][m >> 2]);
      if (M > 1u) a[M - 1u][t][m >> 2] = fma(xr1[m & 3u], wv, a[M - 1u][t][m >> 2]);
    }
  }
}
// turboexl3_gemv_members (EXL3_KU 0) on the 32-bit-indexed tile loads
template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT, uint M, uint DEF = 0u>
inline void turboexl3_gemv_x(const device uint* wbase, const device half* xrow0, const device half* xrow1, uint tk0, uint tk1, uint blk, uint L, threadgroup float* red_sg) {
  const uint qb = exl3_gen_qb<N>(L);
  const uint ga = exl3_gen_a<N>(L);
  const uint lp = exl3_gen_lp<N>(L);
  float a[M][BT][2];
  for (uint j = 0; j < M; ++j) for (uint t = 0; t < BT; ++t) { a[j][t][0] = 0.0f; a[j][t][1] = 0.0f; }
  for (uint tk = tk0; tk < tk1; ++tk) {
    Exl3BitsM<exl3_rmode<N>()> w[BT];
    for (uint t = 0; t < BT; ++t) w[t] = turboexl3_x_tile<N, OUT_TILES, BT>(wbase, tk, blk, t, qb, ga, lp);
    const half4 x0 = turboexl3_x4(xrow0 + tk * 16u);
    half4 x1 = half4(0.0h);
    if (M > 1u) x1 = turboexl3_x4(xrow1 + tk * 16u);
    turboexl3_x_step<N, WMASK, CB, BT, M, DEF>(w, x0, x1, lp, a);
  }
  for (uint j = 0; j < M; ++j) turboexl3_fold<BT>(a[j], red_sg + j * 16u * BT, L);
}
