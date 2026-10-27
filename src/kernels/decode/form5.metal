// coherent(device) (Metal 3.2+): another threadgroup's device stores are only guaranteed visible through it (form 6's
// arrival reduce)
#if defined(__HAVE_COHERENT__)
#define TURBOEXL3_COH coherent(device)
#else
#define TURBOEXL3_COH
#endif
// the row-slots that chose expert e, in slot order: slot rk leads when its occurrence index is a multiple of GM (1..4) and
// takes the next GM - 1 occurrences as its members (top-k picks distinct experts per row, so members are other rows'
// slots); false for a slot some lead covers
template <uint GM, uint RKC, typename SP>
inline bool turboexl3_group(SP slots, uint rk, uint e, uint L, thread uint (&mem)[4], thread uint& m) {
  mem[0] = rk; mem[1] = rk; mem[2] = rk; mem[3] = rk; m = 1u;
  if (GM == 1u) return true;
  uint before = 0u;
  for (uint b = 0u; b < RKC; b += 32u) {
    const uint i = b + L;
    const bool hit = (i < RKC) && (i != rk) && (uint(slots[i]) == e);
    const uint mask = uint(static_cast<simd_vote::vote_t>(simd_ballot(hit)));
    const uint below = (b + 32u <= rk) ? 0xffffffffu : ((b >= rk) ? 0u : ((1u << (rk - b)) - 1u));
    before += popcount(mask & below);
    uint above = mask & ~below;
    while (above != 0u && m < GM) { mem[m] = b + ctz(above); m += 1u; above &= above - 1u; }
  }
  return (before % GM) == 0u;
}
template <uint N, uint WMASK, uint MM, int IR> inline uint turboexl3_cw5(IntWords w, uint lp) {
  if (IR != 0) return exl3_int_cw<N, WMASK, MM>(w, lp);
  return exl3_gen_cw<N, WMASK, MM>(w, lp);
}
// lane L's BT output tiles (ot0..) x 2 columns over input tiles [tk0, tk1) of expert `wbase` for M members, member j's
// prepared input in threadgroup memory at xj (lane order, offset 4 (L & 3), element 0 = input tile xtk0): every weight is
// read and decoded once and fed to each member in the one-member order, so a member's sums are the one-member kernel's
template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT, uint M, uint DEF, int IR>
inline void turboexl3_gemv_tg(const device uint* wbase, const threadgroup half* x0, const threadgroup half* x1, const threadgroup half* x2, const threadgroup half* x3, uint tk0, uint tk1, uint xtk0, uint ot0, uint L, thread float (&a)[M][BT][2]) {
  constexpr uint NW = N / 2u;
  const uint qb = (IR != 0) ? exl3_int_qb<N>(L) : exl3_gen_qb<N>(L);
  const uint ga = (IR != 0) ? exl3_int_a<N>(L) : exl3_gen_a<N>(L);
  const uint lp = (IR != 0) ? exl3_int_b<N>(L) : exl3_gen_lp<N>(L);
  for (uint tk = tk0; tk < tk1; ++tk) {
    float xv[M][4];
    for (uint j = 0; j < M; ++j) {
      const threadgroup half* xp = (j == 0u) ? x0 : ((j == 1u) ? x1 : ((j == 2u) ? x2 : x3));
      const half4 h = *reinterpret_cast<const threadgroup half4*>(xp + (tk - xtk0) * 16u);
      xv[j][0] = float(h.x); xv[j][1] = float(h.y); xv[j][2] = float(h.z); xv[j][3] = float(h.w);
    }
    for (uint t = 0; t < BT; ++t) {
      const device uint* tile = wbase + (ulong(tk) * OUT_TILES + ot0 + t) * ulong(NW);
      const auto ww = exl3_tile_load<N, IR>(tile, qb, ga, lp);
      uint cw[8];
      cw[0] = exl3_tile_cw<N, WMASK, 0, IR>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, IR>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, IR>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, IR>(ww, lp);
      cw[4] = exl3_tile_cw<N, WMASK, 4, IR>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, IR>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, IR>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, IR>(ww, lp);
      for (uint mm = 0; mm < 8u; ++mm) {
        const float w = (DEF != 0u) ? exl3_mul1_t(cw[mm]) : float(exl3_decode<CB>(cw[mm]));
        for (uint j = 0; j < M; ++j) a[j][t][mm >> 2] = fma(xv[j][mm & 3u], w, a[j][t][mm >> 2]);
      }
    }
  }
}
// H128 of a simdgroup's 128 values, lane L holding elements L + 32 q: turboexl3_h128's stages in its order (5 shuffle stages,
// then bits 5 and 6 between the registers), so the same f32 values
inline void turboexl3_h128r(thread float (&v)[4], uint L) {
  for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint q = 0; q < 4u; ++q) { const float p = simd_shuffle_xor(v[q], ushort(bit)); v[q] = (L & bit) ? (p - v[q]) : (v[q] + p); } }
  const float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3];
  v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3;
}
// the gate|up GEMV of one simdgroup for M members, folded over the quad lanes into red_sg[j * 16 BT + output]
template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT, uint M, uint DEF, int IR>
inline void turboexl3_gu5_sg(const device uint* wbase, const threadgroup half* xs0, uint stride, uint tk0, uint tk1, uint xtk0, uint ot0, uint L, threadgroup float* red_sg) {
  float a[M][BT][2];
  for (uint j = 0; j < M; ++j) for (uint t = 0; t < BT; ++t) { a[j][t][0] = 0.0f; a[j][t][1] = 0.0f; }
  turboexl3_gemv_tg<N, WMASK, CB, OUT_TILES, BT, M, DEF, IR>(wbase, xs0, xs0 + stride, xs0 + 2u * stride, xs0 + 3u * stride, tk0, tk1, xtk0, ot0, L, a);
  for (uint j = 0; j < M; ++j) turboexl3_fold<BT>(a[j], red_sg + j * 16u * BT, L);
}
// one output tile's GEMV over NT input tiles from tk0 for M members, input tiles taken in pairs: both tiles' words load
// before either decodes (two tiles in flight per simdgroup), each tile's FMAs in turboexl3_gemv_tg's order (tile tk before
// tk + 1, weights m = 0..7), so the same sums
template <uint N, uint WMASK, int CB, uint OUT_TILES, uint M, uint DEF, int IR, uint NT>
inline void turboexl3_gemv_pair(const device uint* wbase, const threadgroup half* x0, const threadgroup half* x1, const threadgroup half* x2, const threadgroup half* x3, uint tk0, uint ot, uint L, thread float (&a)[M][2]) {
  constexpr uint NW = N / 2u;
  const uint qb = (IR != 0) ? exl3_int_qb<N>(L) : exl3_gen_qb<N>(L);
  const uint ga = (IR != 0) ? exl3_int_a<N>(L) : exl3_gen_a<N>(L);
  const uint lp = (IR != 0) ? exl3_int_b<N>(L) : exl3_gen_lp<N>(L);
  for (uint i = 0; i < NT; i += 2u) {
    const uint nt = (i + 1u < NT) ? 2u : 1u;
    Exl3BitsM<exl3_rmode<N>()> ww[2];
    for (uint u = 0; u < 2u; ++u) if (u < nt) ww[u] = exl3_tile_load<N, IR>(wbase + (ulong(tk0 + i + u) * OUT_TILES + ot) * ulong(NW), qb, ga, lp);
    for (uint u = 0; u < 2u; ++u) if (u < nt) {
      const uint tk = tk0 + i + u;
      float xv[M][4];
      for (uint j = 0; j < M; ++j) {
        const threadgroup half* xp = (j == 0u) ? x0 : ((j == 1u) ? x1 : ((j == 2u) ? x2 : x3));
        const half4 h = *reinterpret_cast<const threadgroup half4*>(xp + tk * 16u);
        xv[j][0] = float(h.x); xv[j][1] = float(h.y); xv[j][2] = float(h.z); xv[j][3] = float(h.w);
      }
      uint cw[8];
      cw[0] = exl3_tile_cw<N, WMASK, 0, IR>(ww[u], lp); cw[1] = exl3_tile_cw<N, WMASK, 1, IR>(ww[u], lp); cw[2] = exl3_tile_cw<N, WMASK, 2, IR>(ww[u], lp); cw[3] = exl3_tile_cw<N, WMASK, 3, IR>(ww[u], lp);
      cw[4] = exl3_tile_cw<N, WMASK, 4, IR>(ww[u], lp); cw[5] = exl3_tile_cw<N, WMASK, 5, IR>(ww[u], lp); cw[6] = exl3_tile_cw<N, WMASK, 6, IR>(ww[u], lp); cw[7] = exl3_tile_cw<N, WMASK, 7, IR>(ww[u], lp);
      for (uint mm = 0; mm < 8u; ++mm) {
        const float w = (DEF != 0u) ? exl3_mul1_t(cw[mm]) : float(exl3_decode<CB>(cw[mm]));
        for (uint j = 0; j < M; ++j) a[j][mm >> 2] = fma(xv[j][mm & 3u], w, a[j][mm >> 2]);
      }
    }
  }
}
// the down GEMV of one simdgroup (one output tile ot) for M members over every input tile in down3's K partition: KS2
// slices of four parts of PER2 tiles, each part from zero and folded over the quad lanes, a slice the left sum of its
// parts, the total the left sum of the slices from zero (down3 then dfin3's order); every lane ends with the sums
template <uint N, uint WMASK, int CB, uint OUT_TILES, uint M, uint DEF, int IR, uint KS2C, uint PER2>
inline void turboexl3_dn5_sg(const device uint* wbase, const threadgroup half* xs0, uint stride, uint ot, uint L, thread float (&tot)[4][2]) {
  for (uint j = 0; j < 4u; ++j) { tot[j][0] = 0.0f; tot[j][1] = 0.0f; }
  for (uint si = 0; si < KS2C; ++si) {
    float sl[M][2];
    for (uint p = 0; p < 4u; ++p) {
      float a[M][2];
      for (uint j = 0; j < M; ++j) { a[j][0] = 0.0f; a[j][1] = 0.0f; }
      const uint tk0 = (si * 4u + p) * PER2;
      turboexl3_gemv_pair<N, WMASK, CB, OUT_TILES, M, DEF, IR, PER2>(wbase, xs0, xs0 + stride, xs0 + 2u * stride, xs0 + 3u * stride, tk0, ot, L, a);
      for (uint j = 0; j < M; ++j) for (uint c = 0; c < 2u; ++c) {
        float v = a[j][c];
        v += simd_shuffle_xor(v, ushort(1)); v += simd_shuffle_xor(v, ushort(2));
        sl[j][c] = (p == 0u) ? v : (sl[j][c] + v);
      }
    }
    for (uint j = 0; j < M; ++j) for (uint c = 0; c < 2u; ++c) tot[j][c] += sl[j][c];
  }
}
