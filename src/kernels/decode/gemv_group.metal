// the grouped banked GEMVs' members: the row-slots that chose packed id e, in slot order; slot rk leads when its
// occurrence index is a multiple of GRP and takes the next GRP - 1 occurrences as its members (into members[0..m)).
// False for a slot some lead covers and for a skipped row-slot (id 0xffffffff: no decode). `slots` as turboexl3_members.
template <uint GRP, uint RKC, typename SP>
inline bool turboexl3_members_g(SP slots, uint rk, uint e, uint L, thread uint* members, thread uint& m) {
  members[0] = rk; m = 1u;
  if (e == 0xffffffffu) return false;
  uint before = 0u;
  for (uint b = 0u; b < RKC; b += 32u) {
    const uint i = b + L;
    const bool hit = (i < RKC) && (i != rk) && (uint(slots[i]) == e);
    const uint mask = uint(static_cast<simd_vote::vote_t>(simd_ballot(hit)));
    const uint below = (b + 32u <= rk) ? 0xffffffffu : ((b >= rk) ? 0u : ((1u << (rk - b)) - 1u));
    before += popcount(mask & below);
    uint above = mask & ~below;
    while (above != 0u && m < GRP) { members[m] = b + ctz(above); m += 1u; above &= above - 1u; }
  }
  return (before % GRP) == 0u;
}
// hw2 (the hit wave's ILP retune): the grouped GEMV's A/B variants, one define each (0: today's). Every variant reads
// the same tile words, decodes the same codewords to the same values and runs the same FMA sequence into the same f32
// accumulators and the same folds, so the same bytes; they differ in when the words load, how many lanes load them,
// the registers the loop holds and the passes over a tile:
//   EXL3_GPF 1: a rolling prefetch: the next input tile's words loaded before this tile decodes (not with EXL3_KU)
//   EXL3_GXH 1: each member's x quad held as half4 and widened at its FMA (float(half) exact), not as four floats
//   EXL3_GM2 1: the four-member path as two passes of two members (the second pass reads and decodes the tile again),
//               so no path holds four members' accumulators
//   EXL3_GSH 1: the one-64-bit-window reader (exl3_rmode 1) with each tile word loaded by one lane (lane i word
//               i mod NW) and taken by the lanes that read it through simd_shuffle (the loop is simdgroup-uniform)
#ifndef EXL3_GPF
#define EXL3_GPF 0
#endif
#ifndef EXL3_GXH
#define EXL3_GXH 0
#endif
#ifndef EXL3_GM2
#define EXL3_GM2 0
#endif
#ifndef EXL3_GSH
#define EXL3_GSH 0
#endif
// EXL3_GSH's per-lane plan: exl3_tile_load's word indexes (last, w1, w2) and shift at IR 0, once per lane
struct Exl3ShPlan { ushort last, w1, w2; uint sh; };
template <uint N> inline Exl3ShPlan exl3_sh_plan(uint qb, uint ga, uint lp) {
  constexpr uint NW = N / 2u;
  constexpr uint CM7 = ((8u * N) >> 4u) - (N >> 4u);
  const uint s = qb * 32u + ga * 16u + lp;
  const uint e = s + CM7 + 16u;
  const uint last = ((e - 1u) >> 5u) % NW;
  const uint w1 = (last + NW - 1u) % NW;
  const uint w2 = (w1 + NW - 1u) % NW;
  Exl3ShPlan p;
  p.last = ushort(last); p.w1 = ushort(w1); p.w2 = ushort(w2); p.sh = (0u - e) & 31u;
  return p;
}
// a tile's words for lane L: exl3_tile_load's, or under EXL3_GSH (rmode 1) the same three words through the shuffles
template <uint N> inline Exl3BitsM<exl3_rmode<N>()> exl3_tile_load_g(const device uint* tile, uint qb, uint ga, uint lp, uint L, Exl3ShPlan p) {
#if EXL3_GSH != 0
  if constexpr (exl3_rmode<N>() == 1) {
    constexpr uint NW = N / 2u;
    const uint mine = tile[L % NW];
    const uint a = simd_shuffle(mine, p.last), b = simd_shuffle(mine, p.w1), c = simd_shuffle(mine, p.w2);
    Exl3BitsM<exl3_rmode<N>()> r;
    r.v = as_type<ulong>(uint2(uint(((ulong(b) << 32u) | ulong(a)) >> p.sh), uint(((ulong(c) << 32u) | ulong(b)) >> p.sh)));
    return r;
  }
#endif
  (void)L; (void)p;
  return exl3_tile_load<N, 0>(tile, qb, ga, lp);
}
// member j's x quad at input tile tk, and its value k as the FMA takes it (EXL3_GXH: the half4 widened there)
#if EXL3_GXH != 0
#define EXL3_G_XDECL(M) half4 xq[M]
#define EXL3_G_XLOAD(j, p) xq[j] = turboexl3_x4(p)
#define EXL3_G_X(j, k) float(xq[j][k])
#else
#define EXL3_G_XDECL(M) float xq[M][4]
#define EXL3_G_XLOAD(j, p) { const half4 x = turboexl3_x4(p); xq[j][0] = float(x.x); xq[j][1] = float(x.y); xq[j][2] = float(x.z); xq[j][3] = float(x.w); }
#define EXL3_G_X(j, k) xq[j][k]
#endif
// turboexl3_gemv_members over m members of at most M (M a compile-time bound; the rows past m read member 0's row and
// are not folded): every weight is read and decoded once and fed to each member's accumulators in the one-member order,
// so a member's bytes equal the one-member kernel's; member j's row is xbase + members[j] * ld; folds into
// red_sg[j * 16 BT + output]
template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT, uint M, uint DEF, typename XP>
inline void turboexl3_gemv_group(const device uint* wbase, XP xbase, ulong ld, thread const uint* members, uint m, uint tk0, uint tk1, uint blk, uint L, threadgroup float* red_sg) {
#if EXL3_GM2 != 0
  if constexpr (M == 4u) {
    turboexl3_gemv_group<N, WMASK, CB, OUT_TILES, BT, 2u, DEF>(wbase, xbase, ld, members, 2u, tk0, tk1, blk, L, red_sg);
    turboexl3_gemv_group<N, WMASK, CB, OUT_TILES, BT, 2u, DEF>(wbase, xbase, ld, members + 2, m - 2u, tk0, tk1, blk, L, red_sg + 2u * 16u * BT);
    return;
  }
#endif
  constexpr uint NW = N / 2u;
  const uint qb = exl3_gen_qb<N>(L);
  const uint ga = exl3_gen_a<N>(L);
  const uint lp = exl3_gen_lp<N>(L);
  const Exl3ShPlan sp = exl3_sh_plan<N>(qb, ga, lp);
  float a[M][BT][2];
  for (uint j = 0; j < M; ++j) for (uint t = 0; t < BT; ++t) { a[j][t][0] = 0.0f; a[j][t][1] = 0.0f; }
#if EXL3_GPF != 0
  Exl3BitsM<exl3_rmode<N>()> wc[BT];
  if (tk0 < tk1) for (uint t = 0; t < BT; ++t) wc[t] = exl3_tile_load_g<N>(wbase + (ulong(tk0) * OUT_TILES + blk * BT + t) * ulong(NW), qb, ga, lp, L, sp);
  for (uint tk = tk0; tk < tk1; ++tk) {
    const bool more = tk + 1u < tk1;
    Exl3BitsM<exl3_rmode<N>()> wn[BT];
    if (more) for (uint t = 0; t < BT; ++t) wn[t] = exl3_tile_load_g<N>(wbase + (ulong(tk + 1u) * OUT_TILES + blk * BT + t) * ulong(NW), qb, ga, lp, L, sp);
    EXL3_G_XDECL(M);
    for (uint j = 0; j < M; ++j) EXL3_G_XLOAD(j, xbase + ulong(members[j < m ? j : 0u]) * ld + tk * 16u);
    for (uint t = 0; t < BT; ++t) {
      const auto ww = wc[t];
      uint cw[8];
      cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
      cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
      for (uint q = 0; q < 8u; ++q) {
        const float w = (DEF != 0u) ? exl3_mul1_t(cw[q]) : float(exl3_decode<CB>(cw[q]));
        for (uint j = 0; j < M; ++j) a[j][t][q >> 2] = fma(EXL3_G_X(j, q & 3u), w, a[j][t][q >> 2]);
      }
    }
    if (more) for (uint t = 0; t < BT; ++t) wc[t] = wn[t];
  }
#else
  constexpr uint KU = (EXL3_KU != 0) ? 2u : 1u;
  for (uint tk2 = tk0; tk2 < tk1; tk2 += KU) {
    Exl3BitsM<exl3_rmode<N>()> wpre[KU][BT];
    for (uint u = 0; u < KU; ++u) if (tk2 + u < tk1) for (uint t = 0; t < BT; ++t) wpre[u][t] = exl3_tile_load_g<N>(wbase + (ulong(tk2 + u) * OUT_TILES + blk * BT + t) * ulong(NW), qb, ga, lp, L, sp);
    for (uint u = 0; u < KU; ++u) {
      const uint tk = tk2 + u;
      if (tk >= tk1) break;
      EXL3_G_XDECL(M);
      for (uint j = 0; j < M; ++j) EXL3_G_XLOAD(j, xbase + ulong(members[j < m ? j : 0u]) * ld + tk * 16u);
      for (uint t = 0; t < BT; ++t) {
        const auto ww = wpre[u][t];
        uint cw[8];
        cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
        cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
        for (uint q = 0; q < 8u; ++q) {
          const float w = (DEF != 0u) ? exl3_mul1_t(cw[q]) : float(exl3_decode<CB>(cw[q]));
          for (uint j = 0; j < M; ++j) a[j][t][q >> 2] = fma(EXL3_G_X(j, q & 3u), w, a[j][t][q >> 2]);
        }
      }
    }
  }
#endif
  for (uint j = 0; j < M; ++j) if (j < m) turboexl3_fold<BT>(a[j], red_sg + j * 16u * BT, L);
}
#undef EXL3_G_XDECL
#undef EXL3_G_XLOAD
#undef EXL3_G_X
