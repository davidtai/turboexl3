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
// turboexl3_gemv_members over m members of at most M (M a compile-time bound; the rows past m read member 0's row and
// are not folded): every weight is read and decoded once and fed to each member's accumulators in the one-member order,
// so a member's bytes equal the one-member kernel's; member j's row is xbase + members[j] * ld; folds into
// red_sg[j * 16 BT + output]
template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT, uint M, uint DEF, typename XP>
inline void turboexl3_gemv_group(const device uint* wbase, XP xbase, ulong ld, thread const uint* members, uint m, uint tk0, uint tk1, uint blk, uint L, threadgroup float* red_sg) {
  constexpr uint NW = N / 2u;
  const uint qb = exl3_gen_qb<N>(L);
  const uint ga = exl3_gen_a<N>(L);
  const uint lp = exl3_gen_lp<N>(L);
  float a[M][BT][2];
  for (uint j = 0; j < M; ++j) for (uint t = 0; t < BT; ++t) { a[j][t][0] = 0.0f; a[j][t][1] = 0.0f; }
  constexpr uint KU = (EXL3_KU != 0) ? 2u : 1u;
  for (uint tk2 = tk0; tk2 < tk1; tk2 += KU) {
    Exl3BitsM<exl3_rmode<N>()> wpre[KU][BT];
    for (uint u = 0; u < KU; ++u) if (tk2 + u < tk1) for (uint t = 0; t < BT; ++t) wpre[u][t] = exl3_tile_load<N, 0>(wbase + (ulong(tk2 + u) * OUT_TILES + blk * BT + t) * ulong(NW), qb, ga, lp);
    for (uint u = 0; u < KU; ++u) {
      const uint tk = tk2 + u;
      if (tk >= tk1) break;
      float xr[M][4];
      for (uint j = 0; j < M; ++j) {
        const half4 x = turboexl3_x4(xbase + ulong(members[j < m ? j : 0u]) * ld + tk * 16u);
        xr[j][0] = float(x.x); xr[j][1] = float(x.y); xr[j][2] = float(x.z); xr[j][3] = float(x.w);
      }
      for (uint t = 0; t < BT; ++t) {
        const auto ww = wpre[u][t];
        uint cw[8];
        cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
        cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
        for (uint q = 0; q < 8u; ++q) {
          const float w = (DEF != 0u) ? exl3_mul1_t(cw[q]) : float(exl3_decode<CB>(cw[q]));
          for (uint j = 0; j < M; ++j) a[j][t][q >> 2] = fma(xr[j][q & 3u], w, a[j][t][q >> 2]);
        }
      }
    }
  }
  for (uint j = 0; j < M; ++j) if (j < m) turboexl3_fold<BT>(a[j], red_sg + j * 16u * BT, L);
}
