// the row-slots that chose expert e, in slot order: slot rk leads when its occurrence index is a multiple of GRP and
// takes the next GRP - 1 occurrences as its members (GRP is 1 or 2); false for a slot some lead covers. `slots` is any
// pointer: MLX binds an input of fewer than 8 elements (a 1-slot call at under 8 rows) in the constant address space
template <uint GRP, uint RKC, typename SP>
inline bool turboexl3_members(SP slots, uint rk, uint e, uint L, thread uint (&members)[2], thread uint& m) {
  members[0] = rk; members[1] = rk; m = 1u;
  if (GRP == 1u) return true;
  uint before = 0u; uint next = 0xffffffffu;
  for (uint b = 0u; b < RKC; b += 32u) {
    const uint i = b + L;
    const bool hit = (i < RKC) && (i != rk) && (uint(slots[i]) == e);
    const uint mask = uint(static_cast<simd_vote::vote_t>(simd_ballot(hit)));
    const uint below = (b + 32u <= rk) ? 0xffffffffu : ((b >= rk) ? 0u : ((1u << (rk - b)) - 1u));
    before += popcount(mask & below);
    const uint above = mask & ~below;
    if (next == 0xffffffffu && above != 0u) next = b + ctz(above);
  }
  if ((before % GRP) != 0u) return false;
  if (next != 0xffffffffu) { members[1] = next; m = 2u; }
  return true;
}
template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT>
inline void turboexl3_gemv_core(const device uint* wbase, const device half* xrow, uint tk0, uint tk1, uint blk, uint L, thread float (&a)[BT][2]) {
  constexpr uint NW = N / 2u;
  // the general fast path (every even N): one select, one shift, one mask per weight
  const uint qb = exl3_gen_qb<N>(L);
  const uint ga = exl3_gen_a<N>(L);
  const uint lp = exl3_gen_lp<N>(L);
  for (uint tk = tk0; tk < tk1; ++tk) {
    const half4 x4 = *reinterpret_cast<const device half4*>(xrow + tk * 16u);
    const float xr[4] = { float(x4.x), float(x4.y), float(x4.z), float(x4.w) };
    for (uint t = 0; t < BT; ++t) {
      const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * BT + t) * ulong(NW);
      const auto ww = exl3_tile_load<N, 0>(tile, qb, ga, lp);
      uint cw[8];
      cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
      cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
      for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(cw[m])), a[t][m >> 2]);
    }
  }
}
// the simdgroup's BT x 2 accumulators folded over the quad lanes into red[sg][16 BT] (output = tile * 16 + column)
template <uint BT>
inline void turboexl3_fold(thread float (&a)[BT][2], threadgroup float* red_sg, uint L) {
  for (uint t = 0; t < BT; ++t) for (uint c = 0; c < 2u; ++c) { a[t][c] += simd_shuffle_xor(a[t][c], ushort(1)); a[t][c] += simd_shuffle_xor(a[t][c], ushort(2)); }
  if ((L & 3u) == 0u) for (uint t = 0; t < BT; ++t) { red_sg[t * 16u + (L >> 2)] = a[t][0]; red_sg[t * 16u + (L >> 2) + 8u] = a[t][1]; }
}
// the input's half4 at p: device memory (a prepare launch's rows) or threadgroup memory (the prepare inside the launch)
inline half4 turboexl3_x4(const device half* p) { return *reinterpret_cast<const device half4*>(p); }
inline half4 turboexl3_x4(const threadgroup half* p) { return *reinterpret_cast<const threadgroup half4*>(p); }
// the core over M members (row-slots that chose this expert): every weight is read and decoded once and fed to each
// member's accumulators in the one-member order, so a member's bytes equal the one-member kernel's; folds into
// red_sg[j * 16 BT + output]
template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT, uint M, uint DEF = 0u, typename XP>
inline void turboexl3_gemv_members(const device uint* wbase, XP xrow0, XP xrow1, uint tk0, uint tk1, uint blk, uint L, threadgroup float* red_sg) {
  constexpr uint NW = N / 2u;
  const uint qb = exl3_gen_qb<N>(L);
  const uint ga = exl3_gen_a<N>(L);
  const uint lp = exl3_gen_lp<N>(L);
  float a[M][BT][2];
  for (uint j = 0; j < M; ++j) for (uint t = 0; t < BT; ++t) { a[j][t][0] = 0.0f; a[j][t][1] = 0.0f; }
  // EXL3_KU 1: input tiles in pairs, both tiles' words loaded before either decodes (2 BT tiles in flight); each tile's FMAs
  // still in tile order, so the same sums
  constexpr uint KU = (EXL3_KU != 0) ? 2u : 1u;
  for (uint tk2 = tk0; tk2 < tk1; tk2 += KU) {
    Exl3BitsM<exl3_rmode<N>()> wpre[KU][BT];
    for (uint u = 0; u < KU; ++u) if (tk2 + u < tk1) for (uint t = 0; t < BT; ++t) wpre[u][t] = exl3_tile_load<N, 0>(wbase + (ulong(tk2 + u) * OUT_TILES + blk * BT + t) * ulong(NW), qb, ga, lp);
    for (uint u = 0; u < KU; ++u) {
      const uint tk = tk2 + u;
      if (tk >= tk1) break;
      const half4 x0 = turboexl3_x4(xrow0 + tk * 16u);
      const float xr0[4] = { float(x0.x), float(x0.y), float(x0.z), float(x0.w) };
      float xr1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
      if (M > 1u) { const half4 x1 = turboexl3_x4(xrow1 + tk * 16u); xr1[0] = float(x1.x); xr1[1] = float(x1.y); xr1[2] = float(x1.z); xr1[3] = float(x1.w); }
      for (uint t = 0; t < BT; ++t) {
        const auto ww = wpre[u][t];
        uint cw[8];
        cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
        cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
        for (uint m = 0; m < 8u; ++m) {
          // DEF: MUL1's byte sum t (the finish applies A and C): no half FMA or half->float convert per weight
          const float w = (DEF != 0u) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
          a[0][t][m >> 2] = fma(xr0[m & 3u], w, a[0][t][m >> 2]);
          if (M > 1u) a[M - 1u][t][m >> 2] = fma(xr1[m & 3u], w, a[M - 1u][t][m >> 2]);
        }
      }
    }
  }
  for (uint j = 0; j < M; ++j) turboexl3_fold<BT>(a[j], red_sg + j * 16u * BT, L);
}
// H128 of 128 values held one per thread (tid): 5 shuffle stages in the simdgroup, 2 exchange stages through xch
inline float turboexl3_h128(float v, threadgroup float* xch, uint tid, uint L) {
  for (uint st = 0; st < 5u; ++st) { const uint bit = 1u << st; const float p = simd_shuffle_xor(v, ushort(bit)); v = (L & bit) ? (p - v) : (v + p); }
  for (uint st = 5; st < 7u; ++st) {
    const uint bit = 1u << st;
    xch[tid] = v; threadgroup_barrier(mem_flags::mem_threadgroup);
    const float p = xch[tid ^ bit]; threadgroup_barrier(mem_flags::mem_threadgroup);
    v = (tid & bit) ? (p - v) : (v + p);
  }
  return v;
}
