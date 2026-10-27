// the v2 GEMV jobs' tile reader, in the loop shapes of dgemv / turboexl3_gemv_pair / turboexl3_dn5_sg (no forced unrolling, every
// thread array indexed by small constant-trip loops): turboexl3_spload issues lane L's raw words (exl3_tile_load<N, 0>'s loads,
// dgemv's general reader) of one part, the OTS output tiles from ot0 for each of its PER input tiles from tk0, so a
// simdgroup can have parts in flight ahead of its decode (and across a barrier); turboexl3_spart funnels them with
// exl3_tile_load's integer arithmetic (the same bits) into the window's 32-bit words and takes each codeword from them
// (turboexl3_scw). The IntWords reader (rmode 0, EXL3_R64 0) loads at the funnel instead.
struct Exl3Raw { uint a, b, c, d; };
template <uint N, uint OUT_TILES, uint PER, uint OTS>
inline void turboexl3_spload(const device uint* wbase, uint tk0, uint ot0, uint L, thread Exl3Raw (&r)[PER][OTS]) {
  constexpr uint NW = N / 2u;
  constexpr uint CM7 = ((8u * N) >> 4u) - (N >> 4u);
  const uint e = exl3_gen_qb<N>(L) * 32u + exl3_gen_a<N>(L) * 16u + exl3_gen_lp<N>(L) + CM7 + 16u;
  const uint last = ((e - 1u) >> 5u) % NW;
  const uint w1 = (last + NW - 1u) % NW;
  const uint w2 = (w1 + NW - 1u) % NW;
  const uint w3 = (w2 + NW - 1u) % NW;
  for (uint i = 0; i < PER; ++i) {
    for (uint t = 0; t < OTS; ++t) {
      const device uint* tile = wbase + (ulong(tk0 + i) * OUT_TILES + ot0 + t) * ulong(NW);
      if (exl3_rmode<N>() != 0) {
        r[i][t].a = tile[last]; r[i][t].b = tile[w1]; r[i][t].c = tile[w2];
        r[i][t].d = (exl3_rmode<N>() == 2) ? tile[w3] : 0u;
      } else {
        r[i][t].a = 0u; r[i][t].b = 0u; r[i][t].c = 0u; r[i][t].d = 0u;
      }
    }
  }
}
// codeword M of a lane's window held as 32-bit words: lo, mid, hi = its bits 0-31, 32-63 and 64-95 counted from the window's
// end (the 96-bit reader's words; a 64-bit window never reaches hi), taken by exl3_tile_cw's 96-bit rules at its
// compile-time offset SH (one bit-field extract, or a two-word funnel): the 16 bits exl3_tile_cw takes from either window
// reader, with no 64-bit window value (rmode 1's ulong)
template <uint N, uint WMASK, uint M>
inline uint turboexl3_scw(uint lo, uint mid, uint hi) {
  constexpr uint SH = (((8u * N) >> 4u) - (N >> 4u)) - ((((M + 1u) * N) >> 4u) - (N >> 4u));
  uint cw;
  if constexpr (SH + 16u <= 32u) cw = metal::extract_bits(lo, SH, 16u);
  else if constexpr (SH < 32u) cw = ((lo >> SH) | (mid << (32u - SH))) & 0xffffu;
  else if constexpr (SH + 16u <= 64u) cw = metal::extract_bits(mid, SH - 32u, 16u);
  else if constexpr (SH < 64u) cw = ((mid >> (SH - 32u)) | (hi << (64u - SH))) & 0xffffu;
  else cw = metal::extract_bits(hi, SH - 64u, 16u);
  return (WMASK == 0xffffu) ? cw : (cw & WMASK);
}
// one part of a K slice (dgemv's simdgroup share: PER input tiles from tk0) for OTS output tiles and RS rows, row j's
// lane-order input at x0 + j XSTRIDE (element 0 = the part's first input tile): from zero in dgemv's order (tile order,
// weights m = 0..7, a[j][t][m / 4] = fma(x[j][m % 4], w, a[j][t][m / 4])), folded over the quad lanes (dgemv's
// turboexl3_fold) and added into the slice's left sum s (dgemv's plane store ((part 0 + part 1) + part 2) + part 3)
template <uint N, uint WMASK, int CB, uint OUT_TILES, uint RS, uint OTS, uint PER, uint DEF, uint XSTRIDE, typename XP>
inline void turboexl3_spart(thread const Exl3Raw (&r)[PER][OTS], const device uint* wbase, uint tk0, uint ot0, XP x0, uint L, bool first, thread float (&s)[RS][OTS][2]) {
  constexpr uint NW = N / 2u;
  constexpr uint CM7 = ((8u * N) >> 4u) - (N >> 4u);
  const uint qb = exl3_gen_qb<N>(L);
  const uint ga = exl3_gen_a<N>(L);
  const uint lp = exl3_gen_lp<N>(L);
  const uint sh = (0u - (qb * 32u + ga * 16u + lp + CM7 + 16u)) & 31u;
  float a[RS][OTS][2];
  for (uint j = 0; j < RS; ++j) for (uint t = 0; t < OTS; ++t) { a[j][t][0] = 0.0f; a[j][t][1] = 0.0f; }
  for (uint i = 0; i < PER; ++i) {
    float xr[RS][4];
    for (uint j = 0; j < RS; ++j) {
      const half4 x4 = turboexl3_x4(x0 + j * XSTRIDE + i * 16u);
      xr[j][0] = float(x4.x); xr[j][1] = float(x4.y); xr[j][2] = float(x4.z); xr[j][3] = float(x4.w);
    }
    for (uint t = 0; t < OTS; ++t) {
      uint cw[8];
      if constexpr (exl3_rmode<N>() != 0) {
        // the window's 32-bit words from the raw words (exl3_tile_load's funnels), hi only for the 96-bit window
        const uint lo = uint(((ulong(r[i][t].b) << 32u) | ulong(r[i][t].a)) >> sh);
        const uint mid = uint(((ulong(r[i][t].c) << 32u) | ulong(r[i][t].b)) >> sh);
        const uint hi = (exl3_rmode<N>() == 2) ? uint(((ulong(r[i][t].d) << 32u) | ulong(r[i][t].c)) >> sh) : 0u;
        cw[0] = turboexl3_scw<N, WMASK, 0>(lo, mid, hi); cw[1] = turboexl3_scw<N, WMASK, 1>(lo, mid, hi); cw[2] = turboexl3_scw<N, WMASK, 2>(lo, mid, hi); cw[3] = turboexl3_scw<N, WMASK, 3>(lo, mid, hi);
        cw[4] = turboexl3_scw<N, WMASK, 4>(lo, mid, hi); cw[5] = turboexl3_scw<N, WMASK, 5>(lo, mid, hi); cw[6] = turboexl3_scw<N, WMASK, 6>(lo, mid, hi); cw[7] = turboexl3_scw<N, WMASK, 7>(lo, mid, hi);
      } else {
        const auto ww = exl3_tile_load<N, 0>(wbase + (ulong(tk0 + i) * OUT_TILES + ot0 + t) * ulong(NW), qb, ga, lp);
        cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
        cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
      }
      for (uint m = 0; m < 8u; ++m) {
        const float wv = (DEF != 0u) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
        for (uint j = 0; j < RS; ++j) a[j][t][m >> 2] = fma(xr[j][m & 3u], wv, a[j][t][m >> 2]);
      }
    }
  }
  for (uint j = 0; j < RS; ++j) {
    for (uint t = 0; t < OTS; ++t) {
      for (uint c = 0; c < 2u; ++c) {
        float v = a[j][t][c];
        v += simd_shuffle_xor(v, ushort(1));
        v += simd_shuffle_xor(v, ushort(2));
        s[j][t][c] = first ? v : (s[j][t][c] + v);
      }
    }
  }
}
// output o of row r in K slice k of planes [KS][RS][DIM]: the natural layout, or lane-major (LM) within each 128-block,
// offset 4 (i % 32) + i / 32 of block o / 128 (i = o % 128), so the finish's lane L reads its values L + 32 q as one float4
template <uint LM, uint RS, uint DIM>
inline ulong turboexl3_sidx(uint k, uint r, uint o) {
  if (LM != 0u) return (ulong(k * RS + r) * ulong(DIM / 128u) + ulong(o >> 7u)) * ulong(128u) + ulong(4u * (o & 31u) + ((o >> 5u) & 3u));
  return ulong(k * RS + r) * ulong(DIM) + ulong(o);
}
// the slice's values to the planes: every lane of a quad holds the quad's 2 RS OTS values (columns L / 4 + 8 c of the OTS
// tiles from ot0); value v = (r OTS + t) 2 + c is stored by the quad's lane v % 4
template <uint LM, uint RS, uint OTS, uint DIM>
inline void turboexl3_sstore(device float* z, thread const float (&s)[RS][OTS][2], uint k, uint ot0, uint L) {
  for (uint r = 0; r < RS; ++r) {
    for (uint t = 0; t < OTS; ++t) {
      for (uint c = 0; c < 2u; ++c) {
        if ((((r * OTS + t) * 2u + c) & 3u) == (L & 3u)) z[turboexl3_sidx<LM, RS, DIM>(k, r, (ot0 + t) * 16u + (L >> 2u) + 8u * c)] = s[r][t][c];
      }
    }
  }
}
// turboexl3_sfinish over planes in either layout: plane 0, then + plane k in slice order (LM: as float4, the same four sums),
// the deferred MUL1 map with the row's input sum, f16, H128, svh, f16
template <uint KS, uint RS, uint DIM, uint DEF, uint LM, typename SV>
inline void turboexl3_sfinish2(const device float* z, SV svh, float xsum, uint row, uint b, uint L, thread float (&v)[4]) {
  constexpr float HAD_SCALE = 0.08838834764831845f;
  const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
  const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
  float acc[4];
  if (LM != 0u) {
    const device float4* z4 = reinterpret_cast<const device float4*>(z);
    float4 t4 = z4[(ulong(row) * ulong(DIM / 128u) + ulong(b)) * ulong(32u) + ulong(L)];
    for (uint k = 1u; k < KS; ++k) t4 += z4[((ulong(k) * ulong(RS) + ulong(row)) * ulong(DIM / 128u) + ulong(b)) * ulong(32u) + ulong(L)];
    acc[0] = t4.x; acc[1] = t4.y; acc[2] = t4.z; acc[3] = t4.w;
  } else {
    for (uint q = 0; q < 4u; ++q) {
      const uint o = b * 128u + L + 32u * q;
      float a = z[ulong(row) * ulong(DIM) + o];
      for (uint k = 1u; k < KS; ++k) a += z[(ulong(k) * ulong(RS) + row) * ulong(DIM) + o];
      acc[q] = a;
    }
  }
  for (uint q = 0; q < 4u; ++q) {
    float a = acc[q];
    if (DEF != 0u) a = fma(a, MUL1_A, MUL1_C * xsum);
    v[q] = float(half(a));
  }
  turboexl3_h128r(v, L);
  for (uint q = 0; q < 4u; ++q) v[q] = float(half(v[q] * HAD_SCALE * float(svh[b * 128u + L + 32u * q])));
}
