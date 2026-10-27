// turboexl3 helpers: template parameters are only visible inside a kernel body, so every helper takes the rate's
// halfwords N (and the window mask / codebook) as its own template argument; the body binds them from NHW, WBITS,
// CODEBOOK once.
struct LaneWords { uint w0, w1, w2, w3; uint wprev; };
// lane L's first needed word: base = (((8L + 1) N >> 4) + 16 N - 16) >> 5; the lane's windows end off_m bits past it
template <uint N> inline uint exl3_lane_base(uint L) { return ((((8u * L + 1u) * N) >> 4u) + 16u * N - 16u) >> 5u; }
template <uint N> inline uint exl3_lane_off(uint L, uint m, uint base) { return (((8u * L + m + 1u) * N) >> 4u) + 16u * N - 32u * base; }
template <uint N> inline LaneWords exl3_lane_words(const device uint* tile, uint base) {
  constexpr uint NW = N / 2u;
  LaneWords r; const uint b = base % NW;
  r.w0 = tile[b]; r.w1 = tile[(b + 1u) % NW]; r.w2 = tile[(b + 2u) % NW]; r.w3 = tile[(b + 3u) % NW]; r.wprev = tile[(b + NW - 1u) % NW];
  return r;
}
inline uint exl3_funnel(uint lo, uint hi, uint sh) { return ((sh == 0u) ? hi : ((hi >> sh) | (lo << (32u - sh)))) & 0xffffu; }
// window m of lane L from its loaded words: off = end bit past word base, q = its word, sh lands its last bit at bit 0
template <uint WMASK> inline uint exl3_lane_cw(LaneWords w, uint off) {
  const uint q = (off - 1u) >> 5u; const uint sh = 32u * (q + 1u) - off;
  const uint hi = (q == 0u) ? w.w0 : ((q == 1u) ? w.w1 : ((q == 2u) ? w.w2 : w.w3));
  const uint lo = (q == 0u) ? w.wprev : ((q == 1u) ? w.w0 : ((q == 2u) ? w.w1 : w.w2));
  return exl3_funnel(lo, hi, sh) & WMASK;
}
// the integer-rate fast path (N % 16 == 0): the lane's base word qb' and b (see lane.zig IntPlan)
template <uint N> inline uint exl3_int_qb(uint L) { constexpr uint K = N / 16u; return (8u * L * K + K + 16u * N - 16u) >> 5u; }
template <uint N> inline uint exl3_int_a(uint L) { constexpr uint K = N / 16u; return ((8u * L * K + K + 16u * N - 16u) >> 4u) & 1u; }
template <uint N> inline uint exl3_int_b(uint L) { constexpr uint K = N / 16u; return ((8u * L * K + K + 16u * N - 16u) >> 3u) & 1u; }
// the 32-bit values at halfword offsets a + 0..5 of the lane's word qb: a lane's eight windows span at most
// 15 + 7 N / 16 + 16 <= 87 bits past that halfword at N <= 128, so four words cover them (h[6], h[7] unreachable)
struct IntWords { uint h[6]; };
template <uint N> inline IntWords exl3_int_words(const device uint* tile, uint qb, uint a) {
  constexpr uint NW = N / 2u;
  const uint b0 = qb % NW;
  const uint w0 = tile[b0], w1 = tile[(b0 + 1u) % NW], w2 = tile[(b0 + 2u) % NW], w3 = tile[(b0 + 3u) % NW];
  const uint o1 = (w0 << 16u) | (w1 >> 16u), o3 = (w1 << 16u) | (w2 >> 16u), o5 = (w2 << 16u) | (w3 >> 16u);
  IntWords r;
  if (a == 0u) { r.h[0] = w0; r.h[1] = o1; r.h[2] = w1; r.h[3] = o3; r.h[4] = w2; r.h[5] = o5; }
  else { r.h[0] = o1; r.h[1] = w1; r.h[2] = o3; r.h[3] = w2; r.h[4] = o5; r.h[5] = w3; }
  return r;
}
// window m: cm = (K mod 8) + K m, j = (cm >> 4) + carry, the 16 bits of h[j] at (8 b + cm) & 15 from the top; M is the unrolled m
template <uint N, uint WMASK, uint M> inline uint exl3_int_cw(IntWords w, uint b) {
  constexpr uint K = N / 16u; constexpr uint CM = (K % 8u) + K * M; constexpr uint JC = CM >> 4u; constexpr bool CARRY_IF_B = (CM & 15u) >= 8u;
  const uint hsel = (CARRY_IF_B && b != 0u) ? w.h[JC + 1u] : w.h[JC];
  const uint cw = metal::extract_bits(hsel, 16u - ((8u * b + CM) & 15u), 16u);
  return (WMASK == 0xffffu) ? cw : (cw & WMASK);
}
// the general fast path (every even N; lane.zig GenPlan): the lane's start bit s, its word qb, halfword a, offset lp
template <uint N> inline uint exl3_gen_s(uint L) { return (((8u * L + 1u) * N) >> 4u) + 16u * N - 16u; }
template <uint N> inline uint exl3_gen_qb(uint L) { return exl3_gen_s<N>(L) >> 5u; }
template <uint N> inline uint exl3_gen_a(uint L) { return (exl3_gen_s<N>(L) >> 4u) & 1u; }
template <uint N> inline uint exl3_gen_lp(uint L) { return exl3_gen_s<N>(L) & 15u; }
// window m: C_m = ((m + 1) N >> 4) - (N >> 4) past the lane's start; j = (lp + C_m) >> 4 is J or J + 1
template <uint N, uint WMASK, uint M> inline uint exl3_gen_cw(IntWords w, uint lp) {
  constexpr uint CM = (((M + 1u) * N) >> 4u) - (N >> 4u); constexpr uint J = CM >> 4u;
  const uint pos = lp + CM;
  const uint hsel = ((pos >> 4u) == J) ? w.h[J] : w.h[J + 1u];
  // the 16 bits at offset (pos & 15) from the top: one bit-field extract (bits 16 - r .. 31 - r)
  const uint cw = metal::extract_bits(hsel, 16u - (pos & 15u), 16u);
  return (WMASK == 0xffffu) ? cw : (cw & WMASK);
}
// A tile's eight codewords of one lane through either reader (EXL3_R64 default 1, the faster reader). EXL3_R64 1 (rates whose eight windows span <= 64 bits:
// n <= 108): the 64 stream bits ending at the lane's last window end (three words, two funnels), then each window at a
// compile-time shift from it; wider rates up to n 128 the same from a 96-bit window (four words, three funnels:
// `exl3_rmode` 2). EXL3_R64 0 (or a span past 96 bits, which no admitted rate has): IntWords and the per-lane extract
// above. The lane's start bit s is rebuilt from the plan the kernel already holds (general: qb 32 + a 16 + lp;
// integer-rate: qb 32 + a 16 + b 8 + (K mod 8)), so every reader returns the same bits.
#ifndef EXL3_R64
#define EXL3_R64 1
#endif
#ifndef EXL3_KU
#define EXL3_KU 0
#endif
// the reader's mode for rate N: 0 IntWords; 1 one 64-bit window (the eight windows span <= 64 bits, n <= 108); 2 a 96-bit
// window as three words (n <= 128: the K7 / K8 trunk)
template <uint N> constexpr int exl3_rmode() {
  constexpr uint span = (((8u * N) >> 4u) - (N >> 4u)) + 16u;
  return EXL3_R64 == 0 ? 0 : (span <= 64u ? 1 : (span <= 96u ? 2 : 0));
}
template <uint N> constexpr bool exl3_r64() { return exl3_rmode<N>() != 0; }
template <int MODE> struct Exl3BitsM { IntWords w; };
template <> struct Exl3BitsM<1> { ulong v; };
template <> struct Exl3BitsM<2> { uint lo, mid, hi; };
template <uint N, int IR> inline Exl3BitsM<exl3_rmode<N>()> exl3_tile_load(const device uint* tile, uint qb, uint ga, uint lp) {
  Exl3BitsM<exl3_rmode<N>()> r;
  if constexpr (exl3_rmode<N>() != 0) {
    constexpr uint NW = N / 2u;
    constexpr uint CM7 = ((8u * N) >> 4u) - (N >> 4u);
    const uint s = (IR != 0) ? (qb * 32u + ga * 16u + lp * 8u + ((N / 16u) & 7u)) : (qb * 32u + ga * 16u + lp);
    const uint e = s + CM7 + 16u;
    const uint last = ((e - 1u) >> 5u) % NW;
    const uint w1 = (last + NW - 1u) % NW;
    const uint w2 = (w1 + NW - 1u) % NW;
    const uint sh = (0u - e) & 31u;
    const uint a = tile[last], b = tile[w1], c = tile[w2];
    if constexpr (exl3_rmode<N>() == 1) {
      r.v = as_type<ulong>(uint2(uint(((ulong(b) << 32u) | ulong(a)) >> sh), uint(((ulong(c) << 32u) | ulong(b)) >> sh)));
    } else {
      const uint d = tile[(w2 + NW - 1u) % NW];
      r.lo = uint(((ulong(b) << 32u) | ulong(a)) >> sh);
      r.mid = uint(((ulong(c) << 32u) | ulong(b)) >> sh);
      r.hi = uint(((ulong(d) << 32u) | ulong(c)) >> sh);
    }
  } else {
    r.w = exl3_int_words<N>(tile, qb, ga);
  }
  return r;
}
template <uint N, uint WMASK, uint M, int IR> inline uint exl3_tile_cw(thread const Exl3BitsM<exl3_rmode<N>()>& t, uint lp) {
  constexpr uint SH = (((8u * N) >> 4u) - (N >> 4u)) - ((((M + 1u) * N) >> 4u) - (N >> 4u));
  if constexpr (exl3_rmode<N>() == 1) {
    const uint cw = uint(t.v >> SH) & 0xffffu;
    return (WMASK == 0xffffu) ? cw : (cw & WMASK);
  } else if constexpr (exl3_rmode<N>() == 2) {
    uint cw;
    if constexpr (SH + 16u <= 32u) cw = metal::extract_bits(t.lo, SH, 16u);
    else if constexpr (SH < 32u) cw = ((t.lo >> SH) | (t.mid << (32u - SH))) & 0xffffu;
    else if constexpr (SH + 16u <= 64u) cw = metal::extract_bits(t.mid, SH - 32u, 16u);
    else if constexpr (SH < 64u) cw = ((t.mid >> (SH - 32u)) | (t.hi << (64u - SH))) & 0xffffu;
    else cw = metal::extract_bits(t.hi, SH - 64u, 16u);
    return (WMASK == 0xffffu) ? cw : (cw & WMASK);
  } else {
    if (IR != 0) return exl3_int_cw<N, WMASK, M>(t.w, lp);
    return exl3_gen_cw<N, WMASK, M>(t.w, lp);
  }
}
// MUL1's codeword as its byte sum t in [0, 1020]: the decoded weight is A t + C with A = half 0x1EEE and
// C = 1024 A + half 0xC931 = -3.453125, both exact in f32, so a GEMV can sum x t and apply A and C once per output.
// EXL3_T1V (default 3, the fastest with EXL3_R64) picks how the two pair sums bs = (b0 + b1) | (b2 + b3) << 16 become t, every variant the same exact value:
// 0 a 32-bit multiply by 0x10001 and an integer-to-float convert; 1 a shift and add instead of the multiply; 2 the
// multiply, then the half 1024 + t (0x6400 + t: ulp 1 on [1024, 2048)) converted and 1024 subtracted; 3 shift and add,
// then the half route (no 32-bit multiply after the hash, no integer convert)
#ifndef EXL3_T1V
#define EXL3_T1V 3
#endif
inline float exl3_mul1_t(uint cw) {
  const uint rr = cw * 0x83DCD12Du;
  const uint bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);
#if EXL3_T1V == 1
  return float((bs + (bs >> 16u)) & 0xFFFFu);
#elif EXL3_T1V == 2
  return float(as_type<half>(ushort((bs * 0x00010001u + 0x64000000u) >> 16u))) - 1024.0f;
#elif EXL3_T1V == 3
  return float(as_type<half>(ushort(bs + (bs >> 16u) + 0x6400u))) - 1024.0f;
#else
  return float((bs * 0x00010001u) >> 16u);
#endif
}
// the same t as a half (exact: t <= 1020), for kernels that feed it to the matrix units
inline half exl3_mul1_th(uint cw) {
  const uint rr = cw * 0x83DCD12Du;
  const uint bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);
#if EXL3_T1V == 1
  return static_cast<half>((bs + (bs >> 16u)) & 0xFFFFu);
#elif EXL3_T1V == 2
  return as_type<half>(ushort((bs * 0x00010001u + 0x64000000u) >> 16u)) - half(1024.0h);
#elif EXL3_T1V == 3
  return as_type<half>(ushort(bs + (bs >> 16u) + 0x6400u)) - half(1024.0h);
#else
  return static_cast<half>((bs * 0x00010001u) >> 16u);
#endif
}
template <int CB> inline half exl3_decode(uint cw) {
  if (CB == 1) {
    const uint mixed = cw * 0xCBAC1FEDu; const uint pair = 0x3B603B60u ^ (mixed & 0x8FFF8FFFu);
    return half(float(as_type<half>(ushort(pair & 0xffffu))) + float(as_type<half>(ushort(pair >> 16u))));
  }
  const uint rr = cw * 0x83DCD12Du;
  const uint bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);
#if EXL3_T1V == 1 || EXL3_T1V == 3
  const half hs = as_type<half>(ushort(bs + (bs >> 16u) + 0x6400u));   // the same half 1024 + t without the multiply
#else
  const half hs = as_type<half>(ushort((bs * 0x00010001u + 0x64000000u) >> 16u));
#endif
  return fma(hs, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u)));
}
