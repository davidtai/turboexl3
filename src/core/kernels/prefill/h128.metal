inline void turboexl3_h128x4(thread float* v, uint L) {
  for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
  const float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3];
  v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3;
}
// the same H128 with lane L holding elements 4L..4L+3: strides 1 and 2 in the lane, 4..64 across lanes (lane bit = stride / 4),
// in turboexl3_h128x4's level order with its lower a + b / upper a - b, so the same f32 value at every element
inline void turboexl3_h128v4(thread float* v, uint L) {
  const float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2; v[1] = a1 + a3; v[2] = a0 - a2; v[3] = a1 - a3;
  for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
}
// a 128-block's sum in the strided kernels' order from the vector layout: lane L adds elements L, L + 32, L + 64, L + 96 (lanes
// L / 4 + 8 j, component L % 4) in that order, then simd_sum: the same per-lane partials, so the same sum
template <int XSO> inline float turboexl3_sum_x4(half4 h, uint L) {
  // XSO 1: the lane's own 4 elements in order, then simd_sum (no shuffles; a different f32 order than the strided kernels')
  if (XSO != 0) return simd_sum(((float(h[0]) + float(h[1])) + float(h[2])) + float(h[3]));
  float s = 0.0f;
  for (uint j = 0; j < 4u; ++j) { const half4 o = simd_shuffle(h, ushort((L >> 2u) + 8u * j)); s += float(o[L & 3u]); }
  return simd_sum(s);
}
// `Options.alay`: the slot of element j of a 32-block in the GEMMs' A layout (`prefill.alayPos`): fragment lane q = (j >> 1) & 3
// holds k 2q, 2q + 1, 2q + 8, 2q + 9 of each 16-deep slice, stored at 8q + 4 (j >> 4) + {0, 1, 2, 3} (alay 1), or at
// 16 (j >> 4) + 4q + {0, 1, 2, 3} (alay 2: per 16-block, `prefill.alayPos16`)
inline uint turboexl3_alay(uint j) { return 8u * ((j >> 1u) & 3u) + 4u * (j >> 4u) + ((j & 1u) | (((j >> 3u) & 1u) << 1u)); }
inline uint turboexl3_alay16(uint j) { return (j & 16u) + 4u * ((j >> 1u) & 3u) + ((j & 1u) | (((j >> 3u) & 1u) << 1u)); }
