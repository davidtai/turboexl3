const auto ww = exl3_tile_load<N, 0>(tile, qb, ga, lp);
const float wv = (DEFER != 0) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
for (uint r = 0; r < RBC; ++r) a[r][t][m >> 2] = fma(xr[r][m & 3u], wv, a[r][t][m >> 2]);
for (uint r = 0; r < RBC; ++r) turboexl3_fold<BTC>(a[r], &red[sg][r * 16u * BTC], L);
= ((red[0][i] + red[1][i]) + red[2][i]) + red[3][i];
const half4 x4 = *reinterpret_cast<const device half4*>(xbase + ulong(row) * ulong(INC) + tk * 16u);
xr[r][0] = float(x4.x); xr[r][1] = float(x4.y); xr[r][2] = float(x4.z); xr[r][3] = float(x4.w);
acc += z
if (DEFER != 0) acc = fma(acc, MUL1_A, MUL1_C * xsum);
v[j] = float(half(acc));
for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
{ float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))));
const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
for (uint k = 1u; k < KSC; ++k) acc += z
if (DEFER != 0) for (uint q = 0; q < INB; ++q) xsum += xs[row * INB + q];
if (DEFER != 0) for (uint q = 0; q < NB; ++q) xsum += xs[row * NB + q];
v[j] = float(half(float(x[row * 
]))) * float(suh[i]); }
const uint q = (r & 7u) >> 1u;
const uint m = (r & 1u) | ((r >> 3u) << 1u);
const half hv = half(v[j] * HAD_SCALE);
sum += float(hv);
sum = simd_sum(sum);
((i & ~15u) + 4u * q + m)
atomic_fetch_add_explicit((device atomic_uint*)cnt + (ob * NCHC + chunk), 1u, memory_order_relaxed)
device TURBOEXL3_COH float* zc = (device TURBOEXL3_COH float*)z;
atomic<float>
atomic_float
