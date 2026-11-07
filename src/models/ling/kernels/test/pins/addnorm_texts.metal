for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
{ float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
const uint q = (r & 7u) >> 1u;
const uint m = (r & 1u) | ((r >> 3u) << 1u);
const half hv = half(v[j] * HAD_SCALE);
xp[row * DIMC + ((LANE_ORDER != 0) ? ((i & ~15u) + 4u * q + m) : i)] = hv;
sum += float(hv);
sum = simd_sum(sum);
if (L == 0u) xs[row * NB + b] = sum;
v[j] = float(half(float(x[i]))) * float(suh[i]);
v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]);
for (uint g = (row * NB + b) * 32u + L; g < uint(NCNT); g += uint(NTH)) cnt[g] = 0u;
suh0, jxp, jxs, 0u * uint(ROWS) + r
suh1, jxp, jxs, 1u * uint(ROWS) + r
suh2, jxp, jxs, 2u * uint(ROWS) + r
constexpr int NC = (D + TN * 4 - 1) / (TN * 4);
delta = float(X[at]);
const T hn = T(float(H[at]) + delta);
H_NEW[at] = hn;
hv[p * 4 + i] = float(hn);
acc += hv[p * 4 + i] * hv[p * 4 + i];
acc = simd_sum(acc);
if (sg == 0) local_sums[lane] = 0.0f;
if (lane == 0) local_sums[sg] = acc;
const float total = simd_sum(local_sums[lane]);
const float inv = metal::precise::rsqrt(total / float(D) + eps);
T(float(W[c]) * float(T(hv[p * 4 + i] * inv)))
