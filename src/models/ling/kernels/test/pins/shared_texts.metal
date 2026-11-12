OUT_T(float(half(v[j] * HAD_SCALE * float(svhm[o - mo0 * 16u]))))
OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))))
const device uint* tile = wm + (ulong(tk) * mtiles + (blk * BTC + t - mo0)) * ulong(NW);
const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * BTC + t) * ulong(NW);
for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
{ float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
const half hv = half(v[j] * HAD_SCALE);
const uint m = (r & 1u) | ((r >> 3u) << 1u);
sum = simd_sum(sum);
v[j] = float(half(float(h))) * float(suh2[o]);
float(half(float(x[row * DIMC + i]))) * float(suh[i])
const OUT_T sig = sigtab[as_type<ushort>(gt)];
const OUT_T act = gt * sig;
const OUT_T h = act * ut;
const float sig = sig32[as_type<ushort>(gt)];
const float act = g * sig;
const float am = metal::isnan(act) ? act : (act < hi ? act : hi);
const float um = metal::isnan(u) ? u : (u > lo ? u : lo);
const float uc = metal::isnan(um) ? um : (um < hi ? um : hi);
const float outv = am * uc;
const OUT_T h = OUT_T(outv);
arrived_last = (old == 1u) ? 1u : 0u;
(device atomic_uint*)pcnt + (pb * NCHC + chunk)
const uint old = atomic_fetch_add_explicit((device atomic_uint*)cnt + (ob * NCHC + chunk), 1u, memory_order_relaxed);
