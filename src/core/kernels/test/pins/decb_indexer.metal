uint base = L * 4;
float v = float(src[base + i]);
acc += v * v;
acc = simd_sum(acc);
float inv = metal::precise::rsqrt(acc / float(IHD) + float(qeps));
nrm[i] = qn[base + i] * OUT_T(float(src[base + i]) * inv);
constexpr uint half_rd = uint(RD) / 2;
partner[i] = simd_shuffle(float(nrm[i]), L ^ (half_rd / 4));
r = OUT_T(float(nrm[i]) * arow[j] - partner[i] * arow[half_rd + j]);
r = OUT_T(partner[i] * arow[j] + float(nrm[i]) * arow[j - half_rd]);
r = nrm[i];
float acc = 0.0f;
const OUT_T pooled = OUT_T(acc * invr);
pxs[j] = float(pooled);
sumsq += pxs[j] * pxs[j];
sumsq = simd_sum(sumsq);
float pinv = metal::precise::rsqrt(sumsq / 128.0f + peps);
const OUT_T nv = kn[lane * 4 + j] * OUT_T(pxs[j] * pinv);
const float nf = float(nv);
const float up = simd_shuffle_down(nf, ushort(HALF / 4));
const float dn = simd_shuffle_up(nf, ushort(HALF / 4));
const OUT_T rot = pd < uint(HALF) ? OUT_T(-up) : OUT_T(dn);
const OUT_T pa = OUT_T(nf * float(pcos[pd]));
const OUT_T pb = OUT_T(float(rot) * float(psin[pd]));
po = OUT_T(float(pa) + float(pb));
+ acc;
for (uint r = 0; r < uint(RATIO); ++r) acc = float(
simd_sum
