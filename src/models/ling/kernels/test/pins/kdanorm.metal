//! copied_from: mlx-serve kernels/gdn/gdn_normgate.metal @ bc6c5f2c
uint base = (row * uint(HV) + head) * uint(DV) + lane * 4;
float xs[4];
float sumsq = 0.0f;
for (uint i = 0; i < 4; ++i) {
xs[i] = float(y[base + i]);
sumsq += xs[i] * xs[i];
sumsq = simd_sum(sumsq);
float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
const T normed = norm_w[c] * T(xc * inv);
uint zbase = row * uint(ZSTRIDE) + uint(ZOFF) + head * uint(DV) + lane * 4;
const T zv = z[zbase + i];
const T zv = z[row * uint(ZSTRIDE) + uint(ZOFF) + head * uint(DV) + c];
if constexpr (TAB) sig = sigtab[as_type<ushort>(zv)];
sig = sigtab[as_type<ushort>(zv)];
out[base + i] = SWISH ? (zv * sig) * normed : normed * sig;
const T ov = normed * sig;
normed * sig
row * uint(ZSTRIDE) + uint(ZOFF) + head * uint(DV) + 
norm_w[lane * 4 + i]
norm_w[c]
T(xs[i] * inv)
T(xc * inv)
v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]);
v[j] = float(half(float(ov))) * float(suh[b * 128u + c]);
float(half(float(
) * float(suh[
const T zv = z[
c];
T sig;
const float xc = float(y[(row * uint(HV) + head) * uint(DV) + c]);
(row * uint(HV) + head) * uint(DV) + 
c]);
const uint i = b * 128u + L + 32u * j;
const uint c = L + 32u * j;
const uint lane = L;
const uint head = b;
uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;
uint head = threadgroup_position_in_grid.z;
const uint b = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.y;
const uint L = thread_index_in_simdgroup;
