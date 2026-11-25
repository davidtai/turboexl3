OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))))
const uint o = b * 128u + L + 32u * j;
acc += thread_x[i] * thread_x[i];
acc = simd_sum(acc);
acc = simd_sum(local_sums[simd_lane_id]);
local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
out[i] = wr[i] * static_cast<float>(thread_x[i] * local_inv_mean[0]);
const float qn = w[e] * static_cast<float>(thread_x[i] * local_inv_mean[0]);
v[j] = float(half(ys[i])) * float(suh[i]);
v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]);
H_T((yv * sg) * uv)
hp[j] = float(half(float(H_T((yv * sg) * uv)))) * float(suh[o]);
k_pe[ulong(row) * uint(ROPE) + (c - NOPEC)] = second ? rx2 : rx1;
const float pv = simd_shuffle_xor(qv, ushort(1));
float inv_freq = metal::precise::exp2(-d * base);
