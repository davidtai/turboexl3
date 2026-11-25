OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))))
const uint o = b * 128u + L + 32u * j;
float costheta = metal::fast::cos(theta);
float sintheta = metal::fast::sin(theta);
float rx1 = x1 * costheta - x2 * sintheta;
float rx2 = x1 * sintheta + x2 * costheta;
float theta = Lp * inv_freq;
float inv_freq = metal::precise::exp2(-d * base);
float d = static_cast<float>(ip) / static_cast<float>(RH);
const float pv = simd_shuffle_xor(qv, ushort(1));
