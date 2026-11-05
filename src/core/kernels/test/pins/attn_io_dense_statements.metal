simd_shuffle_xor
float t0 = v[0] + v[1]
v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]);
xh[j] = float(half(float(x[row * DIMC + i])));
v[j] = xh[j] * float(suh[i]);
v[j] = float(half(float(xv))) * float(suh[i]);
half(v[j] * HAD_SCALE);
xp[row * DIMC + i] = half(v[j] * HAD_SCALE);
v[j] = float(half(float(z[row * DIMC + b * 128u + L + 32u * j])));
v[j] = float(half(float(z[row * dimc + blk * 128u + L + 32u * j])));
OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))))
y[j] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))));
