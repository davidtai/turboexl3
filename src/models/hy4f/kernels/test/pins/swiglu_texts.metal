OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))))
const uint o = b * 128u + L + 32u * j;
const float yv = static_cast<float>(gate[ulong(row) * DIMC + o]);
h[ulong(row) * DIMC + o] = H_T((yv * sg) * uv);
