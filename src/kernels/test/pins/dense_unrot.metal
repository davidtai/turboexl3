for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
{ float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
z[row * DIMC + b * 128u + L + 32u * j]
w[ulong(row) * DIMC + b * 128u + L + 32u * j]
v[j] = t[(L + 32u * j) * 33u + c];
OUT_T(t[(e >> 5u) * 33u + (e & 31u)] * 0.0078125f * float(suh[i]) * float(svh[o]))
