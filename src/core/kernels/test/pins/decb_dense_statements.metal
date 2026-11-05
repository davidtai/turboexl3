OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))))
if (DEFER != 0) for (uint q = 0; q < INB; ++q) xsum += xs[row * INB + q];
if (DEFER != 0) for (uint q = 0; q < INB; ++q) xsum += xs[rq * INB + q];
float acc = z[ulong(row) * ulong(DIMC) + o];
float acc = z[ulong(rq) * ulong(DIMC) + o];
for (uint k = 1u; k < KSC; ++k) acc += z[(ulong(k) * ulong(ROWSC) + row) * ulong(DIMC) + o];
for (uint k = 1u; k < KSC; ++k) acc += z[(ulong(k) * ulong(ROWSC) + rq) * ulong(DIMC) + o];
if (DEFER != 0) acc = fma(acc, MUL1_A, MUL1_C * xsum);
v[j] = float(half(acc));
const uint o = b * 128u + L + 32u * j;
constexpr
threadgroup_position_in_grid
thread_index_in_simdgroup
float xsum = 0.0f;
float v[4];
for (uint j = 0; j < 4u; ++j) {
const T yv = xv * sigtab[as_type<ushort>(gv)];
v[j] = float(half(float(yv))) * float(suh[i]);
gst[d] = xv * sigtab[as_type<ushort>(gv)];
v[j] = float(half(float(gst[sgi * 128u + L + 32u * j]))) * float(suh[i]); }
const T xv = T(acc / Z);
= T(acc / Z);
x[row * DIMC + i]
for (uint g = (row * NB + b) * 32u + L; g < uint(NCNT); g += uint(NTH)) cnt[g] = 0u;
