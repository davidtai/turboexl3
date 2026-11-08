OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))))
const uint o = b * 128u + L + 32u * j;
const OUT_T gt = gate[row * DIMC + o];
y[row * DIMC + o] = h;
const float hi = lim[0];
