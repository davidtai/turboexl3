y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))));
OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))))
const float ea = metal::precise::exp(static_cast<float>(A_log[c]));
const float apd = static_cast<float>(a[i]) + static_cast<float>(dt_bias[c]);
const float sc = ea * apd;
const float sy = 1 / (1 + metal::precise::exp(metal::abs(sc)));
const float sg = (sc < 0) ? sy : 1 - sy;
const float gl = sg * lb;
g[i] = metal::precise::exp(gl);
      const float ea = metal::precise::exp(static_cast<float>(A_log[o]));
      const float apd = static_cast<float>(yv) + static_cast<float>(dt_bias[o]);
      const float sc = ea * apd;
      const float sy = 1 / (1 + metal::precise::exp(metal::abs(sc)));
      const float sg = (sc < 0) ? sy : 1 - sy;
      const float gl = sg * lb;
      y1[row * DIMC + o] = metal::precise::exp(gl);
      const float ea = gea[o];
A_log[c]
A_log[o]
a[i]
yv
dt_bias[c]
dt_bias[o]
g[i] = 
y1[row * DIMC + o] = 
      
      const OUT_T yv = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))));
      if (mem != 1) { y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); continue; }
gea[c] = metal::precise::exp(static_cast<float>(A_log[c]));
metal::precise::exp(static_cast<float>(A_log[c]))
