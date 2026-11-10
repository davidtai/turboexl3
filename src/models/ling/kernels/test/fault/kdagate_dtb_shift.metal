    for (uint j = 0; j < 4u; ++j) {
      const uint o = b * 128u + L + 32u * j;
      if (mem != 1) { y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); continue; }
      // the planted fault (a red check, never served): the gate reads the next channel's dt_bias
      const OUT_T yv = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))));
      const float ea = metal::precise::exp(static_cast<float>(A_log[o]));
      const float apd = static_cast<float>(yv) + static_cast<float>(dt_bias[(o + 1u) % DIMC]);
      const float sc = ea * apd;
      const float sy = 1 / (1 + metal::precise::exp(metal::abs(sc)));
      const float sg = (sc < 0) ? sy : 1 - sy;
      const float gl = sg * lb;
      y1[row * DIMC + o] = metal::precise::exp(gl);
    }
    }
