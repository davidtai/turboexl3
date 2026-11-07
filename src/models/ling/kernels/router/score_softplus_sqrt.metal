// The predictor's score of a product word z (bf16, widened to f32) with the next layer's bias b, DeepSeek's form (the
// host streamer's nextScores: add(sqrt(logaddexp(z, 0)), bias)): MLX's LogAddExp with z its first operand
// (binary_ops.h), its log1p (utils.h), Sqrt (unary_ops.h) and Add, each kernel's statements. MLX's metallib is built
// -fno-fast-math, so the bare exp and log there are the precise ones; this text compiles at MathMode::Safe, where the
// bare names lower to the fast ones, so it spells them precise. No product is contracted into the sum after it: MLX
// rounds log1p's product before LogAddExp adds it.
inline float turboexl3_ling_log1p(float x) {
  #pragma METAL fp contract(off)
  float xp1 = 1.0f + x;
  if (xp1 == metal::numeric_limits<float>::infinity()) {
    return metal::numeric_limits<float>::infinity();
  }
  if (xp1 == 1.0f) {
    return x;
  }
  return x * (metal::precise::log(xp1) / (xp1 - 1.0f));
}

inline float turboexl3_ling_score(float z, float b) {
  #pragma METAL fp contract(off)
  const float y = 0.0f;
  float s;
  if (metal::isnan(z) || metal::isnan(y)) {
    s = metal::numeric_limits<float>::quiet_NaN();
  } else {
    constexpr float inf = metal::numeric_limits<float>::infinity();
    float maxval = metal::max(z, y);
    float minval = metal::min(z, y);
    s = (minval == -inf || maxval == inf) ? maxval : (maxval + turboexl3_ling_log1p(metal::precise::exp(minval - maxval)));
  }
  float q = metal::precise::sqrt(s);
  return q + b;
}
