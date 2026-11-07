// The predictor's score of a product word z (bf16, widened to f32) with the next layer's bias b, Ling's own form (the
// router's sigmoid + bias; candidate C1, turboexl3_ling's MLX_SERVE_LING_PREDICTOR=1: add(sigmoid(z), bias)): MLX's
// Sigmoid (unary_ops.h, its exp spelled precise there) and Add, each kernel's statements.
inline float turboexl3_ling_score(float z, float b) {
  #pragma METAL fp contract(off)
  auto y = 1 / (1 + metal::precise::exp(metal::abs(z)));
  float s = (z < 0) ? y : 1 - y;
  return s + b;
}
