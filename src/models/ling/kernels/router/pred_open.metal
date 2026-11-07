} else {
  // The predictor's threadgroups: the next MoE layer's router pw (bf16 [NE, HK] row-major, the matmul's b transposed)
  // times every row of x, each sum rounded to bf16 (the product's dtype), widened to f32 (the astype) and scored with
  // the next layer's bias pb (turboexl3_ling_score: the header's)
  const uint ptg = tg - uint(R * (NE / 16));
