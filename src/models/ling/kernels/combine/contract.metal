// Ling's routed combine: no product and sum contracted into one FMA, so every product rounds to f32 on its own (MLX's
// Multiply kernel) before the sum reads it (MLX's col_reduce_small)
#pragma METAL fp contract(off)
