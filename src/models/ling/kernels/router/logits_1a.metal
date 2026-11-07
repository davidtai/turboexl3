  // MLX's gemv (gemv.h GEMVKernel::run at BM 4, BN 1, SM 1, SN 32, TM 4, TN 4) over the router's table as the host's
  // loader builds it: [HK, NE] column-major (strides 1, HK: the astype and the + 0 of the weight's transposed view keep
  // its layout), which MLX's matmul reads as b transposed, its memory w [NE, HK] row-major (NE < 4096, 64 < HK < 16 NE).
  // Threadgroup blk of row's NE / 16, its statements in order. MLX's matmul first widens x to f32 (an astype); here x
  // is read as bf16 and widened in registers, the same values. HK % 128 == 0 (no leftover block) and NE % 16 == 0 (no
  // tail rows): the host checks both.
