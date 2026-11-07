  if (R == 1) {
    // One row: MLX's gemv (gemv.h GEMVKernel::run at BM 4, BN 1, SM 1, SN 32, TM 4, TN 4: its route for bf16 x [1, HK]
    // times pw transposed, NE < 4096, 64 < HK < 16 NE), threadgroup ptg of NE / 16, its statements in order. HK % 128 == 0
    // (no leftover block) and NE % 16 == 0 (no tail rows): the host checks both.
    const device bfloat16_t* mat = pw;
