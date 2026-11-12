    // pg2: pg1's copy of MLX's fused NAX GEMM kernel (steel/gemm/kernels/steel_gemm_fused_nax.h `gemm`, MLX 0.32.3; its
    // gemm_loop verbatim) at nn, x and w half, with turboexl3's dense finish (kernels/dense/finish.metal) as its
    // epilogue: y OUT_T [M, N] = finish(z), z = x [M, K] w [K, N] never in device memory. The GEMM's statements are pg1's
    // (MLX's params from the operands' shapes; BM BN BK WM WN SWZ and AM AN AK template constants). The epilogue rounds
    // the accumulators to half as MLX's store rounds them (NAXFrag::store's static_cast), stages them in threadgroup
    // memory at the simdgroup's (tm, tn), and runs the finish's statements per (row, 128-column block) of the tile, a
    // simdgroup a block at a time, the block's halves read from threadgroup memory instead of z. N a multiple of BN, BN of
    // 128 (the host's check).
    const int M = int(x_shape[0]);
    const int K = int(x_shape[1]);
    const int N = int(w_shape[1]);
    const int lda = K;
    const int ldb = N;
    const int tiles_n = (N + BN - 1) / BN;
