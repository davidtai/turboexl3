    // P4: pg2's launch (core/kernels/pg2/gemm_fused_*.metal: MLX's fused NAX GEMM body, its gemm_loop verbatim, with
    // turboexl3's dense finish as the epilogue) over a KDA layer's three projections of one input at prompt rows: member
    // m's y_m OUT_T [M, N_m] = finish(x_m [M, K] w_m [K, N_m]), x_m its prepared rows, w_m its W_hat, svh_m its svh. The
    // grid's N tiles run member 0's, then 1's, then 2's (each N_m a multiple of BN, the host's check), so a threadgroup's
    // tile lies in one member. It reads that member's operands and writes its output (pf_3), and its column counts from
    // the member's first tile (pf_5); every other statement is one of pg2's pieces, so each y_m is its own pg2 launch's,
    // and so its `linear` chain's, word for word.
    const int M = int(x0_shape[0]);
    const int K = int(x0_shape[1]);
    const int N0 = int(w0_shape[1]);
    const int N1 = int(w1_shape[1]);
    const int N2 = int(w2_shape[1]);
    const int T0 = N0 / BN;
    const int T1 = N1 / BN;
    const int lda = K;
    const int tiles_n = T0 + T1 + N2 / BN;
