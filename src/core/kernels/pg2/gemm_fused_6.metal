    const size_t c_row_long = size_t(c_row);
    const size_t c_col_long = size_t(c_col);
    A += c_row_long * lda;
    B += c_col_long;
    constexpr short SM = BM / WM;
    constexpr short SN = BN / WN;
    constexpr short SK = 32;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    const short tm = SM * (simd_group_id / WN);
    const short tn = SN * (simd_group_id % WN);
    const int sgp_sm_int =
        align_M ? int(SM) : min(int(SM), M - (c_row + tm));
    const short sgp_sm = short(sgp_sm_int);
    const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
    const int sgp_sn_int =
        align_N ? int(SN) : min(int(SN), N - (c_col + tn));
    const short sgp_sn = short(sgp_sn_int);
    const bool is_unaligned_sn = align_N ? false : (sgp_sn != SN);
    A += tm * lda;
    B += tn;
    NAXTile<float, TM, TN> Dtile;
    dispatch_bool(align_K, [&](auto kAlignedK) {
      dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
        dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
          Dtile = pg1_gemm_loop<
              T,
              SM,
              SN,
              SK,
              BK,
              false,
              false,
              kAlignedM.value,
              kAlignedN.value,
              kAlignedK.value,
              float>(
              A,
              B,
              lda,
              ldb,
              K,
              gemm_k_iterations_aligned,
              sgp_sm,
              sgp_sn);
        });
      });
    });
    // The epilogue. The tile's halves (MLX's store rounding) in threadgroup memory, row stride BN.
    threadgroup half zt[BM * BN];
    Dtile.template store<half, BN, 1>(&zt[tm * BN + tn]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // turboexl3's dense finish per (row, 128-column block) of the tile: its statements, z's halves read from zt
    const uint DIMC = uint(N);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    constexpr uint NB = uint(BN) / 128u;
    for (uint it = simd_group_id; it < uint(BM) * NB; it += uint(WM * WN)) {
      const uint rt = it / NB;
      const uint bt = it % NB;
      // a row past M (the last row tile's): no output (the whole simdgroup skips it)
      if (c_row + int(rt) >= M) continue;
      const uint b = uint(c_col) / 128u + bt;
      const uint row = uint(c_row) + rt;
    const uint L = thread_index_in_simdgroup;
    float v[4];
    for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(zt[rt * uint(BN) + bt * 128u + L + 32u * j])));
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    for (uint j = 0; j < 4u; ++j) { const uint o = b * 128u + L + 32u * j; y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
    }
