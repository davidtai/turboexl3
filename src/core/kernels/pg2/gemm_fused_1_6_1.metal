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
