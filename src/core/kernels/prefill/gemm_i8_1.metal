    // int8 GEMM (prefill.Options.int8; the gate|up GEMMs NOPS 2, the down GEMM NOPS 1): z_op [S, ND] f16 = 4 A sum over
    // 128-blocks b of sx[r, b] (xq[r, b] . q_b) + (510 A + C) xs[r], per (table row, op, 64-wide output block). BMR / 32 x 2
    // simdgroups of 32 x 32 outputs, each one mpp matmul2d (32 x 32 x 128, int8 in, int32 out) per 128-deep K block: A from
    // device (the table row's sorted rows, 128 spare rows past S), B the block's trellis decoded to int8 in threadgroup
    // memory ([n][k], double-buffered, every simdgroup decoding); each block's int32 products folded into f32 at the rows'
    // block scales. ROWSCALE (one scale a row, per token): one multiply-accumulate over K folded once at the row's scale
    // (every block slot of sx holds it)
    constexpr uint N = uint(NHW);
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int KDC = int(KD);
    constexpr int NDC = int(ND);
    constexpr int BMC = int(BMR);
    constexpr int NSGC = 2 * BMC / 32;
    constexpr int NBK = KDC / 128;
    constexpr int N16 = NDC / 16;
    constexpr int NTG = NDC / 64;
    constexpr int CAP = 32;
    threadgroup uint4 bq_mem[2 * 64 * 128 / 16];
    threadgroup int8_t* bq = reinterpret_cast<threadgroup int8_t*>(bq_mem);
    threadgroup float sxs[2 * BMC];
    threadgroup float xsum[BMC];
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = sg * 32u + L;
    const int g = int(threadgroup_position_in_grid.x);
    const int tile = g / (int(NOPS) * NTG);
    const int rem = g - tile * (int(NOPS) * NTG);
    const int op = rem / NTG;
    const int n0 = (rem - op * NTG) * 64;
    const int e = tiles[4 * tile];
    const int row0 = tiles[4 * tile + 1];
    const int rows = tiles[4 * tile + 2];
    if (rows == 0) return;
    const int tm = 32 * int(sg / 2u);
    const int tn = 32 * int(sg % 2u);
    const int sgp = min(32, max(0, rows - tm));
    const uint qb = (INTR != 0) ? exl3_int_qb<N>(L) : exl3_gen_qb<N>(L);
    const uint ga = (INTR != 0) ? exl3_int_a<N>(L) : exl3_gen_a<N>(L);
    const uint lp = (INTR != 0) ? exl3_int_b<N>(L) : exl3_gen_lp<N>(L);
