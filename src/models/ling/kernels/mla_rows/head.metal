// MLX's gemv (gemv.h `GEMVKernel::run`, MLX 0.32) at SM 1, SN 32, TM 4, TN 4 and the BM, BN that MLX's matmul takes
// for a weight of OUT rows of IN (models/ling/mla_rows.zig `tileFor`), over w bf16 [OUT, IN] row-major (the host's
// loaded [IN, OUT] column-major view, transposed back) for up to RB rows of x bf16 [ROWS, IN] a threadgroup:
// threadgroup (t, c) runs MLX's threadgroup t for rows c * RB .. of x. Each lane loads its weight tile once a K block
// and applies it to each of the chunk's rows. Every row keeps its own accumulators and takes MLX's statements in MLX's
// order: the same products, added in the same order, the same shuffle-down tree and threadgroup sum, so its words are
// MLX's one-row gemv's. IN % (32 BN TN) == 0 and OUT % (BM TM) == 0 (the host checks both): no leftover block and no
// tail simdgroup. A spare accumulator row of a short last chunk reads the last row of x and is never stored.
constexpr int SM = 1;
constexpr int SN = 32;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int blockM = BM * SM * TM;
constexpr int blockN = BN * SN * TN;
constexpr int TGP = BN > 1 ? RB * BN * (blockM + TM) : 1;
threadgroup float tgp_memory[TGP];
const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;
const int thrM = 0;
const int thrN = int(simd_lid);
const int sgN = BN != 1 ? int(simd_gid % uint(BN)) : 0;
const int simdM = BN != 1 ? SM * int(simd_gid / uint(BN)) : SM * int(simd_gid);
const int simdN = BN != 1 ? SN * int(simd_gid % uint(BN)) : 0;
const int bm = (simdM + thrM) * TM;
int bn = (simdN + thrN) * TN;
const int out_row = int(threadgroup_position_in_grid.x) * blockM + bm;
const device bfloat16_t* mat = w + size_t(out_row) * size_t(IN);
const int r0 = int(threadgroup_position_in_grid.y) * RB;
float result[RB][TM];
for (int r = 0; r < RB; r++) {
  for (int tm = 0; tm < TM; tm++) {
    result[r][tm] = 0;
  }
}
for (int i = 0; i < IN / blockN; ++i) {
  // the K block's weight tile, read once for the chunk's rows: MLX's `inter` of each of the lane's TM rows
  bfloat16_t inter[TM][TN];
  for (int tm = 0; tm < TM; tm++) {
    for (int tn = 0; tn < TN; tn++) {
      inter[tm][tn] = static_cast<bfloat16_t>(mat[tm * IN + bn + tn]);
    }
  }
  for (int r = 0; r < RB; r++) {
    const device bfloat16_t* in_vec = x + size_t(min(r0 + r, ROWS - 1)) * size_t(IN);
    float v_coeff[TN];
    for (int tn = 0; tn < TN; tn++) {
      v_coeff[tn] = static_cast<float>(in_vec[bn + tn]);
    }
    for (int tm = 0; tm < TM; tm++) {
      for (int tn = 0; tn < TN; tn++) {
