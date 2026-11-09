// The rows launch's threadgroups (ling/router.zig, Config.router_rows: 2 rows or more over the f16 weight), 128 threads
// each: first the router's, NE / 16 for each group of RG rows of x (up to 4: the R rows in NRG groups), each threadgroup
// 16 logits of each row of its group with every word of the table read once for them; then the predictor's
// (pred_open_rows), which take every row as in router/head's launch
constexpr int RG = (R < 4) ? R : 4;
constexpr int NRG = (R + RG - 1) / RG;
const uint tg = threadgroup_position_in_grid.x;
const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;
if (tg < uint(NRG * (NE / 16))) {
  const uint grp = tg / uint(NE / 16);
  const uint blk = tg % uint(NE / 16);
