// The launch's threadgroups, 128 threads each: first the router's, NE / 16 a row of x (MLX's gemv tiles, 16 logits
// each) for the R rows; then the predictor's, which take every row (pred_open)
const uint tg = threadgroup_position_in_grid.x;
const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;
