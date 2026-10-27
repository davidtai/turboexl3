uint row = thread_position_in_grid.y;
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;

threadgroup float rk[NE];

threadgroup uint sel[NK];

size_t rbase = (size_t)row * (size_t)NE;
for (uint e = tid; e < NE; e += TG) {
  rk[e] = float(logits[rbase + e]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Selection runs on simdgroup 0 only: K rounds of "max, then mask".
// simd_min over the candidate indices makes the tie-break the LOWEST
// expert id, so the pick is a pure function of the keys.
if (tid < 32) {

