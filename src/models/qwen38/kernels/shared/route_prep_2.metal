size_t rbase = (size_t)row * (size_t)NE;
for (uint e = tid; e < NE; e += TG) {
  rk[e] = float(logits[rbase + e]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// phase 1: the selection (simdgroup 0), the softmax tree (block 0's last simdgroup) [, the shared gate's dot (block 0's 1..8)]
if (tid < 32) {
