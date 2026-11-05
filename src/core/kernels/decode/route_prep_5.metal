threadgroup_barrier(mem_flags::mem_threadgroup);
// phase 2: prep2 for row-slot (row, sgi) and block b; block 0's last simdgroup: the router's weights, slots and scores
if (sgi < uint(NK)) {
  constexpr uint DIMC = uint(DIM);
  constexpr uint TOPKC = uint(TOPK);
  constexpr float HAD_SCALE = 0.08838834764831845f;
  const uint rk = row * TOPKC + sgi;
  const uint L = lane;
  const uint e = sel[sgi];
