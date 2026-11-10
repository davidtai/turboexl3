// L6 (G1): a KDA layer's exp(A_log) per channel once, the bounded gate's first statement (ling_kda_gate.metal's): the
// gate epilogue's second form reads gea in place of computing it for each element
const uint c = thread_position_in_grid.x;
if (c >= uint(A_log_shape[0])) {
  return;
}
gea[c] = metal::precise::exp(static_cast<float>(A_log[c]));
