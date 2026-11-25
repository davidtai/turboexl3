// one 16 x 32 x 16 multiply-accumulate on the matrix units, C[16, 32] += A[16, 16] B[16, 32]: A's fragment, B^T's two
// 16-column fragments and C's two as BaseNAXFrag lays them out (steel's NAXFrag::mma, two output fragments), at strict
// precision (relaxed_precision false: half x half products exact in f32, f32 accumulation)
inline void turboexl3_mma16(thread vec<float, 8>& c0, thread vec<float, 8>& c1, thread const vec<half, 8>& a, thread const vec<half, 8>& b0, thread const vec<half, 8>& b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
  auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto cc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  for (short i = 0; i < 8; ++i) ca[i] = a[i];
  for (short i = 0; i < 8; ++i) { cb[i] = b0[i]; cb[8 + i] = b1[i]; }
  for (short i = 0; i < 8; ++i) { cc[i] = c0[i]; cc[8 + i] = c1[i]; }
  op.run(ca, cb, cc);
  for (short i = 0; i < 8; ++i) { c0[i] = cc[i]; c1[i] = cc[8 + i]; }
}
