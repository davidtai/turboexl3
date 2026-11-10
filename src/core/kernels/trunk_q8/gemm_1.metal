    // trunk int8 (trunk_q8.zig, Config.pf_trunk_int8; lossy): y OUT_T [M, N] = finish(z), z half [M, N] = 4 A sx[m]
    // (xq[m] . wq[n]) + (510 A + C) xsum[m] (the routed int8 GEMM's epilogue at one scale a row, gemm_i8_2.metal's
    // ROWSCALE), never in device memory. The dot products: one mpp matmul2d of BM x BN x 64 (int8 in, int32 out,
    // multiply-accumulate over K: exact integer sums) a threadgroup on its WM WN simdgroups, A xq [MP, K] (MP the rows
    // padded to BM: whole tiles read, the pad rows never stored) and B wq [N, K] (k fastest) from device memory
    // (mlx-serve's w8a8 GEMM body, kernels/dit/w8a8_gemm_body.metal). z's halves are staged in threadgroup memory at row
    // stride BN, then pg2's epilogue (gemm_fused_2.metal) runs the finish per (row, 128-column block) of the tile. N a
    // multiple of BN, BN of 128, K of 64 (trunk_q8.takes).
    const int M = int(sx_shape[0]);
    const int K = int(xq_shape[1]);
    const int MP = int(xq_shape[0]);
    const int N = int(wq_shape[0]);
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const int c_row = int(threadgroup_position_in_grid.y) * BM;
    const int c_col = int(threadgroup_position_in_grid.x) * BN;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(BM, BN, 64, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<WM * WN>> op;
    metal::tensor<device int8_t, metal::dextents<int32_t, 2>, metal::tensor_inline> ta((device int8_t*)xq, metal::dextents<int32_t, 2>(K, MP));
    metal::tensor<device int8_t, metal::dextents<int32_t, 2>, metal::tensor_inline> tb((device int8_t*)wq, metal::dextents<int32_t, 2>(K, N));
    auto a0 = ta.template slice<64, BM>(0, c_row);
    auto b0 = tb.template slice<64, BN>(0, c_col);
    auto ct = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
    for (uint16_t i = 0; i < ct.get_capacity(); ++i) ct[i] = 0;
    for (int k0 = 0; k0 < K; k0 += 64) {
        auto a = ta.template slice<64, BM>(k0, c_row);
        auto b = tb.template slice<64, BN>(k0, c_col);
        op.run(a, b, ct);
    }
    // the epilogue: z's halves in threadgroup memory (a pad row's at the last row's scale, never stored)
    const float MA = float(as_type<half>(ushort(0x1EEEu)));
    const float MC = 1024.0f * MA + float(as_type<half>(ushort(0xC931u)));
    threadgroup half zt[BM * BN];
    for (uint16_t i = 0; i < ct.get_capacity(); ++i) {
        const auto ids = ct.get_multidimensional_index(i);
        const int m = min(c_row + int(ids[1]), M - 1);
        zt[int(ids[1]) * BN + int(ids[0])] = half(fma(float(ct[i]) * sx[m], 4.0f * MA, (510.0f * MA + MC) * xsum[m]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
