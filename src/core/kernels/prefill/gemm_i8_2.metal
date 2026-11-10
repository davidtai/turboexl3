    const device uint* wb = wsel + (ulong(e) * ulong((KDC / 16) * N16) + ulong(n0 / 16)) * ulong(N / 2u);
    const device float* sxr = sxsel + ulong(row0) * ulong(NBK);
    const device float* xsr = xssel + ulong(row0) * ulong(NBK);
    metal::tensor<device int8_t, metal::dextents<int32_t, 2>, metal::tensor_inline> ta((device int8_t*)(xqsel + ulong(row0) * ulong(KDC)), metal::dextents<int32_t, 2>(KDC, BMC));
    metal::tensor<threadgroup int8_t, metal::dextents<int32_t, 2>, metal::tensor_inline> tb0(bq, metal::dextents<int32_t, 2>(128, 64));
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(32, 32, 128, false, true, false, (ROWSCALE != 0) ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate : mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> mm;
    auto a0 = ta.template slice<128, 32>(0, tm);
    auto b0 = tb0.template slice<128, 32>(0, tn);
    auto ct = mm.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
    float acc[CAP];
    for (uint16_t i = 0; i < uint16_t(CAP); ++i) {
        acc[i] = 0.0f;
        ct[i] = 0;
    }
    // block 0's weights and scales, and each row's input sum (exl3pf_affine's order)
    exl3pf_stage_q8<N, WMASK, int(INTR), N16, NSGC>(wb, bq, sg, qb, ga, lp, L);
    if (tid < uint(BMC)) {
        if (ROWSCALE == 0) sxs[tid] = sxr[ulong(tid) * ulong(NBK)];
        float s = 0.0f;
        if (int(tid) < rows) for (int b = 0; b < NBK; ++b) s += xsr[ulong(tid) * ulong(NBK) + ulong(b)];
        xsum[tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kb = 0; kb < NBK; ++kb) {
        const int cur = kb & 1;
        if (kb + 1 < NBK) {
            exl3pf_stage_q8<N, WMASK, int(INTR), N16, NSGC>(wb + ulong(8 * (kb + 1) * N16) * ulong(N / 2u), bq + (cur ^ 1) * (64 * 128), sg, qb, ga, lp, L);
            if (ROWSCALE == 0 && tid < uint(BMC)) sxs[(cur ^ 1) * BMC + int(tid)] = sxr[ulong(tid) * ulong(NBK) + ulong(kb + 1)];
        }
        if (sgp > 0) {
            metal::tensor<threadgroup int8_t, metal::dextents<int32_t, 2>, metal::tensor_inline> tb(bq + cur * (64 * 128), metal::dextents<int32_t, 2>(128, 64));
            auto a = ta.template slice<128, 32>(kb * 128, tm);
            auto b = tb.template slice<128, 32>(0, tn);
            mm.run(a, b, ct);
            if (ROWSCALE == 0) {
                for (uint16_t i = 0; i < uint16_t(CAP); ++i) {
                    const int m = int(ct.get_multidimensional_index(i)[1]);
                    acc[i] = fma(float(ct[i]), sxs[cur * BMC + tm + m], acc[i]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sgp > 0) {
        const float MA = float(as_type<half>(ushort(0x1EEEu)));
        const float MC = 1024.0f * MA + float(as_type<half>(ushort(0xC931u)));
        device half* z = zsel;
        for (uint16_t i = 0; i < uint16_t(CAP); ++i) {
            const auto ids = ct.get_multidimensional_index(i);
            const int m = tm + int(ids[1]);
            const float v = (ROWSCALE != 0) ? float(ct[i]) * sxr[ulong(m) * ulong(NBK)] : acc[i];
            if (m < rows) z[ulong(row0 + m) * ulong(NDC) + ulong(n0 + tn + int(ids[0]))] = half(fma(v, 4.0f * MA, (510.0f * MA + MC) * xsum[m]));
        }
    }
