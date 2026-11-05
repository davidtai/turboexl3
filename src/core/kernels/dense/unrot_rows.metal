    constexpr uint DIMC = uint(DIM);
    threadgroup float t[128u * 33u];
    const uint cs = threadgroup_position_in_grid.x;
    const uint rb = threadgroup_position_in_grid.y;
    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    for (uint e = tid; e < 4096u; e += 256u) t[(e >> 5u) * 33u + (e & 31u)] = r[ulong(rb * 128u + (e >> 5u)) * DIMC + cs * 32u + (e & 31u)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = 0; q < 4u; ++q) {
        const uint c = sg + 8u * q;
        float v[4];
        for (uint j = 0; j < 4u; ++j) v[j] = t[(L + 32u * j) * 33u + c];
        for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
        { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
        for (uint j = 0; j < 4u; ++j) t[(L + 32u * j) * 33u + c] = v[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 4096u; e += 256u) {
        const uint i = rb * 128u + (e >> 5u);
        const uint o = cs * 32u + (e & 31u);
        W[ulong(i) * DIMC + o] = OUT_T(t[(e >> 5u) * 33u + (e & 31u)] * 0.0078125f * float(suh[i]) * float(svh[o]));
    }
