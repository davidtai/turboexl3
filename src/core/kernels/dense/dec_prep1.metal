    threadgroup half xt[RBC * SPAN];
    device TURBOEXL3_COH float* xsc = (device TURBOEXL3_COH float*)xsx;
    for (uint pi = sg; pi < SPB * RBC; pi += 4u) {
        const uint rr = pi / SPB;
        const uint b = ksl * SPB + (pi - rr * SPB);
        const uint row = min(r0 + rr, ROWSC - 1u);
        float v[4];
        for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(float(x[row * INC + i]))) * float(suh[i]); }
        for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
        { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
        float sum = 0.0f;
        for (uint j = 0; j < 4u; ++j) {
            const uint i = b * 128u + L + 32u * j;
            const uint r = i & 15u;
            const uint q = (r & 7u) >> 1u;
            const uint m = (r & 1u) | ((r >> 3u) << 1u);
            const half hv = half(v[j] * HAD_SCALE);
            xt[rr * SPAN + ((i & ~15u) + 4u * q + m) - ksl * SPAN] = hv;
            sum += float(hv);
        }
        sum = simd_sum(sum);
        // the deferred map's input sums: the block's first column group publishes them (it arrives like every other)
        if (DEFER != 0 && (blk % GC) == 0u && L == 0u && r0 + rr < ROWSC) xsc[((ob * NCHC + chunk) * RBC + rr) * NB + b] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
