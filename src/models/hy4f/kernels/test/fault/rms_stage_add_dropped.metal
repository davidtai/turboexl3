    // the planted fault (a red check, never served): rms_stage_add without the other branch's y (the linear's y alone)
    threadgroup float ys[DIM];
    for (uint j = 0; j < 4u; ++j) {
        const uint o = b * 128u + L + 32u * j;
        ys[o] = static_cast<float>(OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
