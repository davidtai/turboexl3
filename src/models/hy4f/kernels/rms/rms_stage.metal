    // the linear's y at the row's elements as text_finish's store gives it (rounded to OUT_T), taken to f32 (MLX's
    // astype), staged by element in threadgroup memory: the row the norm reads
    threadgroup float ys[DIM];
    for (uint j = 0; j < 4u; ++j) {
        const uint o = b * 128u + L + 32u * j;
        ys[o] = static_cast<float>(OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
