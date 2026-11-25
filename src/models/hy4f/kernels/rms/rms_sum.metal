    // MLX's rms_single_row (mlx/backend/metal/kernels/rms_norm.metal at the server's MLX c69303b0; T = float, N_READS =
    // RMS_N_READS = 4) over the staged row: its x is `ys`, axis_size the norm's first nrm[0] values, eps eps_in[0]. MLX
    // runs axis_size / 4 threads a row: thread lid takes values 4 lid .. 4 lid + 3, simd_group_id this kernel's simdgroup
    // b, simd_lane_id its lane L; a simdgroup past axis_size / 128 takes MLX's tail branch, zeros, as MLX's zeroed
    // local_sums slots. Its statements in its order.
    constexpr int SIMD_SIZE = 32;
    constexpr int N_READS = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[SIMD_SIZE];
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = L;
    const uint simd_group_id = b;
    const uint axis_size = nrm[0];
    const float eps = eps_in[0];
    float acc = 0;
    float thread_x[N_READS];
    const threadgroup float* x = ys + lid * N_READS;
    if (lid * N_READS + N_READS <= axis_size) {
        for (int i = 0; i < N_READS; i++) {
            thread_x[i] = x[i];
            acc += thread_x[i] * thread_x[i];
        }
    } else {
        for (int i = 0; i < N_READS; i++) {
            thread_x[i] = (lid * N_READS + i < axis_size) ? (float)x[i] : 0;
            acc += thread_x[i] * thread_x[i];
        }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
