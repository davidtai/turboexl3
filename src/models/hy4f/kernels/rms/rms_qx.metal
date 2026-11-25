    // MLX's output values (rms_store's statements) in f32 where QF asks for them (q: the normed row), and each taken to
    // Q_T (MLX's astype to the next linear's input dtype) staged by element for that linear's prepare
    for (int i = 0; i < N_READS; i++) {
        const uint e = lid * N_READS + i;
        if (e < axis_size) {
            const float qn = w[e] * static_cast<float>(thread_x[i] * local_inv_mean[0]);
            if (QF != 0) {
                q[ulong(row) * axis_size + e] = qn;
            }
            ys[e] = static_cast<float>(Q_T(qn));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
