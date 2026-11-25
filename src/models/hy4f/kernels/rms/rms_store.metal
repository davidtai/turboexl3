    // MLX's output loop (T = float): the weight times (x times the inverse mean), one f32 product each, the normed row of
    // axis_size values at y's row
    {
        const device float* wr = w + lid * N_READS;
        device float* out = y + ulong(row) * axis_size + lid * N_READS;
        if (lid * N_READS + N_READS <= axis_size) {
            for (int i = 0; i < N_READS; i++) {
                out[i] = wr[i] * static_cast<float>(thread_x[i] * local_inv_mean[0]);
            }
        } else {
            for (int i = 0; i < N_READS; i++) {
                if ((lid * N_READS + i) < axis_size) {
                    out[i] = wr[i] * static_cast<float>(thread_x[i] * local_inv_mean[0]);
                }
            }
        }
    }
