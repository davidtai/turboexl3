    // the planted fault (a red check, never served): rms_store without the weight (x times the inverse mean alone)
    {
        device float* out = y + ulong(row) * axis_size + lid * N_READS;
        for (int i = 0; i < N_READS; i++) {
            if ((lid * N_READS + i) < axis_size) {
                out[i] = static_cast<float>(thread_x[i] * local_inv_mean[0]);
            }
        }
    }
