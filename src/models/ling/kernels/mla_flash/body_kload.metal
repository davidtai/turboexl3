            if (!align_K && is_last_k) {
              Ktile.load_rows(
                  K + K_load_off,
                  int(k_strides[2]),
                  lim_rows_k - ik * kU);
            } else {
              Ktile.load(K + K_load_off, int(k_strides[2]));
            }
