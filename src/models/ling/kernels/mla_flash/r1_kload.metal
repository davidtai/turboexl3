            if (!align_K && is_last_k) {
              Ktile.load_rows(
                  K_src + K_load_off,
                  K_ld,
                  lim_rows_k - ik * kU);
            } else {
              Ktile.load(K_src + K_load_off, K_ld);
            }
