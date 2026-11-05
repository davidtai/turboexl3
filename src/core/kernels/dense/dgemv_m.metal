            for (uint m = 0; m < 8u; ++m) {
                // DEFER: MUL1's byte sum t; the finish applies A and C (C times the row's input sum)
                const float wv = (DEFER != 0) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
                for (uint r = 0; r < RBC; ++r) a[r][t][m >> 2] = fma(xr[r][m & 3u], wv, a[r][t][m >> 2]);
            }
