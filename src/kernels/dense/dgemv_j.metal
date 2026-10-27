            // by input element: element j feeds codeword j (accumulator 0) and codeword j + 4 (accumulator 1)
            for (uint j = 0; j < 4u; ++j) {
                // DEFER: MUL1's byte sum t; the finish applies A and C (C times the row's input sum)
                const float w0 = (DEFER != 0) ? exl3_mul1_t(cw[j]) : float(exl3_decode<CB>(cw[j]));
                const float w1 = (DEFER != 0) ? exl3_mul1_t(cw[j + 4u]) : float(exl3_decode<CB>(cw[j + 4u]));
                for (uint r = 0; r < RBC; ++r) {
                    // a chunk's spare rows (ROWS not a multiple of RB) reread the last row; their sums are never written
                    const float xv = float(xbase[ulong(min(r0 + r, ROWSC - 1u)) * ulong(INC) + tk * 16u + j]);
                    a[r][t][0] = fma(xv, w0, a[r][t][0]);
                    a[r][t][1] = fma(xv, w1, a[r][t][1]);
                }
            }
