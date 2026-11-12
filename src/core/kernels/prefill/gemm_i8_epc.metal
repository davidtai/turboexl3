            // the weights' center QC2 / 2 (prefill.Options.int8_c2): 4 q + 510 sits half a unit of t above t on average
            // (exl3pf_q8 rounds halves up), so QC2 = 1019 rebuilds at 509.5, unbiased; 1020 is gemm_i8_2_2.metal's 510
            if (m < rows) z[ulong(row0 + m) * ulong(NDC) + ulong(n0 + tn + int(ids[0]))] = half(fma(v, 4.0f * MA, (0.5f * float(QC2) * MA + MC) * xsum[m]));
