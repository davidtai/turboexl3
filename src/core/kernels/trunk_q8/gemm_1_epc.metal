        // the weights' center QC2 / 2 (Config.pf_int8_c2): QC2 = 1019 rebuilds at 509.5, exl3pf_q8's unbiased center
        // (prefill/gemm_i8_epc.metal); 1020 is gemm_1_2.metal's 510
        zt[int(ids[1]) * BN + int(ids[0])] = half(fma(float(ct[i]) * sx[m], 4.0f * MA, (0.5f * float(QC2) * MA + MC) * xsum[m]));
