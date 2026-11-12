    // wave3: the split-K planes through coherent(device) pointers (a finish unit's last threadgroup reads the stores of
    // every threadgroup over the unit), and each member's arrival verdict
    device TURBOEXL3_COH float* zg = (device TURBOEXL3_COH float*)zgp;
    device TURBOEXL3_COH float* zu = (device TURBOEXL3_COH float*)zup;
    threadgroup uint arrived_last[4];
