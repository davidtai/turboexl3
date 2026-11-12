    // wave3: the split-K planes through a coherent(device) pointer (a finish unit's last threadgroup reads the stores of
    // every threadgroup over the unit), each member's arrival verdict, and dfin3's exchange for its H128 (dfin3_1b's,
    // declared here at kernel scope)
    device TURBOEXL3_COH float* zd = (device TURBOEXL3_COH float*)zdp;
    threadgroup uint arrived_last[4];
    threadgroup float xch[128];
