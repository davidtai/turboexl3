    if (rows == 0) return;
    const bool consumer = int(sg) < NSGC;
    const bool producer = NPR == 0 || !consumer;
    const int dsg = (NPR > 0) ? int(sg) - NSGC : int(sg);
    const int tm = SM * int(sg / 2u);
    const int tn = SN * int(sg % 2u);
    const int sgp = consumer ? min(SM, max(0, rows - tm)) : 0;
    // the expert's trellis from the threadgroup's first output column; column block cb starts 4 cb tiles further
