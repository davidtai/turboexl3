    // The member whose columns this tile covers: its widths, its first tile, its operands and its output
    const int mem = tid_x < T0 ? 0 : (tid_x < T0 + T1 ? 1 : 2);
    const int N = mem == 0 ? N0 : (mem == 1 ? N1 : N2);
    const int ldb = N;
    const int first_tile = mem == 0 ? 0 : (mem == 1 ? T0 : T0 + T1);
    const device T* A = mem == 0 ? x0 : (mem == 1 ? x1 : x2);
    const device T* B = mem == 0 ? w0 : (mem == 1 ? w1 : w2);
    const device half* svh = mem == 0 ? svh0 : (mem == 1 ? svh1 : svh2);
    device OUT_T* y = mem == 0 ? y0 : (mem == 1 ? y1 : y2);
