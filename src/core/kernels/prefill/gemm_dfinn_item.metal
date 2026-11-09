    const uint b = uint(fi) % (HC / 128u);
    const uint r = uint(fi) / (HC / 128u);
    // a row past the call, or a row whose expert has DFMIN rows or more (the GEMM half wrote it): no item
    if (r >= uint(inds_shape[0]) || ecnt[uint(inds[r])] >= DFMIN) return;
