    // pg2's launch (gate.zig `linearGateMulAt` at a pg2 tile): the prompt's rows are the GEMM's M, not the grid's y
    const uint S = uint(M);
