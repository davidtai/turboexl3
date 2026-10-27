    const uint i = thread_position_in_grid.x;
    const float gv = g[i];
    h[i] = (gv / (1.0f + metal::exp(-gv))) * u[i];
