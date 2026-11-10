    // trunk int8 (trunk_q8.zig, Config.pf_trunk_int8; lossy): the prepare at one scale a row. One threadgroup a row, eight
    // simdgroups, simdgroup sg taking the row's 128-blocks sg, sg + 8, ...; each block's f16 values are text_prep's
    // (dense/prep_2.metal: x's f16 rounding, suh, the H128, HAD_SCALE and the half rounding below), held in registers until
    // the row's amax is known
    constexpr uint DIMC = uint(DIM);
    constexpr uint NB = DIMC / 128u;
    constexpr uint BPS = (NB + 7u) / 8u;
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint row = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float amx[8];
    threadgroup float bsum[NB];
    half h[BPS][4];
    float am = 0.0f;
    for (uint k = 0; k < BPS; ++k) {
    const uint b = sg + 8u * k;
    if (b >= NB) break;
