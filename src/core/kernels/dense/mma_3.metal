    vec<float, 8> acc[NP][2];
    for (uint p = 0; p < NP; ++p) for (short i = 0; i < 8; ++i) { acc[p][0][i] = 0.0f; acc[p][1][i] = 0.0f; }
    for (uint tk = tk0; tk < tk0 + PER; ++tk) {
        vec<half, 8> af;
        for (short i = 0; i < 8; ++i) af[i] = 0.0h;
        if (live) {
            const half4 x4 = *reinterpret_cast<const device half4*>(xrow + tk * 16u);
            af[0] = x4.x; af[1] = x4.y; af[2] = x4.z; af[3] = x4.w;
        }
        const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * BTC) * ulong(NW);
        for (uint p = 0; p < NP; ++p) {
            // each weight decoded once, as a half (DEFER: MUL1's byte sum t, exact in half), for every row of the chunk
            vec<half, 8> b0;
            vec<half, 8> b1;
            exl3pf_bfrag<N, WMASK, CB, int(DEFER), 0>(b0, tile + ulong(2u * p) * ulong(NW), qb, ga, lp);
            exl3pf_bfrag<N, WMASK, CB, int(DEFER), 0>(b1, tile + ulong(2u * p + 1u) * ulong(NW), qb, ga, lp);
            // the pair's 16-deep sum on the matrix units from zero, then added to the f32 accumulators on the ALUs (round to
            // nearest), one add per output and input tile
            vec<float, 8> t0;
            vec<float, 8> t1;
            for (short i = 0; i < 8; ++i) { t0[i] = 0.0f; t1[i] = 0.0f; }
            turboexl3_mma16(t0, t1, af, b0, b1);
            for (short i = 0; i < 8; ++i) { acc[p][0][i] += t0[i]; acc[p][1][i] += t1[i]; }
        }
    }
    // lane L holds row fm, columns 4q .. 4q + 3 of both tiles of each pair (elements 4..7: row fm + 8, zero)
    if (fm < RBC) for (uint p = 0; p < NP; ++p) for (uint h = 0; h < 2u; ++h) {
        threadgroup float* o = &red[sg][fm * 16u * BTC + (2u * p + h) * 16u + 4u * q];
        o[0] = acc[p][h][0]; o[1] = acc[p][h][1]; o[2] = acc[p][h][2]; o[3] = acc[p][h][3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
