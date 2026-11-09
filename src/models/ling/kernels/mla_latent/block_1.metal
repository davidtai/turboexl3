    // Launch 2 with more 8 x 8 tiles a simdgroup (Config.mla_latent_block; mla_latent.Attn.block): the scores of 4
    // sub-chunks (4 CK keys) at a time, each simdgroup on 2 x 2 tiles (M tiles ma, ma + 1 of 8 rows; N tiles na, na + 1
    // of 8 keys), so a query word serves 2 tiles and a key word 2; then each sub-chunk's online softmax (launch 2's
    // statements, attn_2) and P times the latents with each simdgroup on 4 x 4 output tiles (the 4 M tiles at column
    // tiles 4 sgi .. 4 sgi + 3), so a weight word serves 4 tiles and a latent word 4. Every tile is launch 2's (attn_1 ..
    // attn_4) at the same place, every product and sum of it in the same order: a score tile's DL / 8 steps over the
    // latent's dims, then DR / 8 over the rope's; an output tile's rescale, then its CK / 8 steps a sub-chunk. A key past
    // lim (or the group) reads as 0 and scores -inf, and a sub-chunk at or past k_hi runs no softmax update and no
    // product, as launch 2 runs none. So pm, pl and po are launch 2's words.
    const ushort lane = thread_index_in_simdgroup;
    const ushort sgi = simdgroup_index_in_threadgroup;
    const int T = dims[0];
    const int NJ = dims[1];
    const int j = int(threadgroup_position_in_grid.y);
    const int zg = int(threadgroup_position_in_grid.z);
    const int s = zg / (H / 32);
    const int hb = (zg % (H / 32)) * 32;
    const int lim = T - S + s;
    const int k_lo = j * KPT;
    const int k_hi = min(k_lo + KPT, lim + 1);
    const ulong lt = ulong(lat_strides[2]);
    const ulong ld = ulong(lat_strides[3]);
    const ulong rt = ulong(rot_strides[2]);
    const ulong rd = ulong(rot_strides[3]);
    const float sc = scale[0];
    // the lane's two elements of an 8 x 8 tile: row fm, columns fn and fn + 1 (MLX steel's fragment coordinates)
    const ushort qid = lane / 4u;
    const ushort fm = (qid & 4u) + ((lane / 2u) % 4u);
    const ushort fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    // the scores (then weights) of a block's 4 sub-chunks, each [32 rows][CK keys] as launch 2 keeps its one
    threadgroup float ssb[4 * 32 * CK];
    threadgroup float msh[32];
    threadgroup float lsh[32];
    threadgroup float ash[32];
    {
        const uint tid = uint(sgi) * 32u + uint(lane);
        if (tid < 32u) {
            msh[tid] = -INFINITY;
            lsh[tid] = 0.0f;
            ash[tid] = 1.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the score tiles: M tiles ma, ma + 1 (this lane's rows: heads h0 and h0 + 8), N tiles na, na + 1 of a block
    const int ma = int(sgi / 8u) * 2;
    const int na = int(sgi % 8u) * 2;
    const int h0 = hb + ma * 8 + int(fm);
    const device float* ql0 = ql + (ulong(s) * ulong(H) + ulong(h0)) * ulong(DL);
    const device float* ql1 = ql + (ulong(s) * ulong(H) + ulong(h0 + 8)) * ulong(DL);
    const device QT* qr0 = qr + (ulong(h0) * ulong(S) + ulong(s)) * ulong(DR);
    const device QT* qr1 = qr + (ulong(h0 + 8) * ulong(S) + ulong(s)) * ulong(DR);
    // the output tiles: o[4 m + c] is M tile m at column tile 4 sgi + c (columns 32 sgi + 8 c .. + 7)
    simdgroup_float8x8 o[16];
    for (ushort i = 0; i < 16u; ++i) o[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int kb0 = k_lo; kb0 < k_hi; kb0 += 4 * CK) {
        // A. the block's scores: this lane's keys t0, t0 + 1 (N tile na) and t1, t1 + 1 (N tile na + 1)
        {
            const int t0 = kb0 + na * 8 + int(fn);
            const int t1 = t0 + 8;
            const bool v0a = t0 < k_hi;
            const bool v0b = t0 + 1 < k_hi;
            const bool v1a = t1 < k_hi;
            const bool v1b = t1 + 1 < k_hi;
            const device CT* l0a = lat + ulong(v0a ? t0 : k_lo) * lt;
            const device CT* l0b = lat + ulong(v0b ? t0 + 1 : k_lo) * lt;
            const device CT* l1a = lat + ulong(v1a ? t1 : k_lo) * lt;
            const device CT* l1b = lat + ulong(v1b ? t1 + 1 : k_lo) * lt;
            const device CT* r0a = rot + ulong(v0a ? t0 : k_lo) * rt;
            const device CT* r0b = rot + ulong(v0b ? t0 + 1 : k_lo) * rt;
            const device CT* r1a = rot + ulong(v1a ? t1 : k_lo) * rt;
            const device CT* r1b = rot + ulong(v1b ? t1 + 1 : k_lo) * rt;
            simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int d0 = 0; d0 < DL; d0 += 8) {
                simdgroup_float8x8 qa0;
                simdgroup_float8x8 qa1;
                simdgroup_float8x8 kt0;
                simdgroup_float8x8 kt1;
                qa0.thread_elements()[0] = ql0[d0 + int(fn)];
                qa0.thread_elements()[1] = ql0[d0 + int(fn) + 1];
                qa1.thread_elements()[0] = ql1[d0 + int(fn)];
                qa1.thread_elements()[1] = ql1[d0 + int(fn) + 1];
                kt0.thread_elements()[0] = v0a ? float(l0a[ulong(d0 + int(fm)) * ld]) : 0.0f;
                kt0.thread_elements()[1] = v0b ? float(l0b[ulong(d0 + int(fm)) * ld]) : 0.0f;
                kt1.thread_elements()[0] = v1a ? float(l1a[ulong(d0 + int(fm)) * ld]) : 0.0f;
                kt1.thread_elements()[1] = v1b ? float(l1b[ulong(d0 + int(fm)) * ld]) : 0.0f;
                simdgroup_multiply_accumulate(acc00, qa0, kt0, acc00);
                simdgroup_multiply_accumulate(acc01, qa0, kt1, acc01);
                simdgroup_multiply_accumulate(acc10, qa1, kt0, acc10);
                simdgroup_multiply_accumulate(acc11, qa1, kt1, acc11);
            }
            for (int d0 = 0; d0 < DR; d0 += 8) {
                simdgroup_float8x8 qa0;
                simdgroup_float8x8 qa1;
                simdgroup_float8x8 kt0;
                simdgroup_float8x8 kt1;
                qa0.thread_elements()[0] = float(qr0[d0 + int(fn)]);
                qa0.thread_elements()[1] = float(qr0[d0 + int(fn) + 1]);
                qa1.thread_elements()[0] = float(qr1[d0 + int(fn)]);
                qa1.thread_elements()[1] = float(qr1[d0 + int(fn) + 1]);
                kt0.thread_elements()[0] = v0a ? float(r0a[ulong(d0 + int(fm)) * rd]) : 0.0f;
                kt0.thread_elements()[1] = v0b ? float(r0b[ulong(d0 + int(fm)) * rd]) : 0.0f;
                kt1.thread_elements()[0] = v1a ? float(r1a[ulong(d0 + int(fm)) * rd]) : 0.0f;
                kt1.thread_elements()[1] = v1b ? float(r1b[ulong(d0 + int(fm)) * rd]) : 0.0f;
                simdgroup_multiply_accumulate(acc00, qa0, kt0, acc00);
                simdgroup_multiply_accumulate(acc01, qa0, kt1, acc01);
                simdgroup_multiply_accumulate(acc10, qa1, kt0, acc10);
                simdgroup_multiply_accumulate(acc11, qa1, kt1, acc11);
            }
            // each tile's scores where launch 2 stores them: N tiles na and na + 1 lie in sub-chunk na / 4 of the block,
            // at its columns 8 (na % 4) + fn and 8 further
            threadgroup float* sq = ssb + (na / 4) * (32 * CK);
            const int c0 = (na % 4) * 8 + int(fn);
            const int r0 = ma * 8 + int(fm);
            const int r1 = r0 + 8;
            sq[r0 * CK + c0] = v0a ? acc00.thread_elements()[0] * sc : -INFINITY;
            sq[r0 * CK + c0 + 1] = v0b ? acc00.thread_elements()[1] * sc : -INFINITY;
            sq[r0 * CK + c0 + 8] = v1a ? acc01.thread_elements()[0] * sc : -INFINITY;
            sq[r0 * CK + c0 + 9] = v1b ? acc01.thread_elements()[1] * sc : -INFINITY;
            sq[r1 * CK + c0] = v0a ? acc10.thread_elements()[0] * sc : -INFINITY;
            sq[r1 * CK + c0 + 1] = v0b ? acc10.thread_elements()[1] * sc : -INFINITY;
            sq[r1 * CK + c0 + 8] = v1a ? acc11.thread_elements()[0] * sc : -INFINITY;
            sq[r1 * CK + c0 + 9] = v1b ? acc11.thread_elements()[1] * sc : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int q = 0; q < 4; ++q) {
            const int kb = kb0 + q * CK;
            if (kb >= k_hi) {
                break;
            }
            // sub-chunk q's scores, where launch 2's online softmax update (attn_2) reads them and writes its weights
            threadgroup float* ssh = ssb + q * (32 * CK);
