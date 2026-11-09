    // Launch 2 for up to RPT query rows a threadgroup (Config.mla_latent_rows; mla_latent.Attn.rows2, rows4): threadgroup
    // (j, z) takes key group j for heads hb .. hb + 31 of rows s0 .. s0 + RPT - 1 below S (z = (s0 / RPT) (H / 32) +
    // hb / 32), so the group's latents and rope keys are read once for those rows, not once a row. Each row keeps launch
    // 2's arithmetic (attn_1 .. attn_4): its own scores, max, sum, weights and 512 accumulators, each 8 x 8 tile at the
    // same place, every product and sum of it in the same order. Row s0 + i sits at key position T - S + s0 + i and sees
    // the keys below k_hi[i]: a key it does not see reads as 0 for it and scores -inf, and a sub-chunk at or past
    // k_hi[i] runs nothing for it, as launch 2 runs none. So each row's pm, pl and po are launch 2's words.
    const ushort lane = thread_index_in_simdgroup;
    const ushort sgi = simdgroup_index_in_threadgroup;
    const ushort mt = sgi / 4u;
    const ushort cs = sgi % 4u;
    const int T = dims[0];
    const int NJ = dims[1];
    const int j = int(threadgroup_position_in_grid.y);
    const int zg = int(threadgroup_position_in_grid.z);
    const int s0 = (zg / (H / 32)) * RPT;
    const int nr = min(RPT, S - s0);
    const int hb = (zg % (H / 32)) * 32;
    const int k_lo = j * KPT;
    // row s0 + i's key bound (launch 2's k_hi at its position), and the last row's: the keys the threadgroup reads
    int k_hi[RPT];
    for (int i = 0; i < RPT; ++i) k_hi[i] = min(k_lo + KPT, T - S + s0 + i + 1);
    const int k_top = min(k_lo + KPT, T - S + s0 + nr);
    const ulong lt = ulong(lat_strides[2]);
    const ulong ld = ulong(lat_strides[3]);
    const ulong rt = ulong(rot_strides[2]);
    const ulong rd = ulong(rot_strides[3]);
    const float sc = scale[0];
    // the lane's two elements of an 8 x 8 tile: row fm, columns fn and fn + 1 (MLX steel's fragment coordinates)
    const ushort qid = lane / 4u;
    const ushort fm = (qid & 4u) + ((lane / 2u) % 4u);
    const ushort fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    // each row's sub-chunk scores (then weights), max, sum and rescale, as launch 2 keeps them for its one row
    threadgroup float ssh_r[RPT * 32 * CK];
    threadgroup float msh_r[RPT * 32];
    threadgroup float lsh_r[RPT * 32];
    threadgroup float ash_r[RPT * 32];
    {
        const uint tid = uint(sgi) * 32u + uint(lane);
        if (tid < uint(RPT) * 32u) {
            msh_r[tid] = -INFINITY;
            lsh_r[tid] = 0.0f;
            ash_r[tid] = 1.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // this lane's row of each query row (M tile mt, row fm): head hb + 8 mt + fm
    const int head = hb + int(mt) * 8 + int(fm);
    simdgroup_float8x8 o[RPT][16];
    for (int i = 0; i < RPT; ++i) {
        for (ushort c = 0; c < 16u; ++c) o[i][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (int kb = k_lo; kb < k_top; kb += CK) {
        // the rows this sub-chunk runs for: launch 2 runs it for row s0 + i while kb < k_hi[i]
        bool live[RPT];
        for (int i = 0; i < RPT; ++i) live[i] = i < nr && kb < k_hi[i];
        // A. each row's scores of its M tile's rows at keys kb + 8 cs .. + 7: the lane's two keys' words read once (as
        // far as the last row sees), then each row's tile on its own query, a key the row does not see read as 0
        {
            const int ta = kb + int(cs) * 8 + int(fn);
            const int tb = ta + 1;
            const bool ua = ta < k_top;
            const bool ub = tb < k_top;
            const device CT* la = lat + ulong(ua ? ta : k_lo) * lt;
            const device CT* lb = lat + ulong(ub ? tb : k_lo) * lt;
            const device CT* ra = rot + ulong(ua ? ta : k_lo) * rt;
            const device CT* rb = rot + ulong(ub ? tb : k_lo) * rt;
            simdgroup_float8x8 acc[RPT];
            for (int i = 0; i < RPT; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int d0 = 0; d0 < DL; d0 += 8) {
                const float xa = ua ? float(la[ulong(d0 + int(fm)) * ld]) : 0.0f;
                const float xb = ub ? float(lb[ulong(d0 + int(fm)) * ld]) : 0.0f;
                for (int i = 0; i < RPT; ++i) {
                    if (live[i]) {
                        const device float* qlr = ql + (ulong(s0 + i) * ulong(H) + ulong(head)) * ulong(DL);
                        simdgroup_float8x8 qa;
                        simdgroup_float8x8 kt;
                        qa.thread_elements()[0] = qlr[d0 + int(fn)];
                        qa.thread_elements()[1] = qlr[d0 + int(fn) + 1];
                        kt.thread_elements()[0] = ta < k_hi[i] ? xa : 0.0f;
                        kt.thread_elements()[1] = tb < k_hi[i] ? xb : 0.0f;
                        simdgroup_multiply_accumulate(acc[i], qa, kt, acc[i]);
                    }
                }
            }
            for (int d0 = 0; d0 < DR; d0 += 8) {
                const float xa = ua ? float(ra[ulong(d0 + int(fm)) * rd]) : 0.0f;
                const float xb = ub ? float(rb[ulong(d0 + int(fm)) * rd]) : 0.0f;
                for (int i = 0; i < RPT; ++i) {
                    if (live[i]) {
                        const device QT* qrr = qr + (ulong(head) * ulong(S) + ulong(s0 + i)) * ulong(DR);
                        simdgroup_float8x8 qa;
                        simdgroup_float8x8 kt;
                        qa.thread_elements()[0] = float(qrr[d0 + int(fn)]);
                        qa.thread_elements()[1] = float(qrr[d0 + int(fn) + 1]);
                        kt.thread_elements()[0] = ta < k_hi[i] ? xa : 0.0f;
                        kt.thread_elements()[1] = tb < k_hi[i] ? xb : 0.0f;
                        simdgroup_multiply_accumulate(acc[i], qa, kt, acc[i]);
                    }
                }
            }
            for (int i = 0; i < RPT; ++i) {
                if (live[i]) {
                    const float e0 = ta < k_hi[i] ? acc[i].thread_elements()[0] * sc : -INFINITY;
                    const float e1 = tb < k_hi[i] ? acc[i].thread_elements()[1] * sc : -INFINITY;
                    threadgroup float* sr = ssh_r + i * (32 * CK);
                    sr[(int(mt) * 8 + int(fm)) * CK + int(cs) * 8 + int(fn)] = e0;
                    sr[(int(mt) * 8 + int(fm)) * CK + int(cs) * 8 + int(fn) + 1] = e1;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // each row's online softmax update: launch 2's statements (attn_2) on the row's own scores, max, sum and rescale
        for (int i = 0; i < RPT; ++i) {
            if (live[i]) {
                threadgroup float* ssh = ssh_r + i * (32 * CK);
                threadgroup float* msh = msh_r + i * 32;
                threadgroup float* lsh = lsh_r + i * 32;
                threadgroup float* ash = ash_r + i * 32;
