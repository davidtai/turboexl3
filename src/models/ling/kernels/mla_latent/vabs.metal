    // Launch 3, the merge and the value half: threadgroup (h, s) merges row (s, h)'s key-group partials (pm, pl, po f32
    // from launch 2) over the groups that hold a key it sees (j KPT <= its position), in group order: the max over them,
    // then each group's weight exp(pm - max) into the sum and the 512 sums by fma; the latent output is those sums over
    // the sum. Then each of the DV outputs is that latent dotted with kv_b's value-half row h (DN + DV) + DN + e (a lane:
    // the columns lane + 32 i, i ascending, by fma; then simd_sum), cast to OT: out [1, H, S, DV].
    const uint tid = thread_position_in_threadgroup.x;
    const ushort lane = thread_index_in_simdgroup;
    const ushort sgi = simdgroup_index_in_threadgroup;
    const int h = int(threadgroup_position_in_grid.y);
    const int s = int(threadgroup_position_in_grid.z);
    const int T = dims[0];
    const int NJ = dims[1];
    const int lim = T - S + s;
    const int nj = lim / KPT + 1;
    const ulong base = (ulong(s) * ulong(H) + ulong(h)) * ulong(NJ);
    threadgroup float olat[DL];
    {
        float mx = -INFINITY;
        for (int j = 0; j < nj; ++j) mx = metal::max(mx, pm[base + ulong(j)]);
        float l = 0.0f;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        const uint c0 = tid * 4u;
        for (int j = 0; j < nj; ++j) {
            const float e = metal::exp(pm[base + ulong(j)] - mx);
            l = fma(pl[base + ulong(j)], e, l);
            const device float* pr = po + (base + ulong(j)) * ulong(DL) + ulong(c0);
            a0 = fma(pr[0], e, a0);
            a1 = fma(pr[1], e, a1);
            a2 = fma(pr[2], e, a2);
            a3 = fma(pr[3], e, a3);
        }
        olat[c0] = a0 / l;
        olat[c0 + 1u] = a1 / l;
        olat[c0 + 2u] = a2 / l;
        olat[c0 + 3u] = a3 / l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(sgi) * (DV / 4); e < (int(sgi) + 1) * (DV / 4); ++e) {
        const device WT* wr = w + (ulong(h) * ulong(DN + DV) + ulong(DN + e)) * ulong(DL);
        float acc = 0.0f;
        for (int i = 0; i < DL / 32; ++i) acc = fma(olat[int(lane) + 32 * i], float(wr[int(lane) + 32 * i]), acc);
        const float tot = simd_sum(acc);
        if (lane == 0u) out[(ulong(h) * ulong(S) + ulong(s)) * ulong(DV) + ulong(e)] = OT(tot);
    }
