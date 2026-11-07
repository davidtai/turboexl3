    // Launch 2, the attention's partials: threadgroup (j, z) takes key group j (keys [j KPT, j KPT + KPT)) for 32 rows,
    // heads hb .. hb + 31 of query row s (z = s (H / 32) + hb / 32), over the latent cache: lat [1, 1, T, DL] (the normed
    // latents) and rot [1, 1, T, DR] (the roped shared keys), read through their strides; query row s sits at key
    // position T - S + s and sees the keys up to it (lim). Each row's score of key t is
    // (ql[s, h] . lat[t] + qr[h, s] . rot[t]) * scale, in f32 from the exact operands (ql f32 [S, H, DL], qr [1, H, S, DR]
    // row-contiguous). The group runs in sub-chunks of CK keys, counted from position 0: the scores of a sub-chunk (one
    // 8 x 8 f32 simdgroup-matrix tile a simdgroup: its M tile's 8 rows at 8 keys, the latent's dims then the rope's, in
    // that order), an online softmax update per row (the sub-chunk's max, then exp(score - max), the sum by a 16-lane
    // butterfly), then P (the sub-chunk's weights) times the latents into each row's 512 accumulators (a simdgroup: its M
    // tile's 8 rows at 128 of them). A key past lim (or past the group) reads as 0 and scores -inf: its weight is 0 and
    // its latent 0. So a row's every value depends on its own operands and its position only, never on the rows beside it
    // (a verify row has the bits of a one-row call at its position). Outputs per (s, h, j): pm the max, pl the sum, po
    // the 512 weighted sums (f32).
    const ushort lane = thread_index_in_simdgroup;
    const ushort sgi = simdgroup_index_in_threadgroup;
    const ushort mt = sgi / 4u;
    const ushort cs = sgi % 4u;
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
    threadgroup float ssh[32 * CK];
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
    // this lane's row (M tile mt, row fm): head hb + 8 mt + fm
    const int head = hb + int(mt) * 8 + int(fm);
    const device float* qlr = ql + (ulong(s) * ulong(H) + ulong(head)) * ulong(DL);
    const device QT* qrr = qr + (ulong(head) * ulong(S) + ulong(s)) * ulong(DR);
    simdgroup_float8x8 o[16];
    for (ushort i = 0; i < 16u; ++i) o[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int kb = k_lo; kb < k_hi; kb += CK) {
        // A. the scores of the M tile's rows at keys kb + 8 cs .. + 7 (the lane's: keys ta, tb, both of its row)
        {
            const int ta = kb + int(cs) * 8 + int(fn);
            const int tb = ta + 1;
            const bool va = ta < k_hi;
            const bool vb = tb < k_hi;
            const device CT* la = lat + ulong(va ? ta : k_lo) * lt;
            const device CT* lb = lat + ulong(vb ? tb : k_lo) * lt;
            const device CT* ra = rot + ulong(va ? ta : k_lo) * rt;
            const device CT* rb = rot + ulong(vb ? tb : k_lo) * rt;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int d0 = 0; d0 < DL; d0 += 8) {
                simdgroup_float8x8 qa;
                simdgroup_float8x8 kt;
                qa.thread_elements()[0] = qlr[d0 + int(fn)];
                qa.thread_elements()[1] = qlr[d0 + int(fn) + 1];
                kt.thread_elements()[0] = va ? float(la[ulong(d0 + int(fm)) * ld]) : 0.0f;
                kt.thread_elements()[1] = vb ? float(lb[ulong(d0 + int(fm)) * ld]) : 0.0f;
                simdgroup_multiply_accumulate(acc, qa, kt, acc);
            }
            for (int d0 = 0; d0 < DR; d0 += 8) {
                simdgroup_float8x8 qa;
                simdgroup_float8x8 kt;
                qa.thread_elements()[0] = float(qrr[d0 + int(fn)]);
                qa.thread_elements()[1] = float(qrr[d0 + int(fn) + 1]);
                kt.thread_elements()[0] = va ? float(ra[ulong(d0 + int(fm)) * rd]) : 0.0f;
                kt.thread_elements()[1] = vb ? float(rb[ulong(d0 + int(fm)) * rd]) : 0.0f;
                simdgroup_multiply_accumulate(acc, qa, kt, acc);
            }
            const float s0 = va ? acc.thread_elements()[0] * sc : -INFINITY;
            const float s1 = vb ? acc.thread_elements()[1] * sc : -INFINITY;
            ssh[(int(mt) * 8 + int(fm)) * CK + int(cs) * 8 + int(fn)] = s0;
            ssh[(int(mt) * 8 + int(fm)) * CK + int(cs) * 8 + int(fn) + 1] = s1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // B. the online softmax update of row r (16 lanes a row, 2 keys a lane): the new max, each key's weight, the
        // row's rescale of what it summed so far, and its new sum
        {
            const int r = int(sgi) * 2 + int(lane / 16u);
            const int kk = int(lane % 16u) * 2;
            const float v0 = ssh[r * CK + kk];
            const float v1 = ssh[r * CK + kk + 1];
            float cm = metal::max(v0, v1);
            for (ushort off = 8; off > 0; off >>= 1) cm = metal::max(cm, simd_shuffle_xor(cm, off));
            const float mo = msh[r];
            const float mn = metal::max(mo, cm);
            float p0 = 0.0f;
            float p1 = 0.0f;
            float al = 1.0f;
            if (mn != -INFINITY) {
                p0 = (v0 == -INFINITY) ? 0.0f : metal::exp(v0 - mn);
                p1 = (v1 == -INFINITY) ? 0.0f : metal::exp(v1 - mn);
                al = metal::exp(mo - mn);
            }
            float ps = p0 + p1;
            for (ushort off = 8; off > 0; off >>= 1) ps += simd_shuffle_xor(ps, off);
            ssh[r * CK + kk] = p0;
            ssh[r * CK + kk + 1] = p1;
            if ((lane % 16u) == 0u) {
                lsh[r] = fma(lsh[r], al, ps);
                msh[r] = mn;
                ash[r] = al;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // C. the rows' accumulators rescaled, then P [8 rows, CK keys] times the latents [CK keys, 128 columns]
        {
            const float a = ash[int(mt) * 8 + int(fm)];
            for (ushort i = 0; i < 16u; ++i) {
                o[i].thread_elements()[0] *= a;
                o[i].thread_elements()[1] *= a;
            }
            for (int k0 = 0; k0 < CK; k0 += 8) {
                simdgroup_float8x8 pw;
                pw.thread_elements()[0] = ssh[(int(mt) * 8 + int(fm)) * CK + k0 + int(fn)];
                pw.thread_elements()[1] = ssh[(int(mt) * 8 + int(fm)) * CK + k0 + int(fn) + 1];
                const int t = kb + k0 + int(fm);
                const bool vt = t < k_hi;
                const device CT* vr = lat + ulong(vt ? t : k_lo) * lt + ulong(int(cs) * 128 + int(fn)) * ld;
                for (ushort i = 0; i < 16u; ++i) {
                    simdgroup_float8x8 vv;
                    vv.thread_elements()[0] = vt ? float(vr[ulong(i * 8u) * ld]) : 0.0f;
                    vv.thread_elements()[1] = vt ? float(vr[ulong(i * 8u + 1u) * ld]) : 0.0f;
                    simdgroup_multiply_accumulate(o[i], pw, vv, o[i]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // the group's partials of each row: its 512 sums (this lane's row, 128 columns a simdgroup), then its max and sum
    {
        const ulong slot = (ulong(s) * ulong(H) + ulong(head)) * ulong(NJ) + ulong(j);
        device float* por = po + slot * ulong(DL) + ulong(int(cs) * 128 + int(fn));
        for (ushort i = 0; i < 16u; ++i) {
            por[i * 8u] = o[i].thread_elements()[0];
            por[i * 8u + 1u] = o[i].thread_elements()[1];
        }
        const int r = int(sgi) * 2 + int(lane / 16u);
        if ((lane % 16u) == 0u) {
            const ulong sr = (ulong(s) * ulong(H) + ulong(hb + r)) * ulong(NJ) + ulong(j);
            pm[sr] = msh[r];
            pl[sr] = lsh[r];
        }
    }
