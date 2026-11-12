    // Launch 2 with each sub-chunk's latents and rope keys converted to f32 once a threadgroup (Config.mla_latent_stage;
    // mla_latent.Attn.stage): the threadgroup stages them in threadgroup memory, the scores' keys a slab of 64 dims at a
    // time (two slabs: one staged while the other is read), the values 8 keys of DL columns at a time, and each simdgroup
    // reads its tiles there instead of from the cache. The tiles and every product and sum of them are launch 2's (attn_1
    // .. attn_4) in the same order, and a staged word is the word launch 2 reads: the cache word converted to f32, or 0
    // for a key past lim (or the group). So pm, pl and po are launch 2's words.
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
    const uint tid = uint(sgi) * 32u + uint(lane);
    threadgroup float ssh[32 * CK];
    threadgroup float msh[32];
    threadgroup float lsh[32];
    threadgroup float ash[32];
    // the staged words, one buffer for both phases: the scores' two slabs [2][CK keys][68] (a key's 64 dims, then 4
    // words that put a tile's reads on distinct banks), or the values' [8 keys][DL + 4]
    threadgroup float stg[2 * CK * 68];
    if (tid < 32u) {
        msh[tid] = -INFINITY;
        lsh[tid] = 0.0f;
        ash[tid] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // this lane's row (M tile mt, row fm): head hb + 8 mt + fm
    const int head = hb + int(mt) * 8 + int(fm);
    const device float* qlr = ql + (ulong(s) * ulong(H) + ulong(head)) * ulong(DL);
    const device QT* qrr = qr + (ulong(head) * ulong(S) + ulong(s)) * ulong(DR);
    simdgroup_float8x8 o[16];
    for (ushort i = 0; i < 16u; ++i) o[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int kb = k_lo; kb < k_hi; kb += CK) {
        // A. the scores of the M tile's rows at keys kb + 8 cs .. + 7 (the lane's: keys ta, tb, both of its row), slab
        // by slab: slab p < DL / 64 holds the latent's dims 64 p .. 64 p + 63, slab DL / 64 the rope key's DR. Step q
        // stages slab q (while q <= DL / 64) into buffer q % 2 and runs slab q - 1's 8 steps from the other buffer.
        {
            const int ta = kb + int(cs) * 8 + int(fn);
            const int tb = ta + 1;
            const bool va = ta < k_hi;
            const bool vb = tb < k_hi;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int q = 0; q <= DL / 64 + 1; ++q) {
                if (q <= DL / 64) {
                    threadgroup float* nx = stg + (q % 2) * (CK * 68);
                    for (uint w = tid; w < uint(CK) * 64u; w += 512u) {
                        const int t = kb + int(w / 64u);
                        const bool v = t < k_hi;
                        const ulong u = ulong(v ? t : k_lo);
                        float x = 0.0f;
                        if (q < DL / 64) {
                            x = v ? float(lat[u * lt + ulong(q * 64 + int(w % 64u)) * ld]) : 0.0f;
                        } else {
                            x = v ? float(rot[u * rt + ulong(w % 64u) * rd]) : 0.0f;
                        }
                        nx[(w / 64u) * 68u + (w % 64u)] = x;
                    }
                }
                if (q > 0) {
                    const int p = q - 1;
                    threadgroup float* kq = stg + (p % 2) * (CK * 68);
                    for (int dd = 0; dd < 64; dd += 8) {
                        simdgroup_float8x8 qa;
                        simdgroup_float8x8 kt;
                        if (p < DL / 64) {
                            qa.thread_elements()[0] = qlr[p * 64 + dd + int(fn)];
                            qa.thread_elements()[1] = qlr[p * 64 + dd + int(fn) + 1];
                        } else {
                            qa.thread_elements()[0] = float(qrr[dd + int(fn)]);
                            qa.thread_elements()[1] = float(qrr[dd + int(fn) + 1]);
                        }
                        kt.thread_elements()[0] = kq[(int(cs) * 8 + int(fn)) * 68 + dd + int(fm)];
                        kt.thread_elements()[1] = kq[(int(cs) * 8 + int(fn) + 1) * 68 + dd + int(fm)];
                        simdgroup_multiply_accumulate(acc, qa, kt, acc);
                    }
                    if (p == DL / 64) {
                        const float s0 = va ? acc.thread_elements()[0] * sc : -INFINITY;
                        const float s1 = vb ? acc.thread_elements()[1] * sc : -INFINITY;
                        ssh[(int(mt) * 8 + int(fm)) * CK + int(cs) * 8 + int(fn)] = s0;
                        ssh[(int(mt) * 8 + int(fm)) * CK + int(cs) * 8 + int(fn) + 1] = s1;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
