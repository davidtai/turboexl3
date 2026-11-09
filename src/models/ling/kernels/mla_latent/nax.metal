    // Launch 2 on the matrix units (Config.mla_latent_nax; mla_latent.Attn.nax_*): launch 2's partials (pm, pl, po) from
    // MLX steel's NAX tiles (16 x 32 x 16 multiply-accumulates into f32) over operands in the cache's dtype (bf16 or
    // f16). Threadgroup (j, z) takes key group j (keys [j KPT, j KPT + KPT), counted from position 0 as launch 2's) for
    // 16 MT heads of query row s (z = s (H / (16 MT)) + the head block); simdgroup g scores the 16 heads of its M tile
    // mt = g / 4 and sums the weights times the latent's columns 128 cs .. 128 cs + 127 (cs = g % 4), so 4 simdgroups
    // share an M tile. The query (qlb: launch 1's absorbed query rounded to the cache's dtype [S, H, DL]; qrb: its rope
    // half in that dtype [1, H, S, DR]; each contiguous) meets the cache (lat [1, 1, T, DL], rot [1, 1, T, DR], read
    // through their strides: a key's dims as one run where both views hold them contiguous, else word by word) one 16-dim
    // slice at a time, 36 slices (32 of the latent, 4 of the rope key). With SD 0 each simdgroup scores every slice; with SD 1 a simdgroup scores 9 of them and the M tile's
    // four partial score tiles meet in threadgroup memory, summed in simdgroup order, so its 4 simdgroups hold the same
    // scores. Each step of BKN keys: the scores times the scale; a key past the row's bound (lim, or the group's end)
    // masked to the finite minimum, its K and V rows read as 0; each row's online softmax in natural units (MLX
    // attention_nax's tile row ops); the accumulators rescaled; then P (f32) times V (the latent rows) into the
    // simdgroup's 128 columns. A threadgroup's rows are one query row's heads, so one bound serves them all. Outputs per
    // (s, h, j), as launch 2's: pm the max, pl the sum, po the 512 weighted sums (f32), which launch 3 merges as it
    // merges launch 2's. Not launch 2's words: the query rounds to the cache's dtype and the products sum in the matrix
    // units' order, held to a tolerance of the f64 reference (mla_latent_test.zig).
    const ushort sgi = simdgroup_index_in_threadgroup;
    const short mt = short(sgi / 4u);
    const short cs = short(sgi % 4u);
    const int T = dims[0];
    const int NJ = dims[1];
    const int j = int(threadgroup_position_in_grid.y);
    const int zg = int(threadgroup_position_in_grid.z);
    const int s = zg / (H / (16 * MT));
    const int hb = (zg % (H / (16 * MT))) * (16 * MT) + int(mt) * 16;
    const int lim = T - S + s;
    const int k_lo = j * KPT;
    const int k_hi = min(k_lo + KPT, lim + 1);
    const ulong lt = ulong(lat_strides[2]);
    const ulong ld = ulong(lat_strides[3]);
    const ulong rt = ulong(rot_strides[2]);
    const ulong rd = ulong(rot_strides[3]);
    // a key's dims contiguous in both views (the served caches): each slice's 16 dims read as one run; else word by word
    const bool unit = ld == 1 && rd == 1;
    const float sc = scale[0];

    constexpr short kU = 16;
    constexpr short TK = BKN / kU;
    constexpr short TDL = DL / kU;
    constexpr short TD = (DL + DR) / kU;
    constexpr short TDS = SD != 0 ? TD / 4 : TD;
    constexpr short TV = 128 / kU;
    constexpr int BKP = BKN + 4;
    static_assert(BKN % 32 == 0 && TD % 4 == 0 && DL == 4 * 128 && DR % kU == 0, "nax: the widths the text assumes");

    using stile_t = NAXTile<float, 1, TK>;
    using otile_t = NAXTile<float, 1, TV>;
    using frag_t = typename otile_t::NAXFrag_t;
    otile_t Otile;
    Otile.clear();
    constexpr short kRowsPT = otile_t::kRowsPerThread;
    metal::vec<float, kRowsPT> max_score;
    metal::vec<float, kRowsPT> sum_score{0};
    for (short i = 0; i < kRowsPT; ++i) max_score[i] = mla_nax_finite_min;
    const short2 coord = frag_t::get_coord();
    const short sn = coord.x;
    const short sm = coord.y;

    // this M tile's 16 heads' queries: the latent half (qlb, row stride DL) and the rope half (qrb, row stride S DR)
    const device CT* qlm = qlb + (ulong(s) * ulong(H) + ulong(hb)) * ulong(DL);
    const device CT* qrm = qrb + (ulong(hb) * ulong(S) + ulong(s)) * ulong(DR);
    const short d_lo = SD != 0 ? short(cs * TDS) : short(0);
    threadgroup float sp[SD != 0 ? MT * 4 * 16 * BKP : 1];

    for (int kb = k_lo; kb < k_hi; kb += BKN) {
        // the step's keys this row sees (a key past them reads as 0 and scores the finite minimum)
        const short kn = short(min(BKN, k_hi - kb));
        // S = Q K^T over this simdgroup's slices: 16 heads x BKN keys, f32
        stile_t Stile;
        Stile.clear();
        for (short id = d_lo; id < d_lo + TDS; ++id) {
            NAXTile<CT, 1, 1> Qtile;
            if (id < TDL) {
                Qtile.load(qlm + id * kU, DL);
            } else {
                Qtile.load(qrm + (id - TDL) * kU, S * DR);
            }
            const bool latent = id < TDL;
            const ulong kst = latent ? lt : rt;
            const ulong kdd = latent ? ld : rd;
            for (short ik = 0; ik < TK; ik += 2) {
                NAXTile<CT, 2, 1> Ktile;
                const device CT* kp = latent ? lat + ulong(kb + ik * kU) * lt + ulong(id * kU) * ld : rot + ulong(kb + ik * kU) * rt + ulong((id - TDL) * kU) * rd;
                if (unit) {
                    frag_t::load_rows(Ktile.frag_at(0, 0), kp, kst, Int<1>{}, short(kn - ik * kU));
                    frag_t::load_rows(Ktile.frag_at(1, 0), kp + ulong(kU) * kst, kst, Int<1>{}, short(kn - (ik + 1) * kU));
                } else {
                    frag_t::load_rows(Ktile.frag_at(0, 0), kp, kst, kdd, short(kn - ik * kU));
                    frag_t::load_rows(Ktile.frag_at(1, 0), kp + ulong(kU) * kst, kst, kdd, short(kn - (ik + 1) * kU));
                }
                frag_t::mma(
                    Stile.frag_at(0, ik),
                    Stile.frag_at(0, ik + 1),
                    Qtile.frag_at(0, 0),
                    metal::false_type{},
                    Ktile.frag_at(0, 0),
                    Ktile.frag_at(1, 0),
                    metal::true_type{});
            }
        }
        // SD: the M tile's four partial score tiles summed in simdgroup order (each simdgroup the same sum)
        if constexpr (SD != 0) {
            threadgroup float* mine = sp + (int(mt) * 4 + int(cs)) * 16 * BKP;
            for (short ik = 0; ik < TK; ++ik) frag_t::store(Stile.frag_at(0, ik), mine, Int<BKP>{}, Int<1>{}, Int<0>{}, short(ik * kU));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short p = 0; p < 4; ++p) {
                const threadgroup float* part = sp + (int(mt) * 4 + int(p)) * 16 * BKP;
                for (short ik = 0; ik < TK; ++ik) {
                    typename stile_t::frag_type v;
                    frag_t::load(v, part, Int<BKP>{}, Int<1>{}, Int<0>{}, short(ik * kU));
                    if (p == 0) {
                        Stile.frag_at(0, ik) = v;
                    } else {
                        Stile.frag_at(0, ik) += v;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // the scale, then a key past the row's bound masked
        for (short ii = 0; ii < stile_t::kElemsPerTile; ++ii) Stile.elems()[ii] *= sc;
        if (kn < BKN) {
            for (short ik = 0; ik < TK; ++ik) {
                thread auto& fg = Stile.frag_at(0, ik);
                for (short ii = 0; ii < stile_t::kFragThrRows; ++ii) {
                    for (short jj = 0; jj < stile_t::kFragThrCols; ++jj) {
                        if (ik * kU + sn + jj >= kn) fg[ii * stile_t::kFragThrCols + jj] = mla_nax_finite_min;
                    }
                }
            }
        }
        // each row's online softmax: the new max, exp(score - max), the rescale of what it summed so far, its new sum
        metal::vec<float, kRowsPT> new_max;
        metal::vec<float, kRowsPT> factor;
        for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
        Stile.template row_reduce<NaxMaxOp>(new_max);
        Stile.template row_bin_op<NaxExpSubOp>(new_max);
        for (short i = 0; i < kRowsPT; ++i) {
            factor[i] = metal::exp(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
        }
        for (short i = 0; i < kRowsPT; ++i) sum_score[i] = sum_score[i] * factor[i];
        Stile.template row_reduce<NaxSumOp>(sum_score);
        Otile.template row_bin_op<NaxMulOp>(factor);
        // O += P V over the simdgroup's 128 columns (the latent rows of the step's keys)
        for (short id = 0; id < TV; id += 2) {
            for (short ik = 0; ik < TK; ++ik) {
                NAXTile<CT, 1, 2> Vtile;
                const device CT* vp = lat + ulong(kb + ik * kU) * lt + ulong(int(cs) * 128 + id * kU) * ld;
                if (unit) {
                    frag_t::load_rows(Vtile.frag_at(0, 0), vp, lt, Int<1>{}, short(kn - ik * kU));
                    frag_t::load_rows(Vtile.frag_at(0, 1), vp + kU, lt, Int<1>{}, short(kn - ik * kU));
                } else {
                    frag_t::load_rows(Vtile.frag_at(0, 0), vp, lt, ld, short(kn - ik * kU));
                    frag_t::load_rows(Vtile.frag_at(0, 1), vp + ulong(kU) * ld, lt, ld, short(kn - ik * kU));
                }
                frag_t::mma(
                    Otile.frag_at(0, id),
                    Otile.frag_at(0, id + 1),
                    Stile.frag_at(0, ik),
                    metal::false_type{},
                    Vtile.frag_at(0, 0),
                    Vtile.frag_at(0, 1),
                    metal::false_type{});
            }
        }
    }
    // the group's partials of each row: the simdgroup's 128 sums (row stride NJ DL), then its max and sum (once a row)
    {
        const ulong slot0 = (ulong(s) * ulong(H) + ulong(hb)) * ulong(NJ) + ulong(j);
        device float* por = po + slot0 * ulong(DL) + ulong(int(cs) * 128);
        const ulong rstr = ulong(NJ) * ulong(DL);
        for (short id = 0; id < TV; ++id) frag_t::store(Otile.frag_at(0, id), por, rstr, Int<1>{}, Int<0>{}, short(id * kU));
        if (cs == 0 && sn == 0) {
            for (short i = 0; i < kRowsPT; ++i) {
                const ulong sr = slot0 + ulong(int(sm) + int(i) * 8) * ulong(NJ);
                pm[sr] = max_score[i];
                pl[sr] = sum_score[i];
            }
        }
    }
