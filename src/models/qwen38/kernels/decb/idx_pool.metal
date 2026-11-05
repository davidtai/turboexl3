        uint lane = L;
        uint kbase = lane * 4;
        float pxs[4];
        float sumsq = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            float acc = 0.0f;
            for (uint r = 0; r < uint(RATIO); ++r) acc = float((r < uint(ln)) ? OUT_T(ring[r * 128 + kbase + j]) : OUT_T(ist[r - uint(ln)][kbase + j])) + acc;
            const OUT_T pooled = OUT_T(acc * invr);
            pxs[j] = float(pooled);
            sumsq += pxs[j] * pxs[j];
        }
        sumsq = simd_sum(sumsq);
        float pinv = metal::precise::rsqrt(sumsq / 128.0f + peps);
        for (uint j = 0; j < 4; ++j) {
            const OUT_T nv = kn[lane * 4 + j] * OUT_T(pxs[j] * pinv);
            const float nf = float(nv);
            const float up = simd_shuffle_down(nf, ushort(HALF / 4));
            const float dn = simd_shuffle_up(nf, ushort(HALF / 4));
            uint pd = lane * 4 + j;
            OUT_T po = nv;
            if (pd < uint(RD)) {
                const OUT_T rot = pd < uint(HALF) ? OUT_T(-up) : OUT_T(dn);
                const OUT_T pa = OUT_T(nf * float(pcos[pd]));
                const OUT_T pb = OUT_T(float(rot) * float(psin[pd]));
                po = OUT_T(float(pa) + float(pb));
            }
            op[pd] = po;
        }
