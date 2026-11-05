            uint base = L * 4;
            float acc = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                float v = float(src[base + i]);
                acc += v * v;
            }
            acc = simd_sum(acc);
            float inv = metal::precise::rsqrt(acc / float(IHD) + float(qeps));
            OUT_T nrm[4];
            for (uint i = 0; i < 4; ++i) {
                nrm[i] = qn[base + i] * OUT_T(float(src[base + i]) * inv);
            }
            constexpr uint half_rd = uint(RD) / 2;
            float partner[4];
            for (uint i = 0; i < 4; ++i) {
                partner[i] = simd_shuffle(float(nrm[i]), L ^ (half_rd / 4));
            }
            for (uint i = 0; i < 4; ++i) {
                uint j = base + i;
                OUT_T r;
                if (j < half_rd) {
                    r = OUT_T(float(nrm[i]) * arow[j] - partner[i] * arow[half_rd + j]);
                } else if (j < uint(RD)) {
                    r = OUT_T(partner[i] * arow[j] + float(nrm[i]) * arow[j - half_rd]);
                } else {
                    r = nrm[i];
                }
                dst[j] = r;
            }
