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
