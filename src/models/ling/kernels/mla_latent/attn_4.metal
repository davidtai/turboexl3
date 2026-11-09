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
