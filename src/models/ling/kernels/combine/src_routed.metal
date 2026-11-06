        // the joined routed array [rows, K, D]: pair (row, k)'s output row
        const device float* pk = routed;
        const ulong ok = (ulong(row) * ulong(K) + ulong(k)) * ulong(D) + ulong(col);
