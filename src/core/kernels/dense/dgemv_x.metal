        float xr[RBC][4];
        for (uint r = 0; r < RBC; ++r) {
            // a chunk's spare rows (ROWS not a multiple of RB) reread the last row; their sums are never written
            const uint row = min(r0 + r, ROWSC - 1u);
            const half4 x4 = *reinterpret_cast<const device half4*>(xbase + ulong(row) * ulong(INC) + tk * 16u);
            xr[r][0] = float(x4.x); xr[r][1] = float(x4.y); xr[r][2] = float(x4.z); xr[r][3] = float(x4.w);
        }
