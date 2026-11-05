    // the down linear's input as the prepare below reads it (x[row * DIM + i]): the host's clamped SwiGLU of the gate's
    // and up's outputs in the activation dtype T (mlx-serve's clampedSwiGLU: MLX's Minimum, clip as Maximum then Minimum,
    // the sigmoid read from the host's table of MLX's own results, Multiply, Multiply). The gate's row r starts at
    // r * GSTRIDE, the up's at r * GSTRIDE + UOFF (one joined output [rows, 2 DIM]: GSTRIDE 2 DIM, UOFF DIM, gate = up)
    struct RicSwigluIn {
        const device T* g;
        const device T* u;
        const device T* tab;
        T hi;
        T lo;
        T operator[](uint idx) const {
            const uint r = idx / uint(DIM);
            const uint i = idx - r * uint(DIM);
            const T gv = g[ulong(r) * uint(GSTRIDE) + i];
            const T uv = u[ulong(r) * uint(GSTRIDE) + uint(UOFF) + i];
            const T gm = metal::isnan(float(gv)) ? gv : (gv < hi ? gv : hi);
            const T um = metal::isnan(float(uv)) ? uv : (uv > lo ? uv : lo);
            const T uc = metal::isnan(float(um)) ? um : (um < hi ? um : hi);
            const T sig = tab[as_type<ushort>(gm)];
            const T act = gm * sig;
            return act * uc;
        }
    };
    const RicSwigluIn x = { gate, up, sigtab, lim[0], T(-float(lim[0])) };
