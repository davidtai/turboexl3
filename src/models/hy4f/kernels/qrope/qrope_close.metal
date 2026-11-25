        // the rope part's rotated value (its pair's first or second), q's value elsewhere; f32, MLX's T (no rounding)
        q[ulong(row) * DIMC + o] = (c >= NOPEC) ? (second ? rx2 : rx1) : qv;
    }
