    // Ripple's value input as the prepare below reads it (x[i]): lerp(x0, xa, blend) = x0 + blend (xa - x0) in f32, back
    // in T, one host op a statement (astype f32, subtract, multiply by the f32 scalar, add, astype T); xa itself where
    // `on` is false
    struct RicBlendIn {
        const device T* a;
        const device T* b;
        float lam;
        bool on;
        T operator[](uint i) const {
            if (!on) return a[i];
            const float base = float(b[i]);
            const float d = float(a[i]) - base;
            const float p = lam * d;
            const float m = base + p;
            return T(m);
        }
    };
