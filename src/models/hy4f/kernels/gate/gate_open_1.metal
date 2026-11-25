    // text_finish's store loop with the gate's value kept (models/hy4f/gate.zig): MLX's Sigmoid of it in f32
    // (gate_sig), o at the same element read in o's own [1, H, S, VH] layout (gate_src), and their product stored in G_T
    // where text_finish stores y (gate_close)
    constexpr uint VHC = uint(VH);
    static_assert(uint(H) * uint(VH) == uint(DIM), "hy4f gate: H heads of VH are the gate's DIM outputs");
