// The prompt-width tail's norm operands (tail.zig `moeTailWide`): WE f32 [D + 1] holds the norm's weight widened to f32
// (exact: every bf16 or f16 word is an f32 word) and eps after it, one buffer, so the launch binds MLX's 31 (H, s0..s23,
// loc, WT, XS, WE and its two outputs); tail_2 reads them as the decode launch's W and eps
const device float* W = WE;
const float eps = WE[D];
