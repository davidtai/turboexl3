//! The kernel texts more than one kernel splices, MLX-free (this file imports nothing): turboexl3's own kernels take
//! them from here, and so may a consumer's Metal, so each text has one source. A consumer that splices one pins it in its
//! own tests, since a change here changes every kernel built from it. mlx-serve's `qwen38_exl3` model plugin splices
//! the gate's gemv and sum into its router kernel (the router's `softmax_gate_gemv` mode).

/// The 128-point Hadamard butterfly on `v[4]` (lane L holds block element L + 32 j): the five xor-shuffle stages and the
/// four-point register stage every dense prepare and finish runs, verbatim. `decb.zig`'s gated prepares and merges and
/// `gdn_out.zig`'s GDN tail splice it.
pub const had128 = @embedFile("kernels/decb/had128.metal");

/// The shared gate's logit for one row exactly as MLX's gemv sums it at the verify widths, as one simdgroup's part
/// (`decode.text_gate_gemv` gives the order).
pub const gate_gemv = @embedFile("kernels/decode/gate_gemv.metal");

/// The gate's sum of the gemv's 8 partials in its simdgroup 0 lane 0 order, and its bf16 store (`decode.text_gate_sum`).
pub const gate_sum = @embedFile("kernels/decode/gate_sum.metal");
