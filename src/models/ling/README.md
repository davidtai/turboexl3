# Ling-3.1-flash

The code only Ling-3.1-flash uses, on the core's routed chain and DeepSeek-V4.1's banked decode:

| file | what it holds |
|---|---|
| `pair.zig` | `linearPair` (a same-input pair, each member in its own buffers, in one DEC-ONE launch: the prepare inside, or one prepare launch for both members then the GEMV over their rows), each member's `linear` bytes |
| `kernels/pair/` | the pair's pieces: the prepare's member choice (its suh, its rows), the GEMVs' member choice (its prepared rows or its suh) and the member's trellis, output tiles and svh |
| `api.zig` | `moeLing` (the routed experts with Ling's SwiGLU: none, or clamped after the SiLU) and `moeLingBanked` with its halves (the same over an expert streamer's three slot banks) |
| `ling_test.zig` | the texts' compositions, the reference clamp, the refusals and the trunk shapes (CPU); `moeLing` against `moe` and the reference, `moeLingBanked` against `moeLing` byte for byte, `linear` at every trunk shape, a one-row call on two rows' launch shape (`Config.dec_one_row_wide`) against one row's own and the three-launch chain byte for byte, and the pair against each member's `linear` byte for byte in both of its forms (GPU) |

Ling's own Metal is the pair's pieces (`kernels/pair/`); its clamp pieces are the core's (`kernels/decode/fin7_swiglu_clamp_out.metal`,
`kernels/prefill/mid_swiglu_clamp_out.metal`). docs/models/ling/ describes the model's needs.
