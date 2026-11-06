# Ling-3.1-flash

The code only Ling-3.1-flash uses, on the core's routed chain and DeepSeek-V4.1's banked decode:

| file | what it holds |
|---|---|
| `api.zig` | `moeLing` (the routed experts with Ling's SwiGLU: none, or clamped after the SiLU) and `moeLingBanked` with its halves (the same over an expert streamer's three slot banks) |
| `ling_test.zig` | the texts' compositions, the reference clamp, the refusals and the trunk shapes (CPU); `moeLing` against `moe` and the reference, `moeLingBanked` against `moeLing` byte for byte, `linear` at every trunk shape, and a one-row call on two rows' launch shape (`Config.dec_one_row_wide`) against one row's own and the three-launch chain byte for byte (GPU) |

Ling has no Metal of its own: its clamp pieces are the core's (`kernels/decode/fin7_swiglu_clamp_out.metal`,
`kernels/prefill/mid_swiglu_clamp_out.metal`). docs/models/ling/ describes the model's needs.
