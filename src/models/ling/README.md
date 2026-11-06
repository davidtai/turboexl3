# Ling-3.1-flash

The code only Ling-3.1-flash uses, on the core's routed chain and DeepSeek-V4.1's banked decode:

| file | what it holds |
|---|---|
| `api.zig` | `moeLing` (the routed experts with Ling's SwiGLU: none, or clamped after the SiLU) and `moeLingBanked` with its halves (the same over an expert streamer's three slot banks); the combine's entries |
| `ling_test.zig` | the texts' compositions, the reference clamp, the refusals and the trunk shapes (CPU); `moeLing` against `moe` and the reference, `moeLingBanked` against `moeLing` byte for byte, `linear` at every trunk shape (GPU) |
| `combine.zig` | `combineRouted` and `combineParts` (a wide routed call's combine in one launch, over a joined array or the call's unjoined outputs), the texts' CPU test |
| `combine_test.zig` | both against MLX's composed combine word for word, the refusals (GPU) |
| `kernels/combine/` | the combine's pieces: the contract pragma, the source choice, the fold's head, each text's row read, the fold and store |

Ling's routed experts have no Metal of their own: their clamp pieces are the core's
(`kernels/decode/fin7_swiglu_clamp_out.metal`, `kernels/prefill/mid_swiglu_clamp_out.metal`). Its own Metal is the
routed combine's (`kernels/combine/`; `kernels/test/pins/combine.metal` what its CPU test pins). docs/models/ling/
describes the model's needs and its kernels.
