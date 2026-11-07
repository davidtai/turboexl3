# Ling-3.1-flash

The code only Ling-3.1-flash uses, on the core's routed chain, DeepSeek-V4.1's banked decode and the core's dense linear:

| file | what it holds |
|---|---|
| `pair.zig` | `linearPair` (a same-input pair, each member in its own buffers, in one DEC-ONE launch: the prepare inside, or one prepare launch for both members then the GEMV over their rows), each member's `linear` bytes |
| `kernels/pair/` | the pair's pieces: the prepare's member choice (its suh, its rows), the GEMVs' member choice (its prepared rows or its suh) and the member's trellis, output tiles and svh |
| `api.zig` | `moeLing` (the routed experts with Ling's SwiGLU: none, or clamped after the SiLU) and `moeLingBanked` with its halves (the same over an expert streamer's three slot banks); the combine's entries |
| `ling_test.zig` | the texts' compositions, the reference clamp, the refusals and the trunk shapes (CPU); `moeLing` against `moe` and the reference, `moeLingBanked` against `moeLing` byte for byte, `linear` at every trunk shape, a one-row call on two rows' launch shape (`Config.dec_one_row_wide`) against one row's own and the three-launch chain byte for byte, and the pair against each member's `linear` byte for byte in both of its forms (GPU) |
| `shared.zig` | `sharedExpert` (the shared expert at decode rows: the gate and the up in one DEC-ONE launch, each in its own buffers, whose pairs of output blocks run the host's SwiGLU and the down's prepare, then the down's DEC-ONE GEMV), `sharedTakes`, `sigmoidTable` (the host's tables) |
| `shared_test.zig` | the joint texts' compositions and pinned statements (CPU); `sharedExpert` against the composed chain word for word at rows 1 to 4, limits 0 and 10 (GPU) |
| `kernels/shared/` | the pieces the shared expert's texts add to dense's DEC-ONE: the member choice and its tile address, the member's store, the pair's arrival, the two SwiGLUs and the down's prepare; `kernels/test/pins/` what the CPU test pins |
| `combine.zig` | `combineRouted` and `combineParts` (a wide routed call's combine in one launch, over a joined array or the call's unjoined outputs), the texts' CPU test |
| `combine_test.zig` | both against MLX's composed combine word for word, the refusals (GPU) |
| `kernels/combine/` | the combine's pieces: the contract pragma, the source choice, the fold's head, each text's row read, the fold and store |
| `addnorm.zig` | `addNormLinears` (a decode layer's mixer input: the residual add, its RMSNorm and up to three EXL3 members' prepares in one launch, then each member's GEMV over its prepared rows) and `addNormPrepare` (the launch alone), the texts' CPU test |
| `addnorm_test.zig` | `addNormLinears` against MLX's add, MLX's fast rms_norm and each member's `linear` word for word at decode rows 1 to 8, 12 and 16, in bf16 and f16, each member's own `linear` with its prepare inside or in its own launch; the refusals (GPU) |
| `kernels/addnorm/` | the launch's pieces: the member's prepare (a function in its header), the add and norm, each member's call and the counters |

Ling's own Metal is the pairs', the shared expert's, the routed combine's and the mixer input's add and norm pieces
(`kernels/pair/`, `kernels/shared/`, `kernels/combine/`, `kernels/addnorm/`; `kernels/test/pins/` what the CPU tests pin); the routed experts' clamp pieces are the core's
(`kernels/decode/fin7_swiglu_clamp_out.metal`, `kernels/prefill/mid_swiglu_clamp_out.metal`). docs/models/ling/
describes the model's needs and its kernels.
