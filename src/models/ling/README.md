# Ling-3.1-flash

The code only Ling-3.1-flash uses, on the core's routed chain, DeepSeek-V4.1's banked decode and the core's dense linear:

| file | what it holds |
|---|---|
| `pair.zig` | `linearPair` (a same-input pair, each member in its own buffers, in one DEC-ONE launch: the prepare inside, or one prepare launch for both members then the GEMV over their rows), each member's `linear` bytes |
| `kdaproj.zig` | `kdaProj` (a KDA layer's q|k|v, f and g in one DEC-ONE GEMV launch after one prepare launch, each member in its own buffers at its own K split), `kdaProjPrepared` (the GEMV alone over a host's prepared rows), `ncntFor`; each member's `linear` bytes |
| `kernels/kdaproj/` | the KDA projections' pieces: the prepare's member choice, DEC-ONE's head over three members (each member's widths and K split), the member's prepared rows, trellis, svh, output, partial sums and counters |
| `kernels/pair/` | the pair's pieces: the prepare's member choice (its suh, its rows), the GEMVs' member choice (its prepared rows or its suh) and the member's trellis, output tiles and svh |
| `api.zig` | `moeLing` (the routed experts with Ling's SwiGLU: none, or clamped after the SiLU) and `moeLingBanked` with its halves (the same over an expert streamer's three slot banks); the combine's entries |
| `ling_test.zig` | the texts' compositions, the reference clamp, the refusals and the trunk shapes (CPU); `moeLing` against `moe` and the reference, `moeLingBanked` against `moeLing` byte for byte, `linear` at every trunk shape, a one-row call on two rows' launch shape (`Config.dec_one_row_wide`) and on the one-row table's launch (`Config.dec_one_row_table`, whose per-shape choice the CPU tests pin) against one row's own and the three-launch chain byte for byte, and the pair against each member's `linear` byte for byte in both of its forms (GPU) |
| `shared.zig` | `sharedExpert` (the shared expert at decode rows: the gate and the up in one DEC-ONE launch, each in its own buffers, whose pairs of output blocks run the host's SwiGLU and the down's prepare, then the down's DEC-ONE GEMV), `sharedTakes`, `sigmoidTable` (the host's tables) |
| `shared_test.zig` | the joint texts' compositions and pinned statements (CPU); `sharedExpert` against the composed chain word for word at rows 1 to 4, limits 0 and 10 (GPU) |
| `kernels/shared/` | the pieces the shared expert's texts add to dense's DEC-ONE: the member choice and its tile address, the member's store, the pair's arrival, the two SwiGLUs and the down's prepare; `kernels/test/pins/` what the CPU test pins |
| `combine.zig` | `combineRouted` and `combineParts` (a wide routed call's combine in one launch, over a joined array or the call's unjoined outputs), the texts' CPU test |
| `combine_test.zig` | both against MLX's composed combine word for word, the refusals (GPU) |
| `kernels/combine/` | the combine's pieces: the contract pragma, the source choice, the fold's head, each text's row read, the fold and store |
| `addnorm.zig` | `addNormLinears` (a decode layer's mixer input: the residual add, its RMSNorm and up to three EXL3 members' prepares in one launch, then each member's GEMV over its prepared rows) and `addNormPrepare` (the launch alone), the texts' CPU test |
| `addnorm_test.zig` | `addNormLinears` against MLX's add, MLX's fast rms_norm and each member's `linear` word for word at decode rows 1 to 8, 12 and 16, in bf16 and f16, each member's own `linear` with its prepare inside or in its own launch; the refusals (GPU) |
| `kernels/addnorm/` | the launch's pieces: the member's prepare (a function in its header), the add and norm, each member's call and the counters |
| `router.zig` | `route`, `routerLogits`, `select` and `tableF32` (a MoE layer's router at decode rows 1 to 8 in two launches: MLX's one-row gemv over the host loader's column-major table for the logits, or over the loaded f16 weight widened then + 0, beside its gemv / gemv_wide for the read-ahead predictor's scores, then the host's fused router text over the logits with its weights in bf16; the loader's f32 table rebuilt from the f16 weight), the texts' and the plan's CPU tests |
| `router_test.zig` | the logits, both predictor scores, the ids and the weights against the host's served path word for word at rows 1 to 8 (every operand built as the host builds it: the loaded f16 weight, the loader's column-major table, the MoE input, denseRows), a weight holding every finite f16 word, the rebuilt table at the stock widths, with `TURBOEXL3_LING_PACK=<pack dir>` every router weight of the pack as stored (the MoE layers' and the NextN layer's), the refusals (GPU) |
| `kernels/router/` | the router's pieces: launch 1's head, MLX's gemv statements cut at the table's read (an f32 word, or an f16 word widened then + 0), the predictor's branch (gemv at one row, gemv_wide at 2 to 8), the two scores (headers) and the host's fused router text (`select.metal`) |

Ling's own Metal is the pairs', the shared expert's, the routed combine's, the mixer input's add and norm and the
router's pieces (`kernels/pair/`, `kernels/shared/`, `kernels/combine/`, `kernels/addnorm/`, `kernels/router/`;
`kernels/test/pins/` what the CPU tests pin); the routed experts' clamp pieces are the core's
(`kernels/decode/fin7_swiglu_clamp_out.metal`, `kernels/prefill/mid_swiglu_clamp_out.metal`). docs/models/ling/
describes the model's needs and its kernels.
