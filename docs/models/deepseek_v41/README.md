# DeepSeek-V4.1

DeepSeek-V4.1 is a mixture-of-experts model whose routed experts use a clamped SwiGLU (`swiglu_limit` 10) and whose
attention output runs through a grouped (block-diagonal) linear, `wo_a`. turboexl3 serves its EXL3 linears for a host
(mlx-serve's DeepSeek-V4.1 plugins); its tests run at DeepSeek-V4.1-Flash's geometry (hidden 5120, expert width 2304,
384 routed experts, top-6, MUL1).

What it runs on turboexl3:

| part of the model | entry points | kernels |
|---|---|---|
| routed experts | `moeClamped` | the core's routed decode (`turboexl3_fin7c`) and prefill (`turboexl3_pf_midc`, `_mid_vc`, `_t128_midc`) chains with the clamp ([engines](../../core/engines.md)) |
| routed experts over an expert streamer's slot banks | `moeClampedBanked`, `moeClampedBankedGateUp`, `moeClampedBankedDown` | the banked decode texts on the core's routed decode (`src/models/deepseek_v41/banked.zig`: `turboexl3_prep2b`, `gu3b`, `fin7cb`, `down3b`, `dfin3b` and their grouped variants) |
| the DSpark stages (top-3) and the shared expert as a one-expert bank | `moe`, `moeClamped` | the core's decode chain below a top-k of 4 (form 3's `dfin3`) |
| the shared expert at decode rows | `sharedSwiglu` | F3 (`src/models/deepseek_v41/f3.zig`), `turboexl3_dense_swiglu_prep`: the gate and up finishes, the clamped SwiGLU and the down's prepare in one launch |
| the trunk linears, the vision tower and the aligner | `linear`, `linearBias` | the core's dense linear |
| the attention's `wo_a` | `linearGrouped` | `src/models/deepseek_v41/grouped.zig` ([kernels](kernels.md)) |

Packs: no DeepSeek-V4.1 EXL3 pack is published from this repository; the tests run on synthetic weights at its shapes
(K2, K3 and K6 MUL1).
