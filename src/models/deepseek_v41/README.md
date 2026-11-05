# DeepSeek-V4.1

The code only DeepSeek-V4.1 uses, on turboexl3's core:

| file | what it holds |
|---|---|
| `grouped.zig` | the grouped (block-diagonal) linear at decode rows, the attention's `wo_a` (`root.linearGrouped`) |
| `deepseek_test.zig`, `deepseek_gemv_test.zig` | the trunk's linears (`linear`, `linearBias`, `linearGrouped`) at every trunk and vision shape against the host reference, and the dense decode GEMV's launch shapes |
| `f3_test.zig` | F3: the shared expert's gate and up finishes, the clamped SwiGLU and the down's prepare in one launch (`dense.text_swiglu_prep`) |
| `clamped_test.zig` | the clamped SwiGLU (`root.moeClamped`) at DeepSeek-V4.1-Flash's routed geometry |
| `banked_test.zig`, `banked_grp_test.zig`, `banked_ilp_test.zig` | the banked decode over an expert streamer's slot banks (`root.moeClampedBanked`) and its grouped GEMVs' variants |
| `kernels/<module>/` | the Metal texts of `grouped.zig`; `kernels/test/pins/` what the tests pin |

The kernels these tests hold (the clamped fin7 and prefill mids, the banked and grouped GEMVs, F3) stay in the core
engines that launch them (`src/core/decode.zig`, `prefill.zig`, `dense.zig`); docs/core/kernels.md names the model of every
kernel.
