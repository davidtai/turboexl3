# DeepSeek-V4.1

The code only DeepSeek-V4.1 uses, on turboexl3's core:

| file | what it holds |
|---|---|
| `api.zig` | DeepSeek-V4.1's entry points, re-exported by `root.zig` under their own names |
| `grouped.zig` | the grouped (block-diagonal) linear at decode rows, the attention's `wo_a` (`root.linearGrouped`) |
| `banked.zig` | the banked decode over an expert streamer's slot banks and its grouped GEMVs (`Banked`, kept per `decode.Kernels3`) |
| `f3.zig` | F3: the shared expert's gate and up finishes, the clamped SwiGLU and the down's prepare in one launch (`F3`, kept per `dense.Dense`; `root.sharedSwiglu`) |
| `deepseek_test.zig`, `deepseek_gemv_test.zig` | the trunk's linears (`linear`, `linearBias`, `linearGrouped`) at every trunk and vision shape against the host reference, and the dense decode GEMV's launch shapes |
| `f3_test.zig` | F3's tests (`f3.text_swiglu_prep`) |
| `clamped_test.zig` | the clamped SwiGLU (`root.moeClamped`) at DeepSeek-V4.1-Flash's routed geometry |
| `banked_test.zig`, `banked_grp_test.zig`, `banked_ilp_test.zig` | the banked decode over an expert streamer's slot banks (`root.moeClampedBanked`) and its grouped GEMVs' variants |
| `kernels/<module>/` | the Metal texts of `grouped.zig`, `banked.zig` and `f3.zig`; `kernels/test/pins/` what the tests pin |

The clamped fin7 and prefill mids these tests hold stay in the core (`src/core/decode.zig`, `prefill.zig`): DeepSeek-V4
and V4.1 and GLM share them. docs/core/kernels.md and docs/models/deepseek_v41/kernels.md name the users of every kernel.
