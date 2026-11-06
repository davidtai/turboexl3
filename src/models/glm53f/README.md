# GLM-5.3-Flash

GLM-5.3-Flash runs on the core and DeepSeek-V4.1's banked decode; this folder holds only its tests:

| file | what it holds |
|---|---|
| `glm53f_test.zig` | every dense EXL3 shape of turboderp's GLM-5.3-Flash SC packs (trunk, MTP layer, head, vision tower) at each K they store it at: admitted by the dense kernels (CPU), and `linear` / `linearBias` against the host reference (GPU) |
| `routed_test.zig` | the routed geometry (4096 x 2048, 288 experts, top-8, `swiglu_limit` 10) at K2, K2.5, K3, K3.5, K4 and K5: admitted (CPU); `moeClamped` against the clamped reference, and `moeClampedBanked` against `moeClamped` byte for byte (GPU) |

docs/models/glm53f/ describes the model's needs and its packs' rates.
