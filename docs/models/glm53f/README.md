# GLM-5.3-Flash

GLM-5.3-Flash is a mixture-of-experts model: 45 layers (KDA linear attention, every fourth layer DeepSeek sparse
attention), hidden 4096, 288 routed experts of width 2048 at top-8 and one shared expert, three dense layers, one MTP
layer and a vision tower. Its routed experts and its dense MLPs use DeepSeek-V4's clamped SwiGLU (`swiglu_limit` 10:
the gate clamped from above, the up to [-10, 10], before the SiLU). turboexl3 serves its EXL3 linears for the outside
plugin `turboexl3_glm53f` (mlx-serve).

The packs are turboderp's self-calibrated EXL3 branches of
[turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) (exllamav3 1.6, MUL1, head 6
bits). Each stores every linear as EXL3. The rates, read from each branch's tensor headers:

| part | shape (in -> out) | sc_2.10bpw | sc_2.60bpw | sc_3.10bpw | sc_4.10bpw |
|---|---|---|---|---|---|
| routed experts gate, up / down | 4096 -> 2048 / 2048 -> 4096 | K2, layer 3 K3 | K2.5, layer 3 K3.5 | K3, layer 3 K4 | K4, layer 3 K5 |
| KDA `qkv_proj`, `o_proj` | 4096 -> 24576, 8192 -> 4096 | K5 | K6 | K6 | K8 |
| DSA `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj`; indexer `wq_b` | 4096 -> 1536, 1536 -> 16384, 4096 -> 512, 16384 -> 4096; 1536 -> 4096 | K5 | K6 | K6 | K8 |
| shared expert gate, up / down | 4096 -> 2048 / 2048 -> 4096 | K6 | K6 | K6 | K8 |
| dense layers' MLP (layers 0-2) | 4096 -> 12288 / 12288 -> 4096 | K5 | K6 | K6 | K6 |
| MTP layer: attention, shared expert | as the trunk's | K3 | K4 | K4 | K6 |
| MTP layer: `eh_proj` | 8192 -> 4096 | K3 | K4 | K4 | K5 |
| MTP layer: routed experts | as the trunk's | K3 | K4 | K4 | K4 |
| `lm_head` | 4096 -> 154880 | K6 | K6 | K6 | K6 |
| vision blocks: q, k, v, proj (biased) / MLP (biased) | 1024 -> 1024 / 1024 -> 4096, 4096 -> 1024 | K6 | K6 | K6 | K6 |
| vision merger: proj / gate, up / down | 4096 -> 4096 / 4096 -> 10240 / 10240 -> 4096 | K6 | K6 | K6 | K6 |

K2.5 and K3.5 are exllamav3 1.6's half-integer rates: 40 and 56 halfwords per 256-weight tile.

What it runs on turboexl3:

| part of the model | entry points | kernels |
|---|---|---|
| routed experts in a prompt (a streamer's slot rows) and the MTP layer's resident bank | `moeClamped` | the core's routed decode and prefill chains with the clamp |
| routed experts over an expert streamer's slot banks at decode rows | `moeClampedBanked` | DeepSeek-V4.1's banked decode (`src/models/deepseek_v41/banked.zig`) |
| the trunk, the head, the MTP layer's linears, the vision merger | `linear` | the core's dense linear |
| the vision blocks' biased linears | `linearBias` | the core's dense linear, then the bias |

GLM-5.3-Flash has no kernel of its own. Its tests (`src/models/glm53f/`) hold every shape and rate in the table above:
`glm53f_test.zig` holds each dense shape at each of its K against `format.project`, and `routed_test.zig` holds the
routed experts (`moeClamped`, at every routed rate) against the clamped reference and the banked decode against
`moeClamped`, byte for byte. [Tests](../../core/testing.md) gives their variables.

No speed or quality result comes from this repository: the standard test through the outside plugin decides.
