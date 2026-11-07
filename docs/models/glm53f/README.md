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

The plugin's opt-in fused paths (each off by default, each byte for byte the chain it replaces, `fused_test.zig`):

| part of the model | entry point | what it replaces | the plugin's switch |
|---|---|---|---|
| routed experts in a prompt call over a slot bank | `moeClampedRows` | the call's gather (`take`), its cast to f32 and the f32 output's cast to bf16 around `moeClamped` at top-1: the chain's prepare reads each row-slot's x row through a row map (`turboexl3_pf_prep_r`, `prefill.text_prep_rows`) and its finish stores bf16 | `TURBOEXL3_GLM53F_PF_ROWS=1` |
| routed experts over the slot banks at decode rows | `moeClampedBankedAs` (bf16) | the f32 output's cast to bf16 after `moeClampedBanked`: the banked down finish stores bf16 | `TURBOEXL3_GLM53F_DEC_BF16=1` |
| the shared expert's gate and up at decode rows | `glm53f.shared_gu.gateUp` | `linear` of the gate and of the up (two DEC-ONE calls, 4 launches): one DEC-ONE launch over both trellises as they lie, each member at its own K split (8, where the joined width would take 4), suh and svh (`turboexl3_glm53f_shared_gu`, `shared_gu_test.zig`) | `TURBOEXL3_GLM53F_SHARED_GU=1` |

The plugin's opt-in schedules (each the default's bytes, `fused_test.zig`):

| part of the model | entry point | what changes | the plugin's switch |
|---|---|---|---|
| routed experts over the slot banks at decode rows | `configure` (`banked_group` 4, `banked_m3`, `banked_variant_gu`, `banked_variant_dn`) | the banked decode on the grouped texts (a weight's read and decode shared by up to four row-slots of one expert) and the grouped GEMVs' variants but `sh` and `pfsh` (their EXL3_GSH schedule gives other bytes at the K5 rate): the same reads, decodes and sums | `TURBOEXL3_GLM53F_BANKED=<texts>[.<gu>[.<dn>]]` |
| routed experts in a prompt call over a slot bank | `moeClamped`, `moeClampedRows` | the call's rows at most (2048 to 16384): a row's bytes do not depend on its call | `TURBOEXL3_GLM53F_PF_CALL_ROWS=<n>` |
| routed experts in a prompt call over a slot bank | `configure` (`pf_bk128`, `pf_half_rows`) | the routed prefill's down GEMM in 128-deep K stages, and a simdgroup whose rows end within its first 16 multiplying only that fragment row: the same MMAs in the same order | `TURBOEXL3_GLM53F_PF_OPTS=<bk128\|hr\|bk128.hr>` |
| the trunk's linears at prefill rows | `pgemm.host_table` (from `glm53f.pg2Table`, `api.zig`) | pg2 at a tile a linear from `pgemm.candidates`: MLX's fused NAX GEMM loop at that tile with the finish as its epilogue, one launch for the matmul and the finish (the chain's words where MLX runs its regular kernel) | `TURBOEXL3_GLM53F_PG2=<one code a trunk linear>` |

GLM-5.3-Flash has two kernels of its own: the row-map prepare (a composition of the core prepare's pieces) and the
shared expert's gate | up launch (dense's DEC-ONE pieces around its member choice; [kernels.md](kernels.md)). Its tests
(`src/models/glm53f/`) hold every shape and rate in the tables above: `glm53f_test.zig` holds each dense shape at each of
its K against `format.project`, `routed_test.zig` holds the routed experts (`moeClamped`, at every routed rate) against
the clamped reference and the banked decode against `moeClamped`, byte for byte, and `fused_test.zig` holds each fused
path to the chain it replaces and each schedule to the default's, byte for byte, and every pg2 candidate tile at every
trunk linear of `api.zig` to the chain's words; its timing gives the tile code a linear gets on this machine. [Tests](../../core/testing.md) gives their variables.

turboexl3 compiles each geometry's kernels once and keeps them for the process, in fixed caches (`src/root.zig`:
`dense_cache_slots` 64, `engine_cache_slots` 16 for the decode chains, `prefill_cache_slots` 16). One served pack
needs 20 dense (in, out, K) per activation dtype, 4 decode and 5 prefill geometries; a CPU test in `glm53f_test.zig`
holds each pack to the slots. The GPU tests run all four packs' rates in one process (52 dense configurations at two
dtypes, every routed rate), more than the caches hold, so they free the engines after each case
(`releaseEngines`).

No speed or quality result comes from this repository: the standard test through the outside plugin decides.
