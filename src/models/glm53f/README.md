# GLM-5.3-Flash

GLM-5.3-Flash runs on the core and DeepSeek-V4.1's banked decode (its prompt calls' row-map prepare is a core prefill
text, `prefill.text_prep_rows`, and its entry `root.moeClampedRows`); this folder holds its tests:

| file | what it holds |
|---|---|
| `glm53f_test.zig` | every dense EXL3 shape of turboderp's GLM-5.3-Flash SC packs (trunk, MTP layer, head, vision tower) at each K they store it at: admitted by the dense kernels (CPU), and `linear` / `linearBias` against the host reference (GPU) |
| `routed_test.zig` | the routed geometry (4096 x 2048, 288 experts, top-8, `swiglu_limit` 10) at K2, K2.5, K3, K3.5, K4 and K5: admitted (CPU); `moeClamped` against the clamped reference, and `moeClampedBanked` against `moeClamped` byte for byte (GPU) |
| `fused_test.zig` | the plugin's fused paths at that geometry and every rate, byte for byte: `moeClampedRows` against the prompt call's gather, casts and `moeClamped` (the row-map prepare `prefill.text_prep_rows`, CPU text check too), and `moeClampedBankedAs` at bf16 against `moeClampedBanked` cast to bf16 (GPU); the plugin's schedules: the banked decode's grouped texts and GEMV variants but `sh` and `pfsh` (selected per call and configured) against the served banked call, a prompt call in calls of 2048, 8192 and 16384 rows against its 4096-row calls, and under the routed prefill's bk128 and half_rows against the served options; every pg2 candidate tile at every trunk linear of `api.zig` against the chain (GPU); `TURBOEXL3_GLM_TIME=1` times each fused path against its chain, the sc_3.10bpw trunk's linears at decode and prompt rows, the banked schedules, the prompt call sizes and options, and each pg2 candidate against the chain (the code a linear gets) |
| `pf_stage_test.zig` | the routed prefill's 128-row tile launches at 128-deep K stages (`prefill.Options.t128_bk128`, `Config.pf_t128_bk128`): where the option is taken and the depth each launch runs (CPU); its bytes and timing are `fused_test.zig`'s prefill options tests |
| `zfit_test.zig` | the mixed launch's outputs at the rows they hold (`prefill.Options.t128_zfit`, `Config.pf_t128_zfit`, `root.moeClampedRowsFit`): the rows `prefill.t128ZCap` gives hold every remainder the route leaves (CPU), and a prompt call with it gives the served words, the call's experts named or not, at every SC rate over drawn, 455-row and remainder-edge routings (GPU, TURBOEXL3_GLM_PF_ZFIT_BYTES) |
| `api.zig` | the host helpers (root's `glm53f`): the trunk linears at prefill rows and the pg2 table a host builds from one tile code a linear (`pg2Table`), with CPU tests of the codes and of the host table's lookup |

docs/models/glm53f/ describes the model's needs and its packs' rates.
