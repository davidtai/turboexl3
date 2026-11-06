# Ling-3.1-flash

Ling-3.1-flash (inclusionAI, `BailingMoeV3ForCausalLM`, model type `bailing_hybrid`) has 72 layers at hidden 4096:
63 KDA linear-attention layers and 9 MLA layers (every eighth), the first 4 layers dense (FFN 10240), then a grouped
sigmoid MoE of 512 experts at width 1280, top-8, and one shared expert of 1280. turboderp's EXL3 packs
(`turboderp/Ling-3.1-flash-exl3`) quantize the routed experts at one rate on every layer (2.60 bpw: K 2.5, n 40; 2.10
bpw: K 2, n 32; MUL1), every trunk linear and the head at K6 and the dense FFN at K4. The outside plugin
`turboexl3_ling` (mlx-serve) is the host; it streams the routed experts from SSD.

New against Ling 3.0: a per-layer clamp of the SwiGLU, `expert_swiglu_limit_list` (routed, 10 on layers 65 to 71) and
`share_expert_swiglu_limit_list` (shared, 10 on layers 64 to 71), 0 elsewhere (off). The clamp is applied AFTER the
SiLU: `min(silu(g), L) · clamp(u, −L, L)` (exllamav3's `act_limit` with SiLU, `activation_kernels.cuh`; inclusionAI's
vLLM and SGLang the same). DeepSeek-V4's and GLM's clamp the gate before it, `silu(min(g, L)) · clamp(u, −L, L)`; the
two part wherever a gate's SiLU passes L. A geometry carries which one (`format.ClampAt`: `gate_in`, `silu_out`).

What it runs on turboexl3 (`root.ling`, `src/models/ling/api.zig`):

| part | entry point | kernels |
|---|---|---|
| a routed layer's experts over a stacked bank, any rows | `moeLing` (`limit` 0 or the layer's clamp) | `moe`'s chain at limit 0; with a limit, decode form 9 with `turboexl3_fin7co` and the prefill chain with `turboexl3_pf_t128_midco` |
| a decode wave's routed pairs over an expert streamer's three slot banks | `moeLingBanked` (`moeLingBankedGateUp`, `moeLingBankedDown`) | DeepSeek-V4.1's banked decode with `turboexl3_fin7b` (limit 0) or `turboexl3_fin7cob`, or their grouped texts (`fin7bg`, `fin7cobg`) under `Config.banked_group` 4 |
| every trunk linear and the head | `root.linear` | the core's dense linear |
| the shared expert at decode rows (1 to 4) | `ling.sharedExpert` (`root.lingSharedExpert`; `sharedTakes`, `sigmoidTable`) | `turboexl3_ling_sgu` (no clamp) or `turboexl3_ling_sguco` (the clamp after the SiLU), then the core's `turboexl3_dense_dec0` ([its catalog](kernels.md)) |

The routed shape is new to the library: an expert width of 1280 is ten 128-blocks (the split-K planes, the fin7 grid
and the prefill tiles take it as any multiple of 128), and 512 experts at top-8. Every kernel is a composition of the
core's or DeepSeek-V4.1's pieces with one SwiGLU line swapped, so its bytes are the base launch's wherever no value
reaches the limit. The tests (`src/models/ling/ling_test.zig`) hold `moeLing` to `ref.moeRowClampedAt` at `silu_out`,
`moeLingBanked` to `moeLing` byte for byte, and `linear` to the host reference at every trunk shape. The kernels:
[the core's catalog](../../core/kernels.md) (`turboexl3_fin7co`, `turboexl3_pf_midco`, `turboexl3_pf_mid_vco`,
`turboexl3_pf_t128_midco`) and [DeepSeek-V4.1's](../deepseek_v41/kernels.md) (`turboexl3_fin7b`, `turboexl3_fin7cob`,
`turboexl3_fin7bg`, `turboexl3_fin7cobg`).

`TURBOEXL3_LING_TIME=1` (with `TURBOEXL3_GPU=1`) times the Ling kernels at the model's shapes for the serving plan: every
trunk linear, `moeLing` and `moeLingBanked`, and the decode forward's linears chained over the 72 layers
(`TURBOEXL3_LING_TIME` and `TURBOEXL3_LING_CHAIN` lines). It is a diagnostic: it asserts nothing, and its numbers steer
work; they never decide it.

The shared expert (`src/models/ling/shared.zig`) takes the composed chain's four dependent levels at decode rows (the
gate and the up, the host's SwiGLU, the down's prepare, its GEMV; nine with the clamp's MLX ops) in two: one DEC-ONE
launch of the gate and the up, each member in its own buffers (no joined copy), whose output blocks' finishes, taken
in pairs (the gate's 128-block and the up's), run the host's SwiGLU and the down linear's prepare in the pair's second
finish, then the down's DEC-ONE GEMV. Its bytes are the composed chain's: each member's `linear`, mlx-serve's
fusedSwiGLU (the sigmoid from the host's table) or the plugin's swigluClampOut (in f32, MLX's f32 sigmoid from a
table of every 16-bit gate), and `linear(down)` (`src/models/ling/shared_test.zig`, word for word).

No speed verdict comes from this repository: the standard test through the outside plugin decides.
