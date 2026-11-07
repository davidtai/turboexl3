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
| a decode wave's routed pairs over an expert streamer's three slot banks | `moeLingBanked` (`moeLingBankedGateUp`, `moeLingBankedDown`; `moeLingBankedSel` on a selection of the texts) | DeepSeek-V4.1's banked decode with `turboexl3_fin7b` (limit 0) or `turboexl3_fin7cob`, or their grouped texts (`fin7bg`, `fin7cobg`) under `Config.banked_group` 4; with `Config.banked_gu8` the gate\|up's prepare inside its launch (`turboexl3_gu8bg`, `turboexl3_gu8bgm3`, the same bytes) |
| every trunk linear and the head | `root.linear` | the core's dense linear; a host's opt-in `Config.dec_one_row_wide` runs a one-row call on two rows' launch shape, and `Config.dec_one_row_table` on its trunk shape's `dense.one_row_table` launch (the same bytes) |
| a wide (prompt) routed call's combine: the routed rows times their routing weights, summed over the top-8 | `root.ling.combineRouted` (a joined array), `root.ling.combineParts` (the call's unjoined outputs, JOINLESS) | Ling's own: `turboexl3_ling_combine`, `turboexl3_ling_combine_parts` ([kernels.md](kernels.md)), the words of the host's `sum(multiply(astype(routed, f32), expand_dims(weights, -1)), -2)` |
| the shared expert at decode rows (1 to 4) | `ling.sharedExpert` (`root.lingSharedExpert`; `sharedTakes`, `sigmoidTable`) | `turboexl3_ling_sgu` (no clamp) or `turboexl3_ling_sguco` (the clamp after the SiLU), then the core's `turboexl3_dense_dec0` ([its catalog](kernels.md)) |
| a same-input pair at decode rows: a KDA layer's z \| a, the shared expert's and the dense FFN's gate \| up | `root.ling.linearPair` (a host's opt-in; each member in its own buffers) | `turboexl3_ling_pair_dec1` (after dense's counter zeroing) where the members' calls take the prepare inside, else `turboexl3_ling_pair_prep` then `turboexl3_ling_pair_dec0` ([kernels](kernels.md)): each member's `linear` bytes |
| a KDA layer's q\|k\|v, f and g at decode rows | `root.ling.kdaProj` (a host's opt-in; each member in its own buffers at its own K split), `root.ling.kdaProjPrepared` over the host's prepared rows | `turboexl3_ling_kdaproj_prep` then `turboexl3_ling_kdaproj_dec0` ([kernels](kernels.md)): each member's `linear` bytes |
| a decode layer's mixer input (rows 1 to 16): the residual add, its RMSNorm and the mixer's EXL3 projections (KDA's qkv, f and g; MLA's q_a and kv_a) | `root.ling.addNormLinears` (a host's opt-in; `addNormPrepare` for a GEMV that reads the prepared rows itself) | `turboexl3_ling_addnorm1`, `2` or `3` (the add, the norm and each member's prepare, one threadgroup a row), then each member's `turboexl3_dense_dec0` (or the chain's GEMV and finish past DEC-ONE's rows) ([kernels](kernels.md)): MLX's add, MLX's rms_norm and each member's `linear`, word for word |
| a MoE layer's router at decode rows (1 to 8) with the read-ahead predictor's scores (the next MoE layer's router on the call's rows) | `root.ling.router.route` (`routerLogits`, `select`; a host's opt-in) | `turboexl3_ling_router_pred` (DeepSeek's sqrt(softplus(z)) + bias), `turboexl3_ling_router_pred_sig` (Ling's sigmoid(z) + bias) or, on the last MoE layer, `turboexl3_ling_router_logits`, then `turboexl3_ling_router_select` ([kernels](kernels.md)): the host's logits, scores, ids and bf16 weights word for word |

The pairs share each member's K split (z \| a at KS 2, the shared gate \| up at KS 8, the dense gate \| up at KS 1);
qkv (KS 1) and MLA's q_a (KS 8) \| kv_a (KS 16) cannot join a member of another K split byte for byte.

The routed shape is new to the library: an expert width of 1280 is ten 128-blocks (the split-K planes, the fin7 grid
and the prefill tiles take it as any multiple of 128), and 512 experts at top-8. Every routed-expert kernel is a composition of the
core's or DeepSeek-V4.1's pieces with one SwiGLU line swapped, so its bytes are the base launch's wherever no value
reaches the limit. The combine is Ling's own kernel, held to MLX's three ops word for word (`combine_test.zig`), and so is the router's pair of launches, held to the host's nine (`router_test.zig`). The tests (`src/models/ling/ling_test.zig`) hold `moeLing` to `ref.moeRowClampedAt` at `silu_out`,
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
