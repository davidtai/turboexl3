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
| a prompt call's routed pairs over an expert streamer's slot bank, the rows gathered from the layer's MoE input | `moeLingRows` (`limit` 0 or the layer's clamp) | `moeLing`'s prefill chain with the row-map prepare `turboexl3_pf_prep_r` in place of `turboexl3_pf_prep`: no gathered copy of x, y in the caller's dtype, `moeLing`'s bytes over `take(x, rows)` |
| a decode wave's routed pairs over an expert streamer's three slot banks | `moeLingBanked` (`moeLingBankedGateUp`, `moeLingBankedDown`; `moeLingBankedSel` on a selection of the texts) | DeepSeek-V4.1's banked decode with `turboexl3_fin7b` (limit 0) or `turboexl3_fin7cob`, or their grouped texts (`fin7bg`, `fin7cobg`) under `Config.banked_group` 4; with `Config.banked_gu8` the gate\|up's prepare inside its launch (`turboexl3_gu8bg`, `turboexl3_gu8bgm3`, the same bytes) |
| every trunk linear and the head | `root.linear` | the core's dense linear; a host's opt-in `Config.dec_one_row_table` runs a one-row call on its trunk shape's `dense.one_row_table` launch, and `Config.dec_variant` DEC-ONE's GEMV over prepared rows on a variant (`dense.dec_variants`: t4, pf, both; `turboexl3_dense_dec0_t4`, `_pf`, `_pft4`), each the same bytes |
| a wide (prompt) routed call's combine: the routed rows times their routing weights, summed over the top-8 | `root.ling.combineRouted` (a joined array), `root.ling.combineParts` (the call's unjoined outputs, JOINLESS) | Ling's own: `turboexl3_ling_combine`, `turboexl3_ling_combine_parts` ([kernels.md](kernels.md)), the words of the host's `sum(multiply(astype(routed, f32), expand_dims(weights, -1)), -2)` |
| the shared expert at decode rows (1 to 4) | `ling.sharedExpert` (`root.lingSharedExpert`; `sharedTakes`, `sigmoidTable`) | `turboexl3_ling_sgu` (no clamp) or `turboexl3_ling_sguco` (the clamp after the SiLU), then the core's `turboexl3_dense_dec0` ([its catalog](kernels.md)) |
| the SwiGLU of the shared expert and of the dense layers' MLP at prompt rows (17 and up) | `root.ling.upSwiglu` (`root.lingUpSwiglu`; `swiglu.takes`; a host's opt-in), after the caller's `linear` of the gate | up's inner product by `linear`'s own launches, then `turboexl3_ling_finish_sgu` (no clamp) or `turboexl3_ling_finish_sguco` (the clamp after the SiLU) in place of its finish ([kernels](kernels.md)): the composed chain's words |
| a same-input pair at decode rows: a KDA layer's z \| a, the shared expert's and the dense FFN's gate \| up | `root.ling.linearPair` (a host's opt-in; each member in its own buffers) | `turboexl3_ling_pair_dec1` (after dense's counter zeroing) where the members' calls take the prepare inside, else `turboexl3_ling_pair_prep` then `turboexl3_ling_pair_dec0` ([kernels](kernels.md)): each member's `linear` bytes |
| a KDA layer's q\|k\|v, f and g at decode rows | `root.ling.kdaProj` (a host's opt-in; each member in its own buffers at its own K split), `root.ling.kdaProjPrepared` over the host's prepared rows, `root.ling.kdaProjB` with the layer's b_proj (bf16) as a fourth member | `turboexl3_ling_kdaproj_prep` then `turboexl3_ling_kdaproj_dec0` ([kernels](kernels.md)): each member's `linear` bytes; with b_proj `turboexl3_ling_kdaproj_dec0b`, b's rows the served one-row matmul's; a host's opt-in (`Config.kda_bfirst`, `kda_bt0` / `kda_bt1`, `dec_variant`) runs b's launch as `turboexl3_ling_kdaproj_dec0bv` (or its t4 and pf twins): b's threadgroups first, q\|k\|v's and f and g's tiles a threadgroup apart, DEC-ONE's GEMV variants, the same bytes |
| a KDA layer's q\|k\|v, f and g at prompt rows | `root.ling.kdaProjPrompt` (a host's opt-in; each member's prepare and W_hat by its `linear`'s launches) | `turboexl3_ling_kdaproj_pf` ([kernels](kernels.md)): pg2's NAX GEMM and finish over the three members in one launch, each member's `linear` bytes |
| a MoE layer's router at decode rows (1 to 8) with the read-ahead predictor's scores (the next MoE layer's router on the call's rows) | `root.ling.router.route` (`routerLogits`, `select`, `tableF32`; a host's opt-in) | `turboexl3_ling_router_pred` (DeepSeek's sqrt(softplus(z)) + bias), `turboexl3_ling_router_pred_sig` (Ling's sigmoid(z) + bias) or, on the last MoE layer, `turboexl3_ling_router_logits`, each over the loader's f32 table (or its `_h` twin over the loaded f16 weight), then `turboexl3_ling_router_select` ([kernels](kernels.md)): the host's logits, scores, ids and bf16 weights word for word |
| an MLA layer's attention at decode rows (1 to 16) over a latent cache (each token's normed 512-wide latent and 64-wide roped shared key) | `mlaLatentDecode` (`mla_latent.zig`) | three of its own: `turboexl3_ling_mla_qabs`, `turboexl3_ling_mla_attn`, `turboexl3_ling_mla_vabs` ([the Ling catalog](kernels.md)); a host's opt-in runs launch 2 on a variant with its words: `turboexl3_ling_mla_attn_rows` (`Config.mla_latent_rows`), `_stage` (`mla_latent_stage`) or `_block` (`mla_latent_block`) |
| an MLA layer's prompt attention over the expanded KV (qk 192, v 128, 64 heads) | `root.ling.mlaFlashPrefill` | `turboexl3_ling_mla_flash` ([kernels.md](kernels.md)): MLX's NAX attention kernel at a tile MLX does not instantiate |

The pairs share each member's K split (z \| a at KS 2, the shared gate \| up at KS 8, the dense gate \| up at KS 1);
qkv (KS 1) and MLA's q_a (KS 8) \| kv_a (KS 16) cannot join a member of another K split byte for byte.

The routed shape is new to the library: an expert width of 1280 is ten 128-blocks (the split-K planes, the fin7 grid
and the prefill tiles take it as any multiple of 128), and 512 experts at top-8. Every routed-expert kernel is a composition of the
core's or DeepSeek-V4.1's pieces with one SwiGLU line swapped, so its bytes are the base launch's wherever no value
reaches the limit. The combine is Ling's own kernel, held to MLX's three ops word for word (`combine_test.zig`), and so is the router's pair of launches, held to the host's nine (`router_test.zig`). The tests (`src/models/ling/ling_test.zig`) hold `moeLing` to `ref.moeRowClampedAt` at `silu_out`,
`moeLingBanked` to `moeLing` byte for byte, `linear` to the host reference at every trunk shape, and each row of a
prompt call's `linear` to the same row of a whole-prompt call wherever both calls run MLX's regular matmul (a
layer-major prompt pass's trunk linears; `TURBOEXL3_LING_ROWINV` lines); `rows_test.zig` holds `moeLingRows` to the outside
plugin's prompt chunk (its take, its casts and `moeLing` at top-1) byte for byte on the loader's views, every chunk of
a prompt layer's call and a padded one (`TURBOEXL3_LING_ROWS` lines), and each variant of its routed chain (the 256-row
tiles, the remainder rule's cut, gate|up's column blocks, the finish as the down GEMM's epilogue, also only for the
experts of at least 64, 80 or 96 rows in the call, the plain launches at bm 128 and 256; `prefill.Options`, off by
default) to the served engine word for word on synthetic calls at every row bucket of an expert, experts on both sides
of each threshold and a lognormal-popularity call (the threshold arms also over a ladder of rows per expert; the fused
finish at its other stage shapes to that stage shape without it), with ten red checks that must each be caught: a
flipped word, four planted kernel faults (a value, a route, two pure reorderings), a rounding change, and for the
threshold each half of the launch writing y marked (exactly its rows differ) and a threshold ignored or off by one
(`TURBOEXL3_LING_ROWS_VARIANT` and `TURBOEXL3_LING_ROWS_RED` lines). The kernels:
[the core's catalog](../../core/kernels.md) (`turboexl3_fin7co`, `turboexl3_pf_midco`, `turboexl3_pf_mid_vco`,
`turboexl3_pf_t128_midco`) and [DeepSeek-V4.1's](../deepseek_v41/kernels.md) (`turboexl3_fin7b`, `turboexl3_fin7cob`,
`turboexl3_fin7bg`, `turboexl3_fin7cobg`).

**The MLA prompt attention.** MLX 0.32.5 runs attention at a query head dim of 192 on its composed path at prompt
widths (`ScaledDotProductAttention::use_fallback`): the scores and the probabilities in device memory at the activation
dtype, about 9 TFLOPS on Ling's 4096-row chunks. `mlaFlashPrefill` is one launch of MLX's own NAX attention kernel
(`attention_nax`) at bq 64, bk 32, bd 192, bv 128 and 4 simdgroups, causal from the bottom right as MLX aligns it, the
keys and values read in place from the cache's views. Its numerics are MLX's fused kernel's (f32 scores, running max and
sum, one output rounding), so it is a reordering against the composed path: held to an f64 reference within tolerance,
and to MLX's fused kernel word for word at the head dims MLX instantiates it for. The plugin's switch decides whether a
prompt takes it; the standard test decides that switch.

`TURBOEXL3_LING_TIME=1` (with `TURBOEXL3_GPU=1`) times the Ling kernels at the model's shapes for the serving plan:
`mlaLatentDecode` at the MTP head's cache of the standard request (19,404 keys, rows 1 to 4) on launch 2 and each of
its variants, every trunk linear, `moeLing` and `moeLingBanked`, and the decode forward's linears chained over the 72 layers
(`TURBOEXL3_LING_TIME` and `TURBOEXL3_LING_CHAIN` lines), and the prompt call over gathered rows against the plugin's
chunk at its call shape (`TURBOEXL3_LING_ROWS_TIME` lines) with each routed-chain variant's arm against the served
engine, there, at uniform popularity and over a ladder of rows per expert (`TURBOEXL3_LING_ROWS_TIME_ARM` lines). It is a diagnostic: it asserts nothing, and its numbers steer
work; they never decide it.

The shared expert (`src/models/ling/shared.zig`) takes the composed chain's four dependent levels at decode rows (the
gate and the up, the host's SwiGLU, the down's prepare, its GEMV; nine with the clamp's MLX ops) in two: one DEC-ONE
launch of the gate and the up, each member in its own buffers (no joined copy), whose output blocks' finishes, taken
in pairs (the gate's 128-block and the up's), run the host's SwiGLU and the down linear's prepare in the pair's second
finish, then the down's DEC-ONE GEMV. Its bytes are the composed chain's: each member's `linear`, mlx-serve's
fusedSwiGLU (the sigmoid from the host's table) or the plugin's swigluClampOut (in f32, MLX's f32 sigmoid from a
table of every 16-bit gate), and `linear(down)` (`src/models/ling/shared_test.zig`, word for word).

The MLA attention in the latent reads a layer's cache once at 1,152 B a token (the expanded cache: 40,960). It computes
in f32 and sums in its own orders, so it is within tolerance of the host's composed latent attention, not byte for
byte (`mla_latent_test.zig`: against an f64 reference), and a verify call's rows have the bits of one-row calls at
their positions (the key groups and sub-chunks count from position 0). Its launch 2 has three variants, each a host's
opt-in and each launch 2's words (the output and the partials, word for word, `mla_latent_test.zig`): the rows of a call
in one threadgroup (`Config.mla_latent_rows` 2 or 4: a key group's latents read once for its rows), the cache's words
staged in f32 in threadgroup memory (`mla_latent_stage`), and more tiles a simdgroup (`mla_latent_block`: 2 × 2 score
tiles over 4 sub-chunks, 4 × 4 output tiles). A product or sum in another order is not a variant: it would change the
words.

No speed verdict comes from this repository: the standard test through the outside plugin decides.
