# API and configuration

The module root `src/root.zig`: its entry points and the one `Config` a consumer sets. Which model uses which entry point: [the docs index](../README.md).

## API

`src/root.zig` is the module root. The expert entry points mirror Sushi's `sushi_exl3` module, so a host switches
engines by changing one import. The entry points of one model live in its folder's `api.zig`
(`src/models/qwen38/api.zig`, `src/models/deepseek_v41/api.zig`); root re-exports each under its own name, so a
consumer imports root alone. An entry point that does not serve a call returns `error.Exl3EngineNotImplemented`
before it launches anything. The caller then runs its own chain. Malformed operands return `error.BadExl3Shape` or
`error.BadExl3Dtype`.

The configuration (see [Configuration](#configuration)):

| name | what it does |
|---|---|
| `Config`, `configure` | the choices of the process, set once before the first call; `configure` refuses a value that the library does not serve |

The contract of a pack (`spec.zig`, with Sushi's names and refusals):

| name | what it does |
|---|---|
| `Spec`, `parseExpertQuant` | read the `expert_quant` block of a pack's config: the rate, the codebook and the window |
| `admitTopK` | refuse a top-k above `REDUCE_MAX_TOPK` (32) with `error.Exl3TopKExceedsReduceBank` |
| `trellisAdmitted` | check a stacked trellis `[E, in/16, out/16, n]` against a bank's geometry and rate |
| `kernels` | `DECODE_ROWS_MAX` (16), `REDUCE_MAX_TOPK` (32), `usesPrefillArm`, `rowsOfShape` |

The routed experts:

| name | what it does |
|---|---|
| `Proj`, `Bank`, `Geometry`, `bankGeometry` | one projection's arrays, a bank's gate, up and down, and the geometry that the shapes give |
| `moe` | the routed experts of one MoE layer: decode rows through the split-K GEMV chain, prefill rows through the fused GEMM chain; its argument `verify_rows` is Sushi's and changes nothing here |
| `moeWithShared` | `moe`, then an MLX add of a shared-expert output that the caller has already gated |
| `moeClamped` | `moe` with DeepSeek-V4's (and GLM's) clamped SwiGLU, `swiglu_limit` 1 to 128: the gate clamped from above and up to [−limit, limit] before SwiGLU; decode rows on fin7's clamped variant, prefill rows on the clamped mid |
| `moeClampedBanked` (`moeClampedBankedGateUp`, `moeClampedBankedDown`) | DeepSeek-V4.1 (`models/deepseek_v41/api.zig`): `moeClamped`'s decode launches over an expert streamer's three slot banks: up to 64 routed pairs, each a row-slot whose id is packed (bank << 24 \| slot row), each pair's output at weight 1 with `moeClamped`'s bytes on its bank at top-1; the gate\|up and the down as two calls (the gate\|up half also reads the down's suh: it prepares the down's input) |
| `moeSwigluRows` | `moeSwiglu`'s prefill chain at top-1 over gathered rows: row-slot p reads x's row `rows[p]` (x [n, hidden], the gather in the chain's prepare, `prefill.text_prep_rows`) through slot `inds[p]` at weight `scores[p]`, y [S, hidden] in a chosen dtype, `limit` 0 to 128 at either clamp; row p is `moeSwiglu` over `take(x, rows)` at top-1, its y cast to that dtype, bit for bit (an expert streamer's prompt calls without their gather and casts) |
| `moeSwigluWith`, `moeSwigluRowsWith` | `moeSwiglu` and `moeSwigluRows` with the call's own choices (`prefill.Call`; `.{}` is the plain entry): `.{ .dfin = false }` keeps a prefill call off the fused finish its geometry's engine was built with (`pf_dfin`), on the served route, down launches and finish of that engine, the same bytes. A call keeps the fused finish only where its engine has it |

Ling-3.1-flash's routed experts, shared expert, trunk pairs, combine and router (`models/ling/api.zig`):

| name | what it does |
|---|---|
| `moeLing` | `moe` with Ling-3.1's SwiGLU: `limit` 0 none (`moe`'s bytes), 1 to 128 clamped after the SiLU, min(silu(g), L) · clamp(u, −L, L); decode rows on fin7's Ling variant, prefill rows on the Ling mid |
| `moeLingBanked` (`moeLingBankedGateUp`, `moeLingBankedDown`; `moeLingBankedSel` on a `BankedSel`'s texts) | `moeClampedBanked`'s launches with Ling-3.1's SwiGLU (`moeLing`'s `limit`, 0 included): each pair's bytes `moeLing`'s on its bank at top-1 with score 1, on every selection of the texts |
| `moeLingRows` | a streamer's prompt call over gathered rows (`moeSwigluRows` with Ling-3.1's SwiGLU, `moeLing`'s `limit`): row-slot p reads x's row `rows[p]` in the chain's prepare, y [S, hidden] in a chosen dtype; row p is `moeLing` over `take(x, rows)` at top-1, its y cast to that dtype, bit for bit (turboexl3_ling's prompt chunk without its take and casts, MLX_SERVE_LING_ROWS_PATH=1) |
| `moeLingWith`, `moeLingRowsWith` | `moeLing` and `moeLingRows` with the call's own choices (`prefill.Call`, through `moeSwigluWith` and `moeSwigluRowsWith`): turboexl3_ling's PF_DFIN whole-call switch decides once a wide prompt call, from its rows alone, whether its chunks keep the fused finish (`.{}`) or run the served ending (`.{ .dfin = false }`), the same bytes either way |
| `ling.linearPair` | two EXL3 projections of one input (z \| a, the shared or the dense gate \| up), each in its own buffers, in one DEC-ONE GEMV launch at decode rows (1 to 16): y [rows, out_a + out_b], each member's columns its `linear` bytes; error.Exl3EngineNotImplemented where the members' K splits, forms or tiles differ |
| `lingSharedExpert` (`lingSharedTakes`, `lingSigmoidTable`) | the shared expert at decode rows 1 to 4: the gate and the up in one DEC-ONE launch (each in its own buffers) whose pairs of output blocks run the host's SwiGLU (`limit` 0: mlx-serve's fusedSwiGLU from the host's table; 1 to 128: the clamp after the SiLU in f32) and the down's prepare, then the down's DEC-ONE GEMV: the composed chain's bytes; error.Exl3EngineNotImplemented where `lingSharedTakes` does not hold |
| `ling.combineRouted`, `ling.combineParts` | a wide routed call's combine in one launch (`models/ling/combine.zig`): routed f32 [n, K, D] (or the call's unjoined outputs, at most 24, and each pair's (source, row): JOINLESS) times weights f32 [n, K], summed over K <= 8 -> f32 [n, D], the words of MLX's `sum(multiply(astype(routed, f32), expand_dims(weights, -1)), -2)` |
| `ling.mlaFlashPrefillRope1` (`mla_flash.dimsOfRope1`) | the MLA prompt attention over the rope-once cache in one causal flash launch (`models/ling/mla_flash.zig`, `turboexl3_ling_mla_flash_r1`): q T [B, H, qL, 192], k T [B, Hk, kL, 128] (each head's nope keys), kr T [B, 1, kL, 64] (the token's one roped key) and v T [B, Hk, kL, 128], each read in place at its own strides -> o T [B, H, qL, 128], the words of `ling.mlaFlashPrefill` over the expanded key (k ++ kr for every head); error.Exl3EngineNotImplemented unless `Config.mla_rope_once` and the NAX gate hold |
| `ling.router.route` (`routerLogits`, `select`; `tableF32`) | the MoE router at decode rows 1 to 8 in two launches (`models/ling/router.zig`): x bf16 [rows, hidden] over the router's weight as [512, hidden] row-major (the host loader's column-major f32 table transposed back, or the loaded f16 weight, widened then + 0) -> the logits (each row the host's one-row gemv words) and, with a `Predictor` (the next MoE layer's bf16 router and bias), its read-ahead scores f32 [rows, 512] (MLX's gemv at one row, gemv_wide at 2 to 8, then sqrt(softplus(z)) + bias or sigmoid(z) + bias); then the host's fused router (`sigmoid_bias_grouped`, its text) -> ids u32 [rows, 8] and weights bf16 [rows, 8]: the host's words. `tableF32` rebuilds the loader's f32 table from the f16 weight's view (astype, then + 0) |

Qwen3.8's shared expert and router, at 1 to 4 decode rows (`models/qwen38/api.zig`):

| name | what it does |
|---|---|
| `moeFusedShared` | a gated shared expert inside the decode launches of the routed experts |
| `moeFusedSharedRouted` | the same, with the router's top-k and softmax folded into the prepare launch |

The dense projections:

| name | what it does |
|---|---|
| `linear` | one dense EXL3 projection. 1 to 4 rows: DEC-ONE. 5 to 16 rows: the multi-row GEMV chain of three launches, which decodes each weight once per row chunk: 5 rows in one chunk, 6 to 8 rows in one chunk on the lean GEMV, and 9 to 16 rows in two chunks. 17 to 32 rows: the register-path GEMM. More rows: dequantize, then MLX's matmul |
| `linearSplit` | `linear` with the inner product and the finish of prefill rows on a second MLX stream; the same bytes |
| `linearInnerSplit`, `linearPrepared`, `linearFinish`, `linearPrep` | `linear` in pieces, so the host can run its fused norm-gate of the GDN layers between them (users: Qwen3.8's GDN layers) |
| `gdnOutPrep` | Qwen3.8 (`models/qwen38/api.zig`): the GDN layer's tail at prefill rows, z's finish, the gated norm and out_proj's prepare in one launch |
| `linearPiecesBegin`, `linearPieceInner`, `linearPiecePrepared`, `linearPiecesFinish` | `linear` of one call cut into 2 to 4 row pieces, so the host can overlap them with its own launches (users: Qwen3.8's GDN layers, after the fused tail) |
| `linearGrouped` | DeepSeek-V4.1 (`models/deepseek_v41/api.zig`): a grouped (block-diagonal) linear, its `wo_a`: x `[..., G, in]` through G stacked slices. 1 to 16 rows: three launches whatever G. More rows: `linear` of each slice. Each slice has `linear`'s values |
| `linearBias` | `linear`, then an MLX add of a bias `[out]` in x's dtype (users: DeepSeek-V4.1's vision tower and aligner) |
| `sharedSwigluTakes`, `sharedSwiglu` | DeepSeek-V4.1 (`models/deepseek_v41/api.zig`, F3): its shared expert at decode rows, the clamped SwiGLU and the down's prepare in one launch, the bytes of three `linear`s and the host's SwiGLU |

Qwen3.8's attention projections (`models/qwen38/api.zig`):

| name | what it does |
|---|---|
| `attnIn` | ATTN-IN: q, k and v with QK-norm and partial RoPE in the finish, at prefill rows |
| `attnPrep4`, `linearPreparedIdx` | ATTN-IN's prepare with the QSA indexer projection as a fourth member, and the indexer projection from its prepared copy |
| `linearGated` | ATTN-OUT: `o_proj` over the sigmoid-gated output of the attention core, at prefill rows |
| `decode_merge` | at decode rows: `decodeMergeGated`, `decodeMergeQmergeGated` (with the QSA merge), `decodeMergeQPlanes`, `decodeMergeIndexer` |

The resident and un-rotated trunk:

| name | what it does |
|---|---|
| `dense.residentArm` | keep the W_hat of each admitted trellis after its first call on the matmul path (more than 32 rows by default); the same bytes as a dequantize per call |
| `dense.residentRelease`, `residentPending`, `residentStats`, `residentKeeps` | drop the kept arrays, the bytes admitted but not kept yet, the counts, and whether one trellis is kept |
| `trunkUnrot`, `UnrotStats` | at load, put W_std in the activation dtype (`bf16` or `f16`) in place of each admitted trellis's W_hat; opt-in, with a different rounding |
| `dense.residentUnrot`, `residentStd`, `residentJoined` | whether a trellis has a W_std, its W_std, and its joined W_std (up to 4 projections that read one x, joined along the outputs) |
| `attnInStd`, `linearGatedStd` | ATTN-IN and ATTN-OUT over the un-rotated trunk |

The namespaces are `config`, `format` (the reference), `spec`, `lane` (the codeword readers), `decode`, `dense`,
`pgemm`, `prefill`, `sources`, `unfused` and `ref` (the core); `qwen38` (its entries), `attn_io`, `decode_merge`,
`gdn_out` and `shared_heads` (Qwen3.8); `deepseek_v41` (its entries), `grouped`, `banked` and `f3` (DeepSeek-V4.1); `ling` (Ling-3.1-flash);
and `ricursive` (Ripple). The doc comment of each function states its
shapes, its dtypes, the chain whose bytes it gives and when it refuses.

## Configuration

turboexl3 reads no environment. A consumer sets its choices once, before the first call, with
`turboexl3.configure(.{ ... })`. The argument is one `Config` (`src/core/config.zig`). The default of each field is the
served configuration.

`configure` refuses a value that the library does not serve with `error.InvalidTurboexl3Config`, and then it changes
nothing. A kernel cache keeps what it built. So a change after the first launch reaches only the geometries that later
calls build.

| field | default | what it controls | output bits |
|---|---|---|---|
| `decode_form` | 9 | the form of the decode chain of the routed experts: 3, 5, 6, 7, 8 or 9 | unchanged, tested: forms 5–9 give the bytes of form 3 |
| `shared_spt` | 4 | the fused shared expert: K slices per gate and up GEMV threadgroup, of its 20: 1, 2, 4, 5, 10 or 20 | unchanged, tested |
| `fused_fin_fold` | true | the fused shared expert from two rows: the finishes of the gate and up planes (`fin7s`) run in the gate and up launch (`turboexl3_gu3sf`), in the threadgroups that complete each block; false: `gu3s` then `fin7s` | unchanged, tested |
| `pf_alay` | 2 | prefill: the A layout of the GEMMs: 0 natural; 1 fragment order per 32 elements; 2 fragment order per 16 elements | unchanged, tested |
| `pf_t128` | true | prefill: each expert's full runs of 128 rows in 128-row tiles | unchanged, tested |
| `pf_dfin` | false | prefill, every top-1 routed call (an expert streamer's prompt calls): the finish as the down GEMM's epilogue, one down launch writing y (over the 128-row tiles' two tables). A call can decline it (`prefill.Call`, the `…With` entries): it then runs the served route, down launches and finish of the same engine. A geometry built with it prints `[turboexl3] prefill variant engaged: …` | unchanged, tested |
| `prefill` | true | false: `moe` refuses prefill rows with `error.Exl3EngineNotImplemented` | not applicable |
| `nax` | true | whether the device passes the NAX gate; the consumer tells. False turns off every kernel that uses the NAX: `moe` refuses prefill rows, and a dense call of 17–32 rows takes dequantize + MLX's matmul | can differ: those dense rows sum in another order |
| `dec_one` | true | DEC-ONE for dense projections at decode rows; false: the three-launch chain | unchanged, tested |
| `dec_one_row_table` | false | a one-row dense decode call on its trunk shape's entry of `dense.one_row_table` (Ling-3.1-flash's shapes on the M5 Max: output tiles per GEMV threadgroup, and DEC-ONE's prepare inside the launch or in `prep_sc`'s); a shape the table does not name keeps the rest of the configuration's. The first one-row call on each entry prints `[turboexl3] one-row table engaged: in=… out=… n=… BT=… prepare=…` | unchanged, tested |
| `router_rows` | false | Ling-3.1-flash's router (`ling.routerLogits`, and `route`) at a call of 2 rows or more over the f16 weight on the rows launch (`turboexl3_ling_router_logits_hr`, `_pred_hr`, `_pred_sig_hr`): each router threadgroup computes its 16 logits for a group of up to 4 rows, so each weight word is read once a group, not once a row; one row and the f32 table keep the served launch. Its first call prints `[turboexl3] router rows engaged: …` | unchanged, tested |
| `dense_lean` | true | the row chunks of the dense GEMV chain: up to 8 rows, and a chunk of 6 to 8 rows on the lean GEMV (its codeword loop by input element, so fewer live registers); false: up to 5 rows, all on the plain GEMV | unchanged, tested |
| `banked_variant_gu`, `banked_variant_dn` | 0 | the grouped gate and up GEMV and the grouped down GEMV of the routed decode on an entry of `banked.group_variants` (1 to 11; any but 0 needs `banked_group` 4): another schedule of the same loads, decodes and sums. The first call of each GEMV on a variant prints `[turboexl3] banked variant engaged: the grouped gate_up GEMV on variant … (… texts)` (`down` for the down GEMV) | unchanged, tested |
| `mla_latent_stage` | false | the same launch with each sub-chunk's latents and rope keys staged in f32 in threadgroup memory (`turboexl3_ling_mla_attn_stage`), at the calls `mla_latent_rows` does not take; prints `… engaged: stage …` | unchanged, tested |
| `mla_latent_nax` | false | launch 2 on the matrix units (`turboexl3_ling_mla_attn_nax`, `mla_latent.nax_served`): the query rounded to the cache's dtype, the scores and the weighted sums on MLX steel's NAX tiles into f32, ahead of `mla_latent_stage` at every call with a bf16 or f16 cache (an f32 cache keeps the rest); not with `mla_latent_rows` or `mla_latent_block`; prints `… engaged: nax_… (launch 2 on the matrix units, …)` | not the same words: within `naxBound` of the f64 reference (`mla_latent_test.zig`); a host serves it where the words move no output (the plugin: the MTP head's drafts, which the exact verify checks) |
| `mla_rope_once` | false | Ling-3.1-flash's MLA prompt attention over the rope-once cache (`ling.mlaFlashPrefillRope1`, `turboexl3_ling_mla_flash_r1`): each head's nope keys and the token's one roped key read in place, the expanded key never built; off refuses the calls. The first call prints `[turboexl3] Ling MLA prompt attention over the rope-once cache engaged: …` | unchanged, tested: `ling.mlaFlashPrefill`'s words over the expanded key |
| `slot_stats` | false | a diagnostic: it reads the routing of each decode call back to the host (a sync per call). Every 480 calls it prints how often the row-slots of a call repeat an expert. The router fold refuses while it is on | unchanged |
| `pf_capture_dir` | null | a diagnostic: a directory; the routing of the first 512 prefill calls goes there (a sync per call) | unchanged |

The "output bits" column has these values:

- **unchanged, tested**: the field moves work only, and a GPU test compares the bits of the settings.
- **unchanged**: the field moves work only, as the source states, but no GPU test compares the bits.
- **can differ**: the setting can change the bits.

Every other launch shape is a constant of the file that reads it (`root.zig`, `dense.zig`, `decode.zig`). This
includes the prefill tiles, the GEMV tiles and K splits, the deferred MUL1 map and the dense paths. The tests also
build other values of these constants, as variants.

The plugin sets one field: `nax`, from turboexl3_kit's test of the NAX gate (`kit/nax.zig`).
