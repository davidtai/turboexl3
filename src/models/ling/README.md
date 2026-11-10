# Ling-3.1-flash

The code only Ling-3.1-flash uses, on the core's routed chain, DeepSeek-V4.1's banked decode and the core's dense linear:

| file | what it holds |
|---|---|
| `pair.zig` | `linearPair` (a same-input pair, each member in its own buffers, in one DEC-ONE launch: the prepare inside, or one prepare launch for both members then the GEMV over their rows), each member's `linear` bytes |
| `kdaproj.zig` | `kdaProj` (a KDA layer's q|k|v, f and g in one DEC-ONE GEMV launch after one prepare launch, each member in its own buffers at its own K split), `kdaProjPrepared` (the GEMV alone over a host's prepared rows), `ncntFor`; each member's `linear` bytes; `kdaProjB` adds the layer's b_proj (bf16) as a fourth member of the GEMV launch, MLX's gemv over each row (`kernels/kdaproj/bmember_1.metal`, `bmember_2.metal`) |
| `kdaproj_pf.zig` | `kdaProjPrompt` (a KDA layer's q|k|v, f and g at prompt rows: each member's prepare and W_hat by its `linear`'s launches, then the three products and finishes in one launch, pg2's fused NAX GEMM and finish over the members), `kdaProjPromptW` (the same over W_hats the caller decoded once: each member's prepare, then the one launch over the given W_hats), `takes`, `tileFor`; each member's `linear` bytes; L6 (the outside plugin's KDA_GATE_EPI): `kdaProjPromptGate` and `kdaProjPromptWGate` (the same launch with member 1's output its bounded gate, g f32, made in the store loop: G0 taking exp(A_log) for each element, G1 reading it from `gateEa`'s array), `gateTakes`, `gateEa`; y_0 and y_2 `kdaProjPrompt`'s words, g the served one-launch gate's |
| `kdanorm.zig` | `kdaNormGatePrep` (L5: a KDA layer's gated norm at prompt rows, the host's fused norm-gate at its table with the sigmoid gate, written straight into o_proj's prepared rows in one launch: dense's prepare with the norm-gate's statements in its load's place; o_proj's matmul and finish then `Dense.piecePrepared` and `Dense.finishOf`), `takes`, `shapeOk`; xp the prepare's words of the host's output |
| `kernels/kdaproj/` | the KDA projections' pieces: the prepare's member choice, DEC-ONE's head over three members (each member's widths and K split), the member's prepared rows, trellis, svh, output, partial sums and counters; b_proj's member (`bmember_1`, `bmember_2`); the prompt-row launch's member choice, its column and the macro's end around pg2's pieces (`pf_*.metal`) |
| `kernels/kdanorm/` | the norm-gate's statements at the prepare's lanes (`gate.metal`, under its `//! copied_from:` line), each lane's four values as dense's prepare takes them |
| `kernels/pair/` | the pair's pieces: the prepare's member choice (its suh, its rows), the GEMVs' member choice (its prepared rows or its suh) and the member's trellis, output tiles and svh |
| `api.zig` | `moeLing` (the routed experts with Ling's SwiGLU: none, or clamped after the SiLU) and `moeLingBanked` with its halves (the same over an expert streamer's three slot banks); the combine's entries |
| `ling_test.zig` | the texts' compositions, the reference clamp, the refusals and the trunk shapes (CPU); `moeLing` against `moe` and the reference, `moeLingBanked` against `moeLing` byte for byte, `linear` at every trunk shape, a one-row call on the one-row table's launch (`Config.dec_one_row_table`, whose per-shape choice the CPU tests pin) against one row's own and the three-launch chain byte for byte, the pair against each member's `linear` byte for byte in both of its forms, and KDA_EPILOGUE's o_proj from prep_sc's rows (`linearFromPrepared`) against `linear` word for word at every trunk shape (GPU) |
| `shared.zig` | `sharedExpert` (the shared expert at decode rows: the gate and the up in one DEC-ONE launch, each in its own buffers, whose pairs of output blocks run the host's SwiGLU and the down's prepare, then the down's DEC-ONE GEMV), `sharedTakes`, `sigmoidTable` (the host's tables) |
| `shared_test.zig` | the joint texts' compositions and pinned statements (CPU); `sharedExpert` against the composed chain word for word at rows 1 to 4, limits 0 and 10 (GPU) |
| `kernels/shared/` | the pieces the shared expert's texts add to dense's DEC-ONE: the member choice and its tile address, the member's store, the pair's arrival, the two SwiGLUs and the down's prepare; `kernels/test/pins/` what the CPU test pins |
| `swiglu.zig` | `upSwiglu` (the prefill SwiGLU epilogue: up's inner product by `linear`'s own launches at prompt rows, then its finish running the host's SwiGLU of the gate's value and its own), `upSwigluW` (the same over a W_hat the caller decoded once, at the matmul path's rows), `takes` (where it serves), the two texts (dense's `finish_v`, the store loop's opening and close in `kernels/swiglu/`, the shared expert's SwiGLU statements) |
| `combine.zig` | `combineRouted` and `combineParts` (a wide routed call's combine in one launch, over a joined array or the call's unjoined outputs), the texts' CPU test |
| `combine_test.zig` | both against MLX's composed combine word for word, the refusals (GPU) |
| `kernels/combine/` | the combine's pieces: the contract pragma, the source choice, the fold's head, each text's row read, the fold and store |
| `router.zig` | `route`, `routerLogits`, `select` and `tableF32` (a MoE layer's router at decode rows 1 to 8 in two launches: MLX's one-row gemv over the host loader's column-major table for the logits, or over the loaded f16 weight widened then + 0, beside its gemv / gemv_wide for the read-ahead predictor's scores, then the host's fused router text over the logits with its weights in bf16; the loader's f32 table rebuilt from the f16 weight); `routerLogitsWith`, `rowsRuns` (the rows launch under `Config.router_rows`: each router threadgroup 16 logits of a group of up to 4 rows, the f16 weight read once for them); the texts' and the plan's CPU tests |
| `router_test.zig` | the logits, both predictor scores, the ids and the weights against the host's served path word for word at rows 1 to 8 (every operand built as the host builds it: the loaded f16 weight, the loader's column-major table, the MoE input, denseRows), a weight holding every finite f16 word, the rebuilt table at the stock widths, with `TURBOEXL3_LING_PACK=<pack dir>` every router weight of the pack as stored (the MoE layers' and the NextN layer's), the refusals; the rows launch against the served launch word for word at rows 1 to 8, with red checks (GPU); the rows launch's TIME arm (`timeRouterRows`) |
| `bproj_test.zig` | `kdaProjB`'s b_proj member against the served b_proj (the one-row matmul, the exact verify's denseRows) and its three members against `kdaProj`'s word for word at rows 1 to 8, operands as the loader keeps them; the refusals (GPU); the member's text and gemv statements (CPU) |
| `kdaproj_pf_test.zig` | `kdaProjPrompt`'s members against each `root.linear` word for word at prompt rows 33 to 16384, operands as the loader keeps them, the refusal at 32 rows (GPU); `kdaProjPromptW` over each member's W_hat, decoded once, against `kdaProjPrompt` word for word at rows 33, 1965 and 4096, its declines and refusal, and a swapped W_hat caught (GPU); its timing against the three `linear` calls over a short tile sweep (TURBOEXL3_LING_TIME) |
| `kdaproj_gate_test.zig` | `kdaProjPromptGate`'s y_0 and y_2 against `kdaProjPrompt`'s and its g against the served one-launch gate (its test-only copy, kernels/test/copy) over `kdaProjPrompt`'s y_1, word for word, G0 and G1, rows 33 to 16384, and over given W_hats; the declines, the gate's refusals and five planted faults (GPU); its timing against the served chain and the gate alone (TURBOEXL3_LING_TIME) |
| `kdanorm_test.zig` | `kdaNormGatePrep`'s xp against `Dense.prepare` of the served norm-gate's output (its test-only copy, kernels/test/copy/gdn_normgate.metal) and o_proj's y over it against `root.linear`, word for word at prompt rows 33 to 16384; its refusals, at the GEMV's and the register path's rows with each row's path asserted; four planted faults caught (GPU); its timing against the served pair and o_proj's chain (TURBOEXL3_LING_TIME) |
| `rows_test.zig` | `moeLingRows` against the outside plugin's prompt chunk (take, bf16, f32, `moeLing` at top-1) byte for byte on the loader's views; the routed chain's variants (`variants`: the fused finish over the 128-row tiles' two tables, at the served stage shape and three others) against the served engine word for word on synthetic calls at every row bucket, both sides of 64, 80 and 96 rows and a lognormal-popularity call, and five red checks caught: a flipped word, three planted kernel faults (`kernels/test/fault/`: a value, two pure reorderings) and a rounding change; PF_DFIN's whole-call switch (`prefill.Call`): every synthetic call at both choices on the fused finish's engine, the served words and the ending each call ran (`prefill.Steps`), the plugin's rule at 1,120, 5,119, 5,120 and 16,384 rows and call_min 0, and three red checks only the engagement check sees: a choice ignored and the rule off by one either way (GPU); the rows path's and each variant's timing (TURBOEXL3_LING_TIME) |
| `swiglu_test.zig` | `upSwiglu` against the composed chain (the gate's and the up's `linear`, the host's SwiGLU as MLX's ops) word for word at the shared expert's and the dense layers' shapes, rows 17 to 16384, limits 0 and 10, its declines and refusals (GPU); `upSwigluW` over one decoded W_hat against `upSwiglu` word for word at the matmul path's rows, its declines and refusals (GPU); the texts' composition and pinned statements (CPU); its timing (TURBOEXL3_LING_TIME) |
| `kernels/router/` | the router's pieces: launch 1's head, MLX's gemv statements cut at the table's read (an f32 word, or an f16 word widened then + 0), the predictor's branch (gemv at one row, gemv_wide at 2 to 8), the two scores (headers) and the host's fused router text (`select.metal`); the rows launch's head, gemv opening and close and predictor opening (`head_rows`, `logits_rows_1`, `logits_rows_2`, `pred_open_rows`) |
| `mla_latent.zig` | `decodeLatent` (`mlaLatentDecode`): the MLA attention at decode rows in the latent, three launches (the query absorbed into the latent, the attention's partials over the latent cache, the merge and the value half); launch 2's three variants, each launch 2's words and a host's opt-in in `Config` (`Attn`: 2 or 4 rows a threadgroup, the cache staged in f32, more tiles a simdgroup), `decodeLatentWith` and `decodeLatentParts` (a chosen text, launch 2's partials kept); its texts and plan (CPU tests) |
| `mla_latent_test.zig` | the host's f64 reference, absorbed against expanded (CPU); `decodeLatent` against it within tolerance, a call's rows against one-row calls and strided cache views against contiguous ones, byte for byte; launch 2's variants against launch 2, the output and the partials word for word on the plugin's operands, with red checks (GPU) |
| `kernels/mla_latent/` | the three launches' texts and their header, launch 2 in four pieces (`attn_1` to `attn_4`: its scores, the online softmax update the variants share, the weights times the latents, the outputs) and its variants' pieces (`rows_*`, `stage_*`, `block_*`); `kernels/test/pins/` what the CPU tests pin |
| `mla_flash.zig` | `mlaFlashPrefill` (`api.zig` exports it): the MLA prompt attention as one causal flash launch, MLX's NAX attention kernel at Ling's head dims (qk 192, v 128); `mlaFlashPrefillRope1` (`dimsOfRope1`; `rope1On` on a test's kernel): the same launch over the rope-once cache (each head's nope keys and the token's one roped key read in place, `Config.mla_rope_once`), the same words |
| `mla_flash_test.zig` | the flash kernel's text (CPU) and its standalone translation units (TURBOEXL3_METAL_DUMP=1); against MLX's fused SDPA word for word where MLX instantiates one, and within tolerance of an f64 reference at Ling's heads (GPU); the rope-once text, its fault text and its refusals (CPU); the rope-once launch against the base over the expanded key, every word, with two red checks (a planted fault, a changed roped key; GPU) |
| `kernels/mla_flash/` | the flash kernel's pieces: MLX's attention_nax body less its unused arms (`body_1` to `body_5`, cut at the key reads: `body_kofs`, `body_kload`, `body_kstep`), its row ops, the kernel attribute's define and undefine; the rope-once variant's key reads (`r1_setup`, `r1_kofs`, `r1_kload`, `r1_kstep`); `kernels/test/` its pin files, dump data and the rope-once fault (`fault/mla_flash_r1_kofs_nope`) |

Ling's own Metal is the pairs', the shared expert's, the routed combine's and the router's pieces (`kernels/pair/`,
`kernels/shared/`, `kernels/combine/`, `kernels/router/`),
the MLA attention in the latent's three launches (`kernels/mla_latent/`)
and the MLA prompt attention's flash kernel (`kernels/mla_flash/`;
`kernels/test/pins/` what the CPU tests pin); the routed experts' clamp pieces are the core's
(`kernels/decode/fin7_swiglu_clamp_out.metal`, `kernels/prefill/mid_swiglu_clamp_out.metal`). docs/models/ling/
describes the model's needs and its kernels.

## L5 (KDA_NORM_PREP): the design record

Recorded 2026-11-10 on branch `ling-l5-normprep`, which starts from ling-strip2 at c2783f5. L5 is the work item, and
KDA_NORM_PREP is the name of the plugin option planned to serve it. The CPU and GPU tests are written. The GPU byte test
has not run yet, and the outside plugin turboexl3_ling does not call the launch yet. Terms such as prepare, H128, suh,
W_hat, W_std, NAX, register path, simdgroup and un-rotated trunk are in `docs/core/glossary.md`.

Paths: `core/...` and `models/...` are under this repository's `src/`, `kit/` and `docs/` are at its root, and
`kernels/...` is beside this README. mlx-serve, MLX and mlx-c paths are at the named commits. The plugin is
CommunityServe/plugins `plugins/turboexl3_ling/`, branch turboexl3_ling-m12 at 1a16e40.

A KDA layer is Ling's linear-attention layer. At its prompt rows, the plugin runs mlx-serve's fused norm-gate, from its
copy of mlx-serve's `decode_fused.zig` (`gdnNormGateFusedTab`); this record calls it the host's norm-gate. Then o_proj's
`linear` prepares the norm-gate's output o for its matmul. `kdanorm.kdaNormGatePrep` does both in one launch,
`turboexl3_ling_kdanorm_prep`. It writes xp, o_proj's prepared rows, so o never reaches device memory. That saves one
launch, and the write and read of o, per KDA layer and span (a span is one of the prompt's row ranges).

- The norm-gate reads y [B, S, HV, DV], the KDA recurrence's output: HV heads of DV = 128 values (64 heads on
  Ling-3.1-flash). It also reads z (the gate's rows), norm_w [DV], an f32 eps, and sigtab (MLX's sigmoid of every
  16-bit pattern). For each row and head, inv = rsqrt(sum(y²) / DV + eps), and each output is
  o = (norm_w · T(y · inv)) · sigmoid(z), in y's dtype T. o is [rows, HV·DV], bf16 [rows, 8192] here.
- Its text is mlx-serve bc6c5f2c `src/kernels/gdn/gdn_normgate.metal`, here with TAB 1 (the sigmoid read from sigtab)
  and SWISH 0 (the sigmoid gate).
- The prepare computes `xp = f16(H128(f16(o) · suh))` per 128-value block. suh is o_proj's f16 vector over its input
  channels, and H128 is the 128-point Hadamard transform, run as a butterfly. The launch writes xp in natural order
  (LANE_ORDER 0), the order the matmul reads.

Both kernels give one simdgroup to each row and 128-value block, and at DV 128 one block is one head. The launch sums the
head's squares with the host's statements, in the host's order: each lane's four consecutive channels, then `simd_sum`.
It makes each value the prepare loads (channel L + 32 j) with the host's per-element statements, then runs the
prepare's statements on it.

In this record, words are an output's bit patterns, which the tests compare. The CPU tests in `kdanorm.zig` pin the
launch's statements against a test-only copy of the host's text, and the copy against mlx-serve's file by its sha256.

The sections below give the conditions for equal words, the refusal rows, the shared compile options, and where
contraction cannot change a word.

### The conditions, and who checks each

o_proj's output is `linear`'s words only where the caller runs o_proj's matmul and finish over xp. These are
`Dense.piecePrepared`, MLX's matmul of xp against a W_hat that `Dense.dequantize` decoded, and then `Dense.finishOf`.
`linear` runs that chain on only part of its matmul path.

The planned caller is the plugin's LAYER_WHAT. Under LAYER_MAJOR the plugin runs each layer over every span of a prompt
before the next layer. LAYER_WHAT decodes each dense linear's W_hat once a layer and reuses it for every span
(`src/ling_layer_what.zig`). Its `matmulPath` (lines 228-232) decides where it serves a call. KDA_NORM_PREP must call
the launch only where `matmulPath` returns true for o_proj.

| condition | checked by | why |
|---|---|---|
| the shapes and dtypes of y, z, norm_w, eps, sigtab and suh; o_proj's input width HV·DV | `kdanorm.takes` (`shapeOk`) | the launch's geometry and the host's own rules |
| `Dense.pathFor(rows) == .matmul`: more than 16 rows and, with `Dense.fused_prefill` (a tests-only fused GEMM) off, more than `small_rows` (32 on NAX, else 0) | `kdanorm.takes` | on the other paths `linear` runs other kernels (next section) |
| no un-rotated trunk at more than 32 rows | the caller (`matmulPath`) | `takes` reads o_proj's `Dense`, never its trellis |

Under the un-rotated trunk, a host opt-in, dense's resident table holds o_proj's W_std in place of its W_hat. Past 32
rows (`unrot_min_rows`), `linear` then runs one MLX matmul of x against W_std (`core/dense.zig:1068`). That arm has no
prepare and no finish and rounds at other points, yet `Dense.pathFor` still reports `.matmul` there. `Dense.dequantize`
still returns a W_hat, so the chain over xp would give other words than `linear`.

pg2 is the host's switch that runs a dense linear's prefill matmul and finish as one launch (`core/pgemm.zig`). It
keeps the chain's words (`core/pgemm.zig:10-12`), and its table has no entry for o_proj's shape, 8192 x 4096
(`core/pgemm.zig:87-100`). So pg2 is no condition for equal words. LAYER_WHAT declines every call while pg2 is on
(`matmulPath`, line 230), so the planned caller runs with pg2 off.

### The refusal rows

The API's next step, `piecePrepared`, multiplies xp by a W_hat, and `linear` makes that step only on its matmul path.
At 16 rows or fewer, `linear` runs the GEMV (`decode.Kernels.max_rows`, `core/decode.zig:86`). The GEMV prepares its
rows in its own lane order (LANE_ORDER 1) and decodes the trellis itself. On NAX (the M5 GPU's matrix units,
`Config.nax`), 17 to 32 rows run the register path (`small_rows`, `core/dense.zig:445`), whose GEMM also decodes the
trellis. On those rows a chain over xp and a W_hat is not `linear`'s chain, so `takes` refuses them. Off NAX,
`small_rows` is 0, so 32 rows take the matmul path and `takes` returns true. The GPU test asserts `.gemv` at 16 rows,
`.small` at 32 and `.matmul` at 33, and that `takes` refuses 16 and 32 rows. These assertions hold only on NAX, and
the served Ling path runs on NAX. So the whole GPU test, byte checks included, skips where `Config.nax` is false.

### The shared compile options

MLX's `fast::metal_kernel` JIT compiles each of the four kernels at its first use, with MLX's default compile options.

| kernel | made in service by | made in the tests by |
|---|---|---|
| the host's norm-gate | the plugin's patched copy of mlx-serve's `src/sdk/blocks/gdn/decode_fused.zig` (lines 243 and 260 at bc6c5f2c): the embedded text, mlx-c's `mlx_fast_metal_kernel_new` with an empty header | not made; the tests use its copy |
| its test-only copy | not made | `kdanorm_test.zig` through the test consumer's `newKernel` |
| o_proj's prepare | turboexl3's `newKernel` through the host table | the test consumer's `newKernel` |
| this launch | turboexl3's `newKernel` through the host table | the test consumer's `newKernel` |

The host table is the server's function table that a plugin binds at load (`kit/kernel_launch.zig:1-6, 18-19`). With
it bound, `newKernel` asks the server, whose `makeKernel` (mlx-serve bc6c5f2c `src/host/plugin_host_impl.zig:174-195`)
calls `mlxlc_kernel_new` on a server built with the mlx-stream shims, else `mlx_fast_metal_kernel_new`.
`mlxlc_kernel_new` is `mlx_fast_metal_kernel_new` plus a launch cache over MLX's own pipeline
(`src/sdk/weights/streamed/csrc/mlx_launch_cache.inc:218-228`). A test binary binds no host table, so its `newKernel`
calls `mlx_fast_metal_kernel_new` itself (the compute lane's consumer on m5-max-eigen,
`~/stq-tests/consumer/sdk/kernel_launch.zig:26`).

mlx-serve bc6c5f2c pins MLX c69303b and mlx-c 56b2d39, and the plugin runs on them. The GPU ticket builds its test
binary on the host's MLX build; on m5-max-eigen, `~/lingp/mlx-serve/lib/mlx/.version` read
`mlx=c69303b00f5f mlxc=56b2d39fc831` on 2026-11-10. mlx-c passes no compile options to MLX (`mlx/c/fast.cpp:445`). MLX
then uses `fast::CompileOptions{}`, whose math mode is `MathMode::Safe` (`mlx/backend/common/custom_kernel.h:23`,
`mlx/fast.h:98`). `Device::build_library_` sets `MTLMathModeSafe` and the Metal language version of the OS
(`mlx/backend/metal/device.cpp:53-79, 81-96, 755-756`): 4.0 on macOS 26 and 4.1 on macOS 27. Every kernel in one
process gets the same version.

None of the four kernels comes from a precompiled metallib. Contraction is the compiler's fusion of `a * b + c` into
one `fma`, which rounds once instead of twice. Ling's combine and router texts set `#pragma METAL fp contract(off)`
(`kernels/combine/contract.metal:3`, `kernels/router/score_sigmoid.metal:5`). These four texts and their headers set
no `#pragma`. The host's and the test copy's headers are empty, and o_proj's prepare and this launch use
`decode.header`. o_proj's dequantize, matmul and finish are the same launches in both chains, so their compile path
cannot make the chains differ.

### Where contraction cannot change a word

Contraction cannot change a word when y is f16. It also cannot when y is bf16, no nonzero element is below 2^-63 in
magnitude, and eps is 0 or at least 2^-100. The texts hold three candidates:

- `sumsq += xs[i] * xs[i]`;
- `sumsq / float(DV) + eps`, if the compiler turns the divide by 128 into a multiply;
- `f16(o) · suh`, two f16 values multiplied in f32, which feeds the butterfly's first add or subtract.

MLX and these texts do not say whether safe mode contracts them. On the inputs above, each product is exact in f32, so
`fma(a, b, c)` and `a * b + c` round to the same f32.

- A widened bf16 has 8 significant bits, so its square needs at most 16, and an f32 significand holds 24. A bf16 of
  magnitude 2^-63 or more has a square of 2^-126 or more, inside f32's normal range.
- A finite f16 is 0 or between 2^-24 and 65504 in magnitude. So a product of two f16 values needs at most 22 bits and
  is 0 or between 2^-48 and 2^32, inside f32's normal range.
- The factor 1/128 is 2^-7, so sumsq · 2^-7 is exact while it is 0 or at least 2^-126. Below that, it is far under
  half an ulp of an eps of 2^-100 or more, so both forms give eps. With eps 0, both forms round the same product.

mlx-serve relies on the same fact for its SwiGLU multiplies, without the range limit (bc6c5f2c
`src/sdk/blocks/mlp/fused_activations.zig:41-43`). From 2^64 up, a bf16 square overflows, and both forms give an
infinite sum, so these elements are covered too. A nonzero bf16 element below 2^-63 in magnitude has a square under
2^-126, the smallest normal f32. That square can lose bits, so this argument does not cover such inputs. The GPU test
does not reach them either, because it draws standard-normal bf16 values.

A red check is a planted fault that the test must catch: its words must differ from the served chain's, the host's
norm-gate and then o_proj's `linear`. A fault could force the other choice at a candidate, with an explicit `fma` or
`#pragma METAL fp contract(off)`. On every covered input it gives the host's words, whichever way the host compiles, so
the test could never catch it. The GPU test therefore has no contraction red check. Its reduction red check is
`kernels/test/fault/kdanorm_sum_strided.metal`, which sums each lane's strided channels.
