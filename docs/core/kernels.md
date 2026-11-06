# The kernel catalog: the core

Every kernel made in `src/core/`, one entry each, by the name `ks.newKernel` gives it. The kernels made in a model's
folder are in that model's catalog: [Qwen3.8](../models/qwen38/kernels.md), [DeepSeek-V4.1](../models/deepseek_v41/kernels.md).
[The engines](engines.md) tell how the kernels chain into each path, with diagrams; [the glossary](glossary.md) defines
every term.

## How an entry reads

Each entry has two parts:

- **The generated block**, between `<!-- kernel-facts: ... -->` markers, is `tools/kernel_docs.py`'s from the code:
  where the kernel is made, its input and output names, and the `.metal` pieces its source and header are the
  concatenation of (`dense/dec_head` is `src/core/kernels/dense/dec_head.metal`; `qwen38:attn_io/prep3` a file of
  `src/models/qwen38/kernels/`; `text '...'` a string the Zig computes), with each text's size and hash. Never edit it
  by hand: `python3 tools/kernel_docs.py --write` rewrites it.
- **The prose fields**, written by hand beside the facts: who uses it (the models and entry points), the shapes and
  dtypes, the grid and threadgroup, the template arguments and what they serve, the chain it replaces, its bits against
  that chain or `format.project`, and the tests that pin it.

A family of names the code computes (a variant table) is one entry headed `<prefix>_<variant>`, its block listing every
name. `zig build doc-check` fails when a kernel has no entry, an entry names no kernel, a generated block is not the
code's, or a prose field is missing.

## `src/core/decode.zig`

The routed experts at decode rows (1 to 16): form 1 (`Kernels`, not served), form 3's split-K chain and its variants 7,
8 and 9 (`Kernels3`; form 9 served), forms 5 and 6 (`Kernels5`); the clamped fin7 (DeepSeek-V4 and V4.1 and GLM,
`moeClamped`). The fused shared expert and the router fold in those launches are Qwen3.8's
([its catalog](../models/qwen38/kernels.md), `src/models/qwen38/shared.zig`); the banked decode over an expert
streamer's slot banks and its grouped variants DeepSeek-V4.1's ([its catalog](../models/deepseek_v41/kernels.md),
`src/models/deepseek_v41/banked.zig`).

### `turboexl3_dfin3`

<!-- kernel-facts: turboexl3_dfin3 -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:345` (`const Kernels3`) |
| inputs | `zd`, `slots`, `scores`, `down_svh`, `hsx` |
| outputs | `y` |
| source | `text_dfin3` = `decode/dfin3_1` ++ `decode/dfin3_expert` ++ `decode/dfin3_2` (1324 B, `f7c3494d91f8`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: launch 5 of form 3's chain: under `Config.decode_form` 3, and under forms 7, 8 and 9 whenever TOPK
  < 4 (`turboexl3_dfin7` needs a top-k of 4 or more; `Kernels3.moeForm`), through `root.moe` / `moeWithShared` /
  `root.moeClamped`. Not served at Qwen3.8's top-10 or DeepSeek-V4.1's top-6 by default; the topk test names its
  below-top-4 callers: a shared expert served as a one-expert bank (top-1) and DeepSeek-V4.1's DSpark stages (top-3).
  Its text also runs as `turboexl3_grouped_finish` (DeepSeek-V4.1's `wo_a`, `root.linearGrouped`, top-1 with unit
  scores).
- **Shapes and dtypes:** zd f32 [KS2, RK, HID] (`down3`'s planes); slots i32 or u32 [RK]; scores S [RK] (S bf16, f16 or
  f32); down_svh f16 [E, HID]; hsx f32 [RK, INTER/128] (`fin`'s hs). Out: y OUT_T [rows, HID] (OUT_T = x's dtype).
- **Grid and threadgroup:** `((HID/128)·128, rows, 1)` / `(128, 1, 1)`: one threadgroup per (128-wide hidden block,
  row), looping over the row's TOPK slots in slot order.
- **Template args:** HID, TOPK, RKC, KS (KS2, the down planes it adds), OUT_T (y's dtype: bf16 served, f32 in the
  tests), DEFER, NBI (INTER/128, the block sums it adds). No trellis read.
- **Replaces:** `turboexl3_finish` and `turboexl3_reduce` of the unfused chain (2 launches, 2 levels). Base of form 3's
  chain.
- **Bits:** within tolerance of `format.project` (through `ref.moeRowDec`), as a launch of form 3's chain: it rounds
  where `format.project` rounds but sums in form 3's fixed split-K order. The bars: `root.moe` (form 9, form 3's bytes)
  in bf16 within 1.5e-2 (`ref.compare`); form 3 at top-1 to top-3 within 5e-3 beyond the output store's rounding
  (`ref.compareRounded`, 2^-11 for f32, 2^-8 + 2^-11 for bf16). A row's bytes do not depend on how many rows share the
  call (the K partition is the geometry's alone). Each slot's finished value rounds to f16 before its score `fma`; y
  rounds once to OUT_T.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: decode chain (form 3), a row's bytes are the same alone and
  among 2..16 rows (TURBOEXL3_GPU=1)" (GPU, bytes: row-count independence); `src/core/topk_test.zig` "turboexl3 gpu
  topk: root.moe at top-1, 2 and 3 gives form 3's bytes under decode forms 7, 8 and 9, ..." (GPU: forms 7 to 9 give form
  3's bytes at top-1, 2 and 3 (this launch finishes them), bf16 and f32; form 3 within 5e-3 of the reference);
  `src/models/deepseek_v41/grouped.zig` "turboexl3 grouped: the reused texts read every operand the launches bind and
  every template constant they pass" (CPU: the text under `turboexl3_grouped_finish` reads every operand and constant);
  `src/models/deepseek_v41/deepseek_gemv_test.zig` "turboexl3 deepseek gemv: the grouped dense-GEMV texts are dense's
  GEMV statements from the input loads to ..." (CPU: the grouped GEMV's plane store matches this text's plane read);
  `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode texts: each is its base text with the bank's lines
  in place of the expert line ..." (CPU: `turboexl3_dfin3b` is this text with the expert line swapped).

### `turboexl3_dfin7`

<!-- kernel-facts: turboexl3_dfin7 -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:353` (`const Kernels3`) |
| inputs | `zd`, `slots`, `scores`, `down_svh`, `hsx` |
| outputs | `y` |
| source | `text_dfin7` = `decode/dfin7` (1771 B, `8412c8c0b289`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** core: launch 5 of forms 7, 8 and 9 (the default) when TOPK ≥ 4 (below that `turboexl3_dfin3` runs;
  `Kernels3.moeForm`): `root.moe` / `moeWithShared` / `root.moeClamped` at 1 to 16 decode rows. The fused shared expert
  runs `turboexl3_dfin7s` instead. Qwen3.8 at the decode rows its fused shared-expert entries do not take (more than 4
  rows, or a call they refuse: `moeFusedShared` leaves the caller its chain) (top-10); DeepSeek-V4.1 through
  `moeClamped` (its streamed decode over an expert streamer runs the banked texts of `moeClampedBanked` instead)
  (top-6).
- **Shapes and dtypes:** as `turboexl3_dfin3`: zd f32 [KS2, RK, HID]; slots i32 or u32 [RK]; scores S [RK] (bf16, f16 or
  f32); down_svh f16 [E, HID]; hsx f32 [RK, INTER/128]. Out: y OUT_T [rows, HID].
- **Grid and threadgroup:** `((HID/128)·32·TOPK, rows, 1)` / `(32·TOPK, 1, 1)`: per (128-wide hidden block, row),
  simdgroup k finishes slot k into threadgroup memory, then the first 128 threads run the score `fma` chain.
- **Template args:** HID, TOPK, RKC, KS (KS2), OUT_T, DEFER, NBI: exactly `dfin3`'s. No trellis read.
- **Replaces:** `turboexl3_dfin3` (form 3's launch 5: `turboexl3_finish`, `turboexl3_reduce` of the unfused chain), with
  the slots finished side by side: one H128 deep where dfin3 runs TOPK of them in a row.
- **Bits:** byte-identical to `turboexl3_dfin3`: each slot's planes, deferred map, finish and f16 round as dfin3, then
  dfin3's `fma` chain in slot order.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: decode chain forms 7, 8 and 9 equal form 3 byte for byte, at
  any tiles per threadgroup (TURBOEXL3_GPU=1)" (GPU, bytes against form 3, rows 1 to 5, 8, 16, f32 and bf16);
  `src/core/gpu_test.zig` "turboexl3 gpu: root.moe at one decode row on the host's unreshaped operands gives the
  reshaped chain's bytes (TURBOEXL3_GPU=1)" (GPU, bytes: form 9 at one row on the unreshaped operands);
  `src/core/gpu_test.zig` "turboexl3 gpu: root.moe (the sushi_exl3-shaped entry) equals the reference through the bf16
  host operands (TURBOEXL3_GPU=1)" (GPU, within 1.5e-2 in bf16 at top-10); `src/core/decode.zig` "turboexl3 decode forms
  7 and 8 texts: every template constant read is passed, every signature name is used" (CPU: constants and names).

### `turboexl3_dn5`

<!-- kernel-facts: turboexl3_dn5 -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:650` (`const Kernels5`) |
| inputs | `zg`, `zu`, `xsg`, `xsu`, `slots`, `gate_svh`, `up_svh`, `down_suh`, `down_t`, `down_svh` |
| outputs | `wv` |
| source | `text_dn5` = `decode/dn5_head` ++ `decode/dn5_tail` (5054 B, `58b2ca539f44`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** core: launch 2 of form 5 (`decode.Kernels5`), `root.moe` / `moeWithShared` under `Config.decode_form` 5
  (`root.kernels5For`: BT 4, `ks_max` 4, `gmax` 4). Unclamped only (`Kernels5` refuses a clamped geometry; `moeClamped`
  under form 5 runs form 9). Not served: the default is form 9.
- **Shapes and dtypes:** zg, zu f32 [KS, RK, INTER] and xsg, xsu f32 [RK, HID/128] (`gu5`'s); slots i32 or u32 [RK];
  gate_svh, up_svh, down_suh f16 [E, INTER]; down_t U16 [E, INTER/16, HID/16, NHW]; down_svh f16 [E, HID]. Out: wv f16
  [RK, HID] (each row-slot's down output finished up to its score).
- **Grid and threadgroup:** `((HID/128)·256, RK, 1)` / `(256, 1, 1)`: per (128-wide hidden block, lead row-slot), 8
  simdgroups; a non-lead row-slot's threadgroup returns.
- **Template args:** NHW, CODEBOOK, WBITS, HID, INTER, RKC, KS (the gate|up planes), KS2 (the down's slices, form 3's
  partition), GMAX (row-slots per expert group: min(`gmax`, rows), 1 to 4), DEFER, IR (1: the integer-rate reader, NHW %
  16 = 0, the same codewords as the general reader). Every admitted rate, codebook and window.
- **Replaces:** `turboexl3_fin3` → `turboexl3_down3` → `turboexl3_dfin3`'s per-slot finish (3 launches, 3 levels;
  unfused: `turboexl3_finish` × 3, `turboexl3_swiglu`, `turboexl3_prep`, `turboexl3_gemv`). Form 5 is 3 launches in 3
  levels against form 3's 5.
- **Bits:** byte-identical to form 3: fin3's mid in fin3's order, down3's K partition and per-lane FMA order, dfin3's
  finish and f16 round (form 5's output equals form 3's bit for bit).
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: decode chain form 5 equals form 3 byte for byte
  (TURBOEXL3_GPU=1)" (GPU, bytes against form 3 at rows 1 to 5, 8, 16, `gmax` 1 to 4, BT 1, 2, 4, 8, MUL1's deferred map
  on and off, f32); `src/core/gpu_test.zig` "turboexl3 gpu: root.moe at one decode row on the host's unreshaped operands
  gives the reshaped chain's bytes (TURBOEXL3_GPU=1)" (GPU, bytes: form 5 at one row on the unreshaped operands, bf16);
  `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: at a limit no value reaches, every clamped launch
  gives the unclamped one's bytes ..." (GPU, bytes: `root.moe` at form 5, bf16 and f32, rows 1, 4, 16, equals
  `moeClamped`'s form 9 at a limit no value reaches); `src/core/decode.zig` "turboexl3 decode form 5 texts: every
  template constant read is passed, every signature name is used" (CPU: constants and names);
  `src/models/deepseek_v41/clamped_test.zig` "turboexl3 clamped refusals: moeClamped takes a limit in 1..128; form 1
  (Kernels) and forms 5 and 6 ..." (CPU: `Kernels5` refuses a clamped geometry).

### `turboexl3_dn6`

<!-- kernel-facts: turboexl3_dn6 -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:652` (`const Kernels5`) |
| inputs | `zg`, `zu`, `xsg`, `xsu`, `slots`, `gate_svh`, `up_svh`, `down_suh`, `down_t`, `down_svh`, `cnt`, `scores` |
| outputs | `wv`, `y` |
| source | `text_dn6` = `decode/dn5_head` ++ `decode/dn6_tail` (6350 B, `020e3537b6a1`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** core: launch 2 of form 6 (`decode.Kernels5.moe` with `fused`), `root.moe` / `moeWithShared` under
  `Config.decode_form` 6 (`root.kernels5For`'s shape). Unclamped only. Not served: the default is form 9.
- **Shapes and dtypes:** `dn5`'s inputs, then cnt u32 [rows · HID/128] (the arrival counters `gu5` zeroed) and scores S
  [RK] (bf16, f16 or f32). Out: wv f16 [RK, HID] (the exchange between the slots, stored through coherent pointers; the
  host frees it); y OUT_T [rows, HID].
- **Grid and threadgroup:** as `dn5`: `((HID/128)·256, RK, 1)` / `(256, 1, 1)`.
- **Template args:** `dn5`'s (NHW, CODEBOOK, WBITS, HID, INTER, RKC, KS, KS2, GMAX, DEFER, IR), then TOPK (the arrivals
  per (row, block)) and OUT_T.
- **Replaces:** `turboexl3_fin3` → `turboexl3_down3` → `turboexl3_dfin3` (3 launches, 3 levels), or form 5's `dn5` →
  `red5` (2 launches, 2 levels), in one launch: each slot's simdgroup counts its arrival per (row, 128-block of hidden)
  with a relaxed device `fetch_add` between seq_cst fences, and the one that brings the row's TOPK-th slot runs `red5`'s
  `fma` chain. Form 6 is 2 launches in 2 levels against form 3's 5.
- **Bits:** byte-identical to form 3 (and to form 5): `dn5`'s statements, then `red5`'s `fma` chain in slot order.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: decode chain form 5 equals form 3 byte for byte
  (TURBOEXL3_GPU=1)" (GPU, bytes: form 6 against form 3, the same variants and rows as form 5);
  `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: at a limit no value reaches, every clamped launch
  gives the unclamped one's bytes ..." (GPU, bytes: `root.moe` at form 6, bf16 and f32, equals `moeClamped`'s form 9 at
  a limit no value reaches); `src/core/decode.zig` "turboexl3 decode form 5 texts: every template constant read is
  passed, every signature name is used" (CPU: constants and names).

### `turboexl3_down`

<!-- kernel-facts: turboexl3_down -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:92` (`const Kernels`) |
| inputs | `h`, `slots`, `scores`, `down_t`, `down_suh`, `down_svh` |
| outputs | `y` |
| source | `text_down` = `decode/down` (5902 B, `41783dbdc215`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** test only: launch 2 of form 1 (`decode.Kernels`), made only in `src/core/gpu_test.zig`. Not served:
  `Config.decode_form` has no form 1, so `moe` cannot select it; `Kernels` refuses a clamped geometry.
- **Shapes and dtypes:** h f32 [rows · TOPK, INTER] (`gu`'s); slots u32 [rows · TOPK]; scores f32 [rows · TOPK]; down_t
  U16 [E, INTER/16, HID/16, NHW]; down_suh f16 [E, INTER]; down_svh f16 [E, HID]. Out: y f32 [rows, HID].
- **Grid and threadgroup:** `((HID/128)·128, rows, 1)` / `(128, 1, 1)`: one threadgroup per (128-wide hidden block,
  row), the row's slots in order.
- **Template args:** NHW, CODEBOOK, WBITS, HID, INTER, TOPK, PROBE (timing probes only: 0 served text, 1 skips the
  codebook hash, 2 the funnel too); the same list as `gu`. Any admitted rate, codebook and window (the test: n 32, 40,
  48, 64, 80, 128; MUL1 and MCG; a 12-bit window).
- **Replaces:** `turboexl3_prep`, `turboexl3_gemv`, `turboexl3_finish` and `turboexl3_reduce` of the unfused chain (4
  launches, 4 levels).
- **Bits:** within 5e-3 of `format.project` through `ref.moeRowDec` (`ref.compare`, f32): `format.project`'s rounding
  points, its own summation order (the input split four ways per threadgroup), the slots summed in slot order with no
  atomics.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: the two-launch decode chain equals the host reference at every
  rate and codebook (TURBOEXL3_GPU=1)" (GPU: within 5e-3 at every listed rate and codebook, Qwen3.8's geometry at 1, 4
  and 8 rows); `src/core/decode.zig` "turboexl3 decode texts: every template constant the texts read is one the launch
  passes" (CPU: every constant read is passed, every input named, the y store pinned);
  `src/models/deepseek_v41/clamped_test.zig` "turboexl3 clamped refusals: moeClamped takes a limit in 1..128; form 1
  (Kernels) and forms 5 and 6 ..." (CPU: `Kernels` refuses a clamped geometry).

### `turboexl3_down3`

<!-- kernel-facts: turboexl3_down3 -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:344` (`const Kernels3`) |
| inputs | `hdp`, `slots`, `down_t` |
| outputs | `zd` |
| source | `text_down3` = `decode/down3_1` ++ `decode/rk_line` ++ `decode/down3_2` ++ `decode/down3_w` ++ `decode/down3_3` (2340 B, `873aac06a4a9`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: launch 4 of every `decode.Kernels3` form (3, 7, 8 and 9) at every row count: `root.moe` /
  `moeWithShared` / `root.moeClamped` at 1 to 16 decode rows. Its text also runs as `turboexl3_down3s` (the fused shared
  expert) and as `turboexl3_grouped_gemv` (DeepSeek-V4.1's `wo_a`, `root.linearGrouped`). Qwen3.8 at the decode rows its
  fused shared-expert entries do not take (more than 4 rows, or a call they refuse: `moeFusedShared` leaves the caller
  its chain); DeepSeek-V4.1 through `moeClamped` (its streamed decode over an expert streamer runs the banked texts of
  `moeClampedBanked` instead).
- **Shapes and dtypes:** hdp f16 [RK, INTER] (`fin3`'s or `fin7`'s hd); slots i32 or u32 [RK]; down_t U16 [E, INTER/16,
  HID/16, NHW]. Out: zd f32 [KS2, RK, HID] (the down's split-K planes).
- **Grid and threadgroup:** `((HID/(16·BT))·128, RK, z)` / `(128, 1, 1)`, z = max(1, KS2 / kdiv[min(rows, 4) − 1]) (form
  9: `kdiv9d` or `kdiv9`, all 1 in `root`, so z = KS2); BT as `gu3`. The four simdgroups take the four parts of a slice.
- **Template args:** NHW, CODEBOOK, WBITS, HID, INTER, RKC, KS (here KS2, the down's slices: `ks_max` 4 halved until
  (INTER/16) divides into KS2 × 4 parts of at least 2 tiles; 2 at 2560 × 640, 4 at 5120 × 2304), SPT (KS2 / z), BT,
  GROUP, DEFER. Every admitted rate, codebook and window.
- **Replaces:** `turboexl3_gemv` of the unfused chain (the down inner product: 1 launch). Base of form 3's chain.
- **Bits:** within tolerance of `format.project` (through `ref.moeRowDec`), as a launch of form 3's chain: it rounds
  where `format.project` rounds but sums in form 3's fixed split-K order. The bars: `root.moe` (form 9, form 3's bytes)
  in bf16 within 1.5e-2 (`ref.compare`); form 3 at top-1 to top-3 within 5e-3 beyond the output store's rounding
  (`ref.compareRounded`, 2^-11 for f32, 2^-8 + 2^-11 for bf16); clamped (DeepSeek-V4.1's geometry, form 9) within 5e-3
  of `ref.moeRowClamped` (f32 `ref.compare`, bf16 `ref.compareRounded` at 2^-8). A row's bytes do not depend on how many
  rows share the call (the K partition is the geometry's alone).
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: decode chain (form 3), a row's bytes are the same alone and
  among 2..16 rows (TURBOEXL3_GPU=1)" (GPU, bytes: row-count independence); `src/core/gpu_test.zig` "turboexl3 gpu:
  root.moe (the sushi_exl3-shaped entry) equals the reference through the bf16 host operands (TURBOEXL3_GPU=1)" (GPU,
  within 1.5e-2 in bf16); `src/core/gpu_test.zig` "turboexl3 gpu: decode chain forms 7, 8 and 9 equal form 3 byte for
  byte, at any tiles per threadgroup (TURBOEXL3_GPU=1)" (GPU, bytes: in every form, at BT 1, 2, 4 and several K
  divisors); `src/models/deepseek_v41/grouped.zig` "turboexl3 grouped: the reused texts read every operand the launches
  bind and every template constant they pass" (CPU: the text read under `turboexl3_grouped_gemv` reads every operand and
  constant); `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode texts: each is its base text with the
  bank's lines in place of the expert line ..." (CPU: `turboexl3_down3b` is this text with the weight base swapped).

### `turboexl3_fin3`

<!-- kernel-facts: turboexl3_fin3 -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:343` (`const Kernels3`) |
| inputs | `zg`, `zu`, `slots`, `gate_svh`, `up_svh`, `down_suh`, `xsg`, `xsu` |
| outputs | `hd`, `hs` |
| source | `text_fin3` = `decode/fin3` (2206 B, `187e822beba3`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: launch 3 of form 3's chain, made by every `decode.Kernels3` but launched only under
  `Config.decode_form` 3 (`root.moe` / `moeWithShared`; `Kernels3.moe` in the tests and the bench), unclamped only (a
  clamped `Kernels3` refuses form 3; `moeClamped` under form 3 runs form 9). Not served: the default form 9 runs
  `turboexl3_fin7` in its place.
- **Shapes and dtypes:** zg, zu f32 [KS, RK, INTER] (`gu3`'s planes, or `gu8`'s); slots i32 or u32 [RK]; gate_svh,
  up_svh f16 [E, INTER]; down_suh f16 [E, INTER]; xsg, xsu f32 [RK, HID/128] (`prep2`'s sg, su). Out: hd f16 [RK, INTER]
  (the down's prepared input, lane order); hs f32 [RK, INTER/128] (its 128-block sums).
- **Grid and threadgroup:** `((INTER/128)·128, RK, 1)` / `(128, 1, 1)`: one threadgroup per (128-wide intermediate
  block, row-slot); the H128s go through threadgroup memory.
- **Template args:** INTER, RKC (RK), KS (the gate|up slices it adds, in slice order), DEFER (the deferred MUL1 map: A ×
  acc + C × Σx), NBH (HID/128, the block sums it adds). No trellis read: one text for every rate, codebook and window.
- **Replaces:** `turboexl3_finish` × 2, `turboexl3_swiglu` and `turboexl3_prep` of the unfused chain (4 launches, 3
  levels). Base of form 3's chain; `turboexl3_fin7` is its form-7 counterpart.
- **Bits:** within tolerance of `format.project` (through `ref.moeRowDec`), as a launch of form 3's chain: it rounds
  where `format.project` rounds but sums in form 3's fixed split-K order. The bars: `root.moe` (form 9, form 3's bytes)
  in bf16 within 1.5e-2 (`ref.compare`); form 3 at top-1 to top-3 within 5e-3 beyond the output store's rounding
  (`ref.compareRounded`, 2^-11 for f32, 2^-8 + 2^-11 for bf16). A row's bytes do not depend on how many rows share the
  call (the K partition is the geometry's alone). SwiGLU in f32 with `metal::exp`.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: decode chain (form 3), a row's bytes are the same alone and
  among 2..16 rows (TURBOEXL3_GPU=1)" (GPU, bytes: row-count independence); `src/core/gpu_test.zig` "turboexl3 gpu:
  decode chain forms 7, 8 and 9 equal form 3 byte for byte, at any tiles per threadgroup (TURBOEXL3_GPU=1)" (GPU: form 3
  is the reference forms 7 to 9 equal); `src/core/gpu_test.zig` "turboexl3 gpu: decode chain form 5 equals form 3 byte
  for byte (TURBOEXL3_GPU=1)" (GPU: form 3 is the reference forms 5 and 6 equal); `src/core/topk_test.zig` "turboexl3
  gpu topk: root.moe at top-1, 2 and 3 gives form 3's bytes under decode forms 7, 8 and 9, ..." (GPU: form 3 within 5e-3
  of the reference at top-1 to top-3); `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: at a limit no
  value reaches, every clamped launch gives the unclamped one's bytes ..." (GPU, bytes: `root.moe` at form 3 equals
  `moeClamped` at a limit no value reaches).

### `turboexl3_fin7`

<!-- kernel-facts: turboexl3_fin7 -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:348` (`const Kernels3`) |
| inputs | `zg`, `zu`, `slots`, `gate_svh`, `up_svh`, `down_suh`, `xsg`, `xsu` |
| outputs | `hd`, `hs` |
| source | `text_fin7` = `decode/fin7_1` ++ `decode/expert_line` ++ `decode/fin7_2` ++ `decode/fin7_planes` ++ `decode/fin7_3` ++ `decode/fin7_sums` ++ `decode/fin7_4` ++ `decode/fin7_swiglu` ++ `decode/fin7_5` (2152 B, `7360fcbe736a`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** core: launch 3 of forms 7, 8 and 9 (the default) at 1 to 16 decode rows, unclamped geometry
  (`Geometry.limit` 0; a clamped geometry builds `turboexl3_fin7c` in its place): `root.moe` / `moeWithShared`. The
  fused shared-expert entries run its text under `turboexl3_fin7s` / `turboexl3_gu3sf`. Qwen3.8 at the decode rows its
  fused shared-expert entries do not take (more than 4 rows, or a call they refuse: `moeFusedShared` leaves the caller
  its chain).
- **Shapes and dtypes:** as `turboexl3_fin3`: zg, zu f32 [KS, RK, INTER]; slots i32 or u32 [RK]; gate_svh, up_svh,
  down_suh f16 [E, INTER]; xsg, xsu f32 [RK, HID/128]. Out: hd f16 [RK, INTER] (lane order); hs f32 [RK, INTER/128].
- **Grid and threadgroup:** `((INTER/128)·32, RK, 1)` / `(32, 1, 1)`: one simdgroup per (128-wide intermediate block,
  row-slot), lane L holding elements L + 32q; the H128s in registers (`turboexl3_h128r`), no threadgroup barriers.
- **Template args:** INTER, RKC, KS (the gate|up slices), DEFER, NBH: exactly `fin3`'s five. No trellis read.
- **Replaces:** `turboexl3_fin3` (form 3's launch 3: `turboexl3_finish` × 2, `turboexl3_swiglu`, `turboexl3_prep` of the
  unfused chain, 4 launches, 3 levels), in one launch of the same depth.
- **Bits:** byte-identical to `turboexl3_fin3`: fin3's arithmetic in fin3's order (`turboexl3_h128r` runs
  `turboexl3_h128`'s stages, the block sum the same four simd sums). SwiGLU in f32 with `metal::exp`.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: decode chain forms 7, 8 and 9 equal form 3 byte for byte, at
  any tiles per threadgroup (TURBOEXL3_GPU=1)" (GPU, bytes against form 3 at rows 1 to 5, 8 and 16, f32 and bf16, MUL1's
  deferred map on and off); `src/core/gpu_test.zig` "turboexl3 gpu: root.moe at one decode row on the host's unreshaped
  operands gives the reshaped chain's bytes (TURBOEXL3_GPU=1)" (GPU, bytes: form 9 at one row on the host's unreshaped
  operands); `src/core/gpu_test.zig` "turboexl3 gpu: root.moe (the sushi_exl3-shaped entry) equals the reference through
  the bf16 host operands (TURBOEXL3_GPU=1)" (GPU, within 1.5e-2 of the reference in bf16); `src/core/decode.zig`
  "turboexl3 decode forms 7 and 8 texts: every template constant read is passed, every signature name is used" (CPU:
  every constant read is passed, every input and output named); `src/models/deepseek_v41/clamped_test.zig` "turboexl3
  clamped texts: each clamped kernel is its base text with the SwiGLU line swapped for the clamp ..." (CPU: `fin7c` is
  this text with the SwiGLU line swapped and nothing else; this text reads no LIMIT).

### `turboexl3_fin7c`

<!-- kernel-facts: turboexl3_fin7c -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:350` (`const Kernels3`) |
| inputs | `zg`, `zu`, `slots`, `gate_svh`, `up_svh`, `down_suh`, `xsg`, `xsu` |
| outputs | `hd`, `hs` |
| source | `text_fin7c` = `decode/fin7_1` ++ `decode/expert_line` ++ `decode/fin7_2` ++ `decode/fin7_planes` ++ `decode/fin7_3` ++ `decode/fin7_sums` ++ `decode/fin7_4` ++ `decode/fin7_swiglu_clamp` ++ `decode/fin7_5` (2505 B, `a82ad5c1492b`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moeClamped` at 1 to 16 decode rows: launch 3 of forms 7, 8 and 9 as configured, and of form 9
  when `Config.decode_form` is 3, 5 or 6 (those forms have no clamp; form 9 shares their bytes). A clamped geometry
  (`Geometry.limit` 1 to 128) builds it in `fin7`'s place, so every fin7 launch of that `Kernels3` clamps. DeepSeek-V4.1
  (swiglu_limit 10) through `moeClamped`; its streamed decode runs `turboexl3_fin7cb` (this text with the bank's expert
  line) instead. The clamp's users: DeepSeek-V4 and V4.1 (swiglu_limit 10) and GLM (the same clamp), each through
  `root.moeClamped`; Ripple's clamped SwiGLU is its own (`turboexl3_ric_swiglu_prep_sc` / `_prep`,
  `src/models/ricursive/`).
- **Shapes and dtypes:** as `turboexl3_fin7`: zg, zu f32 [KS, RK, INTER]; slots i32 or u32 [RK]; gate_svh, up_svh,
  down_suh f16 [E, INTER]; xsg, xsu f32 [RK, HID/128]. Out: hd f16 [RK, INTER]; hs f32 [RK, INTER/128].
- **Grid and threadgroup:** as `fin7`: `((INTER/128)·32, RK, 1)` / `(32, 1, 1)`.
- **Template args:** fin7's five (INTER, RKC, KS, DEFER, NBH), then LIMIT (the swiglu_limit, an integer 1 to 128, exact
  in f16): the gate clamped from above at LIMIT and up to [−LIMIT, LIMIT], in f32 on the finished f16 values, before the
  SiLU; a NaN passes through.
- **Replaces:** `turboexl3_fin7`'s launch with the clamp (unfused: `turboexl3_finish` × 2, `turboexl3_swiglu`,
  `turboexl3_prep`, 4 launches, 3 levels; the unfused chain has no clamp).
- **Bits:** byte-identical to `turboexl3_fin7` (so to `fin3`) where no value reaches the limit; with the clamp acting,
  within 5e-3 of `ref.moeRowClamped` (f32 `ref.compare`; bf16 `ref.compareRounded` at 2^-8) at DeepSeek-V4.1-Flash's
  routed geometry, the inputs scaled so at least 1% of the gates and of the ups pass the limit.
- **Pinned by:** `src/models/deepseek_v41/clamped_test.zig` "turboexl3 clamped texts: each clamped kernel is its base
  text with the SwiGLU line swapped for the clamp ..." (CPU: fin7's text with the SwiGLU line swapped for the clamp
  piece, nothing else; only the clamp reads LIMIT); `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped:
  at a limit no value reaches, every clamped launch gives the unclamped one's bytes ..." (GPU, bytes: at limit 128 on
  Qwen3.8's geometry every decode form, bf16 and f32, rows 1, 4, 16, equals the unclamped launch);
  `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: root.moeClamped at DeepSeek-V4.1-Flash's routed
  geometry (5120 x 2304, 384 experts, ..." (GPU: K2, K3 and K6 at decode rows 1, 4, 16 within 5e-3 of the clamped
  reference); `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode texts: each is its base text with the
  bank's lines in place of the expert line ..." (CPU: `turboexl3_fin7cb` is this text with the expert line swapped).

### `turboexl3_gu`

<!-- kernel-facts: turboexl3_gu -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:91` (`const Kernels`) |
| inputs | `x`, `slots`, `gate_t`, `gate_suh`, `gate_svh`, `up_t`, `up_suh`, `up_svh` |
| outputs | `h` |
| source | `text_gu` = `decode/gu` (8530 B, `12117a37337c`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** test only: launch 1 of form 1 (`decode.Kernels`), made only in `src/core/gpu_test.zig`. Not served: `moe`
  cannot select form 1 (`Config.decode_form` 3, 5 to 9).
- **Shapes and dtypes:** x f32 [rows, HID] (the doc comment's; the kernel reads it through an f16 round); slots u32
  [rows · TOPK]; gate_t, up_t U16 [E, HID/16, INTER/16, NHW]; gate_suh, up_suh f16 [E, HID]; gate_svh, up_svh f16 [E,
  INTER]. Out: h f32 [rows · TOPK, INTER] (silu(gate) · up).
- **Grid and threadgroup:** `((INTER/128)·128, TOPK, rows)` / `(128, 1, 1)`: one threadgroup per (128-wide intermediate
  block, slot, row).
- **Template args:** NHW, CODEBOOK, WBITS, HID, INTER, TOPK, PROBE (as `down`). Any admitted rate, codebook and window;
  geometry admitted when HID and INTER are multiples of 128 and TOPK is 1 to 32.
- **Replaces:** `turboexl3_prep` × 2, `turboexl3_gemv` × 2, `turboexl3_finish` × 2 and `turboexl3_swiglu` of the unfused
  chain (7 launches, 4 levels). Form 1 (`gu` → `down`) is 2 launches in 2 levels against the unfused 11 in 8.
- **Bits:** within 5e-3 of `format.project` through `ref.moeRowDec` (`ref.compare`, f32): its rounding points, its own
  order (the input tiles split four ways per threadgroup). SwiGLU in f32 with `metal::exp`.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: the two-launch decode chain equals the host reference at every
  rate and codebook (TURBOEXL3_GPU=1)" (GPU: within 5e-3, n 32 to 128, MUL1 and MCG, a 12-bit window);
  `src/core/decode.zig` "turboexl3 decode texts: every template constant the texts read is one the launch passes" (CPU:
  constants and names, the h store pinned); `src/core/decode.zig` "turboexl3 decode geometry admission: 128-aligned
  widths, top-k within the reduce bank" (CPU: the geometry admission every decode form shares).

### `turboexl3_gu3`

<!-- kernel-facts: turboexl3_gu3 -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:342` (`const Kernels3`) |
| inputs | `xgp`, `xup`, `slots`, `gate_t`, `up_t` |
| outputs | `zg`, `zu` |
| source | `text_gu3` = `decode/gu3_1` ++ `decode/rk_line` ++ `decode/gu3_2` ++ `decode/gu3_w` ++ `decode/gu3_3` ++ `decode/gu3_stores` ++ `decode/gu3_4` (2506 B, `89eae969537a`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: launch 2 of form 3's chain (every `decode.Kernels3`), after `turboexl3_prep2` wherever that runs:
  `root.moe` / `moeWithShared` / `root.moeClamped` in form 9 from two rows (and one row without `gu8`), forms 3 and 7 at
  every row count, form 8 where `gu8` is not admitted. The fused shared-expert entries run its text under
  `turboexl3_gu3s` / `turboexl3_gu3sf` instead. Qwen3.8 at the decode rows its fused shared-expert entries do not take
  (more than 4 rows, or a call they refuse: `moeFusedShared` leaves the caller its chain); DeepSeek-V4.1 through
  `moeClamped` (its streamed decode over an expert streamer runs the banked texts of `moeClampedBanked` instead).
- **Shapes and dtypes:** xgp, xup f16 [RK, HID] (`prep2`'s xg, xu); slots i32 or u32 [RK]; gate_t, up_t U16 [E, HID/16,
  INTER/16, NHW]. Out: zg, zu f32 [KS, RK, INTER] (the split-K planes, one per K slice, each element written once, no
  atomics).
- **Grid and threadgroup:** `((INTER/(16·BT))·128, RK, z)` / `(128, 1, 1)`, z = max(1, KS / kdiv[min(rows, 4) − 1])
  threadgroups along K per row-slot (`kThreadgroups`; KS under `kpar` `full`), each looping over SPT = KS / z slices. As
  `root` builds it: forms 3, 7 and 8 at BT 1 with kdiv (1, 2, 2, 4); form 9 at BT 2 (`bt1`, `bt9`) with z = KS (`kdiv9`
  all 1). The four simdgroups are projection × half of a slice.
- **Template args:** NHW (the rate n, halfwords per 16 × 16 tile: even 16 to 128, K = NHW/16), CODEBOOK (MUL1 or MCG),
  WBITS (the codeword window, 8 to 16), HID, INTER, RKC (RK), KS (the gate|up K slices: `ks_max` 4 in `root`, halved
  until (HID/16) divides into KS × 2 parts), SPT (slices per threadgroup), BT (output tiles per threadgroup: 1, 2, 4 or
  8), GROUP (row-slots per lead threadgroup: 1 at one row, `group` (2 in `root`) from two rows, so two row-slots on one
  expert share one decode), DEFER (1: MUL1's deferred map, on for MUL1 packs). Every admitted rate, codebook and window.
- **Replaces:** `turboexl3_gemv` × 2 of the unfused chain (the gate and up inner products: 2 launches, 1 level). Base of
  form 3's chain.
- **Bits:** within tolerance of `format.project` (through `ref.moeRowDec`), as a launch of form 3's chain: it rounds
  where `format.project` rounds but sums in form 3's fixed split-K order. The bars: `root.moe` (form 9, form 3's bytes)
  in bf16 within 1.5e-2 (`ref.compare`); form 3 at top-1 to top-3 within 5e-3 beyond the output store's rounding
  (`ref.compareRounded`, 2^-11 for f32, 2^-8 + 2^-11 for bf16); clamped (DeepSeek-V4.1's geometry, form 9) within 5e-3
  of `ref.moeRowClamped` (f32 `ref.compare`, bf16 `ref.compareRounded` at 2^-8). A row's bytes do not depend on how many
  rows share the call (the K partition is the geometry's alone).
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: decode chain (form 3), a row's bytes are the same alone and
  among 2..16 rows (TURBOEXL3_GPU=1)" (GPU, bytes: row-count independence); `src/core/gpu_test.zig` "turboexl3 gpu:
  root.moe (the sushi_exl3-shaped entry) equals the reference through the bf16 host operands (TURBOEXL3_GPU=1)" (GPU,
  within 1.5e-2 in bf16); `src/core/topk_test.zig` "turboexl3 gpu topk: root.moe at top-1, 2 and 3 gives form 3's bytes
  under decode forms 7, 8 and 9, ..." (GPU: form 3 within 5e-3 at top-1 to top-3); `src/core/gpu_test.zig` "turboexl3
  gpu: decode chain forms 7, 8 and 9 equal form 3 byte for byte, at any tiles per threadgroup (TURBOEXL3_GPU=1)" (GPU:
  form 3 as the reference, and form 9's GEMV from two rows at BT 1, 2, 4, 8 and several K divisors, byte for byte);
  `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode texts: each is its base text with the bank's lines
  in place of the expert line ..." (CPU: `turboexl3_gu3b` is this text with the weight base swapped).

### `turboexl3_gu5`

<!-- kernel-facts: turboexl3_gu5 -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:649` (`const Kernels5`) |
| inputs | `x`, `slots`, `gate_suh`, `up_suh`, `gate_t`, `up_t` |
| outputs | `zg`, `zu`, `xsg`, `xsu`, `cnt` |
| source | `text_gu5` = `decode/gu5` (4502 B, `fc4946388503`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** core: launch 1 of forms 5 and 6 (`decode.Kernels5`), `root.moe` / `moeWithShared` under
  `Config.decode_form` 5 or 6 (`root.kernels5For`: BT 4, `ks_max` 4, `gmax` 4). Unclamped only. Not served: the default
  is form 9.
- **Shapes and dtypes:** x T [rows, HID] (bf16, f16 or f32); slots i32 or u32 [RK]; gate_suh, up_suh f16 [E, HID];
  gate_t, up_t U16 [E, HID/16, INTER/16, NHW]. Out: zg, zu f32 [KS, RK, INTER] (form 3's planes); xsg, xsu f32 [RK,
  HID/128] (the block sums, written by the blk 0 threadgroups); cnt u32 [rows · HID/128] (form 6's arrival counters,
  zeroed here).
- **Grid and threadgroup:** `((INTER/(16·BT))·128, RK, KS)` / `(128, 1, 1)`: per (BT output tiles, lead row-slot, K
  slice); the members' inputs over the slice are prepared in threadgroup memory.
- **Template args:** NHW, CODEBOOK, WBITS, HID, INTER, RKC, KS (form 3's gate|up partition), BT (1, 2, 4 or 8;
  (INTER/16) % BT = 0), GMAX (min(`gmax`, rows), 1 to 4), TOPK, DEFER, IR (the integer-rate reader where NHW % 16 = 0).
  Every admitted rate, codebook and window.
- **Replaces:** `turboexl3_prep2` → `turboexl3_gu3` (2 launches, 2 levels; unfused: `turboexl3_prep` × 2,
  `turboexl3_gemv` × 2), with each expert's weights decoded once per group of up to GMAX row-slots.
- **Bits:** byte-identical to form 3: `prep2`'s arithmetic into threadgroup memory, then `gu3`'s GEMV in its K partition
  and FMA order, the same planes.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: decode chain form 5 equals form 3 byte for byte
  (TURBOEXL3_GPU=1)" (GPU, bytes against form 3 (forms 5 and 6)); `src/core/gpu_test.zig` "turboexl3 gpu: root.moe at
  one decode row on the host's unreshaped operands gives the reshaped chain's bytes (TURBOEXL3_GPU=1)" (GPU, bytes: form
  5 at one row, bf16); `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: at a limit no value reaches,
  every clamped launch gives the unclamped one's bytes ..." (GPU, bytes: forms 5 and 6 through `root.moe` equal
  `moeClamped`'s form 9 at a limit no value reaches); `src/core/decode.zig` "turboexl3 decode form 5 texts: every
  template constant read is passed, every signature name is used" (CPU: constants and names).

### `turboexl3_gu8`

<!-- kernel-facts: turboexl3_gu8 -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:354` (`const Kernels3`) |
| inputs | `x`, `slots`, `gate_suh`, `up_suh`, `gate_t`, `up_t` |
| outputs | `zg`, `zu`, `xsg`, `xsu` |
| source | `text_gu8` = `decode/gu8_1` ++ `decode/rk_line` ++ `decode/gu8_2` (4061 B, `b1354b8afc06`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** core: launch 1 of form 9 (the default) at up to `gu8_rows` (1) row, and of form 8 at every row count,
  where `gu8Admitted` holds (a threadgroup's K span 128-aligned, and its threadgroup memory within 32 KB;
  DeepSeek-V4.1's 5120-wide hidden at two rows would need 43,008 bytes, so its prepare runs in `prep2` there);
  `root.moe` / `moeWithShared` / `root.moeClamped`. The fused shared expert runs `turboexl3_gu8s` (this text after the
  shared head) at one row instead. Qwen3.8 one-row calls through `moe` (outside the fused entries); DeepSeek-V4.1
  one-row calls through `moeClamped`.
- **Shapes and dtypes:** x T [rows, HID] (bf16, f16 or f32); slots i32 or u32 [RK]; gate_suh, up_suh f16 [E, HID];
  gate_t, up_t U16 [E, HID/16, INTER/16, NHW]. Out: zg, zu f32 [KS, RK, INTER] (`gu3`'s planes); xsg, xsu f32 [RK,
  HID/128] (`prep2`'s block sums, written by the blk 0 threadgroups).
- **Grid and threadgroup:** as `gu3`: `((INTER/(16·BT))·128, RK, z)` / `(128, 1, 1)` (form 9 at one row: BT `bt1` 2, z =
  KS).
- **Template args:** `gu3`'s (NHW, CODEBOOK, WBITS, HID, INTER, RKC, KS, SPT, BT, GROUP, DEFER), then TOPK (the row of a
  member). Every admitted rate, codebook and window.
- **Replaces:** `turboexl3_prep2` → `turboexl3_gu3` (2 launches, 2 levels; unfused: `turboexl3_prep` × 2,
  `turboexl3_gemv` × 2). Form 9 at one row is 4 launches in 4 levels (`gu8` → `fin7` → `down3` → `dfin7`).
- **Bits:** byte-identical to `prep2` + `gu3`: prep2's arithmetic over the threadgroup's span into threadgroup memory,
  then gu3's GEMV on it, the same planes.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: decode chain forms 7, 8 and 9 equal form 3 byte for byte, at
  any tiles per threadgroup (TURBOEXL3_GPU=1)" (GPU, bytes against form 3 where gu8 is admitted (form 8, form 9 at one
  row), at several BT and K divisors); `src/core/gpu_test.zig` "turboexl3 gpu: root.moe at one decode row on the host's
  unreshaped operands gives the reshaped chain's bytes (TURBOEXL3_GPU=1)" (GPU, bytes: form 9 at one row on the
  unreshaped operands); `src/core/decode.zig` "turboexl3 decode forms 7 and 8 texts: every template constant read is
  passed, every signature name is used" (CPU: constants and names).

### `turboexl3_prep2`

<!-- kernel-facts: turboexl3_prep2 -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:341` (`const Kernels3`) |
| inputs | `x`, `slots`, `gate_suh`, `up_suh` |
| outputs | `xg`, `xu`, `sg`, `su` |
| source | `text_prep2` = `decode/prep2_1` ++ `decode/expert_line` ++ `decode/prep2_row` ++ `decode/prep2_2` (1991 B, `b0711f523a54`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: launch 1 of form 3's chain, made by every `decode.Kernels3`. `root.moe` (and `moeWithShared`) and
  `root.moeClamped` at 1 to 16 decode rows run it in form 9 (the default) from two rows, and at one row where
  `gu8Admitted` fails; in forms 3 and 7 at every row count; in form 8 where `gu8` is not admitted. `moeFusedShared` runs
  it as the routed prepare at 2 to 4 rows (rows past `gu8_rows`); `moeFusedSharedRouted` folds it into
  `turboexl3_route_prep` / `turboexl3_route_prepg`. Qwen3.8 at the decode rows its fused shared-expert entries do not
  take (more than 4 rows, or a call they refuse: `moeFusedShared` leaves the caller its chain), and Qwen3.8's fused
  shared-expert call at 2 to 4 rows; DeepSeek-V4.1 through `moeClamped` (its streamed decode over an expert streamer
  runs the banked texts of `moeClampedBanked` instead).
- **Shapes and dtypes:** x T [rows, HID] (T bf16, f16 or f32, as the host holds it); slots i32 or u32 [RK] (RK = rows ·
  TOPK row-slots); gate_suh, up_suh f16 [E, HID]. Out: xg, xu f16 [RK, HID] (each row-slot's prepared gate and up input,
  in lane order within each 16-wide tile); sg, su f32 [RK, HID/128] (each 128-block's sum of the rounded prepared input,
  for the deferred MUL1 map).
- **Grid and threadgroup:** `((HID/128)·32, RK, 1)` / `(32, 1, 1)`: one simdgroup per (128-wide block of x, row-slot).
- **Template args:** DIM (= HID, the input width), TOPK (the row of row-slot rk is rk / TOPK). Reads no trellis, so it
  is the same text for every rate, codebook and window.
- **Replaces:** `turboexl3_prep` × 2 of the unfused chain (`unfused.Steps`: the gate and up prepares, 2 launches, 1
  level). It is the base of form 3's chain (`prep2` → `gu3` → `fin3` → `down3` → `dfin3`).
- **Bits:** within tolerance of `format.project` (through `ref.moeRowDec`), as a launch of form 3's chain: it rounds
  where `format.project` rounds but sums in form 3's fixed split-K order. The bars: `root.moe` (form 9, form 3's bytes)
  in bf16 within 1.5e-2 (`ref.compare`); form 3 at top-1 to top-3 within 5e-3 beyond the output store's rounding
  (`ref.compareRounded`, 2^-11 for f32, 2^-8 + 2^-11 for bf16); clamped (DeepSeek-V4.1's geometry, form 9) within 5e-3
  of `ref.moeRowClamped` (f32 `ref.compare`, bf16 `ref.compareRounded` at 2^-8). A row's bytes do not depend on how many
  rows share the call (the K partition is the geometry's alone). `turboexl3_route_prep` and `turboexl3_route_prepg` are
  held to its bytes.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: decode chain (form 3), a row's bytes are the same alone and
  among 2..16 rows (TURBOEXL3_GPU=1)" (GPU, bytes: each row alone against rows 2 to 16 together, MUL1's deferred map on
  and off); `src/core/gpu_test.zig` "turboexl3 gpu: root.moe (the sushi_exl3-shaped entry) equals the reference through
  the bf16 host operands (TURBOEXL3_GPU=1)" (GPU, within 1.5e-2 in bf16: 3 rows at top-10, 4 rows at top-1);
  `src/core/topk_test.zig` "turboexl3 gpu topk: root.moe at top-1, 2 and 3 gives form 3's bytes under decode forms 7, 8
  and 9, ..." (GPU: form 3 within 5e-3 of the reference at top-1 to top-3); `src/core/gpu_test.zig` "turboexl3 gpu: the
  shared expert fused into form 9's launches equals the host's chain byte for byte (TURBOEXL3_GPU=1)" (GPU, bytes:
  `moeShared` with this prepare at 2 to 4 rows against the host's chain); `src/core/gpu_test.zig` "turboexl3 gpu: the
  router folded into form 9's prepare launch (and the shared gate's logit with it) ..." (GPU, bytes:
  `turboexl3_route_prep` against the router kernel + this launch at `cfgs9`'s config); `src/core/decode.zig` "turboexl3
  decode router-folded prepare texts: the merged launch carries the router's and prep2's ..." (CPU: prep2's body lands
  verbatim in the merged texts); `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode texts: each is its
  base text with the bank's lines in place of the expert line ..." (CPU: `turboexl3_prep2b` is this text with the expert
  line swapped).

### `turboexl3_red5`

<!-- kernel-facts: turboexl3_red5 -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:651` (`const Kernels5`) |
| inputs | `wv`, `scores` |
| outputs | `y` |
| source | `text_red5` = `decode/red5` (391 B, `71b235db1931`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** core: launch 3 of form 5 (`decode.Kernels5`), `root.moe` / `moeWithShared` under `Config.decode_form` 5.
  Not served: the default is form 9 (form 6 folds this launch into `dn6`).
- **Shapes and dtypes:** wv f16 [RK, HID] (`dn5`'s); scores S [RK] (bf16, f16 or f32). Out: y OUT_T [rows, HID].
- **Grid and threadgroup:** `(HID, rows, 1)` / `(128, 1, 1)`: one thread per output.
- **Template args:** HID, TOPK, OUT_T. No trellis read.
- **Replaces:** the score-weighted sum of `turboexl3_dfin3` (unfused: `turboexl3_reduce`).
- **Bits:** byte-identical to form 3: dfin3's `fma` chain over the f16-rounded slot outputs, in slot order.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: decode chain form 5 equals form 3 byte for byte
  (TURBOEXL3_GPU=1)" (GPU, bytes: form 5 against form 3); `src/core/gpu_test.zig` "turboexl3 gpu: root.moe at one decode
  row on the host's unreshaped operands gives the reshaped chain's bytes (TURBOEXL3_GPU=1)" (GPU, bytes: form 5 at one
  row, bf16); `src/core/decode.zig` "turboexl3 decode form 5 texts: every template constant read is passed, every
  signature name is used" (CPU: names).

### `turboexl3_fin7co`

<!-- kernel-facts: turboexl3_fin7co -->

| fact | from the code |
|---|---|
| made at | `src/core/decode.zig:352` (`const Kernels3`) |
| inputs | `zg`, `zu`, `slots`, `gate_svh`, `up_svh`, `down_suh`, `xsg`, `xsu` |
| outputs | `hd`, `hs` |
| source | `text_fin7co` = `decode/fin7_1` ++ `decode/expert_line` ++ `decode/fin7_2` ++ `decode/fin7_planes` ++ `decode/fin7_3` ++ `decode/fin7_sums` ++ `decode/fin7_4` ++ `decode/fin7_swiglu_clamp_out` ++ `decode/fin7_5` (2561 B, `e9f1d528ba34`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moeLing` at 1 to 16 decode rows with a limit (launch 3 of forms 7, 8 and 9 as configured,
  form 9 for forms 3, 5 and 6): a geometry with `limit` 1 to 128 and `clamp` `silu_out` builds it in `fin7`'s place.
  Ling-3.1-flash (`models/ling/api.zig`): its routed experts on layers 65 to 71 (`expert_swiglu_limit_list` 10), whose
  SwiGLU is clamped after the SiLU: min(silu(g), L) · clamp(u, −L, L), exllamav3's `act_limit` with SiLU (inclusionAI's
  vLLM and SGLang the same).
- **Shapes and dtypes:** as `turboexl3_fin7c`.
- **Grid and threadgroup:** as `fin7`: `((INTER/128)·32, RK, 1)` / `(32, 1, 1)`.
- **Template args:** fin7's five (INTER, RKC, KS, DEFER, NBH), then LIMIT (an integer 1 to 128, Ling-3.1's 10): the
  gate's SiLU clamped from above at LIMIT and up to [−LIMIT, LIMIT], in f32 on the finished f16 values; a NaN passes
  through.
- **Replaces:** `turboexl3_fin7`'s launch with Ling's clamp (the unfused chain has no clamp).
- **Bits:** byte-identical to `turboexl3_fin7` where no value reaches the limit; with the clamp acting, within 5e-3 of
  `ref.moeRowClampedAt` at `silu_out` (f32 `ref.compare`; bf16 `ref.compareRounded` at 2^-8) at Ling-3.1-flash's routed
  geometry (4096 x 1280, 512 experts, top-8), at least 1% of the gates' SiLUs and of the ups past the limit.
- **Pinned by:** `src/models/ling/ling_test.zig` "turboexl3 ling texts: each Ling kernel is its base text with the
  SwiGLU line swapped for Ling's clamp ..." (CPU: this text is its base with the one swap, nothing else; only a clamp
  reads LIMIT); `src/models/ling/ling_test.zig` "turboexl3 gpu ling: root.moeLing at Ling-3.1-flash's routed geometry
  ..." (GPU: at limit 128 with x ~ N(0, 1) `moe`'s bytes at rows 1, 16 and 400, bf16 and f32; at limits 0 and 10 with x
  ~ N(0, 8^2) within 5e-3 of `ref.moeRowClampedAt` at `silu_out` on sampled rows of 1, 4, 16, 32 and 1328, n 40 and 32).

### `turboexl3_dense_dec0`

<!-- kernel-facts: turboexl3_dense_dec0 -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:405` (`const Dense`) |
| inputs | `xp`, `xs`, `w`, `svh`, `cnt` |
| outputs | `y`, `z` |
| source | `text_dec0` = `dense/dec_head` ++ `dense/dec_gemv0` ++ `dense/dec_gemv_arrive_1` ++ `dense/dec_tile` ++ `dense/dec_gemv_arrive_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` (6206 B, `1f49ed16dcfc`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.linear` (and every entry built on it) at 1 to `dec_rows` (4) decode rows with
  `Config.dec_one` on, where the prepare does not run inside the launch: 2 to 4 rows at OUT > `dec_prep_out` (1024), or
  any DEC-ONE call where `Dense.decPrepFits` fails; `turboexl3_dense_prep_sc` prepares the rows first. Qwen3.8's decode
  merges `decodeMergeGated` and `decodeMergeQmergeGated` (`decb.preparedRows`) also feed it rows and zeroed counters
  from their own prepare launches. Every dense EXL3 linear: Qwen3.8's trunk, shared expert, `lm_head` and MTP;
  DeepSeek-V4.1's trunk; Ripple's projections and head.
- **Shapes and dtypes:** xp f16 [rows, IN] in lane order and xs f32 [rows, IN/128] (from `turboexl3_dense_prep_sc`); w
  the trellis U16 or I16 [IN/16, OUT/16, NHW]; svh f16 [OUT]; cnt u32 [max(OUT/128 · chunks, 8)], zeroed by the prepare.
  Out: y OUT_T [rows, OUT] (OUT_T = x's dtype T: bf16, f16 or f32); z f32 [KS · rows, OUT] (the partial planes,
  scratch).
- **Grid and threadgroup:** `((OUT/16/BT) · 128, chunks, KS)` / `(128, 1, 1)`, chunks = ceil(rows / RB).
- **Template args:** NHW (even 16 to 128), CODEBOOK (MUL1, MCG), WBITS (8 to 16), IN, OUT, KS (K split), BT
  (`Dense.btDec`: `bt_set`, else 4 at RB 1 and 2 at RB 2 to 4), RB (`Dense.rbFor`: one chunk up to 4 rows), ROWS, DEFER
  (the deferred MUL1 map), OUT_T: `turboexl3_dense_dec1`'s list. Every rate, codebook and window.
- **Replaces:** `turboexl3_dense_dgemv` -> `turboexl3_dense_dfinish` (2 launches, 2 levels); with
  `turboexl3_dense_prep_sc` before it, the three-launch chain `prep_s` -> `dgemv` -> `dfinish` (3 levels) becomes 2
  launches and 2 levels.
- **Bits:** byte-identical to that chain: its GEMV and finish lines are `text_dgemv`'s and `text_dfinish`'s, the planes
  are summed in slice order, and an integer arrival counter (no float atomics) picks the threadgroup that runs the
  finish.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: the one-launch decode linear (DEC-ONE) equals the three-launch
  chain byte for byte (TURBOEXL3_GPU=1)" (GPU, bytes: the "own_launch" arm, `dec_prep_rows` and `dec_prep_out` 0 and
  `dec_rows` raised to 16, at rows 1 to 9, 11 and 16, against the chain and each row alone); `src/core/dense.zig`
  "turboexl3 dense one-launch decode (DEC-ONE) texts: every value-setting line is the three-launch chain's, ..." (CPU,
  the texts, arrival fences and signature names); `src/core/gpu_test.zig` "turboexl3 gpu: dense.linear equals
  format.project per row at the trunk's rates, GEMV and prefill paths (TURBOEXL3_GPU=1)" and
  `src/models/deepseek_v41/deepseek_test.zig` "turboexl3 gpu deepseek: root.linear and root.linearBias equal the host
  reference at every DeepSeek-V4.1 ..." (GPU, within 5e-3 of the reference at the rows it takes);
  `src/models/qwen38/decb_test.zig` "turboexl3 gpu decode merges: o_proj's prepare inside the gate kernel and inside the
  QSA merge equals the ..." (GPU, bytes through Qwen3.8's gated merges).

### `turboexl3_dense_dec1`

<!-- kernel-facts: turboexl3_dense_dec1 -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:406` (`const Dense`) |
| inputs | `x`, `suh`, `w`, `svh`, `cnt` |
| outputs | `y`, `z`, `xsx` |
| source | `text_dec1` = `dense/dec_head` ++ `dense/dec_prep1` ++ `dense/dec_gemv1` ++ `dense/dec_gemv_arrive_1` ++ `dense/dec_tile` ++ `dense/dec_gemv_arrive_2` ++ `dense/dec_xsum1` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` (7726 B, `d4555320479f`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.linear` (and every entry built on it) at 1 to `dec_rows` (4) decode rows with
  `Config.dec_one` on, the prepare inside the launch. Every dense EXL3 linear: Qwen3.8's trunk, shared expert,
  `lm_head` and MTP; DeepSeek-V4.1's trunk; Ripple's projections and head.
- **Shapes and dtypes:** x T [rows, IN] (T bf16, f16 or f32); suh f16 [IN]; w the trellis U16 or I16 [IN/16, OUT/16,
  NHW]; svh f16 [OUT]; cnt u32 [max(OUT/128 · chunks, 8)] from `turboexl3_dense_dec_zero`. Out: y OUT_T [rows, OUT];
  z f32 [KS · rows, OUT] (the partial planes, scratch); xsx f32 (the published block sums, scratch).
- **Grid and threadgroup:** `((OUT/16/BT) · 128, chunks, KS)` / `(128, 1, 1)`.
- **Template args:** NHW (even 16 to 128), CODEBOOK (MUL1, MCG), WBITS (8 to 16), IN, OUT, KS (K split), BT (output
  tiles per threadgroup: 1, 2, 4 or 8), RB (rows per chunk, `Dense.rbFor`), ROWS, DEFER (the deferred MUL1 map),
  OUT_T. Taken when rows <= `dec_prep_rows` (1) or OUT <= `dec_prep_out` (1024) and `Dense.decPrepFits`; otherwise
  `turboexl3_dense_dec0` after `turboexl3_dense_prep_sc`. Every rate, codebook and window.
- **Replaces:** `turboexl3_dense_prep_s` -> `turboexl3_dense_dgemv` -> `turboexl3_dense_dfinish` (3 launches, 3
  levels), with `turboexl3_dense_dec_zero` beside the previous op (it waits on nothing): 1 dependent level.
- **Bits:** byte-identical to that chain.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: the one-launch decode linear (DEC-ONE) equals the three-launch
  chain byte for byte (TURBOEXL3_GPU=1)" (GPU, bytes); `src/core/dense.zig` "turboexl3 dense one-launch decode
  (DEC-ONE) texts: every value-setting line is the three-launch chain's, ..." (CPU, the texts).

### `turboexl3_dense_dec_zero`

<!-- kernel-facts: turboexl3_dense_dec_zero -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:408` (`const Dense`) |
| inputs | `svh` |
| outputs | `cnt` |
| source | `text_dec_zero` = `dense/dec_zero` (83 B, `c4a6f024bb78`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.linear` (and every entry built on it) on DEC-ONE's calls whose prepare runs inside the GEMV
  launch (`turboexl3_dense_dec1`): 1 row, or up to `dec_rows` (4) rows at OUT <= 1024, where `Dense.decPrepFits` holds
  (`Config.dec_one`). Qwen3.8's decode merges `decodeMergeQPlanes` and `decodeMergeIndexer` launch it before their own
  DEC-ONE variants. Every dense EXL3 linear: Qwen3.8's trunk, shared expert, `lm_head` and MTP; DeepSeek-V4.1's trunk;
  Ripple's projections and head.
- **Shapes and dtypes:** svh f16 [OUT], an input for ordering only (a weight, so the launch waits on nothing; the text
  never reads it). Out: cnt u32 [NCNT], NCNT = max(OUT/128 · chunks, 8) (at least 8: MLX binds a smaller input in the
  constant address space, which has no atomics).
- **Grid and threadgroup:** `(ceil(NCNT/128) · 128, 1, 1)` / `(128, 1, 1)`.
- **Template args:** NCNT. It reads no weight, so every rate, codebook and window.
- **Replaces:** nothing: it is DEC-ONE's counter zeroing, a launch beside the previous op, so `turboexl3_dense_dec1`
  stays 1 dependent level.
- **Bits:** writes zeros only; y's bytes are `turboexl3_dense_dec1`'s, the three-launch chain's.
- **Pinned by:** `src/core/dense.zig` "turboexl3 dense one-launch decode (DEC-ONE) texts: every value-setting line is
  the three-launch chain's, ..." (CPU: its signature names, svh the one input it may leave unread, no MLX shape
  buffers); `src/core/gpu_test.zig` "turboexl3 gpu: the one-launch decode linear (DEC-ONE) equals the three-launch chain
  byte for byte (TURBOEXL3_GPU=1)" (GPU, through the "inside" arm's bytes).

### `turboexl3_dense_dequant`

<!-- kernel-facts: turboexl3_dense_dequant -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:384` (`const Dense`) |
| inputs | `w` |
| outputs | `W` |
| source | `text_dequant` = `dense/dequant` (1463 B, `022fd8ffb1d9`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: `Dense.dequantize` / `Dense.decodeTrellis` on the matmul path of `root.linear` and `linearSplit`
  (on the side stream) at prefill rows past `small_rows` (32), or past 16 with `Config.nax` false; also
  `linearInnerSplit`, `linearPiecesBegin` (once for every piece of a call), Qwen3.8's ATTN-IN and ATTN-OUT
  (`attn_io.prefillInner`), and at load the resident trunk's first call per admitted trellis (`dense.residentArm`) and
  `trunkUnrot` (before `turboexl3_unrot_cols`). A kept resident W_hat skips it after that first call; an un-rotated
  linear's calls past 32 rows never run it. Every dense EXL3 linear at prefill rows: Qwen3.8's trunk and MTP;
  DeepSeek-V4.1's trunk, vision tower and aligner (and `wo_a`'s slices at 17+ rows through `linearGrouped`); Ripple's
  projections.
- **Shapes and dtypes:** w the trellis U16 or I16 [IN/16, OUT/16, NHW]. Out: W f16 [IN, OUT] (W_hat).
- **Grid and threadgroup:** `(ceil((IN/16) · (OUT/16) / 4) · 128, 1, 1)` / `(128, 1, 1)`: one simdgroup per 16 x 16
  tile, four tiles per threadgroup.
- **Template args:** NHW, CODEBOOK, WBITS, IN, OUT. Every rate, codebook and window.
- **Replaces:** nothing: it is the base of the dequantize + MLX matmul path (`turboexl3_dense_prep` and `dequant` -> MLX
  matmul -> `turboexl3_dense_finish`).
- **Bits:** each weight's f16 value from its own tile (no reduction); the matmul path on it is within 5e-3 of
  `format.project` per row with f32 x and y and within 2.5e-2 with bf16 (`ref.compare`). The resident W_hat is its
  output kept, so the same bytes.
- **Pinned by:** `src/core/dense.zig` "turboexl3 dense texts: inputs read, outputs written, the dequant layout names the
  GEMV's row/column rule" (CPU: the layout lines naming the GEMV's row/column rule); `src/core/gpu_test.zig` "turboexl3
  gpu: dense.linear equals format.project per row at the trunk's rates, GEMV and prefill paths (TURBOEXL3_GPU=1)" (GPU,
  rows 40, 129 and 200; bf16 at 40); `src/models/deepseek_v41/deepseek_test.zig` "turboexl3 gpu deepseek: root.linear
  and root.linearBias equal the host reference at every DeepSeek-V4.1 ..." (GPU, 512 rows); `src/core/gpu_test.zig`
  "turboexl3 gpu: root.linear on real trunk trellises from the full pack equals format.project (TURBOEXL3_REAL_SHARD)"
  (GPU, 40 rows of real trellises); `src/core/gpu_test.zig` "turboexl3 gpu resident trunk: a resident W_hat gives the
  per-call dequantize's bytes on every trunk linear kind, ..." (GPU, bytes: kept against per call).

### `turboexl3_dense_dfinish`

<!-- kernel-facts: turboexl3_dense_dfinish -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:387` (`const Dense`) |
| inputs | `z`, `svh`, `xs` |
| outputs | `y` |
| source | `text_dfinish` = `dense/dfinish_v` ++ `dense/dfinish_store` (1452 B, `d89e31ec36d1`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: the finish of the multi-row GEMV chain: `root.linear` (and every entry built on it) at 5 to 16
  decode rows (1 to 16 with `Config.dec_one` off), F3's down linear (`Dense.linearPrepared` in `root.sharedSwiglu`), and
  Qwen3.8's decode merges where DEC-ONE does not take the call (`decb.preparedRows`). Every dense EXL3 linear: Qwen3.8's
  trunk, shared expert, `lm_head` and MTP; DeepSeek-V4.1's trunk; Ripple's projections and head.
- **Shapes and dtypes:** z f32 [KS · rows, OUT] (the GEMV's planes); svh f16 [OUT]; xs f32 [rows, IN/128] (`prep_s`'s
  block sums, read when DEFER). Out: y OUT_T [rows, OUT] (OUT_T = x's dtype T).
- **Grid and threadgroup:** `((OUT/128) · 32, rows, 1)` / `(32, 1, 1)`: one simdgroup per (128-wide output block, row).
- **Template args:** DIM (= OUT), OUT_T, KS, ROWS, INBLK (IN/128), DEFER. It reads no weight, so every rate, codebook
  and window.
- **Replaces:** nothing: it is the base of the multi-row GEMV chain (`turboexl3_dense_prep_s` -> `dgemv` -> `dfinish`).
- **Bits:** its own: the planes summed in slice order, the deferred map `A · acc + C · Σx`, f16, H128, svh, f16, OUT_T;
  its chain is within 5e-3 of `format.project` per row with f32 x and y and within 2.5e-2 with bf16 (`ref.compare`).
- **Pinned by:** `src/core/dense.zig` "turboexl3 dense one-launch decode (DEC-ONE) texts: every value-setting line is
  the three-launch chain's, ..." and `src/models/deepseek_v41/f3_test.zig` "turboexl3 f3u: the fused shared-expert text
  is text_dfinish's finishes, MLX's clamped SwiGLU and text_prep_s's prepare" (CPU: its finish lines, which DEC-ONE's
  and F3's texts must carry); `src/core/gpu_test.zig` "turboexl3 gpu: dense.linear equals format.project per row at the
  trunk's rates, GEMV and prefill paths (TURBOEXL3_GPU=1)" (GPU, the bars); `src/core/gpu_test.zig` "turboexl3 gpu:
  dense multi-row GEMV, a row's bytes are the same alone and among 2..16 rows (TURBOEXL3_GPU=1)" (GPU, a row's bytes
  alone and among 2 to 16 rows); `src/core/gpu_test.zig` "turboexl3 gpu: the one-launch decode linear (DEC-ONE) equals
  the three-launch chain byte for byte (TURBOEXL3_GPU=1)" (GPU, the reference arms).

### `turboexl3_dense_dgemv`

<!-- kernel-facts: turboexl3_dense_dgemv -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:385` (`const Dense`) |
| inputs | `xp`, `w` |
| outputs | `z` |
| source | `text_dgemv` = `dense/dgemv_1` ++ `dense/dgemv_x` ++ `dense/dgemv_2` ++ `dense/dgemv_m` ++ `dense/dgemv_fold` ++ `dense/dgemv_store` (3314 B, `e1f321adab55`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: the multi-row GEMV chain at decode rows DEC-ONE does not take (5 to 16; 1 to 16 with
  `Config.dec_one` off), for chunks of up to `max_rb` (5) rows. With `Config.dense_lean` on: rows 1 to 5, 9 and 10,
  except chunks of 5 in a launch of at least 2048 threadgroups at BT 1 under `Config.dense_lean5` (those take
  `turboexl3_dense_dgemv_lean`); with it off, every chunk (`Dense.rbFor`: 6 = 3 + 3, 16 = 4 x 4). Through `root.linear`,
  F3 (`Dense.prepareGemv` and `linearPrepared` in `root.sharedSwiglu`) and Qwen3.8's decode merges. Every dense EXL3
  linear: Qwen3.8's trunk, shared expert, `lm_head` and MTP; DeepSeek-V4.1's trunk; Ripple's projections and head.
- **Shapes and dtypes:** xp f16 [rows, IN] in lane order (from `prep_s`); w the trellis U16 or I16 [IN/16, OUT/16, NHW].
  Out: z f32 [KS · rows, OUT] (K slice ks's partial sums of row r at row ks · ROWS + r).
- **Grid and threadgroup:** `((OUT/16/BT) · 128, chunks, KS)` / `(128, 1, 1)`, chunks = ceil(rows / RB).
- **Template args:** NHW (even 16 to 128), CODEBOOK (MUL1, MCG), WBITS (8 to 16), IN, OUT, KS (K split, `Dense.ksFor`:
  the smallest split that puts 160 threadgroups in flight at one row, fixed per shape), BT (output tiles per
  threadgroup: 1, 2, 4 or 8), RB (rows per chunk), ROWS, DEFER (the deferred MUL1 map, MUL1 only). BT is `Dense.btAt`'s:
  `bt_set`, else `btFor` (4 at RB 1, 2 at RB 2 to 4, 1 at 5); RB is `Dense.chainRb`'s. Every rate, codebook and window.
- **Replaces:** the row-slot GEMV `turboexl3_dense_gemv`, which decodes the trellis once per row: this one decodes each
  weight once per threadgroup for a whole chunk.
- **Bits:** within tolerance as part of the chain: its chain is within 5e-3 of `format.project` per row with f32 x and y
  and within 2.5e-2 with bf16 (`ref.compare`). A row's bytes are the same alone and among 2 to 16 rows (its summation
  order depends on KS alone).
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: dense.linear equals format.project per row at the trunk's
  rates, GEMV and prefill paths (TURBOEXL3_GPU=1)" (GPU, the bars at rows 1 to 16); `src/core/gpu_test.zig` "turboexl3
  gpu: dense multi-row GEMV, a row's bytes are the same alone and among 2..16 rows (TURBOEXL3_GPU=1)" (GPU, bytes per
  row); `src/core/gpu_test.zig` "turboexl3 gpu: the one-launch decode linear (DEC-ONE) equals the three-launch chain
  byte for byte (TURBOEXL3_GPU=1)" (GPU, the "chain_plain" reference arm); `src/core/dense.zig` "turboexl3 dense lean
  GEMV text: text_dgemv but its input loads and codeword loop, every accumulator's FMAs in its order" and
  `src/core/dense.zig` "turboexl3 dense one-launch decode (DEC-ONE) texts: every value-setting line is the three-launch
  chain's, ..." (CPU, the text's pieces and lines); `src/core/dense.zig` "turboexl3 dense: the served plan per shape
  (DEC-ONE from the config, K split, row tiling, MUL1's deferred map)" (CPU, `rbFor`, `btFor`, `ksFor`).

### `turboexl3_dense_dgemv_lean`

<!-- kernel-facts: turboexl3_dense_dgemv_lean -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:386` (`const Dense`) |
| inputs | `xp`, `w` |
| outputs | `z` |
| source | `text_dgemv_lean` = `dense/dgemv_1` ++ `dense/dgemv_2` ++ `dense/dgemv_j` ++ `dense/dgemv_fold` ++ `dense/dgemv_store` (3379 B, `dfc748cf15e0`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: the chain's chunks past `max_rb` with `Config.dense_lean` (on by default): chunks of 6 to 8 rows
  (`Dense.chainRb` under `lean_rb` 8: rows 6 to 8 in one chunk, 11 to 16 in two), and chunks of 5 rows (rows 5, 9, 10)
  in a launch of at least `lean_bt2_threadgroups` (2048) threadgroups at BT 1 with `Config.dense_lean5`
  (`Dense.wideLean`; the config names DeepSeek-V4.1's LM head, wq_b and wo_a there). Through `root.linear`, F3 and
  Qwen3.8's decode merges. Every dense EXL3 linear: Qwen3.8's trunk, shared expert, `lm_head` and MTP; DeepSeek-V4.1's
  trunk; Ripple's projections and head.
- **Shapes and dtypes:** the same as `turboexl3_dense_dgemv`: xp f16 [rows, IN] in lane order; w the trellis U16 or I16
  [IN/16, OUT/16, NHW]. Out: z f32 [KS · rows, OUT].
- **Grid and threadgroup:** `((OUT/16/BT) · 128, chunks, KS)` / `(128, 1, 1)`; BT is `bt_set`'s, else 2 where
  `Dense.wideLean` holds, else 1.
- **Template args:** NHW (even 16 to 128), CODEBOOK (MUL1, MCG), WBITS (8 to 16), IN, OUT, KS (K split, `Dense.ksFor`:
  the smallest split that puts 160 threadgroups in flight at one row, fixed per shape), BT (output tiles per
  threadgroup: 1, 2, 4 or 8), RB (rows per chunk), ROWS, DEFER (the deferred MUL1 map, MUL1 only). RB 5 to 8. Every
  rate, codebook and window.
- **Replaces:** `turboexl3_dense_dgemv` over two chunks of the same rows, each decoding every weight again: one chunk of
  up to 8 rows decodes each weight once.
- **Bits:** byte-identical to `turboexl3_dense_dgemv`: every accumulator gets its FMAs in its order on the same
  operands.
- **Pinned by:** `src/core/dense.zig` "turboexl3 dense lean GEMV text: text_dgemv but its input loads and codeword loop,
  every accumulator's FMAs in its order" (CPU: `dgemv`'s text with only the input loads and the codeword loop replaced);
  `src/core/gpu_test.zig` "turboexl3 gpu: dense multi-row GEMV, a row's bytes are the same alone and among 2..16 rows
  (TURBOEXL3_GPU=1)" (GPU, bytes: lean on and off at 2 to 9, 11 and 16 rows against each row alone);
  `src/core/gpu_test.zig` "turboexl3 gpu: the one-launch decode linear (DEC-ONE) equals the three-launch chain byte for
  byte (TURBOEXL3_GPU=1)" (GPU, bytes: the lean "chain" arm against the plain chain and DEC-ONE);
  `src/core/gpu_test.zig` "turboexl3 gpu: dense.linear equals format.project per row at the trunk's rates, GEMV and
  prefill paths (TURBOEXL3_GPU=1)" and `src/models/deepseek_v41/deepseek_test.zig` "turboexl3 gpu deepseek: root.linear
  and root.linearBias equal the host reference at every DeepSeek-V4.1 ..." (GPU, the bars at 8 and 16 rows).

### `turboexl3_dense_finish`

<!-- kernel-facts: turboexl3_dense_finish -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:383` (`const Dense`) |
| inputs | `z`, `svh` |
| outputs | `y` |
| source | `text_finish` = `dense/finish` (854 B, `73f767f084a6`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: `linear`'s finish at prefill rows: after the register path (17 to `small_rows`, 32), dequantize +
  MLX matmul (past 32, unless pg2 serves the call) and the fused GEMM (tests only); `Dense.finishOf`
  (`root.linearPrepared`, `linearFinish`, Qwen3.8's attention paths over a prepared copy) and Qwen3.8's ATTN-OUT
  (`root.linearGated` launches it with `linear`'s finish config). At decode rows only the row-slot reference
  (`Dense.rowslot`, not served). Every dense EXL3 linear at prefill rows: Qwen3.8's trunk and MTP; DeepSeek-V4.1's
  trunk, vision tower and aligner (and `wo_a`'s slices at 17+ rows through `linearGrouped`); Ripple's projections.
- **Shapes and dtypes:** z [rows, OUT]: f16 from the prefill inner products, f32 from the row-slot GEMV; svh f16 [OUT].
  Out: y OUT_T [rows, OUT] (x's dtype T: bf16, f16 or f32).
- **Grid and threadgroup:** `((OUT/128) · 32, rows, 1)` / `(32, 1, 1)`: one simdgroup per (128-wide output block, row).
- **Template args:** DIM (= OUT), OUT_T. It reads no weight, so every rate, codebook and window.
- **Replaces:** nothing: it is the base finish of the prefill paths and of the row-slot chain.
- **Bits:** `format.project`'s finish rounding points (f16 z, H128, svh, f16, then OUT_T); its paths are within 5e-3 of
  `format.project` per row with f32 x and y and within 2.5e-2 with bf16.
- **Pinned by:** `src/core/dense.zig` "turboexl3 dense texts: inputs read, outputs written, the dequant layout names the
  GEMV's row/column rule" (CPU, its y store); `src/core/pgemm.zig` "pgemm: the epilogue runs the finish's statements,
  the loop is MLX's gemm_loop by its statements, ..." (CPU: pg2's epilogue carries its 10 lines); `src/core/dense.zig`
  "turboexl3 dense un-rotated trunk texts: the un-rotation kernels repeat the dense butterfly verbatim, ..." (CPU, its
  butterfly lines); `src/core/gpu_test.zig` "turboexl3 gpu: dense.linear equals format.project per row at the trunk's
  rates, GEMV and prefill paths (TURBOEXL3_GPU=1)" (GPU, rows 17 to 200); `src/core/gpu_test.zig` "turboexl3 gpu:
  dense.linearOn across two streams gives linear's bytes on both prefill paths (TURBOEXL3_GPU=1)" (GPU, bytes on two
  streams).

### `turboexl3_dense_finish_pieces`

<!-- kernel-facts: turboexl3_dense_finish_pieces -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:855` (`const Dense`) |
| inputs | `zp0`, `zp1`, `zp2`, `zp3`, `svh`, `R1`, `R2`, `R3` |
| outputs | `y` |
| source | `text_finish_pieces` = `dense/finish_pieces` (1102 B, `577ca60ab2f0`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.linearPiecesFinish` (`Dense.finishPieces`, made on first use): one finish over a call cut
  into 2 to 4 row pieces, every piece and the whole call on the matmul path (`Dense.piecesServe`), for the host's
  overlap of Qwen3.8's GDN layers (the plugin table's `linear_pieces`).
- **Shapes and dtypes:** zp0, zp1, zp2, zp3 f16 [rows_p, OUT] (each piece's inner product; an unused piece is bound to
  zp0); svh f16 [OUT]; R1, R2, R3 0-dim int arrays (each piece's first row in the call, an unused piece's bound at the
  call's rows). Out: y OUT_T [rows, OUT], rows the pieces' sum.
- **Grid and threadgroup:** `((OUT/128) · 32, rows, 1)` / `(32, 1, 1)`: `turboexl3_dense_finish`'s launch at the call's
  rows.
- **Template args:** DIM (= OUT), OUT_T.
- **Replaces:** nothing: it is `turboexl3_dense_finish` over row pieces, each row reading its own piece's row, one
  finish for the whole call.
- **Bits:** byte-identical to the one-call finish (`text_finish`'s statements after the piece select). With `pieceInner`
  per piece the result is `linear` of the concatenated rows bit for bit wherever MLX's matmul runs the whole call's
  kernel at every piece's row count (the caller's check, the host's `gdnOverlapGemmSame`).
- **Pinned by:** `src/core/chains_test.zig` "turboexl3 gpu linear pieces: the pieces' one finish gives the one-call
  finish's words over their joined inner products (TURBOEXL3_GPU=1)" (GPU: 2, 3 and 4 pieces at 2560 -> 6144, word for
  word against `root.linearFinish` of the joined inner products; against `linear` of the joined rows reported).

### `turboexl3_dense_gemv`

<!-- kernel-facts: turboexl3_dense_gemv -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:382` (`const Dense`) |
| inputs | `xp`, `slots`, `w` |
| outputs | `z` |
| source | `unfused.text_gemv` = `unfused/gemv` (3087 B, `eeb649b8f300`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core, not served: `Dense.linearOn` at decode rows with `Dense.rowslot` set (false by default; DEC-ONE
  does not take row-slot calls), between `turboexl3_dense_prep` (lane order) and `turboexl3_dense_finish`. Only the
  dense timing test sets it.
- **Shapes and dtypes:** xp f16 [rows, IN] in lane order; slots u32 [rows] (all zero: each row is a row-slot of expert
  0); w the trellis U16 or I16 [IN/16, OUT/16, NHW]. Out: z f32 [rows, OUT] (unrounded inner sums).
- **Grid and threadgroup:** `((OUT/128) · 128, rows, 1)` / `(128, 1, 1)`: one threadgroup per (128-wide output block,
  row).
- **Template args:** NHW, CODEBOOK, WBITS, IN, OUT. Every rate, codebook and window.
- **Replaces:** nothing: it is the GEMV that `turboexl3_dense_dgemv` replaces (one trellis decode per row).
- **Bits:** its own summation order (the four simdgroups split the input tiles); no test holds its values to a bar.
- **Pinned by:** `src/core/unfused.zig` "turboexl3 unfused texts: the inputs each signature names are read, the outputs
  written" (CPU: its text, `unfused.text_gemv`, reads xp, slots and w and stores z); `src/core/gpu_test.zig` "turboexl3
  gpu: dense timing, multi-row vs row-slot vs affine q8 (TURBOEXL3_DENSE_TIME=1)" (GPU, runs the row-slot arm, no value
  check).

### `turboexl3_dense_pf_gemm`

<!-- kernel-facts: turboexl3_dense_pf_gemm -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:393` (`const Dense`) |
| inputs | `tiles`, `x0`, `w0`, `xs0` |
| outputs | `z0` |
| source | `prefill.gemm1_text` = `prefill/gemm_1` ++ `prefill/gemm_2` ++ `prefill/gemm_3` ++ `prefill/gemm_op1_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op1_xsz` ++ `prefill/gemm_5` (6517 B, `dd6225a4d3f1`) |
| header | `prefill.header_nax` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` (60754 B, `4c8c1fc745df`) |

<!-- /kernel-facts -->

- **Used by:** core, test only: `root.linear` at prefill rows with `Dense.fused_prefill` set (false by default, set only
  in tests), up to `pf_tiles_cap` (512) tiles of 64 rows; a longer call takes dequantize + matmul. Before it
  `turboexl3_dense_prep_s` in natural order for MUL1 (the deferred map) or `turboexl3_dense_prep` for MCG; after it
  `turboexl3_dense_finish`.
- **Shapes and dtypes:** tiles i32 [max(tmax, 2), 4] (the host-built table: expert 0, first row 64t, row count, 0; tmax
  = ceil(rows/64)); x0 f16 [rows, IN] in natural order; w0 the trellis U16 or I16 [IN/16, OUT/16, NHW]; xs0 f32 [rows,
  IN/128] (`prep_s`'s block sums; f32 [8] zeros with DEFER off). Out: z0 f16 [rows, OUT].
- **Grid and threadgroup:** `(tmax · (OUT/64) · 2 · BM, 1, 1)` / `(2 · BM, 1, 1)` with BM = `pf.bm` (64):
  `(tmax · (OUT/64) · 128, 1, 1)` / `(128, 1, 1)`.
- **Template args:** NHW, CODEBOOK, WBITS, KD (= IN), ND (= OUT), NOPS (1), BMR (64), DBUF (1), PROBE (0), WSP (0),
  MFIRST (0), UNR (1), FRAGB (1), INTR (1 where NHW % 16 == 0), DEFER (1 for MUL1), NBX (IN/128), BKS (`prefill.BK`,
  64), NCB (1), ALAY (0, natural order), H16 (0): `PfOpts.forShape`, the expert chain's served set less what the rate or
  codebook cannot take. IN and OUT multiples of 64; every even rate, both codebooks.
- **Replaces:** `turboexl3_dense_dequant` and MLX's matmul (2 launches, an f16 W_hat written and read): one launch that
  decodes the trellis per 64-deep stage into threadgroup memory.
- **Bits:** its own MMA summation order. The timing test holds it to dequantize + matmul with bf16 outputs: RMS
  difference <= 3.9e-3 of the RMS and `ref.compare` max ratio <= 8e-2. No test in the repo sets `fused_prefill` against
  `format.project` (`src/core/gpu_test.zig` "turboexl3 gpu: dense.linear equals format.project per row at the trunk's
  rates, GEMV and prefill paths (TURBOEXL3_GPU=1)" mentions it in a comment but keeps the default).
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: dense.linearOn across two streams gives linear's bytes on both
  prefill paths (TURBOEXL3_GPU=1)" (GPU, bytes across two streams at 8192, 1328 and 40 rows); `src/core/gpu_test.zig`
  "turboexl3 gpu: dense timing, multi-row vs row-slot vs affine q8 (TURBOEXL3_DENSE_TIME=1)" with
  TURBOEXL3_DENSE_TIME_PREFILL=1 (GPU, the bars above at 1359, 4096 and 8192 rows).

### `turboexl3_dense_prep`

<!-- kernel-facts: turboexl3_dense_prep -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:381` (`const Dense`) |
| inputs | `x`, `suh` |
| outputs | `xp` |
| source | `text_prep` = `dense/prep` (1085 B, `110e3a6a2eb5`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: `linear`'s prepare at prefill rows in natural order (LANE_ORDER 0): before the register path,
  dequantize + MLX matmul and the fused GEMM for MCG (tests only); also `Dense.prepare` (`root.linearPrep`),
  `Dense.innerOn` (`root.linearInnerSplit`) and `Dense.pieceInner` (`root.linearPieceInner`). In lane order (LANE_ORDER
  1) the row-slot reference at decode rows (not served). Every dense EXL3 linear at prefill rows: Qwen3.8's trunk and
  MTP; DeepSeek-V4.1's trunk, vision tower and aligner (and `wo_a`'s slices at 17+ rows through `linearGrouped`);
  Ripple's projections.
- **Shapes and dtypes:** x T [rows, IN] (T bf16, f16 or f32); suh f16 [IN]. Out: xp f16 [rows, IN].
- **Grid and threadgroup:** `((IN/128) · 32, rows, 1)` / `(32, 1, 1)`: one simdgroup per (128-block, row).
- **Template args:** DIM (= IN), LANE_ORDER (0 natural, 1 the GEMV's lane order). It reads no weight, so every rate,
  codebook and window.
- **Replaces:** nothing: it is the prepare of the prefill paths (`prep` -> inner product -> `finish`).
- **Bits:** `format.project`'s prepare rounding points (f16(x) · suh, H128, f16); the paths it starts are within 5e-3 of
  `format.project` per row with f32 x and y and within 2.5e-2 with bf16.
- **Pinned by:** `src/core/dense.zig` "turboexl3 dense texts: inputs read, outputs written, the dequant layout names the
  GEMV's row/column rule" (CPU, its xp store); `src/core/dense.zig` "turboexl3 dense un-rotated trunk texts: the
  un-rotation kernels repeat the dense butterfly verbatim, ..." (CPU, its butterfly lines); `src/core/gpu_test.zig`
  "turboexl3 gpu: dense.linear equals format.project per row at the trunk's rates, GEMV and prefill paths
  (TURBOEXL3_GPU=1)" (GPU, rows 17 to 200); `src/core/gpu_test.zig` "turboexl3 gpu: dense.linearOn across two streams
  gives linear's bytes on both prefill paths (TURBOEXL3_GPU=1)" (GPU, bytes on two streams).

### `turboexl3_dense_prep_s`

<!-- kernel-facts: turboexl3_dense_prep_s -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:388` (`const Dense`) |
| inputs | `x`, `suh` |
| outputs | `xp`, `xs` |
| source | `text_prep_s` = `dense/prep_s` (1236 B, `4b5d658122db`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: the multi-row GEMV chain's prepare in lane order (LANE_ORDER 1): `root.linear` (and every entry
  built on it) at 5 to 16 decode rows (1 to 16 with `Config.dec_one` off), F3's gate and up (`Dense.prepareGemv` in
  `root.sharedSwiglu`) and Qwen3.8's decode merges `decodeMergeQPlanes` and `decodeMergeIndexer` where DEC-ONE does not
  take the call. In natural order (LANE_ORDER 0) before the fused GEMM for MUL1 (tests only). Every dense EXL3 linear:
  Qwen3.8's trunk, shared expert, `lm_head` and MTP; DeepSeek-V4.1's trunk; Ripple's projections and head.
- **Shapes and dtypes:** x T [rows, IN] (T bf16, f16 or f32); suh f16 [IN]. Out: xp f16 [rows, IN]; xs f32 [rows,
  IN/128] (each 128-block's sum of its rounded values, for the deferred MUL1 map).
- **Grid and threadgroup:** `((IN/128) · 32, rows, 1)` / `(32, 1, 1)`: one simdgroup per (128-block, row).
- **Template args:** DIM (= IN), LANE_ORDER. It reads no weight, so every rate, codebook and window.
- **Replaces:** nothing: it is the base of the multi-row GEMV chain (`prep_s` -> `dgemv` -> `dfinish`).
- **Bits:** `format.project`'s prepare rounding points; xs sums the f16 values in f32 (`simd_sum`). Its chain is within
  5e-3 of `format.project` per row with f32 x and y and within 2.5e-2 with bf16 (`ref.compare`).
- **Pinned by:** `src/core/dense.zig` "turboexl3 dense one-launch decode (DEC-ONE) texts: every value-setting line is
  the three-launch chain's, ..." and `src/models/deepseek_v41/f3_test.zig` "turboexl3 f3u: the fused shared-expert text
  is text_dfinish's finishes, MLX's clamped SwiGLU and text_prep_s's prepare" (CPU: its prepare lines, which DEC-ONE's
  `dec1` and F3 must carry; `text_prep_sc` holds it whole); `src/core/gpu_test.zig` "turboexl3 gpu: dense.linear equals
  format.project per row at the trunk's rates, GEMV and prefill paths (TURBOEXL3_GPU=1)", `src/core/gpu_test.zig`
  "turboexl3 gpu: dense multi-row GEMV, a row's bytes are the same alone and among 2..16 rows (TURBOEXL3_GPU=1)" and
  `src/core/gpu_test.zig` "turboexl3 gpu: the one-launch decode linear (DEC-ONE) equals the three-launch chain byte for
  byte (TURBOEXL3_GPU=1)" (GPU, as the chain's first launch).

### `turboexl3_dense_prep_sc`

<!-- kernel-facts: turboexl3_dense_prep_sc -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:407` (`const Dense`) |
| inputs | `x`, `suh` |
| outputs | `xp`, `xs`, `cnt` |
| source | `text_prep_sc` = `dense/prep_s` ++ `dense/prep_sc` (1429 B, `6e36fd607e96`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core: DEC-ONE with the prepare in its own launch, before `turboexl3_dense_dec0`: `root.linear` (and every
  entry built on it) at 2 to 4 rows with OUT > 1024, or any DEC-ONE call where `Dense.decPrepFits` fails
  (`Config.dec_one`). Qwen3.8's decode merges `decodeMergeQPlanes` and `decodeMergeIndexer` launch it before their own
  DEC-ONE variants. Every dense EXL3 linear: Qwen3.8's trunk, shared expert, `lm_head` and MTP; DeepSeek-V4.1's trunk;
  Ripple's projections and head.
- **Shapes and dtypes:** x T [rows, IN] (T bf16, f16 or f32); suh f16 [IN]. Out: xp f16 [rows, IN] in lane order; xs f32
  [rows, IN/128]; cnt u32 [NCNT] zeroed, NCNT = max(OUT/128 · chunks, 8).
- **Grid and threadgroup:** `((IN/128) · 32, rows, 1)` / `(32, 1, 1)`.
- **Template args:** DIM (= IN), LANE_ORDER (1), NCNT, NTH ((IN/128) · 32 · rows, the launch's threads: the zeroing
  loop's stride). It reads no weight, so every rate, codebook and window.
- **Replaces:** `turboexl3_dense_prep_s` (its text whole) plus the counter zeroing DEC-ONE needs, so `dec0` needs no
  zeroing launch.
- **Bits:** xp and xs byte-identical to `turboexl3_dense_prep_s` (the same text); cnt zeros.
- **Pinned by:** `src/core/dense.zig` "turboexl3 dense one-launch decode (DEC-ONE) texts: every value-setting line is
  the three-launch chain's, ..." (CPU: `text_prep_sc` contains `text_prep_s`, its signature names, no MLX shape
  buffers); `src/core/gpu_test.zig` "turboexl3 gpu: the one-launch decode linear (DEC-ONE) equals the three-launch chain
  byte for byte (TURBOEXL3_GPU=1)" (GPU, bytes through the "own_launch" arm); `src/core/gpu_test.zig` "turboexl3 gpu:
  dense.linear equals format.project per row at the trunk's rates, GEMV and prefill paths (TURBOEXL3_GPU=1)" (GPU, the
  bars at the rows it takes).

### `turboexl3_dense_rs_gemm`

<!-- kernel-facts: turboexl3_dense_rs_gemm -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:397` (`const Dense`) |
| inputs | `tiles`, `x0`, `w0`, `xs0` |
| outputs | `z0` |
| source | `prefill.gemm1s_text` = `prefill/gemm_small_1` ++ `prefill/gemm_small_2` ++ `prefill/gemm_small_3` ++ `prefill/gemm_small_op1_wx` ++ `prefill/gemm_small_4` ++ `prefill/gemm_small_op1_xsz` ++ `prefill/gemm_small_5` (3747 B, `34fd7442f5b6`) |
| header | `prefill.header_nax_small` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` (53783 B, `e637445d0580`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.linear` (and every entry built on it) at 17 to `small_rows` (32) rows where `Config.nax`
  holds (`small_rows` is 0 otherwise) and `fused_prefill` is off: a prefill's snapshot tail. Also `Dense.innerOf` /
  `innerOn` (`root.linearPrepared`, `linearInnerSplit`) and Qwen3.8's ATTN-IN and ATTN-OUT (`attn_io.prefillInner`).
  Between `turboexl3_dense_prep` (natural order) and `turboexl3_dense_finish`. Qwen3.8's trunk and MTP, DeepSeek-V4.1's
  trunk (and `wo_a`'s slices through `linearGrouped`), Ripple's projections.
- **Shapes and dtypes:** tiles i32 [max(tmax, 2), 4] (32-row tiles of the one "expert": 0, 32t, row count, 0; tmax =
  ceil(rows/32)); x0 f16 [rows, IN] in natural order; w0 the trellis U16 or I16 [IN/16, OUT/16, NHW]; xs0 f32 [8] zeros
  (DEFER 0; 8 so MLX binds it as `device`). Out: z0 f16 [rows, OUT].
- **Grid and threadgroup:** `(ceil(items/SGT) · 32 · SGT, 1, 1)` / `(32 · SGT, 1, 1)` with SGT 4 and items = tmax ·
  (OUT/32): `(ceil(items/4) · 128, 1, 1)` / `(128, 1, 1)`.
- **Template args:** NHW, CODEBOOK, WBITS, KD (= IN), ND (= OUT), NOPS (1), SGT (4), DEFER (0: f16 weight values), NBX
  (IN/128), SINTR (1 where NHW % 16 == 0, the integer-rate reader), ALAY (0), H16 (0). Every rate and codebook; up to
  `small_rows_cap` (128) rows where a test raises `small_rows`.
- **Replaces:** `turboexl3_dense_dequant` and MLX's matmul (2 launches, an f16 W_hat written and read): one launch that
  decodes each weight once per 32-row tile straight into the B registers.
- **Bits:** its own MMA summation order: within 5e-3 of `format.project` per row with f32 x and y. With `nax` false
  these rows take dequantize + matmul, whose bytes can differ.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: dense.linear equals format.project per row at the trunk's
  rates, GEMV and prefill paths (TURBOEXL3_GPU=1)" (GPU, rows 17, 31 and 32);
  `src/models/deepseek_v41/deepseek_test.zig` "turboexl3 gpu deepseek: root.linear and root.linearBias equal the host
  reference at every DeepSeek-V4.1 ..." (GPU, 32 rows); `src/core/gpu_test.zig` "turboexl3 gpu: dense.linearOn across
  two streams gives linear's bytes on both prefill paths (TURBOEXL3_GPU=1)" (GPU, 31 rows stay on the first stream, the
  same bytes); `src/core/gpu_test.zig` "turboexl3 gpu un-rotated trunk: root.linear on an un-rotated linear is one plain
  matmul within the bars at ..." (GPU, the chain's bytes at register-path rows on an un-rotated linear).

### `turboexl3_unrot_cols`

<!-- kernel-facts: turboexl3_unrot_cols -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:1298` (`fn unrotKernels`) |
| inputs | `w` |
| outputs | `r` |
| source | `text_unrot_cols` = `dense/unrot_cols` (731 B, `1e71a2b282ba`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core, opt-in at load: `root.trunkUnrot`, the host's un-rotated trunk on an armed resident table (the
  plugin table calls it), after `turboexl3_dense_dequant` decodes each member's W_hat; made on first use, never per
  call. The code's comments name the Qwen3.8 pack (bf16) and its joins (GDN in_proj_qkv | in_proj_z, QSA q | k | v |
  indexer, the shared expert's gate | up). Pass 1 (`dense.unrotCols`).
- **Shapes and dtypes:** w f16 [IN, OUT] (W_hat; OUT a multiple of 128). Out: r f32 [IN, OUT].
- **Grid and threadgroup:** `((OUT/128) · 32, IN, 1)` / `(32, 1, 1)`: one simdgroup per (128-column block, row).
- **Template args:** DIM (= OUT).
- **Replaces:** nothing per call: pass 1 of the load-time un-rotation. With `turboexl3_unrot_rows` it makes W_std, so a
  call past `unrot_min_rows` (32) rows is one MLX matmul.
- **Bits:** the unscaled +-1 butterfly in f32, no rounding before pass 2; opt-in, its own rounding. The pair's bars:
  each W_std element within one bf16 step of the f64 reference `diag(suh) H W_hat H diag(svh)`, or, where its terms
  cancel, half a step of its own value plus 16 f32 eps of its terms' absolute sum; at most 1 element in 1000 off the
  reference's own bf16 rounding; a call's y within relative RMS 2^-7.5 of the chain, max |diff| <= 2 bf16 steps at max
  |y|'s binade.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu un-rotated trunk: every trunk linear kind's un-rotation equals
  the f64 reference on sampled tiles, ..." (GPU: sampled tiles against the f64 reference on every trunk linear kind of
  the 5.05bpw pack, and the table's bill); `src/core/gpu_test.zig` "turboexl3 gpu un-rotated trunk: root.linear on an
  un-rotated linear is one plain matmul within the bars at ..." (GPU: the call bars at prefill rows, the chain's bytes
  at decode and register-path rows); `src/core/gpu_test.zig` "turboexl3 gpu un-rotated trunk: a join is one buffer whose
  members are zero-copy views, its product's columns ..." (GPU, joins); `src/core/dense.zig` "turboexl3 dense un-rotated
  trunk: the un-rotation equals diag(suh) H W_hat H diag(svh) in f64 within one bf16 ..." (CPU: the arithmetic ported
  statement for statement, the bars); `src/core/dense.zig` "turboexl3 dense un-rotated trunk texts: the un-rotation
  kernels repeat the dense butterfly verbatim, ..." (CPU: its butterfly lines, signature names, no MLX shape buffers).

### `turboexl3_unrot_rows`

<!-- kernel-facts: turboexl3_unrot_rows -->

| fact | from the code |
|---|---|
| made at | `src/core/dense.zig:1300` (`fn unrotKernels`) |
| inputs | `r`, `suh`, `svh` |
| outputs | `W` |
| source | `text_unrot_rows` = `dense/unrot_rows` (1431 B, `aeb52f782bb2`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core, opt-in at load: `root.trunkUnrot`, the host's un-rotated trunk on an armed resident table (the
  plugin table calls it), after `turboexl3_dense_dequant` decodes each member's W_hat; made on first use, never per
  call. The code's comments name the Qwen3.8 pack (bf16) and its joins (GDN in_proj_qkv | in_proj_z, QSA q | k | v |
  indexer, the shared expert's gate | up). Pass 2 (`dense.unrotRows`), on pass 1's r.
- **Shapes and dtypes:** r f32 [IN, OUT] (pass 1's; IN a multiple of 128, OUT of 32); suh f16 [IN]; svh f16 [OUT]. Out:
  W OUT_T [IN, OUT] (W_std; OUT_T bf16 or f16, the activation dtype).
- **Grid and threadgroup:** `((OUT/32) · 256, IN/128, 1)` / `(256, 1, 1)`: one threadgroup per (128-row block, 32-column
  strip).
- **Template args:** DIM (= OUT), OUT_T.
- **Replaces:** per call: `turboexl3_dense_prep` and `turboexl3_dense_dequant` (or the kept W_hat) -> MLX matmul ->
  `turboexl3_dense_finish` (4 launches, 3 levels; 3 launches with a resident W_hat), replaced by one MLX matmul x ·
  W_std (1 level).
- **Bits:** one rounding: x 1/128 (exact), suh and svh in f32, then OUT_T; opt-in, its own rounding. The pair's bars:
  each W_std element within one bf16 step of the f64 reference `diag(suh) H W_hat H diag(svh)`, or, where its terms
  cancel, half a step of its own value plus 16 f32 eps of its terms' absolute sum; at most 1 element in 1000 off the
  reference's own bf16 rounding; a call's y within relative RMS 2^-7.5 of the chain, max |diff| <= 2 bf16 steps at max
  |y|'s binade.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu un-rotated trunk: every trunk linear kind's un-rotation equals
  the f64 reference on sampled tiles, ..." (GPU: sampled tiles against the f64 reference on every trunk linear kind of
  the 5.05bpw pack, and the table's bill); `src/core/gpu_test.zig` "turboexl3 gpu un-rotated trunk: root.linear on an
  un-rotated linear is one plain matmul within the bars at ..." (GPU: the call bars at prefill rows, the chain's bytes
  at decode and register-path rows); `src/core/gpu_test.zig` "turboexl3 gpu un-rotated trunk: a join is one buffer whose
  members are zero-copy views, its product's columns ..." (GPU, joins); `src/core/dense.zig` "turboexl3 dense un-rotated
  trunk: the un-rotation equals diag(suh) H W_hat H diag(svh) in f64 within one bf16 ..." (CPU: the arithmetic ported
  statement for statement, the bars); `src/core/dense.zig` "turboexl3 dense un-rotated trunk texts: the un-rotation
  kernels repeat the dense butterfly verbatim, ..." (CPU: its butterfly lines, signature names, no MLX shape buffers).

## `src/core/pgemm.zig`

pg2: a dense linear's prefill matmul and finish as one launch.

### `turboexl3_pgemm_fused`

<!-- kernel-facts: turboexl3_pgemm_fused -->

| fact | from the code |
|---|---|
| made at | `src/core/pgemm.zig:135` (`fn kernelGet`) |
| inputs | `x`, `w`, `svh` |
| outputs | `y` |
| source | `body` = `pg2/gemm_fused` ++ text '\n#undef kernel\n' (4974 B, `5d135c7c939d`) |
| header | `header` = `steel_nax` ++ text 'using namespace mlx::steel;\n' ++ `pg2/gemm_loop` ++ text (72 B) (34287 B, `0ab3ef8c4fca`) |

<!-- /kernel-facts -->

- **Used by:** core, not served by default: `Dense.linearOn`'s matmul path at prefill rows (`root.linear`,
  `linearSplit`) when the host sets `pgemm.mode` to on (default off), at a `pgemm.table` (K, N) where `Config.nax`
  holds, xp and W are f16 and MLX would run its regular fused NAX kernel (`mlxRegular`: no split K). In `check` mode it
  runs beside the chain and the host's `check_hook` gets both outputs; `pgemm.warm` compiles each tile at load. The
  table's six shapes are DeepSeek-V4.1's: wq_b (1280 -> 32768), wo_b (8192 -> 5120), the shared expert's w1 and w3 (5120
  -> 2304) and w2 (2304 -> 5120), wq_a (5120 -> 1280), wkv and the compressors' wkv and wgate (5120 -> 512).
- **Shapes and dtypes:** x f16 [M, K] (the prepared rows, natural order); w f16 [K, N] (W_hat, dequantized or resident);
  svh f16 [N]. Out: y OUT_T [M, N] (x's dtype).
- **Grid and threadgroup:** `(tiles_n · 2^SWZ · 32, ceil(tiles_m / 2^SWZ) · WN, WM)` / `(32, WN, WM)` (MLX's own
  threadgroup shape), tiles_m = ceil(M/BM), tiles_n = N/BN.
- **Template args:** T (f16), BM, BN, BK, WM, WN, SWZ (the swizzle log), AM, AN, AK (M, N, K aligned to BM, BN, BK),
  OUT_T. The table's tiles: BM 32 (64 at 5120 -> 512), BN 256, BK 256 or 512, WM 1 (2 at 5120 -> 512), WN 4, SWZ 2;
  `Tile.admitted` needs whole 16-row fragments and 32-column N pairs per simdgroup, BK a multiple of 32, BN of 128, BM ·
  BN · 2 <= 32 KB.
- **Replaces:** MLX's f16 matmul -> `turboexl3_dense_finish` (2 launches, 2 levels; z f16 written and read): one launch,
  MLX's fused NAX GEMM body (its `gemm_loop` verbatim, MLX 0.32.3) with the finish as its epilogue.
- **Bits:** byte-identical to dequantize + MLX's matmul + the finish where it takes a call (MLX's regular NAX kernel, no
  split K): the accumulators rounded to half as MLX's store rounds them, then the finish's statements. `check` mode
  hands both to the host to compare.
- **Pinned by:** `src/core/chains_test.zig` "turboexl3 gpu pg2: the fused matmul and finish give dequantize + matmul +
  finish's words at every tabled shape (TURBOEXL3_GPU=1)" (GPU: `root.linear` with `Mode.on` against `Mode.off`, word
  for word, at every tabled (K, N) and rows 64, 100 and 2048 where MLX runs its regular kernel); `src/core/pgemm.zig`
  "pgemm: the epilogue runs the finish's statements, the loop is MLX's gemm_loop by its statements, ..." (CPU: the
  epilogue carries each of `finish.metal`'s 10 lines, the loop is MLX's, every table tile admitted, `mlxRegular`'s
  cases).

## `src/core/prefill.zig`

The routed experts at prefill rows (17 and more): the route, the prepare, the two GEMMs, the mid and the finish, the
clamped mids (DeepSeek-V4.1) and the 128-row tiles.

### `turboexl3_pf_fin`

<!-- kernel-facts: turboexl3_pf_fin -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:652` (`const Prefill`) |
| inputs | `zd`, `inds`, `inv`, `scores`, `svh_d` |
| outputs | `y` |
| source | `text_finish` = `prefill/finish_1` ++ `prefill/finish_2` ++ `prefill/finish_3` ++ `prefill/finish_4` (1033 B, `e68b2ddd88ba`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), only with `Config.pf_t128` false
  (`Prefill.moe` on the mixed option set); with it on (the default) `turboexl3_pf_t128_fin` takes its place.
  `Prefill.finish` launches it when `Options.aux` is 0 (the strided finish). Qwen3.8-Flash-Next's routed experts (512
  experts, top-10) through `moe`, and DeepSeek-V4.1's routed experts through `moeClamped` (swiglu_limit 10); the finish
  takes no clamp.
- **Shapes and dtypes:** zd f16 [S, hidden] (the down inner products in sorted order, S = rows · topk); inds u32 or i32
  [S] (the flat expert ids, S = rows · topk); inv u32 [S] (slot to sorted position, from prep); scores bf16, f16 or f32
  [S]; svh_d f16 [experts, hidden]. Out: y OUT_T [rows, hidden], OUT_T = x's dtype T (bf16, f16 or f32) as `moePrefill`
  passes it.
- **Grid and threadgroup:** `((hidden/128/BPT) · 32 · BPT, rows, 1)` / `(32 · BPT, 1, 1)`: one simdgroup per (128-block
  of hidden, row).
- **Template args:** HID (hidden), TOPK (slots per row), BPT (`bptFor(hidden/128)`: the largest count of 128-blocks, 8
  down to 1, that divides them: 5 at Qwen3.8's 2560, 8 at DeepSeek-V4.1's 5120), OUT_T (y's dtype). No rate, codebook or
  window argument: it reads the f16 inner products, so it serves every rate.
- **Replaces:** nothing: it is the base of the chain's finish (the down finish, H128 · svh rounded to f16, times each
  slot's score, summed in slot order through inv, in one launch).
- **Bits:** within tolerance, as part of the chain: y within 5e-3 of `ref.moeRowDec` (`format.project`'s rounding
  points, `ref.compare`) in f32 on sampled rows (the first, middle and last), and in bf16 within 1.5 × the error of
  Sushi's prefill arm + 2e-3 where Sushi is linked. The slot sum runs in slot order, so a row's bytes do not depend on
  where its slots sit in the sorted order or the tiles.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: the chain at every option set within those bars; with `aux` 0 it is the byte reference for
  `turboexl3_pf_fin_v`); `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles: the call and every stage give
  the mixed chain's ..." (GPU: the mixed chain's finish stage at the served set and four variants, the bytes
  `turboexl3_pf_t128_fin` must give, and the whole call equals its stages); `src/core/prefill.zig` "turboexl3 prefill
  128-row tiles texts: both GEMMs' launches are the mixed chain's ..." (CPU: `text_finish` is `text_finish_t128` without
  the row choice); `src/core/prefill.zig` "turboexl3 prefill metal sources: every kernel as a standalone translation
  unit (TURBOEXL3_METAL_DUMP=1)" (CPU print; the box compiles it with `xcrun metal`).

### `turboexl3_pf_fin_v`

<!-- kernel-facts: turboexl3_pf_fin_v -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:667` (`const Prefill`) |
| inputs | `zd`, `inds`, `inv`, `scores`, `svh_d` |
| outputs | `y` |
| source | `text_finish_v` = `prefill/finish_v` (1182 B, `72667b6b538d`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** not served: `Prefill.init` makes it for every geometry, so `root`'s cache holds it, but nothing in the
  library launches it by default: `Prefill.finish` takes it in place of `turboexl3_pf_fin` when `Options.aux` is 4, 8,
  16 or 32, and `root.pfOptions` leaves `aux` 0 (`t128` also needs aux 0). Tests only.
- **Shapes and dtypes:** zd f16 [S, hidden]; inds u32 or i32 [S] (the flat expert ids, S = rows · topk); inv u32 [S]
  (its `inv_shape[0]` bounds the items); scores bf16, f16 or f32 [S]; svh_d f16 [experts, hidden]. Out: y OUT_T [rows,
  hidden] (OUT_T = T).
- **Grid and threadgroup:** `(ceil(rows · (hidden/128) / ASG) · 32 · ASG, 1, 1)` / `(32 · ASG, 1, 1)`: one simdgroup per
  flattened (row, 128-block) item; items past the end return.
- **Template args:** HID, TOPK, ASG (`Options.aux`: simdgroups per threadgroup, 4, 8, 16 or 32), OUT_T. Each lane holds
  4 consecutive elements (half4 loads, `turboexl3_h128v4`). Every rate.
- **Replaces:** `turboexl3_pf_fin` one for one (the same work in the vector layout: 1 launch, 1 level).
- **Bits:** byte-identical to `turboexl3_pf_fin` (the same values; the GPU test compares the whole chain at `aux`
  against `aux` 0). `aux_lane_sums` does not reach it (it has no XSO).
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: aux 8 on the served launch and aux 32 on the all-register path, byte for byte against aux
  0, and within the chain's bars); `src/core/prefill.zig` "turboexl3 prefill metal sources: every kernel as a standalone
  translation unit (TURBOEXL3_METAL_DUMP=1)" (CPU print).

### `turboexl3_pf_froute`

<!-- kernel-facts: turboexl3_pf_froute -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:658` (`const Prefill`) |
| inputs | `inds` |
| outputs | `order`, `tiles` |
| source | `text_froute` = `prefill/table_1` ++ `prefill/froute_count` ++ `prefill/table_2` (2416 B, `84e4298e16ea`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** not served: `Prefill.init` makes it for every geometry, so `root`'s cache holds it, but nothing in the
  library launches it by default: `Prefill.route` takes it with `Options.froute` on and `Options.small` `.off` or
  `.all`, and `root.pfOptions` sets `froute` with small `.mixed` (`turboexl3_pf_froute_m`, or
  `turboexl3_pf_t128_route`). Tests only.
- **Shapes and dtypes:** inds u32 or i32 [S] (the flat expert ids, S = rows · topk), its length read as `inds_shape[0]`.
  Out: order u32 [S] (sorted position to slot); tiles int32 [ceil(S/BMT) + experts, 4] = (expert, first sorted position,
  rows, 0) in expert order, then rows 0 to the end.
- **Grid and threadgroup:** `(1024, 1, 1)` / `(1024, 1, 1)`: one threadgroup; thread e owns expert e (experts <= 1024,
  `Geometry.admitted`).
- **Template args:** EXP (experts), BMT (`Options.tileRows`: `bm`, or 32 with small `.all`), SPL (`Options.split`; this
  text does not read it). No rate arguments.
- **Replaces:** MLX's argsort of the flat expert ids + `turboexl3_pf_table` (2 launches, 2 dependent levels) with 1: it
  counts each expert's slots with threadgroup atomics, scans the counts into each run's start and scatters the slots
  there (`froute_count` in place of `table_bsearch`).
- **Bits:** the slots of one expert land in arrival order, not slot order, but a row's bytes do not depend on where its
  slots land, so the chain's y is byte-identical to the argsort + table chain's.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: `.{ .froute = true }` and `.{ .small = .all, .froute = true }` byte for byte against the
  same set without `froute`); `src/core/prefill.zig` "turboexl3 prefill metal sources: every kernel as a standalone
  translation unit (TURBOEXL3_METAL_DUMP=1)" (CPU print).

### `turboexl3_pf_froute_m`

<!-- kernel-facts: turboexl3_pf_froute_m -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:659` (`const Prefill`) |
| inputs | `inds` |
| outputs | `order`, `tiles` |
| source | `text_froute_m` = `prefill/table_mixed_1` ++ `prefill/froute_count` ++ `prefill/table_mixed_2` (3328 B, `0ed55b2ce032`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), only with `Config.pf_t128` false
  (`Prefill.moe` on the mixed option set); with it on (the default) `turboexl3_pf_t128_route` takes its place.
  `Prefill.route` launches it with `Options.froute` on and small `.mixed`. Qwen3.8-Flash-Next's routed experts (512
  experts, top-10) through `moe`, and DeepSeek-V4.1's routed experts through `moeClamped` (swiglu_limit 10).
- **Shapes and dtypes:** inds u32 or i32 [S] (the flat expert ids, S = rows · topk). Out: order u32 [S]; tiles int32
  [ceil(S/BMT) + experts + SPL · experts, 4]: the BMT-row tiles in expert order from 0, then from TBIG = ceil(S/BMT) +
  experts the register-path tiles of at most 32 rows (an expert's remainder of at most 32 · SPL rows, or an expert of
  that few rows); unused rows 0.
- **Grid and threadgroup:** `(1024, 1, 1)` / `(1024, 1, 1)`: one threadgroup, thread e owns expert e.
- **Template args:** EXP (experts), BMT (`bm`: 64 served), SPL (`Options.split`: 1 or 2; 1 served). No rate arguments.
- **Replaces:** MLX's argsort + `turboexl3_pf_table_m` (2 launches, 2 dependent levels) with 1 (count, scan and scatter
  in the threadgroup).
- **Bits:** byte-identical chain to the argsort + `turboexl3_pf_table_m` chain (arrival order within an expert changes
  no output).
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: the served launch with `froute`, and with aux 8, alay 1 or 2 and half_rows, byte for byte
  against the set without them); `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles: the call and every
  stage give the mixed chain's ..." (GPU: the mixed chain's route at the served set, the side the 128-row tiles match);
  `src/root.zig` "turboexl3 root: the served prefill options follow the config, and the decode form is the config's"
  (CPU: `pf_t128` false gives the set without `t128`); `src/core/prefill.zig` "turboexl3 prefill metal sources: every
  kernel as a standalone translation unit (TURBOEXL3_METAL_DUMP=1)" (CPU print).

### `turboexl3_pf_gemm1`

<!-- kernel-facts: turboexl3_pf_gemm1 -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:644` (`const Prefill`) |
| inputs | `tiles`, `x0`, `w0`, `xs0` |
| outputs | `z0` |
| source | `gemm1_text` = `prefill/gemm_1` ++ `prefill/gemm_2` ++ `prefill/gemm_3` ++ `prefill/gemm_op1_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op1_xsz` ++ `prefill/gemm_5` (6517 B, `dd6225a4d3f1`) |
| header | `header_nax` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` (60754 B, `4c8c1fc745df`) |

<!-- /kernel-facts -->

- **Used by:** not served: `Prefill.init` makes it for every geometry, so `root`'s cache holds it, but nothing in the
  library launches it by default: `Prefill.gemmD` takes it with `Options.small = .off` (the base set `Options{}` and the
  shared-decode-only sets), and `root.pfOptions` sets small `.mixed`. Tests only. Its source is also
  `turboexl3_pf_t128_dnt`'s.
- **Shapes and dtypes:** tiles int32 [ceil(S/bm) + experts, 4] (from `turboexl3_pf_table` or `turboexl3_pf_froute`); x0
  = xd f16 [S, inter] (mid's down input, sorted, in ALAY's layout); w0 the down trellis u16 [experts, inter/16,
  hidden/16, NHW]; xs0 = xsd f32 [S, inter/128] (its block sums, read with DEFER). Out: z0 = zd f16 [S, hidden], the
  inner products in sorted order.
- **Grid and threadgroup:** `(tmax · (hidden/64/NCB) · T, 1, 1)` / `(T, 1, 1)`, T = 2 · bm + 32 · wsp threads, tmax =
  ceil(S/bm) + experts: one threadgroup per (M tile, NCB consecutive 64-wide output blocks).
- **Template args:** the 23 of `gemmCfg`: NHW (halfwords per 256-weight trellis tile, K = NHW/16: any even 16 to 128),
  CODEBOOK (MUL1, MCG), WBITS (the window, 8 to 16), KD and ND (the op's input and output widths), NOPS, BMR
  (`Options.bm`: 64, 128 or 256), DBUF (`dbuf`), PROBE (`probe`, timing probes only), WSP (`wsp`, decode-only producer
  simdgroups), MFIRST (`mma_first`), UNR (`unroll`), SGT (threads / 32), TMAXS (`split` · experts: where the mixed
  table's register-path tiles start; only the mixed launch reads it), FRAGB (`fragb`), INTR (`intr`, needs n % 16 == 0),
  DEFER (`defer_affine`, the deferred MUL1 map, MUL1 only), NBX (KD/128: block sums per row), BKS (the K stage depth),
  NCB (output blocks per threadgroup), SINTR (`sintr`, the register path's integer-rate reader), ALAY (`alay`, needs
  FRAGB) and H16 (`half_rows`, needs FRAGB). Here KD = inter, ND = hidden, NOPS 1, BKS 64 (128 with `Options.bk128`),
  NCB `Options.ncb` (1, 2 or 4). The source and header do not change with the Options: `Options.small` picks this kernel
  name, and every other option reaches it as a template int. Every rate, codebook and window.
- **Replaces:** nothing: it is the base of the down GEMM (each 64-deep weight stage decoded into threadgroup memory
  inside the loader, then NAX MMAs; no f16 W_hat in device memory).
- **Bits:** within tolerance, as part of the chain: y within 5e-3 of `ref.moeRowDec` (`format.project`'s rounding
  points, `ref.compare`) in f32 on sampled rows (the first, middle and last), and in bf16 within 1.5 × the error of
  Sushi's prefill arm + 2e-3 where Sushi is linked. It sums in the NAX's order (16-deep MMAs, K ascending) and applies
  the deferred map (A · acc + C · Σx) where DEFER is on. `bk128`, `ncb`, `sintr`, `alay` and `half_rows` move loads,
  MMAs or work only: each gives the bytes of the same set without it.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: `.{}`, the padded and fragment-order shared-decode sets, `.{ .bk128 = true, .ncb = 2 }`
  single-buffered, alay 1 + half_rows, within the bars and byte for byte where an option only moves work);
  `src/core/prefill.zig` "turboexl3 prefill texts: the NAX header carries steel's tiles and turboexl3's decoder; ..."
  (CPU: op 0's names resolve); `src/core/prefill.zig` "turboexl3 prefill down options: bk128 doubles the stage (32 KB
  double-buffered in ..." (CPU: the stage sizes and NCB values); `src/core/prefill.zig` "turboexl3 prefill alay: a
  lane's A fragment elements are one aligned 16-byte (32-deep ..." (CPU: the text reads ALAY and H16);
  `src/core/prefill.zig` "turboexl3 prefill stage plan: a sub-stage fills its own BSZH halfs; the carried double ..."
  (CPU: the stage plan and the carried double buffer); `src/core/prefill.zig` "turboexl3 prefill 128-row tiles texts:
  both GEMMs' launches are the mixed chain's ..." (CPU: no write through an input binding); `src/core/prefill.zig`
  "turboexl3 prefill metal sources: every kernel as a standalone translation unit (TURBOEXL3_METAL_DUMP=1)" (CPU print,
  with `_gemm1_kn2_sb`).

### `turboexl3_pf_gemm1m`

<!-- kernel-facts: turboexl3_pf_gemm1m -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:657` (`const Prefill`) |
| inputs | `tiles`, `x0`, `w0`, `xs0` |
| outputs | `z0` |
| source | `gemm1m_text` = `prefill/gemm_mixed_1` ++ `prefill/gemm_1` ++ `prefill/gemm_mixed_bs` ++ `prefill/gemm_3` ++ `prefill/gemm_op1_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op1_xsz` ++ `prefill/gemm_5` ++ `prefill/gemm_mixed_2` ++ `prefill/gemm_small_1` ++ `prefill/gemm_mixed_index` ++ `prefill/gemm_small_3` ++ `prefill/gemm_small_op1_wx` ++ `prefill/gemm_small_4` ++ `prefill/gemm_small_op1_xsz` ++ `prefill/gemm_small_5` ++ `prefill/gemm_mixed_3` (10642 B, `b717e7a11df7`) |
| header | `header_nax_mixed` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` (60754 B, `4c8c1fc745df`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), only with `Config.pf_t128` false
  (`Prefill.moe` on the mixed option set); with it on (the default) `turboexl3_pf_t128_dnm` (the same source over the
  remainder table) takes its place. `Prefill.gemmD` launches it with small `.mixed`. Qwen3.8-Flash-Next's routed experts
  (512 experts, top-10) through `moe`, and DeepSeek-V4.1's routed experts through `moeClamped` (swiglu_limit 10).
- **Shapes and dtypes:** tiles int32 [tmax + SPL · experts, 4] (the mixed table from `turboexl3_pf_table_m` or
  `turboexl3_pf_froute_m`, tmax = ceil(S/bm) + experts); x0 = xd f16 [S, inter] (mid's down input, sorted, in ALAY's
  layout); w0 the down trellis u16 [experts, inter/16, hidden/16, NHW]; xs0 = xsd f32 [S, inter/128] (its block sums,
  read with DEFER). Out: z0 = zd f16 [S, hidden], the inner products in sorted order.
- **Grid and threadgroup:** `((tmax · (hidden/64/NCB) + ceil(SPL · experts · (hidden/32) / NSG)) · T, 1, 1)` / `(T, 1,
  1)`, T = 2 · bm + 32 · wsp, NSG = T/32: threadgroups below GBIGM = tmax · (hidden/64/NCB) run the shared-decode body
  over the large tiles, the rest NSG register-path work items (32 rows × 32 columns) each over the small tiles. Served:
  T 128, NCB 4.
- **Template args:** gemm1's 23 (NHW (halfwords per 256-weight trellis tile, K = NHW/16: any even 16 to 128), CODEBOOK
  (MUL1, MCG), WBITS (the window, 8 to 16), KD and ND (the op's input and output widths), NOPS, BMR (`Options.bm`: 64,
  128 or 256), DBUF (`dbuf`), PROBE (`probe`, timing probes only), WSP (`wsp`, decode-only producer simdgroups), MFIRST
  (`mma_first`), UNR (`unroll`), SGT (threads / 32), TMAXS (`split` · experts: where the mixed table's register-path
  tiles start; only the mixed launch reads it), FRAGB (`fragb`), INTR (`intr`, needs n % 16 == 0), DEFER
  (`defer_affine`, the deferred MUL1 map, MUL1 only), NBX (KD/128: block sums per row), BKS (the K stage depth), NCB
  (output blocks per threadgroup), SINTR (`sintr`, the register path's integer-rate reader), ALAY (`alay`, needs FRAGB)
  and H16 (`half_rows`, needs FRAGB)), KD = inter, ND = hidden, NOPS 1. Served values: BMR 64, DBUF 1, UNR 1, FRAGB 1,
  INTR and SINTR 1 where n % 16 == 0 (else 0), DEFER 1 for MUL1 packs (0 for MCG), PROBE 0, WSP 0, MFIRST 0, H16 0, ALAY
  `Config.pf_alay` (2 by default), TMAXS = experts (split 1), BKS 64 (`bk128` off), NCB 4 (1 where (hidden/64) % 4 !=
  0). `header_nax_mixed` is `header_nax` (one text). The source and header do not change with the Options:
  `Options.small` picks this kernel name, and every other option reaches it as a template int. Every rate, codebook and
  window.
- **Replaces:** `turboexl3_pf_gemm1` and `turboexl3_pf_gemm1s` as one launch (both bodies in 1 level).
- **Bits:** within tolerance, as part of the chain: y within 5e-3 of `ref.moeRowDec` (`format.project`'s rounding
  points, `ref.compare`) in f32 on sampled rows (the first, middle and last), and in bf16 within 1.5 × the error of
  Sushi's prefill arm + 2e-3 where Sushi is linked. It sums in the NAX's order (16-deep MMAs, K ascending) and applies
  the deferred map (A · acc + C · Σx) where DEFER is on. `bk128`, `ncb`, `sintr`, `alay` and `half_rows` move loads,
  MMAs or work only: each gives the bytes of the same set without it.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: the mixed sets, split 2, the served launch with ncb 4, bk128, sintr, alay 1 and 2,
  half_rows); `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles: the call and every stage give the mixed
  chain's ..." (GPU: the mixed chain's zd, the bytes the 128-row tiles' down launches give); `src/core/prefill.zig`
  "turboexl3 prefill alay: a lane's A fragment elements are one aligned 16-byte (32-deep ..." (CPU);
  `src/core/prefill.zig` "turboexl3 prefill 128-row tiles texts: both GEMMs' launches are the mixed chain's ..." (CPU);
  `src/core/prefill.zig` "turboexl3 prefill metal sources: every kernel as a standalone translation unit
  (TURBOEXL3_METAL_DUMP=1)" (CPU print: `_gemm1m_n4_jle`, `_down`, `_k`, `_n2`, `_n4`, `_kn2`, `_kn4`, ...).

### `turboexl3_pf_gemm1s`

<!-- kernel-facts: turboexl3_pf_gemm1s -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:654` (`const Prefill`) |
| inputs | `tiles`, `x0`, `w0`, `xs0` |
| outputs | `z0` |
| source | `gemm1s_text` = `prefill/gemm_small_1` ++ `prefill/gemm_small_2` ++ `prefill/gemm_small_3` ++ `prefill/gemm_small_op1_wx` ++ `prefill/gemm_small_4` ++ `prefill/gemm_small_op1_xsz` ++ `prefill/gemm_small_5` (3747 B, `34fd7442f5b6`) |
| header | `header_nax_small` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` (53783 B, `e637445d0580`) |

<!-- /kernel-facts -->

- **Used by:** not served: `Prefill.init` makes it for every geometry, so `root`'s cache holds it, but nothing in the
  library launches it by default: `Prefill.gemmD` takes it with `Options.small = .all` (every M tile at most 32 rows:
  the table at BMT 32), which `root.pfOptions` never sets. Tests only.
- **Shapes and dtypes:** tiles int32 [ceil(S/32) + experts, 4]; x0 = xd f16 [S, inter] (mid's down input, sorted, in
  ALAY's layout); w0 the down trellis u16 [experts, inter/16, hidden/16, NHW]; xs0 = xsd f32 [S, inter/128] (its block
  sums, read with DEFER). Out: z0 = zd f16 [S, hidden], the inner products in sorted order.
- **Grid and threadgroup:** `(ceil(tmax · (hidden/32) / SGT) · 32 · SGT, 1, 1)` / `(32 · SGT, 1, 1)`, tmax = ceil(S/32)
  + experts: one simdgroup per (tile, 32-column block) work item.
- **Template args:** the 12 of `gemmCfg`'s register-path list: NHW (K = NHW/16, any even 16 to 128), CODEBOOK (MUL1,
  MCG), WBITS (8 to 16), KD, ND, NOPS, SGT (`Options.sgt`: 1, 2, 4 or 8, default 4), DEFER (MUL1 only), NBX (KD/128),
  SINTR (`sintr`: the integer-rate reader, n % 16 == 0), ALAY and H16; KD = inter, ND = hidden, NOPS 1. Header
  `header_nax_small` (no `gemm_helpers`). The source and header do not change with the Options: `Options.small` picks
  this kernel name, and every other option reaches it as a template int. Every rate, codebook and window.
- **Replaces:** `turboexl3_pf_gemm1` when no tile exceeds 32 rows: each lane decodes its four trellis tiles straight
  into its B fragment registers, with no threadgroup memory and no barriers.
- **Bits:** within tolerance, as part of the chain: y within 5e-3 of `ref.moeRowDec` (`format.project`'s rounding
  points, `ref.compare`) in f32 on sampled rows (the first, middle and last), and in bf16 within 1.5 × the error of
  Sushi's prefill arm + 2e-3 where Sushi is linked. It sums in the NAX's order (16-deep MMAs, K ascending) and applies
  the deferred map (A · acc + C · Σx) where DEFER is on. `sintr`, `froute`, `aux`, `alay` and `half_rows` each give the
  bytes of the set without them.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: `.{ .small = .all }`, with sintr, froute, aux 32, alay 2 + half_rows);
  `src/core/prefill.zig` "turboexl3 prefill small: lane L decoding the trellis plan of L' = (b4,b2,b1,b3,b0) ..." (CPU:
  the permuted lane plan holds the NAX B fragment in order); `src/core/prefill.zig` "turboexl3 prefill small sintr: at
  every register-path lane the integer-rate reader ..." (CPU: both readers pick the same codewords);
  `src/core/prefill.zig` "turboexl3 prefill alay: a lane's A fragment elements are one aligned 16-byte (32-deep ..."
  (CPU); `src/core/prefill.zig` "turboexl3 prefill metal sources: every kernel as a standalone translation unit
  (TURBOEXL3_METAL_DUMP=1)" (CPU print, `_gemm1s`, `_gemm1s_j`, `_gemm1s_jle`).

### `turboexl3_pf_gemm2`

<!-- kernel-facts: turboexl3_pf_gemm2 -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:643` (`const Prefill`) |
| inputs | `tiles`, `x0`, `x1`, `w0`, `w1`, `xs0`, `xs1` |
| outputs | `z0`, `z1` |
| source | `gemm2_text` = `prefill/gemm_1` ++ `prefill/gemm_2` ++ `prefill/gemm_3` ++ `prefill/gemm_op2_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op2_xsz` ++ `prefill/gemm_5` (6517 B, `f34c9fcb0bd3`) |
| header | `header_nax` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` (60754 B, `4c8c1fc745df`) |

<!-- /kernel-facts -->

- **Used by:** not served: `Prefill.init` makes it for every geometry, so `root`'s cache holds it, but nothing in the
  library launches it by default: `Prefill.gemmGU` takes it with `Options.small = .off`, and `root.pfOptions` sets small
  `.mixed`. Tests only. Its source is also `turboexl3_pf_t128_gut`'s.
- **Shapes and dtypes:** tiles int32 [ceil(S/bm) + experts, 4]; x0 = xg and x1 = xu f16 [S, hidden] (prep's gate and up
  inputs, sorted, in ALAY's layout); w0 and w1 the gate and up trellises u16 [experts, hidden/16, inter/16, NHW]; xs0 =
  xsg and xs1 = xsu f32 [S, hidden/128] (the block sums, read with DEFER). Out: z0 = zg and z1 = zu f16 [S, inter], the
  inner products in sorted order.
- **Grid and threadgroup:** `(tmax · 2 · (inter/64) · T, 1, 1)` / `(T, 1, 1)`, T = 2 · bm + 32 · wsp, tmax = ceil(S/bm)
  + experts: one threadgroup per (M tile, op, 64-wide output block).
- **Template args:** the 23 of `gemmCfg`: NHW (halfwords per 256-weight trellis tile, K = NHW/16: any even 16 to 128),
  CODEBOOK (MUL1, MCG), WBITS (the window, 8 to 16), KD and ND (the op's input and output widths), NOPS, BMR
  (`Options.bm`: 64, 128 or 256), DBUF (`dbuf`), PROBE (`probe`, timing probes only), WSP (`wsp`, decode-only producer
  simdgroups), MFIRST (`mma_first`), UNR (`unroll`), SGT (threads / 32), TMAXS (`split` · experts: where the mixed
  table's register-path tiles start; only the mixed launch reads it), FRAGB (`fragb`), INTR (`intr`, needs n % 16 == 0),
  DEFER (`defer_affine`, the deferred MUL1 map, MUL1 only), NBX (KD/128: block sums per row), BKS (the K stage depth),
  NCB (output blocks per threadgroup), SINTR (`sintr`, the register path's integer-rate reader), ALAY (`alay`, needs
  FRAGB) and H16 (`half_rows`, needs FRAGB). Here KD = hidden, ND = inter, NOPS 2 (op 0 gate, op 1 up), BKS 64 and NCB 1
  always (`gemmGU` passes them). The source and header do not change with the Options: `Options.small` picks this kernel
  name, and every other option reaches it as a template int. Every rate, codebook and window.
- **Replaces:** nothing: it is the base of the gate and up GEMM (both projections in one launch, the trellis decoded in
  the loader).
- **Bits:** within tolerance, as part of the chain: y within 5e-3 of `ref.moeRowDec` (`format.project`'s rounding
  points, `ref.compare`) in f32 on sampled rows (the first, middle and last), and in bf16 within 1.5 × the error of
  Sushi's prefill arm + 2e-3 where Sushi is linked. It sums in the NAX's order (16-deep MMAs, K ascending) and applies
  the deferred map (A · acc + C · Σx) where DEFER is on. `sintr`, `alay` and `half_rows` each give the bytes of the set
  without them.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: the shared-decode sets, within the bars and byte for byte where an option only moves work);
  `src/core/prefill.zig` "turboexl3 prefill texts: the NAX header carries steel's tiles and turboexl3's decoder; ..."
  (CPU: no `XSECOND` left, op 1's names resolve); `src/core/prefill.zig` "turboexl3 prefill alay: a lane's A fragment
  elements are one aligned 16-byte (32-deep ..." (CPU); `src/core/prefill.zig` "turboexl3 prefill stage plan: a
  sub-stage fills its own BSZH halfs; the carried double ..." (CPU); `src/core/prefill.zig` "turboexl3 prefill 128-row
  tiles texts: both GEMMs' launches are the mixed chain's ..." (CPU); `src/core/prefill.zig` "turboexl3 prefill metal
  sources: every kernel as a standalone translation unit (TURBOEXL3_METAL_DUMP=1)" (CPU print, with `_gemm2_le128`).

### `turboexl3_pf_gemm2m`

<!-- kernel-facts: turboexl3_pf_gemm2m -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:656` (`const Prefill`) |
| inputs | `tiles`, `x0`, `x1`, `w0`, `w1`, `xs0`, `xs1` |
| outputs | `z0`, `z1` |
| source | `gemm2m_text` = `prefill/gemm_mixed_1` ++ `prefill/gemm_1` ++ `prefill/gemm_mixed_bs` ++ `prefill/gemm_3` ++ `prefill/gemm_op2_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op2_xsz` ++ `prefill/gemm_5` ++ `prefill/gemm_mixed_2` ++ `prefill/gemm_small_1` ++ `prefill/gemm_mixed_index` ++ `prefill/gemm_small_3` ++ `prefill/gemm_small_op2_wx` ++ `prefill/gemm_small_4` ++ `prefill/gemm_small_op2_xsz` ++ `prefill/gemm_small_5` ++ `prefill/gemm_mixed_3` (10642 B, `e19ee178960e`) |
| header | `header_nax_mixed` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` (60754 B, `4c8c1fc745df`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), only with `Config.pf_t128` false
  (`Prefill.moe` on the mixed option set); with it on (the default) `turboexl3_pf_t128_gum` (the same source over the
  remainder table) takes its place. `Prefill.gemmGU` launches it with small `.mixed`. Qwen3.8-Flash-Next's routed
  experts (512 experts, top-10) through `moe`, and DeepSeek-V4.1's routed experts through `moeClamped` (swiglu_limit
  10).
- **Shapes and dtypes:** tiles int32 [tmax + SPL · experts, 4] (the mixed table, tmax = ceil(S/bm) + experts); x0 = xg
  and x1 = xu f16 [S, hidden] (prep's gate and up inputs, sorted, in ALAY's layout); w0 and w1 the gate and up trellises
  u16 [experts, hidden/16, inter/16, NHW]; xs0 = xsg and xs1 = xsu f32 [S, hidden/128] (the block sums, read with
  DEFER). Out: z0 = zg and z1 = zu f16 [S, inter], the inner products in sorted order.
- **Grid and threadgroup:** `((tmax · 2 · (inter/64) + ceil(SPL · experts · 2 · (inter/32) / NSG)) · T, 1, 1)` / `(T, 1,
  1)`, T = 2 · bm + 32 · wsp, NSG = T/32: the shared-decode body below GBIGM, the register-path items above it. Served:
  T 128.
- **Template args:** gemm2's 23 (NHW (halfwords per 256-weight trellis tile, K = NHW/16: any even 16 to 128), CODEBOOK
  (MUL1, MCG), WBITS (the window, 8 to 16), KD and ND (the op's input and output widths), NOPS, BMR (`Options.bm`: 64,
  128 or 256), DBUF (`dbuf`), PROBE (`probe`, timing probes only), WSP (`wsp`, decode-only producer simdgroups), MFIRST
  (`mma_first`), UNR (`unroll`), SGT (threads / 32), TMAXS (`split` · experts: where the mixed table's register-path
  tiles start; only the mixed launch reads it), FRAGB (`fragb`), INTR (`intr`, needs n % 16 == 0), DEFER
  (`defer_affine`, the deferred MUL1 map, MUL1 only), NBX (KD/128: block sums per row), BKS (the K stage depth), NCB
  (output blocks per threadgroup), SINTR (`sintr`, the register path's integer-rate reader), ALAY (`alay`, needs FRAGB)
  and H16 (`half_rows`, needs FRAGB)), KD = hidden, ND = inter, NOPS 2, BKS 64, NCB 1. Served values: BMR 64, DBUF 1,
  UNR 1, FRAGB 1, INTR and SINTR 1 where n % 16 == 0 (else 0), DEFER 1 for MUL1 packs (0 for MCG), PROBE 0, WSP 0,
  MFIRST 0, H16 0, ALAY `Config.pf_alay` (2 by default), TMAXS = experts (split 1). `header_nax_mixed` is `header_nax`.
  The source and header do not change with the Options: `Options.small` picks this kernel name, and every other option
  reaches it as a template int. Every rate, codebook and window.
- **Replaces:** `turboexl3_pf_gemm2` and `turboexl3_pf_gemm2s` as one launch (both bodies in 1 level).
- **Bits:** within tolerance, as part of the chain: y within 5e-3 of `ref.moeRowDec` (`format.project`'s rounding
  points, `ref.compare`) in f32 on sampled rows (the first, middle and last), and in bf16 within 1.5 × the error of
  Sushi's prefill arm + 2e-3 where Sushi is linked. It sums in the NAX's order (16-deep MMAs, K ascending) and applies
  the deferred map (A · acc + C · Σx) where DEFER is on. `sintr`, `alay` and `half_rows` each give the bytes of the set
  without them.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: the mixed sets, split 2, the served launch, sintr, alay 1 and 2, half_rows);
  `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles: the call and every stage give the mixed chain's ..."
  (GPU: the mixed chain's zg and zu, the bytes the 128-row tiles' gate|up launches give); `src/core/prefill.zig`
  "turboexl3 prefill alay: a lane's A fragment elements are one aligned 16-byte (32-deep ..." (CPU);
  `src/core/prefill.zig` "turboexl3 prefill 128-row tiles texts: both GEMMs' launches are the mixed chain's ..." (CPU);
  `src/core/prefill.zig` "turboexl3 prefill metal sources: every kernel as a standalone translation unit
  (TURBOEXL3_METAL_DUMP=1)" (CPU print: `_gemm2m`, `_j`, `_jl`, `_je`, `_jle`, `_jle_p4`, `_jl2e`).

### `turboexl3_pf_gemm2s`

<!-- kernel-facts: turboexl3_pf_gemm2s -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:653` (`const Prefill`) |
| inputs | `tiles`, `x0`, `x1`, `w0`, `w1`, `xs0`, `xs1` |
| outputs | `z0`, `z1` |
| source | `gemm2s_text` = `prefill/gemm_small_1` ++ `prefill/gemm_small_2` ++ `prefill/gemm_small_3` ++ `prefill/gemm_small_op2_wx` ++ `prefill/gemm_small_4` ++ `prefill/gemm_small_op2_xsz` ++ `prefill/gemm_small_5` (3747 B, `430611958a11`) |
| header | `header_nax_small` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` (53783 B, `e637445d0580`) |

<!-- /kernel-facts -->

- **Used by:** not served: `Prefill.init` makes it for every geometry, so `root`'s cache holds it, but nothing in the
  library launches it by default: `Prefill.gemmGU` takes it with `Options.small = .all`, which `root.pfOptions` never
  sets. Tests only.
- **Shapes and dtypes:** tiles int32 [ceil(S/32) + experts, 4]; x0 = xg and x1 = xu f16 [S, hidden] (prep's gate and up
  inputs, sorted, in ALAY's layout); w0 and w1 the gate and up trellises u16 [experts, hidden/16, inter/16, NHW]; xs0 =
  xsg and xs1 = xsu f32 [S, hidden/128] (the block sums, read with DEFER). Out: z0 = zg and z1 = zu f16 [S, inter], the
  inner products in sorted order.
- **Grid and threadgroup:** `(ceil(tmax · 2 · (inter/32) / SGT) · 32 · SGT, 1, 1)` / `(32 · SGT, 1, 1)`, tmax =
  ceil(S/32) + experts: one simdgroup per (tile, op, 32-column block) work item.
- **Template args:** the 12 of `gemmCfg`'s register-path list: NHW (K = NHW/16, any even 16 to 128), CODEBOOK (MUL1,
  MCG), WBITS (8 to 16), KD, ND, NOPS, SGT (`Options.sgt`: 1, 2, 4 or 8, default 4), DEFER (MUL1 only), NBX (KD/128),
  SINTR (`sintr`: the integer-rate reader, n % 16 == 0), ALAY and H16; KD = hidden, ND = inter, NOPS 2. Header
  `header_nax_small`. The source and header do not change with the Options: `Options.small` picks this kernel name, and
  every other option reaches it as a template int. Every rate, codebook and window.
- **Replaces:** `turboexl3_pf_gemm2` when no tile exceeds 32 rows (register-path body: no threadgroup memory, no
  barriers).
- **Bits:** within tolerance, as part of the chain: y within 5e-3 of `ref.moeRowDec` (`format.project`'s rounding
  points, `ref.compare`) in f32 on sampled rows (the first, middle and last), and in bf16 within 1.5 × the error of
  Sushi's prefill arm + 2e-3 where Sushi is linked. It sums in the NAX's order (16-deep MMAs, K ascending) and applies
  the deferred map (A · acc + C · Σx) where DEFER is on. `sintr`, `froute`, `aux`, `alay` and `half_rows` each give the
  bytes of the set without them.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: the all-register sets); `src/core/prefill.zig` "turboexl3 prefill small: lane L decoding
  the trellis plan of L' = (b4,b2,b1,b3,b0) ..." (CPU); `src/core/prefill.zig` "turboexl3 prefill small sintr: at every
  register-path lane the integer-rate reader ..." (CPU); `src/core/prefill.zig` "turboexl3 prefill alay: a lane's A
  fragment elements are one aligned 16-byte (32-deep ..." (CPU); `src/core/prefill.zig` "turboexl3 prefill metal
  sources: every kernel as a standalone translation unit (TURBOEXL3_METAL_DUMP=1)" (CPU print: `_gemm2s`, `_j`, `_j0`,
  `_jle`, `_jl2e`).

### `turboexl3_pf_mid`

<!-- kernel-facts: turboexl3_pf_mid -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:647` (`const Prefill`) |
| inputs | `zg`, `zu`, `inds`, `order`, `svh_g`, `svh_u`, `suh_d` |
| outputs | `xd`, `xsd` |
| source | `text_mid` = `prefill/mid_1` ++ `prefill/mid_2` ++ `prefill/mid_3` ++ `prefill/mid_4` ++ `prefill/mid_swiglu` ++ `prefill/mid_5` (1402 B, `c03088145060`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), only with `Config.pf_t128` false
  (`Prefill.moe` on the mixed option set); with it on (the default) `turboexl3_pf_t128_mid` takes its place. Made for an
  unclamped geometry (`Geometry.limit` 0); `Prefill.mid` launches it with `Options.aux` 0. Qwen3.8-Flash-Next's routed
  experts through `moe`; a clamped geometry (DeepSeek-V4.1) makes `turboexl3_pf_midc` instead.
- **Shapes and dtypes:** zg and zu f16 [S, inter] (the gate and up inner products, sorted); inds u32 or i32 [S] (the
  flat expert ids, S = rows · topk); order u32 [S]; svh_g and svh_u f16 [experts, inter]; suh_d f16 [experts, inter].
  Out: xd f16 [S, inter] (the down input, sorted, in ALAY's layout); xsd f32 [S, inter/128] (its block sums).
- **Grid and threadgroup:** `((inter/128/BPT) · 32 · BPT, S, 1)` / `(32 · BPT, 1, 1)`: one simdgroup per (128-block of
  inter, sorted position).
- **Template args:** INTER (inter), BPT (`bptFor(inter/128)`: 5 at Qwen3.8's 640), ALAY (0, 1 or 2; `Config.pf_alay`, 2
  served). No LIMIT (an unclamped mid takes exactly its three). No rate arguments.
- **Replaces:** nothing: it is the base of the chain's mid: the gate and up finishes (H128 · svh, f16), SwiGLU in f32
  with `metal::precise::exp`, and the down prepare (f16, suh, H128, f16) with its block sums, in one launch.
- **Bits:** within tolerance, as part of the chain: y within 5e-3 of `ref.moeRowDec` (`format.project`'s rounding
  points, `ref.compare`) in f32 on sampled rows (the first, middle and last), and in bf16 within 1.5 × the error of
  Sushi's prefill arm + 2e-3 where Sushi is linked. Its SwiGLU uses `metal::precise::exp`, the decode forms
  `metal::exp`, so a row's SwiGLU can round differently at prefill rows. ALAY only moves slots: byte-identical to ALAY
  0.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: every set with `aux` 0); `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles:
  the call and every stage give the mixed chain's ..." (GPU: the mixed chain's xd and xsd, the bytes
  `turboexl3_pf_t128_mid` gives); `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: at a limit no value
  reaches, every clamped launch gives the ..." (GPU: the unclamped side of the `midc` arm);
  `src/models/deepseek_v41/clamped_test.zig` "turboexl3 clamped texts: each clamped kernel is its base text with the
  SwiGLU line ..." (CPU: it holds the SwiGLU line once and no LIMIT); `src/core/prefill.zig` "turboexl3 prefill 128-row
  tiles texts: both GEMMs' launches are the mixed chain's ..." (CPU); `src/core/prefill.zig` "turboexl3 prefill alay: a
  lane's A fragment elements are one aligned 16-byte (32-deep ..." (CPU: it writes the ALAY slot);
  `src/core/prefill.zig` "turboexl3 prefill metal sources: every kernel as a standalone translation unit
  (TURBOEXL3_METAL_DUMP=1)" (CPU print: `_mid`, `_mid_l`, `_mid_l2`).

### `turboexl3_pf_mid_v`

<!-- kernel-facts: turboexl3_pf_mid_v -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:662` (`const Prefill`) |
| inputs | `zg`, `zu`, `inds`, `order`, `svh_g`, `svh_u`, `suh_d` |
| outputs | `xd`, `xsd` |
| source | `text_mid_v` = `prefill/mid_v_1` ++ `prefill/mid_swiglu` ++ `prefill/mid_v_2` (1759 B, `3e3c9dbef907`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** not served: `Prefill.init` makes it for every geometry, so `root`'s cache holds it, but nothing in the
  library launches it by default: `Prefill.mid` takes it (unclamped geometry) when `Options.aux` is 4, 8, 16 or 32, and
  `root.pfOptions` leaves `aux` 0. Tests only.
- **Shapes and dtypes:** zg and zu f16 [S, inter] (the gate and up inner products, sorted); inds u32 or i32 [S] (the
  flat expert ids, S = rows · topk); order u32 [S]; svh_g and svh_u f16 [experts, inter]; suh_d f16 [experts, inter].
  Out: xd f16 [S, inter] (the down input, sorted, natural order); xsd f32 [S, inter/128] (its block sums).
- **Grid and threadgroup:** `(ceil(S · (inter/128) / ASG) · 32 · ASG, 1, 1)` / `(32 · ASG, 1, 1)`: one simdgroup per
  flattened (position, 128-block) item, S read as `order_shape[0]`; items past the end return.
- **Template args:** INTER, ASG (`Options.aux`: 4, 8, 16 or 32), XSO (`Options.aux_lane_sums`: 1 sums each block in the
  lane's own order). ALAY must be 0 with `aux` (`Options.valid`). No rate arguments.
- **Replaces:** `turboexl3_pf_mid` one for one (four consecutive elements per lane, several simdgroups per threadgroup).
- **Bits:** byte-identical to `turboexl3_pf_mid` with XSO 0 (`turboexl3_h128v4` and `turboexl3_sum_x4` give the same
  values); with XSO 1 the block sums are added in another f32 order, so outputs may differ in the last bits.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: aux 8 and aux 32 byte for byte against aux 0); `src/models/deepseek_v41/clamped_test.zig`
  "turboexl3 gpu clamped: at a limit no value reaches, every clamped launch gives the ..." (GPU: the unclamped side of
  the `mid_vc` arm); `src/models/deepseek_v41/clamped_test.zig` "turboexl3 clamped texts: each clamped kernel is its
  base text with the SwiGLU line ..." (CPU); `src/core/prefill.zig` "turboexl3 prefill metal sources: every kernel as a
  standalone translation unit (TURBOEXL3_METAL_DUMP=1)" (CPU print).

### `turboexl3_pf_mid_vc`

<!-- kernel-facts: turboexl3_pf_mid_vc -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:664` (`const Prefill`) |
| inputs | `zg`, `zu`, `inds`, `order`, `svh_g`, `svh_u`, `suh_d` |
| outputs | `xd`, `xsd` |
| source | `text_mid_vc` = `prefill/mid_v_1` ++ `prefill/mid_swiglu_clamp` ++ `prefill/mid_v_2` (2112 B, `31fe60881aaf`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** not served: made in place of `turboexl3_pf_mid_v` for a clamped geometry (`Geometry.limit` != 0:
  `root.moeClamped`, DeepSeek-V4.1's swiglu_limit 10), and launched only with `Options.aux` != 0, which `root.pfOptions`
  never sets. Tests only. The clamp's users: DeepSeek-V4 and V4.1 (swiglu_limit 10) and GLM (the same clamp), each
  through `root.moeClamped`; Ripple's clamped SwiGLU is its own (`turboexl3_ric_swiglu_prep_sc` / `_prep`,
  `src/models/ricursive/`).
- **Shapes and dtypes:** zg and zu f16 [S, inter] (the gate and up inner products, sorted); inds u32 or i32 [S] (the
  flat expert ids, S = rows · topk); order u32 [S]; svh_g and svh_u f16 [experts, inter]; suh_d f16 [experts, inter].
  Out: xd f16 [S, inter] (natural order); xsd f32 [S, inter/128].
- **Grid and threadgroup:** `(ceil(S · (inter/128) / ASG) · 32 · ASG, 1, 1)` / `(32 · ASG, 1, 1)`: one simdgroup per
  flattened (position, 128-block) item, S read as `order_shape[0]`; items past the end return.
- **Template args:** INTER, ASG, XSO, then LIMIT (`Geometry.limit`, 1 to 128, after the mid's constants: `midTempl`). No
  rate arguments.
- **Replaces:** `turboexl3_pf_mid_v` in a clamped geometry: the same text with the SwiGLU line (`mid_swiglu`) swapped
  for the clamp (`mid_swiglu_clamp`): the gate clamped from above at LIMIT and up to [−LIMIT, LIMIT], in f32 before the
  SiLU; a NaN passes.
- **Bits:** byte-identical to `turboexl3_pf_mid_v` where no value reaches the limit (the GPU test at limit 128). Its
  clamped values are not held to a reference on their own; the reference test runs root's served mid,
  `turboexl3_pf_t128_midc`, with the same clamp piece.
- **Pinned by:** `src/models/deepseek_v41/clamped_test.zig` "turboexl3 clamped texts: each clamped kernel is its base
  text with the SwiGLU line ..." (CPU: the base text with the one line swapped, nothing else; only the clamp reads
  LIMIT); `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: at a limit no value reaches, every clamped
  launch gives the ..." (GPU: the `mid_vc` arm, aux 8, rows 40 and 400, f32, byte for byte).

### `turboexl3_pf_midc`

<!-- kernel-facts: turboexl3_pf_midc -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:649` (`const Prefill`) |
| inputs | `zg`, `zu`, `inds`, `order`, `svh_g`, `svh_u`, `suh_d` |
| outputs | `xd`, `xsd` |
| source | `text_midc` = `prefill/mid_1` ++ `prefill/mid_2` ++ `prefill/mid_3` ++ `prefill/mid_4` ++ `prefill/mid_swiglu_clamp` ++ `prefill/mid_5` (1755 B, `d4c84b548820`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** core, clamped geometries only: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through
  `moeSwiglu` and `moePrefill`) at more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), only
  with `Config.pf_t128` false (`Prefill.moe` on the mixed option set); with it on (the default) `turboexl3_pf_t128_midc`
  takes its place. Made in place of `turboexl3_pf_mid` when `Geometry.limit` != 0 (`root.moeClamped`): DeepSeek-V4.1's
  routed experts (swiglu_limit 10). The clamp's users: DeepSeek-V4 and V4.1 (swiglu_limit 10) and GLM (the same clamp),
  each through `root.moeClamped`; Ripple's clamped SwiGLU is its own (`turboexl3_ric_swiglu_prep_sc` / `_prep`,
  `src/models/ricursive/`).
- **Shapes and dtypes:** zg and zu f16 [S, inter] (the gate and up inner products, sorted); inds u32 or i32 [S] (the
  flat expert ids, S = rows · topk); order u32 [S]; svh_g and svh_u f16 [experts, inter]; suh_d f16 [experts, inter].
  Out: xd f16 [S, inter] (the down input, sorted, in ALAY's layout); xsd f32 [S, inter/128] (its block sums).
- **Grid and threadgroup:** `((inter/128/BPT) · 32 · BPT, S, 1)` / `(32 · BPT, 1, 1)`: one simdgroup per (128-block of
  inter, sorted position).
- **Template args:** INTER, BPT (`bptFor(inter/128)`: 6 at DeepSeek-V4.1's 2304), ALAY, then LIMIT (`Geometry.limit`, 1
  to 128). No rate arguments.
- **Replaces:** `turboexl3_pf_mid` in a clamped geometry: the same text with the SwiGLU line swapped for the clamp: the
  gate clamped from above at LIMIT and up to [−LIMIT, LIMIT], in f32 before the SiLU; a NaN passes.
- **Bits:** byte-identical to `turboexl3_pf_mid` where no value reaches the limit (GPU test at limit 128). Its clamped
  values are not held to a reference on their own; the reference test runs `turboexl3_pf_t128_midc`, which has the same
  clamp piece.
- **Pinned by:** `src/models/deepseek_v41/clamped_test.zig` "turboexl3 clamped texts: each clamped kernel is its base
  text with the SwiGLU line ..." (CPU); `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: at a limit no
  value reaches, every clamped launch gives the ..." (GPU: the `midc` arm on the served set without the 128-row tiles,
  rows 40 and 400, f32, byte for byte).

### `turboexl3_pf_prep`

<!-- kernel-facts: turboexl3_pf_prep -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:642` (`const Prefill`) |
| inputs | `x`, `inds`, `order`, `suh_g`, `suh_u` |
| outputs | `xg`, `xu`, `inv`, `xsg`, `xsu` |
| source | `text_prep` = `prefill/prep` (1420 B, `cabcf76552d4`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), in both `Prefill.moeT128` (the default)
  and `Prefill.moe` (`Config.pf_t128` false): `Prefill.prep` launches it when `Options.aux` is 0. Qwen3.8-Flash-Next's
  routed experts (512 experts, top-10) through `moe`, and DeepSeek-V4.1's routed experts through `moeClamped`
  (swiglu_limit 10).
- **Shapes and dtypes:** x T [rows, hidden] (T bf16, f16 or f32); inds u32 or i32 [S] (the flat expert ids, S = rows ·
  topk); order u32 [S]; suh_g and suh_u f16 [experts, hidden]. Out: xg and xu f16 [S, hidden] (the gate and up inputs,
  sorted, in ALAY's layout); inv u32 [S] (inv[order[p]] = p); xsg and xsu f32 [S, hidden/128] (each 128-block's sum: the
  deferred map's C term).
- **Grid and threadgroup:** `((hidden/128/BPT) · 32 · BPT, S, 1)` / `(32 · BPT, 1, 1)`: one simdgroup per (128-block of
  hidden, sorted position).
- **Template args:** HID (hidden), TOPK, BPT (`bptFor(hidden/128)`: 5 at Qwen3.8's 2560, 8 at DeepSeek-V4.1's 5120),
  ALAY (0, 1 or 2; `Config.pf_alay`, 2 served). No rate arguments.
- **Replaces:** nothing: it is the base of the chain's prepare: x rounded to f16, times suh, H128, rounded to f16, for
  gate and up, gathered into sorted order.
- **Bits:** within tolerance, as part of the chain: y within 5e-3 of `ref.moeRowDec` (`format.project`'s rounding
  points, `ref.compare`) in f32 on sampled rows (the first, middle and last), and in bf16 within 1.5 × the error of
  Sushi's prefill arm + 2e-3 where Sushi is linked. ALAY only moves slots (the block sums are the same): byte-identical
  to ALAY 0.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: every set with `aux` 0; the byte reference for `turboexl3_pf_prep_v`);
  `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles: the call and every stage give the mixed chain's ..."
  (GPU: both chains run it at the served set; their intermediates compare in slot order through inv);
  `src/core/prefill.zig` "turboexl3 prefill alay: a lane's A fragment elements are one aligned 16-byte (32-deep ..."
  (CPU: it writes the ALAY slot); `src/core/prefill.zig` "turboexl3 prefill metal sources: every kernel as a standalone
  translation unit (TURBOEXL3_METAL_DUMP=1)" (CPU print: `_prep`, `_prep_l`, `_prep_l2`).

### `turboexl3_pf_prep_v`

<!-- kernel-facts: turboexl3_pf_prep_v -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:660` (`const Prefill`) |
| inputs | `x`, `inds`, `order`, `suh_g`, `suh_u` |
| outputs | `xg`, `xu`, `inv`, `xsg`, `xsu` |
| source | `text_prep_v` = `prefill/prep_v` (1808 B, `31abdef09b06`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** not served: `Prefill.init` makes it for every geometry, so `root`'s cache holds it, but nothing in the
  library launches it by default: `Prefill.prep` takes it when `Options.aux` is 4, 8, 16 or 32, and `root.pfOptions`
  leaves `aux` 0. Tests only.
- **Shapes and dtypes:** x T [rows, hidden] (T bf16, f16 or f32); inds u32 or i32 [S] (the flat expert ids, S = rows ·
  topk); order u32 [S]; suh_g and suh_u f16 [experts, hidden] (S read as `order_shape[0]`). Out: xg and xu f16 [S,
  hidden] (natural order); inv u32 [S]; xsg and xsu f32 [S, hidden/128].
- **Grid and threadgroup:** `(ceil(S · (hidden/128) / ASG) · 32 · ASG, 1, 1)` / `(32 · ASG, 1, 1)`: one simdgroup per
  flattened (position, 128-block) item; items past the end return.
- **Template args:** HID, TOPK, ASG (`Options.aux`), XSO (`Options.aux_lane_sums`). ALAY must be 0 with `aux`. No rate
  arguments.
- **Replaces:** `turboexl3_pf_prep` one for one (four consecutive elements per lane: 8-byte loads and stores).
- **Bits:** byte-identical to `turboexl3_pf_prep` with XSO 0; with XSO 1 the block sums run in the lane's order and may
  differ in the last bits.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: aux 8 and aux 32 byte for byte against aux 0); `src/core/prefill.zig` "turboexl3 prefill
  metal sources: every kernel as a standalone translation unit (TURBOEXL3_METAL_DUMP=1)" (CPU print).

### `turboexl3_pf_t128_dnm`

<!-- kernel-facts: turboexl3_pf_t128_dnm -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:593` (`const T128Kernels`) |
| inputs | `tiles`, `x0`, `w0`, `xs0` |
| outputs | `z0` |
| source | `gemm1m_text` = `prefill/gemm_mixed_1` ++ `prefill/gemm_1` ++ `prefill/gemm_mixed_bs` ++ `prefill/gemm_3` ++ `prefill/gemm_op1_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op1_xsz` ++ `prefill/gemm_5` ++ `prefill/gemm_mixed_2` ++ `prefill/gemm_small_1` ++ `prefill/gemm_mixed_index` ++ `prefill/gemm_small_3` ++ `prefill/gemm_small_op1_wx` ++ `prefill/gemm_small_4` ++ `prefill/gemm_small_op1_xsz` ++ `prefill/gemm_small_5` ++ `prefill/gemm_mixed_3` (10642 B, `b717e7a11df7`) |
| header | `header_nax` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` (60754 B, `4c8c1fc745df`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), in `Prefill.moeT128` (`Config.pf_t128`
  on, the default): `gemmDT128` -> `t128Gemm`'s first launch. Qwen3.8-Flash-Next's routed experts (512 experts, top-10)
  through `moe`, and DeepSeek-V4.1's routed experts through `moeClamped` (swiglu_limit 10).
- **Shapes and dtypes:** tiles int32 [tmax + experts, 4] (the remainder table from `turboexl3_pf_t128_route`, tmax =
  `t128MixedRows(S, experts, 64, 1, 64, true)`); x0 = xd f16 [S, inter] (mid's down input, sorted, in ALAY's layout); w0
  the down trellis u16 [experts, inter/16, hidden/16, NHW]; xs0 = xsd f32 [S, inter/128] (its block sums, read with
  DEFER). Out: z0 = zd f16 [S, hidden], the inner products in sorted order, written only at the remainder tiles' rows
  (the rows before each expert's xbase are `turboexl3_pf_t128_dnt`'s).
- **Grid and threadgroup:** `((tmax · (hidden/64/NCB) + ceil(experts · (hidden/32) / 4)) · 128, 1, 1)` / `(128, 1, 1)`
  (bm 64: T 128, NSG 4).
- **Template args:** `turboexl3_pf_gemm1m`'s 23 (NHW (halfwords per 256-weight trellis tile, K = NHW/16: any even 16 to
  128), CODEBOOK (MUL1, MCG), WBITS (the window, 8 to 16), KD and ND (the op's input and output widths), NOPS, BMR
  (`Options.bm`: 64, 128 or 256), DBUF (`dbuf`), PROBE (`probe`, timing probes only), WSP (`wsp`, decode-only producer
  simdgroups), MFIRST (`mma_first`), UNR (`unroll`), SGT (threads / 32), TMAXS (`split` · experts: where the mixed
  table's register-path tiles start; only the mixed launch reads it), FRAGB (`fragb`), INTR (`intr`, needs n % 16 == 0),
  DEFER (`defer_affine`, the deferred MUL1 map, MUL1 only), NBX (KD/128: block sums per row), BKS (the K stage depth),
  NCB (output blocks per threadgroup), SINTR (`sintr`, the register path's integer-rate reader), ALAY (`alay`, needs
  FRAGB) and H16 (`half_rows`, needs FRAGB)), KD = inter, ND = hidden, NOPS 1, SGT 4, BKS 64, NCB 4 (1 where (hidden/64)
  % 4 != 0). Served values: BMR 64, DBUF 1, UNR 1, FRAGB 1, INTR and SINTR 1 where n % 16 == 0 (else 0), DEFER 1 for
  MUL1 packs (0 for MCG), PROBE 0, WSP 0, MFIRST 0, H16 0, ALAY `Config.pf_alay` (2 by default), TMAXS = experts (split
  1). Source `gemm1m_text`, header `header_nax`. Every rate, codebook and window.
- **Replaces:** `turboexl3_pf_gemm1m` over the whole mixed table, split in two with `turboexl3_pf_t128_dnt` beside it: 2
  launches in 1 dependent level (each writes its own output, so neither waits on the other).
- **Bits:** byte-identical, with its partner launch, to the mixed chain's zd (`turboexl3_pf_gemm1m`): bm 128 and bm 64
  run the same per-simdgroup K loop, so every output sums the same 16-deep MMAs in the same order. Through it the chain
  is within tolerance of `ref.moeRowDec` (the bars of `turboexl3_pf_gemm1m`).
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles: the call and every stage give the
  mixed chain's ..." (GPU: zd merged from both launches against the mixed chain's, in slot order, at the served set and
  its alay 0, no-defer, ncb 1 and half_rows variants; routings with every 128-row tile shape and both sides of the cut,
  a K3 case, and 512 experts skewed at 17 to 8192 rows); `src/core/prefill.zig` "turboexl3 prefill 128-row tiles texts:
  both GEMMs' launches are the mixed chain's ..." (CPU: the launches are the mixed chain's kernels; nothing writes an
  input binding); `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: root.moeClamped at
  DeepSeek-V4.1-Flash's routed geometry (5120 ..." (GPU: the served chain end to end at prefill rows 32 and 1328);
  `src/root.zig` "turboexl3 root: the served prefill options follow the config, and the decode form is the config's"
  (CPU: the served set has `t128`); `src/core/prefill.zig` "turboexl3 prefill 128-row tiles metal sources: the route,
  both GEMMs' launches (both ..." (CPU print).

### `turboexl3_pf_t128_dnt`

<!-- kernel-facts: turboexl3_pf_t128_dnt -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:594` (`const T128Kernels`) |
| inputs | `tiles`, `x0`, `w0`, `xs0` |
| outputs | `z0` |
| source | `gemm1_text` = `prefill/gemm_1` ++ `prefill/gemm_2` ++ `prefill/gemm_3` ++ `prefill/gemm_op1_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op1_xsz` ++ `prefill/gemm_5` (6517 B, `dd6225a4d3f1`) |
| header | `header_nax` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` (60754 B, `4c8c1fc745df`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), in `Prefill.moeT128` (`Config.pf_t128`
  on, the default): `gemmDT128` -> `t128Gemm`'s second launch (`t128Cfg`). Qwen3.8-Flash-Next's routed experts (512
  experts, top-10) through `moe`, and DeepSeek-V4.1's routed experts through `moeClamped` (swiglu_limit 10).
- **Shapes and dtypes:** tiles int32 [tq, 4] (tiles128 from `turboexl3_pf_t128_route`, tq = `t128TileRows(S, experts,
  64, true)`); x0 = xd f16 [S, inter] (mid's down input, sorted, in ALAY's layout); w0 the down trellis u16 [experts,
  inter/16, hidden/16, NHW]; xs0 = xsd f32 [S, inter/128] (its block sums, read with DEFER). Out: z0 = zd f16 [S,
  hidden], the inner products in sorted order, written only at the 128-row tiles' rows.
- **Grid and threadgroup:** `(tq · (hidden/64/NCB) · 256, 1, 1)` / `(256, 1, 1)` (`t128Cfg`: bm 128 and small `.off`, 8
  simdgroups).
- **Template args:** the 23 of `gemmCfg` (NHW (halfwords per 256-weight trellis tile, K = NHW/16: any even 16 to 128),
  CODEBOOK (MUL1, MCG), WBITS (the window, 8 to 16), KD and ND (the op's input and output widths), NOPS, BMR
  (`Options.bm`: 64, 128 or 256), DBUF (`dbuf`), PROBE (`probe`, timing probes only), WSP (`wsp`, decode-only producer
  simdgroups), MFIRST (`mma_first`), UNR (`unroll`), SGT (threads / 32), TMAXS (`split` · experts: where the mixed
  table's register-path tiles start; only the mixed launch reads it), FRAGB (`fragb`), INTR (`intr`, needs n % 16 == 0),
  DEFER (`defer_affine`, the deferred MUL1 map, MUL1 only), NBX (KD/128: block sums per row), BKS (the K stage depth),
  NCB (output blocks per threadgroup), SINTR (`sintr`, the register path's integer-rate reader), ALAY (`alay`, needs
  FRAGB) and H16 (`half_rows`, needs FRAGB)) with BMR 128 and SGT 8; KD = inter, ND = hidden, NOPS 1, BKS 64, NCB 4 (1
  where (hidden/64) % 4 != 0); the rest as served (FRAGB 1, DBUF 1, UNR 1, INTR where n % 16 == 0, DEFER for MUL1, ALAY
  `Config.pf_alay`, H16 0; TMAXS and SINTR passed, not read by this body). Source `gemm1_text` (`turboexl3_pf_gemm1`'s).
  Every rate, codebook and window.
- **Replaces:** as `turboexl3_pf_t128_dnm`: with it, `turboexl3_pf_gemm1m` in 2 launches, 1 level.
- **Bits:** byte-identical, with its partner launch, to the mixed chain's zd (`turboexl3_pf_gemm1m`): bm 128 and bm 64
  run the same per-simdgroup K loop, so every output sums the same 16-deep MMAs in the same order. Through it the chain
  is within tolerance of `ref.moeRowDec` (the bars of `turboexl3_pf_gemm1m`).
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles: the call and every stage give the
  mixed chain's ..." (GPU: zd merged from both launches against the mixed chain's, in slot order, at the served set and
  its alay 0, no-defer, ncb 1 and half_rows variants; routings with every 128-row tile shape and both sides of the cut,
  a K3 case, and 512 experts skewed at 17 to 8192 rows); `src/core/prefill.zig` "turboexl3 prefill 128-row tiles texts:
  both GEMMs' launches are the mixed chain's ..." (CPU: the launches are the mixed chain's kernels; nothing writes an
  input binding); `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: root.moeClamped at
  DeepSeek-V4.1-Flash's routed geometry (5120 ..." (GPU: the served chain end to end at prefill rows 32 and 1328);
  `src/root.zig` "turboexl3 root: the served prefill options follow the config, and the decode form is the config's"
  (CPU: the served set has `t128`); `src/core/prefill.zig` "turboexl3 prefill 128-row tiles metal sources: the route,
  both GEMMs' launches (both ..." (CPU print).

### `turboexl3_pf_t128_fin`

<!-- kernel-facts: turboexl3_pf_t128_fin -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:601` (`const T128Kernels`) |
| inputs | `zd`, `inds`, `inv`, `scores`, `svh_d`, `zdt`, `xbase` |
| outputs | `y` |
| source | `text_finish_t128` = `prefill/finish_1` ++ `prefill/finish_t128_pick` ++ `prefill/finish_2` ++ `prefill/finish_t128_load` ++ `prefill/finish_4` (1099 B, `1aa1ff185322`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), in `Prefill.moeT128` (`Config.pf_t128`
  on, the default): `finishT128`. Qwen3.8-Flash-Next's routed experts (512 experts, top-10) through `moe`, and
  DeepSeek-V4.1's routed experts through `moeClamped` (swiglu_limit 10); the finish takes no clamp.
- **Shapes and dtypes:** zd f16 [S, hidden] (`turboexl3_pf_t128_dnm`'s); inds u32 or i32 [S] (the flat expert ids, S =
  rows · topk); inv u32 [S]; scores bf16, f16 or f32 [S]; svh_d f16 [experts, hidden]; zdt f16 [S, hidden]
  (`turboexl3_pf_t128_dnt`'s); xbase int32 [experts]. A slot at sorted position p of expert e reads zdt when p <
  xbase[e], else zd. Out: y OUT_T [rows, hidden] (OUT_T = T).
- **Grid and threadgroup:** `((hidden/128/BPT) · 32 · BPT, rows, 1)` / `(32 · BPT, 1, 1)`.
- **Template args:** HID, TOPK, BPT (`bptFor(hidden/128)`), OUT_T: `turboexl3_pf_fin`'s. Every rate.
- **Replaces:** `turboexl3_pf_fin` in the 128-row chain (each slot's row read from the launch that computed it; 1
  launch, 1 level).
- **Bits:** byte-identical to `turboexl3_pf_fin` on the mixed chain (y in bf16 and f32, stage and whole call).
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles: the call and every stage give the
  mixed chain's ..." (GPU: y bf16 and f32 against the mixed chain's finish, and the whole call); `src/core/prefill.zig`
  "turboexl3 prefill 128-row tiles texts: both GEMMs' launches are the mixed chain's ..." (CPU: `text_finish_t128`
  without the row choice is `text_finish`; its inputs are fin's plus zdt and xbase);
  `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: at a limit no value reaches, every clamped launch
  gives the ..." (GPU: root at rows 40 and 400, clamped against unclamped); `src/models/deepseek_v41/clamped_test.zig`
  "turboexl3 gpu clamped: root.moeClamped at DeepSeek-V4.1-Flash's routed geometry (5120 ..." (GPU);
  `src/core/prefill.zig` "turboexl3 prefill 128-row tiles metal sources: the route, both GEMMs' launches (both ..." (CPU
  print).

### `turboexl3_pf_t128_gum`

<!-- kernel-facts: turboexl3_pf_t128_gum -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:591` (`const T128Kernels`) |
| inputs | `tiles`, `x0`, `x1`, `w0`, `w1`, `xs0`, `xs1` |
| outputs | `z0`, `z1` |
| source | `gemm2m_text` = `prefill/gemm_mixed_1` ++ `prefill/gemm_1` ++ `prefill/gemm_mixed_bs` ++ `prefill/gemm_3` ++ `prefill/gemm_op2_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op2_xsz` ++ `prefill/gemm_5` ++ `prefill/gemm_mixed_2` ++ `prefill/gemm_small_1` ++ `prefill/gemm_mixed_index` ++ `prefill/gemm_small_3` ++ `prefill/gemm_small_op2_wx` ++ `prefill/gemm_small_4` ++ `prefill/gemm_small_op2_xsz` ++ `prefill/gemm_small_5` ++ `prefill/gemm_mixed_3` (10642 B, `e19ee178960e`) |
| header | `header_nax` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` (60754 B, `4c8c1fc745df`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), in `Prefill.moeT128` (`Config.pf_t128`
  on, the default): `gemmGUT128` -> `t128Gemm`'s first launch. Qwen3.8-Flash-Next's routed experts (512 experts, top-10)
  through `moe`, and DeepSeek-V4.1's routed experts through `moeClamped` (swiglu_limit 10).
- **Shapes and dtypes:** tiles int32 [tmax + experts, 4] (the remainder table, tmax = `t128MixedRows(S, experts, 64, 1,
  64, true)`); x0 = xg and x1 = xu f16 [S, hidden] (prep's gate and up inputs, sorted, in ALAY's layout); w0 and w1 the
  gate and up trellises u16 [experts, hidden/16, inter/16, NHW]; xs0 = xsg and xs1 = xsu f32 [S, hidden/128] (the block
  sums, read with DEFER). Out: z0 = zg and z1 = zu f16 [S, inter], the inner products in sorted order, written only at
  the remainder tiles' rows.
- **Grid and threadgroup:** `((tmax · 2 · (inter/64) + ceil(experts · 2 · (inter/32) / 4)) · 128, 1, 1)` / `(128, 1, 1)`
  (bm 64).
- **Template args:** `turboexl3_pf_gemm2m`'s 23 (NHW (halfwords per 256-weight trellis tile, K = NHW/16: any even 16 to
  128), CODEBOOK (MUL1, MCG), WBITS (the window, 8 to 16), KD and ND (the op's input and output widths), NOPS, BMR
  (`Options.bm`: 64, 128 or 256), DBUF (`dbuf`), PROBE (`probe`, timing probes only), WSP (`wsp`, decode-only producer
  simdgroups), MFIRST (`mma_first`), UNR (`unroll`), SGT (threads / 32), TMAXS (`split` · experts: where the mixed
  table's register-path tiles start; only the mixed launch reads it), FRAGB (`fragb`), INTR (`intr`, needs n % 16 == 0),
  DEFER (`defer_affine`, the deferred MUL1 map, MUL1 only), NBX (KD/128: block sums per row), BKS (the K stage depth),
  NCB (output blocks per threadgroup), SINTR (`sintr`, the register path's integer-rate reader), ALAY (`alay`, needs
  FRAGB) and H16 (`half_rows`, needs FRAGB)), KD = hidden, ND = inter, NOPS 2, SGT 4, BKS 64, NCB 1. Served values: BMR
  64, DBUF 1, UNR 1, FRAGB 1, INTR and SINTR 1 where n % 16 == 0 (else 0), DEFER 1 for MUL1 packs (0 for MCG), PROBE 0,
  WSP 0, MFIRST 0, H16 0, ALAY `Config.pf_alay` (2 by default), TMAXS = experts (split 1). Source `gemm2m_text`, header
  `header_nax`. Every rate, codebook and window.
- **Replaces:** `turboexl3_pf_gemm2m` over the whole mixed table, split in two with `turboexl3_pf_t128_gut` beside it: 2
  launches in 1 dependent level.
- **Bits:** byte-identical, with its partner launch, to the mixed chain's zg and zu (`turboexl3_pf_gemm2m`): bm 128 and
  bm 64 run the same per-simdgroup K loop, so every output sums the same 16-deep MMAs in the same order. Through it the
  chain is within tolerance of `ref.moeRowDec` (the bars of `turboexl3_pf_gemm2m`).
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles: the call and every stage give the
  mixed chain's ..." (GPU: zg and zu merged from both launches against the mixed chain's, in slot order, at the served
  set and its alay 0, no-defer, ncb 1 and half_rows variants; routings with every 128-row tile shape and both sides of
  the cut, a K3 case, and 512 experts skewed at 17 to 8192 rows); `src/core/prefill.zig` "turboexl3 prefill 128-row
  tiles texts: both GEMMs' launches are the mixed chain's ..." (CPU: the launches are the mixed chain's kernels; nothing
  writes an input binding); `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: root.moeClamped at
  DeepSeek-V4.1-Flash's routed geometry (5120 ..." (GPU: the served chain end to end at prefill rows 32 and 1328);
  `src/root.zig` "turboexl3 root: the served prefill options follow the config, and the decode form is the config's"
  (CPU: the served set has `t128`); `src/core/prefill.zig` "turboexl3 prefill 128-row tiles metal sources: the route,
  both GEMMs' launches (both ..." (CPU print).

### `turboexl3_pf_t128_gut`

<!-- kernel-facts: turboexl3_pf_t128_gut -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:592` (`const T128Kernels`) |
| inputs | `tiles`, `x0`, `x1`, `w0`, `w1`, `xs0`, `xs1` |
| outputs | `z0`, `z1` |
| source | `gemm2_text` = `prefill/gemm_1` ++ `prefill/gemm_2` ++ `prefill/gemm_3` ++ `prefill/gemm_op2_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op2_xsz` ++ `prefill/gemm_5` (6517 B, `f34c9fcb0bd3`) |
| header | `header_nax` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` (60754 B, `4c8c1fc745df`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), in `Prefill.moeT128` (`Config.pf_t128`
  on, the default): `gemmGUT128` -> `t128Gemm`'s second launch (`t128Cfg`). Qwen3.8-Flash-Next's routed experts (512
  experts, top-10) through `moe`, and DeepSeek-V4.1's routed experts through `moeClamped` (swiglu_limit 10).
- **Shapes and dtypes:** tiles int32 [tq, 4] (tiles128, tq = `t128TileRows(S, experts, 64, true)`); x0 = xg and x1 = xu
  f16 [S, hidden] (prep's gate and up inputs, sorted, in ALAY's layout); w0 and w1 the gate and up trellises u16
  [experts, hidden/16, inter/16, NHW]; xs0 = xsg and xs1 = xsu f32 [S, hidden/128] (the block sums, read with DEFER).
  Out: z0 = zg and z1 = zu f16 [S, inter], the inner products in sorted order, written only at the 128-row tiles' rows.
- **Grid and threadgroup:** `(tq · 2 · (inter/64) · 256, 1, 1)` / `(256, 1, 1)` (bm 128, 8 simdgroups).
- **Template args:** the 23 of `gemmCfg` (NHW (halfwords per 256-weight trellis tile, K = NHW/16: any even 16 to 128),
  CODEBOOK (MUL1, MCG), WBITS (the window, 8 to 16), KD and ND (the op's input and output widths), NOPS, BMR
  (`Options.bm`: 64, 128 or 256), DBUF (`dbuf`), PROBE (`probe`, timing probes only), WSP (`wsp`, decode-only producer
  simdgroups), MFIRST (`mma_first`), UNR (`unroll`), SGT (threads / 32), TMAXS (`split` · experts: where the mixed
  table's register-path tiles start; only the mixed launch reads it), FRAGB (`fragb`), INTR (`intr`, needs n % 16 == 0),
  DEFER (`defer_affine`, the deferred MUL1 map, MUL1 only), NBX (KD/128: block sums per row), BKS (the K stage depth),
  NCB (output blocks per threadgroup), SINTR (`sintr`, the register path's integer-rate reader), ALAY (`alay`, needs
  FRAGB) and H16 (`half_rows`, needs FRAGB)) with BMR 128 and SGT 8; KD = hidden, ND = inter, NOPS 2, BKS 64, NCB 1; the
  rest as served. Source `gemm2_text` (`turboexl3_pf_gemm2`'s). Every rate, codebook and window.
- **Replaces:** as `turboexl3_pf_t128_gum`: with it, `turboexl3_pf_gemm2m` in 2 launches, 1 level.
- **Bits:** byte-identical, with its partner launch, to the mixed chain's zg and zu (`turboexl3_pf_gemm2m`): bm 128 and
  bm 64 run the same per-simdgroup K loop, so every output sums the same 16-deep MMAs in the same order. Through it the
  chain is within tolerance of `ref.moeRowDec` (the bars of `turboexl3_pf_gemm2m`).
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles: the call and every stage give the
  mixed chain's ..." (GPU: zg and zu merged from both launches against the mixed chain's, in slot order, at the served
  set and its alay 0, no-defer, ncb 1 and half_rows variants; routings with every 128-row tile shape and both sides of
  the cut, a K3 case, and 512 experts skewed at 17 to 8192 rows); `src/core/prefill.zig` "turboexl3 prefill 128-row
  tiles texts: both GEMMs' launches are the mixed chain's ..." (CPU: the launches are the mixed chain's kernels; nothing
  writes an input binding); `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: root.moeClamped at
  DeepSeek-V4.1-Flash's routed geometry (5120 ..." (GPU: the served chain end to end at prefill rows 32 and 1328);
  `src/root.zig` "turboexl3 root: the served prefill options follow the config, and the decode form is the config's"
  (CPU: the served set has `t128`); `src/core/prefill.zig` "turboexl3 prefill 128-row tiles metal sources: the route,
  both GEMMs' launches (both ..." (CPU print).

### `turboexl3_pf_t128_mid`

<!-- kernel-facts: turboexl3_pf_t128_mid -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:596` (`const T128Kernels`) |
| inputs | `zg`, `zu`, `inds`, `order`, `svh_g`, `svh_u`, `suh_d`, `zgt`, `zut`, `xbase` |
| outputs | `xd`, `xsd` |
| source | `text_mid_t128` = `prefill/mid_1` ++ `prefill/mid_t128_pick` ++ `prefill/mid_2` ++ `prefill/mid_t128_loads` ++ `prefill/mid_4` ++ `prefill/mid_swiglu` ++ `prefill/mid_5` (1531 B, `9fa53afb15db`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), in `Prefill.moeT128` (`Config.pf_t128`
  on, the default): `midT128`, for an unclamped geometry (`Geometry.limit` 0). Qwen3.8-Flash-Next's routed experts
  through `moe`; a clamped geometry makes `turboexl3_pf_t128_midc` instead.
- **Shapes and dtypes:** zg and zu f16 [S, inter] (the gate and up inner products, sorted); inds u32 or i32 [S] (the
  flat expert ids, S = rows · topk); order u32 [S]; svh_g and svh_u f16 [experts, inter]; suh_d f16 [experts, inter];
  zgt and zut f16 [S, inter] (`turboexl3_pf_t128_gut`'s); xbase int32 [experts]. zg and zu are
  `turboexl3_pf_t128_gum`'s; position p of expert e reads zgt and zut when p < xbase[e]. Out: xd f16 [S, inter] (the
  down input, sorted, in ALAY's layout); xsd f32 [S, inter/128] (its block sums).
- **Grid and threadgroup:** `((inter/128/BPT) · 32 · BPT, S, 1)` / `(32 · BPT, 1, 1)`: one simdgroup per (128-block of
  inter, sorted position).
- **Template args:** INTER, BPT (`bptFor(inter/128)`: 5 at Qwen3.8's 640), ALAY (`Config.pf_alay`, 2 served). No rate
  arguments.
- **Replaces:** `turboexl3_pf_mid` in the 128-row chain (each row read from the launch that computed it; 1 launch, 1
  level).
- **Bits:** byte-identical to `turboexl3_pf_mid` on the mixed chain (xd and xsd).
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles: the call and every stage give the
  mixed chain's ..." (GPU: xd and xsd against the mixed chain's, slot order); `src/core/prefill.zig` "turboexl3 prefill
  128-row tiles texts: both GEMMs' launches are the mixed chain's ..." (CPU: without the row choice and its two loads
  the text is `text_mid`); `src/models/deepseek_v41/clamped_test.zig` "turboexl3 clamped texts: each clamped kernel is
  its base text with the SwiGLU line ..." (CPU: the base of `turboexl3_pf_t128_midc`);
  `src/models/deepseek_v41/clamped_test.zig` "turboexl3 gpu clamped: at a limit no value reaches, every clamped launch
  gives the ..." (GPU: the unclamped side of root's prefill rows 40 and 400); `src/core/prefill.zig` "turboexl3 prefill
  128-row tiles metal sources: the route, both GEMMs' launches (both ..." (CPU print).

### `turboexl3_pf_t128_midc`

<!-- kernel-facts: turboexl3_pf_t128_midc -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:598` (`const T128Kernels`) |
| inputs | `zg`, `zu`, `inds`, `order`, `svh_g`, `svh_u`, `suh_d`, `zgt`, `zut`, `xbase` |
| outputs | `xd`, `xsd` |
| source | `text_mid_t128c` = `prefill/mid_1` ++ `prefill/mid_t128_pick` ++ `prefill/mid_2` ++ `prefill/mid_t128_loads` ++ `prefill/mid_4` ++ `prefill/mid_swiglu_clamp` ++ `prefill/mid_5` (1884 B, `84c6b6a3cdb5`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), in `Prefill.moeT128` (`Config.pf_t128`
  on, the default): `midT128` for a clamped geometry (`Geometry.limit` != 0): `root.moeClamped`, DeepSeek-V4.1's routed
  experts (swiglu_limit 10). The clamp's users: DeepSeek-V4 and V4.1 (swiglu_limit 10) and GLM (the same clamp), each
  through `root.moeClamped`; Ripple's clamped SwiGLU is its own (`turboexl3_ric_swiglu_prep_sc` / `_prep`,
  `src/models/ricursive/`).
- **Shapes and dtypes:** zg and zu f16 [S, inter] (the gate and up inner products, sorted); inds u32 or i32 [S] (the
  flat expert ids, S = rows · topk); order u32 [S]; svh_g and svh_u f16 [experts, inter]; suh_d f16 [experts, inter];
  zgt and zut f16 [S, inter] (`turboexl3_pf_t128_gut`'s); xbase int32 [experts]. zg and zu are
  `turboexl3_pf_t128_gum`'s; position p of expert e reads zgt and zut when p < xbase[e]. Out: xd f16 [S, inter] (the
  down input, sorted, in ALAY's layout); xsd f32 [S, inter/128] (its block sums).
- **Grid and threadgroup:** `((inter/128/BPT) · 32 · BPT, S, 1)` / `(32 · BPT, 1, 1)`: one simdgroup per (128-block of
  inter, sorted position).
- **Template args:** INTER, BPT (`bptFor(inter/128)`: 6 at DeepSeek-V4.1's 2304), ALAY, then LIMIT (`Geometry.limit`, 1
  to 128). No rate arguments.
- **Replaces:** `turboexl3_pf_midc` in the 128-row chain; it is `turboexl3_pf_t128_mid` with the SwiGLU line swapped for
  the clamp: the gate clamped from above at LIMIT and up to [−LIMIT, LIMIT], in f32 before the SiLU; a NaN passes.
- **Bits:** byte-identical to `turboexl3_pf_t128_mid` where no value reaches the limit (GPU, limit 128). Clamped, the
  served chain through `root.moeClamped` is within 5e-3 of `ref.moeRowClamped`: `ref.compare` in f32, and in bf16 beyond
  the store's rounding (`ref.compareRounded`, half an ulp: 2^-8), at K2, K3 and K6, prefill rows 32 and 1328, with at
  least 1% of gates and ups clamped on the sampled rows.
- **Pinned by:** `src/models/deepseek_v41/clamped_test.zig` "turboexl3 clamped texts: each clamped kernel is its base
  text with the SwiGLU line ..." (CPU: one line swapped, nothing else); `src/models/deepseek_v41/clamped_test.zig`
  "turboexl3 gpu clamped: at a limit no value reaches, every clamped launch gives the ..." (GPU: root's prefill rows 40
  and 400, bf16 and f32, byte for byte against the unclamped call); `src/models/deepseek_v41/clamped_test.zig`
  "turboexl3 gpu clamped: root.moeClamped at DeepSeek-V4.1-Flash's routed geometry (5120 ..." (GPU:
  DeepSeek-V4.1-Flash's routed geometry, 5120 × 2304, 384 experts, top-6, against the clamped reference).

### `turboexl3_pf_t128_route`

<!-- kernel-facts: turboexl3_pf_t128_route -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:590` (`const T128Kernels`) |
| inputs | `inds` |
| outputs | `order`, `tiles`, `tiles128`, `xbase` |
| source | `text_froute_t128` = `prefill/froute_t128_1` ++ `prefill/froute_count` ++ `prefill/froute_t128_2` (4646 B, `107b10c53c14`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moe`, `root.moeWithShared` and `root.moeClamped` (through `moeSwiglu` and `moePrefill`) at
  more than `DECODE_ROWS_MAX` (16) rows (`Config.prefill` and `Config.nax` on), in `Prefill.moeT128` (`Config.pf_t128`
  on, the default): `routeT128`. Qwen3.8-Flash-Next's routed experts (512 experts, top-10) through `moe`, and
  DeepSeek-V4.1's routed experts through `moeClamped` (swiglu_limit 10).
- **Shapes and dtypes:** inds u32 or i32 [S] (the flat expert ids, S = rows · topk). Out: order u32 [S] (froute's count
  and scatter); tiles int32 [tmax + SPL · experts, 4] (the mixed table over each expert's remainder,
  `text_table_mixed`'s layout, tmax = `t128MixedRows(S, experts, BMT, SPL, CUT, true)`); tiles128 int32 [tq, 4] (tq =
  `t128TileRows(S, experts, CUT, true)`); xbase int32 [experts] (each expert's first remainder position). Unused table
  rows 0.
- **Grid and threadgroup:** `(1024, 1, 1)` / `(1024, 1, 1)`: one threadgroup, thread e owns expert e.
- **Template args:** EXP (experts), BMT (`bm`: 64), SPL (`split`: 1), CUT (`t128_cut`, 64: past an expert's full 128-row
  tiles, a remainder of more than min(CUT, 127) rows takes one part-full 128-row tile), T128 (1; 0 would build no
  128-row tiles and only the mixed table). No rate arguments.
- **Replaces:** MLX's argsort + `turboexl3_pf_table_m` (2 launches, 2 dependent levels) with 1, as
  `turboexl3_pf_froute_m` does, and the same launch also builds the 128-row table and xbase.
- **Bits:** its tables equal the CPU rule `prefill.t128Tables` (and the sizes TQ and TBIG equal `t128TileRows` and
  `t128MixedRows`); the chain built on them gives the mixed chain's bytes.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill 128-row tiles: the call and every stage give the
  mixed chain's ..." (GPU: the tables against the CPU rule on each case's counts, and the chain's bytes);
  `src/core/prefill.zig` "turboexl3 prefill 128-row tiles texts: both GEMMs' launches are the mixed chain's ..." (CPU:
  `froute_count` verbatim, the 128-row rule and bounds in the text, four outputs ending in xbase);
  `src/core/prefill.zig` "turboexl3 prefill 128-row tiles tables: every sorted position in exactly one tile; ..." (CPU:
  every sorted position in exactly one tile, the bounds hold); `src/core/prefill.zig` "turboexl3 prefill 128-row tiles
  options: t128 needs bm 64, the mixed table at split ..." (CPU: `t128` needs bm 64, mixed, split 1, froute, aux 0; the
  row bounds' values); `src/core/prefill.zig` "turboexl3 prefill 128-row tiles metal sources: the route, both GEMMs'
  launches (both ..." (CPU print).

### `turboexl3_pf_table`

<!-- kernel-facts: turboexl3_pf_table -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:641` (`const Prefill`) |
| inputs | `inds`, `order` |
| outputs | `tiles` |
| source | `text_table` = `prefill/table_1` ++ `prefill/table_bsearch` ++ `prefill/table_2` (1606 B, `ca3c8b5cc5f5`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** not served: `Prefill.init` makes it for every geometry, so `root`'s cache holds it, but nothing in the
  library launches it by default: `Prefill.route` takes it after MLX's argsort with `Options.froute` false and small
  `.off` or `.all`, and `root.pfOptions` always sets `froute`. Tests only.
- **Shapes and dtypes:** inds u32 or i32 [S] (the flat expert ids, S = rows · topk); order u32 [S] (MLX's argsort of
  inds: sorted position to slot). Out: tiles int32 [ceil(S/BMT) + experts, 4] = (expert, first sorted position, rows, 0)
  in expert order, then rows 0.
- **Grid and threadgroup:** `(1024, 1, 1)` / `(1024, 1, 1)`: one threadgroup, thread e owns expert e.
- **Template args:** EXP (experts), BMT (`Options.tileRows`: `bm`, or 32 with small `.all`), SPL (passed, not read by
  this text). No rate arguments.
- **Replaces:** nothing: it is the base of the route (after MLX's argsort: a binary search per expert for its run, an
  exclusive scan, the tile table).
- **Bits:** integer outputs; the base chain built on it is within tolerance: y within 5e-3 of `ref.moeRowDec`
  (`format.project`'s rounding points, `ref.compare`) in f32 on sampled rows (the first, middle and last), and in bf16
  within 1.5 × the error of Sushi's prefill arm + 2e-3 where Sushi is linked.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: `.{}`, `.{ .small = .all }` and every non-mixed set without `froute`; the byte reference
  for `turboexl3_pf_froute`); `src/core/prefill.zig` "turboexl3 prefill metal sources: every kernel as a standalone
  translation unit (TURBOEXL3_METAL_DUMP=1)" (CPU print).

### `turboexl3_pf_table_m`

<!-- kernel-facts: turboexl3_pf_table_m -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:655` (`const Prefill`) |
| inputs | `inds`, `order` |
| outputs | `tiles` |
| source | `text_table_mixed` = `prefill/table_mixed_1` ++ `prefill/table_bsearch` ++ `prefill/table_mixed_2` (2518 B, `1a7088e45088`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** not served: `Prefill.init` makes it for every geometry, so `root`'s cache holds it, but nothing in the
  library launches it by default: `Prefill.route` takes it after MLX's argsort with `Options.froute` false and small
  `.mixed`, and `root.pfOptions` always sets `froute`. Tests only.
- **Shapes and dtypes:** inds u32 or i32 [S] (the flat expert ids, S = rows · topk); order u32 [S] (MLX's argsort). Out:
  tiles int32 [ceil(S/BMT) + experts + SPL · experts, 4]: the BMT-row tiles in expert order from 0, the register-path
  tiles of at most 32 rows from TBIG = ceil(S/BMT) + experts; unused rows 0.
- **Grid and threadgroup:** `(1024, 1, 1)` / `(1024, 1, 1)`: one threadgroup, thread e owns expert e.
- **Template args:** EXP (experts), BMT (`bm`), SPL (`Options.split`, 1 or 2: a remainder of at most 32 · SPL rows
  becomes SPL or fewer register-path tiles). No rate arguments.
- **Replaces:** nothing: it is the mixed set's table (after MLX's argsort), the base `turboexl3_pf_froute_m` replaces.
- **Bits:** integer outputs; the mixed chain built on it is within tolerance: y within 5e-3 of `ref.moeRowDec`
  (`format.project`'s rounding points, `ref.compare`) in f32 on sampled rows (the first, middle and last), and in bf16
  within 1.5 × the error of Sushi's prefill arm + 2e-3 where Sushi is linked.
- **Pinned by:** `src/core/prefill_test.zig` "turboexl3 gpu prefill: the fused chain equals the host reference and
  Sushi's prefill ..." (GPU: the mixed sets without `froute`, split 1 and 2; the byte reference for
  `turboexl3_pf_froute_m`); `src/core/prefill.zig` "turboexl3 prefill metal sources: every kernel as a standalone
  translation unit (TURBOEXL3_METAL_DUMP=1)" (CPU print).

### `turboexl3_pf_mid_vco`

<!-- kernel-facts: turboexl3_pf_mid_vco -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:666` (`const Prefill`) |
| inputs | `zg`, `zu`, `inds`, `order`, `svh_g`, `svh_u`, `suh_d` |
| outputs | `xd`, `xsd` |
| source | `text_mid_vco` = `prefill/mid_v_1` ++ `prefill/mid_swiglu_clamp_out` ++ `prefill/mid_v_2` (2168 B, `a085e4763004`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** core: as `turboexl3_pf_midco`, in the vector-layout arm (`Options.aux` 8). Ling-3.1-flash
  (`models/ling/api.zig`): its routed experts on layers 65 to 71 (`expert_swiglu_limit_list` 10), whose SwiGLU is
  clamped after the SiLU: min(silu(g), L) · clamp(u, −L, L), exllamav3's `act_limit` with SiLU (inclusionAI's vLLM and
  SGLang the same).
- **Shapes and dtypes:** as `turboexl3_pf_mid_vc`.
- **Grid and threadgroup:** as `turboexl3_pf_mid_v`.
- **Template args:** the mid's three, then LIMIT (an integer 1 to 128, Ling-3.1's 10): the gate's SiLU clamped from
  above at LIMIT and up to [−LIMIT, LIMIT], in f32 on the finished f16 values; a NaN passes through.
- **Replaces:** `turboexl3_pf_mid_v`'s launch with Ling's clamp.
- **Bits:** `turboexl3_pf_mid_v`'s statements but the SwiGLU line; where no value reaches the limit, its bytes.
- **Pinned by:** `src/models/ling/ling_test.zig` "turboexl3 ling texts: each Ling kernel is its base text with the
  SwiGLU line swapped for Ling's clamp ..." (CPU: this text is its base with the one swap, nothing else; only a clamp
  reads LIMIT).

### `turboexl3_pf_midco`

<!-- kernel-facts: turboexl3_pf_midco -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:651` (`const Prefill`) |
| inputs | `zg`, `zu`, `inds`, `order`, `svh_g`, `svh_u`, `suh_d` |
| outputs | `xd`, `xsd` |
| source | `text_midco` = `prefill/mid_1` ++ `prefill/mid_2` ++ `prefill/mid_3` ++ `prefill/mid_4` ++ `prefill/mid_swiglu_clamp_out` ++ `prefill/mid_5` (1811 B, `31aaa1a3940b`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moeLing`'s prefill chain with a limit where the mixed chain runs without the 128-row tiles
  (`prefill.Prefill` at a geometry with `limit` > 0 and `clamp` `silu_out`); `root` serves the 128-row tiles' mid
  (`turboexl3_pf_t128_midco`). Ling-3.1-flash (`models/ling/api.zig`): its routed experts on layers 65 to 71
  (`expert_swiglu_limit_list` 10), whose SwiGLU is clamped after the SiLU: min(silu(g), L) · clamp(u, −L, L),
  exllamav3's `act_limit` with SiLU (inclusionAI's vLLM and SGLang the same).
- **Shapes and dtypes:** as `turboexl3_pf_midc`.
- **Grid and threadgroup:** as `turboexl3_pf_mid`.
- **Template args:** the mid's three, then LIMIT (an integer 1 to 128, Ling-3.1's 10): the gate's SiLU clamped from
  above at LIMIT and up to [−LIMIT, LIMIT], in f32 on the finished f16 values; a NaN passes through.
- **Replaces:** `turboexl3_pf_mid`'s launch with Ling's clamp.
- **Bits:** `turboexl3_pf_mid`'s statements but the SwiGLU line; where no value reaches the limit, its bytes.
- **Pinned by:** `src/models/ling/ling_test.zig` "turboexl3 ling texts: each Ling kernel is its base text with the
  SwiGLU line swapped for Ling's clamp ..." (CPU: this text is its base with the one swap, nothing else; only a clamp
  reads LIMIT).

### `turboexl3_pf_t128_midco`

<!-- kernel-facts: turboexl3_pf_t128_midco -->

| fact | from the code |
|---|---|
| made at | `src/core/prefill.zig:600` (`const T128Kernels`) |
| inputs | `zg`, `zu`, `inds`, `order`, `svh_g`, `svh_u`, `suh_d`, `zgt`, `zut`, `xbase` |
| outputs | `xd`, `xsd` |
| source | `text_mid_t128co` = `prefill/mid_1` ++ `prefill/mid_t128_pick` ++ `prefill/mid_2` ++ `prefill/mid_t128_loads` ++ `prefill/mid_4` ++ `prefill/mid_swiglu_clamp_out` ++ `prefill/mid_5` (1940 B, `c288bc44ac1b`) |
| header | `header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** core: `root.moeLing` at prefill rows (more than 16) with a limit: the served chain's mid (the 128-row
  tiles), built in `turboexl3_pf_t128_mid`'s place at a geometry with `limit` > 0 and `clamp` `silu_out`. Ling-3.1-flash
  (`models/ling/api.zig`): its routed experts on layers 65 to 71 (`expert_swiglu_limit_list` 10), whose SwiGLU is
  clamped after the SiLU: min(silu(g), L) · clamp(u, −L, L), exllamav3's `act_limit` with SiLU (inclusionAI's vLLM and
  SGLang the same).
- **Shapes and dtypes:** as `turboexl3_pf_t128_midc`.
- **Grid and threadgroup:** as `turboexl3_pf_t128_mid`.
- **Template args:** the mid's three, then LIMIT (an integer 1 to 128, Ling-3.1's 10): the gate's SiLU clamped from
  above at LIMIT and up to [−LIMIT, LIMIT], in f32 on the finished f16 values; a NaN passes through.
- **Replaces:** `turboexl3_pf_t128_mid`'s launch with Ling's clamp.
- **Bits:** byte-identical to `turboexl3_pf_t128_mid` where no value reaches the limit; with the clamp acting, within
  5e-3 of `ref.moeRowClampedAt` at `silu_out` at Ling-3.1-flash's routed geometry (prefill rows 32 and 1328).
- **Pinned by:** `src/models/ling/ling_test.zig` "turboexl3 ling texts: each Ling kernel is its base text with the
  SwiGLU line swapped for Ling's clamp ..." (CPU: this text is its base with the one swap, nothing else; only a clamp
  reads LIMIT); `src/models/ling/ling_test.zig` "turboexl3 gpu ling: root.moeLing at Ling-3.1-flash's routed geometry
  ..." (GPU: at limit 128 with x ~ N(0, 1) `moe`'s bytes at rows 1, 16 and 400, bf16 and f32; at limits 0 and 10 with x
  ~ N(0, 8^2) within 5e-3 of `ref.moeRowClampedAt` at `silu_out` on sampled rows of 1, 4, 16, 32 and 1328, n 40 and 32).

### `turboexl3_finish`

<!-- kernel-facts: turboexl3_finish -->

| fact | from the code |
|---|---|
| made at | `src/core/unfused.zig:70` (`const Steps`) |
| inputs | `z`, `slots`, `svh` |
| outputs | `y` |
| source | `text_finish` = `unfused/finish` (878 B, `d393f727d10e`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core, not served: `unfused.Steps.moe`, the unfused routed decode chain (11 launches per layer; nothing in
  the library or its benchmark runs it): the gate's, up's and down's finishes.
- **Shapes and dtypes:** z f32 [RK, DIM] (RK = rows · TOPK row-slots); slots [RK] (each row-slot's expert, read as
  uint); svh f16 [E, DIM] (the bank's per-expert scales). Out: y f32 [RK, DIM].
- **Grid and threadgroup:** `((DIM/128) · 32, RK, 1)` / `(32, 1, 1)`.
- **Template args:** DIM (inter for gate and up, hidden for down).
- **Replaces:** nothing: a step of the unfused chain, the math of `decode.zig`'s form 1 split into separate launches.
- **Bits:** `format.project`'s finish rounding points (f16 z, H128, svh, f16), output f32. The chain it is a step of
  gives form 1's words (`turboexl3_gu`, `turboexl3_down`) and is held within 5e-3 of `ref.moeRowDec`.
- **Pinned by:** `src/core/chains_test.zig` "turboexl3 gpu unfused: the eleven-launch chain gives form 1's words and
  equals the host reference within the decode chain's tolerance, at form 1's cases (TURBOEXL3_GPU=1)" (GPU: the whole
  chain word for word against form 1 (`decode.Kernels`) and within 5e-3 of `ref.moeRowDec`, at 512 x 256 and 2560 x 640,
  MUL1 and MCG, rows 1 to 8); `src/core/unfused.zig` "turboexl3 unfused texts: the inputs each signature names are read,
  the outputs written" (CPU, the texts).

### `turboexl3_gemv`

<!-- kernel-facts: turboexl3_gemv -->

| fact | from the code |
|---|---|
| made at | `src/core/unfused.zig:69` (`const Steps`) |
| inputs | `xp`, `slots`, `w` |
| outputs | `z` |
| source | `text_gemv` = `unfused/gemv` (3087 B, `eeb649b8f300`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core, not served: `unfused.Steps.moe` (the gate, up and down GEMVs; nothing in the library or its
  benchmark runs it). The same text is `turboexl3_dense_gemv`.
- **Shapes and dtypes:** xp f16 [RK, IN] in lane order; slots [RK] (each row-slot's expert); w the bank's trellises [E,
  IN/16, OUT/16, NHW]. Out: z f32 [RK, OUT] (unrounded inner sums).
- **Grid and threadgroup:** `((OUT/128) · 128, RK, 1)` / `(128, 1, 1)`: one threadgroup per (128-wide output block,
  row-slot).
- **Template args:** NHW, CODEBOOK, WBITS, IN, OUT. Every rate, codebook and window.
- **Replaces:** nothing: a step of the unfused chain (form 1's math).
- **Bits:** its own summation order, f32 sums unrounded. The chain it is a step of gives form 1's words (`turboexl3_gu`,
  `turboexl3_down`) and is held within 5e-3 of `ref.moeRowDec`.
- **Pinned by:** `src/core/chains_test.zig` "turboexl3 gpu unfused: the eleven-launch chain gives form 1's words and
  equals the host reference within the decode chain's tolerance, at form 1's cases (TURBOEXL3_GPU=1)" (GPU: the whole
  chain word for word against form 1 (`decode.Kernels`) and within 5e-3 of `ref.moeRowDec`, at 512 x 256 and 2560 x 640,
  MUL1 and MCG, rows 1 to 8); `src/core/unfused.zig` "turboexl3 unfused texts: the inputs each signature names are read,
  the outputs written" (CPU, the texts).

### `turboexl3_prep`

<!-- kernel-facts: turboexl3_prep -->

| fact | from the code |
|---|---|
| made at | `src/core/unfused.zig:68` (`const Steps`) |
| inputs | `x`, `slots`, `suh` |
| outputs | `xp` |
| source | `text_prep` = `unfused/prep` (1321 B, `201694d1debd`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core, not served: `unfused.Steps.moe` (the gate's, up's and down's prepares; nothing in the library or
  its benchmark runs it).
- **Shapes and dtypes:** x [rows, DIM] (the activation, read through half(); for the down's prepare h f32 [RK, inter]
  with TOPK 1); slots [RK] (each row-slot's expert); suh f16 [E, DIM]. Out: xp f16 [RK, DIM] in lane order.
- **Grid and threadgroup:** `((DIM/128) · 32, RK, 1)` / `(32, 1, 1)`, RK = rows · TOPK.
- **Template args:** DIM (hidden or inter), TOPK (the bank's top-k; 1 for the down's prepare, where row-slot i is its
  own row).
- **Replaces:** nothing: a step of the unfused chain (form 1's math).
- **Bits:** `format.project`'s prepare rounding points (f16 x, suh, H128, f16). The chain it is a step of gives form 1's
  words (`turboexl3_gu`, `turboexl3_down`) and is held within 5e-3 of `ref.moeRowDec`.
- **Pinned by:** `src/core/chains_test.zig` "turboexl3 gpu unfused: the eleven-launch chain gives form 1's words and
  equals the host reference within the decode chain's tolerance, at form 1's cases (TURBOEXL3_GPU=1)" (GPU: the whole
  chain word for word against form 1 (`decode.Kernels`) and within 5e-3 of `ref.moeRowDec`, at 512 x 256 and 2560 x 640,
  MUL1 and MCG, rows 1 to 8); `src/core/unfused.zig` "turboexl3 unfused texts: the inputs each signature names are read,
  the outputs written" (CPU, the texts).

### `turboexl3_reduce`

<!-- kernel-facts: turboexl3_reduce -->

| fact | from the code |
|---|---|
| made at | `src/core/unfused.zig:72` (`const Steps`) |
| inputs | `yk`, `scores` |
| outputs | `y` |
| source | `text_reduce` = `unfused/reduce` (325 B, `399457866bf8`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core, not served: `unfused.Steps.moe`'s last launch (nothing in the library or its benchmark runs it).
- **Shapes and dtypes:** yk f32 [RK, DIM] (the down finishes); scores [RK] (each row-slot's weight; its dtype is the
  caller's and is not stated in the code, read into an f32 fma). Out: y f32 [rows, DIM].
- **Grid and threadgroup:** `(DIM, rows, 1)` / `(256, 1, 1)`.
- **Template args:** DIM (hidden), TOPK.
- **Replaces:** nothing: the last step of the unfused chain.
- **Bits:** its own: each row's slots summed in slot order with f32 fma. The chain it is a step of gives form 1's words
  (`turboexl3_gu`, `turboexl3_down`) and is held within 5e-3 of `ref.moeRowDec`.
- **Pinned by:** `src/core/chains_test.zig` "turboexl3 gpu unfused: the eleven-launch chain gives form 1's words and
  equals the host reference within the decode chain's tolerance, at form 1's cases (TURBOEXL3_GPU=1)" (GPU: the whole
  chain word for word against form 1 (`decode.Kernels`) and within 5e-3 of `ref.moeRowDec`, at 512 x 256 and 2560 x 640,
  MUL1 and MCG, rows 1 to 8); `src/core/unfused.zig` "turboexl3 unfused texts: the inputs each signature names are read,
  the outputs written" (CPU, the texts).

### `turboexl3_swiglu`

<!-- kernel-facts: turboexl3_swiglu -->

| fact | from the code |
|---|---|
| made at | `src/core/unfused.zig:71` (`const Steps`) |
| inputs | `g`, `u` |
| outputs | `h` |
| source | `text_swiglu` = `unfused/swiglu` (124 B, `e8eb921fb8b6`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** core, not served: `unfused.Steps.moe` (nothing in the library or its benchmark runs it).
- **Shapes and dtypes:** g, u f32 [RK, inter] (the gate's and up's finishes). Out: h f32 [RK, inter].
- **Grid and threadgroup:** `(RK · inter, 1, 1)` / `(256, 1, 1)`.
- **Template args:** none.
- **Replaces:** nothing: a step of the unfused chain.
- **Bits:** silu(g) · u in f32, the sigmoid through `metal::exp` (its precision follows the compiler's math mode). The
  chain it is a step of gives form 1's words (`turboexl3_gu`, `turboexl3_down`) and is held within 5e-3 of
  `ref.moeRowDec`.
- **Pinned by:** `src/core/chains_test.zig` "turboexl3 gpu unfused: the eleven-launch chain gives form 1's words and
  equals the host reference within the decode chain's tolerance, at form 1's cases (TURBOEXL3_GPU=1)" (GPU: the whole
  chain word for word against form 1 (`decode.Kernels`) and within 5e-3 of `ref.moeRowDec`, at 512 x 256 and 2560 x 640,
  MUL1 and MCG, rows 1 to 8).

## `src/core/gpu_test.zig`

Kernels the GPU tests make as references (test only).

### `turboexl3_test_host_swiglu`

<!-- kernel-facts: turboexl3_test_host_swiglu -->

| fact | from the code |
|---|---|
| made at | `src/core/gpu_test.zig:1341` (`test turboexl3 gpu: the shared expert fused into form 9's launches equals the host's chain byte for byte (TURBOEXL3_GPU=1)`) |
| inputs | `gate`, `up`, `sigtab`, `N_size` |
| outputs | `y` |
| source | `host_swiglu_source` = `test/host_swiglu` (230 B, `f8e34292bc18`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->

- **Used by:** test only: the reference chain of `src/core/gpu_test.zig` "turboexl3 gpu: the shared expert fused into
  form 9's launches equals the host's chain byte for byte (TURBOEXL3_GPU=1)", where it runs the host's SwiGLU between
  the shared gate and up linears (the three-launch chain) and the down linear.
- **Shapes and dtypes:** gate, up T [rows, inter] (bf16 [1 to 4, 640] in the test); sigtab T [65536] (MLX's sigmoid of
  every bf16 pattern, as the host builds it); N_size a 0-dim int (rows · inter). Out: y T [rows, inter].
- **Grid and threadgroup:** `(ceil(cnt/tgs) · tgs, 1, 1)` / `(tgs, 1, 1)`, cnt = rows · inter and tgs = min(256, max(32,
  ceil(cnt/32) · 32)).
- **Template args:** T (bf16).
- **Replaces:** nothing: a reference, the host's SwiGLU kernel (`transformer.zig` SWIGLU_SOURCE, verbatim per the test's
  comment).
- **Bits:** the host's rounding: the table sigmoid in T and MLX's Multiply twice in T; the fused shared expert must
  equal the chain built on it byte for byte.
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: the shared expert fused into form 9's launches equals the
  host's chain byte for byte (TURBOEXL3_GPU=1)" (GPU, as the reference: the shared expert inside form 9's launches, in
  its own launches and through the v2 heads, rows 1 to 4). No test pins its text.

### `turboexl3_test_router`

<!-- kernel-facts: turboexl3_test_router -->

| fact | from the code |
|---|---|
| made at | `src/core/gpu_test.zig:1802` (`test turboexl3 gpu: the router folded into form 9's prepare launch (and the shared gate's logit with it) equals the router kernel + prep2 and the host's chain byte for byte (TURBOEXL3_GPU=1)`) |
| inputs | `logits` |
| outputs | `inds`, `scores` |
| source | `shared_heads.text_router_softmax` = `qwen38:shared/router_softmax_1` ++ `qwen38:shared/router_softmax_2` ++ `qwen38:shared/router_softmax_3` ++ `qwen38:shared/router_softmax_4` ++ `qwen38:shared/router_softmax_5` ++ `qwen38:shared/router_softmax_6` (2853 B, `c86a37992534`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->

- **Used by:** test only: the chain's router in `src/core/gpu_test.zig` "turboexl3 gpu: the router folded into form 9's
  prepare launch (and the shared gate's logit with it) equals ...": the host's softmax router at the host's config,
  which the router folded into form 9's prepare launch must match.
- **Shapes and dtypes:** logits TOUT [rows, NE] (bf16 or f16; NE 512 in the test). Out: inds u32 [rows, NK]; scores TOUT
  [rows, NK].
- **Grid and threadgroup:** `(TG, rows, 1)` / `(TG, 1, 1)`, TG = max(32, min(256, ceil(NE/32) · 32)) (256 at 512
  experts).
- **Template args:** TOUT, NE (experts), NK (top-k, 10 in the test), TG, NORM (1).
- **Replaces:** nothing: a reference, the host's fused softmax router (`transformer.zig` `moeRouterSource(.softmax)`,
  the MLX kernel "mlxserve_moe_router_softmax").
- **Bits:** the host's text byte for byte (`shared.text_router_softmax` in `src/models/qwen38/shared.zig`; the host's
  own test pins this copy to its source).
- **Pinned by:** `src/core/gpu_test.zig` "turboexl3 gpu: the router folded into form 9's prepare launch (and the shared
  gate's logit with it) equals ..." (GPU, as the reference: rows 1 to 4, 8 and 16, bf16 and f16 logits, tie-heavy logits
  every other call); `src/core/decode.zig` "turboexl3 decode router-folded prepare texts: the merged launch carries the
  router's and prep2's statements ..." (CPU: the merged launch carries this text's pieces verbatim, its names used).
