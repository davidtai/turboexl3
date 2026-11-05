# DeepSeek-V4.1 kernels

The kernels made in `src/models/deepseek_v41/`: the grouped linear `wo_a` (`grouped.zig`) and the kernels its tests make.
DeepSeek-V4.1's other kernels (the clamped routed chains, the banked decode, F3) live in the core engines that launch
them; their entries are in [the core catalog](../../core/kernels.md), each naming DeepSeek-V4.1. How an entry reads:
[the core catalog's introduction](../../core/kernels.md#how-an-entry-reads).

## The grouped linear `wo_a`

`root.linearGrouped` runs x `[rows, G, in]` through G stacked slices (DeepSeek-V4.1's `wo_a`: 8 slices of 4096 -> 1024),
each slice a linear with `linear`'s values. A decode call (rows <= 16) is three launches whatever G: the prepare of
every slice, one GEMV over all slices (dense's multi-row GEMV, or the routed down GEMV), and the finish. Prefill rows run
`linear` on each slice.

## `src/models/deepseek_v41/grouped.zig`

The grouped linear `wo_a` at decode rows.

### `turboexl3_grouped_finish`

<!-- kernel-facts: turboexl3_grouped_finish -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/grouped.zig:130` (`const Grouped`) |
| inputs | `zd`, `slots`, `scores`, `down_svh`, `hsx` |
| outputs | `y` |
| source | `decode.text_dfin3` = `decode/dfin3_1` ++ `decode/dfin3_expert` ++ `decode/dfin3_2` (1324 B, `f7c3494d91f8`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: `root.linearGrouped` at decode rows (1 to `kernels.DECODE_ROWS_MAX`, 16), the attention's
  `wo_a` (8 slices of 4096 -> 1024, K6 MUL1), through `grouped.Grouped.decodeRows` (one cached `Grouped` per shape,
  `root.groupedFor`; `root.groupedCached` is the tests' seam). From 17 rows `linearGrouped` runs `linear` of each slice
  instead. Step 3, the last launch.
- **Shapes and dtypes:** zd f32 [KS, rows·G, out] (step 2's planes); slots u32 [max(rows·G, 8)] (row-slot r·G + g holds
  g, which picks slice g's svh); scores f32 [max(rows·G, 8)] (all 1); down_svh the stacked svh f16 [G, out]; hsx f32
  [rows, G·in/128] (`turboexl3_grouped_prep`'s xs). Out: y OUT_T [rows·G, out], T x's dtype, reshaped to [..., G, out].
- **Grid and threadgroup:** `((out/128)·128, rows·G, 1)` / `(128, 1, 1)`.
- **Template args:** HID (out), TOPK (1: nothing summed over slots), RKC (rows·G), KS (step 2's), OUT_T (x's dtype),
  DEFER (1 for MUL1: the deferred map), NBI (in/128). Every rate, codebook and window.
- **Replaces:** with `turboexl3_grouped_prep` and step 2, `linear` of each slice (see `turboexl3_grouped_prep`); this
  launch is the G slices' finishes in one. `decode.text_dfin7` would not do: its store covers 32 · TOPK outputs of each
  128-block, all of them only from TOPK 4.
- **Bits:** each slice holds `linear`'s values on it (DEC-ONE's or the GEMV chain's, one set of bytes) but for the sign
  of an exact zero, since the finish adds a +0 score product; the call is within 5e-3 (f32) and 2.5e-2 (bf16) of the
  host reference's rms (`format.project`'s values, `ref.compare`).
- **Pinned by:** `src/models/deepseek_v41/grouped.zig` "turboexl3 grouped: the reused texts read every operand the
  launches bind and every ..." (CPU: `decode.text_dfin3` reads every operand and constant the launch binds);
  `src/models/deepseek_v41/deepseek_test.zig` "turboexl3 gpu deepseek: root.linearGrouped at wo_a (8 slices of 4096
  -> 1024) equals ..." (GPU: `wo_a` at rows 1, 4 and 16 against the host reference, f32 by 5e-3, bf16 at 4 rows by 2.5e-2,
  and every slice equal to `linear` on it, value for value); `src/models/deepseek_v41/deepseek_gemv_test.zig` "turboexl3
  gpu deepseek gemv: the grouped linear's dense GEMV gives the routed GEMV's bytes, ..." (GPU: the whole call on either
  GEMV, byte for byte).

### `turboexl3_grouped_gemv`

<!-- kernel-facts: turboexl3_grouped_gemv -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/grouped.zig:128` (`const Grouped`) |
| inputs | `hdp`, `slots`, `down_t` |
| outputs | `zd` |
| source | `decode.text_down3` = `decode/down3_1` ++ `decode/rk_line` ++ `decode/down3_2` ++ `decode/down3_w` ++ `decode/down3_3` (2340 B, `873aac06a4a9`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: step 2 of `root.linearGrouped`'s decode rows only with `Grouped.dense_gemv` false (a
  field, true by default): the tests' reference arm. Not served by default.
- **Shapes and dtypes:** hdp f16 [rows, G·in] (`turboexl3_grouped_prep`'s xp, read as rows·G row-slots of in values);
  slots u32 [max(rows·G, 8)] (row-slot r·G + g holds g; MLX binds a smaller input in the constant address space); down_t
  the stacked trellis U16 or I16 [G, in/16, out/16, n] (slot g is slice g). Out: zd f32 [KS, rows·G, out] (the split-K
  planes).
- **Grid and threadgroup:** `((out/(16·BT))·128, rows·G, KS/SPT)` / `(128, 1, 1)`, from `grouped.planFor`: BT 2, SPT 4
  where KS % 4 = 0, else 2 where KS is even, else 1.
- **Template args:** NHW, CODEBOOK, WBITS, HID (out), INTER (in), RKC (rows·G), KS (`dense.Dense.ksFor(in, out)`,
  dense's split for one slice: 16 at `wo_a`), SPT, BT, GROUP (2 from two rows: row-slots r·G + g and (r + 1)·G + g share
  a decode of slice g; 1 at one row), DEFER (1 for MUL1). Every rate, codebook and window.
- **Replaces:** nothing: it is the base of the grouped GEMV step (`decode.text_down3`, the routed down GEMV's text, with
  its own template arguments).
- **Bits:** the K partition is dense's for one slice and the statements run in the dense GEMV's order (per-lane tile
  order, quad fold, simdgroup sum), so each slice's planes, through the finish, give `linear`'s values on it (but for
  the sign of an exact zero).
- **Pinned by:** `src/models/deepseek_v41/grouped.zig` "turboexl3 grouped: the reused texts read every operand the
  launches bind and every ..." (CPU: `decode.text_down3` reads every operand and constant the launch binds);
  `src/models/deepseek_v41/grouped.zig` "turboexl3 grouped: DeepSeek-V4.1's wo_a (8 slices of 4096 -> 1024) is admitted,
  its K split ..." (CPU: `wo_a`'s plan: K split 16, whole parts, SPT and BT dividing);
  `src/models/deepseek_v41/deepseek_gemv_test.zig` "turboexl3 gpu deepseek gemv: the grouped linear's dense GEMV gives
  the routed GEMV's bytes, ..." (GPU: the reference arm the dense GEMV is held to).

### `turboexl3_grouped_gemv_d`

<!-- kernel-facts: turboexl3_grouped_gemv_d -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/grouped.zig:132` (`const Grouped`) |
| inputs | `xp`, `w` |
| outputs | `z` |
| source | `text_gemv_d` = `deepseek_v41:grouped/gemv_head` ++ `dense/dgemv_x` ++ `dense/dgemv_2` ++ `dense/dgemv_m` ++ `dense/dgemv_fold` ++ `deepseek_v41:grouped/gemv_store` (3853 B, `0771aed2b06a`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: `root.linearGrouped` at decode rows (1 to `kernels.DECODE_ROWS_MAX`, 16), the attention's
  `wo_a` (8 slices of 4096 -> 1024, K6 MUL1), through `grouped.Grouped.decodeRows` (one cached `Grouped` per shape,
  `root.groupedFor`; `root.groupedCached` is the tests' seam). From 17 rows `linearGrouped` runs `linear` of each slice
  instead. Step 2 with `Grouped.dense_gemv` true (the default), for chunks of up to 5 rows (`dense.Dense.max_rb`) that
  do not take the lean text.
- **Shapes and dtypes:** xp f16 [rows, G·in] (`turboexl3_grouped_prep`'s xp, lane order); w the stacked trellis U16 or
  I16 [G, in/16, out/16, n]. Out: z f32 [KS, rows·G, out], row-slot r·G + g's K-slice partials at z[ks, r·G + g] (the
  routed down GEMV's plane layout, which `turboexl3_grouped_finish` reads).
- **Grid and threadgroup:** `((out/(16·BT))·128, chunks·G, KS)` / `(128, 1, 1)`: the threadgroup's slice is y mod G, its
  row chunk y / G; chunks = ceil(rows/RB).
- **Template args:** NHW, CODEBOOK, WBITS, IN (a slice's inputs; xp's row stride is G·IN), OUT, KS
  (`dense.Dense.ksFor(in, out)`, 16 at `wo_a`), BT (`bt_set[rows]` if set, else `dense.Dense.btFor(RB)`: 4 at 1 row, 2
  at 2 to 4, 1 beyond; 8 % BT = 0), RB (rows per chunk, `dense.Dense.rbCap(rows, lean_rb 8)`: one chunk up to 8 rows,
  two from 9), ROWS (rows), GROUPS (G), DEFER (1 for MUL1). Taken when RB < 5, or RB = 5 where `dense.Dense.wideLean`
  does not take the launch (at `wo_a`, whose launch has 8192 · chunks threadgroups at BT 1, with `Config.dense_lean5`
  on: rows 1 to 4). Every rate, codebook and window.
- **Replaces:** `turboexl3_grouped_gemv` in step 2 (one launch for one): dense's multi-row GEMV over the slices, so a
  slice's weights are decoded once per row chunk instead of once per row pair.
- **Bits:** byte-identical to `turboexl3_grouped_gemv`: the statements from the input loads to the fold are
  `dense.text_dgemv`'s verbatim and the store sums the simdgroups as it does, on the same K partition, so the same
  planes.
- **Pinned by:** `src/models/deepseek_v41/deepseek_gemv_test.zig` "turboexl3 deepseek gemv: the grouped dense-GEMV texts
  are dense's GEMV statements ..." (CPU: the text is dense's GEMV pieces in dense's order, reads every operand and
  constant, no shape buffer); `src/models/deepseek_v41/deepseek_gemv_test.zig` "turboexl3 deepseek gemv: lean chunks,
  and with dense_lean5 chunks of 5 rows, take BT 2 ..." (CPU: the BT rule `denseGemvCfg` applies at `wo_a`, rows 1 to
  16); `src/models/deepseek_v41/deepseek_gemv_test.zig` "turboexl3 gpu deepseek gemv: the grouped linear's dense GEMV
  gives the routed GEMV's bytes, ..." (GPU: the grouped call on it against the routed GEMV, byte for byte, rows 1 to 16,
  x bf16 and f32, BT served and 1, 2, 4, 8, at `wo_a` (8 x 4096 -> 1024, K6) and 4 x 1024 -> 512 at K4);
  `src/models/deepseek_v41/deepseek_test.zig` "turboexl3 gpu deepseek: root.linearGrouped at wo_a (8 slices of 4096
  -> 1024) equals ..." (GPU: the call against the host reference and `linear`).

### `turboexl3_grouped_gemv_dl`

<!-- kernel-facts: turboexl3_grouped_gemv_dl -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/grouped.zig:134` (`const Grouped`) |
| inputs | `xp`, `w` |
| outputs | `z` |
| source | `text_gemv_dl` = `deepseek_v41:grouped/gemv_head` ++ `dense/dgemv_2` ++ `dense/dgemv_j` ++ `dense/dgemv_fold` ++ `deepseek_v41:grouped/gemv_store` (3918 B, `859700a0cadf`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: `root.linearGrouped` at decode rows (1 to `kernels.DECODE_ROWS_MAX`, 16), the attention's
  `wo_a` (8 slices of 4096 -> 1024, K6 MUL1), through `grouped.Grouped.decodeRows` (one cached `Grouped` per shape,
  `root.groupedFor`; `root.groupedCached` is the tests' seam). From 17 rows `linearGrouped` runs `linear` of each slice
  instead. Step 2 with `Grouped.dense_gemv` true (the default), for chunks of more than 5 rows, and of 5 where
  `dense.Dense.wideLean` takes the launch.
- **Shapes and dtypes:** xp f16 [rows, G·in] (`turboexl3_grouped_prep`'s xp, lane order); w the stacked trellis U16 or
  I16 [G, in/16, out/16, n]. Out: z f32 [KS, rows·G, out], row-slot r·G + g's K-slice partials at z[ks, r·G + g] (the
  routed down GEMV's plane layout, which `turboexl3_grouped_finish` reads).
- **Grid and threadgroup:** `((out/(16·BT))·128, chunks·G, KS)` / `(128, 1, 1)`: the threadgroup's slice is y mod G, its
  row chunk y / G; chunks = ceil(rows/RB).
- **Template args:** NHW, CODEBOOK, WBITS, IN (a slice's inputs; xp's row stride is G·IN), OUT, KS
  (`dense.Dense.ksFor(in, out)`, 16 at `wo_a`), BT (`bt_set[rows]` if set, else `dense.Dense.btLaunch(RB, tgs1)`: 2
  where `wideLean` holds, else `btFor(RB)`, 1; tgs1 = (out/16)·chunks·G·KS), RB (rows per chunk,
  `dense.Dense.rbCap(rows, lean_rb 8)`: one chunk up to 8 rows, two from 9), ROWS (rows), GROUPS (G), DEFER (1 for
  MUL1). Taken when RB > 5, or RB = 5 where `wideLean` holds (`Config.dense_lean5`, a launch of at least 2048
  threadgroups at BT 1); at `wo_a` (8192 · chunks): rows 5 to 16, BT 2. Every rate, codebook and window.
- **Replaces:** `turboexl3_grouped_gemv_d` for those chunks (one launch for one), as dense's lean text (`dgemv_j`, the
  codeword loop by input element) stands in for `text_dgemv`; in place of the routed `turboexl3_grouped_gemv`.
- **Bits:** byte-identical to `turboexl3_grouped_gemv`: from the input loads to the fold it is the lean text's
  statements verbatim, with the same per-lane FMA order, quad fold and simdgroup sum on the same K partition.
- **Pinned by:** `src/models/deepseek_v41/deepseek_gemv_test.zig` "turboexl3 deepseek gemv: the grouped dense-GEMV texts
  are dense's GEMV statements ..." (CPU: the text is dense's lean pieces in dense's order, reads every operand and
  constant); `src/models/deepseek_v41/deepseek_gemv_test.zig` "turboexl3 deepseek gemv: lean chunks, and with
  dense_lean5 chunks of 5 rows, take BT 2 ..." (CPU: the lean chunks' BT 2 rule at `wo_a`'s 8192 threadgroups);
  `src/models/deepseek_v41/deepseek_gemv_test.zig` "turboexl3 gpu deepseek gemv: the grouped linear's dense GEMV gives
  the routed GEMV's bytes, ..." (GPU: rows 5 to 16 run it, byte for byte against the routed GEMV, x bf16 and f32, at the
  served BT and 1, 2, 4, 8 where a lane holds at most 64 accumulators); `src/models/deepseek_v41/deepseek_test.zig`
  "turboexl3 gpu deepseek: root.linearGrouped at wo_a (8 slices of 4096 -> 1024) equals ..." (GPU).

### `turboexl3_grouped_prep`

<!-- kernel-facts: turboexl3_grouped_prep -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/grouped.zig:126` (`const Grouped`) |
| inputs | `x`, `suh` |
| outputs | `xp`, `xs` |
| source | `dense.text_prep_s` = `dense/prep_s` (1236 B, `4b5d658122db`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: `root.linearGrouped` at decode rows (1 to `kernels.DECODE_ROWS_MAX`, 16), the attention's
  `wo_a` (8 slices of 4096 -> 1024, K6 MUL1), through `grouped.Grouped.decodeRows` (one cached `Grouped` per shape,
  `root.groupedFor`; `root.groupedCached` is the tests' seam). From 17 rows `linearGrouped` runs `linear` of each slice
  instead. Step 1 of the three launches, whatever G.
- **Shapes and dtypes:** x T [rows, G·in] (T bf16, f16 or f32: `linearGrouped`'s x reshaped so a row's G slices are one
  row); suh f16 [G, in], read flat as [G·in]. Out: xp f16 [rows, G·in] (the rotated inputs, lane order), xs f32 [rows,
  G·in/128] (each 128-block's sum, so row-slot r·G + g's sums sit at [r·G + g, in/128]).
- **Grid and threadgroup:** `((G·in/128)·32, rows, 1)` / `(32, 1, 1)`.
- **Template args:** DIM (G·in), LANE_ORDER (1). Every shape `grouped.Shape.admitted` takes (G 1 to 64, in and out
  multiples of 128, whole 4-tile parts), every rate, codebook and window.
- **Replaces:** with `turboexl3_grouped_gemv_d` (or `_dl`, or `_gemv`) and `turboexl3_grouped_finish`, `linear` of each
  slice (the route `linearGrouped` takes from 17 rows: G column copies, G `linear` calls, one MLX concatenate); this
  launch is the G slices' prepares (`turboexl3_dense_prep_s`'s text) in one. Three launches, three dependent levels.
- **Bits:** its own rounding, dense's prepare's: every 128-block lies inside one slice, so each slice's prepared values
  and block sums are `turboexl3_dense_prep_s`'s on that slice. The call's bits are under `turboexl3_grouped_finish`.
- **Pinned by:** `src/models/deepseek_v41/grouped.zig` "turboexl3 grouped: the reused texts read every operand the
  launches bind and every ..." (CPU: `dense.text_prep_s` reads every operand and constant the launch binds);
  `src/models/deepseek_v41/deepseek_test.zig` "turboexl3 deepseek: every DeepSeek-V4.1 trunk and vision shape is a width
  the dense kernels admit, ..." (CPU: `wo_a` is an admitted grouped shape); `src/models/deepseek_v41/deepseek_test.zig`
  "turboexl3 gpu deepseek: root.linearGrouped at wo_a (8 slices of 4096 -> 1024) equals ..." (GPU: the whole call, see
  `turboexl3_grouped_finish`); `src/models/deepseek_v41/deepseek_gemv_test.zig` "turboexl3 gpu deepseek gemv: the
  grouped linear's dense GEMV gives the routed GEMV's bytes, ..." (GPU: the whole call on either GEMV, byte for byte).

## `src/models/deepseek_v41/f3_test.zig`

Kernels F3's tests make as references (test only).

### `turboexl3_test_swiglu_one`

<!-- kernel-facts: turboexl3_test_swiglu_one -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/f3_test.zig:236` (`test turboexl3 gpu f3u time: the 40-layer trunk chain with the host's shared expert against F3 at rows 5, 6 and 8, every run the first's words (TURBOEXL3_F3_TIME=1)`) |
| inputs | `gt`, `ut`, `lim` |
| outputs | `y` |
| source | `text_swiglu_one` = text (551 B) (551 B, `9be0c03bd41d`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->

- **Used by:** test only: arm A of `src/models/deepseek_v41/f3_test.zig` "turboexl3 gpu f3u time: the 40-layer trunk
  chain with the host's shared expert against F3 at rows 5, ...": the host's clamped SwiGLU as MLX's compiled decode
  stretch runs it, one elementwise launch between the shared gate and up linears and the down linear. Its text is a
  string literal in `f3_test.zig`.
- **Shapes and dtypes:** gt, ut bf16 [rows, 2304] (the gate's and up's outputs; rows 5, 6 and 8); lim f32 [1] (10). Out:
  y bf16 [rows, 2304].
- **Grid and threadgroup:** `(ceil(NEL/256) · 256, 1, 1)` / `(256, 1, 1)`, NEL = rows · 2304.
- **Template args:** NEL.
- **Replaces:** nothing: a stand-in for the host's MLX ops.
- **Bits:** the clamped SwiGLU functors in f32 in `clampedSwiGLU`'s order (MLX's minimum and maximum, the precise-exp
  sigmoid, multiply), bf16 out; F3's runs must give this chain's words.
- **Pinned by:** `src/models/deepseek_v41/f3_test.zig` "turboexl3 gpu f3u time: the 40-layer trunk chain with the host's
  shared expert against F3 at rows 5, ..." (GPU: every run's 40-layer output, both arms, the first run's words at rows
  5, 6 and 8).

## `src/models/deepseek_v41/banked_ilp_test.zig`

The probes of the banked GEMVs' t routes (test only).

### `turboexl3_t1v_probe_<route>_<dtype>`

<!-- kernel-facts: turboexl3_t1v_probe_<route>_<dtype> -->

| fact | from the code |
|---|---|
| names | `turboexl3_t1v_probe_0_i16`, `turboexl3_t1v_probe_0_u16`, `turboexl3_t1v_probe_1_i16`, `turboexl3_t1v_probe_1_u16`, `turboexl3_t1v_probe_2_i16`, `turboexl3_t1v_probe_2_u16`, `turboexl3_t1v_probe_3_i16`, `turboexl3_t1v_probe_3_u16`, `turboexl3_t1v_probe_4_i16`, `turboexl3_t1v_probe_4_u16` |
| made at | `src/models/deepseek_v41/banked_ilp_test.zig:370` (`test turboexl3 gpu banked ilp t routes: every EXL3_T1V route gives today's route's t, word for word, and the byte sum's, for all 65536 codeword states read from I16 and from U16 trellis words (TURBOEXL3_GPU=1)`) |
| inputs | `code` |
| outputs | `t` |
| source | `t_probe_body.ptr` = text (238 B) (238 B, `ffa699a5daea`) |
| header | `t_probe_headers[ri].ptr`: a variant's defines ++ the base header, one per name |
| header of `turboexl3_t1v_probe_0_i16` | text '#define EXL3_T1V 0\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18019 B, `572b084cfa14`) |
| header of `turboexl3_t1v_probe_0_u16` | text '#define EXL3_T1V 0\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18019 B, `572b084cfa14`) |
| header of `turboexl3_t1v_probe_1_i16` | text '#define EXL3_T1V 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18019 B, `4fc9bff5fd1d`) |
| header of `turboexl3_t1v_probe_1_u16` | text '#define EXL3_T1V 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18019 B, `4fc9bff5fd1d`) |
| header of `turboexl3_t1v_probe_2_i16` | text '#define EXL3_T1V 2\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18019 B, `4a04e9b1d385`) |
| header of `turboexl3_t1v_probe_2_u16` | text '#define EXL3_T1V 2\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18019 B, `4a04e9b1d385`) |
| header of `turboexl3_t1v_probe_3_i16` | text '#define EXL3_T1V 3\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18019 B, `8aba604a8cee`) |
| header of `turboexl3_t1v_probe_3_u16` | text '#define EXL3_T1V 3\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18019 B, `8aba604a8cee`) |
| header of `turboexl3_t1v_probe_4_i16` | text '#define EXL3_T1V 4\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18019 B, `f4cb25c4b89d`) |
| header of `turboexl3_t1v_probe_4_u16` | text '#define EXL3_T1V 4\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18019 B, `f4cb25c4b89d`) |

<!-- /kernel-facts -->

- **Used by:** Test only: `src/models/deepseek_v41/banked_ilp_test.zig`'s t-routes test makes one per route (EXL3_T1V 0
  to 4) and code dtype (i16, u16) with `ks.newKernel` and frees it after its launch. Nothing in the library launches it.
- **Shapes and dtypes:** code I16 or U16 [65536] (every codeword state 0 to 65535 once, in order; read as the GEMVs read
  a trellis: its words as uint, halfword i & 1 of word i >> 1). Out: t f32 [65536], `exl3_mul1_t` of state i.
- **Grid and threadgroup:** `(65536, 1, 1)` / `(256, 1, 1)`.
- **Template args:** None (`.template` empty). The route is the header: `#define EXL3_T1V <route>` before
  `decode.header`, so route 3 is today's default; the dtype suffix only names the code array's dtype (the same text and
  header for both). The routes are `exl3_mul1_t`'s ways from the two pair sums to t: 0 a 32-bit multiply by 0x10001 and
  an integer convert, 1 a shift and add, 2 the multiply then the half 1024 + t, 3 shift and add then the half route, 4
  both pair sums as one half2.
- **Replaces:** nothing: a probe of the MUL1 helper alone, the code the `t4`, `t1` and `t0` variants of the grouped
  banked GEMVs change.
- **Bits:** every route's t equals route 3's and the byte sum of cw · 0x83DCD12D (mod 2^32), word for word (differing
  0), for all 65536 states read from I16 and from U16 words.
- **Pinned by:** `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu banked ilp t routes: every EXL3_T1V route
  gives today's route's t, ..." (GPU: each route against route 3 and the host byte sum).
