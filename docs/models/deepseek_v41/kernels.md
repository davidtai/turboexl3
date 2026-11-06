# DeepSeek-V4.1 kernels

The kernels made in `src/models/deepseek_v41/`: the grouped linear `wo_a` (`grouped.zig`), the banked decode over an
expert streamer's slot banks and its grouped GEMVs (`banked.zig`), F3 (`f3.zig`) and the kernels its tests make. The
clamped routed chains it shares with DeepSeek-V4 and GLM live in the core engines that launch them; their entries are in
[the core catalog](../../core/kernels.md), each naming its users. How an entry reads:
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

## `src/models/deepseek_v41/banked.zig`

The banked decode (`root.moeClampedBanked` and its halves): form 7's launches with the clamped fin7 on form 9's
launch shape, each row-slot one routed pair read from an expert streamer's slot bank, and the grouped GEMVs that share a
decode among the row-slots of one id (`Config.banked_group`). `Banked` keeps them beside a geometry's `decode.Kernels3`.

### `turboexl3_dfin3b`

<!-- kernel-facts: turboexl3_dfin3b -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:264` (`const Banked`) |
| inputs | `zd`, `slots`, `scores`, `down_svh0`, `down_svh1`, `down_svh2`, `hsx` |
| outputs | `y` |
| source | `text_dfin3b` = `decode/dfin3_1` ++ `deepseek_v41:banked/dfin3b_expert` ++ `decode/dfin3_2` (1610 B, `7274aee7befe`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 5, the last, of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts:
  each row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots), in its down half. Served by default: `Config.banked_group` 2 runs the banked texts.
- **Shapes and dtypes:** zd f32 [KS_dn, rows, hidden] (`turboexl3_down3b`'s planes); slots the ids u32 or i32 [rows],
  each packed (bank << 24 | slot row); scores f32 [64] (`bankedOnes`: every score 1); down_svh0/1/2 f16 [E_b, hidden];
  hsx f32 [rows, inter/128] (`turboexl3_fin7cb`'s hs). Out: y f32 [rows, hidden], each pair's expert output at weight 1
  (the caller weighs and sums them).
- **Grid and threadgroup:** `((hidden/128)·128, rows, 1)` / `(128, 1, 1)`.
- **Template args:** HID (hidden), TOPK (1), RKC (rows), KS (KS_dn), OUT_T (f32: `root` builds the banked `Kernels3` at
  out_dtype f32), DEFER (1 for MUL1), NBI (inter/128). KS_dn is the down's K slices, the largest of 4, 2, 1 with
  (inter/16) % (4·KS) = 0 and at least 2 tiles per part.
- **Replaces:** `turboexl3_dfin3` (the top-1 down finish: form 9 runs `dfin3` below top-k 4), launched once per bank by
  each bank's own `moeClamped` call: one launch over the three banks.
- **Bits:** byte-identical to `moeClamped` on the pair's bank at top-1 with score 1: `dfin3b_expert` (the slot row and
  the bank's down svh) replaces only the line that reads a bank's arrays, so every other statement is `dfin3`'s. The
  streamer's sum, each full token's six pairs weighed, is held within 5e-3 of the clamped host reference's rms
  (`ref.compare` against `ref.moeRowClamped`).
- **Pinned by:** `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode texts: each is its base text with
  the bank's lines in place ..." (CPU: the text is its base text with `dfin3b_expert` in place of the base line and
  nothing else, every input read, no shape buffer); `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode
  refusals: the clamp, the ids' shape and dtype, the banks' dtypes, ..." (CPU: the call's refusals, before a launch);
  `src/models/deepseek_v41/banked_test.zig` "turboexl3 gpu banked decode: every pair of moeClampedBanked over three
  banks is moeClamped's ..." (GPU: every pair byte for byte against `moeClamped` at rows 1, 2, 6, 7, 16, 36 and 48, x
  bf16 and f32, trellis U16 and I16, at DeepSeek-V4.1's routed geometry (hidden 5120, width 2304, limit 10, MUL1) over
  synthetic banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32); sampled token sums to the reference by 5e-3).

### `turboexl3_dfin3bg`

<!-- kernel-facts: turboexl3_dfin3bg -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:299` (`const Banked`) |
| inputs | `zd`, `slots`, `scores`, `down_svh0`, `down_svh1`, `down_svh2`, `hsx` |
| outputs | `y` |
| source | `text_dfin3bg` = `decode/dfin3_1` ++ `deepseek_v41:banked/skip_rk_next` ++ `deepseek_v41:banked/dfin3b_expert` ++ `decode/dfin3_2` (1701 B, `f4c14dd1211a`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |
| made at | `src/models/deepseek_v41/banked.zig:333` (`const Banked`) (the same text) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 5 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots) on the grouped texts, g4 and g4m3 (made at both sites). Not served by default:
  `Config.banked_group` 4 selects the grouped texts, or a call through `moeClampedBankedSel*` with `BankedSel` g4 or
  g4m3 (hw4b's check mode, the tests).
- **Shapes and dtypes:** As `turboexl3_dfin3b`; `skip_id` 0xFFFFFFFF (`root.banked_skip_id`) marks a row-slot that runs
  no expert: y f32 [rows, hidden] holds 0 in its row (DEVROUTE's pairs whose expert no bank holds; the caller replaces
  those rows).
- **Grid and threadgroup:** As `turboexl3_dfin3b`: `((hidden/128)·128, rows, 1)` / `(128, 1, 1)`.
- **Template args:** HID, TOPK (1), RKC, KS (KS_dn), OUT_T (f32), DEFER, NBI (inter/128), as `turboexl3_dfin3b`.
- **Replaces:** `turboexl3_dfin3b` in the same launch: its text with `skip_rk_next` (skip the row-slot in the slot loop)
  before the bank lines.
- **Bits:** byte-identical to `turboexl3_dfin3b` for every row-slot not skipped; a skipped row-slot's y row is 0, every
  word.
- **Pinned by:** `src/models/deepseek_v41/banked_test.zig` "turboexl3 grouped banked decode texts: prep2bg, fin7cbg and
  dfin3bg are the banked texts ..." (CPU: the text is the banked text with the skip line before its bank lines);
  `src/models/deepseek_v41/banked_test.zig` "turboexl3 gpu banked group: the grouped banked call's rows are the banked
  call's ..." (GPU: the grouped call's rows equal the banked call's byte for byte, a skipped row-slot's y row 0, rows 6,
  12, 36 and 48, x bf16, trellis I16, at DeepSeek-V4.1's routed geometry (hidden 5120, width 2304, limit 10, MUL1) over
  synthetic banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32)). It runs in every arm of
  `src/models/deepseek_v41/banked_grp_test.zig` "turboexl3 gpu banked grp: the M = 3 texts' gate|up outputs and rows are
  today's, ..." and `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu banked ilp: every grouped GEMV
  variant's gate|up outputs and rows, ..." (GPU), which hold the whole call's hd, hs and y.

### `turboexl3_down3b`

<!-- kernel-facts: turboexl3_down3b -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:262` (`const Banked`) |
| inputs | `hdp`, `slots`, `down_t0`, `down_t1`, `down_t2` |
| outputs | `zd` |
| source | `text_down3b` = `decode/down3_1` ++ `decode/rk_line` ++ `decode/down3_2` ++ `deepseek_v41:banked/down3b_w` ++ `decode/down3_3` (2506 B, `67a8776af847`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 4 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots), the start of its down half (`moeClampedBankedDown`, `banked.Banked.bankedDown`).
  Served by default: `Config.banked_group` 2 runs the banked texts.
- **Shapes and dtypes:** hdp f16 [rows, inter] (`turboexl3_fin7cb`'s hd, lane order); slots the ids u32 or i32 [rows],
  each packed (bank << 24 | slot row); down_t0/1/2 U16 or I16 [E_b, inter/16, hidden/16, n]. Out: zd f32 [KS_dn, rows,
  hidden] (the split-K down planes). KS_dn is the down's K slices, the largest of 4, 2, 1 with (inter/16) % (4·KS) = 0
  and at least 2 tiles per part.
- **Grid and threadgroup:** `((hidden/(16·BT))·128, rows, z)` / `(128, 1, 1)`, z = max(1, KS_dn / kdiv[min(rows, 4) −
  1]) with `kdiv9d` (null: `kdiv9`; 1 by default, so z = KS_dn).
- **Template args:** NHW (n, the halfwords per tile: K = NHW/16), CODEBOOK (MUL1 0, MCG 1), WBITS (the window's bits),
  HID (hidden), INTER (inter), RKC (rows), KS (KS_dn), SPT (KS_dn / z: 1 by default), BT (output tiles per threadgroup:
  `bt1` at one row-slot, `bt9` from two, both 2 by default), GROUP (2 from two row-slots, so two row-slots of one packed
  id share a decode only on one bank's one slot; 1 at one), DEFER (1 for MUL1: the GEMV sums x times the codeword's byte
  sum). Every rate `bankedCall` admits (gate, up and down at one rate), both codebooks and every window.
- **Replaces:** `turboexl3_down3`, launched once per bank by each bank's own `moeClamped` call: one launch over the
  three banks.
- **Bits:** byte-identical to `moeClamped` on the pair's bank at top-1 with score 1: `down3b_w` (the weight base: the
  bank's down trellis at the slot row) replaces only the line that reads a bank's arrays, so every other statement is
  `down3`'s. The streamer's sum, each full token's six pairs weighed, is held within 5e-3 of the clamped host
  reference's rms (`ref.compare` against `ref.moeRowClamped`).
- **Pinned by:** `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode texts: each is its base text with
  the bank's lines in place ..." (CPU: the text is its base text with `down3b_w` in place of the base line and nothing
  else, every input read, no shape buffer); `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode refusals:
  the clamp, the ids' shape and dtype, the banks' dtypes, ..." (CPU: the call's refusals, before a launch);
  `src/models/deepseek_v41/banked_test.zig` "turboexl3 gpu banked decode: every pair of moeClampedBanked over three
  banks is moeClamped's ..." (GPU: every pair byte for byte against `moeClamped` at rows 1, 2, 6, 7, 16, 36 and 48, x
  bf16 and f32, trellis U16 and I16, at DeepSeek-V4.1's routed geometry (hidden 5120, width 2304, limit 10, MUL1) over
  synthetic banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32); sampled token sums to the reference by 5e-3). The texts
  test also pins that the members compare the packed id.

### `turboexl3_down3bg`

<!-- kernel-facts: turboexl3_down3bg -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:297` (`const Banked`) |
| inputs | `hdp`, `slots`, `down_t0`, `down_t1`, `down_t2` |
| outputs | `zd` |
| source | `text_down3bg` = `decode/down3_1` ++ `decode/rk_line` ++ `deepseek_v41:banked/down3g_2` ++ `deepseek_v41:banked/down3b_w` ++ `deepseek_v41:banked/down3g_3` (2585 B, `b9686e67ab4a`) |
| header | `header_group` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26373 B, `b7023ef0567c`) |
| made at | `src/models/deepseek_v41/banked.zig:413` (`const Banked`) |
| inputs | `hdp`, `slots`, `down_t0`, `down_t1`, `down_t2` |
| outputs | `zd` |
| source | `text_down3bg` = `decode/down3_1` ++ `decode/rk_line` ++ `deepseek_v41:banked/down3g_2` ++ `deepseek_v41:banked/down3b_w` ++ `deepseek_v41:banked/down3g_3` (2585 B, `b9686e67ab4a`) |
| header | `group_headers[v].ptr` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26373 B, `b7023ef0567c`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 4 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots) on today's grouped texts (g4): `Config.banked_group` 4 with `banked_m3` false, or
  `BankedSel` g4. Not served by default. Its name is also variant 0's in the variant table (`group_down_names[0]`;
  `groupGemvs` makes only variants 1 and up).
- **Shapes and dtypes:** As `turboexl3_down3b`: hdp f16 [rows, inter]; slots the packed ids u32 or i32 [rows], `skip_id`
  0xFFFFFFFF (`root.banked_skip_id`) marks a row-slot that runs no expert (no decode, no plane rows written);
  down_t0/1/2 U16 or I16 [E_b, inter/16, hidden/16, n]. Out: zd f32 [KS_dn, rows, hidden].
- **Grid and threadgroup:** As `turboexl3_down3b`: `((hidden/(16·BT))·128, rows, z)` / `(128, 1, 1)`, z = KS_dn by
  default. (`cfgsBankedG`).
- **Template args:** NHW (n, the halfwords per tile: K = NHW/16), CODEBOOK (MUL1 0, MCG 1), WBITS (the window's bits),
  HID (hidden), INTER (inter), RKC (rows), KS (KS_dn), SPT (KS_dn / z: 1 by default), BT (output tiles per threadgroup:
  `bt1` at one row-slot, `bt9` from two, both 2 by default), GROUP (4 from two row-slots, 1 at one: up to four row-slots
  of one packed id per lead threadgroup (`turboexl3_members_g`), a skipped row-slot no member), DEFER (1 for MUL1: the
  GEMV sums x times the codeword's byte sum). Every rate `bankedCall` admits (gate, up and down at one rate), both
  codebooks and every window. Inside: the member paths M 1, 2 and 4 of `turboexl3_gemv_group`, so a lead of three
  members runs the M = 4 path with a fourth member's FMAs padding.
- **Replaces:** `turboexl3_down3b` in the same launch: the members widened from two to four (`down3g_2`, `down3g_3`),
  the banked weight base (`down3b_w`) between them.
- **Bits:** byte-identical to the banked text `turboexl3_down3b` (the grouped call's rows are the banked call's): every
  weight is read and decoded once and fed to each member in the one-member order.
- **Pinned by:** `src/models/deepseek_v41/banked_test.zig` "turboexl3 grouped banked decode texts: prep2bg, fin7cbg and
  dfin3bg are the banked texts ..." (CPU: the banked text's constants and grid line, then the four-member head, the
  banked weight base, the four-member tail); `src/models/deepseek_v41/banked_test.zig` "turboexl3 gpu banked group: the
  grouped banked call's rows are the banked call's ..." (GPU: the grouped call against the banked call, byte for byte, a
  skipped row-slot's y row 0, rows 6, 12, 36 and 48, x bf16, trellis I16, at DeepSeek-V4.1's routed geometry (hidden
  5120, width 2304, limit 10, MUL1) over synthetic banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32); up to eight
  row-slots of one id: two leads); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 banked ilp: variant 0 is
  today's (no defines, the stock kernel names), ..." (CPU: variant 0 keeps the stock name and no defines). It is the
  reference arm (g4, variant 0) of `src/models/deepseek_v41/banked_grp_test.zig` "turboexl3 gpu banked grp: the M = 3
  texts' gate|up outputs and rows are today's, ..." and `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu
  banked ilp: every grouped GEMV variant's gate|up outputs and rows, ..." (GPU).

### `turboexl3_down3bg_<variant>`

<!-- kernel-facts: turboexl3_down3bg_<variant> -->

| fact | from the code |
|---|---|
| names | `turboexl3_down3bg_ku`, `turboexl3_down3bg_pf`, `turboexl3_down3bg_xh`, `turboexl3_down3bg_m2`, `turboexl3_down3bg_sh`, `turboexl3_down3bg_t4`, `turboexl3_down3bg_t1`, `turboexl3_down3bg_t0`, `turboexl3_down3bg_pfxh`, `turboexl3_down3bg_pfsh`, `turboexl3_down3bg_xhm2` |
| made at | `src/models/deepseek_v41/banked.zig:413` (`const Banked`) |
| inputs | `hdp`, `slots`, `down_t0`, `down_t1`, `down_t2` |
| outputs | `zd` |
| source | `text_down3bg` = `decode/down3_1` ++ `decode/rk_line` ++ `deepseek_v41:banked/down3g_2` ++ `deepseek_v41:banked/down3b_w` ++ `deepseek_v41:banked/down3g_3` (2585 B, `b9686e67ab4a`) |
| header | `group_headers[v].ptr`: a variant's defines ++ the base header, one per name |
| header of `turboexl3_down3bg_ku` | text '#define EXL3_KU 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26391 B, `893d0732e30e`) |
| header of `turboexl3_down3bg_pf` | text '#define EXL3_GPF 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `719802404fb5`) |
| header of `turboexl3_down3bg_xh` | text '#define EXL3_GXH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `35cacdcfb1d7`) |
| header of `turboexl3_down3bg_m2` | text '#define EXL3_GM2 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `d28773b43d18`) |
| header of `turboexl3_down3bg_sh` | text '#define EXL3_GSH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `1eaa64821218`) |
| header of `turboexl3_down3bg_t4` | text '#define EXL3_T1V 4\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `acaeb8fa61e1`) |
| header of `turboexl3_down3bg_t1` | text '#define EXL3_T1V 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `b0622147baad`) |
| header of `turboexl3_down3bg_t0` | text '#define EXL3_T1V 0\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `9be4db5164be`) |
| header of `turboexl3_down3bg_pfxh` | text '#define EXL3_GPF 1\n#define EXL3_GXH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26411 B, `c5b9183f1adc`) |
| header of `turboexl3_down3bg_pfsh` | text '#define EXL3_GPF 1\n#define EXL3_GSH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26411 B, `2478957c5a9f`) |
| header of `turboexl3_down3bg_xhm2` | text '#define EXL3_GXH 1\n#define EXL3_GM2 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26411 B, `a43b1425ae6c`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 4 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots) at `Config.banked_variant_dn` v (1 to 11, needs `Config.banked_group` 4) on today's
  grouped texts (`banked_m3` false), or `moeClampedBankedDownV` / `...SelV` with that variant; `Banked.groupGemvs`
  makes each on first use (`banked_gv`). hw2's ILP retune; not served by default (variant 0).
- **Shapes and dtypes:** As `turboexl3_down3bg`.
- **Grid and threadgroup:** As `turboexl3_down3bg` (`cfgsBankedG`).
- **Template args:** As `turboexl3_down3bg`. The variant is the header: its Metal defines before `header_group`, under a
  kernel name of its own: `ku` EXL3_KU 1 (input tiles in pairs, both tiles' words loaded before either decodes); `pf`
  EXL3_GPF 1 (a rolling prefetch: the next input tile's words loaded before this tile decodes); `xh` EXL3_GXH 1 (each
  member's x quad held as half4 and widened at its FMA); `m2` EXL3_GM2 1 (the four-member path as two passes of two
  members, the second reading and decoding the tile again); `sh` EXL3_GSH 1 (the one-64-bit-window reader with each tile
  word loaded by one lane and taken by the others through `simd_shuffle`); `t4`, `t1`, `t0` EXL3_T1V 4, 1, 0 (the route
  from a MUL1 codeword to t, default 3: 4 both pair sums as one half2, 1 a shift and add for the 32-bit multiply, 0 the
  multiply by 0x10001 and an integer convert; MUL1 only); `pfxh`, `pfsh`, `xhm2` two of those defines together.
- **Replaces:** `turboexl3_down3bg` in the same launch.
- **Bits:** byte-identical to `turboexl3_down3bg` (variant 0): each variant reads, decodes and sums every weight as
  variant 0 does (the same codewords, decoded values, FMA sequence into the same f32 accumulators, and folds); they
  differ only in when the tile words load, how many lanes load them, the registers the loop holds, the passes over a
  tile and the instructions from a codeword to t.
- **Pinned by:** `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 banked ilp: variant 0 is today's (no defines,
  the stock kernel names), ..." (CPU: each name distinct, its defines, each define's default in `header_group`, the
  config's count of 12); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu banked ilp: every grouped GEMV
  variant's gate|up outputs and rows, ..." (GPU: every variant, both GEMVs at it, on g4 and g4m3, against g4 at variant
  0: hd, hs and y word for word, a skipped row-slot's y row 0, rows 1, 2, 3, 4, 6, 7, 12, 16, 24, 36 and 48, x bf16 and
  f32, trellis I16 and U16, at DeepSeek-V4.1's routed geometry (hidden 5120, width 2304, limit 10, MUL1) over synthetic
  banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32)); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu
  banked ilp t routes: every EXL3_T1V route gives today's route's t, ..." (GPU: the t routes of `t4`, `t1`, `t0` on
  every codeword state, through `turboexl3_t1v_probe_<route>_<dtype>`); `src/core/config.zig` "turboexl3 config: the
  defaults are valid and served; an unserved form, slice count ..." (CPU: a variant needs `banked_group` 4).

### `turboexl3_down3bgm3`

<!-- kernel-facts: turboexl3_down3bgm3 -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:331` (`const Banked`) |
| inputs | `hdp`, `slots`, `down_t0`, `down_t1`, `down_t2` |
| outputs | `zd` |
| source | `text_down3bgm3` = `decode/down3_1` ++ `decode/rk_line` ++ `deepseek_v41:banked/down3g_2` ++ `deepseek_v41:banked/down3b_w` ++ `deepseek_v41:banked/down3g_3m3` (2756 B, `3274179feb75`) |
| header | `header_group` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26373 B, `b7023ef0567c`) |
| made at | `src/models/deepseek_v41/banked.zig:411` (`const Banked`) |
| inputs | `hdp`, `slots`, `down_t0`, `down_t1`, `down_t2` |
| outputs | `zd` |
| source | `text_down3bgm3` = `decode/down3_1` ++ `decode/rk_line` ++ `deepseek_v41:banked/down3g_2` ++ `deepseek_v41:banked/down3b_w` ++ `deepseek_v41:banked/down3g_3m3` (2756 B, `3274179feb75`) |
| header | `group_headers[v].ptr` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26373 B, `b7023ef0567c`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 4 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots) on hw4b's texts (g4m3): `Config.banked_group` 4 with `Config.banked_m3`, or
  `BankedSel` g4m3. Not served by default. Its name is also variant 0's in the M = 3 variant table
  (`group_down_m3_names[0]`).
- **Shapes and dtypes:** As `turboexl3_down3b`: hdp f16 [rows, inter]; slots the packed ids u32 or i32 [rows], `skip_id`
  0xFFFFFFFF (`root.banked_skip_id`) marks a row-slot that runs no expert (no decode, no plane rows written);
  down_t0/1/2 U16 or I16 [E_b, inter/16, hidden/16, n]. Out: zd f32 [KS_dn, rows, hidden].
- **Grid and threadgroup:** As `turboexl3_down3b`: `((hidden/(16·BT))·128, rows, z)` / `(128, 1, 1)`, z = KS_dn by
  default. (`cfgsBankedGM3`).
- **Template args:** NHW (n, the halfwords per tile: K = NHW/16), CODEBOOK (MUL1 0, MCG 1), WBITS (the window's bits),
  HID (hidden), INTER (inter), RKC (rows), KS (KS_dn), SPT (KS_dn / z: 1 by default), BT (output tiles per threadgroup:
  `bt1` at one row-slot, `bt9` from two, both 2 by default), GROUP (4 from two row-slots, 1 at one: up to four row-slots
  of one packed id per lead threadgroup (`turboexl3_members_g`), a skipped row-slot no member), DEFER (1 for MUL1: the
  GEMV sums x times the codeword's byte sum). Every rate `bankedCall` admits (gate, up and down at one rate), both
  codebooks and every window. Inside: the member paths M 1, 2, 3 and 4.
- **Replaces:** `turboexl3_down3bg` in the same launch: its text with one line added (`down3g_3m3`), an M = 3 path for a
  lead with three members, where `down3bg` runs the M = 4 path with a fourth member's FMAs padding.
- **Bits:** byte-identical to `turboexl3_down3bg` (g4): every member fed in the one-member order
  (`turboexl3_gemv_group`); the call's hd, hs and y equal today's, differing words 0, max_ulps 0, a skipped row-slot's y
  row 0.
- **Pinned by:** `src/models/deepseek_v41/banked_grp_test.zig` "turboexl3 banked grp: the M = 3 texts are today's
  grouped texts with the one M = 3 line ..." (CPU: today's grouped text with the one M = 3 line added after the M = 2
  line); `src/models/deepseek_v41/banked_grp_test.zig` "turboexl3 gpu banked grp: the M = 3 texts' gate|up outputs and
  rows are today's, ..." (GPU: g4m3 against g4, hd, hs and y word for word, rows 1 to 9, 12, 16, 24, 36, 42, 48 and 64,
  x bf16 and f32, trellis I16 and U16, one id on up to 16 row-slots, skipped row-slots, at DeepSeek-V4.1's routed
  geometry (hidden 5120, width 2304, limit 10, MUL1) over synthetic banks of 20, 12 and 8 experts, K3 and K2 (n 48,
  32)); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 banked ilp: variant 0 is today's (no defines, the stock
  kernel names), ..." (CPU: the names distinct); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu banked
  ilp: every grouped GEMV variant's gate|up outputs and rows, ..." (GPU: variant 0 on hw4b's texts against g4's variant
  0); `src/core/config.zig` "turboexl3 config: the defaults are valid and served; an unserved form, slice count ..."
  (CPU: `banked_m3` needs `banked_group` 4).

### `turboexl3_down3bgm3_<variant>`

<!-- kernel-facts: turboexl3_down3bgm3_<variant> -->

| fact | from the code |
|---|---|
| names | `turboexl3_down3bgm3_ku`, `turboexl3_down3bgm3_pf`, `turboexl3_down3bgm3_xh`, `turboexl3_down3bgm3_m2`, `turboexl3_down3bgm3_sh`, `turboexl3_down3bgm3_t4`, `turboexl3_down3bgm3_t1`, `turboexl3_down3bgm3_t0`, `turboexl3_down3bgm3_pfxh`, `turboexl3_down3bgm3_pfsh`, `turboexl3_down3bgm3_xhm2` |
| made at | `src/models/deepseek_v41/banked.zig:411` (`const Banked`) |
| inputs | `hdp`, `slots`, `down_t0`, `down_t1`, `down_t2` |
| outputs | `zd` |
| source | `text_down3bgm3` = `decode/down3_1` ++ `decode/rk_line` ++ `deepseek_v41:banked/down3g_2` ++ `deepseek_v41:banked/down3b_w` ++ `deepseek_v41:banked/down3g_3m3` (2756 B, `3274179feb75`) |
| header | `group_headers[v].ptr`: a variant's defines ++ the base header, one per name |
| header of `turboexl3_down3bgm3_ku` | text '#define EXL3_KU 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26391 B, `893d0732e30e`) |
| header of `turboexl3_down3bgm3_pf` | text '#define EXL3_GPF 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `719802404fb5`) |
| header of `turboexl3_down3bgm3_xh` | text '#define EXL3_GXH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `35cacdcfb1d7`) |
| header of `turboexl3_down3bgm3_m2` | text '#define EXL3_GM2 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `d28773b43d18`) |
| header of `turboexl3_down3bgm3_sh` | text '#define EXL3_GSH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `1eaa64821218`) |
| header of `turboexl3_down3bgm3_t4` | text '#define EXL3_T1V 4\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `acaeb8fa61e1`) |
| header of `turboexl3_down3bgm3_t1` | text '#define EXL3_T1V 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `b0622147baad`) |
| header of `turboexl3_down3bgm3_t0` | text '#define EXL3_T1V 0\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `9be4db5164be`) |
| header of `turboexl3_down3bgm3_pfxh` | text '#define EXL3_GPF 1\n#define EXL3_GXH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26411 B, `c5b9183f1adc`) |
| header of `turboexl3_down3bgm3_pfsh` | text '#define EXL3_GPF 1\n#define EXL3_GSH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26411 B, `2478957c5a9f`) |
| header of `turboexl3_down3bgm3_xhm2` | text '#define EXL3_GXH 1\n#define EXL3_GM2 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26411 B, `a43b1425ae6c`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 4 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots) at `Config.banked_variant_dn` v (1 to 11, needs `Config.banked_group` 4) on hw4b's
  texts (`banked_m3` true), or `moeClampedBankedDownV` / `...SelV` with that variant; `Banked.groupGemvs` makes each
  on first use (`banked_gvm3`). hw2's ILP retune; not served by default (variant 0).
- **Shapes and dtypes:** As `turboexl3_down3bgm3`.
- **Grid and threadgroup:** As `turboexl3_down3bgm3` (`cfgsBankedGM3`).
- **Template args:** As `turboexl3_down3bgm3`. The variant is the header: its Metal defines before `header_group`, under
  a kernel name of its own: `ku` EXL3_KU 1 (input tiles in pairs, both tiles' words loaded before either decodes); `pf`
  EXL3_GPF 1 (a rolling prefetch: the next input tile's words loaded before this tile decodes); `xh` EXL3_GXH 1 (each
  member's x quad held as half4 and widened at its FMA); `m2` EXL3_GM2 1 (the four-member path as two passes of two
  members, the second reading and decoding the tile again); `sh` EXL3_GSH 1 (the one-64-bit-window reader with each tile
  word loaded by one lane and taken by the others through `simd_shuffle`); `t4`, `t1`, `t0` EXL3_T1V 4, 1, 0 (the route
  from a MUL1 codeword to t, default 3: 4 both pair sums as one half2, 1 a shift and add for the 32-bit multiply, 0 the
  multiply by 0x10001 and an integer convert; MUL1 only); `pfxh`, `pfsh`, `xhm2` two of those defines together.
- **Replaces:** `turboexl3_down3bgm3` in the same launch.
- **Bits:** byte-identical to `turboexl3_down3bgm3` (variant 0): each variant reads, decodes and sums every weight as
  variant 0 does (the same codewords, decoded values, FMA sequence into the same f32 accumulators, and folds); they
  differ only in when the tile words load, how many lanes load them, the registers the loop holds, the passes over a
  tile and the instructions from a codeword to t. The GPU test holds each to g4's variant 0, so to `turboexl3_down3bg`.
- **Pinned by:** `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 banked ilp: variant 0 is today's (no defines,
  the stock kernel names), ..." (CPU: each name distinct, its defines, each define's default in `header_group`, the
  config's count of 12); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu banked ilp: every grouped GEMV
  variant's gate|up outputs and rows, ..." (GPU: every variant, both GEMVs at it, on g4 and g4m3, against g4 at variant
  0: hd, hs and y word for word, a skipped row-slot's y row 0, rows 1, 2, 3, 4, 6, 7, 12, 16, 24, 36 and 48, x bf16 and
  f32, trellis I16 and U16, at DeepSeek-V4.1's routed geometry (hidden 5120, width 2304, limit 10, MUL1) over synthetic
  banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32)); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu
  banked ilp t routes: every EXL3_T1V route gives today's route's t, ..." (GPU: the t routes of `t4`, `t1`, `t0` on
  every codeword state, through `turboexl3_t1v_probe_<route>_<dtype>`); `src/core/config.zig` "turboexl3 config: the
  defaults are valid and served; an unserved form, slice count ..." (CPU: a variant needs `banked_group` 4).

### `turboexl3_fin7cb`

<!-- kernel-facts: turboexl3_fin7cb -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:256` (`const Banked`) |
| inputs | `zg`, `zu`, `slots`, `gate_svh0`, `gate_svh1`, `gate_svh2`, `up_svh0`, `up_svh1`, `up_svh2`, `down_suh0`, `down_suh1`, `down_suh2`, `xsg`, `xsu` |
| outputs | `hd`, `hs` |
| source | `text_fin7cb` = `decode/fin7_1` ++ `deepseek_v41:banked/fin7b_expert` ++ `decode/fin7_2` ++ `decode/fin7_planes` ++ `decode/fin7_3` ++ `decode/fin7_sums` ++ `decode/fin7_4` ++ `decode/fin7_swiglu_clamp` ++ `decode/fin7_5` (2959 B, `68747f002dd1`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 3 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots), the end of its gate|up half; it prepares the down's input, so it reads every bank's
  down suh as well as its gate and up svh. Served by default: `Config.banked_group` 2 runs the banked texts.
- **Shapes and dtypes:** zg, zu f32 [KS_gu, rows, inter] (`turboexl3_gu3b`'s planes); slots the ids u32 or i32 [rows],
  each packed (bank << 24 | slot row); gate_svh0/1/2 and up_svh0/1/2 f16 [E_b, inter]; down_suh0/1/2 f16 [E_b, inter];
  xsg, xsu f32 [rows, hidden/128] (`turboexl3_prep2b`'s sg, su). Out: hd f16 [rows, inter] (the down's prepared input,
  lane order) and hs f32 [rows, inter/128] (its 128-block sums): `root.BankedHalf`.
- **Grid and threadgroup:** `((inter/128)·32, rows, 1)` / `(32, 1, 1)`.
- **Template args:** INTER (inter), RKC (rows), KS (KS_gu), DEFER (1 for MUL1: the deferred affine map applied here),
  NBH (hidden/128), LIMIT (the clamp, 1 to 128; DeepSeek-V4.1's swiglu_limit 10). A banked call needs a top-1 geometry
  (`bankedKernels` refuses any other); an unclamped one builds `turboexl3_fin7b`, Ling's clamp `turboexl3_fin7cob`.
- **Replaces:** `turboexl3_fin7c` (fin7 with the clamped SwiGLU), launched once per bank by each bank's own `moeClamped`
  call: one launch over the three banks.
- **Bits:** byte-identical to `moeClamped` on the pair's bank at top-1 with score 1: `fin7b_expert` (the slot row and
  the bank's gate svh, up svh and down suh) replaces only the line that reads a bank's arrays, so every other statement
  is `fin7c`'s. The streamer's sum, each full token's six pairs weighed, is held within 5e-3 of the clamped host
  reference's rms (`ref.compare` against `ref.moeRowClamped`).
- **Pinned by:** `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode texts: each is its base text with
  the bank's lines in place ..." (CPU: the text is its base text with `fin7b_expert` in place of the base line and
  nothing else, every input read, no shape buffer); `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode
  refusals: the clamp, the ids' shape and dtype, the banks' dtypes, ..." (CPU: the call's refusals, before a launch);
  `src/models/deepseek_v41/banked_test.zig` "turboexl3 gpu banked decode: every pair of moeClampedBanked over three
  banks is moeClamped's ..." (GPU: every pair byte for byte against `moeClamped` at rows 1, 2, 6, 7, 16, 36 and 48, x
  bf16 and f32, trellis U16 and I16, at DeepSeek-V4.1's routed geometry (hidden 5120, width 2304, limit 10, MUL1) over
  synthetic banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32); sampled token sums to the reference by 5e-3).

### `turboexl3_fin7cbg`

<!-- kernel-facts: turboexl3_fin7cbg -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:291` (`const Banked`) |
| inputs | `zg`, `zu`, `slots`, `gate_svh0`, `gate_svh1`, `gate_svh2`, `up_svh0`, `up_svh1`, `up_svh2`, `down_suh0`, `down_suh1`, `down_suh2`, `xsg`, `xsu` |
| outputs | `hd`, `hs` |
| source | `text_fin7cbg` = `decode/fin7_1` ++ `deepseek_v41:banked/skip_rk` ++ `deepseek_v41:banked/fin7b_expert` ++ `decode/fin7_2` ++ `decode/fin7_planes` ++ `decode/fin7_3` ++ `decode/fin7_sums` ++ `decode/fin7_4` ++ `decode/fin7_swiglu_clamp` ++ `decode/fin7_5` (3076 B, `25e578bf80c8`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |
| made at | `src/models/deepseek_v41/banked.zig:325` (`const Banked`) (the same text) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 3 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots) on the grouped texts, g4 and g4m3 (made at both sites). Not served by default:
  `Config.banked_group` 4 selects the grouped texts, or a call through `moeClampedBankedSel*` with `BankedSel` g4 or
  g4m3 (hw4b's check mode, the tests).
- **Shapes and dtypes:** As `turboexl3_fin7cb`; `skip_id` 0xFFFFFFFF (`root.banked_skip_id`) marks a row-slot that runs
  no expert, and its hd and hs rows are left unwritten.
- **Grid and threadgroup:** As `turboexl3_fin7cb`: `((inter/128)·32, rows, 1)` / `(32, 1, 1)`.
- **Template args:** INTER, RKC (rows), KS (KS_gu), DEFER, NBH (hidden/128), LIMIT, as `turboexl3_fin7cb`.
- **Replaces:** `turboexl3_fin7cb` in the same launch: its text with `skip_rk` before the bank lines.
- **Bits:** byte-identical to `turboexl3_fin7cb` for every row-slot not skipped (hd and hs).
- **Pinned by:** `src/models/deepseek_v41/banked_test.zig` "turboexl3 grouped banked decode texts: prep2bg, fin7cbg and
  dfin3bg are the banked texts ..." (CPU: the text is the banked text with the skip line before its bank lines);
  `src/models/deepseek_v41/banked_test.zig` "turboexl3 gpu banked group: the grouped banked call's rows are the banked
  call's ..." (GPU: the grouped call's rows equal the banked call's byte for byte, a skipped row-slot's y row 0, rows 6,
  12, 36 and 48, x bf16, trellis I16, at DeepSeek-V4.1's routed geometry (hidden 5120, width 2304, limit 10, MUL1) over
  synthetic banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32)). It runs in every arm of
  `src/models/deepseek_v41/banked_grp_test.zig` "turboexl3 gpu banked grp: the M = 3 texts' gate|up outputs and rows are
  today's, ..." and `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu banked ilp: every grouped GEMV
  variant's gate|up outputs and rows, ..." (GPU), which hold the whole call's hd, hs and y.

### `turboexl3_gu3b`

<!-- kernel-facts: turboexl3_gu3b -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:252` (`const Banked`) |
| inputs | `xgp`, `xup`, `slots`, `gate_t0`, `gate_t1`, `gate_t2`, `up_t0`, `up_t1`, `up_t2` |
| outputs | `zg`, `zu` |
| source | `text_gu3b` = `decode/gu3_1` ++ `decode/rk_line` ++ `decode/gu3_2` ++ `deepseek_v41:banked/gu3b_w` ++ `decode/gu3_3` ++ `decode/gu3_stores` ++ `decode/gu3_4` (2717 B, `cf09a195e002`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 2 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots), in its gate|up half. Served by default: `Config.banked_group` 2 runs the banked
  texts.
- **Shapes and dtypes:** xgp, xup f16 [rows, hidden] (`turboexl3_prep2b`'s xg, xu, lane order); slots the ids u32 or i32
  [rows], each packed (bank << 24 | slot row); gate_t0/1/2 and up_t0/1/2 U16 or I16 [E_b, hidden/16, inter/16, n] (bank
  b's trellis, every bank one dtype; I16 is the streamer's code arrays). Out: zg, zu f32 [KS_gu, rows, inter] (the
  split-K gate and up planes). KS_gu is the gate|up's K slices, the largest of 4, 2, 1 with (hidden/16) % (2·KS) = 0
  (`Kernels3.initWith(geo, 1, 4, 2)` from `root.kernelsFor`).
- **Grid and threadgroup:** `((inter/(16·BT))·128, rows, z)` / `(128, 1, 1)`, z = max(1, KS_gu / kdiv9[min(rows, 4) −
  1]) threadgroups along K (`kdiv9` 1 by default, so z = KS_gu).
- **Template args:** NHW (n, the halfwords per tile: K = NHW/16), CODEBOOK (MUL1 0, MCG 1), WBITS (the window's bits),
  HID (hidden), INTER (inter), RKC (rows), KS (KS_gu), SPT (KS_gu / z: 1 by default), BT (output tiles per threadgroup:
  `bt1` at one row-slot, `bt9` from two, both 2 by default), GROUP (2 from two row-slots, so two row-slots of one packed
  id share a decode only on one bank's one slot; 1 at one), DEFER (1 for MUL1: the GEMV sums x times the codeword's byte
  sum). Every rate `bankedCall` admits (gate, up and down at one rate), both codebooks and every window.
- **Replaces:** `turboexl3_gu3`, launched once per bank by each bank's own `moeClamped` call: one launch over the three
  banks.
- **Bits:** byte-identical to `moeClamped` on the pair's bank at top-1 with score 1: `gu3b_w` (the weight base: the
  bank's trellis at the slot row) replaces only the line that reads a bank's arrays, so every other statement is
  `gu3`'s. The streamer's sum, each full token's six pairs weighed, is held within 5e-3 of the clamped host reference's
  rms (`ref.compare` against `ref.moeRowClamped`).
- **Pinned by:** `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode texts: each is its base text with
  the bank's lines in place ..." (CPU: the text is its base text with `gu3b_w` in place of the base line and nothing
  else, every input read, no shape buffer); `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode refusals:
  the clamp, the ids' shape and dtype, the banks' dtypes, ..." (CPU: the call's refusals, before a launch);
  `src/models/deepseek_v41/banked_test.zig` "turboexl3 gpu banked decode: every pair of moeClampedBanked over three
  banks is moeClamped's ..." (GPU: every pair byte for byte against `moeClamped` at rows 1, 2, 6, 7, 16, 36 and 48, x
  bf16 and f32, trellis U16 and I16, at DeepSeek-V4.1's routed geometry (hidden 5120, width 2304, limit 10, MUL1) over
  synthetic banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32); sampled token sums to the reference by 5e-3). The texts
  test also pins that the members compare the packed id.

### `turboexl3_gu3bg`

<!-- kernel-facts: turboexl3_gu3bg -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:287` (`const Banked`) |
| inputs | `xgp`, `xup`, `slots`, `gate_t0`, `gate_t1`, `gate_t2`, `up_t0`, `up_t1`, `up_t2` |
| outputs | `zg`, `zu` |
| source | `text_gu3bg` = `decode/gu3_1` ++ `decode/rk_line` ++ `deepseek_v41:banked/gu3g_2` ++ `deepseek_v41:banked/gu3b_w` ++ `deepseek_v41:banked/gu3g_3` (2796 B, `e427ea27f330`) |
| header | `header_group` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26373 B, `b7023ef0567c`) |
| made at | `src/models/deepseek_v41/banked.zig:408` (`const Banked`) |
| inputs | `xgp`, `xup`, `slots`, `gate_t0`, `gate_t1`, `gate_t2`, `up_t0`, `up_t1`, `up_t2` |
| outputs | `zg`, `zu` |
| source | `text_gu3bg` = `decode/gu3_1` ++ `decode/rk_line` ++ `deepseek_v41:banked/gu3g_2` ++ `deepseek_v41:banked/gu3b_w` ++ `deepseek_v41:banked/gu3g_3` (2796 B, `e427ea27f330`) |
| header | `group_headers[v].ptr` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26373 B, `b7023ef0567c`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 2 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots) on today's grouped texts (g4): `Config.banked_group` 4 with `banked_m3` false, or
  `BankedSel` g4. Not served by default. Its name is also variant 0's in the variant table (`group_gu_names[0]`;
  `groupGemvs` makes only variants 1 and up).
- **Shapes and dtypes:** As `turboexl3_gu3b`: xgp, xup f16 [rows, hidden]; slots the packed ids u32 or i32 [rows],
  `skip_id` 0xFFFFFFFF (`root.banked_skip_id`) marks a row-slot that runs no expert (no decode, no plane rows written);
  gate_t0/1/2, up_t0/1/2 U16 or I16 [E_b, hidden/16, inter/16, n]. Out: zg, zu f32 [KS_gu, rows, inter].
- **Grid and threadgroup:** As `turboexl3_gu3b`: `((inter/(16·BT))·128, rows, z)` / `(128, 1, 1)`, z = KS_gu by default.
  (`cfgsBankedG`).
- **Template args:** NHW (n, the halfwords per tile: K = NHW/16), CODEBOOK (MUL1 0, MCG 1), WBITS (the window's bits),
  HID (hidden), INTER (inter), RKC (rows), KS (KS_gu), SPT (KS_gu / z: 1 by default), BT (output tiles per threadgroup:
  `bt1` at one row-slot, `bt9` from two, both 2 by default), GROUP (4 from two row-slots, 1 at one: up to four row-slots
  of one packed id per lead threadgroup (`turboexl3_members_g`), a skipped row-slot no member), DEFER (1 for MUL1: the
  GEMV sums x times the codeword's byte sum). Every rate `bankedCall` admits (gate, up and down at one rate), both
  codebooks and every window. Inside: the member paths M 1, 2 and 4 of `turboexl3_gemv_group`, so a lead of three
  members runs the M = 4 path with a fourth member's FMAs padding.
- **Replaces:** `turboexl3_gu3b` in the same launch: the members widened from two row-slots of one packed id to four
  (`gu3g_2`, `gu3g_3`), the banked weight base (`gu3b_w`) between them.
- **Bits:** byte-identical to the banked text `turboexl3_gu3b` (the grouped call's rows are the banked call's): every
  weight is read and decoded once and fed to each member in the one-member order.
- **Pinned by:** `src/models/deepseek_v41/banked_test.zig` "turboexl3 grouped banked decode texts: prep2bg, fin7cbg and
  dfin3bg are the banked texts ..." (CPU: the banked text's constants and grid line, then the four-member head, the
  banked weight base, the four-member tail); `src/models/deepseek_v41/banked_test.zig` "turboexl3 gpu banked group: the
  grouped banked call's rows are the banked call's ..." (GPU: the grouped call against the banked call, byte for byte, a
  skipped row-slot's y row 0, rows 6, 12, 36 and 48, x bf16, trellis I16, at DeepSeek-V4.1's routed geometry (hidden
  5120, width 2304, limit 10, MUL1) over synthetic banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32); up to eight
  row-slots of one id: two leads); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 banked ilp: variant 0 is
  today's (no defines, the stock kernel names), ..." (CPU: variant 0 keeps the stock name and no defines). It is the
  reference arm (g4, variant 0) of `src/models/deepseek_v41/banked_grp_test.zig` "turboexl3 gpu banked grp: the M = 3
  texts' gate|up outputs and rows are today's, ..." and `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu
  banked ilp: every grouped GEMV variant's gate|up outputs and rows, ..." (GPU).

### `turboexl3_gu3bg_<variant>`

<!-- kernel-facts: turboexl3_gu3bg_<variant> -->

| fact | from the code |
|---|---|
| names | `turboexl3_gu3bg_ku`, `turboexl3_gu3bg_pf`, `turboexl3_gu3bg_xh`, `turboexl3_gu3bg_m2`, `turboexl3_gu3bg_sh`, `turboexl3_gu3bg_t4`, `turboexl3_gu3bg_t1`, `turboexl3_gu3bg_t0`, `turboexl3_gu3bg_pfxh`, `turboexl3_gu3bg_pfsh`, `turboexl3_gu3bg_xhm2` |
| made at | `src/models/deepseek_v41/banked.zig:408` (`const Banked`) |
| inputs | `xgp`, `xup`, `slots`, `gate_t0`, `gate_t1`, `gate_t2`, `up_t0`, `up_t1`, `up_t2` |
| outputs | `zg`, `zu` |
| source | `text_gu3bg` = `decode/gu3_1` ++ `decode/rk_line` ++ `deepseek_v41:banked/gu3g_2` ++ `deepseek_v41:banked/gu3b_w` ++ `deepseek_v41:banked/gu3g_3` (2796 B, `e427ea27f330`) |
| header | `group_headers[v].ptr`: a variant's defines ++ the base header, one per name |
| header of `turboexl3_gu3bg_ku` | text '#define EXL3_KU 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26391 B, `893d0732e30e`) |
| header of `turboexl3_gu3bg_pf` | text '#define EXL3_GPF 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `719802404fb5`) |
| header of `turboexl3_gu3bg_xh` | text '#define EXL3_GXH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `35cacdcfb1d7`) |
| header of `turboexl3_gu3bg_m2` | text '#define EXL3_GM2 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `d28773b43d18`) |
| header of `turboexl3_gu3bg_sh` | text '#define EXL3_GSH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `1eaa64821218`) |
| header of `turboexl3_gu3bg_t4` | text '#define EXL3_T1V 4\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `acaeb8fa61e1`) |
| header of `turboexl3_gu3bg_t1` | text '#define EXL3_T1V 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `b0622147baad`) |
| header of `turboexl3_gu3bg_t0` | text '#define EXL3_T1V 0\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `9be4db5164be`) |
| header of `turboexl3_gu3bg_pfxh` | text '#define EXL3_GPF 1\n#define EXL3_GXH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26411 B, `c5b9183f1adc`) |
| header of `turboexl3_gu3bg_pfsh` | text '#define EXL3_GPF 1\n#define EXL3_GSH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26411 B, `2478957c5a9f`) |
| header of `turboexl3_gu3bg_xhm2` | text '#define EXL3_GXH 1\n#define EXL3_GM2 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26411 B, `a43b1425ae6c`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 2 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots) at `Config.banked_variant_gu` v (1 to 11, needs `Config.banked_group` 4) on today's
  grouped texts (`banked_m3` false), or `moeClampedBankedGateUpV` / `...SelV` with that variant; `Banked.groupGemvs`
  makes each on first use (`banked_gv`). hw2's ILP retune; not served by default (variant 0).
- **Shapes and dtypes:** As `turboexl3_gu3bg`.
- **Grid and threadgroup:** As `turboexl3_gu3bg` (the configs of its texts, `cfgsBankedG`).
- **Template args:** As `turboexl3_gu3bg`. The variant is the header: its Metal defines before `header_group`, under a
  kernel name of its own (MLX keys a custom kernel's library by its name): `ku` EXL3_KU 1 (input tiles in pairs, both
  tiles' words loaded before either decodes); `pf` EXL3_GPF 1 (a rolling prefetch: the next input tile's words loaded
  before this tile decodes); `xh` EXL3_GXH 1 (each member's x quad held as half4 and widened at its FMA); `m2` EXL3_GM2
  1 (the four-member path as two passes of two members, the second reading and decoding the tile again); `sh` EXL3_GSH 1
  (the one-64-bit-window reader with each tile word loaded by one lane and taken by the others through `simd_shuffle`);
  `t4`, `t1`, `t0` EXL3_T1V 4, 1, 0 (the route from a MUL1 codeword to t, default 3: 4 both pair sums as one half2, 1 a
  shift and add for the 32-bit multiply, 0 the multiply by 0x10001 and an integer convert; MUL1 only); `pfxh`, `pfsh`,
  `xhm2` two of those defines together.
- **Replaces:** `turboexl3_gu3bg` in the same launch.
- **Bits:** byte-identical to `turboexl3_gu3bg` (variant 0): each variant reads, decodes and sums every weight as
  variant 0 does (the same codewords, decoded values, FMA sequence into the same f32 accumulators, and folds); they
  differ only in when the tile words load, how many lanes load them, the registers the loop holds, the passes over a
  tile and the instructions from a codeword to t.
- **Pinned by:** `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 banked ilp: variant 0 is today's (no defines,
  the stock kernel names), ..." (CPU: each name distinct, its defines, each define's default in `header_group`, the
  config's count of 12); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu banked ilp: every grouped GEMV
  variant's gate|up outputs and rows, ..." (GPU: every variant, both GEMVs at it, on g4 and g4m3, against g4 at variant
  0: hd, hs and y word for word, a skipped row-slot's y row 0, rows 1, 2, 3, 4, 6, 7, 12, 16, 24, 36 and 48, x bf16 and
  f32, trellis I16 and U16, at DeepSeek-V4.1's routed geometry (hidden 5120, width 2304, limit 10, MUL1) over synthetic
  banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32)); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu
  banked ilp t routes: every EXL3_T1V route gives today's route's t, ..." (GPU: the t routes of `t4`, `t1`, `t0` on
  every codeword state, through `turboexl3_t1v_probe_<route>_<dtype>`); `src/core/config.zig` "turboexl3 config: the
  defaults are valid and served; an unserved form, slice count ..." (CPU: a variant needs `banked_group` 4).

### `turboexl3_gu3bgm3`

<!-- kernel-facts: turboexl3_gu3bgm3 -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:321` (`const Banked`) |
| inputs | `xgp`, `xup`, `slots`, `gate_t0`, `gate_t1`, `gate_t2`, `up_t0`, `up_t1`, `up_t2` |
| outputs | `zg`, `zu` |
| source | `text_gu3bgm3` = `decode/gu3_1` ++ `decode/rk_line` ++ `deepseek_v41:banked/gu3g_2` ++ `deepseek_v41:banked/gu3b_w` ++ `deepseek_v41:banked/gu3g_3m3` (2967 B, `5d637580686f`) |
| header | `header_group` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26373 B, `b7023ef0567c`) |
| made at | `src/models/deepseek_v41/banked.zig:406` (`const Banked`) |
| inputs | `xgp`, `xup`, `slots`, `gate_t0`, `gate_t1`, `gate_t2`, `up_t0`, `up_t1`, `up_t2` |
| outputs | `zg`, `zu` |
| source | `text_gu3bgm3` = `decode/gu3_1` ++ `decode/rk_line` ++ `deepseek_v41:banked/gu3g_2` ++ `deepseek_v41:banked/gu3b_w` ++ `deepseek_v41:banked/gu3g_3m3` (2967 B, `5d637580686f`) |
| header | `group_headers[v].ptr` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26373 B, `b7023ef0567c`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 2 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots) on hw4b's texts (g4m3): `Config.banked_group` 4 with `Config.banked_m3`, or
  `BankedSel` g4m3. Not served by default. Its name is also variant 0's in the M = 3 variant table
  (`group_gu_m3_names[0]`).
- **Shapes and dtypes:** As `turboexl3_gu3b`: xgp, xup f16 [rows, hidden]; slots the packed ids u32 or i32 [rows],
  `skip_id` 0xFFFFFFFF (`root.banked_skip_id`) marks a row-slot that runs no expert (no decode, no plane rows written);
  gate_t0/1/2, up_t0/1/2 U16 or I16 [E_b, hidden/16, inter/16, n]. Out: zg, zu f32 [KS_gu, rows, inter].
- **Grid and threadgroup:** As `turboexl3_gu3b`: `((inter/(16·BT))·128, rows, z)` / `(128, 1, 1)`, z = KS_gu by default.
  (`cfgsBankedGM3`).
- **Template args:** NHW (n, the halfwords per tile: K = NHW/16), CODEBOOK (MUL1 0, MCG 1), WBITS (the window's bits),
  HID (hidden), INTER (inter), RKC (rows), KS (KS_gu), SPT (KS_gu / z: 1 by default), BT (output tiles per threadgroup:
  `bt1` at one row-slot, `bt9` from two, both 2 by default), GROUP (4 from two row-slots, 1 at one: up to four row-slots
  of one packed id per lead threadgroup (`turboexl3_members_g`), a skipped row-slot no member), DEFER (1 for MUL1: the
  GEMV sums x times the codeword's byte sum). Every rate `bankedCall` admits (gate, up and down at one rate), both
  codebooks and every window. Inside: the member paths M 1, 2, 3 and 4.
- **Replaces:** `turboexl3_gu3bg` in the same launch: its text with one line added (`gu3g_3m3`), an M = 3 path for a
  lead with three members, where `gu3bg` runs the M = 4 path with a fourth member's FMAs padding.
- **Bits:** byte-identical to `turboexl3_gu3bg` (g4): every member fed in the one-member order (`turboexl3_gemv_group`);
  the call's hd, hs and y equal today's, differing words 0, max_ulps 0, a skipped row-slot's y row 0.
- **Pinned by:** `src/models/deepseek_v41/banked_grp_test.zig` "turboexl3 banked grp: the M = 3 texts are today's
  grouped texts with the one M = 3 line ..." (CPU: today's grouped text with the one M = 3 line added after the M = 2
  line); `src/models/deepseek_v41/banked_grp_test.zig` "turboexl3 gpu banked grp: the M = 3 texts' gate|up outputs and
  rows are today's, ..." (GPU: g4m3 against g4, hd, hs and y word for word, rows 1 to 9, 12, 16, 24, 36, 42, 48 and 64,
  x bf16 and f32, trellis I16 and U16, one id on up to 16 row-slots, skipped row-slots, at DeepSeek-V4.1's routed
  geometry (hidden 5120, width 2304, limit 10, MUL1) over synthetic banks of 20, 12 and 8 experts, K3 and K2 (n 48,
  32)); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 banked ilp: variant 0 is today's (no defines, the stock
  kernel names), ..." (CPU: the names distinct); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu banked
  ilp: every grouped GEMV variant's gate|up outputs and rows, ..." (GPU: variant 0 on hw4b's texts against g4's variant
  0); `src/core/config.zig` "turboexl3 config: the defaults are valid and served; an unserved form, slice count ..."
  (CPU: `banked_m3` needs `banked_group` 4).

### `turboexl3_gu3bgm3_<variant>`

<!-- kernel-facts: turboexl3_gu3bgm3_<variant> -->

| fact | from the code |
|---|---|
| names | `turboexl3_gu3bgm3_ku`, `turboexl3_gu3bgm3_pf`, `turboexl3_gu3bgm3_xh`, `turboexl3_gu3bgm3_m2`, `turboexl3_gu3bgm3_sh`, `turboexl3_gu3bgm3_t4`, `turboexl3_gu3bgm3_t1`, `turboexl3_gu3bgm3_t0`, `turboexl3_gu3bgm3_pfxh`, `turboexl3_gu3bgm3_pfsh`, `turboexl3_gu3bgm3_xhm2` |
| made at | `src/models/deepseek_v41/banked.zig:406` (`const Banked`) |
| inputs | `xgp`, `xup`, `slots`, `gate_t0`, `gate_t1`, `gate_t2`, `up_t0`, `up_t1`, `up_t2` |
| outputs | `zg`, `zu` |
| source | `text_gu3bgm3` = `decode/gu3_1` ++ `decode/rk_line` ++ `deepseek_v41:banked/gu3g_2` ++ `deepseek_v41:banked/gu3b_w` ++ `deepseek_v41:banked/gu3g_3m3` (2967 B, `5d637580686f`) |
| header | `group_headers[v].ptr`: a variant's defines ++ the base header, one per name |
| header of `turboexl3_gu3bgm3_ku` | text '#define EXL3_KU 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26391 B, `893d0732e30e`) |
| header of `turboexl3_gu3bgm3_pf` | text '#define EXL3_GPF 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `719802404fb5`) |
| header of `turboexl3_gu3bgm3_xh` | text '#define EXL3_GXH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `35cacdcfb1d7`) |
| header of `turboexl3_gu3bgm3_m2` | text '#define EXL3_GM2 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `d28773b43d18`) |
| header of `turboexl3_gu3bgm3_sh` | text '#define EXL3_GSH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `1eaa64821218`) |
| header of `turboexl3_gu3bgm3_t4` | text '#define EXL3_T1V 4\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `acaeb8fa61e1`) |
| header of `turboexl3_gu3bgm3_t1` | text '#define EXL3_T1V 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `b0622147baad`) |
| header of `turboexl3_gu3bgm3_t0` | text '#define EXL3_T1V 0\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26392 B, `9be4db5164be`) |
| header of `turboexl3_gu3bgm3_pfxh` | text '#define EXL3_GPF 1\n#define EXL3_GXH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26411 B, `c5b9183f1adc`) |
| header of `turboexl3_gu3bgm3_pfsh` | text '#define EXL3_GPF 1\n#define EXL3_GSH 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26411 B, `2478957c5a9f`) |
| header of `turboexl3_gu3bgm3_xhm2` | text '#define EXL3_GXH 1\n#define EXL3_GM2 1\n' ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `deepseek_v41:banked/gemv_group` (26411 B, `a43b1425ae6c`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 2 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots) at `Config.banked_variant_gu` v (1 to 11, needs `Config.banked_group` 4) on hw4b's
  texts (`banked_m3` true), or `moeClampedBankedGateUpV` / `...SelV` with that variant; `Banked.groupGemvs` makes each
  on first use (`banked_gvm3`). hw2's ILP retune; not served by default (variant 0).
- **Shapes and dtypes:** As `turboexl3_gu3bgm3`.
- **Grid and threadgroup:** As `turboexl3_gu3bgm3` (`cfgsBankedGM3`).
- **Template args:** As `turboexl3_gu3bgm3`. The variant is the header: its Metal defines before `header_group`, under a
  kernel name of its own: `ku` EXL3_KU 1 (input tiles in pairs, both tiles' words loaded before either decodes); `pf`
  EXL3_GPF 1 (a rolling prefetch: the next input tile's words loaded before this tile decodes); `xh` EXL3_GXH 1 (each
  member's x quad held as half4 and widened at its FMA); `m2` EXL3_GM2 1 (the four-member path as two passes of two
  members, the second reading and decoding the tile again); `sh` EXL3_GSH 1 (the one-64-bit-window reader with each tile
  word loaded by one lane and taken by the others through `simd_shuffle`); `t4`, `t1`, `t0` EXL3_T1V 4, 1, 0 (the route
  from a MUL1 codeword to t, default 3: 4 both pair sums as one half2, 1 a shift and add for the 32-bit multiply, 0 the
  multiply by 0x10001 and an integer convert; MUL1 only); `pfxh`, `pfsh`, `xhm2` two of those defines together.
- **Replaces:** `turboexl3_gu3bgm3` in the same launch.
- **Bits:** byte-identical to `turboexl3_gu3bgm3` (variant 0): each variant reads, decodes and sums every weight as
  variant 0 does (the same codewords, decoded values, FMA sequence into the same f32 accumulators, and folds); they
  differ only in when the tile words load, how many lanes load them, the registers the loop holds, the passes over a
  tile and the instructions from a codeword to t. The GPU test holds each to g4's variant 0, so to `turboexl3_gu3bg`.
- **Pinned by:** `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 banked ilp: variant 0 is today's (no defines,
  the stock kernel names), ..." (CPU: each name distinct, its defines, each define's default in `header_group`, the
  config's count of 12); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu banked ilp: every grouped GEMV
  variant's gate|up outputs and rows, ..." (GPU: every variant, both GEMVs at it, on g4 and g4m3, against g4 at variant
  0: hd, hs and y word for word, a skipped row-slot's y row 0, rows 1, 2, 3, 4, 6, 7, 12, 16, 24, 36 and 48, x bf16 and
  f32, trellis I16 and U16, at DeepSeek-V4.1's routed geometry (hidden 5120, width 2304, limit 10, MUL1) over synthetic
  banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32)); `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu
  banked ilp t routes: every EXL3_T1V route gives today's route's t, ..." (GPU: the t routes of `t4`, `t1`, `t0` on
  every codeword state, through `turboexl3_t1v_probe_<route>_<dtype>`); `src/core/config.zig` "turboexl3 config: the
  defaults are valid and served; an unserved form, slice count ..." (CPU: a variant needs `banked_group` 4).

### `turboexl3_prep2b`

<!-- kernel-facts: turboexl3_prep2b -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:250` (`const Banked`) |
| inputs | `x`, `slots`, `gate_suh0`, `gate_suh1`, `gate_suh2`, `up_suh0`, `up_suh1`, `up_suh2` |
| outputs | `xg`, `xu`, `sg`, `su` |
| source | `text_prep2b` = `decode/prep2_1` ++ `deepseek_v41:banked/prep2b_expert` ++ `decode/prep2_row` ++ `decode/prep2_2` (2352 B, `affd7b64e88f`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 1 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots), the start of its gate|up half (`moeClampedBankedGateUp`,
  `banked.Banked.bankedGateUp`). Served by default: `Config.banked_group` 2 runs the banked texts.
- **Shapes and dtypes:** x T [rows, hidden] (T bf16, f16 or f32; TOPK 1, so row-slot r reads row r); slots the ids u32
  or i32 [rows], each packed (bank << 24 | slot row); gate_suh0/1/2 and up_suh0/1/2 f16 [E_b, hidden] (bank b's suh, E_b
  its slot count). Out: xg, xu f16 [rows, hidden] (the rotated inputs, lane order); sg, su f32 [rows, hidden/128] (each
  128-block's sum of the rounded inputs, for the deferred MUL1 map).
- **Grid and threadgroup:** `((hidden/128)·32, rows, 1)` / `(32, 1, 1)`.
- **Template args:** DIM (hidden), TOPK (1). Every rate, codebook and window: the prepare reads no weight.
- **Replaces:** `turboexl3_prep2`, launched once per bank by each bank's own `moeClamped` call at top-1: one launch over
  the three banks. The banked chain is form 7's five launches (`prep2b` -> `gu3b` -> `fin7cb` -> `down3b` -> `dfin3b`),
  five dependent levels, in form 9's launch shape.
- **Bits:** byte-identical to `moeClamped` on the pair's bank at top-1 with score 1: `prep2b_expert` (the slot row and
  the bank's gate and up suh) replaces only the line that reads a bank's arrays, so every other statement is `prep2`'s.
  The streamer's sum, each full token's six pairs weighed, is held within 5e-3 of the clamped host reference's rms
  (`ref.compare` against `ref.moeRowClamped`).
- **Pinned by:** `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode texts: each is its base text with
  the bank's lines in place ..." (CPU: the text is its base text with `prep2b_expert` in place of the base line and
  nothing else, every input read, no shape buffer); `src/models/deepseek_v41/banked_test.zig` "turboexl3 banked decode
  refusals: the clamp, the ids' shape and dtype, the banks' dtypes, ..." (CPU: the call's refusals, before a launch);
  `src/models/deepseek_v41/banked_test.zig` "turboexl3 gpu banked decode: every pair of moeClampedBanked over three
  banks is moeClamped's ..." (GPU: every pair byte for byte against `moeClamped` at rows 1, 2, 6, 7, 16, 36 and 48, x
  bf16 and f32, trellis U16 and I16, at DeepSeek-V4.1's routed geometry (hidden 5120, width 2304, limit 10, MUL1) over
  synthetic banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32); sampled token sums to the reference by 5e-3).

### `turboexl3_prep2bg`

<!-- kernel-facts: turboexl3_prep2bg -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:285` (`const Banked`) |
| inputs | `x`, `slots`, `gate_suh0`, `gate_suh1`, `gate_suh2`, `up_suh0`, `up_suh1`, `up_suh2` |
| outputs | `xg`, `xu`, `sg`, `su` |
| source | `text_prep2bg` = `decode/prep2_1` ++ `deepseek_v41:banked/skip_rk` ++ `deepseek_v41:banked/prep2b_expert` ++ `decode/prep2_row` ++ `decode/prep2_2` (2469 B, `110fd5d50005`) |
| header | `header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |
| made at | `src/models/deepseek_v41/banked.zig:319` (`const Banked`) (the same text) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1: launch 1 of `root.moeClampedBanked` (DeepSeek-V4.1's streamed routed experts: each
  row-slot one routed pair at top-1 over an expert streamer's three slot banks, base, ext and transient; 1 to
  `banked_rows_max` (64) row-slots) on the grouped texts, both today's (g4) and hw4b's (g4m3: made at both sites). Not
  served by default: `Config.banked_group` 4 selects the grouped texts, or a call through `moeClampedBankedSel*` with
  `BankedSel` g4 or g4m3 (hw4b's check mode, the tests).
- **Shapes and dtypes:** As `turboexl3_prep2b`; `skip_id` 0xFFFFFFFF (`root.banked_skip_id`) marks a row-slot that runs
  no expert, and its xg, xu, sg and su rows are left unwritten.
- **Grid and threadgroup:** As `turboexl3_prep2b`: `((hidden/128)·32, rows, 1)` / `(32, 1, 1)` (`cfgsBankedG`,
  `cfgsBankedGM3`).
- **Template args:** DIM (hidden), TOPK (1).
- **Replaces:** `turboexl3_prep2b` in the same launch: its text with `skip_rk` (return on a skipped row-slot) before the
  bank lines.
- **Bits:** byte-identical to `turboexl3_prep2b` for every row-slot not skipped.
- **Pinned by:** `src/models/deepseek_v41/banked_test.zig` "turboexl3 grouped banked decode texts: prep2bg, fin7cbg and
  dfin3bg are the banked texts ..." (CPU: the text is the banked text with the skip line before its bank lines);
  `src/models/deepseek_v41/banked_test.zig` "turboexl3 gpu banked group: the grouped banked call's rows are the banked
  call's ..." (GPU: the grouped call's rows equal the banked call's byte for byte, a skipped row-slot's y row 0, rows 6,
  12, 36 and 48, x bf16, trellis I16, at DeepSeek-V4.1's routed geometry (hidden 5120, width 2304, limit 10, MUL1) over
  synthetic banks of 20, 12 and 8 experts, K3 and K2 (n 48, 32)). It runs in every arm of
  `src/models/deepseek_v41/banked_grp_test.zig` "turboexl3 gpu banked grp: the M = 3 texts' gate|up outputs and rows are
  today's, ..." and `src/models/deepseek_v41/banked_ilp_test.zig` "turboexl3 gpu banked ilp: every grouped GEMV
  variant's gate|up outputs and rows, ..." (GPU), which hold the whole call's hd, hs and y.

### `turboexl3_fin7b`

<!-- kernel-facts: turboexl3_fin7b -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:258` (`const Banked`) |
| inputs | `zg`, `zu`, `slots`, `gate_svh0`, `gate_svh1`, `gate_svh2`, `up_svh0`, `up_svh1`, `up_svh2`, `down_suh0`, `down_suh1`, `down_suh2`, `xsg`, `xsu` |
| outputs | `hd`, `hs` |
| source | `text_fin7b` = `decode/fin7_1` ++ `deepseek_v41:banked/fin7b_expert` ++ `decode/fin7_2` ++ `decode/fin7_planes` ++ `decode/fin7_3` ++ `decode/fin7_sums` ++ `decode/fin7_4` ++ `decode/fin7_swiglu` ++ `decode/fin7_5` (2606 B, `972e3be9c60d`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: launch 3 of `root.moeLingBanked` at limit 0 (its layers 4 to 64, whose routed SwiGLU is
  not clamped; `Config.banked_group` 2), the end of its gate|up half; it prepares the down's input. Any banked call
  whose geometry has `limit` 0.
- **Shapes and dtypes:** as `turboexl3_fin7cb`.
- **Grid and threadgroup:** as `turboexl3_fin7cb`.
- **Template args:** INTER, RKC, KS, DEFER, NBH (fin7's five; no LIMIT).
- **Replaces:** `turboexl3_fin7`, launched once per bank by each bank's own `moe` call: one launch over the three banks.
- **Bits:** byte-identical to `moeLing` (so `moe`) at limit 0 on the pair's bank at top-1 with score 1:
  `turboexl3_fin7cb` with fin7's own SwiGLU line in place of the clamp, every other statement the same.
- **Pinned by:** `src/models/ling/ling_test.zig` "turboexl3 ling texts: each Ling kernel is its base text with the
  SwiGLU line swapped for Ling's clamp ..." (CPU: this text is its base with the one swap, nothing else; only a clamp
  reads LIMIT); `src/models/ling/ling_test.zig` "turboexl3 gpu ling banked: every pair of moeLingBanked over three banks
  is moeLing's ..." (GPU: every pair byte for byte against `moeLing` on its bank at top-1, rows 1, 2, 8, 16, 48, limits
  0 and 10, `banked_group` 2 and 4, trellis I16 and U16, Ling's routed geometry over synthetic banks of 20, 12 and 8
  experts at n 40 and 32).

### `turboexl3_fin7bg`

<!-- kernel-facts: turboexl3_fin7bg -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:293` (`const Banked`) |
| inputs | `zg`, `zu`, `slots`, `gate_svh0`, `gate_svh1`, `gate_svh2`, `up_svh0`, `up_svh1`, `up_svh2`, `down_suh0`, `down_suh1`, `down_suh2`, `xsg`, `xsu` |
| outputs | `hd`, `hs` |
| source | `text_fin7bg` = `decode/fin7_1` ++ `deepseek_v41:banked/skip_rk` ++ `deepseek_v41:banked/fin7b_expert` ++ `decode/fin7_2` ++ `decode/fin7_planes` ++ `decode/fin7_3` ++ `decode/fin7_sums` ++ `decode/fin7_4` ++ `decode/fin7_swiglu` ++ `decode/fin7_5` (2723 B, `3f74ff0e778e`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |
| made at | `src/models/deepseek_v41/banked.zig:327` (`const Banked`) (the same text) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: as `turboexl3_fin7b`, under `Config.banked_group` 4 (the grouped texts; with `banked_m3`
  too).
- **Shapes and dtypes:** as `turboexl3_fin7cbg`.
- **Grid and threadgroup:** as `turboexl3_fin7cbg`.
- **Template args:** INTER, RKC, KS, DEFER, NBH (no LIMIT).
- **Replaces:** as `turboexl3_fin7b`.
- **Bits:** byte-identical to `turboexl3_fin7b` on every row-slot `fin7cbg` serves (a skipped row-slot's outputs
  untouched): `turboexl3_fin7cbg` with fin7's own SwiGLU line in place of the clamp.
- **Pinned by:** `src/models/ling/ling_test.zig` "turboexl3 ling texts: each Ling kernel is its base text with the
  SwiGLU line swapped for Ling's clamp ..." (CPU: this text is its base with the one swap, nothing else; only a clamp
  reads LIMIT); `src/models/ling/ling_test.zig` "turboexl3 gpu ling banked: every pair of moeLingBanked over three banks
  is moeLing's ..." (GPU: every pair byte for byte against `moeLing` on its bank at top-1, rows 1, 2, 8, 16, 48, limits
  0 and 10, `banked_group` 2 and 4, trellis I16 and U16, Ling's routed geometry over synthetic banks of 20, 12 and 8
  experts at n 40 and 32).

### `turboexl3_fin7cob`

<!-- kernel-facts: turboexl3_fin7cob -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:260` (`const Banked`) |
| inputs | `zg`, `zu`, `slots`, `gate_svh0`, `gate_svh1`, `gate_svh2`, `up_svh0`, `up_svh1`, `up_svh2`, `down_suh0`, `down_suh1`, `down_suh2`, `xsg`, `xsu` |
| outputs | `hd`, `hs` |
| source | `text_fin7cob` = `decode/fin7_1` ++ `deepseek_v41:banked/fin7b_expert` ++ `decode/fin7_2` ++ `decode/fin7_planes` ++ `decode/fin7_3` ++ `decode/fin7_sums` ++ `decode/fin7_4` ++ `decode/fin7_swiglu_clamp_out` ++ `decode/fin7_5` (3015 B, `8b48ab86e9ab`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: launch 3 of `root.moeLingBanked` with a limit (its layers 65 to 71,
  `expert_swiglu_limit_list` 10; `Config.banked_group` 2).
- **Shapes and dtypes:** as `turboexl3_fin7cb`.
- **Grid and threadgroup:** as `turboexl3_fin7cb`.
- **Template args:** fin7's five, then LIMIT (an integer 1 to 128, Ling-3.1's 10): the gate's SiLU clamped from above at
  LIMIT and up to [−LIMIT, LIMIT], in f32 on the finished f16 values; a NaN passes through.
- **Replaces:** `turboexl3_fin7co`, launched once per bank by each bank's own `moeLing` call: one launch over the three
  banks.
- **Bits:** byte-identical to `moeLing` on the pair's bank at top-1 with score 1: `turboexl3_fin7cb` with Ling's clamp
  piece in place of DeepSeek's, every other statement the same.
- **Pinned by:** `src/models/ling/ling_test.zig` "turboexl3 ling texts: each Ling kernel is its base text with the
  SwiGLU line swapped for Ling's clamp ..." (CPU: this text is its base with the one swap, nothing else; only a clamp
  reads LIMIT); `src/models/ling/ling_test.zig` "turboexl3 gpu ling banked: every pair of moeLingBanked over three banks
  is moeLing's ..." (GPU: every pair byte for byte against `moeLing` on its bank at top-1, rows 1, 2, 8, 16, 48, limits
  0 and 10, `banked_group` 2 and 4, trellis I16 and U16, Ling's routed geometry over synthetic banks of 20, 12 and 8
  experts at n 40 and 32).

### `turboexl3_fin7cobg`

<!-- kernel-facts: turboexl3_fin7cobg -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/banked.zig:295` (`const Banked`) |
| inputs | `zg`, `zu`, `slots`, `gate_svh0`, `gate_svh1`, `gate_svh2`, `up_svh0`, `up_svh1`, `up_svh2`, `down_suh0`, `down_suh1`, `down_suh2`, `xsg`, `xsu` |
| outputs | `hd`, `hs` |
| source | `text_fin7cobg` = `decode/fin7_1` ++ `deepseek_v41:banked/skip_rk` ++ `deepseek_v41:banked/fin7b_expert` ++ `decode/fin7_2` ++ `decode/fin7_planes` ++ `decode/fin7_3` ++ `decode/fin7_sums` ++ `decode/fin7_4` ++ `decode/fin7_swiglu_clamp_out` ++ `decode/fin7_5` (3132 B, `02a993e5f353`) |
| header | `header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |
| made at | `src/models/deepseek_v41/banked.zig:329` (`const Banked`) (the same text) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: as `turboexl3_fin7cob`, under `Config.banked_group` 4.
- **Shapes and dtypes:** as `turboexl3_fin7cbg`.
- **Grid and threadgroup:** as `turboexl3_fin7cbg`.
- **Template args:** fin7's five, then LIMIT (an integer 1 to 128, Ling-3.1's 10): the gate's SiLU clamped from above at
  LIMIT and up to [−LIMIT, LIMIT], in f32 on the finished f16 values; a NaN passes through.
- **Replaces:** as `turboexl3_fin7cob`.
- **Bits:** byte-identical to `turboexl3_fin7cob` on every row-slot it serves: `turboexl3_fin7cbg` with Ling's clamp
  piece in place of DeepSeek's.
- **Pinned by:** `src/models/ling/ling_test.zig` "turboexl3 ling texts: each Ling kernel is its base text with the
  SwiGLU line swapped for Ling's clamp ..." (CPU: this text is its base with the one swap, nothing else; only a clamp
  reads LIMIT); `src/models/ling/ling_test.zig` "turboexl3 gpu ling banked: every pair of moeLingBanked over three banks
  is moeLing's ..." (GPU: every pair byte for byte against `moeLing` on its bank at top-1, rows 1, 2, 8, 16, 48, limits
  0 and 10, `banked_group` 2 and 4, trellis I16 and U16, Ling's routed geometry over synthetic banks of 20, 12 and 8
  experts at n 40 and 32).

### `turboexl3_dense_swiglu_prep`

<!-- kernel-facts: turboexl3_dense_swiglu_prep -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/f3.zig:43` (`const F3`) |
| inputs | `z1`, `xs1`, `svh1`, `z3`, `xs3`, `svh3`, `suh2`, `lim` |
| outputs | `xp`, `xs` |
| source | `text_swiglu_prep` = `deepseek_v41:f3/swiglu_prep` (4775 B, `111d0a28a123`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** DeepSeek-V4.1 (F3): `root.sharedSwiglu` (where `root.sharedSwigluTakes` holds), its shared expert at the
  decode rows the dense chain takes, 5 to 16 (DEC-ONE keeps 1 to 4, so F3 refuses them), one rate and decode for gate,
  up and down, limit > 0 (`swiglu_limit` 10). Made on first use on the gate's `Dense`. Ripple's clamped SwiGLU has the
  same shape.
- **Shapes and dtypes:** z1, z3 f32 [KS · rows, DIM] (the gate's and up's planes, `Dense.prepareGemv`; DIM = their OUT,
  the inter width); xs1, xs3 f32 [rows, IN/128]; svh1, svh3 f16 [DIM]; suh2 f16 [DIM] (the down's suh); lim f32 [1].
  Out: xp f16 [rows, DIM] in lane order; xs f32 [rows, DIM/128] (the down's prepared rows and block sums).
- **Grid and threadgroup:** `((DIM/128) · 32, rows, 1)` / `(32, 1, 1)`: one simdgroup per (128-block, row).
- **Template args:** DIM, KS1 and KS3 (both the gate shape's `ks_split`), ROWS, INBLK (IN/128), DEFER, OUT_T (the gate's
  and up's output dtype, x's), ACT_T (the activation dtype `act`: bf16, f16 or f32; the configs are rebuilt when it
  changes). Every rate, codebook and window (it reads no weight).
- **Replaces:** the gate's and up's `turboexl3_dense_dfinish`, the host's clamped SwiGLU in MLX (astype f32, minimum,
  maximum, minimum, sigmoid, multiplies, astype to `act`) and the down's `turboexl3_dense_prep_s`: three launches fewer
  (`root.sharedSwiglu`'s comment) plus the MLX ops, and the finish -> SwiGLU -> prepare levels become one.
- **Bits:** byte-identical to the host's chain (gate and up `linear`, astype f32, MLX's clamped SwiGLU, astype to `act`,
  down `linear`).
- **Pinned by:** `src/models/deepseek_v41/f3_test.zig` "turboexl3 f3u: the fused shared-expert text is text_dfinish's
  finishes, MLX's clamped SwiGLU and text_prep_s's prepare" (CPU, its lines); `src/models/deepseek_v41/f3_test.zig`
  "turboexl3 gpu f3u bytes: sharedSwiglu gives the host's shared expert's words at rows 1 to 16 (TURBOEXL3_F3=1)" (GPU,
  bytes at rows 1 to 16 in bf16 and 6 and 12 in f32, `sharedSwigluTakes` true exactly from 5 rows);
  `src/models/deepseek_v41/f3_test.zig` "turboexl3 gpu f3u time: the 40-layer trunk chain with the host's shared expert
  against F3 at rows 5, ..." (GPU, every run's 40-layer output the first run's words).

## `src/models/deepseek_v41/f3_test.zig`

Kernels F3's tests make as references (test only).

### `turboexl3_test_swiglu_one`

<!-- kernel-facts: turboexl3_test_swiglu_one -->

| fact | from the code |
|---|---|
| made at | `src/models/deepseek_v41/f3_test.zig:222` (`test turboexl3 gpu f3u time: the 40-layer trunk chain with the host's shared expert against F3 at rows 5, 6 and 8, every run the first's words (TURBOEXL3_F3_TIME=1)`) |
| inputs | `gt`, `ut`, `lim` |
| outputs | `y` |
| source | `text_swiglu_one` = `deepseek_v41:test/swiglu_one` (551 B, `9be0c03bd41d`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->

- **Used by:** test only: arm A of `src/models/deepseek_v41/f3_test.zig` "turboexl3 gpu f3u time: the 40-layer trunk
  chain with the host's shared expert against F3 at rows 5, ...": the host's clamped SwiGLU as MLX's compiled decode
  stretch runs it, one elementwise launch between the shared gate and up linears and the down linear. Its text is
  `src/models/deepseek_v41/kernels/test/swiglu_one.metal`.
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
