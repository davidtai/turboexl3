# Ling-3.1-flash kernels

The kernels made in `src/models/ling/`. Ling's routed experts run the core's and DeepSeek-V4.1's kernels with one
SwiGLU line swapped ([the core's catalog](../../core/kernels.md), [DeepSeek-V4.1's](../deepseek_v41/kernels.md)); the
kernels here are its shared expert's at decode rows and its same-input trunk pairs'. Each is a DEC-ONE text over two
members of one input, each member in its own buffers, with the member chosen after its head and the GEMV's tile address
read in the member's trellis:
- the shared expert's (`shared.zig`): DEC-ONE with the prepare inside (`turboexl3_dense_dec1`'s text) over the gate and
  the up; its finish stores the member's output, and the second finish of each pair of 128-blocks (the gate's and the
  up's) runs the host's SwiGLU and the down linear's prepare;
- the pairs' (`pair.zig`): dense's `prep_sc` with each member, on the grid's z, scaling x by its own suh into its own rows
  of the pair's prepared rows, and DEC-ONE's two texts (`turboexl3_dense_dec0` over those rows, `turboexl3_dense_dec1`
  with the prepare inside) with the member's prepared rows or suh, its trellis and svh, and Ripple's member tile address
  (`kernels/ricursive/jtile.metal`).
How an entry reads: [the core catalog's introduction](../../core/kernels.md#how-an-entry-reads).

## `src/models/ling/shared.zig`

The shared expert at decode rows: the gate and the up in one launch, their pairs' SwiGLU and the down's prepare, then
the down's DEC-ONE GEMV (`turboexl3_dense_dec0`).

### `turboexl3_ling_sgu`

<!-- kernel-facts: turboexl3_ling_sgu -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/shared.zig:71` (`fn kernelOf`) |
| inputs | `x`, `suh0`, `suh1`, `w`, `w1`, `svh0`, `svh1`, `suh2`, `sigtab`, `cnt`, `pcnt` |
| outputs | `y`, `z`, `xsx`, `xp`, `xs` |
| source | `text_sgu` = `dense/dec_head` ++ `ling:shared/msel` ++ `dense/dec_prep1` ++ `dense/dec_gemv1` ++ `dense/dec_gemv_arrive_1` ++ `ling:shared/mtile` ++ `dense/dec_gemv_arrive_2` ++ `dense/dec_xsum1` ++ `dense/dec_finish_v` ++ `ling:shared/sgu_store` ++ `ling:shared/sgu_pair_1` ++ `ling:shared/sgu_swiglu` ++ `ling:shared/sgu_pair_2` (11827 B, `9233e19e7283`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `ling.sharedExpert` (`root.lingSharedExpert`) at limit 0 (the shared expert's
  layers without a clamp), decode rows 1 to 4 where `ling.shared.planFor` admits the call (the down's DEC-ONE takes its
  prepare in a launch of its own, the joint's prepare fits threadgroup memory: Ling's 4096 -> 1280 -> 4096 at every
  row), then `turboexl3_dense_dec0` for the down. The outside plugin turboexl3_ling's MoE block calls it under
  MLX_SERVE_LING_SHARED_F3=1.
- **Shapes and dtypes:** x T [rows, IN] (bf16 or f16); suh0, suh1 f16 [IN] (the gate's, the up's); w, w1 the gate's and
  the up's trellises, U16 or I16 alike, [IN/16, F/16, NHW] each; svh0, svh1 f16 [F]; suh2 f16 [F] (the down's);
  sigtab T [65536] (MLX's sigmoid of every 16-bit pattern of T, the host's table); cnt u32 [max(2F/128 · chunks, 8)],
  pcnt u32 [max(F/128 · chunks, 8)], each from `turboexl3_dense_dec_zero`. Out: y T [rows, 2F] (the gate's columns,
  then the up's: each member's `linear` output, which the pairs read back); z f32 [KS · rows, 2F] and xsx f32 (DEC-ONE's
  scratch); xp f16 [rows, F] (the down's prepared rows, lane order) and xs f32 [rows, F/128] (their block sums), what
  the down's `turboexl3_dense_dec0` reads.
- **Grid and threadgroup:** `((2F/16/BT) · 128, chunks, KS)` / `(128, 1, 1)`, DEC-ONE's grid over the joint's 2F
  outputs; three `turboexl3_dense_dec_zero` launches before it (its arrivals, its pairs, the down's arrivals) at
  `(ceil(NCNT/128) · 128, 1, 1)` / `(128, 1, 1)`, each waiting on nothing.
- **Template args:** NHW (the members' one rate: K6 in the packs, n 96), CODEBOOK, WBITS, IN, OUT (2F), KS (each
  member's own `Dense.ksFor(IN, F)`), BT (`Dense.btFor`), RB (`Dense.rbFor`), ROWS, DEFER (the MUL1 map), OUT_T (T), OB1
  (F/128: the member boundary, and the down's input blocks). Every rate, codebook and window DEC-ONE serves; the members
  share IN, F, the rate and the trellis dtype.
- **Replaces:** the composed shared expert's gate, up, SwiGLU and the down's prepare at those rows: two DEC-ONE calls
  (`dec_zero` + `dec1` each at one row, `prep_sc` + `dec0` each from two at F 1280), mlx-serve's fusedSwiGLU and the
  down's `prep_sc`: 6 launches in 3 dependent levels, with 3 `dec_zero` + this launch: 1 dependent level (the down's
  `dec0` after it: 2 levels for the whole shared expert where the composed one takes 4).
- **Bits:** byte-identical to the composed chain: each member's columns are its `linear`'s (DEC-ONE's text after the
  member is chosen: the same prepare, K split, slice order and finish per output block, each tile's words read from the
  member's trellis, the member's svh); the SwiGLU is mlx-serve's fusedSwiGLU's statements (the table's sigmoid by the
  gate's pattern, MLX's Multiply twice in T); the prepare is `text_prep_s`'s statements (lane order) on the activation,
  so the down's `dec0` gives `linear(down)`'s words.
- **Pinned by:** `src/models/ling/shared_test.zig` "turboexl3 gpu ling shared: the gate and the up in one launch with the
  host's SwiGLU and the down's prepare in their pairs' second finishes, then the down, give the composed shared expert's
  words at rows 1 to 4, limits 0 and 10 (TURBOEXL3_GPU=1)" (GPU, words of the shared expert against the composed chain
  with MLX's ops, and of the joint's columns against each member's `linear`, at rows 1 to 4, bf16 and f16);
  `src/models/ling/shared_test.zig` "turboexl3 ling shared texts: dense's DEC-ONE (prepare inside) around the member
  choice, its tile address and the member's store, then the pair's arrival, the host's SwiGLU and text_prep_s's
  statements" (CPU, the texts).

### `turboexl3_ling_sguco`

<!-- kernel-facts: turboexl3_ling_sguco -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/shared.zig:72` (`fn kernelOf`) |
| inputs | `x`, `suh0`, `suh1`, `w`, `w1`, `svh0`, `svh1`, `suh2`, `sig32`, `lim`, `cnt`, `pcnt` |
| outputs | `y`, `z`, `xsx`, `xp`, `xs` |
| source | `text_sguco` = `dense/dec_head` ++ `ling:shared/msel` ++ `dense/dec_prep1` ++ `dense/dec_gemv1` ++ `dense/dec_gemv_arrive_1` ++ `ling:shared/mtile` ++ `dense/dec_gemv_arrive_2` ++ `dense/dec_xsum1` ++ `dense/dec_finish_v` ++ `ling:shared/sgu_store` ++ `ling:shared/sgu_pair_1` ++ `ling:shared/sgu_swiglu_clamp_out` ++ `ling:shared/sgu_pair_2` (12373 B, `22dd0e8c6666`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `ling.sharedExpert` at a limit L in 1..128 (Ling-3.1's
  `share_expert_swiglu_limit_list`: 10 on layers 64 to 71), at the rows `turboexl3_ling_sgu` takes; MLX_SERVE_LING_SHARED_F3=1
  in turboexl3_ling.
- **Shapes and dtypes:** `turboexl3_ling_sgu`'s, with sig32 f32 [65536] (MLX's f32 sigmoid of every 16-bit pattern of T
  taken to f32, `ling.sigmoidTable(s, T, true)`) in sigtab's place and lim f32 [1] (L).
- **Grid and threadgroup:** `turboexl3_ling_sgu`'s.
- **Template args:** `turboexl3_ling_sgu`'s.
- **Replaces:** `turboexl3_ling_sgu`'s chain with the plugin's swigluClampOut (astype, sigmoid, multiply, minimum, clip,
  multiply, astype: 9 launches) in fusedSwiGLU's place: 14 launches in 8 dependent levels, with 3 `dec_zero` + this
  launch: 1 level (2 with the down's `dec0`, where the composed shared expert takes 9).
- **Bits:** byte-identical to the composed chain with the plugin's swigluClampOut: both values in f32, MLX's Sigmoid
  read from the table by the gate's pattern (the gate is a 16-bit value, so the table holds every input MLX's f32
  sigmoid sees), Multiply, Minimum with L and clip as MLX's functors (NaN first, Maximum then Minimum), Multiply, and the
  activation dtype's rounding; the members and the prepare as `turboexl3_ling_sgu`'s.
- **Pinned by:** `src/models/ling/shared_test.zig` "turboexl3 gpu ling shared: ..." (GPU, words at limit 10 with the
  clamp reached: the SiLU above 10 and the up beyond it either way); `src/models/ling/shared_test.zig` "turboexl3 ling
  shared texts: ..." (CPU, the texts).

## `src/models/ling/pair.zig`

A KDA layer's z | a, the shared expert's gate | up and the dense FFN's gate | up in one GEMV launch at decode rows.

### `turboexl3_ling_pair_dec0`

<!-- kernel-facts: turboexl3_ling_pair_dec0 -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/pair.zig:66` (`fn kernelOf`) |
| inputs | `jxp`, `jxs`, `w0`, `w1`, `svh0`, `svh1`, `cnt` |
| outputs | `y`, `z` |
| source | `text_jdec0` = `dense/dec_head` ++ `ling:pair/jrows` ++ `ling:pair/jw` ++ `dense/dec_gemv0` ++ `dense/dec_gemv_arrive_1` ++ `ricursive:ricursive/jtile` ++ `dense/dec_gemv_arrive_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` (7441 B, `e67032a622fb`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.linearPair` at the decode rows (1 to 16) where the members' own `linear` calls
  take the prepare in a launch of its own (`Dense.decPrepInside` false: past one row, or at one row under
  `Config.dec_one_row_wide`, at more than `dec_prep_out` (1024) outputs), after `turboexl3_ling_pair_prep`.
- **Shapes and dtypes:** jxp f16 [2 · rows, IN] and jxs f32 [2 · rows, IN/128] (member m's prepared rows and their
  128-block sums at rows m · rows.., from `turboexl3_ling_pair_prep`); w0, w1 each member's own trellis U16 or I16
  [IN/16, out_m/16, NHW] (one dtype); svh0, svh1 f16 [out_m]; cnt u32 [max(OUT/128 · chunks, 8)] (zeroed by the
  prepare). Out: y T [rows, out0 + out1] (member 0's columns, then member 1's); z f32 [KS · rows, OUT] (scratch, as
  DEC-ONE's).
- **Grid and threadgroup:** `((OUT/16/BT) · 128, chunks, KS)` / `(128, 1, 1)`, DEC-ONE's grid over the pair's OUT.
- **Template args:** NHW, CODEBOOK, WBITS (the members' one rate and decode), IN, OUT (out0 + out1), KS (each member's
  own `Dense.ksFor(in, out)`: the members must share it), BT (the members' `Dense.btDec`: `bt_set`, else `btFor`), RB
  (`Dense.rbFor`), ROWS, DEFER (the MUL1 map), OUT_T, OB1 (member 0's 128-blocks: the member boundary).
- **Replaces:** each member's DEC-ONE GEMV (`turboexl3_dense_dec0`), after one prepare launch for both members in place
  of two `turboexl3_dense_prep_sc`: 4 launches in 2 dependent levels with 2.
- **Bits:** byte-identical to each member's `linear` at every row count it serves (DEC-ONE's text after the member is
  chosen: its own prepared rows, K split, slice order and finish per output block, each tile's words read from its own
  trellis, its svh read at its own column; DEC-ONE gives the three-launch chain's bytes at rows 5 to 16).
- **Pinned by:** `src/models/ling/ling_test.zig` "turboexl3 gpu ling pair: z | a and the shared and dense gate | up in one launch give
  each member's linear bytes, the prepare inside or in its own launch, at decode rows 1 to 16 (TURBOEXL3_GPU=1)" (GPU,
  words at rows 1 to 6, 8 and 16, bf16 and f32, one row on both launch shapes); `src/models/ling/pair.zig` "turboexl3 ling pair texts: dense's prep_sc and DEC-ONE texts verbatim around the member
  choice, ..." (CPU, the texts).

### `turboexl3_ling_pair_dec1`

<!-- kernel-facts: turboexl3_ling_pair_dec1 -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/pair.zig:67` (`fn kernelOf`) |
| inputs | `x`, `suh0`, `suh1`, `w0`, `w1`, `svh0`, `svh1`, `cnt` |
| outputs | `y`, `z`, `xsx` |
| source | `text_jdec1` = `dense/dec_head` ++ `ling:pair/jsuh` ++ `ling:pair/jw` ++ `dense/dec_prep1` ++ `dense/dec_gemv1` ++ `dense/dec_gemv_arrive_1` ++ `ricursive:ricursive/jtile` ++ `dense/dec_gemv_arrive_2` ++ `dense/dec_xsum1` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` (8776 B, `6f8fa1171cb0`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.linearPair` at the decode rows where the members' own `linear` calls take
  the prepare inside the launch (`Dense.decPrepInside` true: one row, as served by default), after dense's counter
  zeroing (`turboexl3_dense_dec_zero`).
- **Shapes and dtypes:** x T [rows, IN] (bf16, f16 or f32); suh0, suh1 f16 [IN] (each member's own); w0, w1 each
  member's own trellis U16 or I16 [IN/16, out_m/16, NHW]; svh0, svh1 f16 [out_m]; cnt u32 [max(OUT/128 · chunks, 8)]
  from `turboexl3_dense_dec_zero`. Out: y T [rows, out0 + out1]; z f32 [KS · rows, OUT] and xsx f32 (scratch, as
  DEC-ONE's).
- **Grid and threadgroup:** `((OUT/16/BT) · 128, chunks, KS)` / `(128, 1, 1)`, DEC-ONE's grid over the pair's OUT;
  `turboexl3_dense_dec_zero` before it at `(ceil(NCNT/128) · 128, 1, 1)` / `(128, 1, 1)`.
- **Template args:** as `turboexl3_ling_pair_dec0`'s.
- **Replaces:** each member's DEC-ONE call with the prepare inside (`turboexl3_dense_dec_zero` + `turboexl3_dense_dec1`):
  4 launches in 1 dependent level with 2.
- **Bits:** byte-identical to each member's `linear` (DEC-ONE's text with the prepare inside after the member is chosen:
  its suh, K split, slice order and finish per output block, its own trellis and svh).
- **Pinned by:** `src/models/ling/ling_test.zig` "turboexl3 gpu ling pair: z | a and the shared and dense gate | up in one launch give
  each member's linear bytes, the prepare inside or in its own launch, at decode rows 1 to 16 (TURBOEXL3_GPU=1)" (GPU,
  words at rows 1 to 6, 8 and 16, bf16 and f32, one row on both launch shapes); `src/models/ling/pair.zig` "turboexl3 ling pair texts: dense's prep_sc and DEC-ONE texts verbatim around the member
  choice, ..." (CPU, the texts).

### `turboexl3_ling_pair_prep`

<!-- kernel-facts: turboexl3_ling_pair_prep -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/pair.zig:65` (`fn kernelOf`) |
| inputs | `x`, `suh0`, `suh1` |
| outputs | `jxp`, `jxs`, `cnt` |
| source | `text_jprep_sc` = `ling:pair/jprep` ++ `dense/prep_s` ++ `dense/prep_sc` (2002 B, `4cf62ba00a38`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.linearPair` where its members' own calls take the prepare in a launch of
  its own (`turboexl3_ling_pair_dec0`'s rows), before `turboexl3_ling_pair_dec0`.
- **Shapes and dtypes:** x T [rows, IN] (bf16, f16 or f32); suh0, suh1 f16 [IN] (each member's own). Out: jxp f16
  [2 · rows, IN] (member m's prepared rows at rows m · rows.., lane order), jxs f32 [2 · rows, IN/128] (their 128-block
  sums), cnt u32 [NCNT] (`turboexl3_ling_pair_dec0`'s arrival counters, zeroed).
- **Grid and threadgroup:** `((IN/128) · 32, rows, 2)` / `(32, 1, 1)`: `turboexl3_dense_prep_sc`'s grid, the members on z.
- **Template args:** DIM (IN), LANE_ORDER 1, NCNT (the GEMV's counters), NTH ((IN/128) · 32 · rows: each member's
  threads zero every counter), ROWS.
- **Replaces:** the members' two `turboexl3_dense_prep_sc` launches: 2 launches, 1 level, with 1.
- **Bits:** byte-identical to each member's `turboexl3_dense_prep_sc` (its text after the member's suh and rows are
  chosen), so `turboexl3_ling_pair_dec0` reads the rows each member's own call reads.
- **Pinned by:** `src/models/ling/ling_test.zig` "turboexl3 gpu ling pair: z | a and the shared and dense gate | up in one launch give
  each member's linear bytes, the prepare inside or in its own launch, at decode rows 1 to 16 (TURBOEXL3_GPU=1)" (GPU,
  words at rows 1 to 6, 8 and 16, bf16 and f32, one row on both launch shapes); `src/models/ling/pair.zig` "turboexl3 ling pair texts: dense's prep_sc and DEC-ONE texts verbatim around the member
  choice, ..." (CPU, the texts).
