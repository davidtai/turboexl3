# Ling-3.1-flash kernels

The kernels made in `src/models/ling/`. Ling's routed experts run the core's and DeepSeek-V4.1's kernels with one
SwiGLU line swapped ([the core's catalog](../../core/kernels.md), [DeepSeek-V4.1's](../deepseek_v41/kernels.md)); the
kernels here are Ling's own: a wide (prompt) routed call's combine (`combine.zig`), its MoE router at decode rows with
the read-ahead predictor's scores (`router.zig`), both below, and its shared expert's at decode rows and its same-input
trunk pairs'. Those two are DEC-ONE texts over two members of one input, each member in
its own buffers, with the member chosen after its head and the GEMV's tile address read in the member's trellis:
- the shared expert's (`shared.zig`): DEC-ONE with the prepare inside (`turboexl3_dense_dec1`'s text) over the gate and
  the up; its finish stores the member's output, and the second finish of each pair of 128-blocks (the gate's and the
  up's) runs the host's SwiGLU and the down linear's prepare;
- the pairs' (`pair.zig`): dense's `prep_sc` with each member, on the grid's z, scaling x by its own suh into its own rows
  of the pair's prepared rows, and DEC-ONE's two texts (`turboexl3_dense_dec0` over those rows, `turboexl3_dense_dec1`
  with the prepare inside) with the member's prepared rows or suh, its trellis and svh, and Ripple's member tile address
  (`kernels/ricursive/jtile.metal`).

A KDA layer's three projections of one input (`kdaproj.zig`) are dense's `prep_sc` with each member on the grid's z, then
dense's DEC-ONE text over prepared rows (`turboexl3_dense_dec0`'s, verbatim after its head) under a head that gives each
threadgroup its member's widths and K split; each member keeps its own buffers.

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
| source | `text_sgu` = `dense/dec_head` ++ `ling:shared/msel` ++ `dense/dec_prep1` ++ `dense/dec_gemv1` ++ `dense/dec_gemv_arrive_1` ++ `ling:shared/mtile` ++ `dense/dec_gemv_arrive_2_1` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum1` ++ `dense/dec_finish_v` ++ `ling:shared/sgu_store` ++ `ling:shared/sgu_pair_1` ++ `ling:shared/sgu_swiglu` ++ `ling:shared/sgu_pair_2` (11827 B, `9233e19e7283`) |
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
| source | `text_sguco` = `dense/dec_head` ++ `ling:shared/msel` ++ `dense/dec_prep1` ++ `dense/dec_gemv1` ++ `dense/dec_gemv_arrive_1` ++ `ling:shared/mtile` ++ `dense/dec_gemv_arrive_2_1` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum1` ++ `dense/dec_finish_v` ++ `ling:shared/sgu_store` ++ `ling:shared/sgu_pair_1` ++ `ling:shared/sgu_swiglu_clamp_out` ++ `ling:shared/sgu_pair_2` (12373 B, `22dd0e8c6666`) |
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
| source | `text_jdec0` = `dense/dec_head` ++ `ling:pair/jrows` ++ `ling:pair/jw` ++ `dense/dec_gemv0` ++ `dense/dec_gemv_arrive_1` ++ `ricursive:ricursive/jtile` ++ `dense/dec_gemv_arrive_2_1` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` (7441 B, `e67032a622fb`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.linearPair` at the decode rows (1 to 16) where the members' own `linear` calls
  take the prepare in a launch of its own (`Dense.decPrepInside` false: past one row, or at one row at more
  than `dec_prep_out` (1024) outputs, or at a shape `Config.dec_one_row_table`'s
  table runs that way), after `turboexl3_ling_pair_prep`.
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
| source | `text_jdec1` = `dense/dec_head` ++ `ling:pair/jsuh` ++ `ling:pair/jw` ++ `dense/dec_prep1` ++ `dense/dec_gemv1` ++ `dense/dec_gemv_arrive_1` ++ `ricursive:ricursive/jtile` ++ `dense/dec_gemv_arrive_2_1` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum1` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` (8776 B, `6f8fa1171cb0`) |
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

## `src/models/ling/kdaproj.zig`

A KDA layer's q|k|v, f and g (three projections of one input) in one GEMV launch at decode rows, each member at its own
K split.

### `turboexl3_ling_kdaproj_dec0`

<!-- kernel-facts: turboexl3_ling_kdaproj_dec0 -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj.zig:67` (`fn kernelOf`) |
| inputs | `jxp`, `jxs`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `jcnt` |
| outputs | `y0`, `y1`, `y2`, `z0`, `z1`, `z2` |
| source | `text_kdec0` = `ling:kdaproj/head` ++ `ling:kdaproj/member` ++ `dense/dec_gemv0` ++ `dense/dec_gemv_arrive_1` ++ `dense/dec_tile` ++ `dense/dec_gemv_arrive_2_1` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` (7971 B, `a9c43d427f96`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.kdaProj` at decode rows 1 to 16, after `turboexl3_ling_kdaproj_prep`; and
  `root.ling.kdaProjPrepared` over a host's own prepare of the same layout (a norm launch that writes its readers'
  prepared rows).
- **Shapes and dtypes:** jxp f16 [3 · rows, IN] and jxs f32 [3 · rows, IN/128] (member m's prepared rows, as its own
  `turboexl3_dense_prep_sc` writes them, and their 128-block sums at rows m · rows..); w0, w1, w2 each member's own
  trellis U16 or I16 [IN/16, out_m/16, NHW] (one dtype); svh0, svh1, svh2 f16 [out_m]; jcnt u32
  [max((OUT0 + OUT1 + OUT2)/128 · chunks, 8)] zeroed (`kdaproj.ncntFor`; member m's counters after the members before
  it). Out: y0, y1, y2 T [rows, out_m] (each member's own); z0, z1, z2 f32 [KS_m · rows, out_m] (scratch, as DEC-ONE's).
- **Grid and threadgroup:** `((OUT0/128 · KS0 + (OUT1 + OUT2)/128 · KS1) · (8/BT) · 128, chunks, 1)` / `(128, 1, 1)`:
  member 0's (K slice, block) threadgroups, then member 1's, then member 2's.
- **Template args:** NHW, CODEBOOK, WBITS (the members' one rate and decode), IN, OUT0, OUT1, OUT2, KS0 (member 0's own
  `Dense.ksFor(in, out)`), KS1 (members 1 and 2's: they must share it), BT (member 0's `Dense.btDec`), RB
  (`Dense.rbFor`), ROWS, DEFER (the MUL1 map), OUT_T.
- **Replaces:** each member's DEC-ONE GEMV (`turboexl3_dense_dec0` or `turboexl3_dense_dec1`, with its own
  `turboexl3_dense_prep_sc` or `turboexl3_dense_dec_zero`): 6 launches in 2 dependent levels with 2.
- **Bits:** byte-identical to each member's `linear` at every row count it serves (DEC-ONE's text over prepared rows
  after its head: the member's own prepared rows, K split, slice order and finish per output block, its own trellis,
  svh and output; the tiles per threadgroup and where the prepare runs move no byte).
- **Pinned by:** `src/models/ling/ling_test.zig` "turboexl3 gpu ling kda proj: q|k|v, f and g in one launch give each
  member's linear bytes, ..." (GPU, words at rows 1 to 8 and 16, bf16 and f32, after its own prepare and over each
  member's own prep_sc rows joined, at one row on three launch shapes); `src/models/ling/kdaproj.zig` "turboexl3 ling
  kdaproj texts: ..." (CPU, the texts), "... plan: ..." and "... grid: ..." (CPU, the K splits and the grid).

### `turboexl3_ling_kdaproj_prep`

<!-- kernel-facts: turboexl3_ling_kdaproj_prep -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj.zig:66` (`fn kernelOf`) |
| inputs | `x`, `suh0`, `suh1`, `suh2` |
| outputs | `jxp`, `jxs`, `cnt` |
| source | `text_kprep_sc` = `ling:kdaproj/prep` ++ `dense/prep_s` ++ `dense/prep_sc` (2057 B, `74bf60b729c3`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** `root.ling.kdaProj`, before `turboexl3_ling_kdaproj_dec0`.
- **Shapes and dtypes:** x T [rows, IN] (bf16, f16 or f32); suh0, suh1, suh2 f16 [IN] (each member's own). Out: jxp f16
  [3 · rows, IN], jxs f32 [3 · rows, IN/128], cnt u32 [NCNT] zeroed.
- **Grid and threadgroup:** `(IN/128 · 32, rows, 3)` / `(32, 1, 1)`: dense's `prep_sc` grid, the members on z.
- **Template args:** DIM (IN), LANE_ORDER 1, NCNT, NTH (IN/128 · 32 · rows, the threads that zero the counters), ROWS.
- **Replaces:** each member's `turboexl3_dense_prep_sc` (or `turboexl3_dense_dec_zero` where its call takes the prepare
  inside).
- **Bits:** each member's prepared rows and block sums are the ones its own `turboexl3_dense_prep_sc` writes (its text
  after the member's suh and rows are chosen).
- **Pinned by:** as `turboexl3_ling_kdaproj_dec0`'s.

### `turboexl3_ling_kdaproj_dec0b`

<!-- kernel-facts: turboexl3_ling_kdaproj_dec0b -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj.zig:68` (`fn kernelOf`) |
| inputs | `jxp`, `jxs`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `jcnt`, `xb`, `wb` |
| outputs | `y0`, `y1`, `y2`, `z0`, `z1`, `z2`, `yb` |
| source | `text_kdec0b` = `ling:kdaproj/bmember_1` ++ `ling:kdaproj/bmember_2` ++ `ling:kdaproj/head` ++ `ling:kdaproj/member` ++ `dense/dec_gemv0` ++ `dense/dec_gemv_arrive_1` ++ `dense/dec_tile` ++ `dense/dec_gemv_arrive_2_1` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` (10779 B, `869bb863886d`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** `kdaProjB` (`ling.kdaProjB`), the outside plugin turboexl3_ling's MLX_SERVE_LING_KDA_BPROJ: a KDA layer's q|k|v, f and g as `turboexl3_ling_kdaproj_dec0`, and its b_proj in the same launch.
- **Shapes and dtypes:** dec0's operands, and xb bf16 [ROWS, IN] (x's rows), wb bf16 [OUTB, IN] row-major (the loader's [IN, OUTB] column-major view transposed back). Out: dec0's, and yb bf16 [ROWS, OUTB].
- **Grid and threadgroup:** `((tgs + OUTB / 4) · 128, chunks, 1)` / `(128, 1, 1)`: dec0's threadgroups, then OUTB / 4 of b's a row chunk (four outputs each, MLX's gemv's blockM); in each of b's only simdgroup 0 works.
- **Template args:** dec0's, FB (the members' threadgroups: b's first) and OUTB (b's width).
- **Replaces:** b_proj's own launches: MLX's matmul at one row, the exact verify's denseRows (a concatenate and a batched matmul) at 2 to 8.
- **Bits:** the members' are dec0's. b's are MLX's gemv's (gemv.h `GEMVKernel::run` at BM 1, BN 8, SM 1, SN 32, TM 4, TN 4, MLX 0.32): each K slice's products in its order, the simdgroup's shuffle-down sum, then the eight slices' sums added in simdgroup order, one simdgroup running the eight slices in turn; stored in bf16. It needs IN % 1024 == 0, OUTB % 4 == 0, OUTB < 4096 and IN >= 16 OUTB (`kdaproj.bTakes`), where MLX picks that kernel.
- **Pinned by:** `src/models/ling/bproj_test.zig` "turboexl3 ling bproj texts: ..." (CPU: the composition and the gemv statements, `kernels/test/pins/bproj_texts.metal`) and "turboexl3 gpu ling bproj: ..." (GPU: b's output against the served one-row matmul and denseRows, the members against kdaProj's, word for word at rows 1 to 8).

## `src/models/ling/kdaproj_pf.zig`

A KDA layer's q|k|v (4096 -> 24576), f and g (4096 -> 8192 each) at prompt rows (`kdaProjPrompt`, the outside plugin
turboexl3_ling's MLX_SERVE_LING_KDA_PROJ_PROMPT). `linear` at these rows runs, per member, its prepare, its W_hat
decode, MLX's f16 matmul (z half written) and the finish (z read back). Here each member's prepare and W_hat come from
its own `linear`'s launches (`Dense.prefillOperands`), and one launch runs the three products and finishes: pg2's
kernel (`turboexl3_pgemm_fused`) with its N tiles running member 0's, then 1's, then 2's.

L6 (the outside plugin turboexl3_ling's KDA_GATE_EPI, off by default): the same launch with member 1's (f's) output
its bounded gate (`kdaProjPromptGate`, `kdaProjPromptWGate`). The plugin runs f through its one-launch gate
(`ling_kda_gate`, the ops of mlx-serve's `kdaGateChain`), which reads y_1 back and writes g f32; here member 1's
tiles run that gate's statements on y_1's words in the store loop (`turboexl3_ling_kdaproj_pfg0`, exp(A_log) for
each element; `turboexl3_ling_kdaproj_pfg1`, exp(A_log) read from `turboexl3_ling_kda_gate_ea`'s array, made once
for a layer), and the gate's launch is gone. P4's text is unchanged: pg2's fused_6 is cut at its store loop and
pf_3 at its output line, the same bytes.

### `turboexl3_ling_kdaproj_pf`

<!-- kernel-facts: turboexl3_ling_kdaproj_pf -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj_pf.zig:52` (`fn kernelGet`) |
| inputs | `x0`, `x1`, `x2`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2` |
| outputs | `y0`, `y1`, `y2` |
| source | `text_pf` = `ling:kdaproj/pf_1` ++ `pg2/gemm_fused_2` ++ `ling:kdaproj/pf_3_1` ++ `ling:kdaproj/pf_3_2` ++ `pg2/gemm_fused_4` ++ `ling:kdaproj/pf_5` ++ `pg2/gemm_fused_6_1` ++ `pg2/gemm_fused_6_2` ++ `ling:kdaproj/pf_undef` (5529 B, `7454701fcb9a`) |
| header | `pgemm.header` = `steel_nax` ++ text 'using namespace mlx::steel;\n' ++ `pg2/gemm_loop` ++ text (72 B) (34287 B, `0ab3ef8c4fca`) |

<!-- /kernel-facts -->

- **Used by:** `kdaProjPrompt` (`ling.kdaProjPrompt`), the outside plugin turboexl3_ling's MLX_SERVE_LING_KDA_PROJ_PROMPT: a KDA layer's three EXL3 projections of its normed input at prompt rows (past `linear`'s register path, 32 rows on NAX). `kdaProjPromptW` (`ling.kdaProjPromptW`) runs it over W_hats the caller decoded once, the plugin's MLX_SERVE_LING_LAYER_WHAT: a layer-major layer's wide spans over the W_hats of its first.
- **Shapes and dtypes:** x0, x1, x2 half [M, K] (each member's prepared rows, natural order), w0, w1, w2 half [K, N_m] (each member's W_hat), svh0, svh1, svh2 half [N_m]. Out: y0, y1, y2 OUT_T [M, N_m] (x's dtype).
- **Grid and threadgroup:** pg2's: `((N0 + N1 + N2) / BN · 2^SWZ · 32, ceil(ceil(M / BM) / 2^SWZ) · WN, WM)` / `(32, WN, WM)`; a threadgroup's N tile lies in one member (each N_m a multiple of BN).
- **Template args:** pg2's: T (half), BM, BN, BK, WM, WN, SWZ, AM, AK (M and K aligned to the tile), AN (1) and OUT_T; the tile `kdaproj_pf.tileFor` (K 4096: `ling_tile`).
- **Replaces:** per member, MLX's matmul and the finish: six launches, and z's write and read, for one.
- **Bits:** each member's are its `linear` chain's: pg2's statements (MLX's NAX gemm_loop verbatim, each column's k in MLX's order whatever the tile; the accumulators rounded to half as MLX's store rounds them; the finish's statements per 128-column block), on that member's operands. A call is taken only where `linear` runs that chain at these rows and MLX runs its regular kernel (`pgemm.mlxRegular`: no split K) on the NAX path.
- **Pinned by:** `src/models/ling/kdaproj_pf.zig` "turboexl3 ling kdaproj prompt texts: ..." (CPU: the composition of pg2's pieces) and `src/models/ling/kdaproj_pf_test.zig` "turboexl3 gpu ling kdaproj prompt: ..." (GPU: each member against `root.linear` word for word at rows 33 to 16384, the refusal at 32) and "turboexl3 gpu ling kdaproj prompt over given W_hats: ..." (GPU: `kdaProjPromptW` over each member's W_hat, decoded once, against `kdaProjPrompt` word for word at rows 33, 1965 and 4096).

### `turboexl3_ling_kda_gate_ea`

<!-- kernel-facts: turboexl3_ling_kda_gate_ea -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj_pf.zig:318` (`fn eaKernelGet`) |
| inputs | `A_log` |
| outputs | `gea` |
| source | `text_gate_ea` = `ling:kdaproj/gate_ea` (349 B, `2b2923832ccb`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->

- **Used by:** `kdaproj_pf.gateEa`: the outside plugin turboexl3_ling's KDA_GATE_EPI at G1, once for a KDA layer (its
  A_log), the result kept with the layer and read by `turboexl3_ling_kdaproj_pfg1`.
- **Shapes and dtypes:** A_log f32 [n] (a KDA layer's per-channel A_log; n 8192 on Ling-3.1-flash). Out: gea f32 [n].
- **Grid and threadgroup:** `(ceil(n / 256) · 256, 1, 1)` / `(256, 1, 1)`: one thread a channel.
- **Template args:** none.
- **Replaces:** nothing alone: with `turboexl3_ling_kdaproj_pfg1` it takes the gate's first statement out of the store
  loop, one exp a channel made once in place of one an element.
- **Bits:** `metal::precise::exp(static_cast<float>(A_log[c]))`, the served one-launch gate's first statement
  (ling_kda_gate.metal) on the same input; G1's g carries it word for word.
- **Pinned by:** `src/models/ling/kdaproj_pf.zig` "turboexl3 ling kdaproj gate statements: ..." (CPU: its statement
  takes the gate's first right-hand side, `kernels/test/pins/kdaproj_gate.metal`) and
  `src/models/ling/kdaproj_gate_test.zig` "turboexl3 gpu ling kdaproj gate: kdaProjPromptGate's y_0 and y_2 are ..."
  (GPU: G1's g, which reads it, against the served gate's word for word).

### `turboexl3_ling_kdaproj_pfg0`

<!-- kernel-facts: turboexl3_ling_kdaproj_pfg0 -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj_pf.zig:304` (`fn gateKernelGet`) |
| inputs | `x0`, `x1`, `x2`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `A_log`, `dt_bias`, `lb` |
| outputs | `y0`, `y1`, `y2` |
| source | `text_pfg0` = `ling:kdaproj/pf_1` ++ `pg2/gemm_fused_2` ++ `ling:kdaproj/pf_3_1` ++ `ling:kdaproj/pf_3_2g` ++ `pg2/gemm_fused_4` ++ `ling:kdaproj/pf_5` ++ `pg2/gemm_fused_6_1` ++ `ling:kdaproj/pf_6g0` ++ `ling:kdaproj/pf_undef` (6232 B, `8202282622ad`) |
| header | `pgemm.header` = `steel_nax` ++ text 'using namespace mlx::steel;\n' ++ `pg2/gemm_loop` ++ text (72 B) (34287 B, `0ab3ef8c4fca`) |

<!-- /kernel-facts -->

- **Used by:** `kdaProjPromptGate` and `kdaProjPromptWGate` at G0 (`ling.kdaProjPromptGate`, `ling.kdaProjPromptWGate`):
  the outside plugin turboexl3_ling's KDA_GATE_EPI (off by default), a KDA layer's q|k|v, f and g at prompt rows with
  f's output its bounded gate, in place of `turboexl3_ling_kdaproj_pf` and the plugin's one-launch gate.
- **Shapes and dtypes:** `turboexl3_ling_kdaproj_pf`'s, then A_log and dt_bias f32 [N1] (N1 member 1's width, 8192) and
  lb an f32 scalar (the bound: Ling-3.1-flash's -5). Out: y0 and y2 OUT_T [M, N_m], y1 f32 [M, N1] (the gate g).
- **Grid and threadgroup:** `turboexl3_ling_kdaproj_pf`'s: the same tile, grid and N tiles' members.
- **Template args:** `turboexl3_ling_kdaproj_pf`'s.
- **Replaces:** `turboexl3_ling_kdaproj_pf` and the plugin's one-launch bounded gate (`ling_kda_gate`): f's y_1 never
  reaches device memory (its OUT_T write and the gate's read), one launch fewer for each KDA layer and call.
- **Bits:** y0 and y2 are `turboexl3_ling_kdaproj_pf`'s (the same pieces; members 0 and 2 store as pg2's fused_6_2). y1:
  member 1's value rounded as that store rounds it, then the served gate's statements on it in their order (precise exp
  of A_log, a + dt_bias in f32, the product, the sigmoid's branch form, times the bound, precise exp), channel o: the
  served gate's word.
- **Pinned by:** `src/models/ling/kdaproj_pf.zig` "turboexl3 ling kdaproj gate texts: ..." (CPU: the composition) and
  "turboexl3 ling kdaproj gate statements: ..." (CPU: the gate's statements against its test-only copy,
  `kernels/test/pins/kdaproj_gate.metal`); `src/models/ling/kdaproj_gate_test.zig` "turboexl3 gpu ling kdaproj gate:
  kdaProjPromptGate's y_0 and y_2 are ..." (GPU: y0 and y2 against `kdaProjPrompt`'s, g against the served gate's over
  `kdaProjPrompt`'s y_1, every word, rows 33 to 16384, and over given W_hats; `TURBOEXL3_LING_KDA_GATE`).

### `turboexl3_ling_kdaproj_pfg1`

<!-- kernel-facts: turboexl3_ling_kdaproj_pfg1 -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj_pf.zig:309` (`fn gateKernelGet`) |
| inputs | `x0`, `x1`, `x2`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `gea`, `dt_bias`, `lb` |
| outputs | `y0`, `y1`, `y2` |
| source | `text_pfg1` = `ling:kdaproj/pf_1` ++ `pg2/gemm_fused_2` ++ `ling:kdaproj/pf_3_1` ++ `ling:kdaproj/pf_3_2g` ++ `pg2/gemm_fused_4` ++ `ling:kdaproj/pf_5` ++ `pg2/gemm_fused_6_1` ++ `ling:kdaproj/pf_6g1` ++ `ling:kdaproj/pf_undef` (6214 B, `1de657c4cd05`) |
| header | `pgemm.header` = `steel_nax` ++ text 'using namespace mlx::steel;\n' ++ `pg2/gemm_loop` ++ text (72 B) (34287 B, `0ab3ef8c4fca`) |

<!-- /kernel-facts -->

- **Used by:** `kdaProjPromptGate` and `kdaProjPromptWGate` at G1: as `turboexl3_ling_kdaproj_pfg0`, the gate's
  exp(A_log) read from `gateEa`'s array (made once for the layer).
- **Shapes and dtypes:** `turboexl3_ling_kdaproj_pfg0`'s, with gea f32 [N1] (`turboexl3_ling_kda_gate_ea`'s) in A_log's
  place.
- **Grid and threadgroup:** `turboexl3_ling_kdaproj_pf`'s.
- **Template args:** `turboexl3_ling_kdaproj_pf`'s.
- **Replaces:** as `turboexl3_ling_kdaproj_pfg0`, and the gate's first statement for each element (one exp an element)
  by one read.
- **Bits:** `turboexl3_ling_kdaproj_pfg0`'s: gea[o] is that statement's word (the same statement on the same input,
  `turboexl3_ling_kda_gate_ea`).
- **Pinned by:** `turboexl3_ling_kdaproj_pfg0`'s tests, at G1 (`form=g1`).

## `src/models/ling/combine.zig`

A wide routed call's combine: the call's routed rows (each pair's expert output at weight 1, f32) times their routing
weights, summed over the top-k, in one launch, the words of the host's three MLX ops
`sum(multiply(astype(routed, f32), expand_dims(weights, -1)), -2)`. MLX's Multiply rounds each product to f32 on its
own; its sum over K <= 8 rows is `col_reduce_small` with one thread per row (threadgroup y = min(8, K)): each thread's
`v_k = p_k + 0` (its op against `Sum::init`), then thread 0's `t = v_0` and `t = v_k + t` in k order. The texts keep
those rounding points: `#pragma METAL fp contract(off)` (no product contracted into a sum), the product its own
statement, then the fold in that order.

### `turboexl3_ling_combine`

<!-- kernel-facts: turboexl3_ling_combine -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/combine.zig:75` (`fn kernelsFor`) |
| inputs | `routed`, `weights` |
| outputs | `out` |
| source | `text_routed` = `ling:combine/combine_1` ++ `ling:combine/src_routed` ++ `ling:combine/combine_2` (1286 B, `4040d9527524`) |
| header | `header_routed` = `ling:combine/contract` (219 B, `6aa4db359091`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.combineRouted`, the combine of a joined routed array (the tests' arm and a
  host's that joins).
- **Shapes and dtypes:** routed f32 [n, K, D], weights f32 [n, K], K 1 to 8. Out: f32 [n, D].
- **Grid and threadgroup:** `(n · ceil(D / 4), 1, 1)` / `(256, 1, 1)`: a thread per row and 4 adjacent outputs.
- **Template args:** K (the top-k), D (the hidden size).
- **Replaces:** MLX's astype (a no-op on f32), Multiply and the sum's `col_reduce_small`: 2 launches and the f32
  products [n, K, D] written and read again, with 1.
- **Bits:** the words of `sum(multiply(astype(routed, f32), expand_dims(weights, -1)), -2)` on MLX: each product rounded
  to f32 alone, the K values folded in col_reduce_small's order (K <= 8 only: past 8 its threads take several rows).
- **Pinned by:** `src/models/ling/combine_test.zig` "turboexl3 gpu ling combine: combineRouted gives MLX's
  sum(multiply(astype(routed, f32), expand_dims(weights, -1)), -2) word for word at rows 1, 7, 4096 and 4097 over signed
  zeros, overflowing, tiny and denormal products and cancellations (TURBOEXL3_GPU=1)" (GPU, words); `src/models/ling/
  combine.zig` "turboexl3 ling combine texts: ..." (CPU, the texts).

### `turboexl3_ling_combine_parts`

<!-- kernel-facts: turboexl3_ling_combine_parts -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/combine.zig:77` (`fn kernelsFor`) |
| inputs | `s0`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`, `s7`, `s8`, `s9`, `s10`, `s11`, `s12`, `s13`, `s14`, `s15`, `s16`, `s17`, `s18`, `s19`, `s20`, `s21`, `s22`, `s23`, `loc`, `weights` |
| outputs | `out` |
| source | `text_parts` = `ling:combine/combine_1` ++ `ling:combine/src_parts` ++ `ling:combine/combine_2` (1480 B, `a5ba86627db1`) |
| header | `header_parts` = `ling:combine/contract` ++ `ling:combine/pick` (1686 B, `6ccfe2f7a4bd`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.combineParts`, a wide routed call's combine over the call's unjoined outputs
  (the streamer's JOINLESS route: `routedParts`), so the joined [n, K, D] array is never built; the outside plugin's
  opt-in (turboexl3_ling's `MLX_SERVE_LING_COMBINE=1`).
- **Shapes and dtypes:** s0 to s23 f32 [r_i, D] (the call's outputs, at most 24; an unused slot binds s0), loc int32
  [n, K, 2] (each pair's source and row there, the streamer's `planJoinless` / `fillJoinlessLoc`), weights f32 [n, K],
  K 1 to 8. Out: f32 [n, D].
- **Grid and threadgroup:** `(n · ceil(D / 4), 1, 1)` / `(256, 1, 1)`, `turboexl3_ling_combine`'s.
- **Template args:** K (the top-k), D (the hidden size).
- **Replaces:** the join (`concatenate` of the outputs, then `take` in routed order: the routed rows written twice) and
  `turboexl3_ling_combine`'s three MLX ops: 4 launches, with 1.
- **Bits:** `turboexl3_ling_combine`'s words on the array the join would build (each pair's row read where the join
  copies it from).
- **Pinned by:** `src/models/ling/combine_test.zig` "turboexl3 gpu ling combine: combineParts over a wide call's unjoined
  outputs (5 read in place; 40 through the streamer's merge rule) gives the composed combine's bytes on the array the
  join builds (TURBOEXL3_GPU=1)" (GPU, words); `src/models/ling/combine.zig` "turboexl3 ling combine texts: ..." (CPU,
  the texts).

### `turboexl3_ling_combine_parts_h`

<!-- kernel-facts: turboexl3_ling_combine_parts_h -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/combine.zig:79` (`fn kernelsFor`) |
| inputs | `s0`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`, `s7`, `s8`, `s9`, `s10`, `s11`, `s12`, `s13`, `s14`, `s15`, `s16`, `s17`, `s18`, `s19`, `s20`, `s21`, `s22`, `s23`, `loc`, `weights` |
| outputs | `out` |
| source | `text_parts_h` = `ling:combine/combine_1` ++ `ling:combine/src_parts_h` ++ `ling:combine/combine_2` (1481 B, `901e2c2d2a55`) |
| header | `header_parts_h` = `ling:combine/contract` ++ `ling:combine/pick_h` (1663 B, `b16712bbbc57`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.combineParts` when every source is f16, the routed pairs of a wide prompt
  call stored as f16 (`root.ling.moeLingRowsPairsF16`); the outside plugin's opt-in (turboexl3_ling's
  `MLX_SERVE_LING_PF_F16_PAIRS=1`). A mix of f16 and f32 sources widens the f16 ones and runs
  `turboexl3_ling_combine_parts`.
- **Shapes and dtypes:** s0 to s23 f16 [r_i, D] (each word the f16 value of the f32 word the served call stores: a
  top-1 pair at weight 1 or a pad row at +0), loc int32 [n, K, 2], weights f32 [n, K], K 1 to 8. Out: f32 [n, D].
- **Grid and threadgroup:** `(n · ceil(D / 4), 1, 1)` / `(256, 1, 1)`, `turboexl3_ling_combine_parts`'s.
- **Template args:** K (the top-k), D (the hidden size).
- **Replaces:** `turboexl3_ling_combine_parts` over the same rows in f32: half the bytes read for every routed pair
  (and half written by the chain's finish, `OUT_T` half).
- **Bits:** `turboexl3_ling_combine_parts`' words on the f32 sources the f16 ones widen to: the same statements with
  each source pointer half (`src_parts_h`, `pick_h`), and the product `pk[ok + i] * w` widens the half to f32
  exactly before it rounds.
- **Pinned by:** `src/models/ling/f16pairs_test.zig` "turboexl3 gpu ling f16 pairs: ..." (GPU, words: f16, mixed and
  f32 sources at the standard request's call, a 2,048-token call and a padded call; a flipped f16 word caught);
  `src/models/ling/combine.zig` "turboexl3 ling combine texts: the f16 sources' JOINLESS text is the f32 one with
  each source pointer half ..." (CPU, the texts).

## `src/models/ling/tail.zig`

A MoE layer's end at decode and verify rows in one launch: the routed sum over the call's joined rows, its cast to the
activation dtype, the shared expert's add, then the residual add and the next layer's input norm (the final norm after
the last layer). The text is mlx-serve's `add_norm` (`add_norm.fmt.metal`, its plain arm) word for word, cut at its
delta (`tail_1`, `tail_2`): one threadgroup of D / 4 threads a row, four columns a thread, the simdgroups' sums of
squares folded through a 32-slot table, `precise::rsqrt`, `w * T(x * inv)`. In place of `delta = float(X[at]);` the
delta (`tail_delta`) is the routed sum in `turboexl3_ling_combine`'s statements (each product rounded to f32 alone, the
K values folded in `col_reduce_small`'s order, contraction off in that block), rounded to T, plus the shared expert,
rounded to T again: the MoE block's astype and add.

### `turboexl3_ling_moe_tail`

<!-- kernel-facts: turboexl3_ling_moe_tail -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/tail.zig:97` (`fn kernelOf`) |
| inputs | `H`, `Y`, `WT`, `XS`, `W`, `eps` |
| outputs | `H_NEW`, `HN` |
| source | `text` = `ling:tail/tail_1` ++ `ling:tail/tail_delta` ++ `ling:tail/tail_2` (2241 B, `ebc2e07ccb65`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.moeTail`, a MoE layer's end at decode and verify rows (1 to 8) after its
  routed call's join; the outside plugin's opt-in (turboexl3_ling's `MLX_SERVE_LING_MOE_TAIL=1`).
- **Shapes and dtypes:** H bf16 or f16 [..., D] (the residual; rows 1 to 8, D a multiple of 4 up to 4096), Y f32 [rows, K, D]
  or [rows · K, D] (the call's joined rows, each pair's expert output at weight 1), WT bf16, f16 or f32 [rows, K] (the
  routes' weights, K 1 to 8), XS [..., D] in H's dtype (the shared expert's output), W [D] in H's dtype (the next
  layer's input norm, or the final norm), eps f32 0-dim. Out: H_NEW and HN in H's shape and dtype.
- **Grid and threadgroup:** `(D / 4 · rows, 1, 1)` / `(D / 4, 1, 1)`: mlx-serve's add_norm config, one threadgroup a
  row, four columns a thread.
- **Template args:** T (H's dtype), D (the width), TN (D / 4), K (the top-k).
- **Replaces:** the source's routed sum (the weights' astype to f32, MLX's Multiply and the sum's `col_reduce_small`),
  the MoE block's astype of the sum to T and its add of the shared expert, and mlx-serve's add_norm (`msv_add_norm`):
  6 launches with 1.
- **Bits:** the chain's words: each product `Y · float(WT)` rounded to f32 alone, the K values folded in
  `col_reduce_small`'s order (`t = p_0 + 0`, then `t = (p_k + 0) + t`: the words of `p_k + t`, as `t` is never -0.0),
  rounded to T, plus the shared expert, rounded to T; then the add_norm's statements, so the residual is MLX's add's
  words and the normed rows MLX's fast rms_norm's.
- **Pinned by:** `src/models/ling/tail_test.zig` "turboexl3 gpu ling tail: moeTail gives the served chain's words ..."
  (GPU, words: the residual and the normed rows at rows 1 to 8, limits 0 and 10, the joined and flat rows, planted
  signed zeros).

### `turboexl3_ling_moe_tail_parts`

<!-- kernel-facts: turboexl3_ling_moe_tail_parts -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/tail.zig:106` (`fn partsKernelOf`) |
| inputs | `H`, `s0`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`, `s7`, `loc`, `WT`, `XS`, `W`, `eps` |
| outputs | `H_NEW`, `HN` |
| source | `text_parts` = `ling:tail/tail_1` ++ `ling:tail/tail_delta_parts` ++ `ling:tail/tail_2` (2471 B, `2a1a6a50ab95`) |
| header | `pick` = `ling:tail/pick` (603 B, `f06a112bd2f9`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.moeTailParts`, a MoE layer's end at decode and verify rows (1 to 8) read
  straight from its routed call's waves; the outside plugin's opt-in on top of the tail (turboexl3_ling's
  `MLX_SERVE_LING_ROUTED_JOIN=1` with `MLX_SERVE_LING_MOE_TAIL=1`).
- **Shapes and dtypes:** `turboexl3_ling_moe_tail`'s, with Y replaced by s0..s7 f32 [r_i, D] (1 to 8 sources: the call's
  device wave and its miss parts in the join's order, each pair's expert output at weight 1; an unused slot aliases s0)
  and loc int32 [rows, K, 2] (each routed pair's (source, row), its last carrier in the join; every entry names a bound
  source and one of its rows, which the launch does not check).
- **Grid and threadgroup:** `turboexl3_ling_moe_tail`'s: `(D / 4 · rows, 1, 1)` / `(D / 4, 1, 1)`.
- **Template args:** T, D, TN, K, as `turboexl3_ling_moe_tail`'s.
- **Replaces:** the streamer's join of a decode call (experts.zig `join`: MLX's concatenate, one copy per wave, then its
  take at each pair's last carrier) and `turboexl3_ling_moe_tail` over it: with no miss part the take and the tail
  (a concatenate of one array is the array), 2 launches with 1; with m miss parts the concatenate's 1 + m copies, the
  take and the tail, 3 + m launches with 1.
- **Bits:** `turboexl3_ling_moe_tail`'s words over the join: `tail_delta_parts` is `tail_delta` with Y's address
  `(r · K + k) · D + c` replaced by the picked source's `loc row · D + c`. The join only moves f32 words (the
  concatenate's copies and the take's gather), so each product reads the same value.
- **Pinned by:** `src/models/ling/tail_test.zig` "turboexl3 gpu ling tail parts: moeTailParts reads each pair's row in
  place ..." (GPU, words: the residual and the normed rows against the served chain and against `moeTail` over the
  streamer's join, at rows 1 to 8, limits 0 and 10, one to three sources and all eight, the later miss part's row last
  where two carry a pair; the refusals).

### `turboexl3_ling_moe_tail_wide`

<!-- kernel-facts: turboexl3_ling_moe_tail_wide -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/tail.zig:116` (`fn wideKernelsOf`) |
| inputs | `H`, `s0`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`, `s7`, `s8`, `s9`, `s10`, `s11`, `s12`, `s13`, `s14`, `s15`, `s16`, `s17`, `s18`, `s19`, `s20`, `s21`, `s22`, `s23`, `loc`, `WT`, `XS`, `WE` |
| outputs | `H_NEW`, `HN` |
| source | `text_wide` = `ling:tail/tail_we` ++ `ling:tail/tail_1` ++ `ling:tail/tail_delta_wide` ++ `ling:tail/tail_2` (3036 B, `aac99e729776`) |
| header | `header_wide` = `ling:combine/pick` (1467 B, `7dc35360395d`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.moeTailWide`, a layer-major MoE layer's end at a prompt span's rows, read
  straight from its wide routed call's JOINLESS parts; the outside plugin's opt-in (turboexl3_ling's
  `MLX_SERVE_LING_PF_MOE_TAIL=1`, L3), on f32 sources (PF_F16_PAIRS off).
- **Shapes and dtypes:** H [rows, D] in T (bf16 or f16; any rows, D a multiple of 4 up to 4096), s0..s23 f32
  [r_i, D] (1 to 24 sources: the call's outputs as the streamer plans them, above 24 the smallest merged into the
  last; an unused slot aliases s0), loc int32 [rows, K, 2] (each of the span's pairs' (source, row); unchecked), WT
  [rows, K] (bf16, f16 or f32), XS [rows, D] in T (the span's shared expert output), WE f32 [D + 1] (the next norm's
  weight widened, then eps); H_NEW and HN in H's shape and T. K 1 to 8.
- **Grid and threadgroup:** `turboexl3_ling_moe_tail`'s: `(D / 4 · rows, 1, 1)` / `(D / 4, 1, 1)`, one threadgroup a
  row at any row count (MLX's fast rms_norm runs its single-row kernel the same way at every row count).
- **Template args:** T, D, TN, K, as `turboexl3_ling_moe_tail`'s.
- **Replaces:** a prompt span's end after its call's combine (`turboexl3_ling_combine_parts`'s f32 sum, written and
  read again), the streamed source's astype to T, the span's slice, the block's add of the shared expert, MLX's add of
  the residual and MLX's fast rms_norm at the next layer's start (or the final norm): 5 launches a span and the call's
  combine with one launch a span; about 1.35 GB of passes a MoE layer at the standard request.
- **Bits:** the served chain's words: the delta is `tail_delta_parts`' statements with the combine's 24-source pick
  (each product rounded to f32 alone, `col_reduce_small`'s fold in k order, rounded to T, plus the shared expert,
  rounded to T, as `turboexl3_ling_combine_parts` and the block's astype and add give them), then the add_norm's
  statements, whose residual is MLX's add's words and whose normed rows are MLX's fast rms_norm's (its single-row
  kernel: four columns a thread, the per-simdgroup sums folded through 32 slots, precise rsqrt, `w * T(x * inv)`),
  the same statements at every row count. W and eps come from WE: each f32 word there is the T word widened.
- **Pinned by:** `src/models/ling/tail_test.zig` "turboexl3 gpu ling tail wide: moeTailWide gives the served chain's
  words at prompt widths ..." (GPU, words: the residual and the normed rows against the served chain at the
  standard request's spans 4,096 x 4, 1,965 and 31 of one 18,380-token call over 24 merged sources, f32 at the
  first, 1,965- and 31-row spans; the refusals) and "turboexl3 ling tail wide texts: ..." (CPU).

### `turboexl3_ling_moe_tail_wide_h`

<!-- kernel-facts: turboexl3_ling_moe_tail_wide_h -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/tail.zig:118` (`fn wideKernelsOf`) |
| inputs | `H`, `s0`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`, `s7`, `s8`, `s9`, `s10`, `s11`, `s12`, `s13`, `s14`, `s15`, `s16`, `s17`, `s18`, `s19`, `s20`, `s21`, `s22`, `s23`, `loc`, `WT`, `XS`, `WE` |
| outputs | `H_NEW`, `HN` |
| source | `text_wide_h` = `ling:tail/tail_we` ++ `ling:tail/tail_1` ++ `ling:tail/tail_delta_wide_h` ++ `ling:tail/tail_2` (3037 B, `8c1d4cf631d8`) |
| header | `header_wide_h` = `ling:combine/pick_h` (1444 B, `cbbf84fad66f`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.moeTailWide` over f16 sources, the outside plugin's opt-in (turboexl3_ling's
  `MLX_SERVE_LING_PF_MOE_TAIL=1`, L3) on its served sources (PF_F16_PAIRS, kept: every prompt pair stored as f16).
- **Shapes and dtypes:** `turboexl3_ling_moe_tail_wide`'s, with s0..s23 f16 [r_i, D] (each word the f16 value of the
  f32 word the served f32 call stores); every source f16 (a mix is refused: the served combine widens it).
- **Grid and threadgroup:** `turboexl3_ling_moe_tail_wide`'s.
- **Template args:** `turboexl3_ling_moe_tail_wide`'s.
- **Replaces:** `turboexl3_ling_combine_parts_h`'s f32 sum and the same five launches a span as
  `turboexl3_ling_moe_tail_wide`.
- **Bits:** `turboexl3_ling_moe_tail_wide`'s words: `tail_delta_wide_h` is `tail_delta_wide` with the source pointer
  half and the combine's pick_h, so each product `float(yk[..]) · w` widens the half to f32 exactly before it
  multiplies, as `turboexl3_ling_combine_parts_h` does.
- **Pinned by:** `src/models/ling/tail_test.zig` "turboexl3 gpu ling tail wide: ..." (GPU, words: the residual and the
  normed rows against the served chain at every span of the standard request's call) and "turboexl3 ling tail wide
  texts: ..." (CPU: the f16 text is the f32 one with the source pointer half and pick_h, nothing else).

## `src/models/ling/router.zig`

The MoE router at decode rows 1 to 8 with the read-ahead predictor's scores (the next MoE layer's router on the call's
rows), in two launches in place of the host's nine: launch 1 runs MLX's GEMV statements for the router's logits and the
predictor's words; launch 2 runs the host's fused router over those logits. Each output keeps the host's reduction order
and roundings. The host's loader builds the router table as the f16 weight's transposed view widened to f32, then + 0
(`f32Table`); MLX's astype and add keep the view's layout, so the table is [hidden, 512] column-major (strides 1,
hidden) and the host's matmul runs gemv over its memory, the weight's own [512, hidden] row-major order (b transposed).
The router's logits are that gemv (`gemv.h` GEMVKernel at BM 4, BN 1, SM 1, SN 32, TM 4, TN 4) over x widened to f32,
every row its own, reading the table's memory through its transposed view (f32), or the loaded f16 weight, each word
widened, then + 0 (the table's word: the loader's + 0 turns -0.0 into +0.0; `_h` kernels). The predictor's words are
MLX's gemv at one row and its gemv_wide at 2 to 8 rows (k_lanes 32 to 5 rows, 16 past; MLX's route on M3 and later),
each rounded to bf16 (the product's dtype), widened, then scored with the statements of MLX's elementwise kernels.
MLX's metallib is built `-fno-fast-math`, so its bare exp, log and sqrt are the precise ones: the scores spell them
precise, and their headers keep each statement's rounding (`#pragma METAL fp contract(off)`). The GEMV statements carry
no pragma: they compile as MLX's own do. `tableF32` rebuilds the loader's table from the f16 weight's view (astype, then
+ 0) for a host that keeps the weight in f16 and runs its own matmul where these launches do not serve. Under
`Config.router_rows` a call of 2 rows or more over the f16 weight takes the rows launch (the `_hr` kernels): each router
threadgroup computes its 16 logits for a group of up to 4 rows, so the weight is read once a group, not once a row.

### `turboexl3_ling_router_logits`

<!-- kernel-facts: turboexl3_ling_router_logits -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/router.zig:161` (`fn kernelsFor`) |
| inputs | `x`, `w` |
| outputs | `logits` |
| source | `text_logits` = `ling:router/head` ++ `ling:router/logits_1` ++ `ling:router/logits_w` ++ `ling:router/logits_2` ++ `ling:router/close` (2372 B, `0d948b7632c5`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.router.routerLogits` (and `route`) without a predictor over the f32 table: the last MoE layer (no next router); the outside plugin's opt-in (turboexl3_ling's `MLX_SERVE_LING_ROUTER_FUSED=1`).
- **Shapes and dtypes:** x bf16 [R, HK] (R 1 to 8; HK a multiple of 1024 below 8192), w f32 [512, HK] row-major (the loader's [HK, 512] column-major table, transposed back: a view of its memory). Out: logits f32 [R, 512].
- **Grid and threadgroup:** `(32R · 128, 1, 1)` / `(128, 1, 1)`: 32 threadgroups a row, each one of MLX's gemv tiles of 16 logits.
- **Template args:** R (the rows), NE (512, the experts), HK (the hidden size).
- **Replaces:** MLX's astype of x to f32 and its gemv: 2 launches with 1.
- **Bits:** each row the words of the host's one-row `matmul(x, table)` over the loader's column-major table (MLX widens x to f32, then runs gemv over the table's memory: its statements in order here, x widened in registers to the astype's values). At 2 to 8 rows those are the host's exact verify's words (turboexl3_ling's `denseRows`: a batched one-row gemv a row); MLX's own matmul of several rows takes a steel GEMM there, another order.
- **Pinned by:** `src/models/ling/router_test.zig` "turboexl3 gpu ling router: routerLogits, select and route over the loader's table ..." (GPU, words at rows 1 to 8, every operand built as the host builds it) and "turboexl3 gpu ling router pack weights: ..." (GPU, with `TURBOEXL3_LING_PACK`: every router weight of a pack as stored, rows 1 to 8); `src/models/ling/router.zig` "turboexl3 ling router texts: ..." (CPU, the texts) and "turboexl3 ling router plan: ..." (CPU, the grid).

### `turboexl3_ling_router_pred`

<!-- kernel-facts: turboexl3_ling_router_pred -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/router.zig:163` (`fn kernelsFor`) |
| inputs | `x`, `w`, `pw`, `pb` |
| outputs | `logits`, `scores` |
| source | `text_pred` = `ling:router/head` ++ `ling:router/logits_1` ++ `ling:router/logits_w` ++ `ling:router/logits_2` ++ `ling:router/pred_open` ++ `ling:router/pred_one` ++ `ling:router/pred_wide` ++ `ling:router/close` (6631 B, `fb59ac42e0eb`) |
| header | `score_softplus_sqrt` = `ling:router/score_softplus_sqrt` (1499 B, `0f43a7203e4b`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.router.routerLogits` (and `route`) with a `Predictor` at its default score (`.softplus_sqrt`) over the f32 table: every MoE layer but the last; the outside plugin's opt-in.
- **Shapes and dtypes:** x and w as `turboexl3_ling_router_logits`'s; pw bf16 [512, HK] row-major (the next MoE layer's router: the host's `router_raw`, the loaded weight cast to bf16), pb f32 [512] (its expert bias). Out: logits f32 [R, 512]; scores f32 [R, 512].
- **Grid and threadgroup:** `((32R + P) · 128, 1, 1)` / `(128, 1, 1)`: the router's 32R threadgroups, then the predictor's P: 32 at one row (gemv's tiles, 16 rows of pw each), 128 at 2 to 5 rows (gemv_wide at k_lanes 32: a row of pw a simdgroup) and 64 at 6 to 8 (k_lanes 16: two rows a simdgroup).
- **Template args:** R, NE, HK.
- **Replaces:** the router's 2 launches and the predictor's 5 (MLX's gemv or gemv_wide, its astype to f32, logaddexp, sqrt and add): 7 launches with 1.
- **Bits:** the logits as `turboexl3_ling_router_logits`'s; the scores the words of the host streamer's `nextScores`, MLX's `add(sqrt(logaddexp(astype(matmul(x, transpose(pw)), f32), 0)), pb)` at the call's row count: gemv's (one row) or gemv_wide's (2 to 8 rows) statements, the bf16 rounding, then LogAddExp with utils.h's log1p, Sqrt and Add.
- **Pinned by:** `src/models/ling/router_test.zig` "turboexl3 gpu ling router: routerLogits, select and route over the loader's table ..." (GPU, words at rows 1 to 8, every operand built as the host builds it); `src/models/ling/router.zig` "turboexl3 ling router texts: ..." (CPU, the texts) and "turboexl3 ling router plan: ..." (CPU, the grid).

### `turboexl3_ling_router_pred_sig`

<!-- kernel-facts: turboexl3_ling_router_pred_sig -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/router.zig:165` (`fn kernelsFor`) |
| inputs | `x`, `w`, `pw`, `pb` |
| outputs | `logits`, `scores` |
| source | `text_pred` = `ling:router/head` ++ `ling:router/logits_1` ++ `ling:router/logits_w` ++ `ling:router/logits_2` ++ `ling:router/pred_open` ++ `ling:router/pred_one` ++ `ling:router/pred_wide` ++ `ling:router/close` (6631 B, `fb59ac42e0eb`) |
| header | `score_sigmoid` = `ling:router/score_sigmoid` (522 B, `765e609d7b4e`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: the same with `Predictor.score = .sigmoid`, Ling's own read-ahead score (candidate C1, turboexl3_ling's `MLX_SERVE_LING_PREDICTOR=1`, whose pick runs on the host).
- **Shapes and dtypes:** `turboexl3_ling_router_pred`'s.
- **Grid and threadgroup:** `turboexl3_ling_router_pred`'s.
- **Template args:** R, NE, HK.
- **Replaces:** the router's 2 launches and the predictor's 4 (MLX's gemv or gemv_wide, its astype to f32, sigmoid and add): 6 launches with 1.
- **Bits:** the logits as `turboexl3_ling_router_logits`'s; the scores MLX's `add(sigmoid(astype(matmul(x, transpose(pw)), f32)), pb)` word for word (MLX's two-sided Sigmoid on the precise exp).
- **Pinned by:** `src/models/ling/router_test.zig` "turboexl3 gpu ling router: routerLogits, select and route over the loader's table ..." (GPU, words at rows 1 to 8, every operand built as the host builds it); `src/models/ling/router.zig` "turboexl3 ling router texts: ..." (CPU, the texts) and "turboexl3 ling router plan: ..." (CPU, the grid).

### `turboexl3_ling_router_logits_h`

<!-- kernel-facts: turboexl3_ling_router_logits_h -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/router.zig:167` (`fn kernelsFor`) |
| inputs | `x`, `w` |
| outputs | `logits` |
| source | `text_logits_h` = `ling:router/head` ++ `ling:router/logits_1` ++ `ling:router/logits_wh` ++ `ling:router/logits_2` ++ `ling:router/close` (2550 B, `ce985411885e`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.router.routerLogits` (and `route`) over the loaded f16 weight; the outside plugin's opt-in `MLX_SERVE_LING_ROUTER_F16=1` (with `MLX_SERVE_LING_ROUTER_FUSED=1`), which drops the f32 table and serves the weight as loaded (the launch reads 4 MiB a layer in place of the table's 8). Without a predictor: the last MoE layer and the NextN layer.
- **Shapes and dtypes:** x bf16 [R, HK]; w f16 [512, HK] row-major (the weight as loaded from the pack: the loader's transposed view, transposed back). Out: logits f32 [R, 512].
- **Grid and threadgroup:** `turboexl3_ling_router_logits`'s.
- **Template args:** R, NE, HK.
- **Replaces:** `turboexl3_ling_router_logits`'s 2 launches with 1, reading the router's weight in half the bytes.
- **Bits:** each table word read as the f16 word widened, then + 0, the loader's table word (its astype is exact; its + 0 turns -0.0 into +0.0): the words of `turboexl3_ling_router_logits` over the loader's table.
- **Pinned by:** `src/models/ling/router_test.zig` "turboexl3 gpu ling router f16 words: ..." (GPU, a weight holding every finite f16 word, rows 1 to 8), "turboexl3 gpu ling router pack weights: ..." (GPU, with `TURBOEXL3_LING_PACK`: every router weight of a pack as stored, -0.0 words among them, rows 1 to 8) and "turboexl3 gpu ling router: ..." (GPU); `src/models/ling/router.zig` "turboexl3 ling router texts: ..." (CPU, the texts) and "turboexl3 ling router plan: ..." (CPU, the grid).

### `turboexl3_ling_router_pred_h`

<!-- kernel-facts: turboexl3_ling_router_pred_h -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/router.zig:169` (`fn kernelsFor`) |
| inputs | `x`, `w`, `pw`, `pb` |
| outputs | `logits`, `scores` |
| source | `text_pred_h` = `ling:router/head` ++ `ling:router/logits_1` ++ `ling:router/logits_wh` ++ `ling:router/logits_2` ++ `ling:router/pred_open` ++ `ling:router/pred_one` ++ `ling:router/pred_wide` ++ `ling:router/close` (6809 B, `44f1b8043a90`) |
| header | `score_softplus_sqrt` = `ling:router/score_softplus_sqrt` (1499 B, `0f43a7203e4b`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.router.routerLogits` (and `route`) over the loaded f16 weight; the outside plugin's opt-in `MLX_SERVE_LING_ROUTER_F16=1` (with `MLX_SERVE_LING_ROUTER_FUSED=1`), which drops the f32 table and serves the weight as loaded (the launch reads 4 MiB a layer in place of the table's 8). With a predictor at the default score.
- **Shapes and dtypes:** x bf16 [R, HK]; w f16 [512, HK] row-major (the weight as loaded from the pack: the loader's transposed view, transposed back); pw, pb as `turboexl3_ling_router_pred`'s. Out: logits, scores f32 [R, 512].
- **Grid and threadgroup:** `turboexl3_ling_router_pred`'s.
- **Template args:** R, NE, HK.
- **Replaces:** `turboexl3_ling_router_pred`'s 7 launches with 1, reading the router's weight in half the bytes.
- **Bits:** each table word read as the f16 word widened, then + 0, the loader's table word (its astype is exact; its + 0 turns -0.0 into +0.0): the words of `turboexl3_ling_router_pred` over the loader's table.
- **Pinned by:** `src/models/ling/router_test.zig` "turboexl3 gpu ling router f16 words: ..." (GPU, a weight holding every finite f16 word, rows 1 to 8) and "turboexl3 gpu ling router: ..." (GPU); `src/models/ling/router.zig` "turboexl3 ling router texts: ..." (CPU, the texts) and "turboexl3 ling router plan: ..." (CPU, the grid).

### `turboexl3_ling_router_pred_sig_h`

<!-- kernel-facts: turboexl3_ling_router_pred_sig_h -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/router.zig:171` (`fn kernelsFor`) |
| inputs | `x`, `w`, `pw`, `pb` |
| outputs | `logits`, `scores` |
| source | `text_pred_h` = `ling:router/head` ++ `ling:router/logits_1` ++ `ling:router/logits_wh` ++ `ling:router/logits_2` ++ `ling:router/pred_open` ++ `ling:router/pred_one` ++ `ling:router/pred_wide` ++ `ling:router/close` (6809 B, `44f1b8043a90`) |
| header | `score_sigmoid` = `ling:router/score_sigmoid` (522 B, `765e609d7b4e`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.router.routerLogits` (and `route`) over the loaded f16 weight; the outside plugin's opt-in `MLX_SERVE_LING_ROUTER_F16=1` (with `MLX_SERVE_LING_ROUTER_FUSED=1`), which drops the f32 table and serves the weight as loaded (the launch reads 4 MiB a layer in place of the table's 8). With a predictor at Ling's score (C1).
- **Shapes and dtypes:** x bf16 [R, HK]; w f16 [512, HK] row-major (the weight as loaded from the pack: the loader's transposed view, transposed back); pw, pb as `turboexl3_ling_router_pred`'s. Out: logits, scores f32 [R, 512].
- **Grid and threadgroup:** `turboexl3_ling_router_pred`'s.
- **Template args:** R, NE, HK.
- **Replaces:** `turboexl3_ling_router_pred_sig`'s 6 launches with 1, reading the router's weight in half the bytes.
- **Bits:** each table word read as the f16 word widened, then + 0, the loader's table word (its astype is exact; its + 0 turns -0.0 into +0.0): the words of `turboexl3_ling_router_pred_sig` over the loader's table.
- **Pinned by:** `src/models/ling/router_test.zig` "turboexl3 gpu ling router: routerLogits, select and route over the loader's table ..." (GPU, words at rows 1 to 8, every operand built as the host builds it); `src/models/ling/router.zig` "turboexl3 ling router texts: ..." (CPU, the texts) and "turboexl3 ling router plan: ..." (CPU, the grid).

### `turboexl3_ling_router_select`

<!-- kernel-facts: turboexl3_ling_router_select -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/router.zig:173` (`fn kernelsFor`) |
| inputs | `logits`, `bias`, `scale` |
| outputs | `inds`, `scores` |
| source | `text_select` = `ling:router/select` (3430 B, `f8657dcb970c`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.router.select` (and `route`): a MoE layer's top-8 after launch 1; the outside
  plugin's opt-in.
- **Shapes and dtypes:** logits f32 [R, 512] (R 1 to 8), bias f32 [512] (the layer's expert bias), scale f32 0-dim
  (routed_scaling_factor, 2.5 for Ling). Out: inds u32 [R, 8] (in the selection's order); scores bf16 [R, 8].
- **Grid and threadgroup:** `(256, R, 1)` / `(256, 1, 1)`: the host's config (`moeRouterThreadGroup(512)`).
- **Template args:** TOUT (bfloat16; the host's launch runs float32), NE (512), NK (8), TG (256), NORM (1), NG (8),
  GSZ (64), TOPKG (4).
- **Replaces:** the host's fused router launch and the MoE block's astype of its f32 weights to bf16: 2 launches with 1.
- **Bits:** the host's text byte for byte (`text_select`: sdk/blocks/moe/router_kernel.zig
  `moeRouterSource(.sigmoid_bias_grouped)`; turboexl3_ling's build checks the copy against its source). TOUT bf16
  rounds each f32 weight to bf16 at its store, as MLX's astype does.
- **Pinned by:** `src/models/ling/router_test.zig` "turboexl3 gpu ling router: ..." (GPU, the ids and weights against
  the host's launch then the astype, alone and in `route`, rows 1 to 8); `src/models/ling/router.zig` "turboexl3 ling
  router texts: ..." (CPU, the host's statements).

### `turboexl3_ling_router_logits_hr`

<!-- kernel-facts: turboexl3_ling_router_logits_hr -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/router.zig:413` (`fn rowsKernelOf`) |
| inputs | `x`, `w` |
| outputs | `logits` |
| source | `text_logits_hr` = `ling:router/head_rows` ++ `ling:router/logits_rows_1` ++ `ling:router/logits_wh` ++ `ling:router/logits_rows_2` ++ `ling:router/close` (3210 B, `64478f05ed17`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.router.routerLogits` (and `route`) over the loaded f16 weight at a call of 2
  rows or more without a predictor (the last MoE layer and the NextN layer), under `Config.router_rows`
  (`routerLogitsWith`): the outside plugin turboexl3_ling's candidate ROUTER_ROWS (with ROUTER_F16). Not served by
  default.
- **Shapes and dtypes:** `turboexl3_ling_router_logits_h`'s, R 2 to 8.
- **Grid and threadgroup:** `(32·NRG · 128, 1, 1)` / `(128, 1, 1)`: 32 threadgroups (MLX's gemv tiles of 16 logits)
  for each group of RG = min(R, 4) rows, NRG = ceil(R / RG) groups. Each threadgroup runs its tile for every row of its
  group; a group past the call's last row repeats that row's reads and stores nothing for it.
- **Template args:** R, NE, HK, as `turboexl3_ling_router_logits_h`'s (RG and NRG follow from R).
- **Replaces:** `turboexl3_ling_router_logits_h` at those calls (the GEMV inventory's D5b): 32R threadgroups, each tile's
  16 weight rows read once a row (R reads of the 4 MiB weight a call), with 32·NRG (one read a group of up to 4 rows).
- **Bits:** `turboexl3_ling_router_logits_h`'s words. Each row keeps its own coefficients (its x words widened) and its
  own 4 sums a lane. Each weight word is read once (`ling:router/logits_wh` verbatim: widened, then + 0) and taken by
  every row of the group in that row's own statement, `result[r][tm] += inter[tn] * v_coeff[r][tn]`: the served
  launch's statement with the row's index. A row's statements and their order (the K loop, the 4 × 4 tile, the shuffle
  tree, the store) are those of its threadgroup in the served launch, so its contractions and roundings are too.
- **Pinned by:** `src/models/ling/router.zig` "turboexl3 ling router rows texts: ..." (CPU: the composition, each per-row
  statement a row's own and the served one nowhere, every other gemv statement once,
  `kernels/test/pins/router_rows.metal`) and "turboexl3 ling router rows plan: ..." (CPU: the grid, the calls it takes,
  the head's cover of each (row, tile) once); `src/models/ling/router_test.zig` "turboexl3 gpu ling router rows: ..."
  (GPU: `TURBOEXL3_LING_ROUTER_ROWS`, the logits and scores against the served launch's word for word at rows 1 to 8,
  over the f16 weight and the f32 table; red checks: a flipped word, MLX's several-row matmul, a negated weight word).

### `turboexl3_ling_router_pred_hr`

<!-- kernel-facts: turboexl3_ling_router_pred_hr -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/router.zig:414` (`fn rowsKernelOf`) |
| inputs | `x`, `w`, `pw`, `pb` |
| outputs | `logits`, `scores` |
| source | `text_pred_hr` = `ling:router/head_rows` ++ `ling:router/logits_rows_1` ++ `ling:router/logits_wh` ++ `ling:router/logits_rows_2` ++ `ling:router/pred_open_rows` ++ `ling:router/pred_one` ++ `ling:router/pred_wide` ++ `ling:router/close` (7284 B, `9bc4dd11a86f`) |
| header | `score_softplus_sqrt` = `ling:router/score_softplus_sqrt` (1499 B, `0f43a7203e4b`) |

<!-- /kernel-facts -->

- **Used by:** as `turboexl3_ling_router_logits_hr`, with a `Predictor` at its default score (`.softplus_sqrt`): in
  `turboexl3_ling_router_pred_h`'s place at a call of 2 rows or more under `Config.router_rows`.
- **Shapes and dtypes:** `turboexl3_ling_router_pred_h`'s, R 2 to 8.
- **Grid and threadgroup:** `((32·NRG + P) · 128, 1, 1)` / `(128, 1, 1)`: the router's 32·NRG threadgroups as
  `turboexl3_ling_router_logits_hr`'s, then the predictor's P as `turboexl3_ling_router_pred`'s (128 at 2 to 5 rows, 64
  at 6 to 8).
- **Template args:** R, NE, HK.
- **Replaces:** `turboexl3_ling_router_pred_h` at those calls (D5b).
- **Bits:** the logits as `turboexl3_ling_router_logits_hr`'s; the scores `turboexl3_ling_router_pred_h`'s: the
  predictor's threadgroups run `ling:router/pred_one` and `ling:router/pred_wide` verbatim, their index counted past the
  router's 32·NRG threadgroups (`ling:router/pred_open_rows`) in place of 32R.
- **Pinned by:** as `turboexl3_ling_router_logits_hr`'s (the GPU test's DeepSeek score).

### `turboexl3_ling_router_pred_sig_hr`

<!-- kernel-facts: turboexl3_ling_router_pred_sig_hr -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/router.zig:415` (`fn rowsKernelOf`) |
| inputs | `x`, `w`, `pw`, `pb` |
| outputs | `logits`, `scores` |
| source | `text_pred_hr` = `ling:router/head_rows` ++ `ling:router/logits_rows_1` ++ `ling:router/logits_wh` ++ `ling:router/logits_rows_2` ++ `ling:router/pred_open_rows` ++ `ling:router/pred_one` ++ `ling:router/pred_wide` ++ `ling:router/close` (7284 B, `9bc4dd11a86f`) |
| header | `score_sigmoid` = `ling:router/score_sigmoid` (522 B, `765e609d7b4e`) |

<!-- /kernel-facts -->

- **Used by:** as `turboexl3_ling_router_pred_hr`, with `Predictor.score = .sigmoid` (Ling's score, the plugin's
  PREDICTOR): in `turboexl3_ling_router_pred_sig_h`'s place at a call of 2 rows or more under `Config.router_rows`.
- **Shapes and dtypes:** `turboexl3_ling_router_pred_sig_h`'s, R 2 to 8.
- **Grid and threadgroup:** `turboexl3_ling_router_pred_hr`'s.
- **Template args:** R, NE, HK.
- **Replaces:** `turboexl3_ling_router_pred_sig_h` at those calls (D5b).
- **Bits:** the logits as `turboexl3_ling_router_logits_hr`'s; the scores `turboexl3_ling_router_pred_sig_h`'s (the
  predictor's pieces verbatim, as `turboexl3_ling_router_pred_hr`'s).
- **Pinned by:** as `turboexl3_ling_router_logits_hr`'s (the GPU test's Ling score).

### `turboexl3_test_ling_router_host`

<!-- kernel-facts: turboexl3_test_ling_router_host -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/router_test.zig:226` (`test turboexl3 gpu ling router: routerLogits, select and route over the loader's table (f32, transposed back) and over the loaded f16 weight give the served logits (one-row gemv over the column-major table, denseRows at 2 to 8), predictor scores (DeepSeek's and Ling's, gemv at one row and gemv_wide at 2 to 8) and ids and bf16 weights word for word at rows 1 to 8 (TURBOEXL3_GPU=1)`) |
| inputs | `logits`, `bias`, `scale` |
| outputs | `inds`, `scores` |
| source | `router.text_select` = `ling:router/select` (3430 B, `f8657dcb970c`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->

- **Used by:** test only: the reference in `src/models/ling/router_test.zig`: the host's fused router at the host's own
  config (TOUT float32), whose weights the test casts to bf16 with MLX's astype (the MoE block's).
- **Shapes and dtypes:** logits f32 [R, 512], bias f32 [512], scale f32 0-dim. Out: inds u32 [R, 8]; scores f32 [R, 8].
- **Grid and threadgroup:** `(256, R, 1)` / `(256, 1, 1)`.
- **Template args:** TOUT (float32), NE, NK, TG, NORM, NG, GSZ, TOPKG as `turboexl3_ling_router_select`'s.
- **Replaces:** nothing: a reference, the host's launch.
- **Bits:** the host's: its text (`router.text_select`) and config.
- **Pinned by:** `src/models/ling/router_test.zig` "turboexl3 gpu ling router: ..." (GPU, as the reference). No test pins
  its text beyond `text_select`'s.

## The MLA attention in the latent

Ling's own too: the MLA attention at decode rows in the latent (`mla_latent.zig`, root's `ling.mlaLatentDecode`).

`mlaLatentDecode` attends a decode or verify call's rows (1 to 16) over a layer's latent cache: each cached token's
normed KV latent (512) and roped shared key (64). The query's non-positional half is absorbed into the latent through
kv_b's key half (launch 1), the attention runs over the latent and the rope key (launch 2, the cache read once a
layer: 1,152 B a token where the expanded cache is 40,960), and the latent output leaves through kv_b's value half
(launch 3). Every value is f32 from the exact operands, so the call is within tolerance of the host's composed latent
attention (MLX's matmuls and softmax), not byte for byte. A row's values depend on its own operands and its key
position only, so a verify call's rows have the bits of one-row calls at their positions.

Launch 2 has one variant with its words, a host's opt-in in `Config` (`mla_latent.Attn.stage`;
`turboexl3_ling_mla_attn_stage` below): the cache's f32 words staged in threadgroup memory.

## `src/models/ling/mla_latent.zig`

### `turboexl3_ling_mla_attn`

<!-- kernel-facts: turboexl3_ling_mla_attn -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_latent.zig:143` (`fn kernelOf`) |
| inputs | `ql`, `qr`, `lat`, `rot`, `scale`, `dims` |
| outputs | `pm`, `pl`, `po` |
| source | `text_attn` = `ling:mla_latent/attn_1` ++ `ling:mla_latent/attn_2` ++ `ling:mla_latent/attn_3` ++ `ling:mla_latent/attn_4` (8673 B, `7d50732448a0`) |
| header | `text_header` = `ling:mla_latent/header` (175 B, `0f02dd5ef698`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `ling.mlaLatentDecode`, launch 2 of 3 (after `turboexl3_ling_mla_qabs`).
- **Shapes and dtypes:** ql f32 [S, H, 512] (launch 1's); qr [1, H, S, 64] (q_rot, roped, made row-contiguous) in QT; lat
  [1, 1, T, 512] and rot [1, 1, T, 64] in CT (the cache views after the step's append, read through `lat_strides` and
  `rot_strides`); scale f32 [1]; dims int32 [2] (T, NJ). Out: pm, pl f32 [S·H·NJ] (each row's max and sum per key group),
  po f32 [S·H·NJ, 512] (its weighted sums). Query row s sits at key position T - S + s.
- **Grid and threadgroup:** `(512, NJ, S·H/32)` / `(512, 1, 1)`: a threadgroup for each (key group of 256 keys, 32 heads of
  one row), 16 simdgroups: 4 M tiles of 8 rows, each across 4 simdgroups (a score tile of 8 keys each; 128 of the 512
  sums each). NJ = ceil(T / 256).
- **Template args:** H, S, DL 512, DR 64, KPT 256 (keys a group), CK 32 (keys a sub-chunk), QT, CT. KPT and CK are part of
  the arithmetic (fixed).
- **Replaces:** the host's composed latent scores, softmax and value product: f32 copies of the latent cache and rope keys a
  call, `matmul(q_lat, latentᵀ) + matmul(q_rope, ropeᵀ)`, the scale, the verify mask, `softmax`, `matmul(probs,
  latent)`.
- **Bits:** f32 on 8 × 8 simdgroup matrices from the exact operands: a score sums the latent's dims then the rope's; the online
  softmax runs sub-chunk by sub-chunk (the max, exp, a 16-lane butterfly sum); a key past the row's position (or the
  group) reads as 0 and scores -inf. Key groups and sub-chunks count from position 0, so a row's values depend on its
  operands and position only (a verify row has a one-row call's bits). Within tolerance of the composed path.
- **Pinned by:** `src/models/ling/mla_latent.zig` "turboexl3 ling mla latent texts: ..." (CPU: the operands, the strides, the
  position arithmetic from the pin file); `src/models/ling/mla_latent_test.zig` "turboexl3 gpu ling mla latent:
  decodeLatent at Ling's widths is within tolerance ..." (GPU: within 1e-4 of the f64 reference in f32, 1e-2 in bf16)
  and "turboexl3 gpu ling mla latent: a call's every row has the bits of a one-row call at its position, ..." (GPU,
  bytes: rows against one-row calls, strided views against contiguous),
  and "turboexl3 gpu ling mla latent: no worse than the served expanded attention ..." (GPU: bf16, the RMS error
  within 10% and the worst element within 25% of the served expanded path's, against the same f64 reference, rows 1 to
  4 at keys to 18,480).

### `turboexl3_ling_mla_qabs`

<!-- kernel-facts: turboexl3_ling_mla_qabs -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_latent.zig:142` (`fn kernelOf`) |
| inputs | `qn`, `w` |
| outputs | `ql` |
| source | `text_qabs` = `ling:mla_latent/qabs` (1347 B, `184aa655ae1b`) |
| header | `text_header` = `ling:mla_latent/header` (175 B, `0f02dd5ef698`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `ling.mlaLatentDecode` (`mla_latent.decodeLatent`), launch 1 of 3: a decode or verify
  call's rows through the MLA layers' latent cache (the outside plugin's MLX_SERVE_LING_MLA_FUSED=1 over its
  MLX_SERVE_LING_MLA_LATENT=1 cache).
- **Shapes and dtypes:** qn [1, H, S, 128] (q_nope, made row-contiguous) in QT (bf16, f16 or f32); w kv_b [H·256, 512] in WT
  (head h's key half rows h·256 .. h·256 + 127). Out: ql f32 [S, H, 512].
- **Grid and threadgroup:** `(H·128, 1, 1)` / `(32, 1, 1)`: a thread for 4 columns of one head, every row in turn.
- **Template args:** H (heads, a multiple of 32), S (rows, 1 to 16), DN 128, DV 128, DL 512, QT, WT.
- **Replaces:** the host's composed latent query: `matmul(astype(q_nope, f32), astype(Wk, f32))` (two copies of kv_b's key half
  in f32 a call).
- **Bits:** f32: each term one fma of the exact operands, d ascending from 0; a row's values its own (any rows beside it).
  Within tolerance of the host's matmul (another summation order).
- **Pinned by:** `src/models/ling/mla_latent.zig` "turboexl3 ling mla latent texts: each text reads every operand it binds and
  ..." (CPU: operands, constants, no strides); `src/models/ling/mla_latent_test.zig` "turboexl3 gpu ling mla latent:
  decodeLatent at Ling's widths is within tolerance of the host's f64 absorbed attention, ..." (GPU, the call).

### `turboexl3_ling_mla_vabs`

<!-- kernel-facts: turboexl3_ling_mla_vabs -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_latent.zig:144` (`fn kernelOf`) |
| inputs | `pm`, `pl`, `po`, `w`, `dims` |
| outputs | `out` |
| source | `text_vabs` = `ling:mla_latent/vabs` (2335 B, `d1d2a8e2e320`) |
| header | `text_header` = `ling:mla_latent/header` (175 B, `0f02dd5ef698`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `ling.mlaLatentDecode`, launch 3 of 3 (after `turboexl3_ling_mla_attn`).
- **Shapes and dtypes:** pm, pl f32 [S·H·NJ], po f32 [S·H·NJ, 512] (launch 2's); w kv_b [H·256, 512] in WT (head h's value half
  rows h·256 + 128 .. h·256 + 255); dims int32 [2] (T, NJ). Out: out [1, H, S, 128] in OT (q_nope's dtype): the
  attention's output per head, before the host's transpose.
- **Grid and threadgroup:** `(128, H, S)` / `(128, 1, 1)`: a threadgroup for each (head, row); each thread merges 4 of the 512
  latent columns, each of 4 simdgroups takes 32 of the 128 outputs.
- **Template args:** H, S, DL 512, DN 128, DV 128, KPT 256, WT, OT.
- **Replaces:** the host's composed latent output: `matmul(o_lat, astype(Wvᵀ, f32))`, `astype` to the activation dtype (and the
  softmax's normalization, here the merge's division).
- **Bits:** f32: the merge takes the key groups holding a key the row sees (j·256 <= its position), in group order, by fma;
  each output a 512-term dot (a lane's 16 columns by fma, then `simd_sum`), cast to OT. A row's values its own. Within
  tolerance of the composed path.
- **Pinned by:** `src/models/ling/mla_latent.zig` "turboexl3 ling mla latent texts: ..." (CPU: the merge's group bound from the
  pin file); `src/models/ling/mla_latent_test.zig`'s three GPU tests (the call).

### `turboexl3_ling_mla_attn_stage`

<!-- kernel-facts: turboexl3_ling_mla_attn_stage -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_latent.zig:193` (`fn variantKernel`) |
| inputs | `ql`, `qr`, `lat`, `rot`, `scale`, `dims` |
| outputs | `pm`, `pl`, `po` |
| source | `text_attn_stage` = `ling:mla_latent/stage_1` ++ `ling:mla_latent/attn_2` ++ `ling:mla_latent/stage_2` ++ `ling:mla_latent/attn_4` (9347 B, `8bed636fe06a`) |
| header | `text_header` = `ling:mla_latent/header` (175 B, `0f02dd5ef698`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `ling.mlaLatentDecode`, launch 2 of 3 under `Config.mla_latent_stage`
  (`mla_latent.Attn.stage`): the outside plugin's MTP head on its latent cache (MLA_LATENT_STAGE, kept), at the calls the
  NAX launch (`Config.mla_latent_nax`) does not take.
- **Shapes and dtypes:** `turboexl3_ling_mla_attn`'s operands and outputs.
- **Grid and threadgroup:** launch 2's, `(512, NJ, S·H/32)` / `(512, 1, 1)`, with 17 KiB more threadgroup memory: the scores'
  two slabs of 32 keys × 64 dims (68 words a key), or the values' 8 keys × 516 words. Per sub-chunk 19 barriers (launch
  2: 3): 10 for the 9 slabs, 1 after the softmax update, 8 for the 4 value stages.
- **Template args:** launch 2's. Its text names no CT: it reads `lat` and `rot` themselves, whose element type MLX's
  signature gives from the cache's dtype (the type CT names).
- **Replaces:** `turboexl3_ling_mla_attn` at those calls: each simdgroup's own reads of the cache (a latent word read and
  converted by 8 simdgroups of the threadgroup, 4 for the scores and 4 for the values).
- **Bits:** launch 2's words: a staged word is the word launch 2 reads (the cache word converted to f32, or 0 past the row's
  bound), the tiles, their operands and their order are launch 2's, and the softmax update and the outputs are launch
  2's pieces (`attn_2`, `attn_4`).
- **Pinned by:** `src/models/ling/mla_latent.zig` "turboexl3 ling mla latent variant texts: ..." (CPU: launch 2's softmax
  and output pieces once, the operands and constants, the staged words' statements from the pin file) and "turboexl3 ling
  mla latent variants plan: ..." (CPU: its grid and the calls it takes); `src/models/ling/mla_latent_test.zig` "turboexl3
  gpu ling mla latent variants: ..." (GPU, bytes: the output, pm, pl and po against launch 2's at rows 1 to 4 and keys 1
  to 19,404 on the plugin's operands and two strided caches; a planted fault differs).

### `turboexl3_ling_mla_attn_nax`

<!-- kernel-facts: turboexl3_ling_mla_attn_nax -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_latent.zig:194` (`fn variantKernel`) |
| inputs | `qlb`, `qrb`, `lat`, `rot`, `scale`, `dims` |
| outputs | `pm`, `pl`, `po` |
| source | `text_attn_nax` = `ling:mla_latent/nax` ++ `ling:mla_latent/nax_close` (10326 B, `973fcfb5fb1c`) |
| header | `header_nax` = `steel_nax` ++ `ling:mla_latent/nax_head` (32072 B, `59af53abaaa3`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `ling.mlaLatentDecode`, launch 2 of 3 under `Config.mla_latent_nax` (`mla_latent.Attn`
  `nax_served`, ahead of `mla_latent_stage`) at every call whose cache is bf16 or f16, behind the plugin's candidate switch
  `MLX_SERVE_LING_MLA_LATENT_NAX`. The plugin serves it to the MTP head alone, whose words move the drafts and never the
  verified text. Not served by default.
- **Shapes and dtypes:** launch 2's outputs (`pm`, `pl` f32 [S·H·NJ], `po` f32 [S·H·NJ, DL]); `qlb` [S, H, DL] and `qrb`
  [1, H, S, DR] contiguous in the cache's dtype (launch 1's f32 `ql` and the rope half cast once on the host); `lat`,
  `rot` bf16 or f16 views read through their strides (a key's dims as one run where both views hold them contiguous, else
  word by word). Refused (`mla_latent.naxTakes`): an f32 cache; heads not a multiple of 16·MT (32), which `planOf`
  refuses first.
- **Grid and threadgroup:** `(128·MT, NJ, S·H/(16·MT))` / `(128·MT, 1, 1)`, at MT 2 `(256, NJ, S·H/32)` / `(256, 1, 1)`:
  a threadgroup for each (key group of 256, 32 heads of a row), 4 simdgroups an M tile of 16 heads, each simdgroup 128
  of the latent's 512 columns; 2·4·16·(32 + 4) floats (18 KiB) of threadgroup memory for the partial score tiles, 2
  barriers a step.
- **Template args:** H, S, DL, DR, KPT (launch 2's); MT 2 (M tiles), BKN 32 (keys a step), SD 1 (each simdgroup scores 9
  of the 36 dim slices, the M tile's partials summed in simdgroup order): `mla_latent.nax_shape`, the one shape (D1's
  TIME: the fastest at every row count of the MTP head's calls; the other eight left in strip2); CT (the cache's dtype,
  the matrix units' operand type).
- **Replaces:** `turboexl3_ling_mla_attn_stage` (or launch 2) at those calls: the scores and the weighted sums on f32
  simdgroup matrices at 8 × 8 × 8 (about 1.35 TFLOPS at the head's calls, A1) become MLX steel's NAX tiles at
  16 × 32 × 16 into f32.
- **Bits:** not launch 2's words: the query rounds to the cache's dtype and the products sum in the matrix units' order.
  Held to a bound of the host's f64 absorbed attention (`mla_latent_test.zig` `naxBound`: the RMS error within 2^-8 of the
  reference's RMS at Ling's scale, the worst element within 4 times that, beyond the output's bf16 store). The partials
  keep launch 2's key groups from position 0 and its online softmax in natural units, so launch 3 merges them unchanged
  and a row's values depend on its own operands and position only.
- **Pinned by:** the CPU test's pin file (`kernels/test/pins/mla_latent_nax.metal`: the bound, the step's keys, the mask,
  the split's sum, the partials, the header's row op, finite minimum and attribute); the GPU test's
  `TURBOEXL3_LING_MLA_LATENT_NAX` lines (the shape and the m2_k64 fixture within the bound, the strided caches, the red
  checks, the fallbacks).

## `src/models/ling/mla_latent_test.zig`

The NAX launch's second shape, kept as a test fixture: a kernel made only by the latent tests,
`turboexl3_ling_mla_attn_nax` at 64 keys a step with the whole scores (`kernels/test/fault/`), for a red check that
plants a row-count-dependent split.

### `turboexl3_test_ling_mla_attn_nax_m2_k64`

<!-- kernel-facts: turboexl3_test_ling_mla_attn_nax_m2_k64 -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_latent_test.zig:820` (`fn fixtureM2K64`) |
| inputs | `qlb`, `qrb`, `lat`, `rot`, `scale`, `dims` |
| outputs | `pm`, `pl`, `po` |
| source | `text_nax_m2_k64` = `ling:test/fault/mla_latent_nax_m2_k64` ++ `ling:mla_latent/nax` ++ `ling:mla_latent/nax_close` (10707 B, `1ce96d46b8df`) |
| header | `ml.header_nax` = `steel_nax` ++ `ling:mla_latent/nax_head` (32072 B, `59af53abaaa3`) |

<!-- /kernel-facts -->

- **Used by:** test only: `src/models/ling/mla_latent_test.zig` `fixtureM2K64`, launched through
  `mla_latent.decodeLatentPartsOn` (the NAX launch's plan, checks and refusals on a test's kernel). The NAX GPU test
  holds it to `naxBound` beside the served shape; a red check runs a call's rows on it and each row alone on the served
  shape.
- **Shapes and dtypes:** `turboexl3_ling_mla_attn_nax`'s.
- **Grid and threadgroup:** `turboexl3_ling_mla_attn_nax`'s at MT 2 (the served shape's launch), one float of
  threadgroup memory for the partial score tiles (SD 0 sums none).
- **Template args:** `turboexl3_ling_mla_attn_nax`'s at the served shape's values; the fixture's two definitions
  (`#define BKN 64`, `#define SD 0`, `ling:test/fault/mla_latent_nax_m2_k64` ahead of the body) stand in for BKN and SD.
- **Replaces:** nothing: the shape `nax_m2_k64` was before strip2 (64 keys a step, each simdgroup scoring every slice),
  a real NAX shape that is not the served one.
- **Bits:** not the served shape's words: its scores and weighted sums group the keys 64 a step and sum every slice in
  one simdgroup; within `naxBound` of the f64 reference as the served shape is.
- **Pinned by:** `src/models/ling/mla_latent_test.zig` "turboexl3 ling mla latent nax fixture: ..." (CPU: the NAX text
  after the two definitions and nothing else); "turboexl3 gpu ling mla latent nax: ..." (GPU: the
  `TURBOEXL3_LING_MLA_LATENT_NAX` lines with shape=m2_k64_fixture, within the bound).

## `src/models/ling/mla_flash.zig`

The MLA prompt attention as one causal flash launch (`root.ling.mlaFlashPrefill`; over the rope-once cache,
`root.ling.mlaFlashPrefillRope1`).

### `turboexl3_ling_mla_flash`

<!-- kernel-facts: turboexl3_ling_mla_flash -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_flash.zig:117` (`fn kernelGet`) |
| inputs | `q`, `k`, `v`, `scale` |
| outputs | `o` |
| source | `source` = `ling:mla_flash/body_1` ++ `ling:mla_flash/body_2` ++ `ling:mla_flash/body_kofs` ++ `ling:mla_flash/body_3` ++ `ling:mla_flash/body_kload` ++ `ling:mla_flash/body_4` ++ `ling:mla_flash/body_kstep` ++ `ling:mla_flash/body_5` ++ `ling:mla_flash/close` (10242 B, `f8febea390c3`) |
| header | `header` = `steel_nax` ++ `ling:mla_flash/head` (31972 B, `30be639a8251`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.mlaFlashPrefill`, an MLA layer's prompt attention over the expanded KV (the
  outside plugin turboexl3_ling's prompt spans, behind its own switch, off by default), where `Config.nax` holds. Any
  head dims `mla_flash.admits` takes; Ling's are qk 192 (128 nope + 64 rope) and v 128 at 64 heads.
- **Shapes and dtypes:** q T [B, H, qL, BD]; k T [B, Hk, kL, BD] and v T [B, Hk, kL, BV] at their own strides (a
  cache's views read in place; each last axis contiguous), H a multiple of Hk, kL >= qL; scale f32 [1]. T bf16 or f16.
  Out: o T [B, H, qL, BV], contiguous.
- **Grid and threadgroup:** `(ceil(qL / BQ) · 32, H · WM, B · WN)` / `(32, WM, WN)`: MLX's own launch of
  `attention_nax` (threadgroups (ceil(qL / BQ), H, B)).
- **Template args:** T, BQ 64, BK 32, BD (qk), BV (v), WM 4, WN 1 (MLX's tile for head dims up to 128), ALIGN_Q and
  ALIGN_K (qL and kL whole tiles: attention_nax's align_Q and align_K), CAUSAL 1.
- **Replaces:** MLX's composed SDPA at qk 192 (`use_fallback` at prompt widths): the scaled query, Q K^T, the causal
  mask, the softmax and P V as separate launches, the scores and probabilities written to and read from device memory
  in the activation dtype. One launch, MLX 0.32.5's `attention_nax` (steel/attn/kernels/steel_attention_nax.h) less its
  mask, sinks, log-sum-exp and window arms, its parameters from the operands.
- **Bits:** can differ from the composed path (a reordering): the scores, the running row max and sum and the output
  accumulate in f32, P meets V in the NAX MMA and the output rounds once, where the composed path rounds the scaled
  query, the scores and the probabilities to the activation dtype. At a head-dim pair MLX instantiates its fused
  kernel for (qk = v = 128; qk 96, v 64) it is that kernel, word for word.
- **Pinned by:** `src/models/ling/mla_flash_test.zig` "turboexl3 gpu ling mla flash: at the head dims MLX instantiates
  its fused NAX kernel for, the kernel gives MLX's fused causal SDPA's words (TURBOEXL3_GPU=1)" (GPU, words, against
  `force_fused` SDPA, GQA and views of a wider cache included) and "... at Ling's heads (qk 192, v 128) the causal prompt
  attention is within tolerance of an f64 host reference and no worse than MLX's composed path against it
  (TURBOEXL3_GPU=1)" (GPU, 2.5e-2 beyond the output's rounding over the reference RMS, and an RMS and worst-element
  error no worse than the composed path's against the same reference, bf16 and f16; and at the served sizes, FL-1:
  Ling's 64 heads in bf16 over views of a 32,768-row cache, 4,096 query rows over 12,288 and 16,384 keys, 1,965 over
  18,349, 31 and 17 over 18,380, the reference on heads 0, 21, 42 and 63); "turboexl3 ling mla flash texts: the body is MLX's
  attention_nax statements ..." (CPU, the texts; its pin file); "... metal source: the kernel as standalone translation
  units (TURBOEXL3_METAL_DUMP=1)" (the offline compile); `src/models/ling/mla_flash.zig` "turboexl3 ling mla flash plan:
  ..." (CPU, the plan); "turboexl3 gpu ling mla flash rope once: ..." (GPU: the
  reference of `turboexl3_ling_mla_flash_r1`, every word).

### `turboexl3_ling_mla_flash_r1`

<!-- kernel-facts: turboexl3_ling_mla_flash_r1 -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_flash.zig:127` (`fn kernelGetR1`) |
| inputs | `q`, `k`, `v`, `kr`, `scale` |
| outputs | `o` |
| source | `source_r1` = `ling:mla_flash/body_1` ++ `ling:mla_flash/r1_setup` ++ `ling:mla_flash/body_2` ++ `ling:mla_flash/r1_kofs` ++ `ling:mla_flash/body_3` ++ `ling:mla_flash/r1_kload` ++ `ling:mla_flash/body_4` ++ `ling:mla_flash/body_kstep` ++ `ling:mla_flash/r1_kstep` ++ `ling:mla_flash/body_5` ++ `ling:mla_flash/close` (11012 B, `1478b201f6da`) |
| header | `header` = `steel_nax` ++ `ling:mla_flash/head` (31972 B, `30be639a8251`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `root.ling.mlaFlashPrefillRope1`, an MLA layer's prompt attention over the rope-once
  cache (the outside plugin turboexl3_ling's MLA_ROPE_ONCE candidate, off by default: kv_b's rows kept as they come, each
  head's nope keys then its values, and the roped key once a token, never expanded to every head), where
  `Config.mla_rope_once` and `Config.nax` hold.
- **Shapes and dtypes:** q T [B, H, qL, BD]; k T [B, Hk, kL, KN] (each head's nope keys), kr T [B, 1, kL, BD - KN] (the
  token's roped key, one for every head) and v T [B, Hk, kL, BV], each at its own strides (views of the cache read in
  place; each last axis contiguous), KN and BD - KN multiples of 16, H a multiple of Hk, kL >= qL; scale f32 [1]. T bf16
  or f16. Out: o T [B, H, qL, BV], contiguous.
- **Grid and threadgroup:** `turboexl3_ling_mla_flash`'s.
- **Template args:** `turboexl3_ling_mla_flash`'s, and KN (the nope dims a key reads from k; Ling's 128).
- **Replaces:** `turboexl3_ling_mla_flash` over the expanded key (each head's nope keys ++ the token's roped key, built
  and cached for every head): `text_body` with kr's batch offset and step added and the key tile's offset and load
  swapped (`ling:mla_flash/r1_setup`, `r1_kofs`, `r1_kload`, `r1_kstep`): each 16-wide slice of a key's dims is read
  from k (below KN) or kr (from KN), at that operand's row stride.
- **Bits:** the same as `turboexl3_ling_mla_flash` over the expanded key, word for word: the key tile holds the same
  words (the expanded key's slice is k's or kr's words), and every other statement is the base's in the base's order,
  so no score, sum or output changes order.
- **Pinned by:** `src/models/ling/mla_flash_test.zig` "turboexl3 gpu ling mla flash rope once: over the rope-once cache
  ... (TURBOEXL3_GPU=1)" (GPU, every word against `turboexl3_ling_mla_flash` over the expanded key and the values built
  from the same words, bf16 and f16, GQA and two batches, Ling's 64 heads and FL-1's served calls; red checks: the
  planted fault `turboexl3_test_ling_mla_flash_r1_nope_fault` and token 0's roped key negated must differ); "turboexl3
  ling mla flash rope once texts: ..." (CPU, the composition; pins/mla_flash_r1.metal); "turboexl3 ling mla flash rope
  once refusals: ..." (CPU, the shapes, dtypes and gates).

## `src/models/ling/mla_flash_test.zig`

The rope-once launch's red check: a kernel made only by its GPU byte test, `turboexl3_ling_mla_flash_r1` with the key
tile's source pinned to the nope keys (`kernels/test/fault/`), so the test shows its word comparison catches a key slice
read from the wrong operand.

### `turboexl3_test_ling_mla_flash_r1_nope_fault`

<!-- kernel-facts: turboexl3_test_ling_mla_flash_r1_nope_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_flash_test.zig:596` (`test turboexl3 gpu ling mla flash rope once: over the rope-once cache (each head's nope keys and values one row a token, the roped key once a token, read in place) the launch gives mlaFlashPrefill's words over the expanded key; a planted fault and a changed roped key differ (TURBOEXL3_GPU=1)`) |
| inputs | `q`, `k`, `v`, `kr`, `scale` |
| outputs | `o` |
| source | `text_r1_nope` = `ling:mla_flash/body_1` ++ `ling:mla_flash/r1_setup` ++ `ling:mla_flash/body_2` ++ `ling:test/fault/mla_flash_r1_kofs_nope` ++ `ling:mla_flash/body_3` ++ `ling:mla_flash/r1_kload` ++ `ling:mla_flash/body_4` ++ `ling:mla_flash/body_kstep` ++ `ling:mla_flash/r1_kstep` ++ `ling:mla_flash/body_5` ++ `ling:mla_flash/close` (11099 B, `3d0116dbe206`) |
| header | `flash.header` = `steel_nax` ++ `ling:mla_flash/head` (31972 B, `30be639a8251`) |

<!-- /kernel-facts -->

- **Used by:** test only: the red check in `src/models/ling/mla_flash_test.zig` "turboexl3 gpu ling mla flash rope
  once: ...", launched through `mla_flash.rope1On` at 100 query rows over 100 keys, bf16.
- **Shapes and dtypes:** `turboexl3_ling_mla_flash_r1`'s.
- **Grid and threadgroup:** `turboexl3_ling_mla_flash_r1`'s.
- **Template args:** `turboexl3_ling_mla_flash_r1`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_ling_mla_flash_r1` with every 16-wide slice of a key's dims read
  from k's row at k's stride (`ling:test/fault/mla_flash_r1_kofs_nope` in `r1_kofs`'s place): in the test's rope-once
  cache, the rope dims read the head's values that follow its nope keys.
- **Bits:** wrong on purpose: the keys' rope dims are another operand's words.
- **Pinned by:** `src/models/ling/mla_flash_test.zig` "turboexl3 gpu ling mla flash rope once: ..." (GPU:
  `TURBOEXL3_LING_FLASH_ROPE1_RED` check=nope_fault, which must differ from the served words).

## `src/models/ling/swiglu.zig`

The prefill SwiGLU epilogue: at a prompt's rows the up projection's finish runs the host's SwiGLU of the gate's
finished value and its own, and stores the activation where the composed chain stores up. Rows 1 to 16 (the decode
GEMVs), the fused-GEMM option, pg2 and an un-rotated resident trellis keep the chain (`swiglu.takes`).

### `turboexl3_ling_finish_sgu`

<!-- kernel-facts: turboexl3_ling_finish_sgu -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/swiglu.zig:61` (`fn kernelOf`) |
| inputs | `z`, `svh`, `gate`, `sigtab` |
| outputs | `y` |
| source | `text_finish_sgu` = `dense/finish_v` ++ `ling:swiglu/epi_open` ++ `ling:shared/sgu_swiglu` ++ `ling:swiglu/epi_close` (1515 B, `79dc32c6f4d5`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** `upSwiglu` (`root.lingUpSwiglu`) at limit 0, the outside plugin turboexl3_ling's MLX_SERVE_LING_SWIGLU_EPI:
  the up projection of the shared expert (its unclamped layers) and of the dense layers' MLP at prompt rows (17 and up).
- **Shapes and dtypes:** z f16 [ROWS, DIM] (up's inner product by `linear`'s own launches at these rows), svh f16 [DIM],
  gate T [ROWS, DIM] (the gate's `linear`), sigtab T [65536] (the host's table: MLX's sigmoid of every 16-bit pattern of
  T, in T). Out: y T [ROWS, DIM], the activation h. T bf16 or f16.
- **Grid and threadgroup:** `((DIM / 128) · 32, ROWS, 1)` / `(32, 1, 1)`: `turboexl3_dense_finish`'s, one simdgroup per
  (128-column block, row).
- **Template args:** DIM, OUT_T, as `turboexl3_dense_finish`'s.
- **Replaces:** up's `turboexl3_dense_finish` and the host's SwiGLU launch (mlx-serve's fusedSwiGLU): up's store, and the
  SwiGLU's reads of the gate and up and its store.
- **Bits:** up's value is text_finish's (`dense/finish_v`, then its store expression, kept in a register); h is
  fusedSwiGLU's statements (`ling:shared/sgu_swiglu`: the sigmoid read from the host's table by the gate's pattern, then
  MLX's Multiply twice in T), so the composed chain's words.
- **Pinned by:** `src/models/ling/swiglu_test.zig` "turboexl3 ling swiglu texts: ..." (CPU: the composition, the kept
  store expression and index, the gate's read and h's store, `kernels/test/pins/swiglu_texts.metal`) and "turboexl3 gpu
  ling swiglu epi: ..." (GPU: h against the composed chain word for word at the shared expert's and the dense layers'
  shapes, rows 17 to 16384).

### `turboexl3_ling_finish_sguco`

<!-- kernel-facts: turboexl3_ling_finish_sguco -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/swiglu.zig:62` (`fn kernelOf`) |
| inputs | `z`, `svh`, `gate`, `sig32`, `lim` |
| outputs | `y` |
| source | `text_finish_sguco` = `dense/finish_v` ++ `ling:swiglu/epi_open` ++ `ling:shared/sgu_swiglu_clamp_out` ++ `ling:swiglu/epi_close` (2061 B, `56844ead9ac6`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** `upSwiglu` at a limit L > 0: the shared expert's up projection on the clamped layers (Ling-3.1's
  `share_expert_swiglu_limit_list`: layers 64 to 71 at L 10).
- **Shapes and dtypes:** `turboexl3_ling_finish_sgu`'s, with sig32 f32 [65536] (MLX's f32 sigmoid of every 16-bit pattern
  of T taken to f32) in place of sigtab, and lim f32 [1] (L).
- **Grid and threadgroup:** `turboexl3_ling_finish_sgu`'s.
- **Template args:** DIM, OUT_T.
- **Replaces:** up's `turboexl3_dense_finish` and the plugin's swigluClampOut, MLX's ops in f32 (astype twice, sigmoid,
  multiply, minimum, clip, multiply, astype).
- **Bits:** up's value as `turboexl3_ling_finish_sgu`'s; h is swigluClampOut's statements in f32
  (`ling:shared/sgu_swiglu_clamp_out`: MLX's sigmoid read from the table by the gate's pattern, Multiply, Minimum with L,
  clip to [-L, L], Multiply, back in T).
- **Pinned by:** as `turboexl3_ling_finish_sgu`'s, at limit 10 (the shared expert's shape).

## `src/models/ling/rows_test.zig`

The red checks' planted faults: kernels made only by the routed chain's variant byte test, each a variant's text with
one piece swapped for a fault (`kernels/test/fault/`), so the test shows its byte comparison catches a kernel-level fault
of each kind it guards: a value one tile computes wrong and a reduction whose order alone changed (the finish's H128,
the GEMM's K loop).

### `turboexl3_test_pf_dfin_fault`

<!-- kernel-facts: turboexl3_test_pf_dfin_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/rows_test.zig:811` (`test turboexl3 gpu ling rows variants: the routed chain's fused finish (over both tables at the served options, and at each other stage shape it takes) is the served engine's words on synthetic calls at every row bucket, n 40 and 32, limits 0 and 10; planted faults are caught (TURBOEXL3_GPU=1)`) |
| inputs | `tiles`, `tiles2`, `x0`, `w0`, `xs0`, `order`, `scores`, `svh_d` |
| outputs | `y` |
| source | `comptime prefill.dfinText(prefill.dfin_gemm_4,@embedFile("kernels/test/fault/dfin_4.metal"))` = `prefill/gemm_1` ++ `prefill/gemm_dfin_bs` ++ `prefill/gemm_dfin_decl` ++ `prefill/gemm_3_1` ++ `prefill/gemm_dfin_tile` ++ `prefill/gemm_3_3` ++ `prefill/gemm_op1_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op1_xsz_1` ++ `prefill/gemm_dfin_epi_1` ++ `prefill/finish_2` ++ `prefill/gemm_dfin_load` ++ `ling:test/fault/dfin_4` ++ `prefill/gemm_dfin_epi_2` (10650 B, `874011a6a65a`) |
| header | `prefill.header_nax_dfin` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` ++ `prefill/h128` (63060 B, `d36286083098`) |

<!-- /kernel-facts -->

- **Used by:** test only: the red check in `src/models/ling/rows_test.zig` "turboexl3 gpu ling rows variants: ...",
  swapped into a `Prefill`'s `dfin_k` for one call.
- **Shapes and dtypes:** `turboexl3_pf_dfin`'s.
- **Grid and threadgroup:** `turboexl3_pf_dfin`'s (`Prefill.gemmDFin` launches it).
- **Template args:** `turboexl3_pf_dfin`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_pf_dfin` with tile 0's products left unrounded before the score
  (`ling:test/fault/dfin_4` in `finish_4`'s place).
- **Bits:** wrong on purpose: tile 0's rows of y differ from the chain's.
- **Pinned by:** `src/models/ling/rows_test.zig` "turboexl3 gpu ling rows variants: ..." (GPU: its y must differ from
  the served engine's, `TURBOEXL3_LING_ROWS_RED` fault=dfin_tile0_unrounded).

### `turboexl3_test_pf_dfin_order_fault`

<!-- kernel-facts: turboexl3_test_pf_dfin_order_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/rows_test.zig:842` (`test turboexl3 gpu ling rows variants: the routed chain's fused finish (over both tables at the served options, and at each other stage shape it takes) is the served engine's words on synthetic calls at every row bucket, n 40 and 32, limits 0 and 10; planted faults are caught (TURBOEXL3_GPU=1)`) |
| inputs | `tiles`, `tiles2`, `x0`, `w0`, `xs0`, `order`, `scores`, `svh_d` |
| outputs | `y` |
| source | `comptime prefill.dfinText(prefill.dfin_gemm_4,@embedFile("kernels/test/fault/dfin_4_order.metal"))` = `prefill/gemm_1` ++ `prefill/gemm_dfin_bs` ++ `prefill/gemm_dfin_decl` ++ `prefill/gemm_3_1` ++ `prefill/gemm_dfin_tile` ++ `prefill/gemm_3_3` ++ `prefill/gemm_op1_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op1_xsz_1` ++ `prefill/gemm_dfin_epi_1` ++ `prefill/finish_2` ++ `prefill/gemm_dfin_load` ++ `ling:test/fault/dfin_4_order` ++ `prefill/gemm_dfin_epi_2` (11082 B, `1719a4c95583`) |
| header | `prefill.header_nax_dfin` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` ++ `prefill/h128` (63060 B, `d36286083098`) |

<!-- /kernel-facts -->

- **Used by:** test only: the red check in `src/models/ling/rows_test.zig` "turboexl3 gpu ling rows variants: ...",
  swapped into a `Prefill`'s `dfin_k` for one call.
- **Shapes and dtypes:** `turboexl3_pf_dfin`'s.
- **Grid and threadgroup:** `turboexl3_pf_dfin`'s (`Prefill.gemmDFin` launches it).
- **Template args:** `turboexl3_pf_dfin`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_pf_dfin` with tile 0's H128 run with its two in-lane stages before
  its five shuffle stages (`ling:test/fault/dfin_4_order` in `finish_4`'s place): the same transform in exact
  arithmetic, its f32 sums in another order.
- **Bits:** wrong on purpose: tile 0's rows of y differ from the chain's wherever the reassociated sums round to another
  f16 word.
- **Pinned by:** `src/models/ling/rows_test.zig` "turboexl3 gpu ling rows variants: ..." (GPU: its y must differ from
  the served engine's, `TURBOEXL3_LING_ROWS_RED` fault=dfin_tile0_h128_reordered).

### `turboexl3_test_pf_dfin_korder_fault`

<!-- kernel-facts: turboexl3_test_pf_dfin_korder_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/rows_test.zig:859` (`test turboexl3 gpu ling rows variants: the routed chain's fused finish (over both tables at the served options, and at each other stage shape it takes) is the served engine's words on synthetic calls at every row bucket, n 40 and 32, limits 0 and 10; planted faults are caught (TURBOEXL3_GPU=1)`) |
| inputs | `tiles`, `tiles2`, `x0`, `w0`, `xs0`, `order`, `scores`, `svh_d` |
| outputs | `y` |
| source | `comptime prefill.dfinText(@embedFile("kernels/test/fault/gemm_4_krev.metal"),prefill.dfin_finish_4)` = `prefill/gemm_1` ++ `prefill/gemm_dfin_bs` ++ `prefill/gemm_dfin_decl` ++ `prefill/gemm_3_1` ++ `prefill/gemm_dfin_tile` ++ `prefill/gemm_3_3` ++ `prefill/gemm_op1_wx` ++ `ling:test/fault/gemm_4_krev` ++ `prefill/gemm_op1_xsz_1` ++ `prefill/gemm_dfin_epi_1` ++ `prefill/finish_2` ++ `prefill/gemm_dfin_load` ++ `prefill/finish_4` ++ `prefill/gemm_dfin_epi_2` (10899 B, `0b54f06dbfac`) |
| header | `prefill.header_nax_dfin` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` ++ `prefill/h128` (63060 B, `d36286083098`) |

<!-- /kernel-facts -->

- **Used by:** test only: the red check in `src/models/ling/rows_test.zig` "turboexl3 gpu ling rows variants: ...",
  swapped into a `Prefill`'s `dfin_k` for one call.
- **Shapes and dtypes:** `turboexl3_pf_dfin`'s.
- **Grid and threadgroup:** `turboexl3_pf_dfin`'s (`Prefill.gemmDFin` launches it).
- **Template args:** `turboexl3_pf_dfin`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_pf_dfin` with tile 0's K stages taken in reverse order, stage
  STAGES - 1 first (`ling:test/fault/gemm_4_krev` in `gemm_4`'s place): the same products, its f32 accumulators
  summed in another order.
- **Bits:** wrong on purpose: tile 0's rows of y differ from the chain's wherever the reordered sums round to another
  f16 word.
- **Pinned by:** `src/models/ling/rows_test.zig` "turboexl3 gpu ling rows variants: ..." (GPU: its y must differ from
  the served engine's, `TURBOEXL3_LING_ROWS_RED` fault=dfin_tile0_k_reversed).

## `src/models/ling/tail_test.zig`

The wide tail's red check's planted fault: a kernel made only by the wide tail's byte test.

### `turboexl3_test_ling_moe_tail_wide_tgsum_rev_fault`

<!-- kernel-facts: turboexl3_test_ling_moe_tail_wide_tgsum_rev_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/tail_test.zig:713` (`test turboexl3 gpu ling tail wide: moeTailWide gives the served chain's words at prompt widths (the combine over the call's JOINLESS parts, the block's astype and shared add, MLX's add and fast rms_norm) at the standard request's spans 4096 x 4, 1965 and 31 over 24 merged sources, f16 and f32; a planted reordering of the norm shows, and its refusals (TURBOEXL3_GPU=1)`) |
| inputs | `H`, `s0`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`, `s7`, `s8`, `s9`, `s10`, `s11`, `s12`, `s13`, `s14`, `s15`, `s16`, `s17`, `s18`, `s19`, `s20`, `s21`, `s22`, `s23`, `loc`, `WT`, `XS`, `WE` |
| outputs | `H_NEW`, `HN` |
| source | `text_wide_tgsum_rev` = `ling:tail/tail_we` ++ `ling:tail/tail_1` ++ `ling:tail/tail_delta_wide_h` ++ `ling:test/fault/tail_2_tgsum_rev` (3297 B, `2c3777a337b8`) |
| header | `tail.header_wide_h` = `ling:combine/pick_h` (1444 B, `cbbf84fad66f`) |

<!-- /kernel-facts -->

- **Used by:** the red check only: a planted fault, never served.
- **Shapes and dtypes:** `turboexl3_ling_moe_tail_wide_h`'s.
- **Grid and threadgroup:** `turboexl3_ling_moe_tail_wide_h`'s.
- **Template args:** `turboexl3_ling_moe_tail_wide_h`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_ling_moe_tail_wide_h` with the threadgroup's 32 simdgroup sums
  added one by one in reverse simdgroup order in place of the simd sum over them (`ling:test/fault/tail_2_tgsum_rev` in
  `tail_2`'s place): the same sum of squares in another order.
- **Bits:** wrong on purpose: the residual keeps the chain's words, the normed rows must not (the reordered sum moves
  some rows' rsqrt by an ulp, which turns about one bf16 word in 65,536 of those rows, so the check runs the whole
  call's 18,380 rows). A thread's own four squares in reverse order are no such fault: four squares of bf16 words round
  at most once in practice, and that reversal changed none of 16,777,216 normed words.
- **Pinned by:** `src/models/ling/tail_test.zig` "turboexl3 gpu ling tail wide: ..." (GPU: `TURBOEXL3_LING_PF_TAIL_RED`
  fault=tgsum_rev, whose normed rows must differ from the chain's and whose residual must not).

### `turboexl3_test_ling_kda_gate`

<!-- kernel-facts: turboexl3_test_ling_kda_gate -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj_gate_test.zig:109` (`fn servedGate`) |
| inputs | `A_log`, `a`, `dt_bias`, `lb`, `n_total` |
| outputs | `g` |
| source | `@embedFile("kernels/test/copy/ling_kda_gate.metal")` = `ling:test/copy/ling_kda_gate` (1271 B, `de6533c19f38`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->

- **Used by:** test only: `servedGate` in `src/models/ling/kdaproj_gate_test.zig`, the reference the gate launches are
  held to, and its timing's served arm.
- **Shapes and dtypes:** the plugin's: A_log and dt_bias f32 [W], a [n_total] in the activation dtype (bf16 here), lb
  f32 and n_total int scalars. Out: g f32 of a's shape.
- **Grid and threadgroup:** `(ceil(n_total / 256) · 256, 1, 1)` / `(256, 1, 1)`: one thread an element, as the plugin
  launches it.
- **Template args:** W (the channels: a's last dimension).
- **Replaces:** nothing: a verbatim copy of the outside plugin's `ling_kda_gate` (turboexl3_ling-port 165d8ca,
  overlay/mlx_serve/kernels/kda/ling_kda_gate.metal) under a `//! copied_from:` line; never served.
- **Bits:** the plugin's gate's (the ops of mlx-serve's `kdaGateChain` in their order, each MLX's own formula).
- **Pinned by:** `src/models/ling/kdaproj_pf.zig` "turboexl3 ling kdaproj gate statements: ..." (CPU: its seven
  statements, `kernels/test/pins/kdaproj_gate.metal`); the plugin pins the copy to its own file.

### `turboexl3_test_ling_kdaproj_pfg1_ea_shift_fault`

<!-- kernel-facts: turboexl3_test_ling_kdaproj_pfg1_ea_shift_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj_gate_test.zig:269` (`test turboexl3 gpu ling kdaproj gate: kdaProjPromptGate's y_0 and y_2 are kdaProjPrompt's and its g the served one-launch gate of kdaProjPrompt's y_1, word for word, G0 and G1, at prompt rows 33 to 16384; kdaProjPromptWGate's the same over given W_hats; its declines and refusals; planted faults caught (TURBOEXL3_GPU=1)`) |
| inputs | `x0`, `x1`, `x2`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `gea`, `dt_bias`, `lb` |
| outputs | `y0`, `y1`, `y2` |
| source | `comptime faultText(@embedFile("kernels/test/fault/kdagate_g1_ea_shift.metal"))` = `ling:kdaproj/pf_1` ++ `pg2/gemm_fused_2` ++ `ling:kdaproj/pf_3_1` ++ `ling:kdaproj/pf_3_2g` ++ `pg2/gemm_fused_4` ++ `ling:kdaproj/pf_5` ++ `pg2/gemm_fused_6_1` ++ `ling:test/fault/kdagate_g1_ea_shift` ++ `ling:kdaproj/pf_undef` (6087 B, `2b7ed2d0dfbb`) |
| header | `pgemm.header` = `steel_nax` ++ text 'using namespace mlx::steel;\n' ++ `pg2/gemm_loop` ++ text (72 B) (34287 B, `0ab3ef8c4fca`) |

<!-- /kernel-facts -->

- **Used by:** test only: the red checks in `src/models/ling/kdaproj_gate_test.zig` "turboexl3 gpu ling kdaproj gate:
  kdaProjPromptGate's y_0 and y_2 are ...", swapped into the gate launches for one call
  (`kdaproj_pf.gate_kernel_override`).
- **Shapes and dtypes:** `turboexl3_ling_kdaproj_pfg1`'s.
- **Grid and threadgroup:** `turboexl3_ling_kdaproj_pfg1`'s.
- **Template args:** `turboexl3_ling_kdaproj_pfg1`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_ling_kdaproj_pfg1` with gea read at the next channel
  (`ling:test/fault/kdagate_g1_ea_shift` in the store loop's place).
- **Bits:** wrong on purpose: member 1's g differs from the served gate's.
- **Pinned by:** `src/models/ling/kdaproj_gate_test.zig` "turboexl3 gpu ling kdaproj gate: kdaProjPromptGate's y_0 and
  y_2 are ..." (GPU: its g must differ from the served gate's over `kdaProjPrompt`'s y_1,
  `TURBOEXL3_LING_KDA_GATE_RED`).

### `turboexl3_test_ling_kdaproj_pfg_dtb_shift_fault`

<!-- kernel-facts: turboexl3_test_ling_kdaproj_pfg_dtb_shift_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj_gate_test.zig:266` (`test turboexl3 gpu ling kdaproj gate: kdaProjPromptGate's y_0 and y_2 are kdaProjPrompt's and its g the served one-launch gate of kdaProjPrompt's y_1, word for word, G0 and G1, at prompt rows 33 to 16384; kdaProjPromptWGate's the same over given W_hats; its declines and refusals; planted faults caught (TURBOEXL3_GPU=1)`) |
| inputs | `x0`, `x1`, `x2`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `A_log`, `dt_bias`, `lb` |
| outputs | `y0`, `y1`, `y2` |
| source | `comptime faultText(@embedFile("kernels/test/fault/kdagate_dtb_shift.metal"))` = `ling:kdaproj/pf_1` ++ `pg2/gemm_fused_2` ++ `ling:kdaproj/pf_3_1` ++ `ling:kdaproj/pf_3_2g` ++ `pg2/gemm_fused_4` ++ `ling:kdaproj/pf_5` ++ `pg2/gemm_fused_6_1` ++ `ling:test/fault/kdagate_dtb_shift` ++ `ling:kdaproj/pf_undef` (6133 B, `99ce6389cee5`) |
| header | `pgemm.header` = `steel_nax` ++ text 'using namespace mlx::steel;\n' ++ `pg2/gemm_loop` ++ text (72 B) (34287 B, `0ab3ef8c4fca`) |

<!-- /kernel-facts -->

- **Used by:** test only: the red checks in `src/models/ling/kdaproj_gate_test.zig` "turboexl3 gpu ling kdaproj gate:
  kdaProjPromptGate's y_0 and y_2 are ...", swapped into the gate launches for one call
  (`kdaproj_pf.gate_kernel_override`).
- **Shapes and dtypes:** `turboexl3_ling_kdaproj_pfg0`'s.
- **Grid and threadgroup:** `turboexl3_ling_kdaproj_pfg0`'s.
- **Template args:** `turboexl3_ling_kdaproj_pfg0`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_ling_kdaproj_pfg0` with dt_bias read at the next channel
  (`ling:test/fault/kdagate_dtb_shift` in the store loop's place).
- **Bits:** wrong on purpose: member 1's g differs from the served gate's.
- **Pinned by:** `src/models/ling/kdaproj_gate_test.zig` "turboexl3 gpu ling kdaproj gate: kdaProjPromptGate's y_0 and
  y_2 are ..." (GPU: its g must differ from the served gate's over `kdaProjPrompt`'s y_1,
  `TURBOEXL3_LING_KDA_GATE_RED`).

### `turboexl3_test_ling_kdaproj_pfg_fastexp_fault`

<!-- kernel-facts: turboexl3_test_ling_kdaproj_pfg_fastexp_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj_gate_test.zig:267` (`test turboexl3 gpu ling kdaproj gate: kdaProjPromptGate's y_0 and y_2 are kdaProjPrompt's and its g the served one-launch gate of kdaProjPrompt's y_1, word for word, G0 and G1, at prompt rows 33 to 16384; kdaProjPromptWGate's the same over given W_hats; its declines and refusals; planted faults caught (TURBOEXL3_GPU=1)`) |
| inputs | `x0`, `x1`, `x2`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `A_log`, `dt_bias`, `lb` |
| outputs | `y0`, `y1`, `y2` |
| source | `comptime faultText(@embedFile("kernels/test/fault/kdagate_fastexp.metal"))` = `ling:kdaproj/pf_1` ++ `pg2/gemm_fused_2` ++ `ling:kdaproj/pf_3_1` ++ `ling:kdaproj/pf_3_2g` ++ `pg2/gemm_fused_4` ++ `ling:kdaproj/pf_5` ++ `pg2/gemm_fused_6_1` ++ `ling:test/fault/kdagate_fastexp` ++ `ling:kdaproj/pf_undef` (6109 B, `c30c4c5766ee`) |
| header | `pgemm.header` = `steel_nax` ++ text 'using namespace mlx::steel;\n' ++ `pg2/gemm_loop` ++ text (72 B) (34287 B, `0ab3ef8c4fca`) |

<!-- /kernel-facts -->

- **Used by:** test only: the red checks in `src/models/ling/kdaproj_gate_test.zig` "turboexl3 gpu ling kdaproj gate:
  kdaProjPromptGate's y_0 and y_2 are ...", swapped into the gate launches for one call
  (`kdaproj_pf.gate_kernel_override`).
- **Shapes and dtypes:** `turboexl3_ling_kdaproj_pfg0`'s.
- **Grid and threadgroup:** `turboexl3_ling_kdaproj_pfg0`'s.
- **Template args:** `turboexl3_ling_kdaproj_pfg0`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_ling_kdaproj_pfg0` with the sigmoid's exp on the fast path
  (`ling:test/fault/kdagate_fastexp` in the store loop's place).
- **Bits:** wrong on purpose: member 1's g differs from the served gate's.
- **Pinned by:** `src/models/ling/kdaproj_gate_test.zig` "turboexl3 gpu ling kdaproj gate: kdaProjPromptGate's y_0 and
  y_2 are ..." (GPU: its g must differ from the served gate's over `kdaProjPrompt`'s y_1,
  `TURBOEXL3_LING_KDA_GATE_RED`).

### `turboexl3_test_ling_kdaproj_pfg_member2_fault`

<!-- kernel-facts: turboexl3_test_ling_kdaproj_pfg_member2_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj_gate_test.zig:268` (`test turboexl3 gpu ling kdaproj gate: kdaProjPromptGate's y_0 and y_2 are kdaProjPrompt's and its g the served one-launch gate of kdaProjPrompt's y_1, word for word, G0 and G1, at prompt rows 33 to 16384; kdaProjPromptWGate's the same over given W_hats; its declines and refusals; planted faults caught (TURBOEXL3_GPU=1)`) |
| inputs | `x0`, `x1`, `x2`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `A_log`, `dt_bias`, `lb` |
| outputs | `y0`, `y1`, `y2` |
| source | `comptime faultText(@embedFile("kernels/test/fault/kdagate_member2.metal"))` = `ling:kdaproj/pf_1` ++ `pg2/gemm_fused_2` ++ `ling:kdaproj/pf_3_1` ++ `ling:kdaproj/pf_3_2g` ++ `pg2/gemm_fused_4` ++ `ling:kdaproj/pf_5` ++ `pg2/gemm_fused_6_1` ++ `ling:test/fault/kdagate_member2` ++ `ling:kdaproj/pf_undef` (6129 B, `ebad74cab6c9`) |
| header | `pgemm.header` = `steel_nax` ++ text 'using namespace mlx::steel;\n' ++ `pg2/gemm_loop` ++ text (72 B) (34287 B, `0ab3ef8c4fca`) |

<!-- /kernel-facts -->

- **Used by:** test only: the red checks in `src/models/ling/kdaproj_gate_test.zig` "turboexl3 gpu ling kdaproj gate:
  kdaProjPromptGate's y_0 and y_2 are ...", swapped into the gate launches for one call
  (`kdaproj_pf.gate_kernel_override`).
- **Shapes and dtypes:** `turboexl3_ling_kdaproj_pfg0`'s.
- **Grid and threadgroup:** `turboexl3_ling_kdaproj_pfg0`'s.
- **Template args:** `turboexl3_ling_kdaproj_pfg0`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_ling_kdaproj_pfg0` with the gate on member 2's tiles and member 1
  stored plain (`ling:test/fault/kdagate_member2` in the store loop's place).
- **Bits:** wrong on purpose: member 1's g differs from the served gate's.
- **Pinned by:** `src/models/ling/kdaproj_gate_test.zig` "turboexl3 gpu ling kdaproj gate: kdaProjPromptGate's y_0 and
  y_2 are ..." (GPU: its g must differ from the served gate's over `kdaProjPrompt`'s y_1,
  `TURBOEXL3_LING_KDA_GATE_RED`).

### `turboexl3_test_ling_kdaproj_pfg_unrounded_fault`

<!-- kernel-facts: turboexl3_test_ling_kdaproj_pfg_unrounded_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj_gate_test.zig:265` (`test turboexl3 gpu ling kdaproj gate: kdaProjPromptGate's y_0 and y_2 are kdaProjPrompt's and its g the served one-launch gate of kdaProjPrompt's y_1, word for word, G0 and G1, at prompt rows 33 to 16384; kdaProjPromptWGate's the same over given W_hats; its declines and refusals; planted faults caught (TURBOEXL3_GPU=1)`) |
| inputs | `x0`, `x1`, `x2`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `A_log`, `dt_bias`, `lb` |
| outputs | `y0`, `y1`, `y2` |
| source | `comptime faultText(@embedFile("kernels/test/fault/kdagate_unrounded.metal"))` = `ling:kdaproj/pf_1` ++ `pg2/gemm_fused_2` ++ `ling:kdaproj/pf_3_1` ++ `ling:kdaproj/pf_3_2g` ++ `pg2/gemm_fused_4` ++ `ling:kdaproj/pf_5` ++ `pg2/gemm_fused_6_1` ++ `ling:test/fault/kdagate_unrounded` ++ `ling:kdaproj/pf_undef` (6127 B, `f7f5df3c7278`) |
| header | `pgemm.header` = `steel_nax` ++ text 'using namespace mlx::steel;\n' ++ `pg2/gemm_loop` ++ text (72 B) (34287 B, `0ab3ef8c4fca`) |

<!-- /kernel-facts -->

- **Used by:** test only: the red checks in `src/models/ling/kdaproj_gate_test.zig` "turboexl3 gpu ling kdaproj gate:
  kdaProjPromptGate's y_0 and y_2 are ...", swapped into the gate launches for one call
  (`kdaproj_pf.gate_kernel_override`).
- **Shapes and dtypes:** `turboexl3_ling_kdaproj_pfg0`'s.
- **Grid and threadgroup:** `turboexl3_ling_kdaproj_pfg0`'s.
- **Template args:** `turboexl3_ling_kdaproj_pfg0`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_ling_kdaproj_pfg0` with member 1's word kept at its half round, not
  rounded to OUT_T, before the gate (`ling:test/fault/kdagate_unrounded` in the store loop's place).
- **Bits:** wrong on purpose: member 1's g differs from the served gate's.
- **Pinned by:** `src/models/ling/kdaproj_gate_test.zig` "turboexl3 gpu ling kdaproj gate: kdaProjPromptGate's y_0 and
  y_2 are ..." (GPU: its g must differ from the served gate's over `kdaProjPrompt`'s y_1,
  `TURBOEXL3_LING_KDA_GATE_RED`).

## `src/models/ling/kdanorm.zig`

L5 (KDA_NORM_PREP): a KDA layer's gated norm at prompt rows written straight into o_proj's prepared rows: dense's
prepare (`turboexl3_dense_prep`'s text, cut at its load) with the host's fused norm-gate's statements in the load's
place.

### `turboexl3_ling_kdanorm_prep`

<!-- kernel-facts: turboexl3_ling_kdanorm_prep -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdanorm.zig:47` (`fn kernelGet`) |
| inputs | `y`, `z`, `norm_w`, `eps`, `sigtab`, `suh` |
| outputs | `xp` |
| source | `text_normprep` = `dense/prep_1` ++ `ling:kdanorm/gate` ++ `dense/prep_3` (2098 B, `c70564d5b441`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->
- **Used by:** `kdanorm.kdaNormGatePrep` (`ling.kdaNormGatePrep`): the outside plugin turboexl3_ling's KDA_NORM_PREP
  (off by default), a KDA layer's gated norm at prompt rows (the host's fused norm-gate at its table, TAB 1, with the
  sigmoid gate) written as o_proj's prepared rows, its matmul and finish then `Dense.piecePrepared` and
  `Dense.finishOf`, where o_proj's `linear` runs its matmul path (`kdanorm.takes`).
- **Shapes and dtypes:** y T [B, S, HV, DV] (bf16 or f16; DV 128), the KDA recurrence's output; z T [B, S, ZSTRIDE]
  (head h's gate at ZOFF + h·DV); norm_w T [DV]; eps f32 (0-d); sigtab T [65536] (MLX's sigmoid of every 16-bit pattern
  of T, the host's table); suh f16 [HV·DV] (o_proj's). Out: xp f16 [B·S, HV·DV], natural order.
- **Grid and threadgroup:** `(HV · 32, B·S, 1)` / `(32, 1, 1)`: `turboexl3_dense_prep`'s at IN = HV·DV, a simdgroup each
  (row, 128-block); the block is the head (DV 128), the host's simdgroup for the same (row, head).
- **Template args:** T (y's dtype), DIM (HV·DV), LANE_ORDER 0, HV, DV (128), ZSTRIDE, ZOFF.
- **Replaces:** the host's fused norm-gate (mlx-serve `gdnNormGateFusedTab`, its `mlxserve_gdn_normgate` at TAB 1) and
  o_proj's prepare (`turboexl3_dense_prep` in its `linear`): one launch for two, and the norm-gate's output (T [B·S,
  HV·DV]) never reaches device memory.
- **Bits:** xp is `turboexl3_dense_prep`'s of the host's output, word for word: the head's sum of squares is the host's
  statements in its order (each lane's four consecutive channels, simd_sum, the precise rsqrt), and each value the
  prepare loads (channel L + 32 j) is the host's per-element statements on that element (normed, the table's sigmoid of
  z, their product in T), then the prepare's f16 of it times suh.
- **Pinned by:** `src/models/ling/kdanorm.zig` "turboexl3 ling kdanorm texts: ..." (CPU: the composition, and
  text_prep's bytes kept: its length and sha256) and "turboexl3 ling kdanorm statements: ..." (CPU: the gate's
  statements against the host's copy, `kernels/test/pins/kdanorm.metal`); `src/models/ling/kdanorm_test.zig` "turboexl3
  gpu ling kdanorm: kdaNormGatePrep's xp is o_proj's prepare of the served norm-gate's output and ..." (GPU: xp against
  `Dense.prepare` of the served norm-gate's output and o_proj's y against `root.linear` of it, every word, rows 33 to
  16384; `TURBOEXL3_LING_KDA_NORM`).

## `src/models/ling/kdanorm_test.zig`

The served chain the fused launch is held to (the host's norm-gate, its test-only copy) and the red checks' planted
faults.

### `turboexl3_test_ling_gdn_normgate`

<!-- kernel-facts: turboexl3_test_ling_gdn_normgate -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdanorm_test.zig:64` (`fn servedNormGate`) |
| inputs | `y`, `z`, `norm_w`, `eps`, `sigtab` |
| outputs | `out` |
| source | `@embedFile("kernels/test/copy/gdn_normgate.metal")` = `ling:test/copy/gdn_normgate` (1175 B, `852e385274a4`) |
| header | `""` = (empty) (0 B, `e3b0c44298fc`) |

<!-- /kernel-facts -->
- **Used by:** test only: `servedNormGate` in `src/models/ling/kdanorm_test.zig`, the served chain's first launch the
  fused launch is held to, and its timing's served arm.
- **Shapes and dtypes:** the host's: y T [B, S, HV, DV], z T [B, S, ZSTRIDE], norm_w T [DV], eps f32 (0-d), sigtab T
  [65536]. Out: out T [B, S, HV·DV].
- **Grid and threadgroup:** `(32, B·S, HV)` / `(32, 1, 1)`: a simdgroup each (row, head), as the host launches it.
- **Template args:** T, HV, DV, ZSTRIDE, ZOFF, TAB (1 here), SWISH (0 here: the sigmoid gate), in the host's order.
- **Replaces:** nothing: a verbatim copy of mlx-serve's `kernels/gdn/gdn_normgate.metal` at bc6c5f2c (the outside plugin
  turboexl3_ling's MLX_SERVE_REF) under a `//! copied_from:` comment line; never served.
- **Bits:** the host's norm-gate's.
- **Pinned by:** `src/models/ling/kdanorm.zig` "turboexl3 ling kdanorm statements: ..." (CPU: its statements against the
  served piece's, `kernels/test/pins/kdanorm.metal`).

### `turboexl3_test_ling_kdanorm_nw_shift_fault`

<!-- kernel-facts: turboexl3_test_ling_kdanorm_nw_shift_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdanorm_test.zig:231` (`test turboexl3 gpu ling kdanorm: kdaNormGatePrep's xp is o_proj's prepare of the served norm-gate's output and o_proj's y over it is linear's, word for word, at prompt rows 33 to 16384; its refusals before any launch; planted faults caught (TURBOEXL3_GPU=1)`) |
| inputs | `y`, `z`, `norm_w`, `eps`, `sigtab`, `suh` |
| outputs | `xp` |
| source | `comptime faultText(@embedFile("kernels/test/fault/kdanorm_nw_shift.metal"))` = `dense/prep_1` ++ `ling:test/fault/kdanorm_nw_shift` ++ `dense/prep_3` (2084 B, `df58762a2249`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->
- **Used by:** test only: the red checks in `src/models/ling/kdanorm_test.zig` "turboexl3 gpu ling kdanorm:
  kdaNormGatePrep's xp is o_proj's prepare of the served norm-gate's output and ...", swapped into the fused launch for
  one call (`kdanorm.kernel_override`).
- **Shapes and dtypes:** `turboexl3_ling_kdanorm_prep`'s.
- **Grid and threadgroup:** `turboexl3_ling_kdanorm_prep`'s.
- **Template args:** `turboexl3_ling_kdanorm_prep`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_ling_kdanorm_prep` with norm_w read at the next channel
  (`ling:test/fault/kdanorm_nw_shift` in the gate's place).
- **Bits:** wrong on purpose: its xp differs from the served chain's.
- **Pinned by:** `src/models/ling/kdanorm_test.zig` "turboexl3 gpu ling kdanorm: kdaNormGatePrep's xp is o_proj's
  prepare of the served norm-gate's output and ..." (GPU: its xp must differ from the served chain's at 512 rows,
  `TURBOEXL3_LING_KDA_NORM_RED`).

### `turboexl3_test_ling_kdanorm_suh_next_block_fault`

<!-- kernel-facts: turboexl3_test_ling_kdanorm_suh_next_block_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdanorm_test.zig:234` (`test turboexl3 gpu ling kdanorm: kdaNormGatePrep's xp is o_proj's prepare of the served norm-gate's output and o_proj's y over it is linear's, word for word, at prompt rows 33 to 16384; its refusals before any launch; planted faults caught (TURBOEXL3_GPU=1)`) |
| inputs | `y`, `z`, `norm_w`, `eps`, `sigtab`, `suh` |
| outputs | `xp` |
| source | `comptime faultText(@embedFile("kernels/test/fault/kdanorm_suh_next_block.metal"))` = `dense/prep_1` ++ `ling:test/fault/kdanorm_suh_next_block` ++ `dense/prep_3` (2083 B, `97ed2295d2cb`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->
- **Used by:** test only: the red checks in `src/models/ling/kdanorm_test.zig` "turboexl3 gpu ling kdanorm:
  kdaNormGatePrep's xp is o_proj's prepare of the served norm-gate's output and ...", swapped into the fused launch for
  one call (`kdanorm.kernel_override`).
- **Shapes and dtypes:** `turboexl3_ling_kdanorm_prep`'s.
- **Grid and threadgroup:** `turboexl3_ling_kdanorm_prep`'s.
- **Template args:** `turboexl3_ling_kdanorm_prep`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_ling_kdanorm_prep` with suh read from the next block
  (`ling:test/fault/kdanorm_suh_next_block` in the gate's place).
- **Bits:** wrong on purpose: its xp differs from the served chain's.
- **Pinned by:** `src/models/ling/kdanorm_test.zig` "turboexl3 gpu ling kdanorm: kdaNormGatePrep's xp is o_proj's
  prepare of the served norm-gate's output and ..." (GPU: its xp must differ from the served chain's at 512 rows,
  `TURBOEXL3_LING_KDA_NORM_RED`).

### `turboexl3_test_ling_kdanorm_sum_strided_fault`

<!-- kernel-facts: turboexl3_test_ling_kdanorm_sum_strided_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdanorm_test.zig:233` (`test turboexl3 gpu ling kdanorm: kdaNormGatePrep's xp is o_proj's prepare of the served norm-gate's output and o_proj's y over it is linear's, word for word, at prompt rows 33 to 16384; its refusals before any launch; planted faults caught (TURBOEXL3_GPU=1)`) |
| inputs | `y`, `z`, `norm_w`, `eps`, `sigtab`, `suh` |
| outputs | `xp` |
| source | `comptime faultText(@embedFile("kernels/test/fault/kdanorm_sum_strided.metal"))` = `dense/prep_1` ++ `ling:test/fault/kdanorm_sum_strided` ++ `dense/prep_3` (2129 B, `7cac59b113b2`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->
- **Used by:** test only: the red checks in `src/models/ling/kdanorm_test.zig` "turboexl3 gpu ling kdanorm:
  kdaNormGatePrep's xp is o_proj's prepare of the served norm-gate's output and ...", swapped into the fused launch for
  one call (`kdanorm.kernel_override`).
- **Shapes and dtypes:** `turboexl3_ling_kdanorm_prep`'s.
- **Grid and threadgroup:** `turboexl3_ling_kdanorm_prep`'s.
- **Template args:** `turboexl3_ling_kdanorm_prep`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_ling_kdanorm_prep` with the sum of squares over each lane's strided
  channels (L + 32 i) in place of the host's four consecutive ones (`ling:test/fault/kdanorm_sum_strided` in the gate's
  place).
- **Bits:** wrong on purpose: its xp differs from the served chain's.
- **Pinned by:** `src/models/ling/kdanorm_test.zig` "turboexl3 gpu ling kdanorm: kdaNormGatePrep's xp is o_proj's
  prepare of the served norm-gate's output and ..." (GPU: its xp must differ from the served chain's at 512 rows,
  `TURBOEXL3_LING_KDA_NORM_RED`).

### `turboexl3_test_ling_kdanorm_z_next_head_fault`

<!-- kernel-facts: turboexl3_test_ling_kdanorm_z_next_head_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdanorm_test.zig:232` (`test turboexl3 gpu ling kdanorm: kdaNormGatePrep's xp is o_proj's prepare of the served norm-gate's output and o_proj's y over it is linear's, word for word, at prompt rows 33 to 16384; its refusals before any launch; planted faults caught (TURBOEXL3_GPU=1)`) |
| inputs | `y`, `z`, `norm_w`, `eps`, `sigtab`, `suh` |
| outputs | `xp` |
| source | `comptime faultText(@embedFile("kernels/test/fault/kdanorm_z_next_head.metal"))` = `dense/prep_1` ++ `ling:test/fault/kdanorm_z_next_head` ++ `dense/prep_3` (2080 B, `466a3d8115a1`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->
- **Used by:** test only: the red checks in `src/models/ling/kdanorm_test.zig` "turboexl3 gpu ling kdanorm:
  kdaNormGatePrep's xp is o_proj's prepare of the served norm-gate's output and ...", swapped into the fused launch for
  one call (`kdanorm.kernel_override`).
- **Shapes and dtypes:** `turboexl3_ling_kdanorm_prep`'s.
- **Grid and threadgroup:** `turboexl3_ling_kdanorm_prep`'s.
- **Template args:** `turboexl3_ling_kdanorm_prep`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_ling_kdanorm_prep` with z read from the next head
  (`ling:test/fault/kdanorm_z_next_head` in the gate's place).
- **Bits:** wrong on purpose: its xp differs from the served chain's.
- **Pinned by:** `src/models/ling/kdanorm_test.zig` "turboexl3 gpu ling kdanorm: kdaNormGatePrep's xp is o_proj's
  prepare of the served norm-gate's output and ..." (GPU: its xp must differ from the served chain's at 512 rows,
  `TURBOEXL3_LING_KDA_NORM_RED`).
