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

### `turboexl3_ling_kdaproj_dec0bv`

<!-- kernel-facts: turboexl3_ling_kdaproj_dec0bv -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj.zig:70` (`fn kernelOf`) |
| inputs | `jxp`, `jxs`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `jcnt`, `xb`, `wb` |
| outputs | `y0`, `y1`, `y2`, `z0`, `z1`, `z2`, `yb` |
| source | `text_kdec0bv` = `ling:kdaproj/bfirst` ++ `ling:kdaproj/bmember_2` ++ `ling:kdaproj/head_v` ++ `ling:kdaproj/member` ++ `ling:kdaproj/bt_0` ++ `dense/dec_gemv0` ++ `dense/dec_gemv_arrive_1` ++ `dense/dec_tile` ++ `dense/dec_gemv_arrive_2_1` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` ++ `ling:kdaproj/bt_1` ++ `dense/dec_gemv0` ++ `dense/dec_gemv_arrive_1` ++ `dense/dec_tile` ++ `dense/dec_gemv_arrive_2_1` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` ++ `ling:kdaproj/bt_close` (16101 B, `5fcfec7db1eb`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** `kdaProjB` (`ling.kdaProjB`, `kdaproj.gemvVariant`) where the configuration asks a variant launch
  (`kdaproj.variantFor`): `Config.kda_bfirst`, or `kda_bt0` / `kda_bt1` at a verify's rows (2 or more) where they fit;
  with `Config.dec_variant` t4 or pf its twins below. The outside plugin turboexl3_ling's candidates KDA_BFIRST and
  KDA_BT (DEC_T4, DEC_PF for the twins).
- **Shapes and dtypes:** `turboexl3_ling_kdaproj_dec0b`'s.
- **Grid and threadgroup:** `((tgs + OUTB / 4) · 128, chunks, 1)` / `(128, 1, 1)`, tgs = OUT0/128 · KS0 · 8/BT0 +
  (OUT1 + OUT2)/128 · KS1 · 8/BT1 (`kdaproj.tgsAt`): b's OUTB / 4 threadgroups first (BF 1: each row of the chunk on
  its own simdgroup) or last (BF 0: simdgroup 0 alone, as `turboexl3_ling_kdaproj_dec0b`), the members' in member order.
- **Template args:** dec0b's with BT0 (member 0's output tiles a threadgroup) and BT1 (members 1 and 2's) in BT's place,
  and BF (b's threadgroups first). At BT0 == BT1 one branch is compiled.
- **Replaces:** `turboexl3_ling_kdaproj_dec0b` (the GEMV inventory's D3).
- **Bits:** byte-identical to `turboexl3_ling_kdaproj_dec0b`: each member's threadgroup runs DEC-ONE's statements over
  its own prepared rows (`ling:kdaproj/bt_0` and `bt_1` set BTC to the member's tile count, then dense's pieces
  verbatim), so its output words are its `linear`'s at any tile count (the tiles a threadgroup move no product or sum:
  each lane's accumulators, the quad fold, the simdgroup sum and the K slices' sum in slice order are per output); b's
  rows run `ling:kdaproj/bmember_2`'s statements (MLX's gemv over the row) whichever threadgroup position and simdgroup
  run them.
- **Pinned by:** `src/models/ling/kdaproj.zig` "turboexl3 ling kdaproj variant texts: ..." (CPU: the composition, the
  head's member-independent lines kept, `kernels/test/pins/kdaproj_variants.metal`) and "turboexl3 ling kdaproj variant
  plan: ..." (CPU: the choice per configuration and row count, the grid's cover of each member's (K slice, block) and
  b's rows at every tile pair, b first or last); `src/models/ling/bproj_test.zig` "turboexl3 gpu ling bproj variants:
  ..." (GPU: `TURBOEXL3_LING_KDA_VARIANT`, each arm's four outputs against the served launch's word for word at rows 1
  to 8 and 16; the red checks at another K split and with a word of b's weight changed).

### `turboexl3_ling_kdaproj_dec0bv_t4`

<!-- kernel-facts: turboexl3_ling_kdaproj_dec0bv_t4 -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj.zig:71` (`fn kernelOf`) |
| inputs | `jxp`, `jxs`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `jcnt`, `xb`, `wb` |
| outputs | `y0`, `y1`, `y2`, `z0`, `z1`, `z2`, `yb` |
| source | `text_kdec0bv` = `ling:kdaproj/bfirst` ++ `ling:kdaproj/bmember_2` ++ `ling:kdaproj/head_v` ++ `ling:kdaproj/member` ++ `ling:kdaproj/bt_0` ++ `dense/dec_gemv0` ++ `dense/dec_gemv_arrive_1` ++ `dense/dec_tile` ++ `dense/dec_gemv_arrive_2_1` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` ++ `ling:kdaproj/bt_1` ++ `dense/dec_gemv0` ++ `dense/dec_gemv_arrive_1` ++ `dense/dec_tile` ++ `dense/dec_gemv_arrive_2_1` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` ++ `ling:kdaproj/bt_close` (16101 B, `5fcfec7db1eb`) |
| header | `dense.header5_t4` = `dense/t4` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26815 B, `fd8d5753490b`) |

<!-- /kernel-facts -->

- **Used by:** as `turboexl3_ling_kdaproj_dec0bv` under `Config.dec_variant` 1 (t4).
- **Shapes and dtypes:** `turboexl3_ling_kdaproj_dec0b`'s.
- **Grid and threadgroup:** `turboexl3_ling_kdaproj_dec0bv`'s.
- **Template args:** `turboexl3_ling_kdaproj_dec0bv`'s.
- **Replaces:** `turboexl3_ling_kdaproj_dec0b` (with banked's group variant t4, as `turboexl3_dense_dec0_t4`).
- **Bits:** byte-identical: `turboexl3_ling_kdaproj_dec0bv`'s text after `dense/t4`'s define (the same exact t).
- **Pinned by:** as `turboexl3_ling_kdaproj_dec0bv`'s (arms t4 and, with pf, pft4 and all).

### `turboexl3_ling_kdaproj_dec0bv_pf`

<!-- kernel-facts: turboexl3_ling_kdaproj_dec0bv_pf -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj.zig:72` (`fn kernelOf`) |
| inputs | `jxp`, `jxs`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `jcnt`, `xb`, `wb` |
| outputs | `y0`, `y1`, `y2`, `z0`, `z1`, `z2`, `yb` |
| source | `text_kdec0bv_pf` = `ling:kdaproj/bfirst` ++ `ling:kdaproj/bmember_2` ++ `ling:kdaproj/head_v` ++ `ling:kdaproj/member` ++ `ling:kdaproj/bt_0` ++ `dense/dec_pf_decl` ++ `dense/dec_gemv0` ++ `dense/dec_pf_next` ++ `dense/dec_gemv_arrive_1` ++ `dense/dec_pf_ww` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` ++ `ling:kdaproj/bt_1` ++ `dense/dec_pf_decl` ++ `dense/dec_gemv0` ++ `dense/dec_pf_next` ++ `dense/dec_gemv_arrive_1` ++ `dense/dec_pf_ww` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` ++ `ling:kdaproj/bt_close` (17953 B, `a7e2be1bd90c`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** as `turboexl3_ling_kdaproj_dec0bv` under `Config.dec_variant` 2 (pf).
- **Shapes and dtypes:** `turboexl3_ling_kdaproj_dec0b`'s.
- **Grid and threadgroup:** `turboexl3_ling_kdaproj_dec0bv`'s.
- **Template args:** `turboexl3_ling_kdaproj_dec0bv`'s.
- **Replaces:** `turboexl3_ling_kdaproj_dec0b` (with banked's group variant pf, as `turboexl3_dense_dec0_pf`).
- **Bits:** byte-identical: `turboexl3_ling_kdaproj_dec0bv`'s text with each branch on `turboexl3_dense_dec0_pf`'s body
  (`dense.dec_body0_pf`: each input tile's words a pass ahead, the same words and statements).
- **Pinned by:** as `turboexl3_ling_kdaproj_dec0bv`'s (arm pf).

### `turboexl3_ling_kdaproj_dec0bv_pft4`

<!-- kernel-facts: turboexl3_ling_kdaproj_dec0bv_pft4 -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj.zig:73` (`fn kernelOf`) |
| inputs | `jxp`, `jxs`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2`, `jcnt`, `xb`, `wb` |
| outputs | `y0`, `y1`, `y2`, `z0`, `z1`, `z2`, `yb` |
| source | `text_kdec0bv_pf` = `ling:kdaproj/bfirst` ++ `ling:kdaproj/bmember_2` ++ `ling:kdaproj/head_v` ++ `ling:kdaproj/member` ++ `ling:kdaproj/bt_0` ++ `dense/dec_pf_decl` ++ `dense/dec_gemv0` ++ `dense/dec_pf_next` ++ `dense/dec_gemv_arrive_1` ++ `dense/dec_pf_ww` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` ++ `ling:kdaproj/bt_1` ++ `dense/dec_pf_decl` ++ `dense/dec_gemv0` ++ `dense/dec_pf_next` ++ `dense/dec_gemv_arrive_1` ++ `dense/dec_pf_ww` ++ `dense/dec_gemv_arrive_2_2` ++ `dense/dec_xsum0` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` ++ `ling:kdaproj/bt_close` (17953 B, `a7e2be1bd90c`) |
| header | `dense.header5_t4` = `dense/t4` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26815 B, `fd8d5753490b`) |

<!-- /kernel-facts -->

- **Used by:** as `turboexl3_ling_kdaproj_dec0bv` under `Config.dec_variant` 3 (pft4).
- **Shapes and dtypes:** `turboexl3_ling_kdaproj_dec0b`'s.
- **Grid and threadgroup:** `turboexl3_ling_kdaproj_dec0bv`'s.
- **Template args:** `turboexl3_ling_kdaproj_dec0bv`'s.
- **Replaces:** `turboexl3_ling_kdaproj_dec0b` (pf and t4 together).
- **Bits:** byte-identical: `turboexl3_ling_kdaproj_dec0bv_pf`'s text after `dense/t4`'s define.
- **Pinned by:** as `turboexl3_ling_kdaproj_dec0bv`'s (arms pft4 and all).

## `src/models/ling/kdaproj_pf.zig`

A KDA layer's q|k|v (4096 -> 24576), f and g (4096 -> 8192 each) at prompt rows (`kdaProjPrompt`, the outside plugin
turboexl3_ling's MLX_SERVE_LING_KDA_PROJ_PROMPT). `linear` at these rows runs, per member, its prepare, its W_hat
decode, MLX's f16 matmul (z half written) and the finish (z read back). Here each member's prepare and W_hat come from
its own `linear`'s launches (`Dense.prefillOperands`), and one launch runs the three products and finishes: pg2's
kernel (`turboexl3_pgemm_fused`) with its N tiles running member 0's, then 1's, then 2's.

### `turboexl3_ling_kdaproj_pf`

<!-- kernel-facts: turboexl3_ling_kdaproj_pf -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/kdaproj_pf.zig:49` (`fn kernelGet`) |
| inputs | `x0`, `x1`, `x2`, `w0`, `w1`, `w2`, `svh0`, `svh1`, `svh2` |
| outputs | `y0`, `y1`, `y2` |
| source | `text_pf` = `ling:kdaproj/pf_1` ++ `pg2/gemm_fused_2` ++ `ling:kdaproj/pf_3` ++ `pg2/gemm_fused_4` ++ `ling:kdaproj/pf_5` ++ `pg2/gemm_fused_6` ++ `ling:kdaproj/pf_undef` (5529 B, `7454701fcb9a`) |
| header | `pgemm.header` = `steel_nax` ++ text 'using namespace mlx::steel;\n' ++ `pg2/gemm_loop` ++ text (72 B) (34287 B, `0ab3ef8c4fca`) |

<!-- /kernel-facts -->

- **Used by:** `kdaProjPrompt` (`ling.kdaProjPrompt`), the outside plugin turboexl3_ling's MLX_SERVE_LING_KDA_PROJ_PROMPT: a KDA layer's three EXL3 projections of its normed input at prompt rows (past `linear`'s register path, 32 rows on NAX). `kdaProjPromptW` (`ling.kdaProjPromptW`) runs it over W_hats the caller decoded once, the plugin's MLX_SERVE_LING_LAYER_WHAT: a layer-major layer's wide spans over the W_hats of its first.
- **Shapes and dtypes:** x0, x1, x2 half [M, K] (each member's prepared rows, natural order), w0, w1, w2 half [K, N_m] (each member's W_hat), svh0, svh1, svh2 half [N_m]. Out: y0, y1, y2 OUT_T [M, N_m] (x's dtype).
- **Grid and threadgroup:** pg2's: `((N0 + N1 + N2) / BN · 2^SWZ · 32, ceil(ceil(M / BM) / 2^SWZ) · WN, WM)` / `(32, WN, WM)`; a threadgroup's N tile lies in one member (each N_m a multiple of BN).
- **Template args:** pg2's: T (half), BM, BN, BK, WM, WN, SWZ, AM, AK (M and K aligned to the tile), AN (1) and OUT_T; the tile `kdaproj_pf.tileFor` (K 4096: `ling_tile`).
- **Replaces:** per member, MLX's matmul and the finish: six launches, and z's write and read, for one.
- **Bits:** each member's are its `linear` chain's: pg2's statements (MLX's NAX gemm_loop verbatim, each column's k in MLX's order whatever the tile; the accumulators rounded to half as MLX's store rounds them; the finish's statements per 128-column block), on that member's operands. A call is taken only where `linear` runs that chain at these rows and MLX runs its regular kernel (`pgemm.mlxRegular`: no split K) on the NAX path.
- **Pinned by:** `src/models/ling/kdaproj_pf.zig` "turboexl3 ling kdaproj prompt texts: ..." (CPU: the composition of pg2's pieces) and `src/models/ling/kdaproj_pf_test.zig` "turboexl3 gpu ling kdaproj prompt: ..." (GPU: each member against `root.linear` word for word at rows 33 to 16384, the refusal at 32) and "turboexl3 gpu ling kdaproj prompt over given W_hats: ..." (GPU: `kdaProjPromptW` over each member's W_hat, decoded once, against `kdaProjPrompt` word for word at rows 33, 1965 and 4096).

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
| made at | `src/models/ling/combine.zig:64` (`fn kernelsFor`) |
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
| made at | `src/models/ling/combine.zig:66` (`fn kernelsFor`) |
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
| made at | `src/models/ling/tail.zig:68` (`fn kernelOf`) |
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
| made at | `src/models/ling/tail.zig:77` (`fn partsKernelOf`) |
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

Launch 2 has three variants, each a host's opt-in in `Config` and each launch 2's words (`mla_latent.Attn`;
`turboexl3_ling_mla_attn_rows`, `_stage`, `_block` below): one threadgroup for 2 or 4 rows of a call, the cache's f32
words staged in threadgroup memory, more tiles a simdgroup.

## `src/models/ling/mla_latent.zig`

### `turboexl3_ling_mla_attn`

<!-- kernel-facts: turboexl3_ling_mla_attn -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_latent.zig:130` (`fn kernelOf`) |
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
| made at | `src/models/ling/mla_latent.zig:129` (`fn kernelOf`) |
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
| made at | `src/models/ling/mla_latent.zig:131` (`fn kernelOf`) |
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

### `turboexl3_ling_mla_attn_rows`

<!-- kernel-facts: turboexl3_ling_mla_attn_rows -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_latent.zig:154` (`fn variantKernel`) |
| inputs | `ql`, `qr`, `lat`, `rot`, `scale`, `dims` |
| outputs | `pm`, `pl`, `po` |
| source | `text_attn_rows` = `ling:mla_latent/rows_1` ++ `ling:mla_latent/attn_2` ++ `ling:mla_latent/rows_2` (11270 B, `83520e3440c5`) |
| header | `text_header` = `ling:mla_latent/header` (175 B, `0f02dd5ef698`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `ling.mlaLatentDecode`, launch 2 of 3 at a call of 2 rows or more under
  `Config.mla_latent_rows` 2 or 4 (`mla_latent.Attn.rows2`, `rows4`): the outside plugin's MTP head on its latent cache
  (MTP_HEAD_LATENT), behind a candidate switch. Not served by default.
- **Shapes and dtypes:** `turboexl3_ling_mla_attn`'s operands and outputs.
- **Grid and threadgroup:** `(512, NJ, ceil(S / RPT)·H/32)` / `(512, 1, 1)`: a threadgroup for each (key group, 32 heads of
  RPT rows; a call's last group of rows may hold fewer), 16 simdgroups in launch 2's layout, each running every row's
  tiles. Threadgroup memory: RPT rows' scores and softmax state (4.4 KiB a row).
- **Template args:** launch 2's, then RPT (rows a threadgroup: 2 or 4).
- **Replaces:** `turboexl3_ling_mla_attn` at those calls: a threadgroup a row for each (key group, 32 heads), each reading
  the group's latents and rope keys.
- **Bits:** launch 2's words. Each row keeps launch 2's scores, max, sum, weights and 512 accumulators, each 8 × 8 tile at
  its place with its products and sums in launch 2's order. A key's latent and rope words are read once for the rows (as
  far as the last row sees) and each row takes the word, or 0 for a key past its position, as launch 2 reads it; a
  sub-chunk at or past a row's bound runs nothing for that row, as launch 2 runs none. The online softmax update is
  launch 2's piece (`attn_2`) on the row's own threadgroup words.
- **Pinned by:** `src/models/ling/mla_latent.zig` "turboexl3 ling mla latent variant texts: ..." (CPU: launch 2's softmax
  piece once, the operands and constants, the per-row bound statements from the pin file) and "turboexl3 ling mla
  latent variants plan: ..." (CPU: its grid and the calls it takes); `src/models/ling/mla_latent_test.zig` "turboexl3 gpu
  ling mla latent variants: ..." (GPU, bytes: the output, pm, pl and po against launch 2's at rows 1 to 4 and keys 1 to
  19,404, on the plugin's operands and two strided caches; red checks: a faulted cache must differ).

### `turboexl3_ling_mla_attn_stage`

<!-- kernel-facts: turboexl3_ling_mla_attn_stage -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_latent.zig:155` (`fn variantKernel`) |
| inputs | `ql`, `qr`, `lat`, `rot`, `scale`, `dims` |
| outputs | `pm`, `pl`, `po` |
| source | `text_attn_stage` = `ling:mla_latent/stage_1` ++ `ling:mla_latent/attn_2` ++ `ling:mla_latent/stage_2` ++ `ling:mla_latent/attn_4` (9347 B, `8bed636fe06a`) |
| header | `text_header` = `ling:mla_latent/header` (175 B, `0f02dd5ef698`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `ling.mlaLatentDecode`, launch 2 of 3 under `Config.mla_latent_stage` at the calls
  `mla_latent_rows` does not take (`mla_latent.Attn.stage`), behind the plugin's candidate switch. Not served by default.
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
- **Pinned by:** as `turboexl3_ling_mla_attn_rows` (the staged words' statements from the pin file; the GPU test's stage
  lines).

### `turboexl3_ling_mla_attn_block`

<!-- kernel-facts: turboexl3_ling_mla_attn_block -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_latent.zig:156` (`fn variantKernel`) |
| inputs | `ql`, `qr`, `lat`, `rot`, `scale`, `dims` |
| outputs | `pm`, `pl`, `po` |
| source | `text_attn_block` = `ling:mla_latent/block_1` ++ `ling:mla_latent/attn_2` ++ `ling:mla_latent/block_2` (12459 B, `5e9316b29042`) |
| header | `text_header` = `ling:mla_latent/header` (175 B, `0f02dd5ef698`) |

<!-- /kernel-facts -->

- **Used by:** Ling-3.1-flash: `ling.mlaLatentDecode`, launch 2 of 3 under `Config.mla_latent_block` at the calls
  `mla_latent_rows` does not take (`mla_latent.Attn.block`; not with `mla_latent_stage`), behind the plugin's candidate
  switch. Not served by default.
- **Shapes and dtypes:** `turboexl3_ling_mla_attn`'s operands and outputs.
- **Grid and threadgroup:** launch 2's, `(512, NJ, S·H/32)` / `(512, 1, 1)`, 16 simdgroups: for the scores, 2 × 2 tiles each
  over a block of 4 sub-chunks (M tiles 2·(sgi / 8) and the next, N tiles 2·(sgi % 8) and the next of the block's 16); for
  the weights times the latents, 4 × 4 output tiles each (the 4 M tiles at columns 32·sgi .. 32·sgi + 31). Threadgroup
  memory: a block's scores (16 KiB).
- **Template args:** launch 2's.
- **Replaces:** `turboexl3_ling_mla_attn`'s one score tile and 1 × 16 output tiles a simdgroup: a query word now serves 2 score
  tiles and a key word 2, a weight word 4 output tiles and a latent word 4.
- **Bits:** launch 2's words: each tile at its place with launch 2's operands and its products and sums in launch 2's order
  (a score tile's 64 latent steps, then 8 rope steps; an output tile's rescale, then 4 steps a sub-chunk); the score tiles
  of a block's sub-chunks at or past the row's bound are computed and never read, as those sub-chunks run no softmax
  update and no product. The softmax update is launch 2's piece (`attn_2`).
- **Pinned by:** as `turboexl3_ling_mla_attn_rows` (the tiles' operand, score and rescale statements from the pin file; the
  GPU test's block lines).

## `src/models/ling/mla_flash.zig`

The MLA prompt attention as one causal flash launch (`root.ling.mlaFlashPrefill`).

### `turboexl3_ling_mla_flash`

<!-- kernel-facts: turboexl3_ling_mla_flash -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/mla_flash.zig:87` (`fn kernelGet`) |
| inputs | `q`, `k`, `v`, `scale` |
| outputs | `o` |
| source | `source` = `ling:mla_flash/body` ++ `ling:mla_flash/close` (10242 B, `f8febea390c3`) |
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
  ..." (CPU, the plan).
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
of each kind it guards: a value one tile computes wrong, a tile on the wrong expert, and a reduction whose order alone
changed (the finish's H128, the GEMM's K loop); and, for the fused finish's threshold, which half of the launch writing y
wrote each row (each half marked by a fault that changes every row it writes).

### `turboexl3_test_pf_dfin_fault`

<!-- kernel-facts: turboexl3_test_pf_dfin_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/rows_test.zig:944` (`test turboexl3 gpu ling rows variants: the routed chain's variants (256-row tiles, the remainder cut, gate-up column blocks, the fused finish, the plain launches) are the served engine's words on synthetic calls at every row bucket, n 40 and 32, limits 0 and 10; planted faults are caught (TURBOEXL3_GPU=1)`) |
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

### `turboexl3_test_pf_t256_route_fault`

<!-- kernel-facts: turboexl3_test_pf_t256_route_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/rows_test.zig:960` (`test turboexl3 gpu ling rows variants: the routed chain's variants (256-row tiles, the remainder cut, gate-up column blocks, the fused finish, the plain launches) are the served engine's words on synthetic calls at every row bucket, n 40 and 32, limits 0 and 10; planted faults are caught (TURBOEXL3_GPU=1)`) |
| inputs | `inds` |
| outputs | `order`, `tiles`, `tiles128`, `xbase` |
| source | `comptime prefill.froute256With(@embedFile("kernels/test/fault/t256_store.metal"))` = `prefill/froute_t128_1` ++ `prefill/froute_count` ++ `prefill/froute_t128_2_1` ++ `prefill/froute_t256_nq` ++ `prefill/froute_t128_2_3` ++ `ling:test/fault/t256_store` ++ `prefill/froute_t128_2_5` (4674 B, `1fc222b2295b`) |
| header | `prefill.header_small` = `prefill/h128` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (20306 B, `574dba73f3cd`) |

<!-- /kernel-facts -->

- **Used by:** test only: the red check in `src/models/ling/rows_test.zig` "turboexl3 gpu ling rows variants: ...",
  swapped into a `Prefill`'s `t128_k.route` for one call.
- **Shapes and dtypes:** `turboexl3_pf_t256_route`'s.
- **Grid and threadgroup:** `turboexl3_pf_t256_route`'s.
- **Template args:** `turboexl3_pf_t256_route`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_pf_t256_route` with its first 256-row tile on the next expert
  (`ling:test/fault/t256_store` in `froute_t256_store`'s place).
- **Bits:** wrong on purpose: that tile's rows are computed with another expert's weights.
- **Pinned by:** `src/models/ling/rows_test.zig` "turboexl3 gpu ling rows variants: ..." (GPU: its y must differ from
  the served engine's, `TURBOEXL3_LING_ROWS_RED` fault=t256_tile0_wrong_expert).

### `turboexl3_test_pf_dfin_order_fault`

<!-- kernel-facts: turboexl3_test_pf_dfin_order_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/rows_test.zig:991` (`test turboexl3 gpu ling rows variants: the routed chain's variants (256-row tiles, the remainder cut, gate-up column blocks, the fused finish, the plain launches) are the served engine's words on synthetic calls at every row bucket, n 40 and 32, limits 0 and 10; planted faults are caught (TURBOEXL3_GPU=1)`) |
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
| made at | `src/models/ling/rows_test.zig:1008` (`test turboexl3 gpu ling rows variants: the routed chain's variants (256-row tiles, the remainder cut, gate-up column blocks, the fused finish, the plain launches) are the served engine's words on synthetic calls at every row bucket, n 40 and 32, limits 0 and 10; planted faults are caught (TURBOEXL3_GPU=1)`) |
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

### `turboexl3_test_pf_dfinn_gemm_fault`

<!-- kernel-facts: turboexl3_test_pf_dfinn_gemm_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/rows_test.zig:1036` (`test turboexl3 gpu ling rows variants: the routed chain's variants (256-row tiles, the remainder cut, gate-up column blocks, the fused finish, the plain launches) are the served engine's words on synthetic calls at every row bucket, n 40 and 32, limits 0 and 10; planted faults are caught (TURBOEXL3_GPU=1)`) |
| inputs | `tiles`, `tiles2`, `x0`, `w0`, `xs0`, `order`, `scores`, `svh_d`, `ecnt`, `zd`, `inds`, `inv`, `zdt`, `xbase` |
| outputs | `y` |
| source | `comptime prefill.dfinnText(@embedFile("kernels/test/fault/fin_4_unrounded.metal"),prefill.dfin_finish_4)` = `prefill/gemm_dfinn_1` ++ `prefill/gemm_1` ++ `prefill/gemm_dfinn_bs` ++ `prefill/gemm_dfin_decl` ++ `prefill/gemm_3_1` ++ `prefill/gemm_dfin_tile` ++ `prefill/gemm_dfinn_skip` ++ `prefill/gemm_3_3` ++ `prefill/gemm_op1_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op1_xsz_1` ++ `prefill/gemm_dfin_epi_1` ++ `prefill/finish_2` ++ `prefill/gemm_dfin_load` ++ `ling:test/fault/fin_4_unrounded` ++ `prefill/gemm_dfin_epi_2` ++ `prefill/gemm_dfinn_2` ++ `prefill/finish_1_1` ++ `prefill/gemm_dfinn_item` ++ `prefill/finish_1_3` ++ `prefill/finish_t128_pick` ++ `prefill/finish_2` ++ `prefill/finish_t128_load` ++ `prefill/finish_4` ++ `prefill/gemm_dfinn_3` (12928 B, `fa5d03b88689`) |
| header | `prefill.header_nax_dfin` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` ++ `prefill/h128` (63060 B, `d36286083098`) |

<!-- /kernel-facts -->

- **Used by:** test only: the threshold's red checks in `src/models/ling/rows_test.zig` "turboexl3 gpu ling rows
  variants: ...", swapped into a `Prefill`'s `dfn_k.fused` for one call.
- **Shapes and dtypes:** `turboexl3_pf_dfinn`'s.
- **Grid and threadgroup:** `turboexl3_pf_dfinn`'s (`Prefill.gemmDFinMin` launches it).
- **Template args:** `turboexl3_pf_dfinn`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_pf_dfinn` with its GEMM half's products left unrounded before the
  score (`ling:test/fault/fin_4_unrounded` in the GEMM half's `finish_4`), so every row the GEMM half writes differs
  from the chain's.
- **Bits:** wrong on purpose, in the GEMM half's rows only: at DFMIN 80, exactly the rows of the experts of 80 rows or
  more differ (every one); at 1 (the threshold ignored) and 81 (off by one) the rows the test holds to the other path
  differ or match.
- **Pinned by:** `src/models/ling/rows_test.zig` "turboexl3 gpu ling rows variants: ..." (GPU: `TURBOEXL3_LING_ROWS_RED`
  fault=dfinn_gemm_half_marked, dfinn_threshold_ignored and dfinn_threshold_off_by_one).

### `turboexl3_test_pf_dfinn_fin_fault`

<!-- kernel-facts: turboexl3_test_pf_dfinn_fin_fault -->

| fact | from the code |
|---|---|
| made at | `src/models/ling/rows_test.zig:1038` (`test turboexl3 gpu ling rows variants: the routed chain's variants (256-row tiles, the remainder cut, gate-up column blocks, the fused finish, the plain launches) are the served engine's words on synthetic calls at every row bucket, n 40 and 32, limits 0 and 10; planted faults are caught (TURBOEXL3_GPU=1)`) |
| inputs | `tiles`, `tiles2`, `x0`, `w0`, `xs0`, `order`, `scores`, `svh_d`, `ecnt`, `zd`, `inds`, `inv`, `zdt`, `xbase` |
| outputs | `y` |
| source | `comptime prefill.dfinnText(prefill.dfin_finish_4,@embedFile("kernels/test/fault/fin_4_unrounded.metal"))` = `prefill/gemm_dfinn_1` ++ `prefill/gemm_1` ++ `prefill/gemm_dfinn_bs` ++ `prefill/gemm_dfin_decl` ++ `prefill/gemm_3_1` ++ `prefill/gemm_dfin_tile` ++ `prefill/gemm_dfinn_skip` ++ `prefill/gemm_3_3` ++ `prefill/gemm_op1_wx` ++ `prefill/gemm_4` ++ `prefill/gemm_op1_xsz_1` ++ `prefill/gemm_dfin_epi_1` ++ `prefill/finish_2` ++ `prefill/gemm_dfin_load` ++ `prefill/finish_4` ++ `prefill/gemm_dfin_epi_2` ++ `prefill/gemm_dfinn_2` ++ `prefill/finish_1_1` ++ `prefill/gemm_dfinn_item` ++ `prefill/finish_1_3` ++ `prefill/finish_t128_pick` ++ `prefill/finish_2` ++ `prefill/finish_t128_load` ++ `ling:test/fault/fin_4_unrounded` ++ `prefill/gemm_dfinn_3` (12928 B, `13561a67b1f7`) |
| header | `prefill.header_nax_dfin` = `steel_nax` ++ `prefill/using_steel` ++ `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `prefill/small_helpers` ++ `prefill/gemm_helpers` ++ `prefill/h128` (63060 B, `d36286083098`) |

<!-- /kernel-facts -->

- **Used by:** test only: the threshold's red check in `src/models/ling/rows_test.zig` "turboexl3 gpu ling rows
  variants: ...", swapped into a `Prefill`'s `dfn_k.fused` for one call.
- **Shapes and dtypes:** `turboexl3_pf_dfinn`'s.
- **Grid and threadgroup:** `turboexl3_pf_dfinn`'s (`Prefill.gemmDFinMin` launches it).
- **Template args:** `turboexl3_pf_dfinn`'s.
- **Replaces:** nothing: a planted fault, `turboexl3_pf_dfinn` with its finish half's products left unrounded before
  the score (`ling:test/fault/fin_4_unrounded` in the finish half's `finish_4`), so every row the finish half writes
  differs from the chain's.
- **Bits:** wrong on purpose, in the finish half's rows only: at DFMIN 80, exactly the rows of the experts of fewer
  than 80 rows differ (every one).
- **Pinned by:** `src/models/ling/rows_test.zig` "turboexl3 gpu ling rows variants: ..." (GPU: `TURBOEXL3_LING_ROWS_RED`
  fault=dfinn_finish_half_marked).
