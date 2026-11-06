# Ling-3.1-flash kernels

The kernels made in `src/models/ling/`: the launches of a same-input linear pair (`pair.zig`). Each is an existing
dense text with small pieces of its own: the pair's prepare is dense's `prep_sc` with each member, on the grid's z,
scaling x by its own suh into its own rows of the pair's prepared rows; the pair's GEMVs are DEC-ONE's two texts
(`turboexl3_dense_dec0` over `prep_sc`'s rows, `turboexl3_dense_dec1` with the prepare inside) with the member chosen
after their head (its prepared rows or its suh, its trellis and svh in their own buffers, its output tiles) and the tile
read at Ripple's member tile address (`kernels/ricursive/jtile.metal`). Ling's routed kernels are the core's and
DeepSeek-V4.1's ([README](README.md)). How an entry reads: [the core catalog's
introduction](../../core/kernels.md#how-an-entry-reads).

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
