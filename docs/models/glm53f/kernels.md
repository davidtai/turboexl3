# GLM-5.3-Flash kernels

The kernels made in `src/models/glm53f/`. Both are an existing dense text with pieces of their own: DEC-ONE with the
prepare inside (`turboexl3_dense_dec1`'s text) with the member chosen after its head (its suh, its trellis, its output
tiles), the GEMV's tile address read in the member's trellis, over two trellises as they lie, and the finish's value
stored in the member's own output; the second also lays its members out at their own K splits. How an entry reads: [the core catalog's
introduction](../../core/kernels.md#how-an-entry-reads).

## `src/models/glm53f/shared_gu.zig`

The shared expert's gate | up in one launch.

### `turboexl3_glm53f_shared_gu`

<!-- kernel-facts: turboexl3_glm53f_shared_gu -->

| fact | from the code |
|---|---|
| made at | `src/models/glm53f/shared_gu.zig:48` (`fn kernelOf`) |
| inputs | `x`, `suh0`, `suh1`, `w0`, `w1`, `svh`, `cnt` |
| outputs | `y0`, `y1`, `z`, `xsx` |
| source | `text_gu` = `dense/dec_head` ++ `glm53f:shared_gu/jsel_pair` ++ `dense/dec_prep1` ++ `dense/dec_gemv1_head` ++ `dense/dec_gemv_loop` ++ `dense/dec_gemv1_x` ++ `dense/dec_gemv_arrive_1` ++ `ricursive:ricursive/jtile` ++ `dense/dec_gemv_arrive_2a` ++ `dense/dec_gemv_arrive_2b` ++ `dense/dec_gemv_arrive_2c` ++ `dense/dec_xsum1` ++ `dense/dec_finish_v` ++ `glm53f:shared_gu/jstore_pair` (8868 B, `9ab319baece4`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** GLM-5.3-Flash (the plugin's `TURBOEXL3_GLM53F_SHARED_GU=1`): `glm53f.shared_gu.gateUp` over a layer's
  shared expert gate and up (the 42 MoE layers' at K6, the MTP layer's at K4) at the decode rows `shared_gu.plan` admits:
  1 to 16 rows whose joined prepare fits threadgroup memory (`Dense.decPrepFits`: every decode row at 4096 -> 2048).
- **Shapes and dtypes:** x T [rows, IN] (bf16, f16 or f32); suh0, suh1 f16 [IN] (the gate's, the up's); w0, w1 the
  gate's and the up's trellises as stored, U16 or I16 [IN/16, out_m/16, NHW] (one dtype); svh f16 [out0 + out1]
  (`shared_gu.joinSvh`: the gate's then the up's); cnt u32 [max(OUT/128 · chunks, 8)] from `turboexl3_dense_dec_zero`.
  Out: y0 T [rows, out0] (the gate's) and y1 T [rows, out1] (the up's), each row-contiguous as its member's `linear`
  returns it, so the SwiGLU after them reads them with no copy; z f32 [KS · rows, OUT] and xsx f32 (scratch, as
  DEC-ONE's).
- **Grid and threadgroup:** `((OUT/16/BT) · 128, chunks, KS)` / `(128, 1, 1)`, DEC-ONE's grid over the joined OUT;
  `turboexl3_dense_dec_zero` before it at `(ceil(NCNT/128) · 128, 1, 1)` / `(128, 1, 1)`.
- **Template args:** NHW (the members' one rate: K6 in the trunk, n 96; K4 in the MTP layer, n 64), CODEBOOK, WBITS, IN,
  OUT (out0 + out1), KS (each member's own `Dense.ksFor(in, out)`: 8 at 4096 -> 2048, where the joined width's would be
  4; the members must share it), BT (`Dense.btFor`), RB (`Dense.rbFor`), ROWS, DEFER (the MUL1 map), OUT_T, OB1 (the
  gate's 128-blocks: the member boundary).
- **Replaces:** `root.linear` of the gate and of the up at those rows: two DEC-ONE calls (`prep_sc` + `dec0` past one row,
  `dec_zero` + `dec1` at one), 4 launches, with `dec_zero` + this launch.
- **Bits:** byte-identical to each member's `linear` (DEC-ONE's text after the member is chosen: the same prepare, K
  split, slice order and finish per output block, each tile's words read from the member's own trellis, the same value
  stored; ricursive's joined launch with each member where it lies instead of end to end).
- **Pinned by:** `src/models/glm53f/shared_gu_test.zig` "turboexl3 gpu glm53f shared gu: the shared expert's gate | up in
  one launch gives each member's linear words at the trunk's K6 and the MTP layer's K4, decode rows 1 to 8 and 16, bf16
  and f32 (TURBOEXL3_GPU=1)" (GPU, words, and the refusal at 17 rows); `src/models/glm53f/shared_gu.zig` "turboexl3
  glm53f shared gu texts: dense's DEC-ONE with the prepare inside, the member chosen over two trellises after its head,
  the tile read in the member's own trellis and the finish's value stored in the member's own output" (CPU, the text)
  and "turboexl3 glm53f shared gu plan: the shared
  expert's gate | up at each member's K split 8 (the joined width's own would be 4), every decode row where the prepare
  fits; 17 rows and members of other K splits decline" (CPU, where it serves).

## `src/models/glm53f/pair_ks.zig`

Two linears of one input at their own K splits in one launch (a DSA layer's q_a and kv_a).

### `turboexl3_glm53f_pair_ks`

| fact | from the code |
|---|---|
| made at | `src/models/glm53f/pair_ks.zig:51` (`fn kernelOf`) |
| inputs | `x`, `suh0`, `suh1`, `w0`, `w1`, `svh`, `cnt` |
| outputs | `y0`, `y1`, `z`, `xsx` |
| source | `text_ks` = `glm53f:pair_ks/ks_head` ++ `glm53f:shared_gu/jsel_pair` ++ `dense/dec_prep1_1` ++ `glm53f:pair_ks/ks_prep1_2` ++ `dense/dec_prep1_3` ++ `dense/dec_gemv1_head` ++ `dense/dec_gemv_loop` ++ `dense/dec_gemv1_x` ++ `dense/dec_gemv_arrive_1` ++ `ricursive:ricursive/jtile` ++ `dense/dec_gemv_arrive_2a` ++ `dense/dec_gemv_arrive_2b` ++ `dense/dec_gemv_arrive_2c` ++ `dense/dec_xsum1` ++ `dense/dec_finish_v` ++ `glm53f:shared_gu/jstore_pair` (9831 B, `5c2014de800a`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

- **Used by:** GLM-5.3-Flash (the plugin's `TURBOEXL3_GLM53F_SMALLOP_FUSE=1`): `glm53f.pair_ks.linear2` over a DSA
  layer's q_a (4096 -> 1536, K split 8) and kv_a (4096 -> 512, K split 32), the trunk's DSA layers' at K6 and the MTP
  layer's at K4, at the decode rows `pair_ks.plan` admits: 1 to 16 rows where each member's prepare fits threadgroup
  memory at its own K split (`Dense.decPrepFits`: every decode row at these widths).
- **Shapes and dtypes:** `turboexl3_glm53f_shared_gu`'s, with z f32 [max(KS0, KS1) · rows, OUT].
- **Grid and threadgroup:** `((NG0 · KS0 + NG1 · KS1) · 128, chunks, 1)` / `(128, 1, 1)`: member m's threadgroups a K
  slice and row chunk NGm = (its 128-blocks) · 8 / BT lie flat along x, member 0's slice by slice, then member 1's;
  `turboexl3_dense_dec_zero` before it at `(ceil(NCNT/128) · 128, 1, 1)` / `(128, 1, 1)`.
- **Template args:** `turboexl3_glm53f_shared_gu`'s with KS0 and KS1 (each member's own `Dense.ksFor(in, out)`) in
  place of KS.
- **Replaces:** `root.linear` of q_a and of kv_a at those rows: two DEC-ONE calls, 4 launches, with `dec_zero` + this
  launch.
- **Bits:** byte-identical to each member's `linear`: each threadgroup takes its member's K slice, slice count, tiles a
  slice, span and arrivals (`ks_head`), its prepare stores at the input's offset in the member's slice in a buffer whose
  row stride is the wider member's (`ks_prep1_2`, dense's prepare cut at its store), and the rest is DEC-ONE's text as
  `turboexl3_glm53f_shared_gu` runs it: the same prepare, K split, slice order and finish per output block.
- **Pinned by:** `src/models/glm53f/pair_ks_test.zig` "turboexl3 gpu glm53f pair ks: a DSA layer's q_a and kv_a at their
  own K splits in one launch give each member's linear words at the trunk's K6 and the MTP layer's K4, decode rows 1 to
  8 and 16, bf16 and f32 (TURBOEXL3_GPU=1)" (GPU, words, and the refusal at 17 rows); `src/models/glm53f/pair_ks.zig`
  "turboexl3 glm53f pair ks texts: ..." (CPU, the text, `kernels/test/pins/pair_ks_texts.metal`) and "turboexl3 glm53f
  pair ks plan: ..." (CPU, where it serves).
