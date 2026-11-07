# GLM-5.3-Flash kernels

The kernels made in `src/models/glm53f/`. The one here is an existing dense text with two pieces of its own: DEC-ONE
with the prepare inside (`turboexl3_dense_dec1`'s text) with the member chosen after its head (its suh, its trellis, its
output tiles), the GEMV's tile address read in the member's trellis, over two trellises as they lie, and the finish's
value stored in the member's own output. How an entry reads: [the core catalog's
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
