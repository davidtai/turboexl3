# Ricursive Ripple-65B kernels

The kernels made in `src/models/ricursive/`. Each is an existing dense text with one piece of its own: the joined
launches are DEC-ONE with the prepare inside (`turboexl3_dense_dec1`'s text) with the member's suh chosen after its head;
the prepares are dense's (`turboexl3_dense_prep`, `_prep_sc`) reading their input `x` through a small accessor that forms
the value residual or the clamped SwiGLU. How an entry reads: [the core catalog's introduction](../../core/kernels.md#how-an-entry-reads).

## `src/models/ricursive/ricursive.zig`

The joined launches, the value residual's prepare and the MLP's clamped SwiGLU in the down's prepare.

### `turboexl3_ric_blend_prep`

<!-- kernel-facts: turboexl3_ric_blend_prep -->

| fact | from the code |
|---|---|
| made at | `src/models/ricursive/ricursive.zig:97` (`fn kernelOf`) |
| inputs | `xa`, `x0`, `blend`, `suh` |
| outputs | `xp` |
| source | `text_blend_prep` = `ricursive:ricursive/blend_struct` ++ `ricursive:ricursive/blend_all` ++ `dense/prep` (1795 B, `8f18f915cffe`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** Ripple-65B: `ricursive.valueLinear`, v at prefill rows (more than 16), then `root.linearPrepared` (the
  register path or dequantize + matmul).
- **Shapes and dtypes:** xa, x0 T [rows, IN]; blend f32 [1]; suh f16 [IN] (v's). Out: xp f16 [rows, IN] (natural order).
- **Grid and threadgroup:** `((IN/128) · 32, rows, 1)` / `(32, 1, 1)`, `turboexl3_dense_prep`'s.
- **Template args:** DIM (IN), LANE_ORDER 0 (the natural order the prefill paths read), T (x's dtype).
- **Replaces:** the host's blend (5 launches) and `linear`'s prepare (`turboexl3_dense_prep`): 6 launches, 6 levels,
  with 1.
- **Bits:** byte-identical to the host's blend then `linear(v)`'s prepare, so `valueLinear` gives `linear(v, blended)`'s
  words.
- **Pinned by:** `src/models/ricursive/ricursive_test.zig` "turboexl3 gpu ricursive joint: q | g and gate | up in one
  launch, and k | v with v's value residual in its prepare, give each member's linear words (TURBOEXL3_GPU=1)" (GPU,
  words, `valueLinear` at rows 17, 32, 33 and 64); `src/models/ricursive/ricursive.zig` "turboexl3 ricursive texts:
  dense's DEC-ONE and prepare statements verbatim around the member choice, the blend and the clamped SwiGLU" (CPU, the
  texts).

### `turboexl3_ric_jdec1`

<!-- kernel-facts: turboexl3_ric_jdec1 -->

| fact | from the code |
|---|---|
| made at | `src/models/ricursive/ricursive.zig:95` (`fn kernelOf`) |
| inputs | `x`, `suh0`, `suh1`, `w`, `svh`, `cnt` |
| outputs | `y`, `z`, `xsx` |
| source | `text_jdec1` = `dense/dec_head` ++ `ricursive:ricursive/jsel` ++ `dense/dec_prep1` ++ `dense/dec_gemv1` ++ `dense/dec_gemv_arrive` ++ `dense/dec_xsum1` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` (8013 B, `e308f7a1851d`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** Ripple-65B: `ricursive.jointLinear` over `ricursive.join(q, g)` (and `join(gate, up)` inside
  `ricursive.mlp`) at the decode rows `ricursive.jointPlan` admits: 1 to 16 rows whose joined prepare fits threadgroup
  memory (`Dense.decPrepFits`; q | g and gate | up where a row chunk is at most 3 rows, so 1, 2, 3 and 6, at Ripple's widths).
- **Shapes and dtypes:** x T [rows, IN] (bf16, f16 or f32); suh0, suh1 f16 [IN] (each member's own); w the joined
  trellis U16 or I16 [IN/16, (out0 + out1)/16, NHW] (member 0's tile columns first); svh f16 [out0 + out1]; cnt u32
  [max(OUT/128 · chunks, 8)] from `turboexl3_dense_dec_zero`. Out: y T [rows, out0 + out1] (member 0's columns, then
  member 1's); z f32 [KS · rows, OUT] and xsx f32 (scratch, as DEC-ONE's).
- **Grid and threadgroup:** `((OUT/16/BT) · 128, chunks, KS)` / `(128, 1, 1)`, DEC-ONE's grid over the joined OUT;
  `turboexl3_dense_dec_zero` before it at `(ceil(NCNT/128) · 128, 1, 1)` / `(128, 1, 1)`.
- **Template args:** NHW (the members' one rate: K3 in the pack, n 48), CODEBOOK, WBITS, IN, OUT (out0 + out1), KS (each
  member's own `Dense.ksFor(in, out)`: the members must share it), BT (`Dense.btFor`), RB (`Dense.rbFor`), ROWS, DEFER
  (the MUL1 map), OUT_T, OB1 (member 0's 128-blocks: the member boundary). Every rate, codebook and window DEC-ONE
  serves; the members share IN, the rate and the K split.
- **Replaces:** `root.linear` of each member at those rows: two DEC-ONE calls (`dec_zero` + `dec1`, or `prep_sc` +
  `dec0` past one row at Ripple's widths), 4 launches in 2 to 4 dependent levels, with `dec_zero` + this launch: 1
  dependent level.
- **Bits:** byte-identical to each member's `linear` (DEC-ONE's text after the member's suh is chosen: the same prepare,
  K split, slice order and finish per output block).
- **Pinned by:** `src/models/ricursive/ricursive_test.zig` "turboexl3 gpu ricursive joint: q | g and gate | up in one
  launch, and k | v with v's value residual in its prepare, give each member's linear words (TURBOEXL3_GPU=1)" (GPU,
  words at rows 1 to 6, 8 and 16, bf16 and f32, where it serves; a refusal where it does not);
  `src/models/ricursive/ricursive.zig` "turboexl3 ricursive texts: dense's DEC-ONE and prepare statements verbatim
  around the member choice, the blend and the clamped SwiGLU" (CPU, the texts); `src/models/ricursive/ricursive.zig`
  "turboexl3 ricursive plan: q | g and gate | up join where the row chunks are at most 3 rows (1, 2, 3, 6), k | v at every decode row"
  (CPU, where the joined launch serves).

### `turboexl3_ric_jdec1_blend`

<!-- kernel-facts: turboexl3_ric_jdec1_blend -->

| fact | from the code |
|---|---|
| made at | `src/models/ricursive/ricursive.zig:96` (`fn kernelOf`) |
| inputs | `xa`, `x0`, `blend`, `suh0`, `suh1`, `w`, `svh`, `cnt` |
| outputs | `y`, `z`, `xsx` |
| source | `text_jdec1_blend` = `dense/dec_head` ++ `ricursive:ricursive/jsel` ++ `ricursive:ricursive/blend_struct` ++ `ricursive:ricursive/blend_member` ++ `dense/dec_prep1` ++ `dense/dec_gemv1` ++ `dense/dec_gemv_arrive` ++ `dense/dec_xsum1` ++ `dense/dec_finish_v` ++ `dense/dec_finish_store` (8787 B, `0beac76a58ee`) |
| header | `decode.header5` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` ++ `decode/form5` (26594 B, `41efbbbbd943`) |

<!-- /kernel-facts -->

- **Used by:** Ripple-65B: `ricursive.jointLinearBlend` over `ricursive.join(k, v)`, member 1 (v) reading the value
  residual, at every decode row (1 to 16) at Ripple's widths.
- **Shapes and dtypes:** xa, x0 T [rows, IN] (the block's normed input and the scaled embeddings; bf16, f16 or f32);
  blend f32 [1] (sigmoid(lambda)); suh0, suh1, w, svh, cnt as `turboexl3_ric_jdec1`'s. Out: y T [rows, out0 + out1]; z,
  xsx scratch.
- **Grid and threadgroup:** as `turboexl3_ric_jdec1`: `((OUT/16/BT) · 128, chunks, KS)` / `(128, 1, 1)`, after
  `turboexl3_dense_dec_zero`.
- **Template args:** NHW (the members' one rate: K3 in the pack, n 48), CODEBOOK, WBITS, IN, OUT (out0 + out1), KS (each
  member's own `Dense.ksFor(in, out)`: the members must share it), BT (`Dense.btFor`), RB (`Dense.rbFor`), ROWS, DEFER
  (the MUL1 map), OUT_T, OB1 (member 0's 128-blocks: the member boundary), and T (x's dtype, the accessor's). k | v at
  Ripple's widths: KS 10.
- **Replaces:** `root.linear(k, x)`, the host's blend (astype f32, subtract, multiply, add, astype: 5 launches) and
  `root.linear(v, blended)`: 9 launches in 7 dependent levels, with `dec_zero` + this launch.
- **Bits:** byte-identical to `linear(k, x)` in k's columns and to the host's blend then `linear(v)` in v's (the blend's
  f32 statements one host op each).
- **Pinned by:** `src/models/ricursive/ricursive_test.zig` "turboexl3 gpu ricursive joint: q | g and gate | up in one
  launch, and k | v with v's value residual in its prepare, give each member's linear words (TURBOEXL3_GPU=1)" (GPU,
  words, k | v at rows 1 to 6, 8 and 16, bf16 and f32); `src/models/ricursive/ricursive.zig` "turboexl3 ricursive texts:
  dense's DEC-ONE and prepare statements verbatim around the member choice, the blend and the clamped SwiGLU" (CPU, the
  texts).

### `turboexl3_ric_swiglu_prep`

<!-- kernel-facts: turboexl3_ric_swiglu_prep -->

| fact | from the code |
|---|---|
| made at | `src/models/ricursive/ricursive.zig:99` (`fn kernelOf`) |
| inputs | `gate`, `up`, `sigtab`, `lim`, `suh` |
| outputs | `xp` |
| source | `text_swiglu_prep` = `ricursive:ricursive/swiglu_x` ++ `dense/prep` (2402 B, `a7d5131c1474`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** Ripple-65B: `ricursive.mlp` at prefill rows (more than 16): the down's prepare from the gate's and up's
  `linear` outputs, then `root.linearPrepared` (the register path or dequantize + matmul).
- **Shapes and dtypes:** gate, up T [rows, DIM] (bf16 or f16); sigtab T [65536]; lim T [1]; suh f16 [DIM] (the down's).
  Out: xp f16 [rows, DIM] (natural order).
- **Grid and threadgroup:** `((DIM/128) · 32, rows, 1)` / `(32, 1, 1)`, `turboexl3_dense_prep`'s.
- **Template args:** DIM (40960), LANE_ORDER 0, T (the activation dtype), GSTRIDE (DIM), UOFF (0).
- **Replaces:** the host's clamped SwiGLU (5 launches, or its fused table kernel) and the down's prepare
  (`turboexl3_dense_prep`), with this one launch.
- **Bits:** byte-identical: the down's output equals `linear(down)` of the host's clamped SwiGLU, word for word.
- **Pinned by:** `src/models/ricursive/ricursive_test.zig` "turboexl3 gpu ricursive mlp: gate | up, the clamped SwiGLU
  in the down's prepare and the down give the host's MLP words, decode and prefill rows (TURBOEXL3_GPU=1)" (GPU, words
  at rows 17, 32, 33 and 64 in bf16, row 33 in f16); `src/models/ricursive/ricursive.zig` "turboexl3 ricursive texts:
  dense's DEC-ONE and prepare statements verbatim around the member choice, the blend and the clamped SwiGLU" (CPU, the
  texts).

### `turboexl3_ric_swiglu_prep_sc`

<!-- kernel-facts: turboexl3_ric_swiglu_prep_sc -->

| fact | from the code |
|---|---|
| made at | `src/models/ricursive/ricursive.zig:98` (`fn kernelOf`) |
| inputs | `gate`, `up`, `sigtab`, `lim`, `suh` |
| outputs | `xp`, `xs`, `cnt` |
| source | `text_swiglu_prep_sc` = `ricursive:ricursive/swiglu_x` ++ `dense/prep_s` ++ `dense/prep_sc` (2746 B, `59befcf1aeb4`) |
| header | `decode.header` = `lane/helpers` ++ `decode/common` ++ `decode/gemv_core` (18000 B, `771237d24284`) |

<!-- /kernel-facts -->

- **Used by:** Ripple-65B: `ricursive.mlp` at decode rows (1 to 16): the down's prepare from the gate's and up's outputs
  (joined or two arrays), before the down's DEC-ONE GEMV (`turboexl3_dense_dec0`, rows the down's DEC-ONE takes: its
  prepare outside at Ripple's down) or its GEMV and finish (`Dense.linearPrepared`, rows 5 to 16).
- **Shapes and dtypes:** gate, up T [rows, ...] (bf16 or f16; one joined array [rows, 2 DIM] passed twice, or two [rows,
  DIM]); sigtab T [65536] (MLX's sigmoid of every 16-bit pattern, the host's table); lim T [1] (the limit, 20); suh f16
  [DIM] (the down's). Out: xp f16 [rows, DIM] (lane order), xs f32 [rows, DIM/128] (the block sums), cnt u32 [NCNT] (the
  down's DEC-ONE counters, zeroed).
- **Grid and threadgroup:** `((DIM/128) · 32, rows, 1)` / `(32, 1, 1)`, `turboexl3_dense_prep_sc`'s.
- **Template args:** DIM (the FFN width, 40960), LANE_ORDER 1, NCNT (the down's DEC-ONE counters: max(out/128 · chunks,
  8); 8 where DEC-ONE does not take the rows), NTH (DIM/128 · 32 · rows), T (the activation dtype), GSTRIDE (a gate
  row's stride: 2 DIM joined, DIM apart), UOFF (the up's column offset: DIM joined, 0 apart).
- **Replaces:** the host's clamped SwiGLU in the activation dtype (minimum, clip, sigmoid, two multiplies: 5 launches,
  or its one fused table kernel) and the down's `prep_sc` (or `prep_s`): with this one launch.
- **Bits:** byte-identical: the down's output equals `linear(down)` of the host's clamped SwiGLU of `linear(gate)` and
  `linear(up)`, word for word.
- **Pinned by:** `src/models/ricursive/ricursive_test.zig` "turboexl3 gpu ricursive mlp: gate | up, the clamped SwiGLU
  in the down's prepare and the down give the host's MLP words, decode and prefill rows (TURBOEXL3_GPU=1)" (GPU, words
  at rows 1 to 6, 8 and 16 in bf16 and row 1 in f16, gate | up joined and apart, the clamp reached);
  `src/models/ricursive/ricursive.zig` "turboexl3 ricursive texts: dense's DEC-ONE and prepare statements verbatim
  around the member choice, the blend and the clamped SwiGLU" (CPU, the texts).
