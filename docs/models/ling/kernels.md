# Ling-3.1-flash kernels

The kernels made in `src/models/ling/`. Ling's routed experts run the core's and DeepSeek-V4.1's kernels with one
SwiGLU line swapped ([the core's catalog](../../core/kernels.md), [DeepSeek-V4.1's](../deepseek_v41/kernels.md)); the
kernels here are its shared expert's at decode rows. Each is DEC-ONE with the prepare inside (`turboexl3_dense_dec1`'s
text) over two members of one input, each in its own buffers, with the member chosen after its head and the GEMV's tile
address read in the member's trellis; its finish stores the member's output, and the second finish of each pair of
128-blocks (the gate's and the up's) runs the host's SwiGLU and the down linear's prepare. How an entry reads: [the core
catalog's introduction](../../core/kernels.md#how-an-entry-reads).

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
