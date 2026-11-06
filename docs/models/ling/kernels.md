# Ling-3.1-flash kernels

The kernels made in `src/models/ling/`. Ling's routed experts, its banked decode and its trunk run the core's and
DeepSeek-V4.1's kernels ([the core catalog](../../core/kernels.md), [DeepSeek-V4.1's](../deepseek_v41/kernels.md)); the
kernels here are Ling's own: the routed combine of a wide (prompt) call. How an entry reads: [the core catalog's
introduction](../../core/kernels.md#how-an-entry-reads).

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
