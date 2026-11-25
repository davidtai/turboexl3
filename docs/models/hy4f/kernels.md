# Hy4-flash kernels

The kernels made in `src/models/hy4f/`. The gate's finish is the dense finish's text with a store loop of its own: the
gate's value as text_finish computes it, MLX's sigmoid of it in f32, o read where it lies (in the layout the call names:
its heads' [1, H, S, VH] or its rows' [1, S, H, VH], the template's O_FLAT) and the product stored in o_proj's input
dtype. The SwiGLU's finish is up's dense finish with a store loop of its own: up's value as
text_finish computes it, the gate's y read at the same element, MLX's sigmoid of the gate (the gate finish's piece) and
swigluOut's two products stored in the down linear's input dtype. How an entry reads: [the core catalog's introduction](../../core/kernels.md#how-an-entry-reads).

## `src/models/hy4f/gate.zig`

The attention output gate in its linear's finish (`linearGateMul`, `root.linearGateMul`; the hy4f model plugin's
`gate_fused`) and the red checks' planted faults (`plant`). Rows 1 to 16 (the decode GEMVs), the fused-GEMM option,
pg2, trunk int8 and an un-rotated resident trellis keep the chain (`gate.takes`).

### `turboexl3_hy4f_finish_gate`

- **Used by:** `linearGateMul` at a prompt's rows (17 and up): Hy4-flash's linear_gate (3072 -> 8192 on the 11 indexed
  layers, 3072 -> 16384 on the 30 sliding layers) through the hy4f model plugin's `gate_fused`, its output o_proj's input.
- **Shapes and dtypes:** z f16 [ROWS, DIM] (the gate's inner product by `linear`'s own launches at these rows), svh f16
  [DIM], o_src f32 [1, H, ROWS, VH] (O_FLAT 0: MLX's matmul's un-absorb) or [1, ROWS, H, VH] (O_FLAT 1: the rows the
  host's kv_b kernel writes, `linearGateMulAt(.flat)`), each read where it lies: a launch copies o only where it is not
  row-contiguous. Out: gated G_T [ROWS, DIM]. OUT_T is x's
  dtype (the dtype `linear` rounds the gate's value to), G_T o_proj's input dtype; each bf16, f16 or f32.
- **Grid and threadgroup:** `((DIM / 128) · 32, ROWS, 1)` / `(32, 1, 1)`: `turboexl3_dense_finish`'s, one simdgroup per
  (128-column block, row); at 1024 rows 65536 threadgroups (8192 wide) and 131072 (16384 wide) over the 40 cores.
- **Template args:** DIM (the gate's outputs, H VH), VH (256 on Hy4), H (32 indexed, 64 sliding; DIM = H VH is a
  static_assert), OUT_T, G_T, O_FLAT (o's layout: 0 heads, 1 rows).
- **Replaces:** the gate's `turboexl3_dense_finish` and the chain's MLX launches after it: the f32 copy (bf16 only), the
  sigmoid, o's transpose copy to [1, S, H VH], the multiply and the cast to o_proj's input (bf16 only). Per row of DIM
  outputs it reads z (2 DIM bytes) and o (4 DIM) and writes the product (2 DIM at bf16, 4 DIM at f32): 8 DIM bytes at
  bf16 against the chain's 44 DIM (the finish 4, the copy 6, the sigmoid 8, the transpose 8, the multiply 12, the cast
  6), 10 DIM at f32 against 34 DIM. At DIM 16384: 131 KB a row against 721 KB (bf16).
- **Bits:** the gate's value is text_finish's (`dense/finish_v`, then its store expression) rounded to OUT_T as its store
  rounds it, then taken to f32 (MLX's astype, exact); the sigmoid is MLX's Sigmoid at T = float (unary_ops.h:
  `1 / (1 + metal::precise::exp(metal::abs(x)))`, then the side by the sign); the product one f32 multiply (MLX's
  Multiply), cast to G_T (MLX's astype): the chain's words at either dtype.
- **Pinned by:** `src/models/hy4f/hy4f_test.zig` "turboexl3 hy4f gatemul texts: ..." (CPU: the composition, the kept
  store expression and index, MLX's Sigmoid statements, o's read and the product's store,
  `kernels/test/pins/gate_texts.metal`), "turboexl3 hy4f gatemul refusals: ..." (CPU: the decode rows and wrong operands
  refuse before any engine, in either layout) and "turboexl3 gpu hy4f gatemul pg2arms: ..." (GPU: the chain's words at both of
  linear_gate's widths, rows 20 to 3584, bf16 and f32, with o in its heads layout, as the transposed view of its rows and
  as the rows).

### `turboexl3_test_hy4f_finish_gate_o_layout_fault`

- **Used by:** the red checks only (`plant(.o_layout)`): `hy4f_test.zig`'s gatemul test and the hy4f model plugin's
  `gate_fused` proof.
- **Shapes and dtypes:** `turboexl3_hy4f_finish_gate`'s.
- **Grid and threadgroup:** `turboexl3_hy4f_finish_gate`'s.
- **Template args:** `turboexl3_hy4f_finish_gate`'s.
- **Replaces:** nothing served: `turboexl3_hy4f_finish_gate`'s text with o's read swapped
  (`hy4f:test/fault/gate_src_swapped`).
- **Bits:** o read in the other layout than O_FLAT names (the rows' order on a heads-layout o, the heads' on a rows
  one; the planted fault).
- **Pinned by:** the gatemul GPU test's TURBOEXL3_HY4F_GATEMUL_RED line (its words must differ from the chain's).

### `turboexl3_test_hy4f_finish_gate_fast_exp_fault`

- **Used by:** the red checks only (`plant(.sig_fast_exp)`): `hy4f_test.zig`'s gatemul test.
- **Shapes and dtypes:** `turboexl3_hy4f_finish_gate`'s.
- **Grid and threadgroup:** `turboexl3_hy4f_finish_gate`'s.
- **Template args:** `turboexl3_hy4f_finish_gate`'s.
- **Replaces:** nothing served: `turboexl3_hy4f_finish_gate`'s text with the sigmoid swapped
  (`hy4f:test/fault/gate_sig_fast_exp`).
- **Bits:** MLX's two-sided sigmoid with `metal::fast::exp` in place of `metal::precise::exp` (the planted fault).
- **Pinned by:** the gatemul GPU test's TURBOEXL3_HY4F_GATEMUL_RED line (its words must differ from the chain's).

### `turboexl3_test_hy4f_finish_gate_one_sided_fault`

- **Used by:** the red checks only (`plant(.sig_one_sided)`): `hy4f_test.zig`'s gatemul test.
- **Shapes and dtypes:** `turboexl3_hy4f_finish_gate`'s.
- **Grid and threadgroup:** `turboexl3_hy4f_finish_gate`'s.
- **Template args:** `turboexl3_hy4f_finish_gate`'s.
- **Replaces:** nothing served: `turboexl3_hy4f_finish_gate`'s text with the sigmoid swapped
  (`hy4f:test/fault/gate_sig_one_sided`).
- **Bits:** the one-sided sigmoid `1 / (1 + metal::precise::exp(-x))` in place of MLX's two-sided form (the planted
  fault).
- **Pinned by:** the gatemul GPU test's TURBOEXL3_HY4F_GATEMUL_RED line (its words must differ from the chain's).

### `turboexl3_hy4f_finish_gate_prep`, `turboexl3_hy4f_pg2_gate_prep`

- **Used by:** `linearGateMulPrep` at a prompt's rows (the hy4f model plugin's `gate_prep_fused`): the gate's finish
  writing o_proj's prepared rows; o_proj continues from them by `root.linearOnPrepared` (`linear`'s prefill tail,
  pg2's launch and split K where the served table names o_proj's shape). `_pg2_` is the same store loop in pg2's launch
  (`text_pg2_gate`'s GEMM) where pg2 takes the gate's shape.
- **Shapes and dtypes:** `turboexl3_hy4f_finish_gate`'s (or `turboexl3_hy4f_pg2_gate`'s) inputs and suh_o f16 [DIM]
  (o_proj's suh). Out: xp f16 [ROWS, DIM], natural order.
- **Grid and threadgroup:** `turboexl3_hy4f_finish_gate`'s (`turboexl3_hy4f_pg2_gate`'s).
- **Template args:** theirs and LANE_ORDER 0.
- **Replaces:** the product's store and o_proj's `turboexl3_dense_prep` (the product written and read, xp written: 2 + 2
  + 2 DIM bytes a row at bf16, to 2 DIM).
- **Bits:** the store loop's value G_T(ov * sg) (gate_close's), then o_proj's prepare statements on it (dense
  prep_2_1's expression with suh_o, prep_2_2, prep_3): o_proj's own prepare bytes. Lane L's value j is column
  b 128 + L + 32 j in the gate's store loop and in the prepare alike (one simdgroup a (128-block, row) in both launches).
- **Pinned by:** `hy4f_test.zig` "turboexl3 hy4f finprep gateprep texts and refusals: ..." (CPU) and "turboexl3 gpu hy4f
  finprep gateprep: ..." (GPU: xp against o_proj's `linearPrep` of the product, o_proj from xp against o_proj on the
  product, both widths, rows 3584, 3125 and 31, bf16 and f32, pg2 off and on the served table; the red
  `prep_no_suh`, `hy4f:test/fault/gate_close_prep_nosuh`, in both launches; `plant`'s other faults reach these launches
  too).

## `src/models/hy4f/swiglu.zig`

The shared expert's and the dense MLP's SwiGLU in the up linear's finish (`linearSwigluMul`, `root.linearSwigluMul`; the
host hy4f's `swiglu_fused`) and the red checks' planted faults (`plant`). The gate stays the caller's `linear`. Rows 1
to 16 (the decode GEMVs), the fused-GEMM option, pg2, trunk int8 and an un-rotated resident trellis keep the chain
(`gate.takes`). The routed experts' clamped SwiGLU runs inside `moeSwiglu` and is not this.

### `turboexl3_hy4f_finish_swiglu`

- **Used by:** `linearSwigluMul` at a prompt's rows (17 and up): Hy4-flash's shared expert (gate and up 3072 -> 1536 on
  the 41 MoE layers) and layer 0's dense MLP (3072 -> 13824) through the hy4f model plugin's `swiglu_fused`, its output
  the down linear's input.
- **Shapes and dtypes:** z f16 [ROWS, DIM] (up's inner product by `linear`'s own launches at these rows), svh f16 [DIM],
  gate OUT_T [ROWS, DIM] (the gate's `linear` of the same x). Out: h H_T [ROWS, DIM]. OUT_T is x's dtype (the dtype
  `linear` rounds both values to), H_T the down linear's input dtype; each bf16, f16 or f32.
- **Grid and threadgroup:** `((DIM / 128) · 32, ROWS, 1)` / `(32, 1, 1)`: `turboexl3_dense_finish`'s, one simdgroup per
  (128-column block, row), a simdgroup's lanes reading 32 consecutive z, gate and h elements per step; at 1024 rows
  12288 threadgroups (1536 wide) and 110592 (13824 wide) over the 40 cores.
- **Template args:** DIM (up's outputs), OUT_T, H_T.
- **Replaces:** up's `turboexl3_dense_finish` and the chain's MLX launches after the two linears: the gate's and up's f32
  copies (bf16 only), the sigmoid, the two multiplies and the cast to the down linear's input (bf16 only). Per row of
  DIM outputs it reads z (2 DIM bytes) and the gate's y (2 DIM at bf16, 4 at f32) and writes h (2 DIM at bf16, 4 at
  f32): 6 DIM bytes at bf16 against the chain's 54 DIM (up's finish 4, the copies 12, the sigmoid 8, the multiplies
  24, the cast 6), 10 DIM at f32 against 38 DIM. At bf16: 9,216 B a row against 82,944 (1536 wide), 82,944 against
  746,496 (13824 wide); at 530 GB/s over a 17,492-token prompt, 12.5 ms against 112.2 ms (41 shared layers) and 2.7 ms
  against 24.6 ms (layer 0).
- **Bits:** up's value is text_finish's (`dense/finish_v`, then its store expression) rounded to OUT_T as its store
  rounds it, then taken to f32 (MLX's astype, exact); the gate's y is taken to f32 the same way; the sigmoid is MLX's
  Sigmoid at T = float (`gate/gate_sig`: `1 / (1 + metal::precise::exp(metal::abs(x)))`, then the side by the sign);
  then `(g * sigmoid(g)) * u` as swigluOut's two MLX Multiplys in its order, cast to H_T (MLX's astype): the chain's
  words at either dtype.
- **Pinned by:** `src/models/hy4f/hy4f_test.zig` "turboexl3 hy4f swiglumul texts: ..." (CPU: the composition, the kept
  store expression and index, the gate's read, the products' store, `kernels/test/pins/swiglu_texts.metal`, and MLX's
  Sigmoid statements from `gate_texts.metal`), "turboexl3 hy4f swiglumul refusals: ..." (CPU: the decode rows and wrong
  operands refuse before any engine) and "turboexl3 gpu hy4f swiglumul pg2arms: ..." (GPU: the chain's words at both widths,
  rows 20 to 3584, bf16 and f32).

### `turboexl3_test_hy4f_finish_swiglu_swapped_fault`

- **Used by:** the red checks only (`plant(.swapped)`): `hy4f_test.zig`'s swiglumul test and the hy4f model plugin's
  `swiglu_fused` proof.
- **Shapes and dtypes:** `turboexl3_hy4f_finish_swiglu`'s.
- **Grid and threadgroup:** `turboexl3_hy4f_finish_swiglu`'s.
- **Template args:** `turboexl3_hy4f_finish_swiglu`'s.
- **Replaces:** nothing served: `turboexl3_hy4f_finish_swiglu`'s text with the opening swapped
  (`hy4f:test/fault/swiglu_open_swapped`).
- **Bits:** the gate and up swapped, h = (u * sigmoid(u)) * g (the planted fault).
- **Pinned by:** the swiglumul GPU test's TURBOEXL3_HY4F_SWIGLUMUL_RED line (its words must differ from the chain's).

### `turboexl3_test_hy4f_finish_swiglu_fast_exp_fault`

- **Used by:** the red checks only (`plant(.sig_fast_exp)`): `hy4f_test.zig`'s swiglumul test.
- **Shapes and dtypes:** `turboexl3_hy4f_finish_swiglu`'s.
- **Grid and threadgroup:** `turboexl3_hy4f_finish_swiglu`'s.
- **Template args:** `turboexl3_hy4f_finish_swiglu`'s.
- **Replaces:** nothing served: `turboexl3_hy4f_finish_swiglu`'s text with the sigmoid swapped (the gate finish's fault
  piece `hy4f:test/fault/gate_sig_fast_exp`).
- **Bits:** MLX's two-sided sigmoid with `metal::fast::exp` in place of `metal::precise::exp` (the planted fault).
- **Pinned by:** the swiglumul GPU test's TURBOEXL3_HY4F_SWIGLUMUL_RED line (its words must differ from the chain's).

## `src/models/hy4f/rms.zig`

The RMSNorms and the next linear's prepare in a linear's prompt finish (`linearRms`, `linearRmsPrep`, `linearKv`,
`linearSwigluPrep`; `root.linearRms` and the others; the hy4f model plugin's `addnorm_fused`, `qnorm_fused`,
`kvnorm_fused` and `down_prep_fused`) and the red checks' planted faults (`plant`). The three norm
kernels run one threadgroup a row: one simdgroup a 128-column block runs text_finish's loads, butterfly and value
(`rms/finish_v_row`: `dense/finish_v` with the block the simdgroup's), the row is staged in threadgroup memory as
text_finish's store rounds it (OUT_T, then f32), and MLX's `rms_single_row` statements (`rms_norm.metal` at the server's
MLX c69303b0, T = float, N_READS 4) run over it: MLX's threadgroup of axis / 4 threads is this one's axis / 128
simdgroups, thread `lid` taking values 4 lid to 4 lid + 3, its sums in MLX's order (the square sum MLX's text,
`acc += x * x`). Rows 1 to 16, the fused-GEMM option, pg2, trunk int8, an un-rotated resident trellis (on
this linear, and on the next one where the call prepares its rows) and rows of more than 4096 outputs keep the chain.

### `turboexl3_hy4f_finish_add_rms`

- **Used by:** `linearRms` (its `add`) at a prompt's rows: the shared expert's down (1536 -> 3072) with the routed
  experts' up added and the MLP post-norm, on the 40 MoE layers, through `addnorm_fused`.
- **Shapes and dtypes:** z f16 [ROWS, DIM] (the inner product by `linear`'s own launches, from x or from prepared rows),
  svh f16 [DIM], add [ROWS, DIM] (bf16, f16 or f32: the routed up's y as `linear` gives it), w f32 [DIM], nrm uint32 [1]
  (the norm's width), eps_in f32 [1]. Out: y f32 [ROWS, DIM].
- **Grid and threadgroup:** `((DIM / 128) · 32, ROWS, 1)` / `((DIM / 128) · 32, 1, 1)`: one threadgroup a row (768
  threads at DIM 3072), 12 KB of threadgroup memory for the row.
- **Template args:** DIM (the linear's outputs, at most 4096), OUT_T (the dtype `linear` rounds to).
- **Replaces:** the down's finish, both f32 copies (bf16 only), MLX's add and `rms` launches. Per row: z 2 DIM, add 2
  DIM, y 4 DIM = 8 DIM against the chain's 36 DIM at bf16 (the finish 4, two copies 12, the add 12, the norm 8).
- **Bits:** `f32(add) + f32(OUT_T(value))` (MLX's Add of the two f32 copies, add first, as the chain's `add(up,
  shared)`; the value the one text_finish stores, taken to f32, exact), then MLX's statements: the square sum per
  thread, two `simd_sum`s through `local_sums`, `metal::precise::rsqrt(acc / axis_size + eps)`, and `w[i] * (x[i] *
  inv)`.
- **Pinned by:** `hy4f_test.zig` "turboexl3 hy4f finprep rms texts: ..." (CPU: `finish_v_row` is `finish_v` with the
  block line swapped, the compositions, MLX's statements, `kernels/test/pins/rms_texts.metal`), "turboexl3 hy4f finprep
  rms refusals: ..." (CPU) and "turboexl3 gpu hy4f finprep addnorm: ..." (GPU: the chain's words at the shared down, rows
  20 to 3584, bf16 and f32, from x and from prepared rows).

### `turboexl3_hy4f_finish_rms_prep`

- **Used by:** `linearRmsPrep` at a prompt's rows: q_a (3072 -> 1024) with the q norm and q_b's prepare, through
  `qnorm_fused`; q_b continues from xp (`root.linearPrepared`, or `linearQRopePrepared` under `qb_rope_fused`).
- **Shapes and dtypes:** `turboexl3_hy4f_finish_add_rms`'s inputs without add, and suh f16 [DIM] (q_b's). Out: xp f16 [ROWS, DIM]
  (natural order) and q f32 [ROWS, DIM] where QF is 1 (q_resid, the indexed layers' indexer input), else q [1, 1]
  unwritten.
- **Grid and threadgroup:** `turboexl3_hy4f_finish_add_rms`'s.
- **Template args:** `turboexl3_hy4f_finish_add_rms`'s, Q_T (q_b's input dtype), QF, LANE_ORDER 0.
- **Replaces:** q_a's finish, the f32 copy, MLX's `rms`, the cast to Q_T and q_b's `turboexl3_dense_prep`. Per row of
  DIM: z 2 DIM and xp 2 DIM (q 4 DIM more on indexed layers) against the chain's 28 DIM at bf16.
- **Bits:** the norm's output values by `turboexl3_hy4f_finish_add_rms`'s statements on the staged row (no add), each
  taken to Q_T, then `linear`'s
  prepare statements (`dense/prep_2_1`'s load on the staged row, `prep_2_2`, `prep_3`): q_b's own prepare bytes.
- **Pinned by:** the rms texts and refusals tests and "turboexl3 gpu hy4f finprep qnorm: ..." (GPU: xp against q_b's
  `linearPrep` of the chain's cast, q_resid's words, and q_b from xp against `linear` and `linearQRope` of the cast).

### `turboexl3_hy4f_finish_kv`

- **Used by:** `linearKv` at a prompt's rows: kv_a as the pack stores it (3072 -> 640) with the kv norm of its first
  512 values and the rope of the next 64, through `kvnorm_fused`.
- **Shapes and dtypes:** `turboexl3_hy4f_finish_add_rms`'s inputs without add (w f32 [LAT], nrm LAT) and off int32 [1] (the first row's
  position). Out: y f32 [ROWS, LAT] (c_kv), k_pe f32 [ROWS, ROPE].
- **Grid and threadgroup:** `turboexl3_hy4f_finish_add_rms`'s at DIM 640 (5 simdgroups; the fifth adds MLX's zero
  `local_sums` slot).
- **Template args:** DIM, OUT_T, LAT (a whole number of 128), ROPE (even, at most 128), LOG2B (log2 of the
  rope base as MLX takes it, its f32 bits).
- **Replaces:** kv_a's finish, the f32 copy, the latent's slice copy and MLX's `rms`, the rope part's slice copy and
  MLX's rope. Per row: 3.5 KB against about 15.6 KB at bf16.
- **Bits:** c_kv by `turboexl3_hy4f_finish_add_rms`'s statements (no add) on the first LAT values; k_pe by qrope's pair and rope statements
  (`qrope/qrope_pair`, `qrope/qrope_rot`: MLX's traditional rope at T = float) in the simdgroup holding values LAT to
  LAT + ROPE, every lane shuffling, the rope dims stored.
- **Pinned by:** the rms texts and refusals tests and "turboexl3 gpu hy4f finprep kvnorm: ..." (GPU: both thetas, rows 33
  to 3584 at positions 0 to 17459).

### `turboexl3_hy4f_finish_swiglu_prep`

- **Used by:** `linearSwigluPrep` at a prompt's rows: the shared expert's and the dense MLP's up with the SwiGLU and the
  down linear's prepare, through `down_prep_fused` (with `swiglu_fused`); the down continues from xp.
- **Shapes and dtypes:** `turboexl3_hy4f_finish_swiglu`'s inputs and suh f16 [DIM] (the down's). Out: xp f16 [ROWS,
  DIM] (natural order).
- **Grid and threadgroup:** `turboexl3_hy4f_finish_swiglu`'s: one simdgroup per (128-column block, row).
- **Template args:** DIM, OUT_T, H_T (the down's input dtype), LANE_ORDER 0.
- **Replaces:** h's store and the down's `turboexl3_dense_prep`: 6 DIM bytes a row (h written, read, xp written) to 2
  DIM.
- **Bits:** `turboexl3_hy4f_finish_swiglu`'s value h in H_T, then `linear`'s prepare statements on it (x's f16 rounding
  times suh, `prep_2_2`, `prep_3`): the down's own prepare bytes.
- **Pinned by:** the rms texts and refusals tests and "turboexl3 gpu hy4f finprep downprep: ..." (GPU: xp against the
  down's `linearPrep` of h, and the down from xp against `linear` of h).

### `turboexl3_test_hy4f_finish_add_rms_no_add_fault`, `..._rms_prep_no_suh_fault`, `..._kv_no_weight_fault`, `..._kv_freq_ln_fault`, `..._swiglu_prep_no_suh_fault`

- **Used by:** the red checks only (`plant`): the finprep GPU tests and the hy4f model plugin's proofs.
- **Shapes and dtypes, grid and threadgroup, template args:** their served kernel's.
- **Replaces:** nothing served: the served text with one piece swapped: the output loop without the weight
  (`hy4f:test/fault/rms_store_noweight`, in the rms and kv kernels), the staging without the add
  (`hy4f:test/fault/rms_stage_add_dropped`), the next linear's prepare without its suh
  (`hy4f:test/fault/rms_prep_load_nosuh`, `hy4f:test/fault/swp_close_nosuh`), k_pe's frequency by MLX's CPU fallback
  formula (qrope's `hy4f:test/fault/qrope_rot_freq_ln`).
- **Bits:** the planted fault's.
- **Pinned by:** the finprep GPU tests' `_RED` lines (their words must differ from the chain's; the kv faults move only
  their own output).
