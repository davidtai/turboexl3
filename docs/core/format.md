# The EXL3 format

What turboexl3 decodes, and the reference every kernel is held to. The [glossary](glossary.md) defines every term.

## Formats

- **Rates**: Every even n from 16 to 128, so K = n / 16 bits per weight from 1 to 8. Fractional rates are valid, for
  example n = 36 (K = 2.25) or n = 40 (K = 2.5). Each projection has its own rate. In an expert bank, gate and up
  must share one rate, and `moe` refuses a bank whose down rate differs.
- **Codebooks**: MUL1 and MCG (exllamav3's `codebook.cuh`), named `"mul1"` and `"mcg"` in a pack's config. The MUL1
  map can be deferred to the finish (see the [deferred MUL1 map](glossary.md#deferred-map)).
- **Codeword windows**: From 8 to 16 bits. A pack names its window in the `window` field of `expert_quant`, or uses 16.
  `parseExpertQuant` refuses a width outside 8 to 16 with `error.Exl3WindowUnsupported`.
- **Layout**: A projection is a trellis `U16` `[in/16, out/16, n]` in the tensor-core tile permutation, with `suh`
  `F16` `[in]` and `svh` `F16` `[out]`. An expert bank adds the axis E: `[E, in/16, out/16, n]`, `[E, in]` and
  `[E, out]`. Input and output widths are multiples of 128.
- **Activations**: `bf16`, `f16` or `f32` in, and the output in the input's dtype. Some entry points take only `bf16`
  or `f16`: the fused shared expert and the attention entries. The indexer merge takes only `bf16`. Accumulation is
  `f32` everywhere. The rounding points are exllamav3's.

## The three stages of an EXL3 projection

Every [EXL3](glossary.md#exl3) projection has three stages. [`format.project`](glossary.md#format-project) runs them for
one [row](glossary.md#row) on the CPU, and every kernel is held to it. [H128](glossary.md#h128) includes its scale of
1/√128.

| stage | what it computes | where it rounds |
|---|---|---|
| [prepare](glossary.md#prepare) | `xp = H128(f16(x) · suh)`, per 128-element block of x | x to [`f16`](glossary.md#float-types), then xp to `f16` |
| [inner product](glossary.md#inner-product) | `z = xp · W_hat`, with `f32` accumulation | z to `f16` |
| [finish](glossary.md#finish) | `y = H128(z) · svh`, per 128-element block of z | y to `f16`, then to the output [dtype](glossary.md#dtype) |

Other terms on this page:

- **mid**: The work between the gate and up [GEMV](glossary.md#gemv-gemm) and the down GEMV. It is the gate and up
  finishes, [SwiGLU](glossary.md#swiglu) in `f32`, and the prepare of the down input.
- **planes**: The [split-K](glossary.md#split-k) partial sums, one `f32` plane per K slice. A finish adds them in slice
  order.
- **block sums**: The sum of each 128-element block of a row's prepared input. The [deferred MUL1
  map](glossary.md#deferred-map) needs them for its `C × Σx` term.
- **K span**: The part of the input dimension that one [threadgroup](glossary.md#threadgroup)'s K slices cover.
- **quad**: Four neighboring [lanes](glossary.md#lane) that hold one output column. Two shuffles fold their partial sums
  together.
- **T**: The activation dtype of a call, `bf16` or `f16`.
- **deferred MUL1 map**: The GEMVs and GEMMs multiply by [MUL1](glossary.md#mul1)'s byte sum t, and the finish applies
  `A × acc + C × Σx` once per output. It is always on for MUL1 [packs](glossary.md#pack). The tests also run chains with
  the map off: each weight then rounds to `f16`, as in the reference.
