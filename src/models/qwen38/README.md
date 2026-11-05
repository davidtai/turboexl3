# Qwen3.8-Flash-Next

The kernels only Qwen3.8-Flash-Next's trunk uses, on turboexl3's dense linear (`src/core/dense.zig`):

| file | what it holds |
|---|---|
| `attn_io.zig` | the gated, QK-normed, partially rotated attention's projections at prefill rows: ATTN-IN (q, k, v with the norm and RoPE in the finish), the four-member prepare with the QSA indexer, ATTN-OUT (`o_proj` over the sigmoid-gated core) |
| `decb.zig` | the decode-row merges at those layers (rows 1 to 16): the gate product, the QSA merge, the q planes and the indexer in DEC-ONE's launches |
| `gdn_out.zig` | the GatedDeltaNet layer's tail at prefill rows: z's finish, the gated norm and `out_proj`'s prepare in one launch |
| `*_test.zig` | their tests: byte for byte against the chains they replace, at Qwen3.8's geometry |
| `kernels/<module>/` | their Metal texts; `kernels/test/pins/` what their tests pin |

Qwen3.8 also runs kernels of the core: the routed experts (`decode.zig`, `prefill.zig`), the dense linear, and the fused
shared expert with the router fold (`decode.zig`'s `Kernels3`, `root.moeFusedShared*`), which stay in the core engines
that launch them. docs/KERNELS.md names the model of every kernel.
