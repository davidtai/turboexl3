# Qwen3.8-Flash-Next

The kernels only Qwen3.8-Flash-Next uses, on turboexl3's dense linear (`src/core/dense.zig`) and routed decode
(`src/core/decode.zig`):

| file | what it holds |
|---|---|
| `attn_io.zig` | the gated, QK-normed, partially rotated attention's projections at prefill rows: ATTN-IN (q, k, v with the norm and RoPE in the finish), the four-member prepare with the QSA indexer, ATTN-OUT (`o_proj` over the sigmoid-gated core) |
| `decb.zig` | the decode-row merges at those layers (rows 1 to 16): the gate product, the QSA merge, the q planes and the indexer in DEC-ONE's launches |
| `gdn_out.zig` | the GatedDeltaNet layer's tail at prefill rows: z's finish, the gated norm and `out_proj`'s prepare in one launch |
| `shared.zig` | the gated shared expert and the router fold inside the routed decode launches (`SharedHeads`, kept per `decode.Kernels3`: the shared heads on form 9's launches, the side launches, the v2 heads, the folded router) |
| `api.zig` | Qwen3.8's entry points, re-exported by `root.zig` under their own names |
| `*_test.zig` | their tests: byte for byte against the chains they replace, at Qwen3.8's geometry |
| `kernels/<module>/` | their Metal texts; `kernels/test/pins/` what their tests pin |

Qwen3.8 also runs kernels of the core: the routed experts (`decode.zig`, `prefill.zig`) and the dense linear.
docs/core/kernels.md and docs/models/qwen38/kernels.md name the users of every kernel.
