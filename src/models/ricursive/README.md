# Ricursive Ripple-65B

The code only Ripple-65B uses, on turboexl3's dense linear (`src/core/dense.zig`):

| file | what it holds |
|---|---|
| `ricursive.zig` | `join` (two linears of one input joined along OUT at load), `jointLinear` (q \| g, gate \| up in one DEC-ONE launch), `jointLinearBlend` (k \| v, v's value residual formed in its prepare), `valueLinear` (v at prefill rows), `mlp` (the clamped SwiGLU in the down's prepare) |
| `ricursive_test.zig` | `linear` against `format.project` at every Ripple shape (K3, the head K6); the joined launches, `valueLinear` and `mlp` against the host's chains, word for word |
| `kernels/ricursive/` | the pieces the texts add to dense's: the member's suh choice, the blend accessor, the clamped SwiGLU accessor; `kernels/test/pins/` what the CPU test pins |

docs/models/ricursive/ describes the model's needs and every kernel.
