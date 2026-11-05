# Ricursive Ripple-65B

The code only Ripple-65B uses, on turboexl3's dense linear (`src/core/dense.zig`):

| file | what it holds |
|---|---|
| `ricursive.zig` | `join` (two linears of one input joined end to end at load, each member a no-copy view of the joint: `Joint.members`), `jointLinear` (q \| g, gate \| up in one DEC-ONE launch), `jointLinearBlend` (k \| v, v's value residual formed in its prepare), `valueLinear` (v at prefill rows), `mlp` (over the gate \| up joint; the clamped SwiGLU in the down's prepare) |
| `ricursive_test.zig` | `linear` against `format.project` at every Ripple shape (K3, the head K6); the joined launches, `valueLinear` and `mlp` against the host's chains, word for word; the joint's layout (against the interleaved joint, each view against its member, no copy) |
| `kernels/ricursive/` | the pieces the texts add to dense's: the member choice (suh, trellis base, output tiles) and its tile address, the blend accessor, the clamped SwiGLU accessor; `kernels/test/` the interleaved member choice the layout test's reference reads; `kernels/test/pins/` what the CPU test pins |

docs/models/ricursive/ describes the model's needs and every kernel.
