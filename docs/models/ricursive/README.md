# Ricursive Ripple-65B

Ripple-65B is a dense model: 80 layers, hidden 5120, 80 query heads and 8 kv heads of 128, a 4 : 1 pattern of sliding
(window 1024, NeoX RoPE) and full (NoPE) attention layers, four RMSNorms a block (PeriNorm), an untied head over 100,352
tokens. Its EXL3 pack is uniform K3 (MUL1) for every layer linear and K6 for the head. turboexl3 serves its linears;
the outside plugin `turboexl3_ricursive` (mlx-serve) is the host.

Its linears, all from the block's normed input x except o and down:

| linear | shape | input |
|---|---|---|
| q, g (`g_proj`, the sigmoid output gate, in f32) | 5120 -> 10240 | x |
| k | 5120 -> 1024 | x |
| v | 5120 -> 1024 | the value residual lerp(x0, x, sigmoid(lambda)), x0 the scaled embeddings |
| o | 10240 -> 5120 | the attention output times sigmoid(g) |
| gate, up | 5120 -> 40960 | x |
| down | 40960 -> 5120 | silu(min(gate, 20)) · clip(up, -20, 20) |
| head | 5120 -> 100352 | the final norm's output |

What it runs on turboexl3 (`root.ricursive`, `src/models/ricursive/ricursive.zig`):

| part | entry point | rows | launches |
|---|---|---|---|
| q and g | `jointLinear` over `join(q, g)` | 1 to 3 (where the joined prepare fits threadgroup memory) | `dec_zero` + `turboexl3_ric_jdec1`: one dependent level for both |
| k and v, v's input the value residual | `jointLinearBlend` over `join(k, v)` | 1 to 16 | `dec_zero` + `turboexl3_ric_jdec1_blend`: the blend formed in v's prepare |
| v at prefill rows | `valueLinear` | 17 and more | `turboexl3_ric_blend_prep`, then `linear`'s inner product and finish |
| the MLP | `mlp` (gate \| up joined where `jointLinear` serves) | any | decode: gate \| up, `turboexl3_ric_swiglu_prep_sc` (the clamped SwiGLU in the down's prepare), the down's DEC-ONE GEMV (`turboexl3_dense_dec0`) or GEMV and finish; prefill: gate and up `linear`, `turboexl3_ric_swiglu_prep`, the down's inner product and finish |
| o, the head, and any call the entries above decline | `root.linear` | any | the core's dense linear |

Rows 2 to 8 (prompt-lookup speculation) are served by the same paths: the joined launches where they fit, `linear`
elsewhere. Every fused path gives the words of the chain it replaces (each member's `linear`, the host's blend, the
host's clamped SwiGLU in the activation dtype); `linear` is held to `format.project` at every Ripple shape. The tests:
`src/models/ricursive/ricursive_test.zig`. The kernels: [kernels.md](kernels.md).

No speed verdict comes from this repository: the standard test through the outside plugin decides.
