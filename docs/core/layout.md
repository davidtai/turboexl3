# Source layout

The sources are organized by model: one shared core, and a folder for each model with what only that model uses.

| path | what it holds |
|---|---|
| `src/root.zig` | the module root, with [the API](api.md) (its names are the same whatever folder a file sits in) |
| `src/tests.zig` | the test root: every file's tests |
| `src/core/` | the shared core: the EXL3 reference (`format.zig`), the codeword readers (`lane.zig`), the pack contract (`spec.zig`), `Config` and `configure` (`config.zig`), the routed decode (`decode.zig`) and prefill (`prefill.zig`) engines, the dense linear (`dense.zig`, `pgemm.zig`), the unfused chain (`unfused.zig`), the steel headers (`steel.zig`), the texts more than one kernel splices (`sources.zig`), the host reference (`ref.zig`), the test support and the core's tests |
| `src/models/qwen38/` | Qwen3.8-Flash-Next: the attention projections at prefill rows (`attn_io.zig`), the decode-row merges (`decb.zig`), the GDN tail (`gdn_out.zig`) and their tests |
| `src/models/deepseek_v41/` | DeepSeek-V4.1: the grouped linear `wo_a` (`grouped.zig`) and the tests at its geometry (trunk, F3, clamped SwiGLU, banked decode) |
| `src/models/ricursive/` | Ricursive Ripple-65B: two linears of one input in one launch, the value residual and the clamped SwiGLU in the prepares (`ricursive.zig`) and their tests |
| `<folder>/kernels/<module>/<name>.metal` | the Metal text of each kernel, beside the Zig file that launches it: `<module>` is that file's name without `.zig` (`src/core/kernels/decode/`, `src/models/qwen38/kernels/attn_io/`, ...) |
| `<folder>/kernels/test/` | the Metal texts of that folder's tests: `pins/` holds what a test looks for in the kernels, and `dump/` the parts of the Metal dumps |
| `src/core/kernels/steel_nax.metal` | four steel headers of MLX 0.32.2, inlined, for the prefill and dense GEMMs |
| `src/hip/`, `src/hip_tests.zig`, `src/kernels/hip/` | the AMD (HIP) backend: its launch plans, runtime, tests and kernels (an outside consumer vendors these paths) |
| `plugin/` | the mlx-serve quant plugin ([consuming](consuming.md)) |
| `tools/check_texts.py` | a check that runs on the host, with no Zig: no Metal in a Zig file, and no kernel text changed by a refactor |
| `AGENTS.md` | the rules for a change to this repository; `CLAUDE.md` points to it |

Some kernels of the core serve one model: the fused shared expert and the router fold (Qwen3.8), the banked and grouped
GEMVs (DeepSeek-V4.1), F3 and the clamped SwiGLU (DeepSeek-V4.1, GLM and Ripple). They stay in the core engines that
launch them; [the kernel catalog](kernels.md) names the model of each.

No Metal goes inside a Zig file. Each Zig file imports its Metal texts with `@embedFile`. A variant of a kernel is a
composition, not a search and replace. The base text is cut into numbered pieces (`x_1.metal`, `x_2.metal`, ...), and
the variant joins other pieces between them. [AGENTS.md](../../AGENTS.md) gives these rules.
