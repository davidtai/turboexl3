# Source layout

The sources are organized by model: one shared core, and a folder for each model with what only that model uses.

| path | what it holds |
|---|---|
| `src/root.zig` | the module root, with [the API](api.md): the core's entry points, and each model's re-exported from its `api.zig` (its names are the same whatever folder a file sits in) |
| `src/tests.zig` | the test root: every file's tests |
| `src/core/` | the shared core: the EXL3 reference (`format.zig`), the codeword readers (`lane.zig`), the pack contract (`spec.zig`), `Config` and `configure` (`config.zig`), the routed decode (`decode.zig`) and prefill (`prefill.zig`) engines, the dense linear (`dense.zig`, `pgemm.zig`), the unfused chain (`unfused.zig`), the steel headers (`steel.zig`), the texts more than one kernel splices (`sources.zig`), the host reference (`ref.zig`), the test support and the core's tests |
| `src/models/qwen38/` | Qwen3.8-Flash-Next: its entry points (`api.zig`), the attention projections at prefill rows (`attn_io.zig`), the decode-row merges (`decb.zig`), the GDN tail (`gdn_out.zig`), the gated shared expert and the router fold inside the routed decode launches (`shared.zig`, `SharedHeads` beside a geometry's `decode.Kernels3`) and their tests |
| `src/models/deepseek_v41/` | DeepSeek-V4.1: its entry points (`api.zig`), the grouped linear `wo_a` (`grouped.zig`), the banked decode over an expert streamer's slot banks and its grouped GEMVs (`banked.zig`, `Banked` beside a geometry's `decode.Kernels3`), the shared expert's fused SwiGLU (`f3.zig`, `F3` beside a `dense.Dense`) and the tests at its geometry (trunk, F3, clamped SwiGLU, banked decode) |
| `src/models/glm53f/` | GLM-5.3-Flash: no code of its own (it runs on the core and DeepSeek-V4.1's banked decode); the tests at its shapes and rates (`glm53f_test.zig`: every dense EXL3 linear; `routed_test.zig`: the routed experts and the banked decode) |
| `src/models/ricursive/` | Ricursive Ripple-65B: two linears of one input in one launch, the value residual and the clamped SwiGLU in the prepares (`ricursive.zig`) and their tests |
| `<folder>/kernels/<module>/<name>.metal` | the Metal text of each kernel, beside the Zig file that launches it: `<module>` is that file's name without `.zig` (`src/core/kernels/decode/`, `src/models/qwen38/kernels/attn_io/`, ...) |
| `<folder>/kernels/test/` | the Metal texts of that folder's tests: `pins/` holds what a test looks for in the kernels, and `dump/` the parts of the Metal dumps |
| `src/core/kernels/steel_nax.metal` | four steel headers of MLX 0.32.2, inlined, for the prefill and dense GEMMs |
| `src/hip/`, `src/hip_tests.zig`, `src/kernels/hip/` | the AMD (HIP) backend: its launch plans, runtime, tests and kernels (an outside consumer vendors these paths) |
| `plugin/` | the mlx-serve quant plugin ([consuming](consuming.md)) |
| `tools/check_texts.py` | a check that runs on the host, with no Zig: no Metal in a Zig file, and no kernel text changed by a refactor |
| `AGENTS.md` | the rules for a change to this repository; `CLAUDE.md` points to it |

A model's variant of a core engine lives in the model's folder as an extension beside the core's object: its kernels
and configs in a struct the model file keeps per core object (`shared.SharedHeads` and `banked.Banked` per
`decode.Kernels3`, `f3.F3` per `dense.Dense`), each method taking the core object for its shapes and launches. The core
keeps what more than one model uses: the clamped SwiGLU in the routed engines (`Geometry.limit`: DeepSeek-V4 and V4.1
and GLM; Ripple clamps in its own kernels), `linearBias` (DeepSeek-V4.1's vision tower and aligner) and `linear` in
pieces (Qwen3.8's GDN layers). [The kernel catalog](kernels.md) names the users of each kernel.

No Metal goes inside a Zig file. Each Zig file imports its Metal texts with `@embedFile`. A variant of a kernel is a
composition, not a search and replace. The base text is cut into numbered pieces (`x_1.metal`, `x_2.metal`, ...), and
the variant joins other pieces between them. [AGENTS.md](../../AGENTS.md) gives these rules.
