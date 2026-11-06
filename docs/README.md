# turboexl3 documents

The docs mirror the code: one shared core (`src/core/`, `docs/core/`) and a folder per model (`src/models/<model>/`,
`docs/models/<model>/`). Every kernel the code can build has one entry, in the catalog of the folder whose Zig makes it;
`zig build doc-check` (or `python3 tools/kernel_docs.py --check`) fails when one is missing.

## The core

| document | what it holds |
|---|---|
| [core/format.md](core/format.md) | the EXL3 format turboexl3 decodes: rates, codebooks, windows, layout, activations, and the three stages of a projection every kernel is held to |
| [core/engines.md](core/engines.md) | the shared engines: the routed experts at decode and prefill rows, the dense projections, the building blocks; the chains each path replaces, with diagrams, and the exactness classes |
| [core/kernels.md](core/kernels.md) | the catalog of every kernel made in `src/core/`: its pieces, operands, grid, template arguments, the chain it replaces, its bits and its tests |
| [core/api.md](core/api.md) | the entry points of `src/root.zig` and the configuration (`Config`, `configure`) |
| [core/consuming.md](core/consuming.md) | what a consumer supplies (the `sdk` module), the wiring, and how to build, check and install the mlx-serve quant plugin (`plugin/`) |
| [core/testing.md](core/testing.md) | the test groups and their variables, the text check (`tools/check_texts.py`) and the docs check (`tools/kernel_docs.py`) |
| [core/layout.md](core/layout.md) | the source layout: the core, the model folders, where the Metal lies |
| [core/glossary.md](core/glossary.md) | the definition of every term |

## The models

| model | what it needs | kernels | results |
|---|---|---|---|
| Qwen3.8-Flash-Next | [models/qwen38/README.md](models/qwen38/README.md): the parts it runs on turboexl3, the packs, the coding evals | [models/qwen38/kernels.md](models/qwen38/kernels.md): the shared expert and router in the decode launches, the attention projections, the decode-row merges, the GDN tail | [models/qwen38/results.md](models/qwen38/results.md): speed from 1K to 256K tokens with mlx-serve, coding scores, the kernels against Sushi |
| Ricursive Ripple-65B | [models/ricursive/README.md](models/ricursive/README.md): q \| g and k \| v joined, the value residual and the clamped SwiGLU in the prepares, K3 linears and a K6 head | [models/ricursive/kernels.md](models/ricursive/kernels.md) | none yet: the standard test through the outside plugin `turboexl3_ricursive` decides |
| DeepSeek-V4.1 | [models/deepseek_v41/README.md](models/deepseek_v41/README.md): the clamped SwiGLU, the banked decode, the grouped `wo_a`, F3, the trunk | [models/deepseek_v41/kernels.md](models/deepseek_v41/kernels.md) | none published here |
| Ling-3.1-flash | [models/ling/README.md](models/ling/README.md): its routed SwiGLU, unclamped or clamped after the SiLU, stacked and over an expert streamer's banks; K6 trunk, K4 dense FFN | in [core/kernels.md](core/kernels.md) and [models/deepseek_v41/kernels.md](models/deepseek_v41/kernels.md) (no Metal of its own) | none yet: the standard test through the outside plugin `turboexl3_ling` decides |

## Other

| document | what it holds |
|---|---|
| [../AGENTS.md](../AGENTS.md) | the rules for a change to this repository |
| [media-kit/README.md](media-kit/README.md) | the TurboEXL3 logo for light and dark grounds, its colours and its use; the results as an animated infographic |
| [charts/](charts/coding_evals.png) | the result charts the Qwen3.8 pages embed |
