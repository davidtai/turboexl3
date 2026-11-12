<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media-kit/turboexl3-lockup-animated-dark.svg">
    <img alt="TurboEXL3" src="docs/media-kit/turboexl3-lockup-animated-light.svg" width="440">
  </picture>
</h1>

turboexl3 runs the projections of an EXL3-quantized LLM on Apple GPUs. It is a Zig library of Metal kernels that MLX
launches. The [glossary](docs/core/glossary.md) defines every term that these documents use.

## What it does

A model in the EXL3 format stores each weight as a trellis code of 1 to 8 bits. To use a weight, a kernel must
first decode its code into a number. turboexl3 decodes the codes inside its own kernels:

- **Routed experts of a mixture-of-experts layer.** At decode rows a chain of split-K GEMV launches serves them. At
  prefill rows a chain of GEMM launches serves them. Neither chain writes a decoded copy of the expert weights to
  memory.
- **Dense projections of the trunk.** These are the attention, GDN, shared-expert, `lm_head` and MTP projections.
  Up to 32 rows, the kernels decode the codes in registers (from 17 rows, only where the NAX gate holds). Above 32
  rows, the default path decodes the weight into an `f16` copy (W_hat) and runs MLX's matmul.
- **Fused launches around these projections.** A serving host uses them to cut the launches and dependent levels per
  layer.

## Models

| model | what turboexl3 runs for it | page |
|---|---|---|
| Qwen3.8-Flash-Next | routed experts, the fused shared expert and router, the dense trunk, the attention projections, the decode-row merges, the GDN tail | [docs/models/qwen38](docs/models/qwen38/README.md) |
| Ricursive Ripple-65B | the dense linears at K3 and the K6 head, q \| g and k \| v in joined launches, the value residual and the clamped SwiGLU in the prepares | [docs/models/ricursive](docs/models/ricursive/README.md) |
| DeepSeek-V4.1 | the clamped routed experts, the banked decode, the dense trunk, the grouped `wo_a`, F3 | [docs/models/deepseek_v41](docs/models/deepseek_v41/README.md) |
| GLM-5.3-Flash | the clamped routed experts at every rate of turboderp's self-calibrated packs, the banked decode, the dense trunk, head, MTP layer and vision tower | [docs/models/glm53f](docs/models/glm53f/README.md) |

One host reference holds every kernel to the same numbers: `format.project`, a port of Sushi's EXL3 reference. A kernel
that replaces a chain of other kernels is byte-identical to that chain; a kernel that sums in its own order stays within
a stated tolerance of `format.project`. [docs/core/kernels.md](docs/core/kernels.md) lists every kernel.

turboexl3 was written as the EXL3 engine of [mlx-stream](https://github.com/davidtai/mlx-stream), a plugin for
[mlx-serve](https://github.com/ddalcu/mlx-serve), to serve [Qwen3.8-Flash-Next](docs/core/glossary.md#qwen38) from
[turboderp](docs/core/glossary.md#turboderp)'s EXL3 packs,
[turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3). Now it reaches mlx-serve
as an outside quant plugin. The folder [`plugin/`](plugin/build.zig) builds that plugin with mlx-serve's plugin kit.

## Use it

```zig
const t3 = b.dependency("turboexl3", .{ .target = target, .optimize = optimize }).module("turboexl3");
t3.addImport("sdk", sdk);
consumer.addImport("turboexl3", t3);
```

[docs/core/consuming.md](docs/core/consuming.md) gives what a consumer supplies and how to build the quant plugin;
[docs/core/api.md](docs/core/api.md) the entry points and the configuration.

## Documents

[docs/README.md](docs/README.md) indexes every document: the core (format, engines, API, consuming, tests, layout,
glossary, the kernel catalog) and one folder per model (what it needs, its packs, its kernels, its results).
[docs/media-kit](docs/media-kit/README.md) holds the TurboEXL3 logo and the results infographic.

## License

MIT (LICENSE). Third-party material and its licenses: NOTICE.
