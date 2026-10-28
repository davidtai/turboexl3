# Consuming turboexl3

This page tells a [consumer](docs/GLOSSARY.md#consumer) how to build turboexl3 into its own build. The first section
gives what every consumer supplies. The second section describes the recommended consumer: the mlx-serve
[quant plugin](docs/GLOSSARY.md#quant-plugin) in the folder `plugin/`. A later section can describe another consumer
in the same way.

## What every consumer supplies

turboexl3 is one Zig module, and its root file is `src/root.zig`. Its files import each other by relative path.
Apart from these files, they import only three modules:

| module | who supplies it | what it holds |
|---|---|---|
| `std` | Zig | the standard library |
| `sdk` | the consumer | `sdk.mlx`: mlx-c's bindings, plus `check`, `getShape` and `streamIsGpu`; `sdk.kernel_launch`: `newKernel`, `Launch`, `TemplateArg`, `buildConfig` and `applyConfig`, and optionally `mlx_fast_metal_kernel_free`; optionally `sdk.steel`, the steel headers of the consumer's MLX ([Wiring](README.md#wiring) gives the full surface) |
| `mlx_host` | the consumer, for the tests only | the [host surface](docs/GLOSSARY.md#host-surface); the tests use only its `sushi_exl3`, where it exists, as a second reference |

The `build.zig` of the package gives a consumer two things:

- The module `turboexl3`, over `src/root.zig`. The consumer adds its `sdk` to this module as an import.
  [Wiring](README.md#wiring) shows the three lines.
- The function `testModule`. It creates the module of the test root, `src/tests.zig`, over the consumer's `sdk` and
  `mlx_host`.

The test binary must also link MLX, the consumer's launcher and Metal. The GPU tests also need a Metal device (see
[Tests](README.md#tests)).

The consumer sets the configuration once, before its first call, with `configure` (see
[Configuration](README.md#configuration)). On a machine that fails the [NAX gate](docs/GLOSSARY.md#nax-gate), it
must set `nax` to false.

## The mlx-serve quant plugin (`plugin/`)

The folder `plugin/` is turboexl3 as an outside quant plugin of mlx-serve, in the plugin layout of mlx-serve's `main`
(the rewrite of PR 978, merged on 2026-10-27). It builds against the kit of the branch
[`qwen38-exl3-full`](https://github.com/davidtai/mlx-serve/tree/qwen38-exl3-full) of davidtai/mlx-serve, which adds the
host side of an outside quant plugin. That branch is the head of
[ddalcu/mlx-serve#997](https://github.com/ddalcu/mlx-serve/pull/997) before the pull request became the host support of
the `qwen38_exl3` model plugin, which is to replace this quant plugin. mlx-serve's [plugin kit](docs/GLOSSARY.md#plugin-kit) builds it into one dylib,
`libturboexl3.dylib`. The server loads the dylib at startup and offers it the quantization blocks of each pack.

| file | what it does |
|---|---|
| `build.zig` | calls the builder `plugin()` of the kit, then adds the imports `sdk`, `turboexl3` and `turboexl3_table` |
| `build.zig.zon` | names the staged kit (the path `kit`) and this package (the path `..`) as its dependencies |
| `plugin.zig` | the dylib's root: the table and the claims are turboexl3_table's |
| `config.zig` | the file the kit's `test` step builds; the claims and their tests are in `table/config.zig` |
| `sdk.zig` | turboexl3's `sdk`: the bindings, turboexl3_kit's launcher over the kernel cache and the launcher of the server, and the kit's quant ABI |
| `mlx.zig` | the mlx-c functions that turboexl3 calls, declared with Zig types |

In the package itself, `table/table.zig` is the quant table (`matmul` runs turboexl3's `linear`, `swiglu` runs its
`moe`, and the optional members run its fused paths: [What it runs](#what-it-runs)), `table/config.zig` the claim, the
validation and the bill with no MLX, and `kit/nax.zig` the test of the NAX gate, from `sysctl`, with no MLX.

### What it claims

The plugin answers each block of a pack with a [claim](docs/GLOSSARY.md#claim):

- **A `trunk_quant` block** (the dense linears of the trunk): always `by_marker`. A trunk block has no rate `k`,
  because each linear carries its own rate. No built-in format of mlx-serve runs a dense EXL3 linear.
- **An `expert_quant` block** (the routed banks, with a rate `k`): `by_marker` only when all of these hold. Otherwise
  the claim is `no`, and the built-in Sushi format keeps the banks.
  - The [numerics](docs/GLOSSARY.md#numerics) are reordered.
  - The machine passes the NAX gate.
  - The SwiGLU is not clamped: the model's config has no `swiglu_limit` above 0.
- **Any other block**: `no`. This includes Sushi's STQ family and the other formats.

`by_marker` ranks above Sushi's claim, `by_type`. So the plugin wins each block that it claims, unless another
outside plugin also claims that block at `by_marker`.

The plugin reads `MLX_SERVE_BIT_EXACT` itself, in the same way as the server:

| value | numerics |
|---|---|
| `0` | reordered |
| any other value, for example `1` | exact |
| unset or empty | reordered on a Qwen3.8 pack (`model_type` `qwen4_exp` or `qwen4_exp_text`); exact on every other pack |

The validation refuses a block by the name of the reason:

| refusal | when |
|---|---|
| `Exl3CodebookUnsupported` | a codebook other than `mul1` or `mcg`, or a bank block without a codebook |
| `Exl3WindowUnsupported` | a window outside 8 to 16 bits |
| `Exl3RateUnsupported` | a bank rate `k` for which n = 16 × k is not an even whole number from 16 to 128 |
| `Exl3TopKExceedsReduceBank` | a bank block of a model whose `num_experts_per_tok` is above 32 |
| `NotExl3` | a block that is not EXL3 |

The bill of a block is its stored size, because the plugin uses the arrays that the server loaded, as they are.

### What it runs

- **`open`** binds the host table of the server to `sdk`. Once per process, it also configures turboexl3: the
  defaults, with `nax` from `kit/nax.zig` (turboexl3_kit). Below the gate, it logs that dense prefill rows take dequantize + MLX's
  matmul and that the banks stay with the built-in format.
- **`load_tensor`** (trunk blocks) takes `<base>.weight` (the trellis, `U16` or `I16`), `<base>.scales` (`suh`) and
  `<base>.biases` (`svh`). These are the names of the affine format. It checks the shapes and the dtypes.
- **`load_bank`** (bank blocks) takes `<base>.trellis`, `<base>.suh` and `<base>.svh`, in Sushi's layout.
- **`matmul`** runs turboexl3's `linear`.
- **`swiglu`** runs turboexl3's `moe` on the three banks of one layer. It refuses a clamped SwiGLU, and banks that
  decode differently.
- **`moe`**, for one projection at a time, always refuses by name: the banks of a layer run as one routed SwiGLU.

A refusal during a call returns a verdict that names the error.

The fused paths of turboexl3 are optional members of the table, as `swiglu` is. The plugin fills all of them, so it
builds only with a kit whose table has these members. The kit of mlx-serve's `main` does not have them (the branch
`qwen38-exl3-full` adds them), and the build of the plugin fails against it. Each member refuses, by name, a call that it does not support,
and the host then runs its own chain of operations:

- **`linear_split`, `linear_finish`, `linear_prepared`, `linear_pieces`, `linear_piece`**: the parts of `matmul`
  (the prepare of x, the inner product, the finish) as separate calls, so that the tail of a gated delta net (GDN)
  layer can put its own work between them.
- **`attn_in`, `attn_prep`, `linear_prepared_idx`**: the input projections of a gated attention layer at prefill
  rows: q, k and v with the q / k RMSNorm and the partial RoPE in the finish, and one prepare that four linears share.
- **`gated_out`, `merged_gated_out`, `q_planes`, `indexer`**: the decode-row merges: the output projection over the
  gated attention output (also over the split partials of the sparse attention), q_proj as separate query and gate
  planes, and the indexer projection with its epilogue.
- **`swiglu_shared`**: `swiglu` with the gated shared expert in the same launches, at decode and verify rows. It can
  also run the router in those launches.
- **`resident`, `resident_pending`, `unrotate`, `linear_joined`**: the trunk: decoded matrices that stay in memory
  after their first prefill call, those matrices taken out of the format's basis, and one call for a group of linears
  that read the same input.

### Build it

A fetched turboexl3 package does not hold `plugin/`, so build the plugin in a checkout of turboexl3.

1. In an mlx-serve checkout whose quant table has the fused members (see [What it runs](#what-it-runs)), run
   `zig build kit`. It stages the kit in `zig-out/kit/mlx-serve-plugin-kit-<version>/`, with a tarball beside it.
2. Link the staged kit, or extract its tarball, as `plugin/kit`.
3. In `plugin/`, run `zig build`. It writes `zig-out/lib/libturboexl3.dylib`.
4. In `plugin/`, run `zig build test`. It runs the MLX-free tests of the claim, the validation and the NAX gate.

The dylib links no MLX. Its mlx-c calls resolve against the server's libmlxc when the server loads it. So the process
holds one MLX, and the MLX pin of the kit must equal the server's.

### Check, install and turn it on

1. Run `mlx-serve plugin check zig-out/lib/libturboexl3.dylib --model <pack>`. It loads the dylib and runs the
   conformance suite on the plugin's fixture. With `--model`, the plugin also claims and parses the blocks of the pack.
2. Run `mlx-serve plugin install zig-out/lib/libturboexl3.dylib`. It copies the dylib into `~/.mlx-serve/plugins/`, with
   a receipt of its hash and signature. A downloaded file installs only after you confirm, or with `--yes`.
3. Set `"enabled": true` in `~/.mlx-serve/plugins.json`, for example `{"enabled": true}`. The server loads the
   installed plugins when it starts.

Before the server opens a dylib, it compares the dylib with its receipt. The option `--no-plugins` keeps every outside
plugin unloaded. The list `"disabled"` in `plugins.json` turns off plugins by name.

### What the kit lacks for turboexl3

| gap | what the plugin does instead | what would close the gap |
|---|---|---|
| MLX's steel headers: the kit stages only mlx-c's headers | turboexl3 uses its inlined copy, `src/kernels/steel_nax.metal` (MLX 0.32.2) | the kit stages the four steel headers of the server's MLX, for `sdk.steel` |
| the NAX gate of the server | `kit/nax.zig` (turboexl3_kit) finds it again from `sysctl`: an Apple M5 or later, macOS 26.2 or later, and no `FORCE_GPU_FAMILY_FALLBACK=1` | a member of the host table that gives the server's answer |
| mlx-c bindings with Zig types: the kit's `mlx_c` comes from translate-C, with integer enums and `[*c]` pointers | `plugin/mlx.zig` declares the functions that turboexl3 calls, as mlx-serve's own bindings declare them | bindings with Zig types in the kit |
| extra imports in `plugin()`: the builder takes only a name | `build.zig` adds `sdk` and `turboexl3` to the root module through `import_table.get("plugin")` | an option of `plugin()` for imports |
| GPU tests: the kit runs only MLX-free tests | none | turboexl3's GPU tests still need a consumer surface of their own (see [Tests](README.md#tests)) |
