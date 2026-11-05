# Consuming turboexl3

This page tells a [consumer](glossary.md#consumer) how to build turboexl3 into its own build. The first section
gives what every consumer supplies. The second section describes the recommended consumer: the mlx-serve
[quant plugin](glossary.md#quant-plugin) in the folder `plugin/`. A later section can describe another consumer
in the same way.

## What every consumer supplies

turboexl3 is one Zig module, and its root file is `src/root.zig`. Its files import each other by relative path.
Apart from these files, they import only three modules:

| module | who supplies it | what it holds |
|---|---|---|
| `std` | Zig | the standard library |
| `sdk` | the consumer | `sdk.mlx`: mlx-c's bindings, plus `check`, `getShape` and `streamIsGpu`; `sdk.kernel_launch`: `newKernel`, `Launch`, `TemplateArg`, `buildConfig` and `applyConfig`, and optionally `mlx_fast_metal_kernel_free`; optionally `sdk.steel`, the steel headers of the consumer's MLX; optionally `sdk.shared`, a host's `Shared` (mlx-serve's `sdk/util/shared.zig`), which makes each of turboexl3's module globals the host's one copy: the host then registers `turboexl3.shared_state` and `turboexl3_table.shared_state` ([Wiring](#wiring) gives the full surface) |
| `mlx_host` | the consumer, for the tests only | the [host surface](glossary.md#host-surface); the tests use only its `sushi_exl3`, where it exists, as a second reference |

The `build.zig` of the package gives a consumer two things:

- The module `turboexl3`, over `src/root.zig`. The consumer adds its `sdk` to this module as an import.
  [Wiring](#wiring) shows the three lines.
- The function `testModule`. It creates the module of the test root, `src/tests.zig`, over the consumer's `sdk` and
  `mlx_host`.

The test binary must also link MLX, the consumer's launcher and Metal. The GPU tests also need a Metal device (see
[Tests](testing.md)).

The consumer sets the configuration once, before its first call, with `configure` (see
[Configuration](api.md#configuration)). On a machine that fails the [NAX gate](glossary.md#nax-gate), it
must set `nax` to false.

## Wiring

turboexl3 imports `std` and one module that its consumer supplies, named `sdk`:

```zig
// sdk
pub const mlx = ...;            // mlx-c's bindings (mlx_array, mlx_stream, mlx_reshape, mlx_fast_metal_kernel_config_*, ...)
                                //   plus check(c_int) !void, getShape(mlx_array) []const c_int, streamIsGpu(mlx_stream) bool
pub const kernel_launch = struct {
    pub const TemplateValue = union(enum) { int: i32, dtype: mlx.mlx_dtype };
    pub const TemplateArg = struct { name: [:0]const u8, value: TemplateValue };
    /// one launch: grid (threads), threadgroup, output shapes and dtypes, template arguments
    pub const Launch = struct { grid: [3]u32, threadgroup: [3]u32, out_shapes: []const []const c_int, out_dtypes: []const mlx.mlx_dtype, template: []const TemplateArg };
    /// an MLX fast Metal kernel over `source` with `header` (input / output names in signature order); null when MLX refuses it
    pub fn newKernel(name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool) ?mlx.mlx_fast_metal_kernel;
    /// the MLX config of a launch (the caller frees it)
    pub fn buildConfig(l: Launch) error{MlxError}!mlx.mlx_fast_metal_kernel_config;
    /// one launch on `stream`; `outs` receives new arrays (the caller frees them)
    pub fn applyConfig(kernel: mlx.mlx_fast_metal_kernel, config: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array, outs: []mlx.mlx_array, stream: mlx.mlx_stream) error{MlxError}!void;
};
/// optional: MLX's steel NAX headers from the consumer's pinned MLX (each the file's text: `defines`, `type_traits`,
/// `integral_constant`, `nax`); without it the prefill GEMMs take the vendored copy (`steel_nax.metal`, MLX 0.32.2)
pub const steel = ...;
```

The members of `sdk`:

- `mlx`: mlx-c's bindings, plus three helpers: `check`, `getShape` and `streamIsGpu`.
- `kernel_launch`: `newKernel` makes a kernel, `buildConfig` makes the config of a launch, and `applyConfig` runs one
  launch. An optional `mlx_fast_metal_kernel_free` drops a kernel that `newKernel` made.
- `steel` (optional): the four steel headers of the consumer's own MLX. An MLX update that changes them then reaches
  the kernels. Without `steel`, the GEMMs use `src/core/kernels/steel_nax.metal`.

The plugin's `plugin/sdk.zig` is such a module:

- Its `kernel_launch` gets each kernel from mlx-serve's kernel cache (`Host.kernel`). The cache keeps one kernel
  object for each name and text, for the life of the process.
- It launches each kernel through mlx-serve's launcher (`HostGpu.launch`). The launcher runs MLX's own compiled
  pipeline, but it skips the source generation and hashing that MLX does on each launch. The results are the same.
- Its `mlx` (`plugin/mlx.zig`) declares the mlx-c functions that turboexl3 calls, with Zig types. They resolve
  against the server's libmlxc when the server loads the plugin.
- It declares no `steel`, because the plugin kit has no steel headers.
- It declares no `mlx_fast_metal_kernel_free`, because the server's kernel cache owns every kernel.

turboexl3 keeps its kernels in caches, one per geometry, for the life of the process. It drops a kernel only through
`decode.freeKernel`, on an error path or when it frees a cache. That function calls
`kernel_launch.mlx_fast_metal_kernel_free` when the consumer declares it. mlx-stream's version removes the kernel's
launch-cache entry, then frees the kernel. Without that member, turboexl3 frees no kernel: the consumer keeps them.

A launcher can also add Metal defines to the start of each header. Each header holds the default of each define, and
the defaults are the served values. The plugin passes each header as given. The other values are A/B variants:

| define | default | what it selects | output bits |
|---|---|---|---|
| `EXL3_T1V` | 3 | the route from MUL1's two pair sums to t, from 0 to 3; every route is exact | unchanged |
| `EXL3_R64` | 1 | 1: each lane reads its codewords from one 64-bit window (n ≤ 108) or from three words (n ≤ 128); 0: the reader with a per-lane extract (`IntWords`) | unchanged |
| `EXL3_KU` | 0 | 1: the decode GEMV loads its input tiles in pairs | unchanged |

To use the package, add it to the consumer's `build.zig.zon` and add these lines to its `build.zig`:

```zig
const t3 = b.dependency("turboexl3", .{ .target = target, .optimize = optimize }).module("turboexl3");
t3.addImport("sdk", sdk);
consumer.addImport("turboexl3", t3);
```

[The plugin's section below](#the-mlx-serve-quant-plugin-plugin) shows the wiring of the plugin.

## The mlx-serve quant plugin (`plugin/`)

The folder `plugin/` is turboexl3 as an outside quant plugin of mlx-serve, in the plugin layout of mlx-serve's `main`
(the rewrite of PR 978, merged on 2026-10-27). It builds against the kit of the branch
[`qwen38-exl3-full`](https://github.com/davidtai/mlx-serve/tree/qwen38-exl3-full) of davidtai/mlx-serve, which adds the
host side of an outside quant plugin. That branch is the head of
[ddalcu/mlx-serve#997](https://github.com/ddalcu/mlx-serve/pull/997) before the pull request became the host support of
the `qwen38_exl3` model plugin, which is to replace this quant plugin. mlx-serve's [plugin kit](glossary.md#plugin-kit) builds it into one dylib,
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

The plugin answers each block of a pack with a [claim](glossary.md#claim):

- **A `trunk_quant` block** (the dense linears of the trunk): always `by_marker`. A trunk block has no rate `k`,
  because each linear carries its own rate. No built-in format of mlx-serve runs a dense EXL3 linear.
- **An `expert_quant` block** (the routed banks, with a rate `k`): `by_marker` only when all of these hold. Otherwise
  the claim is `no`, and the built-in Sushi format keeps the banks.
  - The pack is a fully EXL3 Qwen3.8 pack: its config has a `trunk_quant` block, and its `model_type` is `qwen4_exp`
    or `qwen4_exp_text`. Its [numerics](glossary.md#numerics) are reordered. An expert-only pack stays Sushi's.
  - The machine passes the NAX gate.
  - The SwiGLU is not clamped: the model's config has no `swiglu_limit` above 0.
- **Any other block**: `no`. This includes Sushi's STQ family and the other formats.

`by_marker` ranks above Sushi's claim, `by_type`. So the plugin wins each block that it claims, unless another
outside plugin also claims that block at `by_marker`. No environment variable changes a claim.

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
- **`swiglu`** runs turboexl3's `moe` on the three banks of one layer, or `moeClamped` for a clamped SwiGLU (`limit`
  above 0; the claim keeps a clamped pack's banks off this table, so only a model plugin's in-process table gets one). It
  refuses banks that decode differently.
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
| MLX's steel headers: the kit stages only mlx-c's headers | turboexl3 uses its inlined copy, `src/core/kernels/steel_nax.metal` (MLX 0.32.2) | the kit stages the four steel headers of the server's MLX, for `sdk.steel` |
| the NAX gate of the server | `kit/nax.zig` (turboexl3_kit) finds it again from `sysctl`: an Apple M5 or later, macOS 26.2 or later, and no `FORCE_GPU_FAMILY_FALLBACK=1` | a member of the host table that gives the server's answer |
| mlx-c bindings with Zig types: the kit's `mlx_c` comes from translate-C, with integer enums and `[*c]` pointers | `plugin/mlx.zig` declares the functions that turboexl3 calls, as mlx-serve's own bindings declare them | bindings with Zig types in the kit |
| extra imports in `plugin()`: the builder takes only a name | `build.zig` adds `sdk` and `turboexl3` to the root module through `import_table.get("plugin")` | an option of `plugin()` for imports |
| GPU tests: the kit runs only MLX-free tests | none | turboexl3's GPU tests still need a consumer surface of their own (see [Tests](testing.md)) |
