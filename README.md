# turboexl3

EXL3 trellis-quantized linears and MoE experts on Apple GPUs, as MLX custom Metal kernels. turboexl3 runs a model whose
weights are EXL3 (exllamav3's trellis quantization, turboderp's packs) without dequantizing them to memory: the routed experts
(decode GEMV chains and a fused prefill GEMM chain), the dense trunk linears (attention, GDN and shared-expert projections,
lm_head, MTP), and the fused launches around them that a serving host uses to cut launches per layer. Every kernel is held to
one host reference (`format.project`, Sushi's EXL3 reference, ported): within a stated tolerance where a kernel sums in its own
order, and byte for byte where a kernel replaces a chain of other kernels.

It was written as [mlx-stream](https://github.com/davidtai/mlx-stream)'s EXL3 engine for
[mlx-serve](https://github.com/ddalcu/mlx-serve), where it serves Qwen3.8-Flash-Next on turboderp's 5.05 bpw pack.

## Formats

- **Rates:** every even codeword length n in [16, 128], i.e. K = n / 16 bits per weight from 1 to 8, fractional rates
  (n 36, 40, ...) included. A pack may use one rate per projection; an expert bank's gate, up and down share one.
- **Codebooks:** MUL1 and MCG (exllamav3's `codebook.cuh`), MUL1's affine map optionally deferred to the epilogue.
- **Codeword windows:** 8 to 16 bits (a pack names its window; 16 when it names none).
- **Layout:** a projection is a trellis U16 `[in/16, out/16, n]` (an expert bank `[E, in/16, out/16, n]`) in exllamav3's
  tensor-core tile permutation, with its Hadamard scales `suh` F16 `[in]` and `svh` F16 `[out]` (`[E, in]` / `[E, out]` for a
  bank); widths are multiples of 128.
- **Activations:** bf16, f16 or f32 in; outputs in the input's dtype. Accumulation is f32 throughout; the rounding points are
  exllamav3's (f16 rounds around the 128-point Hadamards).

## API

`src/root.zig` is the module root. The expert entry points mirror Sushi's `sushi_exl3` module, so a host switches engines by
import; everything that cannot serve a call returns `error.Exl3EngineNotImplemented` before any launch, and the caller keeps its
own chain.

- **Experts:** `Spec`, `parseExpertQuant`, `admitTopK`, `trellisAdmitted`, `kernels` (`DECODE_ROWS_MAX`,
  `REDUCE_MAX_TOPK`, `usesPrefillArm`, `rowsOfShape`), `Proj`, `Bank`, `Geometry`, `bankGeometry`; `moe` (decode rows through the
  split-K GEMV chain, prefill rows through the fused NAX GEMM chain), `moeWithShared`; `moeClamped` (GLM's clamped SwiGLU)
  refuses for now.
- **Shared expert and router:** `moeFusedShared` (a gated shared expert inside the routed decode launches),
  `moeFusedSharedRouted` (the same with the router's top-k and softmax folded into the prepare launch).
- **Dense linears:** `linear` and `linearSplit` (up to 16 rows: a multi-row GEMV that decodes each weight once per group of
  up to four rows, in one launch with DEC-ONE; 17 to 32 rows: the register-path GEMM; more: dequantize and MLX's matmul), and
  the pieces a host uses to fuse around them: `linearInnerSplit`, `linearPrepared`, `linearFinish`, `linearPrep`, `linearPiecesBegin`,
  `linearPieceInner`, `linearPiecePrepared`, `linearPiecesFinish`.
- **Attention:** `attnIn` (q, k, v projections with the QK-norm and partial RoPE in the finish), `attnPrep4` and
  `linearPreparedIdx` (the indexer projection sharing that prepare), `linearGated` (o_proj over the sigmoid-gated core output);
  at decode rows the merges of `decode_merge` (`decodeMergeGated`, `decodeMergeQmergeGated` with the QSA core's split-K
  merge, `decodeMergeQPlanes`, `decodeMergeIndexer`).
- **Resident trunk:** `dense.residentArm` keeps each admitted trellis's dequantized W_hat after its first prefill call
  (`residentRelease`, `residentPending`, `residentStats`, `residentKeeps`), same bytes as dequantizing per call; `trunkUnrot`
  (`UnrotStats`) replaces them with the linears out of the Hadamard basis in bf16 (opt-in: a different rounding), joined per
  input (`dense.residentUnrot`, `residentStd`, `residentJoined`), with `attnInStd` and `linearGatedStd` over it.
- **Pieces:** the namespaces `format` (the reference), `spec`, `lane` (the codeword readers), `decode`, `dense`, `prefill`,
  `attn_io`, `decode_merge`, `unfused`, `ref`.

Each function's doc comment states its shapes, dtypes, the chain whose bytes it gives and when it refuses.

## Wiring

turboexl3 imports `std` and one module its consumer supplies, named `sdk`:

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
```

mlx-stream's plugin SDK is such a module: its `kernel_launch` launches through the plugin's launch cache (MLX's own pipeline,
without regenerating and hashing the source per launch; same results) and prefixes `EXL3_T1V`, `EXL3_R64` and `EXL3_KU`
defines from the environment onto each header (a launcher that does not leaves the headers' defaults). A plain launcher over
mlx-c's `mlx_fast_metal_kernel_new` / `_apply` serves too. turboexl3 frees its kernels with `sdk.mlx`'s
`mlx_fast_metal_kernel_free` (its per-geometry caches hold them for the process).

In a consumer's `build.zig.zon`, add the package; in its `build.zig`:

```zig
const t3 = b.dependency("turboexl3", .{ .target = target, .optimize = optimize }).module("turboexl3");
t3.addImport("sdk", sdk);
consumer.addImport("turboexl3", t3);
```

[CONSUMING.md](CONSUMING.md) shows mlx-stream's wiring.

## Defaults and switches

The defaults are the served configuration; each switch is read once per process.

| switch | default | effect |
|---|---|---|
| `MLX_SERVE_TURBOEXL3_DECODE_FORM` | 9 | decode chain form 3, 5, 6, 7, 8 or 9 (7, 8 and 9 give form 3's bytes in fewer dependent launches) |
| `MLX_SERVE_TURBOEXL3_SHARED_SPT` | 4 | gate\|up K slices per GEMV threadgroup of the fused shared expert (launch shape only) |
| `MLX_SERVE_TURBOEXL3_PF_ALAY` | 2 | prefill GEMMs' A operands in fragment order per 16 (2) or 32 (1) elements, or natural (0); same bytes |
| `MLX_SERVE_TURBOEXL3_PF_T128` | on | each expert's full 128-row runs in 128-row prefill tiles; `0` off; same bytes |
| `MLX_SERVE_DEC_ONE` | on | dense decode linears in one launch; `0` the three-launch chain; same bytes |
| `MLX_SERVE_TURBOEXL3_PREFILL` | on | `0` hands prefill rows back to the caller |

The remaining `MLX_SERVE_TURBOEXL3_*` knobs (prefill tile shapes, GEMV tiles, the deferred MUL1 map, the dense paths) are
launch-shape and A/B switches documented where they are read (`root.zig`, `dense.zig`).

## Tests

`src/tests.zig` is the test root. CPU tests (texts, plans, tables, the reference) always run; the GPU parity tests run with
`TURBOEXL3_GPU=1` on a Metal device; `TURBOEXL3_METAL_DUMP=1` prints every kernel as a standalone Metal translation unit for an
offline `xcrun metal` compile; `TURBOEXL3_BENCH=1`, `TURBOEXL3_PF_TIME=1`, `TURBOEXL3_DEC_TIME=1` and `TURBOEXL3_DENSE_TIME=1`
time the chains (against Sushi's where the host exports `sushi_exl3`). The tests import `sdk` and `mlx_host`, the host
surface (`mlx_host.sushi_exl3`, when present, is the second reference), so a consumer builds them over its own surface:
`build.zig`'s `testModule`.

## License

MIT (LICENSE). Third-party material and its licenses: NOTICE.
