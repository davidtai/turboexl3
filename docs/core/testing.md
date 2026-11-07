# Tests

`src/tests.zig` is the test root. The tests import `sdk` and `mlx_host`, the host surface. When the host exports
`mlx_host.sushi_exl3`, Sushi is a second reference. So a consumer builds the tests over its own surface, with the
`testModule` function of `build.zig`. The plugin kit cannot run them: a kit plugin builds only its dylib and its
MLX-free tests.

The library reads no environment, but the tests do. These variables turn on test groups:

| variable | file | what it runs |
|---|---|---|
| none | every file | the CPU tests: the kernel texts, the launch plans, the tables and the reference |
| `TURBOEXL3_GPU=1` | `gpu_test.zig`, `prefill_test.zig`, `attn_io_test.zig`, `decb_test.zig`, `clamped_test.zig`, `deepseek_test.zig`, `ling_test.zig`, `shared_test.zig` (Ling's), `rows_test.zig` (Ling's), `combine_test.zig`, `topk_test.zig`, `bench.zig` | the GPU parity and byte tests, on a Metal device |
| `TURBOEXL3_METAL_DUMP=1` | `prefill.zig` | prints each kernel of the routed prefill chain, with its option variants, as one standalone source file for an offline `xcrun metal` compile |
| `TURBOEXL3_BENCH=1` | `bench.zig` | with `TURBOEXL3_GPU=1`: times the form-3 variants of the decode chain against Sushi's chain |
| `TURBOEXL3_PF_TIME=1`, `TURBOEXL3_DEC_TIME=1` | `prefill_test.zig` | with `TURBOEXL3_GPU=1`: time the prefill and decode chains against Sushi's |
| `TURBOEXL3_DS_TIME=1` | `deepseek_test.zig` | with `TURBOEXL3_GPU=1`: times DeepSeek-V4.1's clamped routed MoE (K3, K2; decode rows and a 4096-row chunk), its K6 trunk linears and the grouped `wo_a` |
| `TURBOEXL3_LING_TIME=1` | `ling_test.zig`, `rows_test.zig` | with `TURBOEXL3_GPU=1`: a diagnostic that asserts nothing: times every Ling-3.1-flash trunk linear (rows 1 to 4, 4096 and 16384), `moeLing` (top-8 at rows 1 to 4 and 4096; the outside plugin's top-1 prompt call of 4096 pairs sorted by slot) and `moeLingBanked` (the pairs of 1 to 3 decode rows), then the decode forward's linears chained over the 72 layers (trunk, routed and whole-layer arms at rows 1 to 3); and a sweep of the dense decode GEMV's launch shapes per trunk shape class on the chain at rows 1 to 8 (each choice's chain time, the fastest per class and row count, all of a row count's together; every choice's bytes against the served chain's); and the prompt call over gathered rows (`moeLingRows`) against the plugin's chunk (take, f32, `moeLing`) at its call shape, with both outputs' differing words |
| `TURBOEXL3_DENSE_TIME=1` | `gpu_test.zig` | with `TURBOEXL3_GPU=1`: times the dense GEMV against the row-slot GEMV and MLX's 8-bit affine matmul |
| `TURBOEXL3_REAL_SHARD=<file>` | `gpu_test.zig` | with `TURBOEXL3_GPU=1`: runs `linear` on two real trellises of a trunk shard of the full pack |
| `TURBOEXL3_PERM_FILE=<file>` | `gpu_test.zig` | a CPU-only tool: the column order of the trunk, the EXL3 pack against the affine pack |

These variables set or narrow the cases of a test group. The comments of each file give their values:

| variables | file | what they narrow |
|---|---|---|
| `TURBOEXL3_BENCH_N`, `_ROWS`, `_ARMS`, `_ROUNDS` | `bench.zig` | the decode benchmark |
| `TURBOEXL3_PF_E`, `_ROWS`, `_N`, `_CALLS`, `_ROUNDS`, `_SKEW`, `_ARMS`, `_INDS`, `_STAGES` | `prefill_test.zig` | the prefill parity and timing |
| `TURBOEXL3_DEC_E`, `_ROWS`, `_LAYERS`, `_ROUNDS`, `_INDS` | `prefill_test.zig` | the decode timing |
| `TURBOEXL3_DENSE_TIME_PREFILL`, `_TAIL`, `_MULTI_ONLY` | `gpu_test.zig` | the dense timing: prompt-chunk rows, the rows of a snapshot tail, or the multi-row GEMV alone |
| `TURBOEXL3_SHARED_V2` | `gpu_test.zig` | the configurations of the v2 shared heads |
| `TURBOEXL3_ATTN_ROWS` | `attn_io_test.zig` | the row counts of the attention tests |
| `TURBOEXL3_CLAMP_N`, `_ROWS` | `clamped_test.zig` | the rates (halfwords per tile) and row counts of the clamped SwiGLU's test at DeepSeek-V4.1-Flash's geometry |

`python3 tools/check_texts.py <src before the change>` checks a refactor on the host. It fails on any Zig string that
holds Metal. It also proves that each kernel source and header is byte-identical to the one before the change, or
equal except for comments and whitespace. It names each kernel of the second kind, each kernel that the change
removed, and each kernel that the change added (its list `ADDED_KERNELS`). With the current `src` as both trees, it
runs only the scan for Metal.

## The kernel docs check

`python3 tools/kernel_docs.py --check` (or `zig build doc-check`) fails when a kernel the code can build has no entry in
the docs, an entry names no kernel, an entry's generated block is not the code's, or one of its prose fields is missing.
`python3 tools/kernel_docs.py --write` rewrites every generated block from the code and adds a skeleton entry for a new
kernel; [the kernel catalog](kernels.md) says how an entry reads.

## The consumer check

`zig build consumer-check -Dconsumers=<plugin folder>[,<plugin folder>...]` builds each consumer plugin against this tree
in place of its pinned turboexl3 (`zig build --fork=<this repo>`), as its own CI builds it: its test step (plugin.json's
`test_step`, else `test`), then its ReleaseFast dylib, with its repository's toolchain. The consumers are
CommunityServe's `plugins/turboexl3_qwen38` and `plugins/turboexl3_ds41`. That compiles `turboexl3_table`
(`table/`) and `turboexl3_kit` (`kit/`), which need the host's quant ABI and plugin kit, and every library entry a
consumer reaches. The suite's own build cannot: Zig analyzes only what is referenced, and the tests reference what they
call. The test root also references every pub declaration of the module root and the namespaces it exports
(`refAllDeclsRecursive`, "turboexl3 module: every pub declaration compiles"), so an entry no test calls still
compiles. No GPU work; a change runs both checks with the suite.
