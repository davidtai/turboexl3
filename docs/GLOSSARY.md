# Glossary

This page defines the terms that the turboexl3 documents use: the [README](../README.md),
[KERNELS.md](KERNELS.md), [RESULTS.md](RESULTS.md) and [CONSUMING.md](../CONSUMING.md). Each entry gives the plain
meaning first, then the precise one.

## The weight format

- <a id="exl3"></a>**EXL3**: A format that stores LLM weights in about 1 to 8 bits each. exllamav3 defines it. EXL3
  encodes the weights with QTIP-style trellis codes: each weight has a codeword that a decoder reads from a bitstream.
- <a id="exllamav3"></a>**exllamav3**: turboderp's CUDA library that defines EXL3 and runs EXL3 models on NVIDIA GPUs.
  turboexl3 follows exllamav3 v1.4.2 (`codebook.cuh`, `exl3_dq.cuh`, `pack.cu`, `hadamard.cu` and `quantize.py`).
- <a id="turboderp"></a>**turboderp**: The author of exllamav3. turboderp also publishes EXL3 packs of models.
- <a id="pack"></a>**pack**: A model's weights in EXL3 form. "bpw" is the average bits per weight of the pack, for
  example 5.05 bpw.
- <a id="trellis"></a>**trellis**: The encoded weights of one projection, a `U16` array `[in/16, out/16, n]`. Each 16 ×
  16 tile of the weight matrix is one bitstream of n 16-bit words.
- <a id="tile"></a>**tile**: A 16 × 16 block of the weight matrix, so 256 weights. A tile is the unit that EXL3 encodes
  and decodes.
- <a id="rate"></a>**rate, K, n**: The rate K is the number of bits per weight. n is the number of 16-bit words in one
  encoded tile, so K = n / 16. n is even and from 16 to 128, so K is from 1 to 8. A fractional K is valid, for example K
  = 2.25 at n = 36. n is not a codeword length: every codeword has 16 bits.
- <a id="codeword"></a>**codeword**: The 16 bits of a tile's bitstream that encode one weight. The codeword of weight t
  is the 16-bit window that ends at stream bit `floor((t + 1) × K)`. So consecutive codewords overlap: each weight adds
  K new bits to a sliding window. The stream is circular, so the window of weight 0 takes its first bits from the end of
  the stream.
- <a id="codeword-window"></a>**codeword window (w)**: The number of low bits of each codeword that the decoder uses,
  from 8 to 16. The pack's quantizer searched with that width, so the decoder masks each codeword to its low w bits. The
  bitstream and the bits per weight are the same at every width. A pack that names no width uses 16.
- <a id="codebook"></a>**codebook**: The function that turns a codeword into a weight value. EXL3 computes it: there is
  no lookup table. The function multiplies the codeword by a constant (a hash) and then applies a few bit operations.
  EXL3 has two:
  - <a id="mul1"></a>**MUL1**: The hash is `codeword × 0x83DCD12D` (modulo 2^32). t is the sum of the hash's four bytes,
    from 0 to 1020. The weight is `f16(A × (1024 + t) + B)`, with A = `half` `0x1EEE` and B = `half` `0xC931`.
  - <a id="mcg"></a>**MCG**: The hash is `codeword × 0xCBAC1FED`. Then `0x3B603B60 ^ (hash & 0x8FFF8FFF)` makes two
    `f16` values. The mask keeps the sign bit and the low 12 bits of each 16-bit half, and the XOR sets the other bits.
    The weight is the `f16` sum of the two values.
- <a id="deferred-map"></a>**deferred MUL1 map**: A faster way to apply the MUL1 codebook. A MUL1 weight is affine in t:
  A × t + C, with C = 1024 × A + B = −3.453125. With the map on, a GEMV or GEMM multiplies each input by t and sums. The
  finish then applies `A × acc + C × Σx` once per output, where Σx is the sum of the row's prepared inputs. This uses
  the exact affine weight, but the reference rounds each weight to `f16` first. So the outputs can differ in the last
  bits.
- <a id="permutation"></a>**tensor-core tile permutation**: The order in which exllamav3 stores the 256 weights of a
  tile. It is the order of an NVIDIA tensor-core fragment: code position p holds tile element `perm[p]`
  (`format.tensorCorePerm`). Lane L of a 32-lane simdgroup owns positions 8L to 8L + 7. These are rows
  `2 (L mod 4) + {0, 1, 8, 9}` of the columns `L / 4` and `L / 4 + 8`. So each lane decodes only the weights that its
  share of a GEMV or a matrix-unit fragment needs.
- <a id="h128"></a>**H128**: The 128-point Hadamard transform. It is the Sylvester Hadamard matrix of size 128, with
  entries `(−1)^popcount(i & k)`, times 1/√128. With that scale it is orthonormal and is its own inverse. EXL3 applies
  it to each 128-element block of a vector. The kernels run it as a butterfly of 7 stages. The first 5 stages shuffle
  values across the 32 lanes of a simdgroup, and the last 2 combine the 4 registers of each lane.
- <a id="suh-svh"></a>**suh, svh**: Two `F16` vectors of each projection: `suh` `[in]` for the input channels and `svh`
  `[out]` for the output channels. Each element is a random sign times a channel scale.
- <a id="w-hat"></a>**W_hat, W_std**: W_hat is the weight matrix that the trellis decodes to, `[in, out]`, in the
  rotated basis. W_std is the same projection in the standard basis: `W_std = diag(suh) H W_hat H diag(svh)`, with H the
  block-diagonal H128. So the projection is y = x × W_std.
- <a id="why-rotate"></a>**why EXL3 rotates the inputs and outputs**: The rotation spreads large outlier weights across
  each block. The rotated weights then look like Gaussian noise, which the trellis code fits well. At run time the
  kernels do not build W_std. They rotate the input before the multiply by W_hat and rotate the output after it. Because
  H is its own inverse, the result is the projection by W_std.
- <a id="stages"></a>**prepare, inner product, finish**: The three stages of one EXL3 projection, as `format.project`
  computes them:
  - <a id="prepare"></a>**prepare**: `xp = f16(H128(f16(x) · suh))`, per 128-element block of the input.
  - <a id="inner-product"></a>**inner product**: `z = f16(xp · W_hat)`, with `f32` accumulation.
  - <a id="finish"></a>**finish**: `y = f16(H128(z) · svh)`, per 128-element block of the output, then converted to the
    output dtype.

## Numbers and exactness

- <a id="float-types"></a>**`f32`, `f16`, `bf16`**: Three floating-point types. `f32` is IEEE single precision (24-bit
  significand). `f16` is IEEE half precision (11-bit significand, largest value 65504). `bf16` is bfloat16 (8-bit
  significand, the exponent range of `f32`). The served model's activations are `bf16`. EXL3's intermediate values are
  `f16`.
- <a id="dtype"></a>**dtype**: The element type of an array, for example `bf16`.
- <a id="rounding-point"></a>**rounding point**: A step where a computation rounds a value to a narrower type. The EXL3
  reference rounds to `f16` four times: x, the prepared input, the inner product and the finished output. Then it
  converts to the output dtype. It also rounds each decoded weight to `f16`. Every trellis path of turboexl3 keeps the
  four rounding points of the stages. Only the opt-in un-rotated trunk rounds at other points.
- <a id="format-project"></a>**`format.project`**: The host reference. It is a CPU function in Zig that computes one row
  of one EXL3 projection. It uses exllamav3's rounding points and `f32` arithmetic in a fixed order. It is a port of
  Sushi's `src/exl3/expert_exl3.zig`, and every kernel is held to it.
- <a id="byte-identical"></a>**byte-identical**: Every output element has the same bit pattern as the output of the
  chain that the kernel replaces, on the same inputs. This holds because the kernel runs that chain's arithmetic
  statements, in the same order, on the same values. Only the place where the statements run changes. The GPU tests
  compare the bit patterns.
- <a id="within-tolerance"></a>**within tolerance**: The kernel rounds where the reference rounds, but it sums in a
  different order. Addition in `f32` is not associative: (a + b) + c can differ from a + (b + c) in the last bit. A
  last-bit change can also flip a later `f16` rounding. So the outputs differ slightly from the reference, and a
  tolerance bounds the difference.
- <a id="ref-compare"></a>**`ref.compare`**: The tolerance measure. It computes, in `f64`, the largest |got − ref| over
  the output divided by the root mean square of the reference output. Each kernel states its bar, for example 5e-3.

## GPU, Metal and MLX

- <a id="metal"></a>**Metal**: Apple's interface for GPU programs. Its kernels are in the Metal Shading Language, which
  is based on C++.
- <a id="mlx"></a>**MLX**: Apple's array framework for Apple silicon. Its arrays are lazy: MLX records operations and
  runs them when a result is needed. turboexl3 calls MLX through mlx-c, the C interface of MLX.
- <a id="custom-kernel"></a>**custom Metal kernel**: A kernel that MLX builds from Metal source text.
  `mlx_fast_metal_kernel_new` takes a name, the input and output names, a body and a header. The body holds the
  statements of the kernel function. The header holds helper code that comes before the function. MLX writes the
  function signature from the names.
- <a id="launch-config"></a>**launch config, template arguments**: The settings of one launch: the grid (in threads),
  the threadgroup size, and the output shapes and dtypes. Template arguments are integers and dtypes that become
  compile-time constants in the body. MLX builds and keeps one compiled pipeline for each set of template arguments.
- <a id="launch"></a>**launch**: One call of a GPU kernel, here one application of a custom Metal kernel.
- <a id="dependent-level"></a>**dependent level**: A launch that reads the output of an earlier launch starts one level
  after it. MLX puts a barrier between them. Launches that do not depend on each other share a level and can overlap.
  The number of levels in a chain is its longest path.
- <a id="why-fewer-launches"></a>**why fewer launches and levels are faster at decode**: At decode, one launch does
  little work. So the fixed cost of each launch is a large part of the time. Each level also adds a wait: the next
  launch cannot start until the earlier launch has finished and its writes are visible.
- <a id="threadgroup"></a>**threadgroup**: A group of up to 1024 threads that run on one GPU core. Its threads share
  threadgroup memory and can wait for each other at a barrier. A launch is a grid of threadgroups.
- <a id="threadgroup-memory"></a>**threadgroup memory**: Fast on-chip memory that one threadgroup shares. Apple GPUs
  give a threadgroup at most 32 KB.
- <a id="simdgroup"></a>**simdgroup**: 32 threads of a threadgroup that run in lockstep. They exchange register values
  directly, with shuffles (`simd_shuffle_xor`) and reductions (`simd_sum`), without threadgroup memory or barriers.
- <a id="lane"></a>**lane**: One thread of a simdgroup, numbered 0 to 31 (L in the code).
- <a id="lane-order"></a>**lane order**: An input layout for the GEMV. Within each 16-element tile row, the four
  elements that one lane multiplies (rows `2q + {0, 1, 8, 9}`) are next to each other. So the lane loads them as one
  `half4`.
- <a id="gemv-gemm"></a>**GEMV, GEMM**: A GEMV multiplies a few rows by a weight matrix (matrix-vector product). A GEMM
  multiplies many rows by it (matrix-matrix product). A GEMV uses each weight for few rows, so the reads and decodes of
  weights limit it. A GEMM uses each weight for many rows, so arithmetic limits it.
- <a id="split-k"></a>**split-K, K slice, plane**: Split-K cuts the input dimension (the dimension of the sum, called K
  here) into K slices. Different threadgroups sum different slices. Each slice writes its partial sums to an `f32` plane
  `[slices, rows, out]`. A later finish adds the planes in slice order. At decode, one row gives too few output blocks
  to fill the GPU, so the split adds threadgroups. This K is the input dimension, not the rate K.
- <a id="row-chunk"></a>**row chunk, lean GEMV**: The rows of a call that one threadgroup of the dense GEMV serves
  together. They share one decode of each weight. A lane keeps its sums and inputs for each row of its chunk in
  registers, so the registers limit the rows of a chunk: 5 on the plain GEMV. The lean GEMV
  (`turboexl3_dense_dgemv_lean`) keeps one input per row at a time, so its chunks can have up to 8 rows.
- <a id="nax"></a>**NAX**: The matrix-multiply units ("neural accelerators") in the GPU cores of Apple's M5. MLX uses
  them through the NAX GEMM header of its steel library (`steel/gemm/nax.h`). Its `NAXTile` and `tile_matmad_nax` wrap
  Metal's `mpp::tensor_ops::matmul2d`. turboexl3 carries four steel headers (`defines.h`, `type_traits.h`,
  `integral_constant.h` and `nax.h`), inlined, in `src/kernels/steel_nax.metal` (MLX 0.32.2). A consumer can give the
  headers of its own MLX instead (`sdk.steel`).
- <a id="nax-gate"></a>**NAX gate**: MLX's test for the NAX units: an M5-class GPU and macOS 26.2 or later. Below the
  M5, the GEMMs that use the NAX compute wrong values. So where the gate fails, a consumer sets `Config.nax` to false.
- <a id="mma"></a>**MMA, fragment**: An MMA is one matrix multiply-accumulate instruction on a small tile; here it is 16
  deep along the input dimension. A fragment is the part of an MMA operand that one lane holds in its registers.
- <a id="shared-decode-body"></a>**shared-decode body**: A GEMM body for tiles of 64, 128 or 256 rows. For each 64-deep
  stage of the input dimension, the simdgroups decode the stage's 16 trellis tiles into threadgroup memory once. Then
  every simdgroup multiplies its rows by that stage on the NAX.
- <a id="register-path"></a>**register path**: A GEMM body for tiles of at most 32 rows. Each simdgroup computes one 32
  × 32 output block by itself. It decodes the trellis tiles straight into its B fragment registers, with no threadgroup
  memory and no barriers. Like the shared-decode body, it multiplies on the NAX.

## The model and its serving

- <a id="moe"></a>**mixture of experts (MoE)**: A layer with many small feed-forward networks (experts). A router picks
  a few experts for each token.
- <a id="router"></a>**router, top-k, softmax weights**: The router is a projection that gives one logit per expert.
  Top-k picks the k experts with the largest logits. Their weights are the softmax over all experts, renormalized over
  the k that it picks. The layer output is the weighted sum of the k expert outputs.
- <a id="routed-expert"></a>**routed expert, expert bank**: A routed expert is one of the E experts that the router
  picks from. An expert bank holds one trellis per projection for all experts: `[E, in/16, out/16, n]`.
- <a id="shared-expert"></a>**shared expert**: An expert that every token runs. Qwen3.8-Flash-Next gates it: the layer
  adds sigmoid(logit) × shared(x), where the logit is x times one weight vector, the shared gate.
- <a id="swiglu"></a>**SwiGLU**: The function of each expert: `h = silu(x · W_gate) × (x · W_up)`, then
  `y = h · W_down`, with `silu(g) = g × sigmoid(g)`.
- <a id="row"></a>**row**: One token position of a call. The batch and sequence axes of the input together give the row
  count.
- <a id="row-slot"></a>**row-slot**: One pair of a row and one of its top-k slots. A call of r rows has r × k row-slots.
- <a id="decode-prefill-rows"></a>**decode rows, prefill rows**: Decode makes one token at a time, so a call has 1 row
  or a few verify rows. Prefill reads the prompt in chunks of up to thousands of rows. turboexl3 sends calls of at most
  16 rows (`DECODE_ROWS_MAX`) to its GEMV paths and wider calls to its GEMM paths.
- <a id="why-row-count"></a>**why the row count decides the path**: At a few rows, the reads and decodes of weights take
  most of the time. So a GEMV that decodes each weight once per call, over many threadgroups, is fastest. At many rows,
  arithmetic takes most of the time. So a GEMM on the NAX that decodes each weight tile once per tile of rows is
  fastest.
- <a id="mtp"></a>**MTP, verify rows**: MTP (multi-token prediction) is a small extra head that drafts the next tokens.
  The main model then checks all drafted tokens in one call, one row per token: these are verify rows. At the served MTP
  depth of 4 a verify call has up to 5 rows, and the head's draft steps have 1. The number of verify rows changes from
  call to call. So turboexl3
  keeps the bytes of a row the same at every row count, where this README and KERNELS.md say so.
- <a id="trunk"></a>**trunk**: Every projection that is not a routed expert: attention, GDN, shared expert, `lm_head`
  and MTP. A fully EXL3 model has about a dozen trunk shapes.
- <a id="lm-head"></a>**`lm_head`**: The last projection, from the hidden size to the logits of the vocabulary.
- <a id="gdn"></a>**GDN**: Gated DeltaNet, the model's linear-attention layers. Their projections (`in_proj_qkv`,
  `in_proj_z`, `out_proj`) are trunk projections.
- <a id="qwen38"></a>**Qwen3.8-Flash-Next**: The served model. Its hidden size is 2560. Each MoE layer has 512 routed
  experts (top-10, intermediate width 640) and a gated shared expert. It has GDN layers, 12 full-attention layers and an
  MTP head.
- <a id="attention-pieces"></a>**attention pieces**: The full-attention layers of Qwen3.8-Flash-Next have 24 query heads
  and 2 key-value heads of 256 values. `q_proj` gives each head 512 columns: `[query | gate]`. The fused attention
  kernels serve these pieces:
  - <a id="qk-norm"></a>**QK-norm**: An RMSNorm of each query head and each key head over its 256 values, with a learned
    weight. It runs before RoPE.
  - <a id="rope"></a>**RoPE, partial RoPE**: Rotary position embedding rotates pairs of a head's values by angles that
    depend on the position. Partial RoPE rotates only the first RD values (64 of 256 here). The pairs are value j and
    value j + RD/2 (the NeoX pairing).
  - <a id="output-gate"></a>**sigmoid output gate**: The attention output is multiplied by sigmoid(gate) before
    `o_proj`: y = `o_proj`(attn × sigmoid(gate)).
  - <a id="qsa"></a>**QSA, indexer, pooled key**: QSA is the model's block-sparse attention (the name in the host code).
    The indexer projection gives 4 index query heads and one raw key, 128 values each. The pooled key of a block is the
    mean of the raw keys of its 4 rows (the block ratio), normalized and rotated. Each query keeps the blocks with the
    best index scores, and the attention core reads only those blocks.
  - <a id="qsa-merge"></a>**QSA merge, ring rows**: The attention core runs in splits along the keys. The merge
    (`msv_attn_qsa256_qmerge`) combines the partial sums of the splits. Ring rows are the rows of an incomplete block
    that an earlier call left.
- <a id="head-dim"></a>**head dim (hd)**: The number of values per attention head: 256 for the attention heads, 128 for
  the index heads.

## Engines and hosts

- <a id="host"></a>**host**: The application that calls turboexl3, here mlx-serve. When turboexl3 refuses a call, it
  returns an error that names the reason. The host can then run its own chain for the call.
- <a id="host-surface"></a>**host surface**: The module `mlx_host` that the tests import. It holds the functions and
  types of the host. The tests use only its `sushi_exl3`, where it exists, as a second reference.
- <a id="mlx-serve"></a>**mlx-serve**: David Dalcu's LLM server built on MLX (github.com/ddalcu/mlx-serve). At startup
  it can load outside plugins: dylibs that its plugin kit builds.
- <a id="quant-plugin"></a>**quant plugin**: An outside plugin that adds a weight format to mlx-serve. The server offers
  each quantization block of a pack to every format, and the format answers with a claim. turboexl3's quant plugin is
  the folder `plugin/`.
- <a id="plugin-kit"></a>**plugin kit**: The package that `zig build kit` stages in mlx-serve. It holds the plugin ABI
  and SDK (the module `mlx_serve`), the server's MLX pin, example plugins and the builder `plugin()`. It also holds
  mlx-c's headers, and the builder makes the module `mlx_c` from them.
- <a id="claim"></a>**claim**: A format's answer to one block of a pack: `no`, `by_type` or `by_marker`, in rank order.
  An outside format wins a block only when its claim ranks above every built-in claim. A built-in format wins a tie.
- <a id="numerics"></a>**exact and reordered numerics**: mlx-serve's two numerics modes for EXL3. Exact allows no
  path that reorders a sum, so the routed experts stay with Sushi. Reordered allows such paths. A fully EXL3 Qwen3.8
  pack is reordered and every other pack is exact. The earlier runs set the mode with `MLX_SERVE_BIT_EXACT` (`1` exact,
  `0` reordered).
- <a id="mlx-stream"></a>**mlx-stream**: David Tai's plugin for mlx-serve (github.com/davidtai/mlx-stream). turboexl3
  was written as its EXL3 engine, and then moved out of it into this package.
- <a id="sushi"></a>**Sushi**: Another EXL3 engine for MLX (MIT license), and mlx-serve's built-in EXL3 format.
  turboexl3's expert API mirrors Sushi's `sushi_exl3` module, so a host switches engines by changing one import.
  `format.zig` ports Sushi's reference, and the tests compare with Sushi where the host exports it.

## Zig builds

- <a id="module"></a>**module**: A unit of Zig code with one root file. A build script declares it and gives it a name.
  Code in another module imports it by that name, for example `@import("sdk")`. Inside a module, files import each other
  by path. A path import cannot reach outside the directory of the module's root file.
- <a id="package"></a>**package**: A directory that Zig's package manager can fetch. It holds a `build.zig` and a
  `build.zig.zon`. turboexl3 is a package with one module, `turboexl3`.
- <a id="build-zig"></a>**`build.zig`**: The build script of a package. It declares the modules, the test binaries and
  the named steps that `zig build <step>` runs. An argument `-D<name>=<value>` sets one of its options.
- <a id="build-zig-zon"></a>**`build.zig.zon`**: The manifest of a package. It lists the paths that the package holds
  (`.paths`) and the dependencies, each with a URL and a hash. A fetched package holds only the listed paths.
- <a id="consumer"></a>**consumer**: A package whose build includes turboexl3. It supplies the module `sdk` and, for the
  tests, the host surface. The recommended consumer is the quant plugin in `plugin/` ([CONSUMING.md](../CONSUMING.md)).
- <a id="test-root"></a>**test root**: The root file of a test binary. The binary runs only the tests in its own module,
  not the tests of the modules that it imports.

## Names in this library

- <a id="chain"></a>**chain, unfused chain, fused path**: A chain is a sequence of launches. A fused path is turboexl3's
  replacement for a chain. The unfused chain is the chain that a fused path replaces.
- <a id="form"></a>**form**: One launch arrangement of the decode chain of the routed experts (forms 1, 3 and 5 to 9).
  Form 9 is the served one.
- <a id="dec-one"></a>**DEC-ONE**: The one-launch form of a dense projection at decode rows. One GEMV launch runs the
  GEMV and the finish, and at small sizes the prepare too. The last threadgroup to arrive at each output block runs that
  block's finish.
- <a id="arrival-counter"></a>**arrival counter, last arriver**: A way to run a finish in the launch that computes its
  inputs, so that no later launch waits for them. Each threadgroup writes its part of a block's inputs, then adds 1 to
  the block's integer counter. The threadgroup whose add brings the counter to the number of parts is the last arriver,
  and it runs the finish of the block. Device-scope fences around the add make the writes of the other threadgroups
  visible to it, and no threadgroup waits for another. An earlier launch sets the counters to zero. DEC-ONE, decode
  form 6 and the fin fold of the shared expert (`fused_fin_fold`) use it.
- <a id="attn-in-out"></a>**ATTN-IN, ATTN-OUT**: The fused attention projections at prefill rows. ATTN-IN computes q
  (with the gate), k and v with QK-norm and partial RoPE. ATTN-OUT computes `o_proj` over the gated output of the
  attention core.
- <a id="resident-trunk"></a>**resident trunk**: A mode that the host turns on with `dense.residentArm`. It keeps the
  W_hat of each trunk projection in GPU memory after its first dequantize. Later prefill calls then skip the dequantize
  launch. The outputs are byte-identical.
- <a id="unrotated-trunk"></a>**un-rotated trunk**: A mode that the host turns on at load with `trunkUnrot`. It replaces
  the kept W_hat with W_std in the activation dtype. A prefill call is then one MLX matmul. The rounding differs from
  the trellis paths.
- <a id="variant"></a>**variant**: Another launch arrangement that gives the same bytes, kept for tests and timing
  comparisons.
- <a id="configuration"></a>**configuration**: The choices that a consumer sets once with `configure`, as one `Config`.
  The defaults are the served configuration ([Configuration](../README.md#configuration)).
