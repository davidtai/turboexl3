# turboexl3

Rules for anyone changing this repository, people and coding agents alike. CLAUDE.md points here.

## Metal lives in .metal files

No Metal inside Zig or any other code file: no kernel body, helper block, statement, splice needle or replacement, test
pin or dump preamble in a string literal. Every Metal text is a file under `src/kernels/` (the module's directory;
`test/pins/` for what a test searches the kernels for, `test/dump/` for the TURBOEXL3_METAL_DUMP units), imported with
`@embedFile`.

- A variant is a composition, never a search/replace over kernel text: cut the base at the splice point into numbered
  pieces (`x_1.metal`, `x_2.metal`, ...) and concatenate (`base = x_1 ++ x_2 ++ x_3`, `variant = x_1 ++ insert ++ x_3`).
- A test reads the statements it pins from its pin file (`test_support.pins`), or uses the piece it pins.
- `python3 tools/check_texts.py <src before the change>` fails on any Zig string that holds Metal and proves a refactor
  left every kernel source and header byte-identical, or equal but for comments and whitespace (each such kernel listed
  by name), and lists every kernel a change removed or added (`ADDED_KERNELS`); with the current `src` as both trees it
  runs its scans alone.

## Kernels are dropped through the consumer

A kernel `ks.newKernel` made is dropped with `decode.freeKernel`, never with mlx-c's `mlx_fast_metal_kernel_free`: the
consumer's kernel API decides what a drop does (mlx-stream's launch cache evicts the kernel, then frees it; the plugin's
server keeps every kernel for the process). `tools/check_texts.py` fails on a direct call.
