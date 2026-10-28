# Results

Measured with mlx-serve on one Apple M5 Max (128 GB, macOS 27.0.1), serving [Qwen3.8-Flash-Next](GLOSSARY.md#qwen38)
from the MLX repacks of [turboderp](GLOSSARY.md#turboderp)'s packs (the table in [the README](../README.md#the-packs)).
The same tables are in [ddalcu/mlx-serve#997](https://github.com/ddalcu/mlx-serve/pull/997), the pull request that
serves these packs through turboexl3's [quant plugin](GLOSSARY.md#quant-plugin).

## Across context lengths

One run per cell. Each prompt is the same fixed text (Python source with one coding task) for every pack and arm, sent as
one user message in Qwen's chat template with thinking on, greedy, seed 1234, at most 1,024 new tokens. Each run gets a
fresh server after a warm-up boot on the same prompt and a 40 °C cool gate. Server flags: `--ctx-size 262144
--max-concurrent 1 --prefix-cache-mem 1GB` and the MTP flags below, with `MLX_SERVE_NOCACHE_LOAD=1 MLX_SERVE_NGRAM_WARM=0
MLX_SERVE_GPU_CEILING_MB=114688 MLX_SERVE_CACHE_LIMIT=2147483648`. The rates come from the server's own request line.

- **K2 to K5** are the 2.05, 3.05, 4.05 and 5.05 packs, run on mlx-serve's tree from before its rewrite (PR 978) with
  turboexl3 inside. "Sushi" is [exact numerics](GLOSSARY.md#numerics) (`MLX_SERVE_BIT_EXACT=1`:
  [Sushi](GLOSSARY.md#sushi)'s routed experts, every reordering path off), and "turboexl3" is the default reordered
  numerics, on one binary. K5 ran on build S21 (turboexl3's kernels as of 4a4ea95, with the host fixes of #997) at
  [MTP](GLOSSARY.md#mtp) depth 4 (`--mtp-max-depth 4 --mtp-typical 0.2`), with turboderp's MTP mixer. K2 to K4 ran on
  the earlier build PR17B at MTP depth 3 and were not re-measured.
- **mixed 4/8-bit** is the standard mlx-serve setup, with no turboexl3: upstream main (edebbb3) with only #997's two
  load-time switches, `MLX_SERVE_NOCACHE_LOAD` and `MLX_SERVE_NGRAM_WARM`. It runs ddalcu's
  `Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` at mlx-serve's default MTP depth (auto: it ran depth 6), one arm. Its
  `generation_config.json` gained `default_chat_template_kwargs: {"enable_thinking": true}`, which the EXL3 repacks carry,
  so every column renders the same prompt.
- `*`: the answer reached the 1,024-token cap. `†`: a single run flagged as noisy (K2 8K, K4 1K and 2K), not re-run. With
  MTP, a decode rate also depends on how many drafts its answer accepts, and the arms give different answers.
- MTP off was measured on K5 only.

Prefill, tok/s (MTP on):

| prompt | tokens | mixed 4/8-bit | K2 Sushi | K2 turboexl3 | K3 Sushi | K3 turboexl3 | K4 Sushi | K4 turboexl3 | K5 Sushi | K5 turboexl3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 1,031 | 1,372.9 | 1,648.9 | 1,603.2 | 1,580.5 | 1,572.3 | 1,025.5† | 1,373.0† | 955.7 | 1,093.7 |
| 2K | 2,013 | 1,868.1 | 2,147.0 | 2,114.1 | 2,057.0 | 2,080.1 | 1,916.3† | 1,423.7† | 1,161.2 | 1,625.5 |
| 4K | 4,025 | 2,311.0 | 2,518.1 | 2,554.9 | 2,464.7 | 2,497.5 | 1,938.8 | 2,116.1 | 1,884.1 | 1,987.2 |
| 8K | 8,100 | 2,658.5 | 2,412.1† | 2,809.4† | 2,498.3 | 2,769.8 | 2,488.6 | 2,711.0 | 2,018.0 | 2,555.2 |
| 16K | 16,275 | 2,658.5 | 2,714.5 | 2,771.8 | 2,599.3 | 2,834.5 | 2,538.2 | 2,830.0 | 2,264.2 | 2,711.8 |
| 32K | 32,424 | 2,761.2 | 2,664.4 | 2,767.0 | 2,578.8 | 2,766.3 | 2,492.8 | 2,759.3 | 2,334.2 | 2,674.0 |
| 64K | 64,987 | 2,724.2 | 2,543.3 | 2,680.0 | 2,484.4 | 2,659.6 | 2,485.6 | 2,639.1 | 2,242.2 | 2,550.7 |
| 128K | 129,949 | 2,604.0 | 2,451.4 | 2,544.4 | 2,399.0 | 2,522.2 | 2,398.9 | 2,512.4 | 2,179.0 | 2,500.4 |
| 256K | 258,992 | 2,477.7 | 2,318.8 | 2,362.3 | 2,273.7 | 2,348.0 | 2,270.0 | 2,344.0 | 2,073.1 | 2,329.1 |

Decode, tok/s (MTP on):

| prompt | tokens | mixed 4/8-bit | K2 Sushi | K2 turboexl3 | K3 Sushi | K3 turboexl3 | K4 Sushi | K4 turboexl3 | K5 Sushi | K5 turboexl3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 1,031 | 106.6 | 126.2 | 131.5 | 131.8 | 140.0 | 119.7† | 141.0† | 126.8 | 134.9 |
| 2K | 2,013 | 95.2 | 119.6 | 123.6\* | 126.6 | 131.3\* | 114.2† | 134.8† | 122.0\* | 126.5\* |
| 4K | 4,025 | 111.4 | 113.3 | 125.3 | 124.5 | 135.2 | 116.5 | 132.3 | 123.6 | 138.2 |
| 8K | 8,100 | 96.1 | 118.2† | 127.6† | 128.2 | 134.4 | 116.1 | 135.7 | 123.1 | 126.6 |
| 16K | 16,275 | 112.8\* | 110.7 | 120.6 | 126.6 | 132.3 | 112.5 | 137.6 | 122.3 | 124.2\* |
| 32K | 32,424 | 105.0 | 107.5 | 117.8 | 117.8 | 128.5 | 108.7 | 126.4 | 114.3 | 122.8 |
| 64K | 64,987 | 107.7 | 108.5 | 110.9 | 112.0 | 116.6 | 102.5 | 119.5 | 107.8 | 113.6 |
| 128K | 129,949 | 99.1 | 98.9 | 101.3 | 106.3 | 110.4 | 96.8 | 114.3 | 103.6 | 108.9 |
| 256K | 258,992 | 82.4 | 91.8 | 96.0 | 93.6 | 99.4 | 89.6 | 105.5 | 92.3 | 95.2 |

MTP off (`--no-mtp`), tok/s:

| prompt | tokens | prefill, K5 Sushi | prefill, K5 turboexl3 | decode, K5 Sushi | decode, K5 turboexl3 |
|---|---:|---:|---:|---:|---:|
| 1K | 1,031 | 1,014.1 | 1,157.9 | 56.8 | 55.9 |
| 2K | 2,013 | 1,439.9 | 1,578.1 | 56.0\* | 55.0\* |
| 4K | 4,025 | 1,756.9 | 2,173.5 | 56.2\* | 55.1\* |
| 8K | 8,100 | 2,211.2 | 2,609.6 | 55.9 | 54.9 |
| 16K | 16,275 | 2,277.3 | 2,751.4 | 55.8 | 54.9 |
| 32K | 32,424 | 2,290.3 | 2,698.0 | 55.1 | 54.2 |
| 64K | 64,987 | 2,269.8 | 2,604.9 | 54.2 | 53.2 |
| 128K | 129,949 | 2,203.3 | 2,503.1 | 53.1 | 52.0 |
| 256K | 258,992 | 2,094.2 | 2,345.8 | 51.9 | 51.2 |

## Coding evals

EvalPlus 0.3.1: HumanEval+ (164 problems) and MBPP+ (378), greedy pass@1, seed 1234, EvalPlus's chat prompt,
`reasoning_effort: low` with `max_tokens` 16,384. Every answer finished on its own.

| Setup | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---:|---:|---:|---:|
| 5.05 pack, turboexl3 4a4ea95 as the quant plugin of #997's head, reordered numerics, MTP depth 4 | 0.988 (162) | 0.945 (155) | 0.955 (361) | 0.812 (307) |
| mixed 4/8-bit, upstream main + the two load switches, MTP auto | 0.988 (162) | 0.945 (155) | 0.952 (360) | 0.815 (308) |

The two setups are within one problem of each other on every suite. Their HumanEval answers differ (74 of 164 solutions
identical) but fail nearly the same problems: 8 of the 9 HumanEval+ failures are shared.
