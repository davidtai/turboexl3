# Inkling 2

Inkling 2 (Thinking Machines, `Inkling2ForConditionalGeneration`, model type `inkling2`) has 48 layers at hidden 4608.
Every layer is MoE: 384 routed experts at width 2304, top-8, and no shared expert. The EXL3 packs
(`OpensourceWTF/Inkling-2-Small-exl3`) use the MCG codebook. The experts have one rate a layer (K1 to K4); the trunk
and the head are at K6 (2.99bpw), K5 (1.99bpw) or K3 (1.02bpw). The outside plugin `turboexl3_inkling2` (mlx-serve) is
the quant host; it streams the routed experts from SSD or keeps them resident.

The routed SwiGLU is softcapped (transformers' `Inkling2SoftcappedSwiGLU`):
`cg · tanh(g / cg) · sigmoid(g) · cu · tanh(u / cu)`, in f32, with the sigmoid on the raw gate. The pack's text_config
gives the caps: `mlp_gate_softcap` 4 and `mlp_up_softcap` 25. A geometry carries the softcap as `format.ClampAt`
`softcap`. Its `limit` holds the two caps, one byte each (`format.softcapLimit`: whole numbers 1 to 255).

What it runs on turboexl3 (`root.inkling2`, `src/models/inkling2/api.zig`):

| part | entry point | kernels |
|---|---|---|
| a routed layer's experts over a stacked bank, any rows | `moeInkling2` (`moeInkling2With`: the call keeps or declines the fused finish) | decode form 9 with `turboexl3_fin7cs`; the prefill chain with `turboexl3_pf_t128_midcs` (`turboexl3_pf_midcs`, `turboexl3_pf_mid_vcs` off the 128-row tiles) |
| a prompt call's routed pairs over a streamer's slot bank, the rows gathered from the layer's MoE input | `moeInkling2Rows` (`moeInkling2RowsWith`) | `moeInkling2`'s prefill chain with the row-map prepare `turboexl3_pf_prep_r` |
| a decode wave's routed pairs over a streamer's three slot banks | `moeInkling2Banked` (`moeInkling2BankedGateUp`, `moeInkling2BankedDown`; `moeInkling2BankedSel` on a selection of the texts) | DeepSeek-V4.1's banked decode with `turboexl3_fin7csb`, or `turboexl3_fin7csbg` under `Config.banked_group` 4 |
| every trunk linear and the head | `root.linear` | the core's dense linear |

The compact mixed outputs (`Config.pf_t128_zfit`) and int8 (`Config.pf_int8`) stay off for a softcapped geometry, as for
Ling's clamp. wave3 (`Config.banked_wave3`) does not take it.

Tests: `src/models/inkling2/inkling2_test.zig` (the texts, the reference, the refusals and the widths on the CPU; the
softcap against `ref.moeRowSoftcap`, the banked pairs and the gathered rows on the GPU) and the K1 cases in
`src/core/gpu_test.zig`.
