//! Ling-3.1-flash's entry points (root re-exports each under its own name): its routed experts' SwiGLU, unclamped on most
//! layers and clamped after the SiLU on the late ones (`expert_swiglu_limit_list`: min(silu(g), L) * clamp(u, -L, L),
//! exllamav3's `act_limit` with SiLU), over a stacked bank (`moeLing`) and over an expert streamer's three slot banks
//! (`moeLingBanked`), and over a streamer's slot bank with the prompt rows gathered in the chain's prepare (`moeLingRows`).
//! Every launch is root's routed chain or DeepSeek-V4.1's banked decode; only the SwiGLU line differs
//! (`decode.text_fin7co`, `prefill.text_midco` and the banked `text_fin7b` / `text_fin7cob`). Its trunk's same-input
//! linear pairs run in one DEC-ONE launch through `linearPair` (`pair.zig`), its shared expert at decode rows in two
//! dependent launches of its own (`sharedExpert`, shared.zig), a wide routed call's combine in one launch
//! (`combineRouted`, `combineParts`, combine.zig), and its MoE router at decode rows with the read-ahead predictor's
//! scores in two launches (`router.route`, router.zig): each a host's opt-in. Its MLA attention
//! at decode rows in the latent is `mlaLatentDecode` (`mla_latent.zig`, three launches of its own), and its MLA prompt
//! attention runs as one causal flash launch (`mlaFlashPrefill`, `mla_flash.zig`: MLX's NAX attention at qk 192, v 128).
const std = @import("std");
const mlx = @import("sdk").mlx;
const root = @import("../../root.zig");
const format = @import("../../core/format.zig");
const ds = @import("../deepseek_v41/api.zig");
const combine = @import("combine.zig");
const mla_flash = @import("mla_flash.zig");
const Bank = root.Bank;
const freeArr = root.freeArr;
/// The shared expert at decode rows in two dependent launches (`shared.zig`)
pub const shared = @import("shared.zig");
pub const sharedExpert = shared.sharedExpert;
pub const sharedTakes = shared.sharedTakes;
pub const sigmoidTable = shared.sigmoidTable;

/// The prefill SwiGLU epilogue (`swiglu.zig`): at prompt rows, up's finish running the host's SwiGLU of the gate's value
/// and its own (`upSwiglu`; `swiglu.takes` says where), the composed chain's words.
pub const swiglu = @import("swiglu.zig");
pub const upSwiglu = swiglu.upSwiglu;

/// Same-input linear pairs in one DEC-ONE launch (`pair.zig`): z | a, the shared expert's and the dense FFN's
/// gate | up, each member in its own buffers, each member's `linear` bytes (`linearPair`).
pub const pair = @import("pair.zig");
pub const linearPair = pair.linearPair;

/// A KDA layer's q|k|v, f and g in one DEC-ONE GEMV launch (`kdaproj.zig`), each member in its own buffers at its own
/// K split, each member's `linear` bytes: after its own prepare launch (`kdaProj`) or over the host's prepared rows
/// (`kdaProjPrepared`).
pub const kdaproj = @import("kdaproj.zig");
pub const kdaProj = kdaproj.kdaProj;
pub const kdaProjPrepared = kdaproj.kdaProjPrepared;
/// `kdaProj` with the layer's b_proj (bf16) as a fourth member of the GEMV launch, its output the served one-row
/// matmul's bytes at every row (`kdaproj.zig`; `kdaproj.bTakes` says where it takes b's weight)
pub const kdaProjB = kdaproj.kdaProjB;
/// A KDA layer's q|k|v, f and g at prompt rows: each member's prepare and W_hat by its `linear`'s launches, then the
/// three products and finishes in one launch (pg2's NAX GEMM and finish over the members), each member's `linear` bytes
/// (`kdaproj_pf.zig`; `kdaproj_pf.takes` says where it runs)
pub const kdaproj_pf = @import("kdaproj_pf.zig");
pub const kdaProjPrompt = kdaproj_pf.kdaProjPrompt;

/// The MoE router at decode rows 1 to 8 in two launches, with the read-ahead predictor's scores (`router.zig`): the
/// router's logits and the predictor's scores in one launch (`router.routerLogits`), the host's fused router over them
/// (`router.select`), or both (`router.route`), the host's ids, bf16 weights and scores word for word.
pub const router = @import("router.zig");

/// The MLA attention at decode rows in the latent (`mla_latent.zig`): the absorbed form over a cache of normed latents and
/// roped shared keys, in f32 (within tolerance of the host's composed latent attention), each row's bits its own.
pub const mla_latent = @import("mla_latent.zig");
pub const mlaLatentDecode = mla_latent.decodeLatent;
pub const mlaLatentTakes = mla_latent.takes;

/// `root.moe` with Ling-3.1's SwiGLU: `limit` 0 none (`moe`'s bytes), 1..128 the clamp after the SiLU (each row-slot's
/// gate SiLU clamped from above at `limit` and its up to [-limit, limit], in f32 on the projections' f16-rounded outputs);
/// else error.InvalidExl3Clamp. Decode rows run form 9's launches (fin7's Ling variant where it clamps), prefill rows the
/// fused chain (the Ling mid). `moe`'s refusals hold. `ref.moeRowClampedAt` at `silu_out` is the reference.
pub fn moeLing(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, limit: c_int) !mlx.mlx_array {
    if (limit < 0 or limit > 128) return error.InvalidExl3Clamp;
    return root.moeSwiglu(s, x, bank, inds, scores, dec, @intCast(limit), .silu_out);
}

/// `moeLing`'s prompt call over gathered rows (an expert streamer's slot bank: the outside plugin turboexl3_ling's prompt
/// calls under MLX_SERVE_LING_ROWS_PATH=1): row-slot p reads x's row `rows[p]` (x [n, hidden], the layer's MoE input) in
/// the chain's prepare and runs the expert at slot `inds[p]` at weight `scores[p]`; y [S, hidden] in `out_dtype`
/// (`root.moeSwigluRows` with Ling-3.1's SwiGLU, `moeLing`'s `limit`: 0 none, 1..128 the clamp after the SiLU, else
/// error.InvalidExl3Clamp). Row p is, bit for bit, row p of `moeLing` at top-1 over `take(x, rows)` in any float dtype
/// holding the same values, its y rounded once to `out_dtype`: the plugin's take, its casts and its call in one call with
/// no gathered copy. The operands and refusals as `root.moeSwigluRows` takes them.
pub fn moeLingRows(s: mlx.mlx_stream, x: mlx.mlx_array, rows: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, limit: c_int, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    if (limit < 0 or limit > 128) return error.InvalidExl3Clamp;
    return root.moeSwigluRows(s, x, rows, bank, inds, scores, dec, @intCast(limit), .silu_out, out_dtype);
}

/// `moeClampedBanked` with Ling-3.1's SwiGLU (`moeLing`'s `limit`): each pair x's row r through the expert at `ids[r]`
/// (bank << 24 | slot row) at weight 1, y f32 [rows, hidden], each pair's bytes `moeLing`'s on its bank at top-1 with
/// score 1. The banks, ids and rows as `moeClampedBanked` takes them; `limit` 0..128, else error.InvalidExl3Clamp.
pub fn moeLingBanked(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int) !mlx.mlx_array {
    const h = try moeLingBankedGateUp(s, x, banks, ids, dec, limit);
    defer freeArr(h.hd);
    defer freeArr(h.hs);
    return moeLingBankedDown(s, h, banks, ids, dec, limit);
}

/// hw4b's selection of the banked texts a call runs (DeepSeek-V4.1's `BankedSel`): the configured ones (a geometry's
/// `Config.banked_group` and `banked_m3` as they were at its first banked call), the grouped texts (g4) or the grouped
/// texts with the M = 3 member path (g4m3). Every selection gives a pair the same bytes.
pub const BankedSel = ds.BankedSel;

/// `moeLingBanked` on `sel`'s texts at the configured variants (`moeClampedBankedSel`'s twin): a test names the texts it
/// runs, which the kernel cache otherwise fixes at a geometry's first call.
pub fn moeLingBankedSel(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int, sel: BankedSel) !mlx.mlx_array {
    const a = try act(limit);
    const h = try ds.bankedGateUpAct(s, x, banks, ids, dec, a, sel, root.config.get().banked_variant_gu);
    defer freeArr(h.hd);
    defer freeArr(h.hs);
    return ds.bankedDownAct(s, h, banks, ids, dec, a, sel, root.config.get().banked_variant_dn);
}

/// `moeLingBanked`'s gate|up (prep2b, gu3b and fin7b or fin7cob, or their grouped texts): as `moeClampedBankedGateUp`.
pub fn moeLingBankedGateUp(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int) !ds.BankedHalf {
    return ds.bankedGateUpAct(s, x, banks, ids, dec, try act(limit), .configured, root.config.get().banked_variant_gu);
}

/// `moeLingBanked`'s down over the gate|up's `h` at the same `limit` -> y f32 [rows, hidden]; the caller frees y.
pub fn moeLingBankedDown(s: mlx.mlx_stream, h: ds.BankedHalf, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int) !mlx.mlx_array {
    return ds.bankedDownAct(s, h, banks, ids, dec, try act(limit), .configured, root.config.get().banked_variant_dn);
}

/// A wide routed call's combine at prompt rows (`combine.zig`): the routed rows times their routing weights, summed over
/// the top-k in one launch, MLX's `sum(multiply(astype(routed, f32), expand_dims(weights, -1)), -2)` word for word:
/// over the joined routed array (`combineRouted`) or the call's unjoined outputs (`combineParts`, JOINLESS).
pub const combineRouted = combine.combineRouted;
pub const combineParts = combine.combineParts;
/// The sources `combineParts` binds at most (the streamer's `joinless_sources`)
pub const combine_max_sources = combine.max_sources;

fn act(limit: c_int) !ds.BankedAct {
    if (limit < 0 or limit > 128) return error.InvalidExl3Clamp;
    return .{ .limit = @intCast(limit), .clamp = .silu_out };
}

/// A MoE layer's tail at decode and verify rows in one launch (`tail.zig`): the routed call's joined rows weighted and
/// summed, rounded to the activation dtype, the shared expert added, then the residual add and the next norm, the words
/// of the served chain (the source's sum, the block's astype and add, mlx-serve's add_norm).
pub const tail = @import("tail.zig");
pub const moeTail = tail.moeTail;
/// `moeTail` over the routed call's unjoined outputs (`tail.zig`): each pair's row read in place at its (source, row),
/// the decode join folded in; the words of `moeTail` over the join.
pub const moeTailParts = tail.moeTailParts;

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ shared.shared_state, pair.shared_state, kdaproj.shared_state, router.shared_state, tail.shared_state, mla_latent.shared_state, mla_flash.shared_state, kdaproj_pf.shared_state, swiglu.shared_state };

/// The MLA prompt attention as one causal flash launch (`mla_flash.zig`): o [B, H, qL, v] = softmax(scale q k^T) v, query
/// row i over keys 0 .. kL - qL + i, at Ling's heads (qk 192, v 128) or any pair `mla_flash.admits` takes; within
/// tolerance of MLX's composed SDPA (a reordering), refused off the NAX gate (error.Exl3EngineNotImplemented).
pub const mlaFlashPrefill = mla_flash.mlaFlashPrefill;
