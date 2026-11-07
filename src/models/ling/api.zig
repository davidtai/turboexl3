//! Ling-3.1-flash's entry points (root re-exports each under its own name): its routed experts' SwiGLU, unclamped on most
//! layers and clamped after the SiLU on the late ones (`expert_swiglu_limit_list`: min(silu(g), L) * clamp(u, -L, L),
//! exllamav3's `act_limit` with SiLU), over a stacked bank (`moeLing`) and over an expert streamer's three slot banks
//! (`moeLingBanked`). Every launch is root's routed chain or DeepSeek-V4.1's banked decode; only the SwiGLU line differs
//! (`decode.text_fin7co`, `prefill.text_midco` and the banked `text_fin7b` / `text_fin7cob`). Its trunk's same-input
//! linear pairs run in one DEC-ONE launch through `linearPair` (`pair.zig`), its shared expert at decode rows in two
//! dependent launches of its own (`sharedExpert`, shared.zig), a wide routed call's combine in one launch
//! (`combineRouted`, `combineParts`, combine.zig), and its MoE router at decode rows with the read-ahead predictor's
//! scores in two launches (`router.route`, router.zig): each a host's opt-in.
const std = @import("std");
const mlx = @import("sdk").mlx;
const root = @import("../../root.zig");
const format = @import("../../core/format.zig");
const ds = @import("../deepseek_v41/api.zig");
const combine = @import("combine.zig");
const Bank = root.Bank;
const freeArr = root.freeArr;
/// The shared expert at decode rows in two dependent launches (`shared.zig`)
pub const shared = @import("shared.zig");
pub const sharedExpert = shared.sharedExpert;
pub const sharedTakes = shared.sharedTakes;
pub const sigmoidTable = shared.sigmoidTable;

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

/// The residual add and RMSNorm folded into the next linears' prepare at a decode layer's mixer input (`addnorm.zig`):
/// the residual, the normed rows and each EXL3 member's output from one launch of the add, the norm and the members'
/// prepares, then each member's GEMV (`addNormLinears`), each the bytes of MLX's add, MLX's rms_norm and its `linear`.
pub const addnorm = @import("addnorm.zig");
pub const addNormLinears = addnorm.addNormLinears;
pub const addNormPrepare = addnorm.addNormPrepare;

/// The MoE router at decode rows 1 to 8 in two launches, with the read-ahead predictor's scores (`router.zig`): the
/// router's logits and the predictor's scores in one launch (`router.routerLogits`), the host's fused router over them
/// (`router.select`), or both (`router.route`), the host's ids, bf16 weights and scores word for word.
pub const router = @import("router.zig");

/// `root.moe` with Ling-3.1's SwiGLU: `limit` 0 none (`moe`'s bytes), 1..128 the clamp after the SiLU (each row-slot's
/// gate SiLU clamped from above at `limit` and its up to [-limit, limit], in f32 on the projections' f16-rounded outputs);
/// else error.InvalidExl3Clamp. Decode rows run form 9's launches (fin7's Ling variant where it clamps), prefill rows the
/// fused chain (the Ling mid). `moe`'s refusals hold. `ref.moeRowClampedAt` at `silu_out` is the reference.
pub fn moeLing(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, limit: c_int) !mlx.mlx_array {
    if (limit < 0 or limit > 128) return error.InvalidExl3Clamp;
    return root.moeSwiglu(s, x, bank, inds, scores, dec, @intCast(limit), .silu_out);
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

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ shared.shared_state, pair.shared_state, kdaproj.shared_state, addnorm.shared_state, router.shared_state };
