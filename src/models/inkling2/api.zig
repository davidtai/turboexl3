//! Inkling 2's entry points (root re-exports each under its own name): its routed experts' softcapped SwiGLU,
//! cg tanh(g / cg) sigmoid(g) * cu tanh(u / cu) in f32 with the sigmoid on the raw gate (transformers'
//! Inkling2SoftcappedSwiGLU; the pack's text_config `mlp_gate_softcap` 4 and `mlp_up_softcap` 25), over a stacked bank
//! (`moeInkling2`), over an expert streamer's three slot banks (`moeInkling2Banked`) and over a streamer's slot bank with
//! the prompt rows gathered in the chain's prepare (`moeInkling2Rows`). Every launch is root's routed chain or
//! DeepSeek-V4.1's banked decode; only the SwiGLU line differs (`decode.text_fin7cs`, the prefill mids' `cs` texts and
//! the banked `text_fin7csb` / `text_fin7csbg`). The caps travel as the geometry's `limit` (`format.softcapLimit`), so
//! a softcapped geometry caches its own kernels beside the clamped and plain ones.
const mlx = @import("sdk").mlx;
const root = @import("../../root.zig");
const format = @import("../../core/format.zig");
const ds = @import("../deepseek_v41/api.zig");
const Bank = root.Bank;
const freeArr = root.freeArr;

/// The caps as the entries take them (`format.Softcap`); whole numbers 1..`format.softcap_max`, else
/// error.InvalidExl3Clamp before any operand is read.
pub const Softcap = format.Softcap;

fn limitOf(caps: Softcap) !u32 {
    return format.softcapLimit(caps) orelse error.InvalidExl3Clamp;
}

/// `root.moe` with Inkling 2's softcapped SwiGLU at `caps`: decode rows run form 9's launches with fin7's softcap variant,
/// prefill rows the fused chain with the softcap mid. `moe`'s refusals hold (prefill rows need `Config.prefill` and
/// `Config.nax`; gate, up and down at one rate); the compact mixed outputs and int8 are off for it, as for Ling's clamp.
/// `ref.moeRowSoftcap` is the reference.
pub fn moeInkling2(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, caps: Softcap) !mlx.mlx_array {
    return moeInkling2With(s, x, bank, inds, scores, dec, caps, .{});
}

/// `moeInkling2` with the call's own choices (`prefill.Call`, `root.moeSwigluWith`): `.{ .dfin = false }` keeps a prompt
/// call off the fused finish, the same bytes.
pub fn moeInkling2With(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, caps: Softcap, call: root.prefill.Call) !mlx.mlx_array {
    return root.moeSwigluWith(s, x, bank, inds, scores, dec, try limitOf(caps), .softcap, call);
}

/// `moeInkling2`'s prompt call over gathered rows (`root.moeSwigluRows` at the softcap): row-slot p reads x's row
/// `rows[p]` (x [n, hidden]) in the chain's prepare and runs the expert at slot `inds[p]` at weight `scores[p]`; y
/// [S, hidden] in `out_dtype`. Row p is, bit for bit, row p of `moeInkling2` at top-1 over `take(x, rows)`. The operands
/// and refusals as `root.moeSwigluRows` takes them (S more than `kernels.DECODE_ROWS_MAX`).
pub fn moeInkling2Rows(s: mlx.mlx_stream, x: mlx.mlx_array, rows: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, caps: Softcap, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    return moeInkling2RowsWith(s, x, rows, bank, inds, scores, dec, caps, out_dtype, .{});
}

/// `moeInkling2Rows` with the call's own choices (`prefill.Call`), as `moeInkling2With` is `moeInkling2`'s.
pub fn moeInkling2RowsWith(s: mlx.mlx_stream, x: mlx.mlx_array, rows: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, caps: Softcap, out_dtype: mlx.mlx_dtype, call: root.prefill.Call) !mlx.mlx_array {
    return root.moeSwigluRowsWith(s, x, rows, bank, inds, scores, dec, try limitOf(caps), .softcap, out_dtype, call);
}

/// `moeClampedBanked` with Inkling 2's softcapped SwiGLU: each pair x's row r through the expert at `ids[r]` (bank << 24 |
/// slot row) at weight 1, y f32 [rows, hidden], each pair's bytes `moeInkling2`'s on its bank at top-1 with score 1. The
/// banks, ids and rows as `moeClampedBanked` takes them. The caller frees y.
pub fn moeInkling2Banked(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, caps: Softcap) !mlx.mlx_array {
    const h = try moeInkling2BankedGateUp(s, x, banks, ids, dec, caps);
    defer freeArr(h.hd);
    defer freeArr(h.hs);
    return moeInkling2BankedDown(s, h, banks, ids, dec, caps);
}

/// `moeInkling2Banked` on `sel`'s texts (`moeLingBankedSel`'s twin): a test names the texts it runs, which the kernel
/// cache otherwise fixes at a geometry's first call.
pub fn moeInkling2BankedSel(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, caps: Softcap, sel: ds.BankedSel) !mlx.mlx_array {
    const a = try act(caps);
    const h = try ds.bankedGateUpAct(s, x, banks, ids, dec, a, sel, root.config.get().banked_variant_gu);
    defer freeArr(h.hd);
    defer freeArr(h.hs);
    return ds.bankedDownAct(s, h, banks, ids, dec, a, sel, root.config.get().banked_variant_dn);
}

/// `moeInkling2Banked`'s gate|up (prep2b, gu3b and fin7csb, or their grouped texts): as `moeClampedBankedGateUp`.
pub fn moeInkling2BankedGateUp(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, caps: Softcap) !ds.BankedHalf {
    return ds.bankedGateUpAct(s, x, banks, ids, dec, try act(caps), .configured, root.config.get().banked_variant_gu);
}

/// `moeInkling2Banked`'s down over the gate|up's `h` at the same `caps` -> y f32 [rows, hidden]; the caller frees y.
pub fn moeInkling2BankedDown(s: mlx.mlx_stream, h: ds.BankedHalf, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, caps: Softcap) !mlx.mlx_array {
    return ds.bankedDownAct(s, h, banks, ids, dec, try act(caps), .configured, root.config.get().banked_variant_dn);
}

fn act(caps: Softcap) !ds.BankedAct {
    return .{ .limit = try limitOf(caps), .clamp = .softcap };
}
