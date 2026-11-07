//! DeepSeek-V4.1's entry points (root re-exports each under its own name): the grouped linear of its trunk (`wo_a`,
//! `grouped.zig`), the clamped routed experts over an expert streamer's slot banks (`banked.zig`) and the shared expert's
//! fused SwiGLU (`f3.zig`). Each runs on root's caches of kernels (`root.kernelsFor`).
const std = @import("std");
const mlx = @import("sdk").mlx;
const root = @import("../../root.zig");
const Global = @import("../../core/global.zig").Global;
const config = @import("../../core/config.zig");
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const grouped = @import("grouped.zig");
const banked = @import("banked.zig");
const f3 = @import("f3.zig");
const Bank = root.Bank;
const Geometry = root.Geometry;
const Proj = root.Proj;
const bankGeometryAs = root.bankGeometryAs;
const freeArr = root.freeArr;
const kernels = root.kernels;
const kernelsFor = root.kernelsFor;
const linearOn = root.linearOn;

// ── The grouped linear ──

/// The grouped linears' kernels per (groups, in, out, rate, decode, dtype): DeepSeek-V4.1's `wo_a` is one shape per
/// activation dtype.
const grouped_lock = Global(std.atomic.Value(u32), "root_grouped_lock", std.atomic.Value(u32).init(0));
const grouped_cache = Global([8]?grouped.Grouped, "root_grouped_cache", @splat(null));

fn groupedFor(shape: grouped.Shape) !*grouped.Grouped {
    while (grouped_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer grouped_lock.get().*.store(0, .release);
    var free_slot: ?usize = null;
    for (&grouped_cache.get().*, 0..) |*slot, i| {
        if (slot.*) |*gr| {
            if (gr.shape.eql(shape)) return gr;
        } else if (free_slot == null) free_slot = i;
    }
    const i = free_slot orelse return error.Exl3EngineCacheFull;
    grouped_cache.get().*[i] = try grouped.Grouped.init(shape);
    return &grouped_cache.get().*[i].?;
}

/// Test seam: the `grouped.Grouped` `linearGrouped` runs a shape's decode rows on (the process's cached one), for a test
/// that sets its launch (`dense_gemv`, `bt_set`, then `resetCfgs`)
pub fn groupedCached(shape: grouped.Shape) !*grouped.Grouped {
    return groupedFor(shape);
}

/// The grouped (block-diagonal) linear, DeepSeek-V4.1's attention `wo_a` (`o = einsum("bsgd,grd->bsgr", o, wo_a)`):
/// y [..., G, out] in x's dtype for x [..., G, in] (bf16, f16 or f32). Slice g of y is the linear of slice g of x through
/// slice g of `p`. `p` holds the G slices stacked: trellis U16/I16 [G, in/16, out/16, n], suh F16 [G, in], svh F16
/// [G, out]. Each slice's values are `linear`'s on it at every row count (at decode rows but for the sign of an exact
/// zero).
/// - Decode rows (<= `kernels.DECODE_ROWS_MAX`): three launches whatever G (`grouped.zig`): the prepare of every slice,
///   the routed down GEMV's gather with row-slot r G + g on slice g, and the routed down finish at top-1.
/// - Prefill rows: `linear` on each slice (views of the stacked arrays, no weight copy), then one MLX concatenate.
pub fn linearGrouped(s: mlx.mlx_stream, x: mlx.mlx_array, p: Proj, dec: format.Decode) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(p.trellis);
    if (xsh.len < 2 or xsh.len > 8 or tsh.len != 4 or tsh[0] < 1 or tsh[1] < 1 or tsh[2] < 1 or tsh[3] < 1) return error.BadExl3Shape;
    const groups = tsh[0];
    const in_dim = tsh[1] * 16;
    const out_dim = tsh[2] * 16;
    const rate = format.Rate.fromN(@intCast(tsh[3])) orelse return error.BadExl3Shape;
    if (xsh[xsh.len - 2] != groups or xsh[xsh.len - 1] != in_dim) return error.BadExl3Shape;
    if (!std.mem.eql(c_int, &.{ groups, in_dim }, mlx.getShape(p.suh)) or !std.mem.eql(c_int, &.{ groups, out_dim }, mlx.getShape(p.svh))) return error.BadExl3Shape;
    const td = mlx.mlx_array_dtype(p.trellis);
    if ((td != .uint16 and td != .int16) or mlx.mlx_array_dtype(p.suh) != .float16 or mlx.mlx_array_dtype(p.svh) != .float16) return error.BadExl3Dtype;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    var rows: c_int = 1;
    for (xsh[0 .. xsh.len - 2]) |d| rows *= d;
    if (rows < 1) return error.BadExl3Shape;
    const shape: grouped.Shape = .{ .groups = @intCast(groups), .in = @intCast(in_dim), .out = @intCast(out_dim), .rate = rate, .dec = dec, .out_dtype = xd };
    if (!shape.admitted()) return error.BadExl3Shape;
    // a row's G slices as one row of G in values
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, groups * in_dim }, 2, s));
    const y2 = if (@as(usize, @intCast(rows)) <= kernels.DECODE_ROWS_MAX)
        try (try groupedFor(shape)).decodeRows(s, x2, p.trellis, p.suh, p.svh, @intCast(rows))
    else
        try groupedPrefill(s, x2, p, shape, rows, dec);
    defer freeArr(y2);
    var osh: [8]c_int = undefined;
    @memcpy(osh[0..xsh.len], xsh);
    osh[xsh.len - 1] = out_dim;
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, osh[0..xsh.len].ptr, @intCast(xsh.len), s));
    return y;
}

/// `linearGrouped` at prefill rows: y [rows, G out] from x2 [rows, G in], `linear` of each slice on its columns of x2 (one
/// copy of them, as any strided x of `linear`), the G outputs side by side. The slices are views: the stacked trellis as
/// [G in/16, out/16, n] and suh / svh flat, each slice one contiguous range.
fn groupedPrefill(s: mlx.mlx_stream, x2: mlx.mlx_array, p: Proj, shape: grouped.Shape, rows: c_int, dec: format.Decode) !mlx.mlx_array {
    const gn: c_int = @intCast(shape.groups);
    const in_dim: c_int = @intCast(shape.in);
    const out_dim: c_int = @intCast(shape.out);
    const tsh = mlx.getShape(p.trellis);
    var t3 = mlx.mlx_array_new();
    defer freeArr(t3);
    try mlx.check(mlx.mlx_reshape(&t3, p.trellis, &[_]c_int{ gn * tsh[1], tsh[2], tsh[3] }, 3, s));
    var suh1 = mlx.mlx_array_new();
    defer freeArr(suh1);
    try mlx.check(mlx.mlx_reshape(&suh1, p.suh, &[_]c_int{gn * in_dim}, 1, s));
    var svh1 = mlx.mlx_array_new();
    defer freeArr(svh1);
    try mlx.check(mlx.mlx_reshape(&svh1, p.svh, &[_]c_int{gn * out_dim}, 1, s));
    var ys: [grouped.max_groups]mlx.mlx_array = undefined;
    var made: usize = 0;
    defer for (ys[0..made]) |a| freeArr(a);
    for (0..shape.groups) |gi| {
        const g: c_int = @intCast(gi);
        // x's columns, the trellis's input-tile rows, suh's and svh's ranges of slice g
        var v: [4]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (v) |a| freeArr(a);
        try mlx.check(mlx.mlx_slice(&v[0], x2, &[_]c_int{ 0, g * in_dim }, 2, &[_]c_int{ rows, (g + 1) * in_dim }, 2, &[_]c_int{ 1, 1 }, 2, s));
        try mlx.check(mlx.mlx_slice(&v[1], t3, &[_]c_int{ g * tsh[1], 0, 0 }, 3, &[_]c_int{ (g + 1) * tsh[1], tsh[2], tsh[3] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try mlx.check(mlx.mlx_slice(&v[2], suh1, &[_]c_int{g * in_dim}, 1, &[_]c_int{(g + 1) * in_dim}, 1, &[_]c_int{1}, 1, s));
        try mlx.check(mlx.mlx_slice(&v[3], svh1, &[_]c_int{g * out_dim}, 1, &[_]c_int{(g + 1) * out_dim}, 1, &[_]c_int{1}, 1, s));
        ys[gi] = try linearOn(s, s, v[0], v[1], v[2], v[3], dec);
        made += 1;
    }
    const vec = mlx.mlx_vector_array_new_data(&ys, made);
    defer _ = mlx.mlx_vector_array_free(vec);
    var y2 = mlx.mlx_array_new();
    errdefer freeArr(y2);
    try mlx.check(mlx.mlx_concatenate_axis(&y2, vec, 1, s));
    return y2;
}

// ── The clamped routed experts over an expert streamer's slot banks ──

/// The most row-slots (routed pairs) a banked call takes (`moeClampedBanked`).
pub const banked_rows_max = banked.banked_rows_max;
/// A banked call's id for a row-slot that runs no expert (`Config.banked_group` 4): no decode, its y row 0.
pub const banked_skip_id = banked.skip_id;

/// The banked gate|up's result, its down's input: `hd` f16 [rows, inter] (the down's prepared input, lane order) and `hs`
/// f32 [rows, inter / 128] (its 128-block sums). The caller frees both.
pub const BankedHalf = struct { hd: mlx.mlx_array, hs: mlx.mlx_array };

/// `moeClamped` over an expert streamer's slot banks at decode (DeepSeek-V4.1's streamed routed experts: mlx-serve's
/// streamed EXL3 source serves a call's routed pairs from three banks, base, ext and transient): row-slot r is one pair,
/// x's row r through the expert at `ids[r]`, packed (bank << 24 | slot row), at weight 1, and y f32 [rows, hidden] holds
/// each pair's output (the caller weighs and sums them). Each pair's bytes are `moeClamped`'s on its bank at top-1 with
/// score 1 (the GPU test holds them so): `decode.Kernels3`'s banked texts, form 7's launches (prep2, gu3, fin7 clamped,
/// down3, dfin3) on form 9's launch shape at that many row-slots, each text reading a row-slot's arrays from its bank where
/// the base text reads them. `banks`: base, ext, transient, each its own expert (slot) count at one geometry and rate,
/// the trellis U16 or I16 (the streamer's code arrays), every trellis one dtype; a bank no id names may repeat another's
/// arrays. `ids` u32 or i32 [rows], rows 1..banked_rows_max; x [rows, hidden] bf16, f16 or f32; `limit` 1..128, else
/// error.InvalidExl3Clamp. `moeClampedBankedGateUp` then `moeClampedBankedDown` are its two halves. The gate|up half ends
/// in fin7cb, which prepares the down's input and so reads each bank's down suh as well as its gate and up arrays: a caller
/// whose arrays land in parts (an expert streamer's gate and up, then its down) has every array in place before it, or runs
/// the whole call at its down. With `Config.banked_group` 4 the GEMVs share a decode among up to four row-slots of one id
/// (two at 2), and a row-slot whose id is `banked_skip_id` runs no expert: its y row is 0. Every other row-slot's bytes
/// are the same at either.
pub fn moeClampedBanked(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int) !mlx.mlx_array {
    return moeClampedBankedSelV(s, x, banks, ids, dec, limit, .configured, config.get().banked_variant_gu, config.get().banked_variant_dn);
}

/// hw4b: which banked texts a call runs (`decode.Kernels3.BankedSel`): the configured ones, today's grouped texts (g4) or
/// the grouped texts with the M = 3 member path (g4m3). Every selection gives a row-slot the same bytes.
pub const BankedSel = banked.Banked.BankedSel;

/// hw2: the grouped banked GEMVs' variants (`Config.banked_variant_gu`, `banked_variant_dn`; `decode.group_variants`),
/// on whichever grouped texts a call runs (today's or hw4b's), and a variant's index by name (null: none).
pub const group_variants = banked.group_variants;
pub const groupVariantOf = banked.groupVariantOf;

/// `moeClampedBanked` on `sel`'s texts at the configured variants (hw4b's check mode's calls, the tests).
pub fn moeClampedBankedSel(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int, sel: BankedSel) !mlx.mlx_array {
    return moeClampedBankedSelV(s, x, banks, ids, dec, limit, sel, config.get().banked_variant_gu, config.get().banked_variant_dn);
}

/// `moeClampedBanked` on the configured texts with the grouped gate|up GEMV of variant `vgu` and the grouped down GEMV
/// of `vdn` (`group_variants`; 0 today's, any other needs `Config.banked_group` 4; hw2's check mode's calls): the same
/// bytes (a variant changes a GEMV's schedule, never a decoded value or a sum).
pub fn moeClampedBankedV(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int, vgu: u8, vdn: u8) !mlx.mlx_array {
    return moeClampedBankedSelV(s, x, banks, ids, dec, limit, .configured, vgu, vdn);
}

/// `moeClampedBanked` on `sel`'s texts with the grouped GEMVs of variants `vgu` (gate|up) and `vdn` (down) on them: the
/// same bytes for every selection and variant.
pub fn moeClampedBankedSelV(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int, sel: BankedSel, vgu: u8, vdn: u8) !mlx.mlx_array {
    const h = try moeClampedBankedGateUpSelV(s, x, banks, ids, dec, limit, sel, vgu);
    defer freeArr(h.hd);
    defer freeArr(h.hs);
    return moeClampedBankedDownSelV(s, h, banks, ids, dec, limit, sel, vdn);
}

/// `moeClampedBanked`'s gate|up: prep2b, gu3b and fin7cb (`decode.Kernels3.bankedGateUp`). It reads every bank's gate and
/// up arrays and its down suh (fin7cb prepares the down's input).
pub fn moeClampedBankedGateUp(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int) !BankedHalf {
    return moeClampedBankedGateUpSelV(s, x, banks, ids, dec, limit, .configured, config.get().banked_variant_gu);
}

/// `moeClampedBankedGateUp` on `sel`'s texts at the configured variant.
pub fn moeClampedBankedGateUpSel(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int, sel: BankedSel) !BankedHalf {
    return moeClampedBankedGateUpSelV(s, x, banks, ids, dec, limit, sel, config.get().banked_variant_gu);
}

/// `moeClampedBankedGateUp` with the grouped gate|up GEMV of `variant` (`moeClampedBankedV`).
pub fn moeClampedBankedGateUpV(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int, variant: u8) !BankedHalf {
    return moeClampedBankedGateUpSelV(s, x, banks, ids, dec, limit, .configured, variant);
}

/// `moeClampedBankedGateUp` on `sel`'s texts with the grouped gate|up GEMV of `variant` on them.
pub fn moeClampedBankedGateUpSelV(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int, sel: BankedSel, variant: u8) !BankedHalf {
    if (limit <= 0 or limit > 128) return error.InvalidExl3Clamp;
    return bankedGateUpAct(s, x, banks, ids, dec, .{ .limit = @intCast(limit) }, sel, variant);
}

/// A banked call's SwiGLU: its clamp `limit` (0: none) and where it clamps (`format.ClampAt`). DeepSeek-V4.1's entries
/// pass `limit` 1..128 at `gate_in`; Ling-3.1's (`models/ling/api.zig`) 0, or 1..128 at `silu_out`.
pub const BankedAct = struct { limit: u32, clamp: format.ClampAt = .gate_in };

/// The banked gate|up at `act` (`moeClampedBankedGateUpSelV` for every SwiGLU the banked texts hold): fin7cb's text is the
/// geometry's (`banked.zig` `finText`), every other launch the same.
pub fn bankedGateUpAct(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, act: BankedAct, sel: BankedSel, variant: u8) !BankedHalf {
    const c = try bankedCall(banks, ids, dec, act);
    const xs = mlx.getShape(x);
    if (xs.len != 2 or xs[0] != @as(c_int, @intCast(c.rows)) or xs[1] != @as(c_int, @intCast(c.geo.hidden))) return error.BadExl3Shape;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    const k3 = try kernelsFor(c.geo, .float32);
    const hd = try (try banked.forKernels(k3)).bankedGateUpSelV(k3, s, x, ids, &c.kb, c.rows, sel, variant);
    return .{ .hd = hd[0], .hs = hd[1] };
}

/// The LUT-routed gate|up's result (`bankedGateUpLutAct`): the half, then the row-slots' packed ids, the down's `ids`.
/// The caller frees all three.
pub const BankedLutHalf = struct { hd: mlx.mlx_array, hs: mlx.mlx_array, slots: mlx.mlx_array };

/// The banked gate|up with DEVROUTE's two takes in its prepare (`banked.Banked.bankedGateUpLutSelV`): x [tokens, hidden]
/// (bf16, f16 or f32), `idx` [tokens, topk] or [tokens · topk] (i32 or u32) the routed experts, every one below `lut`'s
/// length, and `lut` u32 [experts] the layer's packed ids (bank << 24 | slot row) by expert. Row-slot r is token
/// r / topk's pair r % topk, and its bytes are `bankedGateUpAct`'s over x's rows taken per pair with ids the LUT taken at
/// `idx`; `slots` holds those ids. Rows 1..banked_rows_max; the grouped texts without gu8b, else
/// error.Exl3EngineNotImplemented.
pub fn bankedGateUpLutAct(s: mlx.mlx_stream, x: mlx.mlx_array, idx: mlx.mlx_array, lut: mlx.mlx_array, banks: *const [3]Bank, dec: format.Decode, act: BankedAct, sel: BankedSel, variant: u8) !BankedLutHalf {
    if (act.limit > 128) return error.InvalidExl3Clamp;
    const xs = mlx.getShape(x);
    if (xs.len != 2 or xs[0] < 1) return error.BadExl3Shape;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    const is = mlx.getShape(idx);
    if (is.len < 1 or is.len > 2 or is[0] < 1 or (is.len == 2 and is[0] != xs[0])) return error.BadExl3Shape;
    const idd = mlx.mlx_array_dtype(idx);
    if (idd != .uint32 and idd != .int32) return error.BadExl3Dtype;
    const tokens: usize = @intCast(xs[0]);
    const rows: usize = mlx.mlx_array_size(idx);
    if (rows > banked_rows_max or rows % tokens != 0) return error.BadExl3Shape;
    const ls = mlx.getShape(lut);
    if (ls.len != 1 or ls[0] < 1) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(lut) != .uint32) return error.BadExl3Dtype;
    const c = try bankedCallRows(banks, rows, dec, act);
    if (xs[1] != @as(c_int, @intCast(c.geo.hidden))) return error.BadExl3Shape;
    const k3 = try kernelsFor(c.geo, .float32);
    const r = try (try banked.forKernels(k3)).bankedGateUpLutSelV(k3, s, x, idx, lut, @intCast(rows / tokens), &c.kb, rows, sel, variant);
    return .{ .hd = r[0], .hs = r[1], .slots = r[2] };
}

/// `moeClampedBanked`'s down: down3b and dfin3b (`decode.Kernels3.bankedDown`) over the gate|up's `h` -> y f32 [rows,
/// hidden]; the caller frees y (and `h`).
pub fn moeClampedBankedDown(s: mlx.mlx_stream, h: BankedHalf, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int) !mlx.mlx_array {
    return moeClampedBankedDownSelV(s, h, banks, ids, dec, limit, .configured, config.get().banked_variant_dn);
}

/// `moeClampedBankedDown` on `sel`'s texts at the configured variant.
pub fn moeClampedBankedDownSel(s: mlx.mlx_stream, h: BankedHalf, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int, sel: BankedSel) !mlx.mlx_array {
    return moeClampedBankedDownSelV(s, h, banks, ids, dec, limit, sel, config.get().banked_variant_dn);
}

/// `moeClampedBankedDown` with the grouped down GEMV of `variant` (`moeClampedBankedV`).
pub fn moeClampedBankedDownV(s: mlx.mlx_stream, h: BankedHalf, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int, variant: u8) !mlx.mlx_array {
    return moeClampedBankedDownSelV(s, h, banks, ids, dec, limit, .configured, variant);
}

/// `moeClampedBankedDown` on `sel`'s texts with the grouped down GEMV of `variant` on them.
pub fn moeClampedBankedDownSelV(s: mlx.mlx_stream, h: BankedHalf, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, limit: c_int, sel: BankedSel, variant: u8) !mlx.mlx_array {
    if (limit <= 0 or limit > 128) return error.InvalidExl3Clamp;
    return bankedDownAct(s, h, banks, ids, dec, .{ .limit = @intCast(limit) }, sel, variant);
}

/// The banked down at `act` (`moeClampedBankedDownSelV` for every SwiGLU the banked texts hold); its kernels are the
/// gate|up's geometry's, so a call passes the `act` its gate|up took.
pub fn bankedDownAct(s: mlx.mlx_stream, h: BankedHalf, banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, act: BankedAct, sel: BankedSel, variant: u8) !mlx.mlx_array {
    const c = try bankedCall(banks, ids, dec, act);
    const r: c_int = @intCast(c.rows);
    const w: c_int = @intCast(c.geo.inter);
    if (!std.mem.eql(c_int, mlx.getShape(h.hd), &.{ r, w }) or !std.mem.eql(c_int, mlx.getShape(h.hs), &.{ r, @divExact(w, 128) })) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(h.hd) != .float16 or mlx.mlx_array_dtype(h.hs) != .float32) return error.BadExl3Dtype;
    const k3 = try kernelsFor(c.geo, .float32);
    return (try banked.forKernels(k3)).bankedDownSelV(k3, s, h.hd, h.hs, ids, &c.kb, c.rows, sel, variant);
}

/// A banked call's checked operands: the decode geometry (top-1, the clamp), its row-slots and the banks as the kernels
/// take them.
const BankedCall = struct { geo: decode.Geometry, rows: usize, kb: [3]decode.Kernels.Bank };

fn bankedCall(banks: *const [3]Bank, ids: mlx.mlx_array, dec: format.Decode, act: BankedAct) !BankedCall {
    if (act.limit > 128) return error.InvalidExl3Clamp;
    const is = mlx.getShape(ids);
    if (is.len != 1 or is[0] < 1 or is[0] > banked_rows_max) return error.BadExl3Shape;
    const idd = mlx.mlx_array_dtype(ids);
    if (idd != .uint32 and idd != .int32) return error.BadExl3Dtype;
    return bankedCallRows(banks, @intCast(is[0]), dec, act);
}

/// `bankedCall` at `rows` row-slots, the ids checked by the caller
fn bankedCallRows(banks: *const [3]Bank, rows: usize, dec: format.Decode, act: BankedAct) !BankedCall {
    const geo = try bankedGeometry(banks);
    if (geo.gate_up.n != geo.down.n) return error.Exl3EngineNotImplemented;
    var kb: [3]decode.Kernels.Bank = undefined;
    for (banks, &kb) |b, *o| o.* = .{
        .gate_t = b.gate.trellis,
        .gate_suh = b.gate.suh,
        .gate_svh = b.gate.svh,
        .up_t = b.up.trellis,
        .up_suh = b.up.suh,
        .up_svh = b.up.svh,
        .down_t = b.down.trellis,
        .down_suh = b.down.suh,
        .down_svh = b.down.svh,
    };
    return .{
        .geo = .{ .hidden = @intCast(geo.hidden), .inter = @intCast(geo.width), .topk = 1, .rate = geo.gate_up, .dec = dec, .limit = act.limit, .clamp = act.clamp },
        .rows = rows,
        .kb = kb,
    };
}

/// The banks' one geometry: each bank as `bankGeometry` checks it, its trellis in bank 0's gate trellis dtype (U16 or
/// I16), every bank at bank 0's widths and rates (its own expert count).
fn bankedGeometry(banks: *const [3]Bank) !Geometry {
    const gs = mlx.getShape(banks[0].gate.trellis);
    if (gs.len != 4 or gs[1] <= 0 or gs[1] > std.math.maxInt(c_int) / 16) return error.BadExl3Shape;
    const td = mlx.mlx_array_dtype(banks[0].gate.trellis);
    if (td != .uint16 and td != .int16) return error.BadExl3Dtype;
    const geo = try bankGeometryAs(banks[0], gs[1] * 16, td);
    for (banks[1..]) |b| {
        const g = try bankGeometryAs(b, geo.hidden, td);
        if (g.width != geo.width or g.gate_up.n != geo.gate_up.n or g.down.n != geo.down.n) return error.BadExl3Shape;
    }
    return geo;
}

/// The shared expert's fused SwiGLU at decode rows (`f3.zig`)
pub const sharedSwigluTakes = f3.sharedSwigluTakes;
pub const sharedSwiglu = f3.sharedSwiglu;

/// The process-wide state here (root's `shared_state` lists it)
pub const shared_state = .{ grouped_lock, grouped_cache };
