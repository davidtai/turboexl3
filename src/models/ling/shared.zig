//! Ling-3.1-flash's shared expert at decode rows (`sharedExpert`), on turboexl3's dense linear (`core/dense.zig`): the
//! gate and the up of one input x in one DEC-ONE launch, each member in its own buffers, whose output blocks' finishes,
//! taken in pairs (the gate's 128-block and the up's), run the host's SwiGLU and the down linear's prepare in the
//! pair's second finish (`text_sgu`, `text_sguco`); then the down's DEC-ONE GEMV (`dense.text_dec0`). Two dependent
//! launches, beside the counters' zeroing, where the composed shared expert takes four levels (the gate and the up, the
//! SwiGLU, the down's prepare, its GEMV). The bytes of `linear(gate)`, `linear(up)`, the host's SwiGLU and
//! `linear(down)`:
//! - limit 0: mlx-serve's fusedSwiGLU in the activation dtype T, act = T(g * tab[g]), h = T(act * u), `tab` MLX's
//!   sigmoid of every 16-bit pattern of T, in T (the host's own table);
//! - limit L > 0 (Ling-3.1's `share_expert_swiglu_limit_list`): the plugin's swigluClampOut, in f32: act = g * sig(g),
//!   h = T(min(act, L) * clip(u, -L, L)), `tab` MLX's f32 sigmoid of every 16-bit pattern of T taken to f32.
//! Decode rows only: 1 to DEC-ONE's `dec_rows` (4), where the joint's prepare fits threadgroup memory and the down's
//! DEC-ONE call takes its prepare in a launch of its own; a call it does not serve returns
//! error.Exl3EngineNotImplemented before any launch (`sharedTakes` says which), and the caller composes the chain.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const root = @import("../../root.zig");
const Proj = root.Proj;

fn freeArr(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

// ── texts ──

/// the member choice after DEC-ONE's head (the suh, trellis base, svh and output tiles of the threadgroup's member) and
/// the GEMV's tile address in that member's own trellis (in place of `dense.dec_tile`)
pub const msel = @embedFile("kernels/shared/msel.metal");
pub const mtile = @embedFile("kernels/shared/mtile.metal");
/// the member's finish stored in y (in place of `dense.dec_finish_store`: the member's svh), then the pair's arrival
/// and the opening of its second finish (the block's gate and up values, row by row)
pub const sgu_store = @embedFile("kernels/shared/sgu_store.metal");
pub const sgu_pair_1 = @embedFile("kernels/shared/sgu_pair_1.metal");
/// the host's SwiGLU of the pair's values: unclamped in the activation dtype (mlx-serve's fusedSwiGLU), or clamped
/// after the SiLU in f32 (the plugin's swigluClampOut)
pub const sgu_swiglu = @embedFile("kernels/shared/sgu_swiglu.metal");
pub const sgu_swiglu_clamp_out = @embedFile("kernels/shared/sgu_swiglu_clamp_out.metal");
/// the down linear's prepare of the pair's block (text_prep_s's statements, lane order, its block sum)
pub const sgu_pair_2 = @embedFile("kernels/shared/sgu_pair_2.metal");

/// DEC-ONE with the prepare inside (`dense.text_dec1`) after its head, the member chosen, the tile read in the member's
/// trellis, the finish stored in y, then the pair's arrival up to the SwiGLU
const sgu_body = dense.dec_prep1 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive_1 ++ mtile ++ dense.dec_gemv_arrive_2 ++ dense.dec_xsum1 ++ dense.dec_finish_v ++ sgu_store ++ sgu_pair_1;
/// the gate and the up in one launch, their pairs' unclamped SwiGLU and the down's prepare: y T [rows, 2 F] (the
/// members' outputs), z, xsx (DEC-ONE's scratch), xp half [rows, F] (lane order) and xs f32 [rows, F / 128]
pub const text_sgu: [:0]const u8 = dense.dec_head ++ msel ++ sgu_body ++ sgu_swiglu ++ sgu_pair_2;
/// the same with the SwiGLU clamped after the SiLU at `lim`
pub const text_sguco: [:0]const u8 = dense.dec_head ++ msel ++ sgu_body ++ sgu_swiglu_clamp_out ++ sgu_pair_2;

const sgu_in = [_][*:0]const u8{ "x", "suh0", "suh1", "w", "w1", "svh0", "svh1", "suh2", "sigtab", "cnt", "pcnt" };
const sguco_in = [_][*:0]const u8{ "x", "suh0", "suh1", "w", "w1", "svh0", "svh1", "suh2", "sig32", "lim", "cnt", "pcnt" };
const sgu_out = [_][*:0]const u8{ "y", "z", "xsx", "xp", "xs" };

// ── kernels and configs: made at their first call ──

const Variant = enum(u8) { plain, clamp_out };
const kernel_lock = Global(std.atomic.Value(u32), "ling_shared_kernel_lock", std.atomic.Value(u32).init(0));
const kernels = Global([2]?mlx.mlx_fast_metal_kernel, "ling_shared_kernels", @splat(null));

fn kernelOf(v: Variant) !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    const slot = &kernels.get().*[@intFromEnum(v)];
    if (slot.*) |made| return made;
    const made = switch (v) {
        .plain => ks.newKernel("turboexl3_ling_sgu", &sgu_in, &sgu_out, text_sgu, decode.header5, true),
        .clamp_out => ks.newKernel("turboexl3_ling_sguco", &sguco_in, &sgu_out, text_sguco, decode.header5, true),
    } orelse return error.KernelCreateFailed;
    slot.* = made;
    return made;
}

/// One call shape's launch configs (cached): the three counters' zeroing (the joint's arrivals, its pairs', the down's)
/// and the joint's launch
const Cfgs = struct {
    key: Key,
    zero_cnt: mlx.mlx_fast_metal_kernel_config,
    zero_pcnt: mlx.mlx_fast_metal_kernel_config,
    zero_dcnt: mlx.mlx_fast_metal_kernel_config,
    joint: mlx.mlx_fast_metal_kernel_config,
};
const Key = struct { in: u32, f: u32, out: u32, n: u32, codebook: u8, window: u8, act: mlx.mlx_dtype, rows: u32, variant: Variant };
const cfg_lock = Global(std.atomic.Value(u32), "ling_shared_cfg_lock", std.atomic.Value(u32).init(0));
const cfg_cache = Global([16]?Cfgs, "ling_shared_cfg_cache", @splat(null));

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernels, cfg_lock, cfg_cache };

// ── the plan ──

/// A call `sharedExpert` serves: its rows, the joint's DEC-ONE shape (the members' one K split, row chunks, tiles), the
/// counters' sizes, and the down's `Dense` and its DEC-ONE configs at those rows.
pub const Plan = struct {
    rows: usize,
    in: u32,
    f: u32,
    out: u32,
    n: u32,
    ks: u32,
    rb: u32,
    bt: u32,
    chunks: u32,
    ncnt: u32,
    npcnt: u32,
    ndcnt: u32,
    nx: u32,
    down: *dense.Dense,
    dc: dense.Dense.DecCfg,
};

fn rowsOf(x: mlx.mlx_array, in: u32) ?usize {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh[xsh.len - 1] != @as(c_int, @intCast(in))) return null;
    var rows: usize = 1;
    for (xsh[0 .. xsh.len - 1]) |d| {
        if (d < 1) return null;
        rows *= @intCast(d);
    }
    return rows;
}

fn projOk(p: Proj, in: u32, out: u32) bool {
    const t = mlx.getShape(p.trellis);
    if (t.len != 3 or t[0] * 16 != @as(c_int, @intCast(in)) or t[1] * 16 != @as(c_int, @intCast(out))) return false;
    const td = mlx.mlx_array_dtype(p.trellis);
    if (td != .uint16 and td != .int16) return false;
    if (mlx.getShape(p.suh).len != 1 or mlx.getShape(p.suh)[0] != @as(c_int, @intCast(in)) or mlx.mlx_array_dtype(p.suh) != .float16) return false;
    if (mlx.getShape(p.svh).len != 1 or mlx.getShape(p.svh)[0] != @as(c_int, @intCast(out)) or mlx.mlx_array_dtype(p.svh) != .float16) return false;
    return true;
}

/// The plan for x through gate and up (in -> F each, one rate and trellis dtype) and the down (F -> out), or null where
/// `sharedExpert` does not serve the call: x bf16 or f16, decode rows the down's DEC-ONE takes with its prepare in a
/// launch of its own, the members' one K split, the joint's prepare inside its launch.
pub fn planFor(x: mlx.mlx_array, gate: Proj, up: Proj, down: Proj, dec: format.Decode) !?Plan {
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    const tg = mlx.getShape(gate.trellis);
    const td = mlx.getShape(down.trellis);
    if (tg.len != 3 or td.len != 3) return null;
    const in: u32 = @intCast(tg[0] * 16);
    const f: u32 = @intCast(tg[1] * 16);
    const out: u32 = @intCast(td[1] * 16);
    if (!projOk(gate, in, f) or !projOk(up, in, f) or !projOk(down, f, out)) return null;
    if (mlx.mlx_array_dtype(gate.trellis) != mlx.mlx_array_dtype(up.trellis)) return null;
    if (tg[2] != mlx.getShape(up.trellis)[2]) return null;
    if (in % 128 != 0 or f % 128 != 0 or out % 128 != 0 or (in / 16) % 4 != 0 or (f / 16) % 4 != 0) return null;
    const rate = format.Rate.fromN(@intCast(tg[2])) orelse return null;
    const drate = format.Rate.fromN(@intCast(td[2])) orelse return null;
    const rows = rowsOf(x, in) orelse return null;
    const d = try root.denseCached(.{ .in = f, .out = out, .rate = drate, .dec = dec, .out_dtype = xd });
    if (!d.decTakes(rows)) return null;
    const dc = try d.decCfgs(rows);
    if (dc.prep_in) return null;
    const ksv = dense.Dense.ksFor(in, f);
    const rows_u: u32 = @intCast(rows);
    const rb = dense.Dense.rbFor(rows_u);
    const bt = dense.Dense.btFor(rb);
    const joint: dense.Shape = .{ .in = in, .out = 2 * f, .rate = rate, .dec = dec, .out_dtype = xd };
    if (!dense.Dense.decPrepFits(joint, ksv, rb, bt)) return null;
    const chunks = (rows_u + rb - 1) / rb;
    const nob = (2 * f) / 128;
    const nb = in / 128;
    return .{
        .rows = rows,
        .in = in,
        .f = f,
        .out = out,
        .n = @intCast(tg[2]),
        .ks = ksv,
        .rb = rb,
        .bt = bt,
        .chunks = chunks,
        .ncnt = @max(nob * chunks, 8),
        .npcnt = @max((f / 128) * chunks, 8),
        // the down's DEC-ONE counters at its own chunks (`Dense.decCfgs`)
        .ndcnt = @max((out / 128) * ((rows_u + dense.Dense.rbFor(rows_u) - 1) / dense.Dense.rbFor(rows_u)), 8),
        .nx = if (dense.Dense.deferFor(dec)) @max(nob * chunks * rb * nb, 8) else 8,
        .down = d,
        .dc = dc,
    };
}

/// Whether `sharedExpert` serves x through gate, up and down (`planFor`)
pub fn sharedTakes(x: mlx.mlx_array, gate: Proj, up: Proj, down: Proj, dec: format.Decode) !bool {
    return (try planFor(x, gate, up, down, dec)) != null;
}

fn zeroCfg(n: u32) !mlx.mlx_fast_metal_kernel_config {
    const shape = [_]c_int{@intCast(n)};
    return ks.buildConfig(.{ .grid = .{ ((n + 127) / 128) * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&shape}, .out_dtypes = &.{.uint32}, .template = &.{.{ .name = "NCNT", .value = .{ .int = @intCast(n) } }} });
}

fn cfgsFor(p: Plan, dec: format.Decode, act: mlx.mlx_dtype, v: Variant) !Cfgs {
    const key: Key = .{ .in = p.in, .f = p.f, .out = p.out, .n = p.n, .codebook = @backingInt(dec.codebook), .window = @intCast(dec.window.bits()), .act = act, .rows = @intCast(p.rows), .variant = v };
    while (cfg_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer cfg_lock.get().*.store(0, .release);
    var free_slot: ?usize = null;
    for (&cfg_cache.get().*, 0..) |*slot, i| {
        if (slot.*) |c| {
            if (std.meta.eql(c.key, key)) return c;
        } else if (free_slot == null) free_slot = i;
    }
    const i = free_slot orelse return error.Exl3EngineCacheFull;
    const r: c_int = @intCast(p.rows);
    const jo: u32 = 2 * p.f;
    const zc = try zeroCfg(p.ncnt);
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(zc);
    const zp = try zeroCfg(p.npcnt);
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(zp);
    const zd = try zeroCfg(p.ndcnt);
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(zd);
    const joint = try ks.buildConfig(.{
        .grid = .{ (jo / 16 / p.bt) * 128, p.chunks, p.ks },
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ r, @intCast(jo) }, &[_]c_int{ @as(c_int, @intCast(p.ks)) * r, @intCast(jo) }, &[_]c_int{@intCast(p.nx)}, &[_]c_int{ r, @intCast(p.f) }, &[_]c_int{ r, @intCast(p.f / 128) } },
        .out_dtypes = &.{ act, .float32, .float32, .float16, .float32 },
        .template = &.{
            .{ .name = "NHW", .value = .{ .int = @intCast(p.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(dec.window.bits()) } },
            .{ .name = "IN", .value = .{ .int = @intCast(p.in) } },
            .{ .name = "OUT", .value = .{ .int = @intCast(jo) } },
            .{ .name = "KS", .value = .{ .int = @intCast(p.ks) } },
            .{ .name = "BT", .value = .{ .int = @intCast(p.bt) } },
            .{ .name = "RB", .value = .{ .int = @intCast(p.rb) } },
            .{ .name = "ROWS", .value = .{ .int = r } },
            .{ .name = "DEFER", .value = .{ .int = if (dense.Dense.deferFor(dec)) 1 else 0 } },
            .{ .name = "OUT_T", .value = .{ .dtype = act } },
            .{ .name = "OB1", .value = .{ .int = @intCast(p.f / 128) } },
        },
    });
    const c: Cfgs = .{ .key = key, .zero_cnt = zc, .zero_pcnt = zp, .zero_dcnt = zd, .joint = joint };
    cfg_cache.get().*[i] = c;
    return c;
}

/// One element as [1] (MLX binds a 0-d input as a scalar reference, not a pointer)
fn lim1(limit: u32) !mlx.mlx_array {
    const v: f32 = @floatFromInt(limit);
    const a = mlx.mlx_array_new_data(&v, &[_]c_int{1}, 1, .float32);
    if (a.ctx == null) return error.MlxError;
    return a;
}

/// The host's table for `sharedExpert` at a limit: MLX's sigmoid of every 16-bit pattern of `act` (bf16 or f16), in
/// `act` (limit 0: what mlx-serve's fusedSwiGLU reads) or of the patterns taken to f32, in f32 (limit > 0: what the
/// plugin's swigluClampOut's f32 sigmoid gives), evaluated on `s` (a GPU stream: the devices round apart). The caller
/// frees it.
pub fn sigmoidTable(s: mlx.mlx_stream, act: mlx.mlx_dtype, clamped: bool) !mlx.mlx_array {
    if (act != .bfloat16 and act != .float16) return error.BadExl3Dtype;
    const pats = try std.heap.page_allocator.alloc(u16, 1 << 16);
    defer std.heap.page_allocator.free(pats);
    for (pats, 0..) |*v, i| v.* = @intCast(i);
    // every pattern as an `act` value (the array takes the words as they are)
    const vals = mlx.mlx_array_new_data(pats.ptr, &[_]c_int{1 << 16}, 1, act);
    if (vals.ctx == null) return error.MlxError;
    defer freeArr(vals);
    var src = mlx.mlx_array_new();
    defer freeArr(src);
    if (clamped) try mlx.check(mlx.mlx_astype(&src, vals, .float32, s)) else try mlx.check(mlx.mlx_array_set(&src, vals));
    var t = mlx.mlx_array_new();
    errdefer freeArr(t);
    try mlx.check(mlx.mlx_sigmoid(&t, src, s));
    try mlx.check(mlx.mlx_array_eval(t));
    return t;
}

/// Ling's shared expert of x [..., in] (bf16 or f16) at decode rows: gate and up (in -> F), the host's SwiGLU at
/// `limit` (0 none, 1..128 the clamp after the SiLU) and the down (F -> out), y [..., out] in x's dtype, the composed
/// chain's bytes. `tab` the host's table at that limit (`sigmoidTable`: [65536] in x's dtype at limit 0, f32 above).
/// error.Exl3EngineNotImplemented where `sharedTakes` does not hold, error.InvalidExl3Clamp above 128, before any launch.
pub fn sharedExpert(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Proj, up: Proj, down: Proj, limit: u32, tab: mlx.mlx_array, dec: format.Decode) !mlx.mlx_array {
    if (limit > 128) return error.InvalidExl3Clamp;
    const p = (try planFor(x, gate, up, down, dec)) orelse return error.Exl3EngineNotImplemented;
    const xd = mlx.mlx_array_dtype(x);
    const want_tab: mlx.mlx_dtype = if (limit == 0) xd else .float32;
    if (mlx.mlx_array_dtype(tab) != want_tab or mlx.mlx_array_size(tab) != 1 << 16) return error.BadExl3Dtype;
    const v: Variant = if (limit == 0) .plain else .clamp_out;
    const k = try kernelOf(v);
    const c = try cfgsFor(p, dec, xd, v);
    const xsh = mlx.getShape(x);
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ @intCast(p.rows), @intCast(p.in) }, 2, s));
    // the counters, each zeroed by dense's DEC-ONE zeroing launch (it reads a weight: it waits on nothing)
    var cnt: [3]mlx.mlx_array = undefined;
    var n_cnt: usize = 0;
    defer for (cnt[0..n_cnt]) |a| freeArr(a);
    for ([_]mlx.mlx_fast_metal_kernel_config{ c.zero_cnt, c.zero_pcnt, c.zero_dcnt }) |zc| {
        var o: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(p.down.dec_zero, zc, &.{gate.svh}, &o, s);
        cnt[n_cnt] = o[0];
        n_cnt += 1;
    }
    var lim: ?mlx.mlx_array = null;
    defer if (lim) |a| freeArr(a);
    var jo: [5]mlx.mlx_array = undefined;
    if (v == .plain) {
        try ks.applyConfig(k, c.joint, &.{ x2, gate.suh, up.suh, gate.trellis, up.trellis, gate.svh, up.svh, down.suh, tab, cnt[0], cnt[1] }, &jo, s);
    } else {
        lim = try lim1(limit);
        try ks.applyConfig(k, c.joint, &.{ x2, gate.suh, up.suh, gate.trellis, up.trellis, gate.svh, up.svh, down.suh, tab, lim.?, cnt[0], cnt[1] }, &jo, s);
    }
    defer for (jo) |a| freeArr(a);
    // the down's DEC-ONE GEMV over the pairs' prepared rows (what its prep_sc makes of the host's activation)
    var o: [2]mlx.mlx_array = undefined;
    try ks.applyConfig(p.down.dec0, p.dc.gemv, &.{ jo[3], jo[4], down.trellis, down.svh, cnt[2] }, &o, s);
    freeArr(o[1]);
    defer freeArr(o[0]);
    var osh: [8]c_int = undefined;
    if (xsh.len > osh.len) return error.BadExl3Shape;
    @memcpy(osh[0..xsh.len], xsh);
    osh[xsh.len - 1] = @intCast(p.out);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, o[0], osh[0..xsh.len].ptr, @intCast(xsh.len), s));
    return y;
}

/// The joint launch alone at x's decode rows (a test seam): y T [rows, 2 F] (each member's `linear` columns), xp half
/// [rows, F] and xs f32 [rows, F / 128] (what the down's prep_sc makes of the host's SwiGLU); the caller frees all three.
pub fn jointParts(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Proj, up: Proj, down: Proj, limit: u32, tab: mlx.mlx_array, dec: format.Decode) ![3]mlx.mlx_array {
    if (limit > 128) return error.InvalidExl3Clamp;
    const p = (try planFor(x, gate, up, down, dec)) orelse return error.Exl3EngineNotImplemented;
    const xd = mlx.mlx_array_dtype(x);
    const want_tab: mlx.mlx_dtype = if (limit == 0) xd else .float32;
    if (mlx.mlx_array_dtype(tab) != want_tab or mlx.mlx_array_size(tab) != 1 << 16) return error.BadExl3Dtype;
    const v: Variant = if (limit == 0) .plain else .clamp_out;
    const k = try kernelOf(v);
    const c = try cfgsFor(p, dec, xd, v);
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ @intCast(p.rows), @intCast(p.in) }, 2, s));
    var cnt: [2]mlx.mlx_array = undefined;
    var n_cnt: usize = 0;
    defer for (cnt[0..n_cnt]) |a| freeArr(a);
    for ([_]mlx.mlx_fast_metal_kernel_config{ c.zero_cnt, c.zero_pcnt }) |zc| {
        var o: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(p.down.dec_zero, zc, &.{gate.svh}, &o, s);
        cnt[n_cnt] = o[0];
        n_cnt += 1;
    }
    var lim: ?mlx.mlx_array = null;
    defer if (lim) |a| freeArr(a);
    var jo: [5]mlx.mlx_array = undefined;
    if (v == .plain) {
        try ks.applyConfig(k, c.joint, &.{ x2, gate.suh, up.suh, gate.trellis, up.trellis, gate.svh, up.svh, down.suh, tab, cnt[0], cnt[1] }, &jo, s);
    } else {
        lim = try lim1(limit);
        try ks.applyConfig(k, c.joint, &.{ x2, gate.suh, up.suh, gate.trellis, up.trellis, gate.svh, up.svh, down.suh, tab, lim.?, cnt[0], cnt[1] }, &jo, s);
    }
    freeArr(jo[1]);
    freeArr(jo[2]);
    return .{ jo[0], jo[3], jo[4] };
}
