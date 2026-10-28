//! turboexl3: EXL3 trellis routed experts on Metal, every admitted rate (even n in [16, 128]: K1..K8, fractional
//! included), both codebooks (MUL1, MCG), every codeword window (8..16), any geometry whose widths are multiples of 128.
//! Its API is the `sushi_exl3` module's (lib/sushi, MIT) so a host switches engines by import: `Spec`, `parseExpertQuant`,
//! `admitTopK`, `trellisAdmitted`, `Bank{gate,up,down: Proj{trellis,suh,svh}}`, `moe`, `moeWithShared`,
//! `moeClamped`, `format.Decode`, `kernels.DECODE_ROWS_MAX`. The reference every kernel is held to is `format.project`.
//!
//! Status: `moe` serves decode rows (<= DECODE_ROWS_MAX) through the form `Config.decode_form` names: forms 3, 7, 8 and 9
//! run form 3's split-K chain (`decode.Kernels3`; 9, the default, in four or five launches), forms 5 and 6 the same bytes
//! in three or two launches with expert-grouped row-slots (`decode.Kernels5`); prefill rows run the fused GEMM chain
//! (`prefill.zig`). The clamped (GLM) form still refuses `error.Exl3EngineNotImplemented`. A consumer's choices are one
//! `Config` (`configure`); the library reads no environment.
const std = @import("std");
const mlx = @import("sdk").mlx;
pub const config = @import("config.zig");
pub const Config = config.Config;
pub const configure = config.configure;
pub const format = @import("format.zig");
pub const spec = @import("spec.zig");
pub const lane = @import("lane.zig");
pub const decode = @import("decode.zig");
pub const ref = @import("ref.zig");
pub const unfused = @import("unfused.zig");
pub const dense = @import("dense.zig");
pub const prefill = @import("prefill.zig");
/// The kernel texts a consumer may splice into its own Metal (MLX-free; `sources.zig`).
pub const sources = @import("sources.zig");
/// The GDN layer's tail in one launch (`gdnOutPrep`).
pub const gdn_out = @import("gdn_out.zig");

pub const Spec = spec.Spec;
pub const parseExpertQuant = spec.parseExpertQuant;
pub const admitTopK = spec.admitTopK;
pub const trellisAdmitted = spec.trellisAdmitted;

/// The dispatch constants a host reads (Sushi's `kernels` namespace names).
pub const kernels = struct {
    pub const DECODE_ROWS_MAX = spec.DECODE_ROWS_MAX;
    pub const REDUCE_MAX_TOPK = spec.REDUCE_MAX_TOPK;
    pub const usesPrefillArm = spec.usesPrefillArm;
    pub const rowsOfShape = spec.rowsOfShape;
};

/// One projection's bank, expert e on axis 0: `trellis` U16 [E, in/16, out/16, n], `suh` F16 [E, in], `svh` F16 [E, out].
pub const Proj = struct { trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array };
pub const Bank = struct { gate: Proj, up: Proj, down: Proj };

/// What a bank's shapes say: expert count, hidden and intermediate widths, the gate/up rate and the down rate.
pub const Geometry = struct {
    experts: c_int,
    hidden: c_int,
    width: c_int,
    gate_up: format.Rate,
    down: format.Rate,
};

/// A projection's shapes and dtypes against the geometry: the refusals Sushi's loader makes, by the same names.
pub fn validateClampedProjection(p: Proj, experts: c_int, input: c_int, output: c_int) !void {
    const sh = mlx.getShape(p.trellis);
    if (input <= 0 or output <= 0 or @mod(input, 128) != 0 or @mod(output, 128) != 0 or
        sh.len != 4 or sh[0] != experts or sh[1] != @divExact(input, 16) or sh[2] != @divExact(output, 16) or sh[3] < 0 or
        format.kFromPackedDim(@intCast(sh[3])) == null or
        !std.mem.eql(c_int, &.{ experts, input }, mlx.getShape(p.suh)) or
        !std.mem.eql(c_int, &.{ experts, output }, mlx.getShape(p.svh))) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(p.trellis) != .uint16 or mlx.mlx_array_dtype(p.suh) != .float16 or mlx.mlx_array_dtype(p.svh) != .float16) return error.BadExl3Dtype;
}

/// The bank's geometry from its shapes, every projection validated; gate and up must share a rate.
pub fn bankGeometry(bank: Bank, hidden: c_int) !Geometry {
    const gs = mlx.getShape(bank.gate.trellis);
    if (gs.len != 4 or gs[0] <= 0 or gs[2] <= 0 or gs[2] > std.math.maxInt(c_int) / 16) return error.BadExl3Shape;
    const width = gs[2] * 16;
    try validateClampedProjection(bank.gate, gs[0], hidden, width);
    try validateClampedProjection(bank.up, gs[0], hidden, width);
    try validateClampedProjection(bank.down, gs[0], width, hidden);
    const us = mlx.getShape(bank.up.trellis);
    if (us[3] != gs[3]) return error.Exl3GateUpRateMismatch;
    const ds = mlx.getShape(bank.down.trellis);
    return .{
        .experts = gs[0],
        .hidden = hidden,
        .width = width,
        .gate_up = format.kFromPackedDim(@intCast(gs[3])).?,
        .down = format.kFromPackedDim(@intCast(ds[3])).?,
    };
}

fn validateCall(x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array) !Geometry {
    const sh = mlx.getShape(x);
    const ix = mlx.getShape(inds);
    if (sh.len != 3 or ix.len != 3 or sh[0] <= 0 or sh[1] <= 0 or sh[0] != ix[0] or sh[1] != ix[1] or ix[2] <= 0 or ix[2] > kernels.REDUCE_MAX_TOPK) return error.BadExl3Shape;
    if (sh[2] <= 0 or @mod(sh[2], 128) != 0 or !std.mem.eql(c_int, ix, mlx.getShape(scores))) return error.BadExl3Shape;
    const xd = mlx.mlx_array_dtype(x);
    const id = mlx.mlx_array_dtype(inds);
    const sd = mlx.mlx_array_dtype(scores);
    if ((xd != .bfloat16 and xd != .float32 and xd != .float16) or
        (id != .uint32 and id != .int32) or
        (sd != .float32 and sd != .bfloat16 and sd != .float16)) return error.BadExl3Dtype;
    _ = std.math.mul(c_int, std.math.mul(c_int, sh[0], sh[1]) catch return error.BadExl3Shape, ix[2]) catch return error.BadExl3Shape;
    return bankGeometry(bank, sh[2]);
}

/// The compiled kernels of every geometry this process has served (a handful per model).
/// a spinlock: the cache fills once per geometry and is read thereafter
var cache_lock: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
var cache: [16]?decode.Kernels3 = @splat(null);

fn kernelsFor(geo: decode.Geometry, out_dtype: mlx.mlx_dtype) !*decode.Kernels3 {
    while (cache_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer cache_lock.store(0, .release);
    var free_slot: ?usize = null;
    for (&cache, 0..) |*slot, i| {
        if (slot.*) |*k| {
            const g = k.geo;
            if (g.hidden == geo.hidden and g.inter == geo.inter and g.topk == geo.topk and g.rate.n == geo.rate.n and
                g.dec.codebook == geo.dec.codebook and g.dec.window == geo.dec.window and k.out_dtype == out_dtype) return k;
        } else if (free_slot == null) free_slot = i;
    }
    const i = free_slot orelse return error.Exl3EngineCacheFull;
    // form 3 at 1 output tile per threadgroup (2 f32 accumulators per lane, 4 grouped) and a K partition of 4 slices, expert-grouped
    // row-slots (2 per lead): the fastest at 4 and 8 rows on the M5 Max, a tie at 1 row
    var k3 = try decode.Kernels3.initWith(geo, 1, 4, 2);
    k3.out_dtype = out_dtype;
    k3.fused_fin_fold = config.get().fused_fin_fold;
    // the shared expert's gate|up K slices per GEMV threadgroup in `moeFusedShared` (of its 20; a value that does not divide
    // them runs 1). Launch shape only: the planes, so the bytes, stay.
    k3.sspt = config.get().shared_spt;
    cache[i] = k3;
    return &cache[i].?;
}

/// `Config.slot_stats`: a diagnostic that reads every decode call's routing back to the host (a sync per layer, so never on in
/// a timed run) and prints, every 480 calls, how the row-slots of a call repeat experts: the multiplicity histogram that
/// sizes the grouping (GROUP 2 covers pairs; 3 and 4 would want wider groups) and the decode work a perfect grouping would
/// save.
const SlotStats = struct {
    var calls: u64 = 0;
    var calls_rows: [17]u64 = @splat(0);
    var slots_total: [17]u64 = @splat(0);
    var distinct_total: [17]u64 = @splat(0);
    var mult: [17][6]u64 = @splat(@splat(0)); // [rows][multiplicity 1..4, 5+]

    fn enabled() bool {
        return config.get().slot_stats;
    }

    fn record(inds: mlx.mlx_array, rows: usize, K: usize) !void {
        const n = rows * K;
        var buf: [16 * 64]u32 = undefined;
        if (rows > 16 or n > buf.len) return;
        try mlx.check(mlx.mlx_array_eval(inds));
        if (mlx.mlx_array_dtype(inds) == .int32) {
            const p = mlx.mlx_array_data_int32(inds) orelse return error.NullData;
            for (0..n) |i| buf[i] = @intCast(p[i]);
        } else {
            const p = mlx.mlx_array_data_uint32(inds) orelse return error.NullData;
            for (0..n) |i| buf[i] = p[i];
        }
        var distinct: usize = 0;
        for (0..n) |i| {
            const e = buf[i];
            var seen = false;
            for (0..i) |j| if (buf[j] == e) {
                seen = true;
                break;
            };
            if (seen) continue;
            distinct += 1;
            var m: usize = 0;
            for (0..n) |j| if (buf[j] == e) {
                m += 1;
            };
            mult[rows][@min(m, 5)] += 1;
        }
        slots_total[rows] += n;
        distinct_total[rows] += distinct;
        calls_rows[rows] += 1;
        calls += 1;
        if (calls % 480 == 0) print();
    }

    fn print() void {
        for (1..17) |r| if (calls_rows[r] > 0) {
            const saved = 100.0 * (1.0 - @as(f64, @floatFromInt(distinct_total[r])) / @as(f64, @floatFromInt(slots_total[r])));
            std.debug.print("[turboexl3 slot-stats] rows={d} calls={d} slots={d} distinct={d} perfect_grouping_saves={d:.1}% experts_by_multiplicity 1={d} 2={d} 3={d} 4={d} 5+={d}\n", .{
                r, calls_rows[r], slots_total[r], distinct_total[r], saved, mult[r][1], mult[r][2], mult[r][3], mult[r][4], mult[r][5],
            });
        };
    }
};

/// Form 5's kernels per geometry, as `cache` holds form 3's.
var cache5: [16]?decode.Kernels5 = @splat(null);

fn kernels5For(geo: decode.Geometry, out_dtype: mlx.mlx_dtype) !*decode.Kernels5 {
    while (cache_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer cache_lock.store(0, .release);
    var free_slot: ?usize = null;
    for (&cache5, 0..) |*slot, i| {
        if (slot.*) |*k| {
            const g = k.geo;
            if (g.hidden == geo.hidden and g.inter == geo.inter and g.topk == geo.topk and g.rate.n == geo.rate.n and
                g.dec.codebook == geo.dec.codebook and g.dec.window == geo.dec.window and k.out_dtype == out_dtype) return k;
        } else if (free_slot == null) free_slot = i;
    }
    const i = free_slot orelse return error.Exl3EngineCacheFull;
    // form 3's K partition (4 slices), so form 3's bytes; 4 output tiles per gate|up threadgroup, up to 4 row-slots per
    // expert group
    var k5 = try decode.Kernels5.init(geo, 4, 4, 4);
    k5.out_dtype = out_dtype;
    cache5[i] = k5;
    if (!decode_form_logged) {
        decode_form_logged = true;
        std.debug.print("[turboexl3] decode engaged: form {d} ({s}, expert groups of up to {d} row-slots, bt={d}, intr={})\n", .{ decodeForm(), if (decodeForm() == 6) "2 launches: the slot reduce by arrival" else "3 launches", k5.gmax, k5.bt, k5.intr });
    }
    return &cache5[i].?;
}
var decode_form_logged = false;

/// The decode form (3, 5, 6 = form 5 with the slot reduce in the down launch, 7 = form 3 with fin / dfin's H128s in registers,
/// 8 = form 7 with the prepare inside the gate|up launch, 9 = form 8 at one row and form 7 from two; 7, 8 and 9 keep form 3's
/// bytes): `decode_form_override` (tests), else `Config.decode_form`.
pub var decode_form_override: ?u8 = null;

fn decodeForm() u8 {
    return decode_form_override orelse config.get().decode_form;
}

/// The dense-linear kernels of every (in, out, rate, decode, dtype) this process has served: the trunk of a fully EXL3 model
/// has a dozen shapes (attention q/k/v/o, indexer, GDN qkv/z/out, shared gate/up/down, lm_head, MTP fc).
var dense_lock: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
var dense_cache: [32]?dense.Dense = @splat(null);

fn denseFor(shape: dense.Shape) !*dense.Dense {
    while (dense_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer dense_lock.store(0, .release);
    var free_slot: ?usize = null;
    for (&dense_cache, 0..) |*slot, i| {
        if (slot.*) |*d| {
            const g = d.shape;
            if (g.in == shape.in and g.out == shape.out and g.rate.n == shape.rate.n and g.dec.codebook == shape.dec.codebook and
                g.dec.window == shape.dec.window and g.out_dtype == shape.out_dtype) return d;
        } else if (free_slot == null) free_slot = i;
    }
    const i = free_slot orelse return error.Exl3EngineCacheFull;
    dense_cache[i] = try dense.Dense.init(shape);
    return &dense_cache[i].?;
}

/// A dense EXL3 linear: y [..., out] in x's dtype for x [..., in], trellis U16/I16 [in/16, out/16, n], suh F16 [in], svh F16
/// [out]. Rows up to `kernels.DECODE_ROWS_MAX` take the GEMV kernels (DEC-ONE's one launch up to 4 rows), rows 17 to 32
/// the register-path GEMM (`Config.nax`), wider calls MLX's matmul against the dequantized trellis: decoded per call, or
/// the W_hat a host's resident table keeps, or the un-rotated W_std it installed (one matmul of x, past 32 rows). The
/// fused GEMM (`Dense.fused_prefill`) is the tests' arm. dense.zig's module comment has each path.
pub fn linear(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, dec: format.Decode) !mlx.mlx_array {
    return linearOn(s, s, x, trellis, suh, svh, dec);
}

/// `linear` split across two MLX streams (the host's side stream `gdnz`): x's reshape and the prepare on
/// `s`, the prefill rows' inner product (dequantize + matmul, or the fused GEMM) and finish on `rs`, y's reshape on `s` again.
/// MLX joins the two with a fence where the prepared rows cross to `rs` and where y2 crosses back, so `rs` starts once the
/// prepare (and everything `s` recorded before it) is done. The same launches, operands and graph as `linear`: the same
/// bytes. Rows the GEMV or the register path take run on `s` alone (no crossing).
pub fn linearSplit(s: mlx.mlx_stream, rs: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, dec: format.Decode) !mlx.mlx_array {
    return linearOn(s, rs, x, trellis, suh, svh, dec);
}

fn linearOn(s: mlx.mlx_stream, rs: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, dec: format.Decode) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1) return error.BadExl3Shape;
    const tsh = mlx.getShape(trellis);
    if (tsh.len != 3) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(tsh[0] * 16);
    const out_dim: u32 = @intCast(tsh[1] * 16);
    const rate = format.Rate.fromN(@intCast(tsh[2])) orelse return error.BadExl3Shape;
    if (xsh[xsh.len - 1] != @as(c_int, @intCast(in_dim))) return error.BadExl3Shape;
    if (mlx.getShape(suh)[0] != @as(c_int, @intCast(in_dim)) or mlx.getShape(svh)[0] != @as(c_int, @intCast(out_dim))) return error.BadExl3Shape;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    var rows: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows *= d;
    if (rows < 1) return error.BadExl3Shape;
    const d = try denseFor(.{ .in = in_dim, .out = out_dim, .rate = rate, .dec = dec, .out_dtype = xd });
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, @intCast(in_dim) }, 2, s));
    const y2 = try d.linearOn(s, rs, x2, trellis, suh, svh, @intCast(rows));
    defer freeArr(y2);
    var osh: [8]c_int = undefined;
    if (xsh.len > osh.len) return error.BadExl3Shape;
    for (xsh, 0..) |v, i| osh[i] = v;
    osh[xsh.len - 1] = @intCast(out_dim);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, osh[0..xsh.len].ptr, @intCast(xsh.len), s));
    return y;
}

// ── The un-rotated trunk: the resident trunk out of the Hadamard basis (dense.zig's module comment) ──

/// What `trunkUnrot` installed: the groups (and of them the joins), their members and W_std bytes (the W_hat bytes they took
/// the place of), and the groups that kept their W_hat path.
pub const UnrotStats = struct { groups: u32 = 0, joined: u32 = 0, modules: u32 = 0, bytes: u64 = 0, failed: u32 = 0 };

/// At load, on an armed resident table (`dense.residentArm`): each group of `members` (`sizes[g]` consecutive
/// projections, 1..`dense.max_join`, reading one x, in the joined W_std's column order) un-rotated into its trellises' slots
/// in the W_hat's place, in `act` (the activation dtype; bf16 or f16), on `s`. A group of one is its member's W_std; a larger
/// group is one joined W_std [in, sum of outs] whose members are column views of it, so nothing is held twice. One member
/// at a time: its W_hat (`Dense.decodeTrellis`) and pass 1's f32 copy are evaluated and released before pass 2 makes its
/// W_std, so the transient is six bytes a weight of one member (189 MB at q_proj's 2560 x 12288), plus a join's copy of
/// its members (the joined bytes again) while it is made. A group that cannot be installed (a member not admitted or not
/// a trunk shape, a non-finite W_std, an MLX failure) keeps its W_hat path: `took[g]` (when given) says which took.
pub fn trunkUnrot(s: mlx.mlx_stream, members: []const Proj, sizes: []const u8, dec: format.Decode, act: mlx.mlx_dtype, took: ?[]bool) !UnrotStats {
    if (act != .bfloat16 and act != .float16) return error.BadExl3Dtype;
    if (took) |t| if (t.len != sizes.len) return error.BadExl3Shape;
    var st: UnrotStats = .{};
    var at: usize = 0;
    for (sizes, 0..) |n, g| {
        if (n == 0 or n > dense.max_join or at + n > members.len) return error.BadExl3Shape;
        const grp = members[at .. at + n];
        at += n;
        if (took) |t| t[g] = false;
        // the previous group's freed transients back to the system, not MLX's buffer cache: the load's peak is one group's
        _ = mlx.mlx_clear_cache();
        unrotGroup(s, grp, dec, act) catch |e| {
            st.failed += 1;
            if (!unrot_failed_said) {
                unrot_failed_said = true;
                std.debug.print("[turboexl3] un-rotated trunk: a group of {d} kept its W_hat path ({s})\n", .{ n, @errorName(e) });
            }
            continue;
        };
        if (took) |t| t[g] = true;
        st.groups += 1;
        if (n > 1) st.joined += 1;
        st.modules += n;
        for (grp) |m| st.bytes += dense.residentBytesOf(mlx.getShape(m.trellis));
    }
    if (at != members.len) return error.BadExl3Shape;
    return st;
}
var unrot_failed_said = false;

fn unrotGroup(s: mlx.mlx_stream, grp: []const Proj, dec: format.Decode, act: mlx.mlx_dtype) !void {
    var ws: [dense.max_join]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (ws) |w| if (w.ctx != null) freeArr(w);
    var outs: [dense.max_join]c_int = undefined;
    var in_dim: c_int = 0;
    var total: c_int = 0;
    for (grp, 0..) |m, i| {
        const tsh = mlx.getShape(m.trellis);
        if (tsh.len != 3 or tsh[0] <= 0 or tsh[1] <= 0) return error.BadExl3Shape;
        if (!dense.residentUnrotAdmits(m.trellis)) return error.Exl3ResidentNotAdmitted;
        if (i == 0) in_dim = tsh[0] * 16 else if (tsh[0] * 16 != in_dim) return error.BadExl3Shape;
        outs[i] = tsh[1] * 16;
        total += outs[i];
        const rate = format.Rate.fromN(@intCast(tsh[2])) orelse return error.BadExl3Shape;
        const d = try denseFor(.{ .in = @intCast(in_dim), .out = @intCast(outs[i]), .rate = rate, .dec = dec, .out_dtype = act });
        // pass 1 evaluated alone, so the W_hat's buffer is released before pass 2 allocates the W_std's
        const r = blk: {
            const w_hat = try d.decodeTrellis(s, m.trellis);
            defer freeArr(w_hat);
            break :blk try dense.unrotCols(s, w_hat);
        };
        defer freeArr(r);
        try mlx.check(mlx.mlx_array_eval(r));
        ws[i] = try dense.unrotRows(s, r, m.suh, m.svh, act);
        try mlx.check(mlx.mlx_array_eval(ws[i]));
    }
    const n = grp.len;
    // the group's W_std: the member's own, or the members' concatenated along the output (their views made below)
    var jw = mlx.mlx_array_new();
    var jw_owned = true;
    defer if (jw_owned) freeArr(jw);
    if (n == 1) {
        try mlx.check(mlx.mlx_array_set(&jw, ws[0]));
    } else {
        const vec = mlx.mlx_vector_array_new_data(&ws, n);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&jw, vec, 1, s));
        try mlx.check(mlx.mlx_array_eval(jw));
        // the members' own copies go now: the joined W_std is the group's one buffer
        for (ws[0..n]) |*w| {
            freeArr(w.*);
            w.* = .{ .ctx = null };
        }
    }
    // a non-finite weight (an f16 or bf16 overflow) keeps the group rotated
    {
        var fin = mlx.mlx_array_new();
        defer freeArr(fin);
        try mlx.check(mlx.mlx_isfinite(&fin, jw, s));
        var all = mlx.mlx_array_new();
        defer freeArr(all);
        try mlx.check(mlx.mlx_all(&all, fin, false, s));
        try mlx.check(mlx.mlx_array_eval(all));
        var ok = false;
        try mlx.check(mlx.mlx_array_item_bool(&ok, all));
        if (!ok) return error.Exl3UnrotNotFinite;
    }
    // each member's W_std: the group's whole (a group of one) or its column view, evaluated (a lazy view would stay bound to
    // this stream; an evaluated slice shares the joined buffer at its row stride)
    var views: [dense.max_join]mlx.mlx_array = @splat(.{ .ctx = null });
    var views_owned = true;
    defer if (views_owned) for (views[0..n]) |v| if (v.ctx != null) freeArr(v);
    var cols: [dense.max_join]u32 = undefined;
    var trellises: [dense.max_join]mlx.mlx_array = undefined;
    var col: c_int = 0;
    for (grp, 0..) |m, i| {
        trellises[i] = m.trellis;
        cols[i] = @intCast(col);
        views[i] = mlx.mlx_array_new();
        if (n == 1) {
            try mlx.check(mlx.mlx_array_set(&views[i], jw));
        } else {
            try mlx.check(mlx.mlx_slice(&views[i], jw, &[_]c_int{ 0, col }, 2, &[_]c_int{ in_dim, col + outs[i] }, 2, &[_]c_int{ 1, 1 }, 2, s));
            try mlx.check(mlx.mlx_array_eval(views[i]));
        }
        col += outs[i];
    }
    std.debug.assert(col == total);
    try dense.residentInstall(trellises[0..n], views[0..n], if (n > 1) jw else null, cols[0..n]);
    // the table owns the views and (a join's) jw now; a group of one's jw handle is this function's to drop
    views_owned = false;
    if (n > 1) jw_owned = false;
}

// ── Decode-row merges at a gated attention layer's EXL3 linears (`decb.zig`) ──
// Rows 1..kernels.DECODE_ROWS_MAX (the GEMV path), each the bytes of the chain it replaces, with DEC-ONE taking the call or
// not; error.Exl3EngineNotImplemented where a call is not served (the caller keeps its chain).

pub const decode_merge = @import("decb.zig");

/// The rows of x [..., in] and x as [rows, in] (the caller frees the view)
fn rowsOf(x: mlx.mlx_array, in_dim: u32, s: mlx.mlx_stream) !struct { rows: usize, x2: mlx.mlx_array } {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh[xsh.len - 1] != @as(c_int, @intCast(in_dim))) return error.BadExl3Shape;
    var rows: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows *= d;
    if (rows < 1) return error.BadExl3Shape;
    var x2 = mlx.mlx_array_new();
    errdefer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, @intCast(in_dim) }, 2, s));
    return .{ .rows = @intCast(rows), .x2 = x2 };
}

/// `decode_merge.gatedDecode`: y [1, rows, out] = o_proj(T(attn * sigmoid(gate))) at decode rows for the attention
/// core's attn [1, heads, rows, hd] (bf16 or f16), the gate read in place, `sigtab` attn's dtype [65536]: the bytes of the
/// host's strided sigmoid-gate kernel and `linear`, with the product formed inside o_proj's prepare (DEC-ONE's dec0 after it).
pub fn decodeMergeGated(s: mlx.mlx_stream, attn: mlx.mlx_array, gate: decode_merge.Gate, sigtab: mlx.mlx_array, o: Proj, dec: format.Decode) !mlx.mlx_array {
    const ash = mlx.getShape(attn);
    if (ash.len != 4 or ash[0] != 1 or ash[1] < 1 or ash[2] < 1 or ash[3] < 1) return error.BadExl3Shape;
    const heads: u32 = @intCast(ash[1]);
    const rows: usize = @intCast(ash[2]);
    const hd: u32 = @intCast(ash[3]);
    const ad = mlx.mlx_array_dtype(attn);
    if (ad != .bfloat16 and ad != .float16) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(gate.base) != ad or mlx.mlx_array_dtype(sigtab) != ad or mlx.mlx_array_size(sigtab) != 65536) return error.BadExl3Dtype;
    const d = try denseOf(o, heads * hd, ad, dec);
    const y2 = try decode_merge.gatedDecode(s, attn, gate, sigtab, .{ .d = d, .trellis = o.trellis, .suh = o.suh, .svh = o.svh }, heads, hd, rows);
    defer freeArr(y2);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &[_]c_int{ 1, @intCast(rows), @intCast(d.shape.out) }, 3, s));
    return y;
}

/// The qmerge form (`decode_merge.mergeGatedDecode`): y [1, rows, out] = o_proj(T(merge(pacc, pml) * sigmoid(gate)))
/// for the host's fused QSA core's split-K partials (pacc f32 [heads, rows, NS, 256], pml f32 [heads, rows, NS, 2]), the gate
/// in place and `sigtab` (the gate's dtype): the bytes of the host's merge launch, its strided gate kernel and `linear`.
pub fn decodeMergeQmergeGated(s: mlx.mlx_stream, pacc: mlx.mlx_array, pml: mlx.mlx_array, gate: decode_merge.Gate, sigtab: mlx.mlx_array, o: Proj, dec: format.Decode) !mlx.mlx_array {
    const psh = mlx.getShape(pacc);
    if (psh.len != 4 or psh[0] < 1 or psh[1] < 1 or psh[3] != 256) return error.BadExl3Shape;
    const heads: u32 = @intCast(psh[0]);
    const rows: usize = @intCast(psh[1]);
    const gd = mlx.mlx_array_dtype(gate.base);
    if (gd != .bfloat16 and gd != .float16) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(sigtab) != gd or mlx.mlx_array_size(sigtab) != 65536) return error.BadExl3Dtype;
    const d = try denseOf(o, heads * 256, gd, dec);
    const y2 = try decode_merge.mergeGatedDecode(s, pacc, pml, gate, sigtab, .{ .d = d, .trellis = o.trellis, .suh = o.suh, .svh = o.svh }, heads, rows);
    defer freeArr(y2);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &[_]c_int{ 1, @intCast(rows), @intCast(d.shape.out) }, 3, s));
    return y;
}

/// `decode_merge.qPlanes`: q_proj of x [..., in] at decode rows as two planes [1, rows, heads, hd] (each head's query
/// half, then its gate half): `linear`'s bytes, relaid out, so neither half is a strided view.
pub fn decodeMergeQPlanes(s: mlx.mlx_stream, x: mlx.mlx_array, q: Proj, heads: u32, hd: u32, dec: format.Decode) ![2]mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return error.BadExl3Dtype;
    const xsh = mlx.getShape(x);
    if (xsh.len < 1) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const d = try denseOf(q, in_dim, xd, dec);
    const v = try rowsOf(x, in_dim, s);
    defer freeArr(v.x2);
    const planes = try decode_merge.qPlanes(s, v.x2, v.rows, .{ .d = d, .trellis = q.trellis, .suh = q.suh, .svh = q.svh }, heads, hd);
    defer for (planes) |p| freeArr(p);
    var out: [2]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null } };
    errdefer for (out) |p| if (p.ctx != null) freeArr(p);
    for (planes, 0..) |p, i| {
        out[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out[i], p, &[_]c_int{ 1, @intCast(v.rows), @intCast(heads), @intCast(hd) }, 4, s));
    }
    return out;
}

/// `decode_merge.indexerQk`: the indexer projection of x [..., in] at decode rows with its finish's epilogue: qk
/// [1, rows, out] (`linear`'s bytes), the index queries normed and rotated [1, heads, rows, 128] and the one pooled key the
/// call completes [1, 1, 128] (each where asked; the caller frees every non-null result).
pub fn decodeMergeIndexer(s: mlx.mlx_stream, x: mlx.mlx_array, idx: Proj, ep: decode_merge.IdxEpilogue, dec: format.Decode) !decode_merge.IdxOut {
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16) return error.BadExl3Dtype;
    const xsh = mlx.getShape(x);
    if (xsh.len < 1) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const d = try denseOf(idx, in_dim, xd, dec);
    const v = try rowsOf(x, in_dim, s);
    defer freeArr(v.x2);
    var r = try decode_merge.indexerQk(s, v.x2, v.rows, .{ .d = d, .trellis = idx.trellis, .suh = idx.suh, .svh = idx.svh }, ep);
    errdefer r.deinit();
    var qk = mlx.mlx_array_new();
    mlx.check(mlx.mlx_reshape(&qk, r.qk, &[_]c_int{ 1, @intCast(v.rows), @intCast(d.shape.out) }, 3, s)) catch |e| {
        freeArr(qk);
        return e;
    };
    freeArr(r.qk);
    r.qk = qk;
    return r;
}

/// The fused attention projections (`attn_io.zig`).
pub const attn_io = @import("attn_io.zig");

/// The `Dense` `linear` would take for one projection of x (in, its dtype), the trellis and suh / svh checked as `linear`
/// checks them (plus their f16 dtype, which the fused texts name).
fn denseOf(p: Proj, in_dim: u32, xd: mlx.mlx_dtype, dec: format.Decode) !*dense.Dense {
    const tsh = mlx.getShape(p.trellis);
    if (tsh.len != 3) return error.BadExl3Shape;
    const out_dim: u32 = @intCast(tsh[1] * 16);
    const rate = format.Rate.fromN(@intCast(tsh[2])) orelse return error.BadExl3Shape;
    if (@as(u32, @intCast(tsh[0] * 16)) != in_dim) return error.BadExl3Shape;
    if (mlx.getShape(p.suh)[0] != @as(c_int, @intCast(in_dim)) or mlx.getShape(p.svh)[0] != @as(c_int, @intCast(out_dim))) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(p.suh) != .float16 or mlx.mlx_array_dtype(p.svh) != .float16) return error.BadExl3Dtype;
    return denseFor(.{ .in = in_dim, .out = out_dim, .rate = rate, .dec = dec, .out_dtype = xd });
}

/// ATTN-IN (`attn_io.attnIn`): an attention layer's q (each head [query | gate]), k and v projections of x [..., in]
/// (bf16 or f16) at prefill rows, with the q / k RMSNorm and partial RoPE in the finish: q, k, v [1, heads, rows, 256]
/// (normed and rotated q and k), the gate [1, rows, heads, 256], each the bytes of `linear` + MLX's split / rms_norm /
/// transpose / rope. error.Exl3EngineNotImplemented where it does not serve (the caller keeps its chain).
pub fn attnIn(s: mlx.mlx_stream, x: mlx.mlx_array, q: Proj, k: Proj, v: Proj, ep: attn_io.Epilogue, dec: format.Decode) !attn_io.InOut {
    const xsh = mlx.getShape(x);
    if (xsh.len < 2) return error.BadExl3Shape;
    var rows_c: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows_c *= d;
    if (rows_c < 1) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return error.BadExl3Dtype;
    const dq = try denseOf(q, in_dim, xd, dec);
    const dk = try denseOf(k, in_dim, xd, dec);
    const dv = try denseOf(v, in_dim, xd, dec);
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows_c, @intCast(in_dim) }, 2, s));
    return attn_io.attnIn(s, x2, @intCast(rows_c), .{ .d = dq, .trellis = q.trellis, .suh = q.suh, .svh = q.svh }, .{ .d = dk, .trellis = k.trellis, .suh = k.suh, .svh = k.svh }, .{ .d = dv, .trellis = v.trellis, .suh = v.suh, .svh = v.svh }, ep);
}

/// `attn_io.prepare4`: ATTN-IN's prepare launch with the QSA indexer projection `idx` as a fourth member, over
/// x [..., in] (bf16 or f16): xp[0..3] q's, k's and v's rotated copies (for `attnIn`'s `Epilogue.prep`), xp[3] the
/// indexer's (for `linearPrepared`), each f16 [rows, in], the caller's. error.Exl3EngineNotImplemented where it does not
/// serve (the caller keeps its launches).
pub fn attnPrep4(s: mlx.mlx_stream, x: mlx.mlx_array, q: Proj, k: Proj, v: Proj, idx: Proj, dec: format.Decode) ![4]mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh.len < 2) return error.BadExl3Shape;
    var rows_c: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows_c *= d;
    if (rows_c < 1) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return error.BadExl3Dtype;
    const dq = try denseOf(q, in_dim, xd, dec);
    const dk = try denseOf(k, in_dim, xd, dec);
    const dv = try denseOf(v, in_dim, xd, dec);
    const di = try denseOf(idx, in_dim, xd, dec);
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows_c, @intCast(in_dim) }, 2, s));
    return attn_io.prepare4(s, x2, @intCast(rows_c), .{ .d = dq, .trellis = q.trellis, .suh = q.suh, .svh = q.svh }, .{ .d = dk, .trellis = k.trellis, .suh = k.suh, .svh = k.svh }, .{ .d = dv, .trellis = v.trellis, .suh = v.suh, .svh = v.svh }, .{ .d = di, .trellis = idx.trellis, .suh = idx.suh, .svh = idx.svh });
}

/// `attn_io.linearPrepared` (named apart from the GDN tail's `linearPrepared`): `linear` of an x of dtype `xd` over its rows
/// already prepared (xp f16 [rows, in], `attnPrep4`'s fourth copy): y [rows, out] in `xd` (the caller reshapes), by
/// `linear`'s inner product and finish launches on that Dense (`denseFor` keys it as `linear` does), so `linear`'s bytes.
/// error.Exl3EngineNotImplemented where `linear`'s path at these rows reads no prepared rows (the GEMV rows, the fused GEMM).
pub fn linearPreparedIdx(s: mlx.mlx_stream, xp: mlx.mlx_array, p: Proj, xd: mlx.mlx_dtype, dec: format.Decode) !mlx.mlx_array {
    const xsh = mlx.getShape(xp);
    if (xsh.len != 2 or xsh[0] < 1 or mlx.mlx_array_dtype(xp) != .float16) return error.BadExl3Shape;
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    const d = try denseOf(p, @intCast(xsh[1]), xd, dec);
    return attn_io.linearPrepared(s, xp, .{ .d = d, .trellis = p.trellis, .suh = p.suh, .svh = p.svh }, @intCast(xsh[0]));
}

/// ATTN-OUT (`attn_io.linearGated`): y [1, rows, out] = o_proj(T(attn * sigmoid(gate))) for the attention core's
/// attn [1, heads, rows, hd] (bf16 or f16), the gate read in place (`attn_io.Gate`) and `sigtab` (MLX's sigmoid of every
/// 16-bit pattern, attn's dtype): the bytes of MLX's Sigmoid, Multiply and `linear`. error.Exl3EngineNotImplemented where
/// it does not serve.
pub fn linearGated(s: mlx.mlx_stream, attn: mlx.mlx_array, gate: attn_io.Gate, sigtab: mlx.mlx_array, o: Proj, dec: format.Decode) !mlx.mlx_array {
    const ash = mlx.getShape(attn);
    if (ash.len != 4 or ash[0] != 1 or ash[1] < 1 or ash[2] < 1 or ash[3] < 1) return error.BadExl3Shape;
    const heads: u32 = @intCast(ash[1]);
    const rows: usize = @intCast(ash[2]);
    const hd: u32 = @intCast(ash[3]);
    const ad = mlx.mlx_array_dtype(attn);
    if (ad != .bfloat16 and ad != .float16) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(gate.base) != ad or mlx.mlx_array_dtype(sigtab) != ad or mlx.mlx_array_size(sigtab) != 65536) return error.BadExl3Dtype;
    const d = try denseOf(o, heads * hd, ad, dec);
    const y2 = try attn_io.linearGated(s, attn, gate, sigtab, .{ .d = d, .trellis = o.trellis, .suh = o.suh, .svh = o.svh }, heads, hd, rows);
    defer freeArr(y2);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &[_]c_int{ 1, @intCast(rows), @intCast(d.shape.out) }, 3, s));
    return y;
}

/// ATTN-IN over the un-rotated trunk (`attn_io.attnInStd`): an attention layer's q (each head [query | gate]), k and v from their
/// standard-basis products in x's dtype, `srcs[m]` [..., stride_m] holding member m at columns [offs[m], ...) (a joined
/// product passed three times, or each member's own plain matmul), the outputs `attnIn`'s. error.Exl3EngineNotImplemented
/// where it does not serve (the caller keeps its chain).
pub fn attnInStd(s: mlx.mlx_stream, srcs: [3]mlx.mlx_array, offs: [3]u32, ep: attn_io.Epilogue) !attn_io.InOut {
    var flat: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (flat) |a| freeArr(a);
    var ins: [3]attn_io.StdIn = undefined;
    var rows: usize = 0;
    for (srcs, offs, &flat, &ins, 0..) |a, off, *f, *in, i| {
        const sh = mlx.getShape(a);
        if (sh.len < 2) return error.BadExl3Shape;
        var r: c_int = 1;
        for (sh[0 .. sh.len - 1]) |d| r *= d;
        if (r < 1) return error.BadExl3Shape;
        if (i == 0) rows = @intCast(r) else if (rows != @as(usize, @intCast(r))) return error.BadExl3Shape;
        try mlx.check(mlx.mlx_reshape(f, a, &[_]c_int{ r, sh[sh.len - 1] }, 2, s));
        in.* = .{ .a = f.*, .stride = @intCast(sh[sh.len - 1]), .off = off };
    }
    return attn_io.attnInStd(s, rows, ins[0], ins[1], ins[2], ep, mlx.mlx_array_dtype(srcs[0]));
}

/// ATTN-OUT over the un-rotated trunk (`attn_io.linearGatedStd`): y [1, rows, out] = T(attn * sigmoid(gate)) W_std for the attention
/// core's attn [1, heads, rows, hd], the gate read in place and `sigtab` (`linearGated`'s operands), o_proj's W_std from the
/// resident table. error.Exl3EngineNotImplemented where o_proj is not un-rotated or the call is not served.
pub fn linearGatedStd(s: mlx.mlx_stream, attn: mlx.mlx_array, gate: attn_io.Gate, sigtab: mlx.mlx_array, o: Proj) !mlx.mlx_array {
    const ash = mlx.getShape(attn);
    if (ash.len != 4 or ash[0] != 1 or ash[1] < 1 or ash[2] < 1 or ash[3] < 1) return error.BadExl3Shape;
    const w = dense.residentStd(o.trellis) orelse return error.Exl3EngineNotImplemented;
    defer freeArr(w);
    const rows: usize = @intCast(ash[2]);
    const y2 = try attn_io.linearGatedStd(s, attn, gate, sigtab, w, @intCast(ash[1]), @intCast(ash[3]), rows);
    defer freeArr(y2);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &[_]c_int{ 1, @intCast(rows), mlx.getShape(y2)[1] }, 3, s));
    return y;
}

// ── `linear` in two pieces around the fused GDN tail ──
// in_proj_z stops at its inner product (`linearInnerSplit`), `gdnOutPrep` runs z's finish, the gated norm and out_proj's
// prepare in one launch, and out_proj continues from those prepared rows (`linearPrepared`). Each piece is `linear`'s own
// launches with `linear`'s configs on `linear`'s `Dense`; `linearFinish` and `linearPrep` are the launches left out, for the
// byte tests. Rows the GEMV or the fused-GEMM option take are refused (error.Exl3EngineNotImplemented) before any launch.

/// `linearSplit` without its finish: the inner product F16 [rows, out] of x [..., in] (bf16, f16 or f32), the
/// prepare on `s` and the dequantize + matmul on `rs` (the register path's GEMM on `s`): the bytes `linear`'s finish reads.
pub fn linearInnerSplit(s: mlx.mlx_stream, rs: mlx.mlx_stream, x: mlx.mlx_array, p: Proj, dec: format.Decode) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    const d = try denseOf(p, in_dim, xd, dec);
    const v = try rowsOf(x, in_dim, s);
    defer freeArr(v.x2);
    return d.innerOn(s, rs, v.x2, p.trellis, p.suh, v.rows);
}

/// `linear` from prepared rows xp F16 [rows, in] (natural order, `linear`'s prepare arithmetic: the host's fused
/// norm-gate prepare): `linear`'s inner product and finish at this row count, y [rows, out] in `out_dtype`.
pub fn linearPrepared(s: mlx.mlx_stream, xp: mlx.mlx_array, p: Proj, dec: format.Decode, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const psh = mlx.getShape(xp);
    if (psh.len != 2 or psh[0] < 1 or mlx.mlx_array_dtype(xp) != .float16) return error.BadExl3Shape;
    if (out_dtype != .bfloat16 and out_dtype != .float32 and out_dtype != .float16) return error.BadExl3Dtype;
    const d = try denseOf(p, @intCast(psh[1]), out_dtype, dec);
    const rows: usize = @intCast(psh[0]);
    const path = d.pathFor(rows);
    if (path != .small and path != .matmul) return error.Exl3EngineNotImplemented;
    const z = try d.innerOf(s, xp, p.trellis, rows, path);
    defer freeArr(z);
    return d.finishOf(s, z, p.svh, rows);
}

/// `linear`'s finish of an inner product z F16 [rows, out] at prefill rows: y [rows, out] in `out_dtype`.
pub fn linearFinish(s: mlx.mlx_stream, z: mlx.mlx_array, p: Proj, dec: format.Decode, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const zsh = mlx.getShape(z);
    const tsh = mlx.getShape(p.trellis);
    if (zsh.len != 2 or zsh[0] < 1 or mlx.mlx_array_dtype(z) != .float16 or tsh.len != 3) return error.BadExl3Shape;
    if (out_dtype != .bfloat16 and out_dtype != .float32 and out_dtype != .float16) return error.BadExl3Dtype;
    const d = try denseOf(p, @intCast(tsh[0] * 16), out_dtype, dec);
    const rows: usize = @intCast(zsh[0]);
    if (zsh[1] != @as(c_int, @intCast(d.shape.out)) or rows <= kernels.DECODE_ROWS_MAX) return error.Exl3EngineNotImplemented;
    return d.finishOf(s, z, p.svh, rows);
}

/// `linear`'s prepare alone at prefill rows: xp F16 [rows, in] (natural order) of x [..., in].
pub fn linearPrep(s: mlx.mlx_stream, x: mlx.mlx_array, p: Proj, dec: format.Decode) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    const d = try denseOf(p, in_dim, xd, dec);
    const v = try rowsOf(x, in_dim, s);
    defer freeArr(v.x2);
    const path = d.pathFor(v.rows);
    if (path != .small and path != .matmul) return error.Exl3EngineNotImplemented;
    return d.prepare(s, v.x2, p.suh, v.rows);
}

/// The GDN layer's tail at prefill rows (`gdn_out.zig`). It returns out_proj's prepared rows xp F16 [rows, hv 128] (natural
/// order, as `linearPrepared` and `linearPiecePrepared` take them). The inputs are the recurrence's output y
/// [B, S, hv, 128] (bf16 or f16) and in_proj_z's inner product zq F16 [B S, hv 128] (`linearInnerSplit`'s), with
/// rows = B S. One launch runs three steps of the chain:
/// - z's finish (`linearFinish` with z's `svh`);
/// - the gated norm: MLX's rms_norm over each 128-wide head with `norm.w` and `norm.eps`, times MLX's sigmoid of z by
///   `norm.sigtab`, in y's dtype;
/// - out_proj's prepare (`linearPrep` with o's `suh`).
/// Every element is the chain's bytes. It refuses with error.Exl3EngineNotImplemented, before any launch, at decode rows
/// (<= `kernels.DECODE_ROWS_MAX`, where `linear` takes the GEMV), for another head width and off the GPU.
pub fn gdnOutPrep(s: mlx.mlx_stream, y: mlx.mlx_array, zq: mlx.mlx_array, z: Proj, norm: gdn_out.Norm, o: Proj) !mlx.mlx_array {
    const ysh = mlx.getShape(y);
    if (ysh.len != 4 or ysh[0] < 1 or ysh[1] < 1 or ysh[2] < 1) return error.BadExl3Shape;
    const dt = mlx.mlx_array_dtype(y);
    if (dt != .bfloat16 and dt != .float16) return error.BadExl3Dtype;
    const rows: usize = @intCast(ysh[0] * ysh[1]);
    if (ysh[3] != gdn_out.head_dim or rows <= kernels.DECODE_ROWS_MAX or !mlx.streamIsGpu(s)) return error.Exl3EngineNotImplemented;
    const hv: u32 = @intCast(ysh[2]);
    const vd: c_int = @intCast(hv * gdn_out.head_dim);
    const zsh = mlx.getShape(zq);
    if (zsh.len != 2 or zsh[0] != ysh[0] * ysh[1] or zsh[1] != vd or mlx.mlx_array_dtype(zq) != .float16) return error.BadExl3Shape;
    for ([_]mlx.mlx_array{ z.svh, o.suh }) |vec| {
        const sh = mlx.getShape(vec);
        if (sh.len != 1 or sh[0] != vd or mlx.mlx_array_dtype(vec) != .float16) return error.BadExl3Shape;
    }
    if (mlx.mlx_array_dtype(norm.w) != dt or mlx.mlx_array_size(norm.w) != gdn_out.head_dim) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(norm.sigtab) != dt or mlx.mlx_array_size(norm.sigtab) != 65536) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(norm.eps) != .float32 or mlx.mlx_array_ndim(norm.eps) != 0) return error.BadExl3Dtype;
    return gdn_out.prepare(s, y, zq, z.svh, norm, o.suh, hv, rows);
}

// ── `linear` of one call split into row pieces ──
// Three steps the host interleaves with its own launches and streams: `linearPiecesBegin` checks that the whole call and
// every piece take the matmul path (refusing with error.Exl3EngineNotImplemented before any launch) and dequantizes the
// trellis once; `linearPieceInner` runs one piece's prepare and matmul on the piece's stream (`linearPiecePrepared`: the
// matmul alone, on rows the host prepared as `linearPrepared` takes them: the fused GDN tail's); `linearPiecesFinish` runs
// one finish over every piece's inner product. Each launch is `linear`'s own on the matmul path (its configs, on `linear`'s
// `Dense`), so the result is `linear` (`linearPrepared`) of the concatenated rows bit for bit wherever MLX's matmul runs the
// whole call's kernel at every piece's row count (the caller's check).

/// The dequantized f16 W [in, out] of linear `p` for a call of x's dtype `xd` split into pieces of `rows` rows
/// (2..`dense.Dense.max_pieces`), made on `ws`, or error.Exl3EngineNotImplemented (nothing launched) where the call or a
/// piece leaves the matmul path.
pub fn linearPiecesBegin(ws: mlx.mlx_stream, p: Proj, rows: []const usize, xd: mlx.mlx_dtype, dec: format.Decode) !mlx.mlx_array {
    const tsh = mlx.getShape(p.trellis);
    if (tsh.len != 3) return error.BadExl3Shape;
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    const d = try denseOf(p, @intCast(tsh[0] * 16), xd, dec);
    if (!d.piecesServe(rows)) return error.Exl3EngineNotImplemented;
    return d.dequantize(ws, p.trellis);
}

/// One piece's inner product f16 [rows, out] of x [..., in] (the call's dtype) against `linearPiecesBegin`'s W,
/// the reshape, prepare and matmul on `s`.
pub fn linearPieceInner(s: mlx.mlx_stream, x: mlx.mlx_array, W: mlx.mlx_array, p: Proj, dec: format.Decode) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    const d = try denseOf(p, in_dim, xd, dec);
    const v = try rowsOf(x, in_dim, s);
    defer freeArr(v.x2);
    return d.pieceInner(s, v.x2, W, p.suh, v.rows);
}

/// One piece's inner product F16 [rows, out] from prepared rows xp F16 [rows, in] (natural order,
/// `linear`'s prepare arithmetic, as `linearPrepared` takes them) against `linearPiecesBegin`'s W, the matmul on `s`; `xd`
/// the call's dtype (the `Dense` the call's other steps take).
pub fn linearPiecePrepared(s: mlx.mlx_stream, xp: mlx.mlx_array, W: mlx.mlx_array, p: Proj, dec: format.Decode, xd: mlx.mlx_dtype) !mlx.mlx_array {
    const psh = mlx.getShape(xp);
    if (psh.len != 2 or psh[0] < 1 or mlx.mlx_array_dtype(xp) != .float16) return error.BadExl3Shape;
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    const d = try denseOf(p, @intCast(psh[1]), xd, dec);
    return d.piecePrepared(s, xp, W, @intCast(psh[0]));
}

/// `linear`'s finish over the pieces' inner products (in row order), one launch on `s`: y [their rows, out] in
/// `out_dtype` (the call's dtype).
pub fn linearPiecesFinish(s: mlx.mlx_stream, inners: []const mlx.mlx_array, p: Proj, dec: format.Decode, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const tsh = mlx.getShape(p.trellis);
    if (tsh.len != 3) return error.BadExl3Shape;
    if (out_dtype != .bfloat16 and out_dtype != .float32 and out_dtype != .float16) return error.BadExl3Dtype;
    const d = try denseOf(p, @intCast(tsh[0] * 16), out_dtype, dec);
    return d.finishPieces(s, inners, p.svh);
}

/// The prefill kernels of every geometry this process has served (one per expert bank shape).
var pf_lock: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
var pf_cache: [8]?prefill.Prefill = @splat(null);
var pf_logged: bool = false;

/// The served prefill variant: 64-row tiles (bm 64) with the register path for <= 32-row remainders in one launch (mixed),
/// the double-buffered weight stage, unrolled MMA sub-steps, the stage in NAX fragment order and the integer-rate reader
/// where the rate allows it (outputs identical across these options). The deferred MUL1 map (faster at small row counts
/// per expert) multiplies by the exact affine weight A t + C where the reference rounds it to f16 first, as the decode
/// chain does. The GEMMs' A operands in the fragments' slot order (`Config.pf_alay`) and each expert's full 128-row runs
/// in 128-row tiles beside the mixed launch (`Config.pf_t128`) give the same bytes either way.
fn pfOptions() prefill.Options {
    // bm 64: faster than 128 on captured routing at every call size, bytes identical
    // ncb 4: the down GEMM's threadgroups take 4 column blocks each (faster on small calls, neutral on large; bytes identical)
    // sintr: the register path's integer-rate reader (faster at every call size, bytes identical)
    // froute: argsort + table in one launch (one launch fewer, bytes identical)
    var o: prefill.Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .defer_affine = true, .ncb = 4, .sintr = true, .froute = true };
    o.alay = config.get().pf_alay;
    o.t128 = config.get().pf_t128;
    return o;
}

fn prefillFor(geo: prefill.Geometry) !*prefill.Prefill {
    while (pf_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer pf_lock.store(0, .release);
    var free_slot: ?usize = null;
    for (&pf_cache, 0..) |*slot, i| {
        if (slot.*) |*p| {
            const g = p.geo;
            if (g.hidden == geo.hidden and g.inter == geo.inter and g.topk == geo.topk and g.experts == geo.experts and
                g.rate.n == geo.rate.n and g.dec.codebook == geo.dec.codebook and g.dec.window == geo.dec.window) return p;
        } else if (free_slot == null) free_slot = i;
    }
    const i = free_slot orelse return error.Exl3EngineCacheFull;
    var o = pfOptions();
    if (geo.rate.n % 16 != 0) o.intr = false; // the integer-rate reader needs whole bits per weight
    if (!o.intr or o.small == .off) o.sintr = false; // the register path's use of it: intr's, where a register path runs
    if (geo.dec.codebook != .mul1) o.defer_affine = false; // the deferred affine map is MUL1's
    if (o.small != .mixed) o.split = 1; // split remainders are the mixed table's
    if (o.small == .all or o.stageBytes(128) > 32768) o.bk128 = false; // the shared-decode body's, within 32 KB
    if (o.small == .all or (geo.hidden / prefill.BN) % o.ncb != 0) o.ncb = 1; // whole column groups only
    // the A layout and the first-fragment-row MMA: the fragment-order body (or the all-register path); the layout's writers
    // are the strided prep and mid
    if (!o.fragb and o.small != .all) {
        o.alay = 0;
        o.half_rows = false;
    }
    if (o.aux != 0) o.alay = 0;
    // the 128-row tiles: their remainder tiles ride the mixed launch (froute's route at split 1, the strided mid and finish),
    // the 128-row tiles the bm-128 body beside bm 64's
    if (o.small != .mixed or o.split != 1 or !o.froute or o.aux != 0 or o.bm != 64) o.t128 = false;
    pf_cache[i] = try prefill.Prefill.init(geo, o);
    if (!pf_logged) {
        pf_logged = true;
        std.debug.print("[turboexl3] prefill engaged: fused EXL3 GEMM chain (bm={d} dbuf={} unroll={} fragb={} intr={} sintr={} defer={} split={d} small={s} down_bk={d} down_ncb={d} froute={} aux={d} alay={d} t128={})\n", .{ o.bm, o.dbuf, o.unroll, o.fragb, o.intr, o.sintr, o.defer_affine, o.split, @tagName(o.small), o.downStage(), o.ncb, o.froute, o.aux, o.alay, o.t128 });
    }
    return &pf_cache[i].?;
}

fn freeArr(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

/// Routed SwiGLU experts over `x` [B, S, D] for the top-k `inds`/`scores` [B, S, K], in x's shape and dtype.
/// Rows up to `kernels.DECODE_ROWS_MAX` take the decode form `Config.decode_form` names (form 9 by default: four or five
/// launches); wider calls take the fused prefill chain (`moePrefill`), or refuse `Exl3EngineNotImplemented` when
/// `Config.prefill` or `Config.nax` is off (the host keeps its prefill arm for them). Gate, up and down must share a rate.
pub fn moe(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool) !mlx.mlx_array {
    _ = verify_rows;
    const geo = try validateCall(x, bank, inds, scores);
    if (geo.gate_up.n != geo.down.n) return error.Exl3EngineNotImplemented;
    const xsh = mlx.getShape(x);
    const rows_c: c_int = xsh[0] * xsh[1];
    const rows: usize = @intCast(rows_c);
    const K: c_int = mlx.getShape(inds)[2];
    if (rows > kernels.DECODE_ROWS_MAX) {
        if (!config.get().prefill or !config.get().nax) return error.Exl3EngineNotImplemented;
        return moePrefill(s, x, bank, inds, scores, dec, geo, rows, K);
    }
    if (SlotStats.enabled()) SlotStats.record(inds, rows, @intCast(K)) catch {};
    const dgeo: decode.Geometry = .{ .hidden = @intCast(geo.hidden), .inter = @intCast(geo.width), .topk = @intCast(K), .rate = geo.gate_up, .dec = dec };
    // the operands the kernels take as the host holds them: x [rows, D] (bf16/f16/f32), slots [rows * K] (i32/u32), scores
    // [rows * K] (bf16/f16/f32); the kernels convert in place and write y in x's dtype, so no cast launches per layer. At one
    // row (`flatOperands`) the host's x [1, 1, D], inds and scores [1, 1, K] go as they are: every kernel indexes them flat
    // and MLX hands a launch row-contiguous inputs whatever their shape (copying one that is not, as the reshape did), so
    // three reshape ops fewer to evaluate per MoE layer of a serial step, the same bytes.
    const flat = flatOperands(rows);
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    var inds1 = mlx.mlx_array_new();
    defer freeArr(inds1);
    var s1 = mlx.mlx_array_new();
    defer freeArr(s1);
    if (!flat) {
        try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows_c, xsh[2] }, 2, s));
        try mlx.check(mlx.mlx_reshape(&inds1, inds, &[_]c_int{rows_c * K}, 1, s));
        try mlx.check(mlx.mlx_reshape(&s1, scores, &[_]c_int{rows_c * K}, 1, s));
    }
    const xk = if (flat) x else x2;
    const ik = if (flat) inds else inds1;
    const sk = if (flat) scores else s1;
    const kb: decode.Kernels.Bank = .{
        .gate_t = bank.gate.trellis,
        .gate_suh = bank.gate.suh,
        .gate_svh = bank.gate.svh,
        .up_t = bank.up.trellis,
        .up_suh = bank.up.suh,
        .up_svh = bank.up.svh,
        .down_t = bank.down.trellis,
        .down_suh = bank.down.suh,
        .down_svh = bank.down.svh,
    };
    const form = decodeForm();
    if (form >= 7 and !decode_form_logged) {
        decode_form_logged = true;
        std.debug.print("[turboexl3] decode engaged: form {d} (form 3's bytes in fewer dependent launches)\n", .{form});
    }
    const y0 = if (form == 5 or form == 6)
        try (try kernels5For(dgeo, mlx.mlx_array_dtype(x))).moe(s, xk, ik, sk, kb, rows, form == 6)
    else
        try (try kernelsFor(dgeo, mlx.mlx_array_dtype(x))).moeForm(s, xk, ik, sk, kb, rows, form);
    defer freeArr(y0);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y0, xsh.ptr, 3, s));
    return y;
}

/// Whether `moe` hands the decode kernels the host's operands without reshaping them: one row, where x [1, 1, D] and the
/// [1, 1, K] slots and scores are already the flat rows the kernels index.
pub fn flatOperands(rows: usize) bool {
    return rows == 1;
}

/// Prefill rows (> DECODE_ROWS_MAX) through the fused chain (`prefill.zig`): sorted by expert, the trellis decoded inside the
/// GEMMs' weight loader; output in x's shape and dtype.
/// `Config.pf_capture_dir`: each prefill call's routing (u32 [rows, K], the slots `moePrefill` reads) written to
/// <dir>/pf-<call>-r<rows>.u32 for the first 512 calls, so the prefill timing test can replay real routing
/// (TURBOEXL3_PF_INDS). The eval syncs every call: a capture run, never a timed one.
const PfCapture = struct {
    var calls: u32 = 0;
    extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
    extern "c" fn fwrite(ptr: *const anyopaque, size: usize, n: usize, f: *anyopaque) usize;
    extern "c" fn fclose(f: *anyopaque) c_int;

    fn record(inds1: mlx.mlx_array, rows: usize) !void {
        const dir = config.get().pf_capture_dir orelse return;
        if (calls >= 512) return;
        defer calls += 1;
        try mlx.check(mlx.mlx_array_eval(inds1));
        const n = mlx.mlx_array_size(inds1);
        var path_buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/pf-{d:0>4}-r{d}.u32", .{ dir, calls, rows }, 0);
        const f = fopen(path, "wb") orelse return error.CaptureOpen;
        defer _ = fclose(f);
        const ptr: *const anyopaque = switch (mlx.mlx_array_dtype(inds1)) {
            .int32 => @ptrCast(mlx.mlx_array_data_int32(inds1) orelse return error.NullData),
            .uint32 => @ptrCast(mlx.mlx_array_data_uint32(inds1) orelse return error.NullData),
            else => return error.CaptureDtype,
        };
        if (fwrite(ptr, 4, n, f) != n) return error.CaptureWrite;
    }
};

fn moePrefill(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, geo: Geometry, rows: usize, K: c_int) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const rows_c: c_int = @intCast(rows);
    const pf = try prefillFor(.{ .hidden = @intCast(geo.hidden), .inter = @intCast(geo.width), .topk = @intCast(K), .experts = @intCast(geo.experts), .rate = geo.gate_up, .dec = dec });
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows_c, xsh[2] }, 2, s));
    var inds1 = mlx.mlx_array_new();
    defer freeArr(inds1);
    try mlx.check(mlx.mlx_reshape(&inds1, inds, &[_]c_int{rows_c * K}, 1, s));
    PfCapture.record(inds1, rows) catch |e| std.debug.print("[turboexl3] prefill capture failed: {s}\n", .{@errorName(e)});
    var s1 = mlx.mlx_array_new();
    defer freeArr(s1);
    try mlx.check(mlx.mlx_reshape(&s1, scores, &[_]c_int{rows_c * K}, 1, s));
    const ops: prefill.Operands = .{
        .gate_t = bank.gate.trellis,
        .gate_suh = bank.gate.suh,
        .gate_svh = bank.gate.svh,
        .up_t = bank.up.trellis,
        .up_suh = bank.up.suh,
        .up_svh = bank.up.svh,
        .down_t = bank.down.trellis,
        .down_suh = bank.down.suh,
        .down_svh = bank.down.svh,
    };
    const y0 = if (pf.opts.t128) try pf.moeT128(s, x2, inds1, s1, ops, rows, mlx.mlx_array_dtype(x)) else try pf.moe(s, x2, inds1, s1, ops, rows, mlx.mlx_array_dtype(x));
    defer freeArr(y0);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y0, xsh.ptr, 3, s));
    return y;
}

/// `moe` plus an already gated shared expert, added at the routed output's rounding point.
pub fn moeWithShared(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool, shared: mlx.mlx_array) !mlx.mlx_array {
    if (mlx.mlx_array_dtype(shared) != mlx.mlx_array_dtype(x) or mlx.mlx_array_size(shared) != mlx.mlx_array_size(x)) return error.BadExl3Shape;
    const routed = try moe(s, x, bank, inds, scores, dec, verify_rows);
    defer freeArr(routed);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_add(&y, routed, shared, s));
    return y;
}

/// Decode / verify rows (up to `decode.Kernels3.shared_rows_max`) of a MoE with a gated shared expert, both in form 9's
/// launches (`decode.Kernels3.moeShared`): y [x's shape, x's dtype] = routed + sigmoid(logit) * shared(x), the bytes of the
/// host's chain over `moe` (form 3), three `linear`s (dense.zig), its table SwiGLU (`sigtab` OUT_T [65536]: MLX's sigmoid of
/// every 16-bit pattern) and MLX's Sigmoid, Multiply and Add. `shared`: the dense linears (trellis [in/16, out/16, n], suh,
/// svh), `logit` the shared gate's logit per row ([rows] or [.., rows, 1]). error.Exl3EngineNotImplemented for a call the
/// fused launches do not carry (the caller keeps its chain). The shared expert's heads run on the routed kernels
/// (`moeShared`); `decode.Kernels3`'s side launches and v2 heads give the same bytes.
pub fn moeFusedShared(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, shared: Bank, sigtab: mlx.mlx_array, logit: mlx.mlx_array) !mlx.mlx_array {
    const geo = try validateCall(x, bank, inds, scores);
    if (geo.gate_up.n != geo.down.n) return error.Exl3EngineNotImplemented;
    const xsh = mlx.getShape(x);
    const rows_c: c_int = xsh[0] * xsh[1];
    const rows: usize = @intCast(rows_c);
    if (rows > decode.Kernels3.shared_rows_max) return error.Exl3EngineNotImplemented;
    const K: c_int = mlx.getShape(inds)[2];
    const xd = mlx.mlx_array_dtype(x);
    if (mlx.mlx_array_dtype(sigtab) != xd or mlx.mlx_array_size(sigtab) != 65536 or mlx.mlx_array_dtype(logit) != xd or mlx.mlx_array_size(logit) != rows) return error.Exl3EngineNotImplemented;
    const gsh = mlx.getShape(shared.gate.trellis);
    const dsh = mlx.getShape(shared.down.trellis);
    if (gsh.len != 3 or dsh.len != 3) return error.Exl3EngineNotImplemented;
    const inter_s: u32 = @intCast(gsh[1] * 16);
    const sb: decode.Kernels3.SharedBank = .{
        .gate_t = shared.gate.trellis,
        .gate_suh = shared.gate.suh,
        .gate_svh = shared.gate.svh,
        .up_t = shared.up.trellis,
        .up_suh = shared.up.suh,
        .up_svh = shared.up.svh,
        .down_t = shared.down.trellis,
        .down_suh = shared.down.suh,
        .down_svh = shared.down.svh,
        .n_gu = @intCast(gsh[2]),
        .n_dn = @intCast(dsh[2]),
        .ks_gu = dense.Dense.ksFor(@intCast(geo.hidden), inter_s),
        .ks_dn = dense.Dense.ksFor(inter_s, @intCast(geo.hidden)),
        .defer_affine = dense.Dense.deferFor(dec),
        .sigtab = sigtab,
        .logit = logit,
    };
    const dgeo: decode.Geometry = .{ .hidden = @intCast(geo.hidden), .inter = @intCast(geo.width), .topk = @intCast(K), .rate = geo.gate_up, .dec = dec };
    const k3 = try kernelsFor(dgeo, xd);
    if (!k3.sharedAdmitted(rows, sb)) return error.Exl3EngineNotImplemented;
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows_c, xsh[2] }, 2, s));
    var inds1 = mlx.mlx_array_new();
    defer freeArr(inds1);
    try mlx.check(mlx.mlx_reshape(&inds1, inds, &[_]c_int{rows_c * K}, 1, s));
    var s1 = mlx.mlx_array_new();
    defer freeArr(s1);
    try mlx.check(mlx.mlx_reshape(&s1, scores, &[_]c_int{rows_c * K}, 1, s));
    var l1 = mlx.mlx_array_new();
    defer freeArr(l1);
    try mlx.check(mlx.mlx_reshape(&l1, logit, &[_]c_int{rows_c}, 1, s));
    var sbr = sb;
    sbr.logit = l1;
    const kb: decode.Kernels.Bank = .{
        .gate_t = bank.gate.trellis,
        .gate_suh = bank.gate.suh,
        .gate_svh = bank.gate.svh,
        .up_t = bank.up.trellis,
        .up_suh = bank.up.suh,
        .up_svh = bank.up.svh,
        .down_t = bank.down.trellis,
        .down_suh = bank.down.suh,
        .down_svh = bank.down.svh,
    };
    const y0 = try k3.moeShared(s, x2, inds1, s1, kb, rows, sbr);
    defer freeArr(y0);
    if (!fused_shared_logged) {
        fused_shared_logged = true;
        std.debug.print("[turboexl3] shared expert fused into the routed decode launches (gate|up K{d} split {d}, down K{d} split {d})\n", .{ sb.n_gu / 16, sb.ks_gu, sb.n_dn / 16, sb.ks_dn });
    }
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y0, xsh.ptr, 3, s));
    return y;
}
var fused_shared_logged = false;

/// `moeFusedShared` with the routing folded into form 9's routed prepare launch (`decode.Kernels3.moeSharedRouted`).
/// `logits` [.., rows, E] is the router
/// gemv's output in x's dtype and `k` the experts per token: the top-k and the renormalized softmax weights are the host's
/// fused router kernel's (softmax mode, `mlxserve_moe_router_softmax`, `decode.text_router_softmax`), computed inside
/// the launch. `logit` is the shared gate's logit per row as `moeFusedShared` takes it, or null with `gate_vec` the gate's
/// weight vector [hidden] (the logit computed in the same launch in MLX's gemv order, bf16 at 2..4 rows). The bytes of the
/// router kernel + `moeFusedShared`, one dispatch and one dependent level fewer. error.Exl3EngineNotImplemented, before any
/// launch, for a call the routed launch does not carry (the caller runs its chain): rows the gate|up launch prepares itself
/// (gu8, one row by default), the slot statistics, a geometry or dtype outside the kernel.
pub fn moeFusedSharedRouted(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, logits: mlx.mlx_array, k: u32, dec: format.Decode, shared: Bank, sigtab: mlx.mlx_array, logit: ?mlx.mlx_array, gate_vec: ?mlx.mlx_array) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const lsh = mlx.getShape(logits);
    if (xsh.len != 3 or lsh.len != 3 or xsh[0] <= 0 or xsh[1] <= 0 or xsh[0] != lsh[0] or xsh[1] != lsh[1]) return error.BadExl3Shape;
    if ((logit == null) == (gate_vec == null)) return error.BadExl3Shape;
    const rows_c: c_int = xsh[0] * xsh[1];
    const rows: usize = @intCast(rows_c);
    if (rows > decode.Kernels3.shared_rows_max or k == 0 or k > 31) return error.Exl3EngineNotImplemented;
    const xd = mlx.mlx_array_dtype(x);
    if (mlx.mlx_array_dtype(logits) != xd) return error.Exl3EngineNotImplemented;
    const geo = try bankGeometry(bank, xsh[2]);
    if (geo.gate_up.n != geo.down.n) return error.Exl3EngineNotImplemented;
    if (lsh[2] != geo.experts or lsh[2] > 2048 or lsh[2] < @as(c_int, @intCast(k))) return error.Exl3EngineNotImplemented;
    if (mlx.mlx_array_dtype(sigtab) != xd or mlx.mlx_array_size(sigtab) != 65536) return error.Exl3EngineNotImplemented;
    if (logit) |l| {
        if (mlx.mlx_array_dtype(l) != xd or mlx.mlx_array_size(l) != rows) return error.Exl3EngineNotImplemented;
    }
    if (gate_vec) |gv| {
        // MLX's matmul of x [R, D] and [D, 1] runs gemv_bfloat16_bm1_bn8_sm1_sn32_tm{1,4}_tn4 only at R = 2..4, D > 64, D >= 16 R
        const d = xsh[2];
        if (xd != .bfloat16 or mlx.mlx_array_dtype(gv) != .bfloat16 or mlx.mlx_array_size(gv) != @as(usize, @intCast(d)) or rows < 2 or rows > 4 or d <= 64 or d < 16 * rows_c) return error.Exl3EngineNotImplemented;
    }
    const gsh = mlx.getShape(shared.gate.trellis);
    const dsh = mlx.getShape(shared.down.trellis);
    if (gsh.len != 3 or dsh.len != 3) return error.Exl3EngineNotImplemented;
    const inter_s: u32 = @intCast(gsh[1] * 16);
    const sb: decode.Kernels3.SharedBank = .{
        .gate_t = shared.gate.trellis,
        .gate_suh = shared.gate.suh,
        .gate_svh = shared.gate.svh,
        .up_t = shared.up.trellis,
        .up_suh = shared.up.suh,
        .up_svh = shared.up.svh,
        .down_t = shared.down.trellis,
        .down_suh = shared.down.suh,
        .down_svh = shared.down.svh,
        .n_gu = @intCast(gsh[2]),
        .n_dn = @intCast(dsh[2]),
        .ks_gu = dense.Dense.ksFor(@intCast(geo.hidden), inter_s),
        .ks_dn = dense.Dense.ksFor(inter_s, @intCast(geo.hidden)),
        .defer_affine = dense.Dense.deferFor(dec),
        .sigtab = sigtab,
        // with a gate vector the merged launch's logit takes this one's place in dfin7s (never read; any array of the dtype)
        .logit = logit orelse sigtab,
    };
    const dgeo: decode.Geometry = .{ .hidden = @intCast(geo.hidden), .inter = @intCast(geo.width), .topk = k, .rate = geo.gate_up, .dec = dec };
    const k3 = try kernelsFor(dgeo, xd);
    if (!k3.sharedAdmitted(rows, sb)) return error.Exl3EngineNotImplemented;
    // the prepare in a launch of its own (form 9 above gu8_rows), no slot stats
    if (rows <= k3.gu8_rows and k3.gu8Admitted(rows, k3.kdiv9)) return error.Exl3EngineNotImplemented;
    if (SlotStats.enabled()) return error.Exl3EngineNotImplemented;
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows_c, xsh[2] }, 2, s));
    var l2 = mlx.mlx_array_new();
    defer freeArr(l2);
    try mlx.check(mlx.mlx_reshape(&l2, logits, &[_]c_int{ rows_c, lsh[2] }, 2, s));
    var l1 = mlx.mlx_array_new();
    defer freeArr(l1);
    var sbr = sb;
    if (logit) |l| {
        try mlx.check(mlx.mlx_reshape(&l1, l, &[_]c_int{rows_c}, 1, s));
        sbr.logit = l1;
    }
    const kb: decode.Kernels.Bank = .{
        .gate_t = bank.gate.trellis,
        .gate_suh = bank.gate.suh,
        .gate_svh = bank.gate.svh,
        .up_t = bank.up.trellis,
        .up_suh = bank.up.suh,
        .up_svh = bank.up.svh,
        .down_t = bank.down.trellis,
        .down_suh = bank.down.suh,
        .down_svh = bank.down.svh,
    };
    const y0 = try k3.moeSharedRouted(s, x2, l2, kb, rows, sbr, gate_vec);
    defer freeArr(y0);
    if (!routed_logged[@intFromBool(gate_vec != null)]) {
        routed_logged[@intFromBool(gate_vec != null)] = true;
        std.debug.print("[turboexl3] router folded into form 9's prepare launch ({d} rows, {d} experts, top-{d}){s}\n", .{ rows, lsh[2], k, if (gate_vec != null) ", the shared gate's logit with it" else "" });
    }
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y0, xsh.ptr, 3, s));
    return y;
}
var routed_logged = [2]bool{ false, false };

/// GLM's gate is upper-clamped and its up branch symmetrically clamped before SwiGLU and the down projection.
pub fn moeClamped(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, limit: c_int) !mlx.mlx_array {
    _ = s;
    _ = dec;
    if (limit <= 0 or limit > 128) return error.InvalidExl3Clamp;
    _ = try validateCall(x, bank, inds, scores);
    return error.Exl3EngineNotImplemented;
}

test {
    _ = config;
    _ = format;
    _ = spec;
    _ = lane;
    _ = decode;
    _ = ref;
    _ = unfused;
    _ = prefill;
}

test "turboexl3 root: the served prefill options follow the config, and the decode form is the config's" {
    const t = std.testing;
    const saved = config.get().*;
    defer config.configure(saved) catch unreachable;
    // the served set takes the 128-row tiles in fragment order; the vector-layout prep / mid / finish and bm 128 do not
    const o = pfOptions();
    try t.expect(o.valid() and o.t128 and o.alay == 2 and o.bm == 64);
    var v = o;
    v.bm = 128;
    try t.expect(!v.valid());
    v = o;
    v.alay = 0;
    v.aux = 8;
    try t.expect(!v.valid());
    try config.configure(.{ .pf_alay = 0, .pf_t128 = false, .decode_form = 5 });
    const p = pfOptions();
    try t.expect(p.alay == 0 and !p.t128);
    try t.expectEqual(@as(u8, 5), decodeForm());
    decode_form_override = 3;
    defer decode_form_override = null;
    try t.expectEqual(@as(u8, 3), decodeForm());
}
