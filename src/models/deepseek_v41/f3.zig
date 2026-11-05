//! F3: DeepSeek-V4.1's shared expert at decode rows (`sharedSwiglu`), the gate's and up's finishes, the host's clamped
//! SwiGLU and the down's prepare in one launch (`text_swiglu_prep`) between the dense chain's GEMVs, on turboexl3's
//! dense linear (`core/dense.zig`). Its state (the kernel, its configs) sits beside the gate's `Dense` (`forDense`).
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const spec = @import("../../core/spec.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const root = @import("../../root.zig");
const Dense = dense.Dense;
const Proj = root.Proj;

fn freeArr(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

/// F3 (`F3.swigluPrep`): the shared expert's gate and up finishes (text_dfinish's statements) of one 128-wide block of
/// their outputs, the host's clamped SwiGLU in f32 (MLX's functors), the activation dtype, then the down linear's prepare
/// (text_prep_s, lane order) on it: xp half [ROWS, DIM] and xs f32 [ROWS, DIM / 128], the gate's and up's outputs never
/// stored
pub const text_swiglu_prep: [:0]const u8 = @embedFile("kernels/f3/swiglu_prep.metal");
const swiglu_prep_in = [_][*:0]const u8{ "z1", "xs1", "svh1", "z3", "xs3", "svh3", "suh2", "lim" };
const swiglu_prep_out = [_][*:0]const u8{ "xp", "xs" };

/// F3's state beside one gate|up `Dense`
pub const F3 = struct {
    /// F3's launch on its Dense's (a gate's) shape (`text_swiglu_prep`), made on first use, its configs per decode row count (for
    /// `swiglu_act`, the activation dtype they were built for) and the limit's one-element array
    swiglu_prep: ?mlx.mlx_fast_metal_kernel = null,
    swiglu_cfg: [decode.Kernels.max_rows + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null),
    swiglu_act: ?mlx.mlx_dtype = null,
    swiglu_lim: ?struct { v: f32, a: mlx.mlx_array } = null,

    /// F3 on the gate's shape (the up's the same): from the gate's and up's planes and block sums (`prepareGemv`), one
    /// launch of their finishes, the clamped SwiGLU at `limit` (> 0) in f32, `act`'s rounding and the down linear's
    /// prepare (its suh `suh2` [out]): the down's xp half [rows, out] and xs f32 [rows, out / 128], what its prep_s makes
    /// of the host's activation, bit for bit. The caller frees both.
    pub fn swigluPrep(self: *F3, d: *Dense, s: mlx.mlx_stream, z1: mlx.mlx_array, xs1: mlx.mlx_array, svh1: mlx.mlx_array, z3: mlx.mlx_array, xs3: mlx.mlx_array, svh3: mlx.mlx_array, suh2: mlx.mlx_array, limit: f32, act: mlx.mlx_dtype, rows: usize) ![2]mlx.mlx_array {
        if (!d.preparedTakes(rows) or !(limit > 0)) return error.Exl3EngineNotImplemented;
        if (self.swiglu_prep == null) self.swiglu_prep = ks.newKernel("turboexl3_dense_swiglu_prep", &swiglu_prep_in, &swiglu_prep_out, text_swiglu_prep, decode.header, true) orelse return error.KernelCreateFailed;
        if (self.swiglu_act) |a| if (a != act) {
            for (&self.swiglu_cfg) |*c| if (c.*) |cc| {
                _ = mlx.mlx_fast_metal_kernel_config_free(cc);
                c.* = null;
            };
        };
        self.swiglu_act = act;
        if (self.swiglu_lim) |l| if (l.v != limit) {
            _ = mlx.mlx_array_free(l.a);
            self.swiglu_lim = null;
        };
        if (self.swiglu_lim == null) {
            const a = mlx.mlx_array_new_data(&limit, &[_]c_int{1}, 1, .float32);
            if (a.ctx == null) return error.MlxError;
            self.swiglu_lim = .{ .v = limit, .a = a };
        }
        const cfg = self.swiglu_cfg[rows] orelse blk: {
            const r: c_int = @intCast(rows);
            const dim: u32 = d.shape.out;
            const c = try ks.buildConfig(.{
                .grid = .{ (dim / 128) * 32, @intCast(rows), 1 },
                .threadgroup = .{ 32, 1, 1 },
                .out_shapes = &.{ &[_]c_int{ r, @intCast(dim) }, &[_]c_int{ r, @intCast(dim / 128) } },
                .out_dtypes = &.{ .float16, .float32 },
                .template = &.{
                    .{ .name = "DIM", .value = .{ .int = @intCast(dim) } },
                    .{ .name = "KS1", .value = .{ .int = @intCast(d.ks_split) } },
                    .{ .name = "KS3", .value = .{ .int = @intCast(d.ks_split) } },
                    .{ .name = "ROWS", .value = .{ .int = r } },
                    .{ .name = "INBLK", .value = .{ .int = @intCast(d.shape.in / 128) } },
                    .{ .name = "DEFER", .value = .{ .int = if (d.defer_affine) 1 else 0 } },
                    .{ .name = "OUT_T", .value = .{ .dtype = d.shape.out_dtype } },
                    .{ .name = "ACT_T", .value = .{ .dtype = act } },
                },
            });
            self.swiglu_cfg[rows] = c;
            break :blk c;
        };
        var o: [2]mlx.mlx_array = undefined;
        try ks.applyConfig(self.swiglu_prep.?, cfg, &.{ z1, xs1, svh1, z3, xs3, svh3, suh2, self.swiglu_lim.?.a }, &o, s);
        return o;
    }

    pub fn deinit(self: *F3) void {
        for (&self.swiglu_cfg) |*c| if (c.*) |cc| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cc);
            c.* = null;
        };
        if (self.swiglu_prep) |k| decode.freeKernel(k);
        if (self.swiglu_lim) |l| _ = mlx.mlx_array_free(l.a);
        self.* = undefined;
    }
};

const Entry = struct { key: *const Dense, f: F3 };
const cache_lock = Global(std.atomic.Value(u32), "deepseek_v41_f3_lock", std.atomic.Value(u32).init(0));
const cache = Global([16]?Entry, "deepseek_v41_f3_cache", @splat(null));

/// F3's state beside `d` (root's `Dense` of the gate's shape), made at its first call
pub fn forDense(d: *const Dense) !*F3 {
    while (cache_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer cache_lock.get().*.store(0, .release);
    var free_slot: ?usize = null;
    for (&cache.get().*, 0..) |*slot, i| {
        if (slot.*) |*e| {
            if (e.key == d) return &e.f;
        } else if (free_slot == null) free_slot = i;
    }
    const i = free_slot orelse return error.Exl3EngineCacheFull;
    cache.get().*[i] = .{ .key = d, .f = .{} };
    return &cache.get().*[i].?.f;
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ cache_lock, cache };

/// F3 at x [..., in] through the gate `p1`, the up `p3` (in -> inter each) and the down `p2` (inter -> out): the rows and
/// the two Dense when `sharedSwiglu` serves the call (decode rows the dense chain takes: 5 to DECODE_ROWS_MAX, DEC-ONE
/// keeping up to 4; one rate and decode; the down's input in `act`); else null.
fn swigluPlan(x: mlx.mlx_array, p1: Proj, p3: Proj, p2: Proj, act: mlx.mlx_dtype, dec: format.Decode) !?struct { rows: usize, gu: *dense.Dense, down: *dense.Dense } {
    const xsh = mlx.getShape(x);
    const t1 = mlx.getShape(p1.trellis);
    const t3 = mlx.getShape(p3.trellis);
    const t2 = mlx.getShape(p2.trellis);
    if (xsh.len < 1 or t1.len != 3 or t3.len != 3 or t2.len != 3) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return null;
    if (act != .bfloat16 and act != .float32 and act != .float16) return null;
    if (!std.mem.eql(c_int, t1, t3) or t2[0] != t1[1] or t2[2] != t1[2]) return null;
    const in_dim = t1[0] * 16;
    const inter = t1[1] * 16;
    const out_dim = t2[1] * 16;
    if (xsh[xsh.len - 1] != in_dim) return null;
    const rate = format.Rate.fromN(@intCast(t1[2])) orelse return null;
    for ([_]Proj{ p1, p3 }) |p| if (mlx.getShape(p.suh).len != 1 or mlx.getShape(p.suh)[0] != in_dim or mlx.getShape(p.svh).len != 1 or mlx.getShape(p.svh)[0] != inter) return null;
    if (mlx.getShape(p2.suh).len != 1 or mlx.getShape(p2.suh)[0] != inter or mlx.getShape(p2.svh).len != 1 or mlx.getShape(p2.svh)[0] != out_dim) return null;
    var rows: usize = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows *= @intCast(d);
    if (rows < 1 or rows > spec.DECODE_ROWS_MAX) return null;
    const gu = try root.denseCached(.{ .in = @intCast(in_dim), .out = @intCast(inter), .rate = rate, .dec = dec, .out_dtype = xd });
    const down = try root.denseCached(.{ .in = @intCast(inter), .out = @intCast(out_dim), .rate = rate, .dec = dec, .out_dtype = act });
    if (!gu.preparedTakes(rows) or !down.preparedTakes(rows)) return null;
    return .{ .rows = rows, .gu = gu, .down = down };
}

/// F3: whether `sharedSwiglu` serves x through `p1`, `p3` and `p2` (`swigluPlan`)
pub fn sharedSwigluTakes(x: mlx.mlx_array, p1: Proj, p3: Proj, p2: Proj, act: mlx.mlx_dtype, dec: format.Decode) !bool {
    return (try swigluPlan(x, p1, p3, p2, act, dec)) != null;
}

/// F3: the host's shared expert at decode rows, `linear(astype(clampedSwiGLU(astype(linear(x, p1), f32),
/// astype(linear(x, p3), f32), limit), act), p2)` (limit > 0), in three launches fewer: the gate's and up's prepares and
/// GEMVs, one launch of their finishes, the SwiGLU and the down's prepare (`dense.text_swiglu_prep`), then the down's GEMV
/// and finish, bit for bit; y [..., out] in `act`. error.Exl3EngineNotImplemented where `sharedSwigluTakes` does not hold.
pub fn sharedSwiglu(s: mlx.mlx_stream, x: mlx.mlx_array, p1: Proj, p3: Proj, p2: Proj, limit: f32, act: mlx.mlx_dtype, dec: format.Decode) !mlx.mlx_array {
    const plan = (try swigluPlan(x, p1, p3, p2, act, dec)) orelse return error.Exl3EngineNotImplemented;
    if (!(limit > 0)) return error.Exl3EngineNotImplemented;
    const xsh = mlx.getShape(x);
    const gu = plan.gu;
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ @intCast(plan.rows), @intCast(gu.shape.in) }, 2, s));
    const g = try gu.prepareGemv(s, x2, p1.trellis, p1.suh, plan.rows);
    defer for (g) |a| freeArr(a);
    const u = try gu.prepareGemv(s, x2, p3.trellis, p3.suh, plan.rows);
    defer for (u) |a| freeArr(a);
    const pre = try (try forDense(gu)).swigluPrep(gu, s, g[0], g[1], p1.svh, u[0], u[1], p3.svh, p2.suh, limit, act, plan.rows);
    defer for (pre) |a| freeArr(a);
    const y2 = try plan.down.linearPrepared(s, pre[0], pre[1], p2.trellis, p2.svh, plan.rows);
    defer freeArr(y2);
    var osh: [8]c_int = undefined;
    if (xsh.len > osh.len) return error.BadExl3Shape;
    @memcpy(osh[0..xsh.len], xsh);
    osh[xsh.len - 1] = @intCast(plan.down.shape.out);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, osh[0..xsh.len].ptr, @intCast(xsh.len), s));
    return y;
}
