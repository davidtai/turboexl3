//! Ling-3.1-flash's MLA chain's plain projections at decode and verify rows in one GEMV launch (`mlaRows`, the outside
//! plugin turboexl3_ling's MLX_SERVE_LING_MLA_ROWS): kv_b (bf16 512 -> 16384, 16 MiB a layer) and the head gate g (bf16
//! 4096 -> 64) of each MLA layer. The host keeps each weight as the f16 checkpoint's [out, in] tensor viewed [in, out]
//! and cast to bf16 (its column-major layout kept) and runs a call's rows on MLX's gemv over that weight: one row as its
//! matmul, an exact verify's 2 to 16 rows as its denseRows (each row stored twice and the first viewed, one batched
//! matmul: a one-row gemv a row, so the weight is read once a row). `turboexl3_ling_mla_rows` runs MLX's gemv
//! statements at the tile MLX's matmul takes for the weight (`tileFor`) over up to `Config.mla_rows` rows a
//! threadgroup: each lane loads its weight tile once a K block and applies it to each of the chunk's rows, every row in
//! its own accumulators. A row's products, the order they are added in, the shuffle-down tree and the threadgroup sum
//! are the one-row gemv's, so each row's words are the host's at every row count and every cap: the rows a threadgroup
//! takes move no word.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const config = @import("../../core/config.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");

// ── texts ──

/// MLX's gemv statements up to the accumulate: the lane's place, the chunk's accumulators, a K block's weight tile (read
/// once for the chunk) and each row's input words
pub const rows_head = @embedFile("kernels/mla_rows/head.metal");
/// the accumulate: a row's product added to its accumulator, MLX's `result[tm] += inter[tn] * v_coeff[tn]`
pub const rows_acc = @embedFile("kernels/mla_rows/acc.metal");
/// the K loop's close, each row's shuffle-down sum, and with more than one simdgroup a K block (BN > 1) each
/// simdgroup's sums into threadgroup memory
pub const rows_simd = @embedFile("kernels/mla_rows/simd.metal");
/// the threadgroup sum: simdgroup 0 adds the other simdgroups' sums in simdgroup order
pub const rows_tgsum = @embedFile("kernels/mla_rows/tgsum.metal");
/// each stored row's outputs in bf16
pub const rows_store = @embedFile("kernels/mla_rows/store.metal");
/// `turboexl3_ling_mla_rows`
pub const text_rows: [:0]const u8 = rows_head ++ rows_acc ++ rows_simd ++ rows_tgsum ++ rows_store;

/// The launch's operands: x bf16 [ROWS, IN] and w bf16 [OUT, IN] row-major; out y bf16 [ROWS, OUT]
pub const rows_in = [_][*:0]const u8{ "x", "w" };
pub const rows_out = [_][*:0]const u8{"y"};

// ── the plan ──

/// The rows a call takes: decode's one to a verify's 16
pub const max_rows: u32 = @intCast(decode.Kernels.max_rows);
/// The most rows a threadgroup takes (`Config.mla_rows` at most): a lane keeps 4 accumulators a row, 32 at 8 rows
pub const max_cap: u32 = 8;

/// MLX's gemv tile at SM 1, SN 32, TM 4, TN 4: BM simdgroups of output rows by BN of K
pub const Tile = struct { bm: u32, bn: u32 };

/// The tile MLX's matmul takes for a weight it reads as `out` rows of `in` (MLX 0.32's `gemv_axbpy`, the matrix not
/// transposed: the host's column-major [in, out] weight): BM 8 at out >= 4096, else BM 4; BM 1 and BN 8 where in >= 16
/// out. Null where MLX takes a tile these texts do not hold (in <= 64: SM 8, SN 4; out < 4: TM 1).
pub fn tileFor(in: u32, out: u32) ?Tile {
    if (in <= 64 or out < 4) return null;
    if (in >= 16 * out) return .{ .bm = 1, .bn = 8 };
    return .{ .bm = if (out >= 4096) 8 else 4, .bn = 1 };
}

/// A call's launch: MLX's tile, the rows a threadgroup takes (`dense.Dense.rbCap`: the fewest chunks of at most `cap`
/// rows, the rows spread evenly over them) and the chunks, MLX's threadgroups a chunk (OUT / (BM TM)) and their threads
pub const Plan = struct { tile: Tile, rb: u32, chunks: u32, tgs: u32, threads: u32 };

/// The plan for `rows` rows of a weight of `out` rows of `in` at `cap` rows a threadgroup at most, or null where the
/// launch does not serve it: rows outside 1 to 16, a cap outside 1 to `max_cap`, a tile MLX's matmul runs that these
/// texts do not hold, a leftover K block (in not a multiple of 128 BN) or a tail simdgroup (out not one of 4 BM).
pub fn planFor(rows: u32, in: u32, out: u32, cap: u32) ?Plan {
    if (rows < 1 or rows > max_rows or cap < 1 or cap > max_cap) return null;
    const t = tileFor(in, out) orelse return null;
    if (in % (t.bn * 32 * 4) != 0 or out % (t.bm * 4) != 0) return null;
    const rb = dense.Dense.rbCap(rows, cap);
    return .{ .tile = t, .rb = rb, .chunks = (rows + rb - 1) / rb, .tgs = out / (t.bm * 4), .threads = 32 * t.bm * t.bn };
}

/// Whether `mlaRows` takes `rows` rows of a bf16 weight of `out` rows of `in` under the configuration in force
pub fn takes(rows: u32, in: u32, out: u32) bool {
    const cap: u32 = config.get().mla_rows;
    return cap != 0 and planFor(rows, in, out, cap) != null;
}

// ── the kernel and its configs: made at their first call ──

/// The weight shapes whose launch configs are kept (Ling's two: kv_b and the head gate); a call at another shape builds
/// its config for itself
const max_shapes = 4;
const Kept = struct {
    /// each kept shape's (in, out); (0, 0) free
    io: [max_shapes][2]u32 = @splat(.{ 0, 0 }),
    /// its configs by rows and cap
    cfg: [max_shapes][max_rows][max_cap]?mlx.mlx_fast_metal_kernel_config = @splat(@splat(@splat(null))),
    /// its engagement line printed
    said: [max_shapes]bool = @splat(false),
};
const lock = Global(std.atomic.Value(u32), "ling_mla_rows_lock", std.atomic.Value(u32).init(0));
const kernel_slot = Global(?mlx.mlx_fast_metal_kernel, "ling_mla_rows_kernel", null);
const kept = Global(Kept, "ling_mla_rows_kept", .{});

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ lock, kernel_slot, kept };

fn acquire() void {
    while (lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}

fn release() void {
    lock.get().*.store(0, .release);
}

fn kernelGet() !mlx.mlx_fast_metal_kernel {
    acquire();
    defer release();
    if (kernel_slot.get().*) |made| return made;
    const made = ks.newKernel("turboexl3_ling_mla_rows", &rows_in, &rows_out, text_rows, "", true) orelse return error.KernelCreateFailed;
    kernel_slot.get().* = made;
    return made;
}

/// x [rows, in] and w [out, in] as a call hands them
const Operands = struct { rows: u32, in: u32, out: u32 };

/// The call's operands: error.BadExl3Shape where x and w are not 2-D or their inputs differ; error.Exl3EngineNotImplemented
/// where either is not bf16 (the host's own matmul serves it)
fn operandsOf(x: mlx.mlx_array, w: mlx.mlx_array) !Operands {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len != 2 or ws.len != 2 or xs[0] < 1 or xs[1] < 1 or ws[0] < 1 or ws[1] != xs[1]) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .bfloat16) return error.Exl3EngineNotImplemented;
    return .{ .rows = @intCast(xs[0]), .in = @intCast(xs[1]), .out = @intCast(ws[0]) };
}

fn ci(v: u32) c_int {
    return @intCast(v);
}

/// The launch config of a plan: MLX's threadgroups a chunk on x, the chunks on y; template IN, OUT, BM, BN, RB, ROWS
fn buildCfg(op: Operands, pl: Plan) !mlx.mlx_fast_metal_kernel_config {
    return ks.buildConfig(.{
        .grid = .{ pl.tgs * pl.threads, pl.chunks, 1 },
        .threadgroup = .{ pl.threads, 1, 1 },
        .out_shapes = &.{&[_]c_int{ ci(op.rows), ci(op.out) }},
        .out_dtypes = &.{.bfloat16},
        .template = &.{
            .{ .name = "IN", .value = .{ .int = ci(op.in) } },
            .{ .name = "OUT", .value = .{ .int = ci(op.out) } },
            .{ .name = "BM", .value = .{ .int = ci(pl.tile.bm) } },
            .{ .name = "BN", .value = .{ .int = ci(pl.tile.bn) } },
            .{ .name = "RB", .value = .{ .int = ci(pl.rb) } },
            .{ .name = "ROWS", .value = .{ .int = ci(op.rows) } },
        },
    });
}

/// A call's config: the kept one of its shape, rows and cap (`shape`: the kept shape's index), or past the kept shapes a
/// new one the caller frees (`owned`)
const CallCfg = struct { cfg: mlx.mlx_fast_metal_kernel_config, owned: bool, shape: ?usize };

fn cfgFor(op: Operands, pl: Plan, cap: u32) !CallCfg {
    acquire();
    defer release();
    const k = kept.get();
    var at: ?usize = null;
    for (&k.io, 0..) |*io, j| {
        if (io[0] == op.in and io[1] == op.out) {
            at = j;
            break;
        }
        if (io[0] == 0) {
            io.* = .{ op.in, op.out };
            at = j;
            break;
        }
    }
    const si = at orelse return .{ .cfg = try buildCfg(op, pl), .owned = true, .shape = null };
    const slot = &k.cfg[si][op.rows - 1][cap - 1];
    if (slot.* == null) slot.* = try buildCfg(op, pl);
    return .{ .cfg = slot.*.?, .owned = false, .shape = si };
}

/// The first call of a kept shape on the launch: its line
fn engaged(op: Operands, pl: Plan, cap: u32, shape: ?usize) void {
    const i = shape orelse return;
    acquire();
    defer release();
    const k = kept.get();
    if (k.said[i]) return;
    k.said[i] = true;
    std.debug.print("[turboexl3] MLA rows engaged: one GEMV launch for a call's rows of a bf16 weight of {d} x {d} (MLX's gemv tile BM {d}, BN {d}), each weight tile read once for up to {d} rows; its first call {d} rows in {d} chunk(s) (Config.mla_rows)\n", .{ op.out, op.in, pl.tile.bm, pl.tile.bn, cap, op.rows, pl.chunks });
}

// ── the calls ──

/// Each row of x bf16 [rows, in] (rows 1 to 16) times w bf16 [out, in] row-major (the host's loaded [in, out]
/// column-major view transposed back: a view, no copy) in one launch, its threadgroups taking up to `Config.mla_rows`
/// rows each: y bf16 [rows, out], each row the words of MLX's one-row matmul of that row with the loaded weight (the
/// host's one-row call, and each row of its exact verify's denseRows). error.Exl3EngineNotImplemented before any launch
/// where it does not serve the call (`Config.mla_rows` 0, x or w not bf16, `planFor` null): the caller runs its own
/// matmul. error.BadExl3Shape where x and w do not fit together. The caller frees y.
pub fn mlaRows(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
    const cap: u32 = config.get().mla_rows;
    if (cap == 0) return error.Exl3EngineNotImplemented;
    return mlaRowsWith(s, x, w, cap);
}

/// `mlaRows` at `cap` rows a threadgroup at most (1 to `max_cap`), whatever the configuration (the tests, the timing).
/// The first call of each weight shape prints `[turboexl3] MLA rows engaged: ...`.
pub fn mlaRowsWith(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, cap: u32) !mlx.mlx_array {
    const op = try operandsOf(x, w);
    const pl = planFor(op.rows, op.in, op.out, cap) orelse return error.Exl3EngineNotImplemented;
    const k = try kernelGet();
    const c = try cfgFor(op, pl, cap);
    defer if (c.owned) {
        _ = mlx.mlx_fast_metal_kernel_config_free(c.cfg);
    };
    var o: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(k, c.cfg, &.{ x, w }, &o, s);
    engaged(op, pl, cap, c.shape);
    return o[0];
}

/// `mlaRowsWith` on another kernel of the launch's operands and template arguments (a test's planted fault), its config
/// built for the call; no engagement line
pub fn mlaRowsOn(s: mlx.mlx_stream, k: mlx.mlx_fast_metal_kernel, x: mlx.mlx_array, w: mlx.mlx_array, cap: u32) !mlx.mlx_array {
    const op = try operandsOf(x, w);
    const pl = planFor(op.rows, op.in, op.out, cap) orelse return error.Exl3EngineNotImplemented;
    const c = try buildCfg(op, pl);
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    var o: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(k, c, &.{ x, w }, &o, s);
    return o[0];
}

// ── tests (CPU) ──

const testing = std.testing;

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

test "turboexl3 ling mla rows texts: the launch is its five pieces in order; MLX's gemv statements (each row's tile read, products, shuffle-down sum, threadgroup sum and store) once each, the accumulate and the threadgroup sum each a piece of their own, no contraction pragma; the operands and template arguments it names, and no shape buffer" {
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/mla_rows_texts.metal"));
    try testing.expectEqualStrings(rows_head ++ rows_acc ++ rows_simd ++ rows_tgsum ++ rows_store, text_rows);
    // MLX's statements compile as MLX's own: no contraction pragma
    try testing.expectEqual(@as(usize, 0), count(text_rows, pin[0]));
    for (pin[1..]) |p| try testing.expectEqual(@as(usize, 1), count(text_rows, p));
    // what the planted faults swap (mla_rows_test.zig): the accumulate alone, and the threadgroup sum's loop
    try testing.expectEqualStrings(pin[3], std.mem.trim(u8, rows_acc, " \n"));
    try testing.expectEqual(@as(usize, 1), count(rows_tgsum, pin[6]));
    for (rows_in ++ rows_out) |n| try testing.expect(count(text_rows, std.mem.span(n)) > 0);
    for ([_][]const u8{ "IN", "OUT", "BM", "BN", "RB", "ROWS" }) |n| try testing.expect(count(text_rows, n) > 0);
    // MLX adds a shape, strides or ndim buffer for an operand named with those suffixes
    for (rows_in ++ rows_out) |n| for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| {
        var buf: [24]u8 = undefined;
        try testing.expectEqual(@as(usize, 0), count(text_rows, std.fmt.bufPrint(&buf, "{s}{s}", .{ std.mem.span(n), suf }) catch unreachable));
    };
}

test "turboexl3 ling mla rows plan: MLX's gemv tile for kv_b (BM 8) and the head gate (BN 8), and the router's (BM 4) its text names; the rows a threadgroup takes and their chunks at rows 1 to 16 and caps 1 to 8; a verify's 2 to 4 rows in one chunk from cap 4; the declines" {
    // kv_b 512 -> 16384; the head gate 4096 -> 64 (b_proj's shape: kdaproj/bmember runs BM 1, BN 8 there)
    try testing.expectEqual(Tile{ .bm = 8, .bn = 1 }, tileFor(512, 16384).?);
    try testing.expectEqual(Tile{ .bm = 1, .bn = 8 }, tileFor(4096, 64).?);
    // the router's 4096 -> 512 (router/logits_1: BM 4, BN 1)
    try testing.expectEqual(Tile{ .bm = 4, .bn = 1 }, tileFor(4096, 512).?);
    // MLX's other tiles: SM 8, SN 4 at K <= 64; TM 1 below 4 outputs
    try testing.expect(tileFor(64, 4096) == null and tileFor(4096, 2) == null);
    for (1..max_rows + 1) |ru| {
        const rows: u32 = @intCast(ru);
        for ([_]u32{ 1, 2, 4, 8 }) |cap| {
            for ([_][2]u32{ .{ 512, 16384 }, .{ 4096, 64 } }) |io| {
                const p = planFor(rows, io[0], io[1], cap).?;
                try testing.expect(p.rb >= 1 and p.rb <= cap and p.rb <= rows);
                // the fewest chunks of at most cap rows, every row in one
                try testing.expectEqual((rows + cap - 1) / cap, p.chunks);
                try testing.expectEqual((rows + p.rb - 1) / p.rb, p.chunks);
                try testing.expectEqual(@as(u32, 256), p.threads);
                try testing.expectEqual(if (io[1] == 16384) @as(u32, 512) else @as(u32, 16), p.tgs);
            }
        }
    }
    for (2..5) |ru| try testing.expectEqual(@as(u32, 1), planFor(@intCast(ru), 512, 16384, 4).?.chunks);
    try testing.expectEqual(@as(u32, 2), planFor(9, 512, 16384, 8).?.chunks);
    try testing.expectEqual(@as(u32, 5), planFor(9, 512, 16384, 8).?.rb);
    // the declines: no rows, past 16, cap 0 or past 8, a leftover K block (in not a multiple of 128 at BN 1, of 1024 at
    // BN 8), a tail simdgroup (out not a multiple of 4 BM)
    try testing.expect(planFor(0, 512, 16384, 8) == null);
    try testing.expect(planFor(max_rows + 1, 512, 16384, 8) == null);
    try testing.expect(planFor(4, 512, 16384, 0) == null);
    try testing.expect(planFor(4, 512, 16384, max_cap + 1) == null);
    try testing.expect(planFor(4, 576, 16384, 8) == null);
    try testing.expect(planFor(4, 1536, 64, 8) == null);
    try testing.expect(planFor(4, 512, 16400, 8) == null);
    try testing.expect(planFor(4, 64, 4096, 8) == null);
}
