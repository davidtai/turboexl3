//! The grouped (block-diagonal) EXL3 linear at decode rows: DeepSeek-V4.1's attention `wo_a` (the release's
//! inference/model.py, `o = einsum("bsgd,grd->bsgr", o, wo_a)`). x [rows, G, in] meets G slices, each a linear in -> out
//! (`root.Proj` with the slices stacked: trellis U16/I16 [G, in/16, out/16, n], suh F16 [G, in], svh F16 [G, out]), and
//! y [rows, G, out] holds slice g's linear of x[r, g] at [r, g]. `root.linearGrouped` is the entry; it serves prefill rows
//! as `linear` of each slice.
//!
//! A decode call (rows <= DECODE_ROWS_MAX) is three launches whatever G, each an existing kernel's statements with its own
//! template arguments:
//! 1. `turboexl3_grouped_prep`: dense's prepare with its block sums (`dense.text_prep_s`, lane order) over each row's G
//!    slices as one row of G in values. suh [G, in] is read flat and every 128-block lies inside one slice, so each slice's
//!    prepared values and block sums are dense's.
//! 2. `turboexl3_grouped_gemv_d` (below: dense's multi-row GEMV over the slices), or `turboexl3_grouped_gemv`: the routed
//!    down GEMV (`decode.text_down3`) with row-slot r G + g on expert g: split-K planes over the stacked trellis. From two
//!    rows up the row-slots on one slice pair under one decode (GROUP 2).
//! 3. `turboexl3_grouped_finish`: the routed down finish (`decode.text_dfin3`) at top-1 with every score 1, so nothing is
//!    summed over slots: each row-slot's planes in slice order, the deferred MUL1 map, the finish and the output dtype.
//!    (`text_dfin7` would not do: its store covers 32 TOPK outputs of each 128-block, all of them only from TOPK 4.)
//! The K partition is dense's for one slice (`dense.Dense.ksFor(in, out)`), and each step runs the dense chain's
//! statements in its order (the GEMV's per-lane tile order, quad fold and simdgroup sum; the finish's planes in slice
//! order). So each slice holds the values `linear` gives on it (DEC-ONE's or the GEMV chain's: one set of bytes), but for
//! the sign of an exact zero: the finish's score product adds +0.
//!
//! The GEMV of step 2 has two texts with one set of planes. The routed down GEMV decodes a slice's weights once per row
//! pair (GROUP 2: row-slots r G + g and (r + 1) G + g share a lead threadgroup), so ceil(rows / 2) times; `text_gemv_d`
//! (`Grouped.dense_gemv`, the default) is dense's multi-row GEMV over the slices, one threadgroup per (output tiles, row
//! chunk and slice, K slice), so a slice's weights are decoded once per row chunk (`dense.Dense.chainRb`'s: one chunk up
//! to 8 rows, on the lean text past 5). Its statements from the input loads to the fold are `dense.text_dgemv`'s (or the
//! lean text's) verbatim and its store sums the simdgroups as text_dgemv's does, into the routed GEMV's plane layout:
//! the same per-lane FMA order, quad fold and simdgroup sum on the same K partition, so the same planes.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const format = @import("format.zig");
const decode = @import("decode.zig");
const dense = @import("dense.zig");

const prep_in = [_][*:0]const u8{ "x", "suh" };
const prep_out = [_][*:0]const u8{ "xp", "xs" };
const gemv_in = [_][*:0]const u8{ "hdp", "slots", "down_t" };
const gemv_out = [_][*:0]const u8{"zd"};
const finish_in = [_][*:0]const u8{ "zd", "slots", "scores", "down_svh", "hsx" };
const finish_out = [_][*:0]const u8{"y"};
pub const gemv_d_in = [_][*:0]const u8{ "xp", "w" };
pub const gemv_d_out = [_][*:0]const u8{"z"};

/// The GEMV on dense's multi-row GEMV: z f32 [KS, ROWS G, OUT], row-slot r G + g's K-slice partials at z[ks, r G + g] (the
/// routed down GEMV's planes); xp half [ROWS, G IN] (the prepare's rows, lane order), w the stacked trellis [G, IN/16,
/// OUT/16, n]. Grid (OUT / (16 BT) threadgroups x 128, chunks x G, KS): the threadgroup's slice g = y mod G, its row chunk
/// y / G. `text_gemv_d` for chunks of up to `dense.Dense.max_rb` rows, `text_gemv_dl` (the lean codeword loop) past them.
pub const text_gemv_d: [:0]const u8 = gemv_head ++ dense.dgemv_x ++ dense.dgemv_2 ++ dense.dgemv_m ++ dense.dgemv_fold ++ gemv_store;
pub const text_gemv_dl: [:0]const u8 = gemv_head ++ dense.dgemv_2 ++ dense.dgemv_j ++ dense.dgemv_fold ++ gemv_store;
/// `dense.text_dgemv`'s setup with slice g's trellis and inputs (the template IN is a slice's inputs, xp's row G IN)
pub const gemv_head = @embedFile("kernels/grouped/gemv_head.metal");
/// `dense.text_dgemv`'s store (the simdgroups summed in order) at row-slot r G + g of the routed planes
pub const gemv_store = @embedFile("kernels/grouped/gemv_store.metal");

/// The most slices a grouped linear takes (row-slots: at most 16 rows x 64)
pub const max_groups: u32 = 64;

/// One grouped linear's shape: G slices of in -> out at one rate and decode, the output in the activation dtype.
pub const Shape = struct {
    groups: u32,
    in: u32,
    out: u32,
    rate: format.Rate,
    dec: format.Decode,
    out_dtype: mlx.mlx_dtype,

    pub fn eql(a: Shape, b: Shape) bool {
        return a.groups == b.groups and a.in == b.in and a.out == b.out and a.rate.n == b.rate.n and
            a.dec.codebook == b.dec.codebook and a.dec.window == b.dec.window and a.out_dtype == b.out_dtype;
    }

    /// Dense's widths (multiples of 128, whole 4-tile parts per K slice) and 1..max_groups slices
    pub fn admitted(sh: Shape) bool {
        return sh.groups >= 1 and sh.groups <= max_groups and sh.in > 0 and sh.out > 0 and sh.in % 128 == 0 and
            sh.out % 128 == 0 and (sh.in / 16) % 4 == 0;
    }
};

/// The decode launch shape. `ks`: the K slices, dense's split for one slice, so a slice's sums are the dense GEMV's.
/// `spt`: slices per GEMV threadgroup; `bt`: output tiles per GEMV threadgroup. Those two are launch shape only: a
/// row-slot's sums do not depend on them.
pub const Plan = struct { ks: u32, spt: u32, bt: u32 };

pub fn planFor(in: u32, out: u32) Plan {
    const k = dense.Dense.ksFor(in, out);
    // out % 128 == 0, so two output tiles per threadgroup always tile it
    return .{ .ks = k, .spt = if (k % 4 == 0) 4 else if (k % 2 == 0) 2 else 1, .bt = 2 };
}

/// The kernels of one grouped shape and its launches per decode row count.
pub const Grouped = struct {
    shape: Shape,
    plan: Plan,
    /// MUL1: the GEMV sums x times the codeword's byte sum and the finish applies the affine map (dense's `deferFor`)
    defer_affine: bool,
    prep: mlx.mlx_fast_metal_kernel,
    gemv: mlx.mlx_fast_metal_kernel,
    finish: mlx.mlx_fast_metal_kernel,
    /// the GEMV on dense's multi-row GEMV (`text_gemv_d`, `text_gemv_dl`)
    gemv_d: mlx.mlx_fast_metal_kernel,
    gemv_dl: mlx.mlx_fast_metal_kernel,
    /// step 2 on `text_gemv_d` (one decode of a slice's weights per row chunk); false the routed down GEMV (one per row
    /// pair), the tests' reference arm. The same bytes. Set before the configs of a row count are built (`resetCfgs`).
    dense_gemv: bool = true,
    /// `text_gemv_d`'s output tiles per threadgroup per row count, 0 for `dense.Dense.btLaunch`'s: launch shape only
    bt_set: [decode.Kernels.max_rows + 1]u8 = @splat(0),
    cfg: [decode.Kernels.max_rows + 1]?Cfg = @splat(null),

    /// One decode row count's three configs and constant operands: `slots` u32 (row-slot r G + g holds g) and `ones` f32
    /// (the unit scores), each at least 8 long (MLX binds a smaller input in the constant address space). `dense`: `gemv`
    /// is `text_gemv_d`'s config (`text_gemv_dl`'s with `lean`), else the routed down GEMV's.
    pub const Cfg = struct {
        prep: mlx.mlx_fast_metal_kernel_config,
        gemv: mlx.mlx_fast_metal_kernel_config,
        finish: mlx.mlx_fast_metal_kernel_config,
        slots: mlx.mlx_array,
        ones: mlx.mlx_array,
        dense: bool = false,
        lean: bool = false,
    };

    pub fn init(shape: Shape) !Grouped {
        if (!shape.admitted()) return error.BadExl3Shape;
        const prep = ks.newKernel("turboexl3_grouped_prep", &prep_in, &prep_out, dense.text_prep_s, decode.header, true) orelse return error.KernelCreateFailed;
        errdefer decode.freeKernel(prep);
        const gemv = ks.newKernel("turboexl3_grouped_gemv", &gemv_in, &gemv_out, decode.text_down3, decode.header, true) orelse return error.KernelCreateFailed;
        errdefer decode.freeKernel(gemv);
        const finish = ks.newKernel("turboexl3_grouped_finish", &finish_in, &finish_out, decode.text_dfin3, decode.header, true) orelse return error.KernelCreateFailed;
        errdefer decode.freeKernel(finish);
        const gemv_d = ks.newKernel("turboexl3_grouped_gemv_d", &gemv_d_in, &gemv_d_out, text_gemv_d, decode.header, true) orelse return error.KernelCreateFailed;
        errdefer decode.freeKernel(gemv_d);
        const gemv_dl = ks.newKernel("turboexl3_grouped_gemv_dl", &gemv_d_in, &gemv_d_out, text_gemv_dl, decode.header, true) orelse return error.KernelCreateFailed;
        return .{ .shape = shape, .plan = planFor(shape.in, shape.out), .defer_affine = dense.Dense.deferFor(shape.dec), .prep = prep, .gemv = gemv, .finish = finish, .gemv_d = gemv_d, .gemv_dl = gemv_dl };
    }

    /// Drops every built row count's configs and operands, so the next call of each builds them from the fields as they
    /// are then (`dense_gemv`, `bt_set`)
    pub fn resetCfgs(self: *Grouped) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep, cc.gemv, cc.finish }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            _ = mlx.mlx_array_free(cc.slots);
            _ = mlx.mlx_array_free(cc.ones);
            c.* = null;
        };
    }

    pub fn deinit(self: *Grouped) void {
        self.resetCfgs();
        inline for (.{ self.prep, self.gemv, self.finish, self.gemv_d, self.gemv_dl }) |k| decode.freeKernel(k);
    }

    /// `text_gemv_d`'s launch at `rows` rows: dense's chain chunks (`dense.Dense.chainRb` with the lean cap: one chunk up
    /// to 8 rows, the lean text past `dense.Dense.max_rb`), its tiles per threadgroup (`bt_set`, else
    /// `dense.Dense.btLaunch`'s for its threadgroups at BT 1)
    fn denseGemvCfg(self: *const Grouped, rows: usize) !struct { cfg: mlx.mlx_fast_metal_kernel_config, lean: bool } {
        const sh = self.shape;
        const rows_u: u32 = @intCast(rows);
        const rb = dense.Dense.rbCap(rows_u, dense.Dense.lean_rb);
        const chunks = (rows_u + rb - 1) / rb;
        const o = self.bt_set[rows];
        const bt: u32 = if (o != 0) o else dense.Dense.btLaunch(rb, (sh.out / 16) * chunks * sh.groups * self.plan.ks);
        if (8 % bt != 0) return error.BadExl3Shape;
        const cfg = try ks.buildConfig(.{
            .grid = .{ (sh.out / (16 * bt)) * 128, chunks * sh.groups, self.plan.ks },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{&[_]c_int{ @intCast(self.plan.ks), @intCast(rows_u * sh.groups), @intCast(sh.out) }},
            .out_dtypes = &.{.float32},
            .template = &.{
                .{ .name = "NHW", .value = .{ .int = @intCast(sh.rate.n) } },
                .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(sh.dec.codebook) } },
                .{ .name = "WBITS", .value = .{ .int = @intCast(sh.dec.window.bits()) } },
                .{ .name = "IN", .value = .{ .int = @intCast(sh.in) } },
                .{ .name = "OUT", .value = .{ .int = @intCast(sh.out) } },
                .{ .name = "KS", .value = .{ .int = @intCast(self.plan.ks) } },
                .{ .name = "BT", .value = .{ .int = @intCast(bt) } },
                .{ .name = "RB", .value = .{ .int = @intCast(rb) } },
                .{ .name = "ROWS", .value = .{ .int = @intCast(rows_u) } },
                .{ .name = "GROUPS", .value = .{ .int = @intCast(sh.groups) } },
                .{ .name = "DEFER", .value = .{ .int = if (self.defer_affine) 1 else 0 } },
            },
        });
        return .{ .cfg = cfg, .lean = rb > dense.Dense.max_rb };
    }

    /// The launches of `rows` decode rows (cached): the prepare over [rows, G in], the GEMV over rows G row-slots, the
    /// finish into [rows G, out]
    pub fn cfgs(self: *Grouped, rows: usize) !Cfg {
        if (rows == 0 or rows > decode.Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const sh = self.shape;
        const p = self.plan;
        const rk: u32 = @as(u32, @intCast(rows)) * sh.groups;
        const rkc: c_int = @intCast(rk);
        const r: c_int = @intCast(rows);
        // a row's G slices as one row
        const width: u32 = sh.groups * sh.in;
        const OUT: c_int = @intCast(sh.out);
        const ksv: c_int = @intCast(p.ks);
        const def_t = ks.TemplateArg{ .name = "DEFER", .value = .{ .int = if (self.defer_affine) 1 else 0 } };
        const prep = try ks.buildConfig(.{
            .grid = .{ (width / 128) * 32, @intCast(rows), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ r, @intCast(width) }, &[_]c_int{ r, @intCast(width / 128) } },
            .out_dtypes = &.{ .float16, .float32 },
            .template = &.{ .{ .name = "DIM", .value = .{ .int = @intCast(width) } }, .{ .name = "LANE_ORDER", .value = .{ .int = 1 } } },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(prep);
        // step 2 on dense's multi-row GEMV, or the routed down GEMV (its `errdefer` below frees either)
        const gd = if (self.dense_gemv) try self.denseGemvCfg(rows) else null;
        const gemv = if (gd) |d| d.cfg else try ks.buildConfig(.{
            .grid = .{ (sh.out / (16 * p.bt)) * 128, rk, p.ks / p.spt },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{&[_]c_int{ ksv, rkc, OUT }},
            .out_dtypes = &.{.float32},
            .template = &.{
                .{ .name = "NHW", .value = .{ .int = @intCast(sh.rate.n) } },
                .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(sh.dec.codebook) } },
                .{ .name = "WBITS", .value = .{ .int = @intCast(sh.dec.window.bits()) } },
                .{ .name = "HID", .value = .{ .int = OUT } },
                .{ .name = "INTER", .value = .{ .int = @intCast(sh.in) } },
                .{ .name = "RKC", .value = .{ .int = rkc } },
                .{ .name = "KS", .value = .{ .int = ksv } },
                .{ .name = "SPT", .value = .{ .int = @intCast(p.spt) } },
                .{ .name = "BT", .value = .{ .int = @intCast(p.bt) } },
                .{ .name = "GROUP", .value = .{ .int = if (rows >= 2) 2 else 1 } },
                def_t,
            },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gemv);
        const finish = try ks.buildConfig(.{
            .grid = .{ (sh.out / 128) * 128, rk, 1 },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{&[_]c_int{ rkc, OUT }},
            .out_dtypes = &.{sh.out_dtype},
            .template = &.{
                .{ .name = "HID", .value = .{ .int = OUT } },
                .{ .name = "TOPK", .value = .{ .int = 1 } },
                .{ .name = "RKC", .value = .{ .int = rkc } },
                .{ .name = "KS", .value = .{ .int = ksv } },
                .{ .name = "OUT_T", .value = .{ .dtype = sh.out_dtype } },
                def_t,
                .{ .name = "NBI", .value = .{ .int = @intCast(sh.in / 128) } },
            },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(finish);
        var sbuf: [decode.Kernels.max_rows * max_groups]u32 = undefined;
        var obuf: [decode.Kernels.max_rows * max_groups]f32 = undefined;
        const n: usize = @max(rk, 8);
        for (sbuf[0..n], obuf[0..n], 0..) |*sl, *o, i| {
            sl.* = @intCast(i % sh.groups);
            o.* = 1.0;
        }
        const slots = mlx.mlx_array_new_data(&sbuf, &[_]c_int{@intCast(n)}, 1, .uint32);
        if (slots.ctx == null) return error.MlxError;
        errdefer _ = mlx.mlx_array_free(slots);
        const ones = mlx.mlx_array_new_data(&obuf, &[_]c_int{@intCast(n)}, 1, .float32);
        if (ones.ctx == null) return error.MlxError;
        errdefer _ = mlx.mlx_array_free(ones);
        try mlx.check(mlx.mlx_array_eval(slots));
        try mlx.check(mlx.mlx_array_eval(ones));
        const c: Cfg = .{ .prep = prep, .gemv = gemv, .finish = finish, .slots = slots, .ones = ones, .dense = gd != null, .lean = if (gd) |d| d.lean else false };
        self.cfg[rows] = c;
        return c;
    }

    /// y [rows G, out] in `shape.out_dtype` (row-slot r G + g = slice g of row r) for x [rows, G in] (any float dtype), the
    /// stacked trellis, suh and svh: the three launches. The caller frees y.
    pub fn decodeRows(self: *Grouped, s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        const c = try self.cfgs(rows);
        // the prepared rows (lane order) and their block sums
        var po: [2]mlx.mlx_array = undefined;
        try ks.applyConfig(self.prep, c.prep, &.{ x, suh }, &po, s);
        defer for (po) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var zd: [1]mlx.mlx_array = undefined;
        if (c.dense)
            try ks.applyConfig(if (c.lean) self.gemv_dl else self.gemv_d, c.gemv, &.{ po[0], trellis }, &zd, s)
        else
            try ks.applyConfig(self.gemv, c.gemv, &.{ po[0], c.slots, trellis }, &zd, s);
        defer _ = mlx.mlx_array_free(zd[0]);
        var y: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(self.finish, c.finish, &.{ zd[0], c.slots, c.ones, svh, po[1] }, &y, s);
        return y[0];
    }
};

// ── tests ──

const testing = std.testing;

fn has(text: []const u8, name: []const u8) bool {
    return std.mem.indexOf(u8, text, name) != null;
}

test "turboexl3 grouped: the reused texts read every operand the launches bind and every template constant they pass" {
    for (prep_in ++ prep_out) |n| try testing.expect(has(dense.text_prep_s, std.mem.span(n)));
    for ([_][]const u8{ "DIM", "LANE_ORDER" }) |n| try testing.expect(has(dense.text_prep_s, n));
    for (gemv_in ++ gemv_out) |n| try testing.expect(has(decode.text_down3, std.mem.span(n)));
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "RKC", "KS", "SPT", "BT", "GROUP", "DEFER" }) |n| try testing.expect(has(decode.text_down3, n));
    for (finish_in ++ finish_out) |n| try testing.expect(has(decode.text_dfin3, std.mem.span(n)));
    for ([_][]const u8{ "HID", "TOPK", "RKC", "KS", "OUT_T", "DEFER", "NBI" }) |n| try testing.expect(has(decode.text_dfin3, n));
}

test "turboexl3 grouped: DeepSeek-V4.1's wo_a (8 slices of 4096 -> 1024) is admitted, its K split is the dense linear's" {
    const sh: Shape = .{ .groups = 8, .in = 4096, .out = 1024, .rate = format.Rate.fromK(6), .dec = .mul1, .out_dtype = .bfloat16 };
    try testing.expect(sh.admitted());
    const p = planFor(sh.in, sh.out);
    try testing.expectEqual(dense.Dense.ksFor(sh.in, sh.out), p.ks);
    try testing.expectEqual(@as(u32, 16), p.ks);
    // the GEMV's partition: four parts of whole input tiles per slice; the launch shape divides the slices and the tiles
    try testing.expectEqual(@as(u32, 0), (sh.in / 16) % (p.ks * 4));
    try testing.expectEqual(@as(u32, 0), p.ks % p.spt);
    try testing.expectEqual(@as(u32, 0), (sh.out / 16) % p.bt);
    var bad = sh;
    bad.out = 1000;
    try testing.expect(!bad.admitted());
    bad = sh;
    bad.in = 4160;
    try testing.expect(!bad.admitted());
    bad = sh;
    bad.groups = 0;
    try testing.expect(!bad.admitted());
    bad = sh;
    bad.groups = max_groups + 1;
    try testing.expect(!bad.admitted());
    try testing.expect(sh.eql(sh));
    bad = sh;
    bad.out_dtype = .float32;
    try testing.expect(!sh.eql(bad));
}
