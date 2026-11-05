//! The unfused decode chain: the same math as `decode.zig`'s form 1 (two launches), as the separately launched steps a
//! host would compose from GEMMs: per projection a prepare (f16 round, suh, H128, f16 round), a raw trellis GEMV (f32
//! inner sums) and a finish (f16 round, H128, svh, f16 round); SwiGLU; the top-k weighted reduce. Eleven launches per
//! layer (`Steps`; nothing in the library or its benchmark runs it now, and dense.zig's row-slot reference arm launches
//! `text_gemv`). Every kernel reads the same lane plan and decode helpers as the fused chain, so a comparison of the two
//! measures fusion alone (launch count, intermediate traffic), not the reader.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const format = @import("format.zig");
const lane = @import("lane.zig");
const decode = @import("decode.zig");

/// prepare: xp half [RK, DIM] = H128(f16(x[row]) * suh[slot]) per 128-block, f16; one simdgroup per (block, row-slot)
pub const text_prep = @embedFile("kernels/unfused/prep.metal");

/// raw GEMV: z f32 [RK, OUT] = xp[rk] . W[slot] (inner sums, unrounded); one 128-thread threadgroup per (128-wide out
/// block, row-slot), the four simdgroups split the input tiles, each lane loads its quad's four rows of a tile as one half4
pub const text_gemv = @embedFile("kernels/unfused/gemv.metal");

/// finish: y f32 [RK, DIM] = f16(H128(f16(z)) * svh[slot]) per 128-block; one simdgroup per (block, row-slot)
pub const text_finish = @embedFile("kernels/unfused/finish.metal");

/// SwiGLU: h = silu(g) * u, elementwise over [RK, DIM]
pub const text_swiglu = @embedFile("kernels/unfused/swiglu.metal");

/// reduce: y f32 [R, DIM] = sum_k scores[r k] * yk[r k], in slot order
pub const text_reduce = @embedFile("kernels/unfused/reduce.metal");

pub const header: [:0]const u8 = decode.header;

const prep_in = [_][*:0]const u8{ "x", "slots", "suh" };
const prep_out = [_][*:0]const u8{"xp"};
const gemv_in = [_][*:0]const u8{ "xp", "slots", "w" };
const gemv_out = [_][*:0]const u8{"z"};
const finish_in = [_][*:0]const u8{ "z", "slots", "svh" };
const finish_out = [_][*:0]const u8{"y"};
const swiglu_in = [_][*:0]const u8{ "g", "u" };
const swiglu_out = [_][*:0]const u8{"h"};
const reduce_in = [_][*:0]const u8{ "yk", "scores" };
const reduce_out = [_][*:0]const u8{"y"};

/// The step kernels of one geometry and their configs per row count.
pub const Steps = struct {
    geo: decode.Geometry,
    prep: mlx.mlx_fast_metal_kernel,
    gemv: mlx.mlx_fast_metal_kernel,
    finish: mlx.mlx_fast_metal_kernel,
    swiglu: mlx.mlx_fast_metal_kernel,
    reduce: mlx.mlx_fast_metal_kernel,
    cfg: [decode.Kernels.max_rows + 1]?Cfgs = @splat(null),

    const Cfgs = struct {
        prep_h: mlx.mlx_fast_metal_kernel_config, // prepare over hidden (gate, up)
        prep_i: mlx.mlx_fast_metal_kernel_config, // prepare over inter (down)
        gemv_gu: mlx.mlx_fast_metal_kernel_config,
        gemv_d: mlx.mlx_fast_metal_kernel_config,
        fin_i: mlx.mlx_fast_metal_kernel_config,
        fin_h: mlx.mlx_fast_metal_kernel_config,
        swiglu: mlx.mlx_fast_metal_kernel_config,
        reduce: mlx.mlx_fast_metal_kernel_config,
    };

    pub fn init(geo: decode.Geometry) !Steps {
        if (!geo.admitted()) return error.BadExl3Shape;
        return .{
            .geo = geo,
            .prep = ks.newKernel("turboexl3_prep", &prep_in, &prep_out, text_prep, header, true) orelse return error.KernelCreateFailed,
            .gemv = ks.newKernel("turboexl3_gemv", &gemv_in, &gemv_out, text_gemv, header, true) orelse return error.KernelCreateFailed,
            .finish = ks.newKernel("turboexl3_finish", &finish_in, &finish_out, text_finish, header, true) orelse return error.KernelCreateFailed,
            .swiglu = ks.newKernel("turboexl3_swiglu", &swiglu_in, &swiglu_out, text_swiglu, header, true) orelse return error.KernelCreateFailed,
            .reduce = ks.newKernel("turboexl3_reduce", &reduce_in, &reduce_out, text_reduce, header, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *Steps) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.prep_i, cc.gemv_gu, cc.gemv_d, cc.fin_i, cc.fin_h, cc.swiglu, cc.reduce }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        inline for (.{ self.prep, self.gemv, self.finish, self.swiglu, self.reduce }) |k| decode.freeKernel(k);
    }

    fn cfgs(self: *Steps, rows: usize) !Cfgs {
        if (rows == 0 or rows > decode.Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const g = self.geo;
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nhw = ks.TemplateArg{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } };
        const cb = ks.TemplateArg{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = ks.TemplateArg{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const topk = ks.TemplateArg{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } };
        const dim_h = ks.TemplateArg{ .name = "DIM", .value = .{ .int = H } };
        const dim_i = ks.TemplateArg{ .name = "DIM", .value = .{ .int = I } };
        const c: Cfgs = .{
            .prep_h = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float16}, .template = &.{ dim_h, topk } }),
            .prep_i = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float16}, .template = &.{ dim_i, ks.TemplateArg{ .name = "TOPK", .value = .{ .int = 1 } } } }),
            .gemv_gu = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 128, rk, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float32}, .template = &.{ nhw, cb, wb, .{ .name = "IN", .value = .{ .int = H } }, .{ .name = "OUT", .value = .{ .int = I } } } }),
            .gemv_d = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 128, rk, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float32}, .template = &.{ nhw, cb, wb, .{ .name = "IN", .value = .{ .int = I } }, .{ .name = "OUT", .value = .{ .int = H } } } }),
            .fin_i = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float32}, .template = &.{dim_i} }),
            .fin_h = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float32}, .template = &.{dim_h} }),
            .swiglu = try ks.buildConfig(.{ .grid = .{ rk * g.inter, 1, 1 }, .threadgroup = .{ 256, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float32}, .template = &.{} }),
            .reduce = try ks.buildConfig(.{ .grid = .{ g.hidden, @intCast(rows), 1 }, .threadgroup = .{ 256, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{.float32}, .template = &.{ dim_h, topk } }),
        };
        self.cfg[rows] = c;
        return c;
    }

    fn one(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, c, ins, &outs, s);
        return outs[0];
    }

    fn drop(a: mlx.mlx_array) void {
        _ = mlx.mlx_array_free(a);
    }

    /// The eleven launches; the caller frees y.
    pub fn moe(self: *Steps, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: decode.Kernels.Bank, rows: usize) !mlx.mlx_array {
        const c = try self.cfgs(rows);
        // gate
        const xg = try one(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh }, s);
        defer drop(xg);
        const zg = try one(self.gemv, c.gemv_gu, &.{ xg, slots, bank.gate_t }, s);
        defer drop(zg);
        const yg = try one(self.finish, c.fin_i, &.{ zg, slots, bank.gate_svh }, s);
        defer drop(yg);
        // up
        const xu = try one(self.prep, c.prep_h, &.{ x, slots, bank.up_suh }, s);
        defer drop(xu);
        const zu = try one(self.gemv, c.gemv_gu, &.{ xu, slots, bank.up_t }, s);
        defer drop(zu);
        const yu = try one(self.finish, c.fin_i, &.{ zu, slots, bank.up_svh }, s);
        defer drop(yu);
        // SwiGLU, then down over h as [RK, inter] rows (prepare with TOPK 1: row-slot i is its own x row)
        const h = try one(self.swiglu, c.swiglu, &.{ yg, yu }, s);
        defer drop(h);
        const hd = try one(self.prep, c.prep_i, &.{ h, slots, bank.down_suh }, s);
        defer drop(hd);
        const zd = try one(self.gemv, c.gemv_d, &.{ hd, slots, bank.down_t }, s);
        defer drop(zd);
        const yk = try one(self.finish, c.fin_h, &.{ zd, slots, bank.down_svh }, s);
        defer drop(yk);
        return one(self.reduce, c.reduce, &.{ yk, scores }, s);
    }
};

test "turboexl3 unfused texts: the inputs each signature names are read, the outputs written" {
    const pin = comptime @import("test_support.zig").pins(@embedFile("kernels/test/pins/unfused_texts.metal"));
    const t = std.testing;
    for (prep_in) |n| try t.expect(std.mem.indexOf(u8, text_prep, std.mem.span(n)) != null);
    for (gemv_in) |n| try t.expect(std.mem.indexOf(u8, text_gemv, std.mem.span(n)) != null);
    for (finish_in) |n| try t.expect(std.mem.indexOf(u8, text_finish, std.mem.span(n)) != null);
    try t.expect(std.mem.indexOf(u8, text_prep, pin[0]) != null);
    try t.expect(std.mem.indexOf(u8, text_gemv, pin[1]) != null);
    try t.expect(std.mem.indexOf(u8, text_reduce, pin[2]) != null);
}
