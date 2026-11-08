//! `upSwiglu` (models/ling/swiglu.zig; the outside plugin turboexl3_ling's MLX_SERVE_LING_SWIGLU_EPI): up's finish at
//! prompt rows running the host's SwiGLU of the gate's finished value and its own, held word for word to the chain it
//! replaces.
//! - CPU (always): text_finish is finish_v ++ finish_store; each epilogue text is finish_v, the store loop's opening,
//!   the shared expert's SwiGLU statements and the loop's close; the opening keeps text_finish's store expression and
//!   element index, reads the gate's value at that element, and the close stores h (pins in
//!   kernels/test/pins/swiglu_texts.metal).
//! - GPU (TURBOEXL3_GPU=1, else skipped), at Ling-3.1-flash's shapes with the loader's I16 trellis words: the shared
//!   expert's gate and up (4096 -> 1280 at K 6) at limits 0 and 10 (`share_expert_swiglu_limit_list`), the dense layers'
//!   (4096 -> 10240 at K 4) at limit 0; x the MLP's input [1, S, 4096] in bf16 (f16 at two row counts), the gate's and the
//!   up's outputs reaching past the limit. At rows 17, 31 and 32 (the register path) and 33, 100, 1965, 4096 and 16384
//!   (the matmul path; the standard request's spans are 4096, 1965 and 31 rows, a prompt without spans runs layer-major
//!   calls of 16384), h against the composed chain (the gate and the up by
//!   `linear`; at limit 0 the host's SwiGLU as MLX's sigmoid, multiply and multiply in x's dtype, the words of mlx-serve's
//!   fusedSwiGLU; at 10 the plugin's swigluClampOut, MLX's ops in f32), word for word; the declines (rows 1 to 16, an f32
//!   x) and the refusals (a clamp above 128, the wrong table, a gate of the wrong size). Lines: TURBOEXL3_LING_SWIGLU_EPI
//!   (then TURBOEXL3_LING_SWIGLU_EPI_DONE).
//! - Timing (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; asserts nothing, guidance only): at rows 4096 and 16384, up's
//!   `linear` alone, then with one elementwise launch over the gate and up (fusedSwiGLU's launch and traffic), with
//!   swigluClampOut's ops (limit 10), and `upSwiglu`. Lines: TURBOEXL3_LING_TIME swiglu_epi.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const dense = @import("../../core/dense.zig");
const root = @import("../../root.zig");
const shared = @import("shared.zig");
const swiglu = @import("swiglu.zig");
const ds_test = @import("../deepseek_v41/deepseek_test.zig");

const testing = std.testing;
const A = mlx.mlx_array;

const hidden: u32 = 4096;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

test "turboexl3 ling swiglu texts: text_finish is finish_v ++ finish_store; each epilogue is finish_v, the store loop with up's value kept and the gate's read, the host's SwiGLU, h stored" {
    try testing.expectEqualStrings(dense.finish_v ++ dense.finish_store, dense.text_finish);
    try testing.expectEqualStrings(dense.finish_v ++ swiglu.epi_open ++ shared.sgu_swiglu ++ swiglu.epi_close, swiglu.text_finish_sgu);
    try testing.expectEqualStrings(dense.finish_v ++ swiglu.epi_open ++ shared.sgu_swiglu_clamp_out ++ swiglu.epi_close, swiglu.text_finish_sguco);
    const pin = comptime support.pins(@embedFile("kernels/test/pins/swiglu_texts.metal"));
    // up's value and element: text_finish's store expression and index, once each in the finish's store and the opening
    for (pin[0..2]) |line| {
        try testing.expectEqual(@as(usize, 1), count(dense.finish_store, line));
        try testing.expectEqual(@as(usize, 1), count(swiglu.epi_open, line));
    }
    try testing.expectEqual(@as(usize, 1), count(swiglu.epi_open, pin[2]));
    try testing.expectEqual(@as(usize, 1), count(swiglu.epi_close, pin[3]));
    // the operands by name, no shape buffers
    for ([_][]const u8{ swiglu.text_finish_sgu, swiglu.text_finish_sguco }) |t| for ([_][]const u8{ "gate_shape", "gate_strides", "gate_ndim", "z_shape", "svh_shape" }) |n| try testing.expectEqual(@as(usize, 0), count(t, n));
    try testing.expect(count(swiglu.text_finish_sgu, "sigtab") > 0 and count(swiglu.text_finish_sgu, "sig32") == 0);
    try testing.expect(count(swiglu.text_finish_sguco, "sig32") > 0 and count(swiglu.text_finish_sguco, pin[4]) == 1 and count(swiglu.text_finish_sguco, "sigtab") == 0);
}

// ── GPU ──

/// A projection [in -> out] of random trellis words as the loader hands them (I16), its svh scaled so outputs reach
/// about `scale` in magnitude
fn randProj(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, rnd: std.Random, in: u32, out: u32, n: u32, scale: f32) !root.Proj {
    const t = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(t);
    rnd.bytes(std.mem.sliceAsBytes(t));
    const su = try a.alloc(u16, in);
    defer a.free(su);
    for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, out);
    defer a.free(sv);
    for (sv) |*v| v.* = format.f32ToF16Bits(scale * (0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    const u = try g.hostArray(std.mem.sliceAsBytes(t), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16);
    var v = mlx.mlx_array_new();
    mlx.check(mlx.mlx_view(&v, u, .int16, s)) catch |e| {
        _ = mlx.mlx_array_free(v);
        return e;
    };
    return .{
        .trellis = try g.adopt(v),
        .suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(out)}, .float16),
    };
}

/// x [1, rows, hidden] in `dt`: normal draws, every fifth row times 16 (its outputs well past the limit)
fn randX(a: std.mem.Allocator, g: *G, rnd: std.Random, rows: u32, dt: mlx.mlx_dtype) !A {
    const xh = try a.alloc(f32, @as(usize, rows) * hidden);
    defer a.free(xh);
    for (xh, 0..) |*v, i| {
        const big: f32 = if ((i / hidden) % 5 == 4) 16 else 1;
        v.* = @bitCast(@as(u32, support.bf16Bits(big * rnd.floatNorm(f32))) << 16);
    }
    return g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ 1, ci(rows), ci(hidden) }, .float32), dt);
}

fn op2(g: *G, comptime f: anytype, a: A, b: A) !A {
    var r = mlx.mlx_array_new();
    mlx.check(f(&r, a, b, g.s)) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
}

/// MLX's clip (mlx/ops.cpp): maximum with the lower bound, then minimum with the upper
fn clip(g: *G, a: A, lo: A, hi: A) !A {
    return op2(g, mlx.mlx_minimum, try op2(g, mlx.mlx_maximum, a, lo), hi);
}

/// The host's SwiGLU of the served gate and up, op for op: at limit 0 MLX's sigmoid, multiply and multiply in their
/// dtype (the words of mlx-serve's fusedSwiGLU, which reads MLX's sigmoid from its table); above, the plugin's
/// swigluClampOut (astype f32, sigmoid, multiply, minimum, clip, multiply, astype)
fn hostSwiglu(g: *G, gv: A, uv: A, lim: u32) !A {
    if (lim == 0) return g.mul(try g.mul(gv, try g.sigmoid(gv)), uv);
    const g32 = try g.astype(gv, .float32);
    const u32_ = try g.astype(uv, .float32);
    const act = try g.mul(g32, try g.sigmoid(g32));
    const hi = try g.scalar(@floatFromInt(lim), .float32);
    const lo = try g.scalar(-@as(f64, @floatFromInt(lim)), .float32);
    const am = try op2(g, mlx.mlx_minimum, act, hi);
    const uc = try clip(g, u32_, lo, hi);
    return g.astype(try g.mul(am, uc), mlx.mlx_array_dtype(gv));
}

fn wordsDiffer(a: std.mem.Allocator, g: *G, x: A, y: A, n: usize) !usize {
    const p = try a.alloc(f32, n);
    defer a.free(p);
    const q = try a.alloc(f32, n);
    defer a.free(q);
    _ = try g.hostF32(try g.astype(x, .float32), p);
    _ = try g.hostF32(try g.astype(y, .float32), q);
    var d: usize = 0;
    for (p, q) |u, w| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return d;
}

/// The values of `x` (n of them) past `lim`: the SiLU above it, and beyond +-lim
fn overLimit(a: std.mem.Allocator, g: *G, gv: A, uv: A, n: usize, lim: u32) ![2]usize {
    const hv = try a.alloc(f32, n);
    defer a.free(hv);
    const L: f32 = @floatFromInt(lim);
    var over: [2]usize = .{ 0, 0 };
    _ = try g.hostF32(try g.astype(gv, .float32), hv);
    for (hv) |v| over[0] += @intFromBool(v * (1.0 / (1.0 + @exp(-v))) > L);
    _ = try g.hostF32(try g.astype(uv, .float32), hv);
    for (hv) |v| over[1] += @intFromBool(@abs(v) > L);
    return over;
}

const Case = struct { name: []const u8, out: u32, n: u32, lims: []const u32 };
/// The shared expert's gate | up (K 6, both limits) and the dense layers' (K 4, no clamp)
const cases = [_]Case{
    .{ .name = "shared", .out = 1280, .n = 96, .lims = &.{ 0, 10 } },
    .{ .name = "dense", .out = 10240, .n = 64, .lims = &.{0} },
};
const test_rows = [_]u32{ 17, 31, 32, 33, 100, 1965, 4096, 16384 };

test "turboexl3 gpu ling swiglu epi: up's finish running the host's SwiGLU gives the composed chain's words at the shared expert's and the dense layers' shapes, rows 17 to 16384, limits 0 and 10 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    std.debug.print("TURBOEXL3_LING_SWIGLU_EPI_START\n", .{});
    var prng = std.Random.DefaultPrng.init(0x11A6E6);
    const rnd = prng.random();
    var lines: usize = 0;
    var bad: usize = 0;
    var clamped_seen = false;
    for (cases) |c| {
        const cm = g.mark();
        defer g.resetTo(cm);
        defer _ = mlx.mlx_clear_cache();
        const gate = try randProj(a, &g, s, rnd, hidden, c.out, c.n, 8);
        const up = try randProj(a, &g, s, rnd, hidden, c.out, c.n, 8);
        for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
            const tabs = [2]A{ try g.adopt(try shared.sigmoidTable(s, dt, false)), try g.adopt(try shared.sigmoidTable(s, dt, true)) };
            for (c.lims) |lim| for (test_rows) |rows| {
                if (dt == .float16 and rows != 33 and rows != 4096) continue;
                const rm = g.mark();
                defer g.resetTo(rm);
                const x = try randX(a, &g, rnd, rows, dt);
                const tab = tabs[@intFromBool(lim != 0)];
                const gv = try g.adopt(try root.linear(s, x, gate.trellis, gate.suh, gate.svh, .mul1));
                const uv = try g.adopt(try root.linear(s, x, up.trellis, up.suh, up.svh, .mul1));
                const want = try hostSwiglu(&g, gv, uv, lim);
                const got = try g.adopt((try swiglu.upSwiglu(s, x, up, gv, lim, tab, .mul1)) orelse return error.TestUnexpectedResult);
                try testing.expectEqualSlices(c_int, &.{ 1, ci(rows), ci(c.out) }, mlx.getShape(got));
                const nel: usize = @as(usize, rows) * c.out;
                const d = try wordsDiffer(a, &g, got, want, nel);
                if (lim != 0 and rows == 4096) {
                    const over = try overLimit(a, &g, gv, uv, nel, lim);
                    std.debug.print("TURBOEXL3_LING_SWIGLU_EPI_CLAMP {s} rows={d} x={s} silu_over={d}/{d} ups_over={d}/{d}\n", .{ c.name, rows, @tagName(dt), over[0], nel, over[1], nel });
                    if (over[0] > 0 and over[1] > 0) clamped_seen = true;
                }
                const dn = root.denseCached(.{ .in = hidden, .out = c.out, .rate = format.Rate.fromN(c.n).?, .dec = .mul1, .out_dtype = dt }) catch unreachable;
                lines += 1;
                bad += @intFromBool(d != 0);
                std.debug.print("TURBOEXL3_LING_SWIGLU_EPI {s} {d}x{d} n={d} rows={d} x={s} limit={d} path={s} differing={d}/{d} {s}\n", .{ c.name, hidden, c.out, c.n, rows, @tagName(dt), lim, @tagName(dn.pathFor(rows)), d, nel, if (d == 0) "PASS" else "FAIL" });
            };
        }
        // the declines (null, no launch): the decode GEMVs' rows, an f32 x
        const tab0 = try g.adopt(try shared.sigmoidTable(s, .bfloat16, false));
        for ([_]u32{ 1, 4, 16 }) |rows| {
            const x = try randX(a, &g, rnd, rows, .bfloat16);
            const gv = try g.adopt(try root.linear(s, x, gate.trellis, gate.suh, gate.svh, .mul1));
            try testing.expect((try swiglu.upSwiglu(s, x, up, gv, 0, tab0, .mul1)) == null);
        }
        const x33 = try randX(a, &g, rnd, 33, .bfloat16);
        const x33f = try g.astype(x33, .float32);
        const g33 = try g.adopt(try root.linear(s, x33, gate.trellis, gate.suh, gate.svh, .mul1));
        try testing.expect((try swiglu.upSwiglu(s, x33f, up, try g.astype(g33, .float32), 0, tab0, .mul1)) == null);
        // the refusals, before any launch: a clamp above 128, the table of the other limit, a gate of the wrong size
        try testing.expectError(error.InvalidExl3Clamp, swiglu.upSwiglu(s, x33, up, g33, 129, tab0, .mul1));
        const tab32 = try g.adopt(try shared.sigmoidTable(s, .bfloat16, true));
        try testing.expectError(error.BadExl3Dtype, swiglu.upSwiglu(s, x33, up, g33, 0, tab32, .mul1));
        try testing.expectError(error.BadExl3Dtype, swiglu.upSwiglu(s, x33, up, g33, 10, tab0, .mul1));
        const short = try g.slice(g33, &.{ 0, 0, 0 }, &.{ 1, 32, ci(c.out) }, &.{ 1, 1, 1 });
        try testing.expectError(error.BadExl3Shape, swiglu.upSwiglu(s, x33, up, short, 0, tab0, .mul1));
    }
    std.debug.print("TURBOEXL3_LING_SWIGLU_EPI_DONE lines={d} differing_lines={d} clamp_reached={} {s}\n", .{ lines, bad, clamped_seen, if (bad == 0 and clamped_seen) "BYTES_OK" else "BYTES_DIFFER" });
    try testing.expect(clamped_seen);
    try testing.expectEqual(@as(usize, 0), bad);
}

// ── timing (TURBOEXL3_LING_TIME=1): guidance for the serving plan, never a speed verdict ──

test "turboexl3 gpu ling swiglu timing: up's linear and the host's SwiGLU against upSwiglu at the shared expert's and the dense layers' shapes, rows 4096 and 16384 (TURBOEXL3_LING_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_LING_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x11A6E7);
    const rnd = prng.random();
    const Arm = enum { up_linear, up_and_one_launch, up_and_clamp_ops, epi };
    for (cases) |c| {
        const cm = g.mark();
        defer g.resetTo(cm);
        defer _ = mlx.mlx_clear_cache();
        const gate = try randProj(a, &g, s, rnd, hidden, c.out, c.n, 8);
        const up = try randProj(a, &g, s, rnd, hidden, c.out, c.n, 8);
        const tabs = [2]A{ try g.adopt(try shared.sigmoidTable(s, .bfloat16, false)), try g.adopt(try shared.sigmoidTable(s, .bfloat16, true)) };
        for ([_]u32{ 4096, 16384 }) |rows| for (c.lims) |lim| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const x = try randX(a, &g, rnd, rows, .bfloat16);
            const gv = try g.adopt(try root.linear(s, x, gate.trellis, gate.suh, gate.svh, .mul1));
            try g.evalAll(&.{ x, gv });
            var us: [4]f64 = @splat(0);
            inline for (.{ Arm.up_linear, Arm.up_and_one_launch, Arm.up_and_clamp_ops, Arm.epi }) |arm| {
                if (arm != .up_and_clamp_ops or lim != 0) {
                    const Ctx = struct {
                        s: mlx.mlx_stream,
                        x: A,
                        gv: A,
                        up: root.Proj,
                        lim: u32,
                        tab: A,
                        g: *G,
                        pub fn call(cx: @This()) !A {
                            switch (arm) {
                                .up_linear => return root.linear(cx.s, cx.x, cx.up.trellis, cx.up.suh, cx.up.svh, .mul1),
                                .up_and_one_launch => {
                                    const uv = try root.linear(cx.s, cx.x, cx.up.trellis, cx.up.suh, cx.up.svh, .mul1);
                                    defer _ = mlx.mlx_array_free(uv);
                                    var r = mlx.mlx_array_new();
                                    try mlx.check(mlx.mlx_multiply(&r, cx.gv, uv, cx.s));
                                    return r;
                                },
                                .up_and_clamp_ops => {
                                    const m = cx.g.mark();
                                    defer cx.g.resetTo(m);
                                    const uv = try cx.g.adopt(try root.linear(cx.s, cx.x, cx.up.trellis, cx.up.suh, cx.up.svh, .mul1));
                                    const h = try hostSwiglu(cx.g, cx.gv, uv, cx.lim);
                                    return cx.g.keep(h);
                                },
                                .epi => return (try swiglu.upSwiglu(cx.s, cx.x, cx.up, cx.gv, cx.lim, cx.tab, .mul1)) orelse return error.TestUnexpectedResult,
                            }
                        }
                    };
                    const ctx: Ctx = .{ .s = s, .x = x, .gv = gv, .up = up, .lim = lim, .tab = tabs[@intFromBool(lim != 0)], .g = &g };
                    const st = try ds_test.timeIt(&g, ctx, 2);
                    us[@intFromEnum(arm)] = st.min;
                    std.debug.print("TURBOEXL3_LING_TIME swiglu_epi {s} {d}x{d} n={d} rows={d} limit={d} arm={s} min_us={d:.1} med_us={d:.1}\n", .{ c.name, hidden, c.out, c.n, rows, lim, @tagName(arm), st.min, st.med });
                }
            }
            const chain = if (lim == 0) us[1] else us[2];
            std.debug.print("TURBOEXL3_LING_TIME swiglu_epi {s} rows={d} limit={d} chain_min_us={d:.1} epi_min_us={d:.1} saved_us={d:.1}\n", .{ c.name, rows, lim, chain, us[3], chain - us[3] });
        };
    }
}
