//! `root.ling.addNormLinears` on the device: the residual add, its RMSNorm and up to three EXL3 linears' prepares in one
//! launch (addnorm.zig) against the composed ops it replaces at a decode layer's mixer input: MLX's add, MLX's fast
//! rms_norm, then each member's `root.linear`, word for word, at decode rows 1 to 8, 12 and 16, in bf16 and f16, with each
//! member's own `linear` taking its prepare inside its GEMV launch or in a launch of its own (`dense.Dense.dec_prep_rows`).
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const addnorm = @import("addnorm.zig");
const dense = @import("../../core/dense.zig");

const testing = std.testing;

const hidden: u32 = addnorm.width;
const n_k6: u32 = 96;
const eps: f32 = 1e-6;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// `v` rounded to bf16 (as MLX rounds), back in f32: a value both sides read alike.
fn bfExact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
}

/// Ling's mixer inputs: a KDA layer's qkv, f and g (the plugin's order), an MLA layer's q_a and kv_a, and one member
const Case = struct { name: []const u8, outs: []const u32 };
const cases = [_]Case{
    .{ .name = "kda.qkv+f+g", .outs = &.{ 24576, 8192, 8192 } },
    .{ .name = "mla.q_a+kv_a", .outs = &.{ 1536, 640 } },
    .{ .name = "shared.gate", .outs = &.{1280} },
};
const rows_list = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16 };

fn synthProj(a: std.mem.Allocator, g: *G, rnd: std.Random, in: u32, out: u32, n: u32) !struct { lin: ref.Linear, p: root.Proj } {
    var lin = try ref.synthLinear(a, rnd, 1, in, out, format.Rate.fromN(n).?, .mul1);
    errdefer lin.deinit(a);
    const p: root.Proj = .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(out)}, .float16),
    };
    return .{ .lin = lin, .p = p };
}

/// Words of `x` (any float dtype) that differ from `want`'s, read through f32 (exact for bf16 and f16)
fn differing(g: *G, x: mlx.mlx_array, want: mlx.mlx_array, a: std.mem.Allocator) !usize {
    const n: usize = @intCast(mlx.mlx_array_size(x));
    if (n != @as(usize, @intCast(mlx.mlx_array_size(want)))) return n;
    const u = try a.alloc(f32, n);
    defer a.free(u);
    const v = try a.alloc(f32, n);
    defer a.free(v);
    _ = try g.hostF32(try g.astype(x, .float32), u);
    _ = try g.hostF32(try g.astype(want, .float32), v);
    var d: usize = 0;
    for (u, v) |p, q| d += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
    return d;
}

test "turboexl3 gpu ling addnorm: the add, its RMSNorm and up to three linears' prepares in one launch give MLX's add, MLX's fast rms_norm and each linear's bytes at decode rows 1 to 8, 12 and 16, each linear's prepare inside or in its own launch (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var failed: usize = 0;
    var checks: usize = 0;
    var forms = [2]usize{ 0, 0 };
    const max_rows: u32 = 16;
    for (cases) |c| {
        var prng = std.Random.DefaultPrng.init(0xE3 ^ (@as(u64, c.outs[0]) << 20) ^ c.outs.len);
        const rnd = prng.random();
        const mark = g.mark();
        defer g.resetTo(mark);
        var lins: [addnorm.max_members]ref.Linear = undefined;
        var members: [addnorm.max_members]root.Proj = undefined;
        for (c.outs, 0..) |o, i| {
            const sp = try synthProj(a, &g, rnd, hidden, o, n_k6);
            lins[i] = sp.lin;
            members[i] = sp.p;
        }
        defer for (lins[0..c.outs.len]) |*l| l.deinit(a);
        const hs = try a.alloc(f32, max_rows * hidden);
        defer a.free(hs);
        const ds = try a.alloc(f32, max_rows * hidden);
        defer a.free(ds);
        const ws = try a.alloc(f32, hidden);
        defer a.free(ws);
        for (hs) |*v| v.* = bfExact(4 * rnd.floatNorm(f32));
        for (ds) |*v| v.* = bfExact(rnd.floatNorm(f32));
        for (ws) |*v| v.* = bfExact(1 + 0.25 * rnd.floatNorm(f32));
        for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
            // each member's cached Dense decides its own `linear`'s form; the wide arm gives one row two rows' launch
            // shape with the prepare in its own launch (`Config.dec_one_row_wide`'s fields)
            var dens: [addnorm.max_members]*dense.Dense = undefined;
            var saved_prep: [addnorm.max_members]u32 = undefined;
            var saved_bt: [addnorm.max_members]u8 = undefined;
            for (members[0..c.outs.len], 0..) |m, i| {
                dens[i] = try root.denseOf(m, hidden, dt, .mul1);
                saved_prep[i] = dens[i].dec_prep_rows;
                saved_bt[i] = dens[i].bt_set[1];
            }
            defer for (dens[0..c.outs.len], 0..) |d, i| {
                d.dec_prep_rows = saved_prep[i];
                d.bt_set[1] = saved_bt[i];
                d.resetCfgs();
            };
            for ([_]bool{ false, true }) |wide| {
                for (dens[0..c.outs.len]) |d| {
                    d.dec_prep_rows = if (wide) 0 else 1;
                    d.bt_set[1] = if (wide) 2 else 0;
                    d.resetCfgs();
                }
                for (rows_list) |rows| {
                    const rm = g.mark();
                    defer g.resetTo(rm);
                    const h = try g.astype(try g.hostArray(std.mem.sliceAsBytes(hs[0 .. rows * hidden]), &.{ 1, ci(rows), ci(hidden) }, .float32), dt);
                    const delta = try g.astype(try g.hostArray(std.mem.sliceAsBytes(ds[0 .. rows * hidden]), &.{ 1, ci(rows), ci(hidden) }, .float32), dt);
                    const w = try g.astype(try g.hostArray(std.mem.sliceAsBytes(ws), &.{ci(hidden)}, .float32), dt);
                    const eps_arr = try g.scalar(eps, .float32);
                    // the composed ops: MLX's add, MLX's fast rms_norm, each member's linear
                    const h_ref = try g.add(h, delta);
                    var n_ref = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_fast_rms_norm(&n_ref, h_ref, w, eps, s));
                    n_ref = try g.adopt(n_ref);
                    var y_ref: [addnorm.max_members]mlx.mlx_array = undefined;
                    for (members[0..c.outs.len], 0..) |m, i| y_ref[i] = try g.adopt(try root.linear(s, n_ref, m.trellis, m.suh, m.svh, .mul1));
                    var got = try root.ling.addNormLinears(s, h, delta, w, eps_arr, members[0..c.outs.len], .mul1);
                    defer got.deinit();
                    const dh = try differing(&g, got.h, h_ref, a);
                    const dn = try differing(&g, got.normed, n_ref, a);
                    var dy: [addnorm.max_members]usize = @splat(0);
                    var shapes_ok = std.mem.eql(c_int, mlx.getShape(got.h), mlx.getShape(h)) and std.mem.eql(c_int, mlx.getShape(got.normed), mlx.getShape(h));
                    for (0..c.outs.len) |i| {
                        dy[i] = try differing(&g, got.ys[i], y_ref[i], a);
                        shapes_ok = shapes_ok and std.mem.eql(c_int, mlx.getShape(got.ys[i]), mlx.getShape(y_ref[i]));
                    }
                    const inside = try dens[0].decPrepInside(rows);
                    forms[@intFromBool(inside)] += 1;
                    var ok = shapes_ok and dh == 0 and dn == 0;
                    for (dy[0..c.outs.len]) |d| ok = ok and d == 0;
                    checks += 1;
                    std.debug.print("TURBOEXL3_LING_ADDNORM {s} x={s} wide={} rows={d} member0_own_linear={s} differing h={d} normed={d} y={d},{d},{d} of {d} rows {s}\n", .{ c.name, @tagName(dt), wide, rows, if (inside) "inside" else "own_launch", dh, dn, dy[0], dy[1], dy[2], rows, if (ok) "PASS" else "FAIL" });
                    if (!ok) failed += 1;
                }
            }
        }
    }
    std.debug.print("TURBOEXL3_LING_ADDNORM_DONE checks={d} inside={d} own_launch={d} {s}\n", .{ checks, forms[1], forms[0], if (failed == 0) "PASS" else "FAIL" });
    try testing.expect(forms[0] > 0 and forms[1] > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}

test "turboexl3 gpu ling addnorm: refusals come before any launch: a width other than 4096, mixed dtypes, a non-f32 or non-scalar eps, a suh of another width (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const h = try g.zeros(&.{ 1, 1, ci(hidden) }, .bfloat16);
    const w = try g.zeros(&.{ci(hidden)}, .bfloat16);
    const eps_arr = try g.scalar(eps, .float32);
    const suh = try g.zeros(&.{ci(hidden)}, .float16);
    const narrow = try g.zeros(&.{ 1, 1, 2048 }, .bfloat16);
    try testing.expectError(error.BadExl3Shape, addnorm.addNormPrepare(s, narrow, narrow, w, eps_arr, &.{suh}, 1, 8));
    try testing.expectError(error.BadExl3Dtype, addnorm.addNormPrepare(s, h, try g.zeros(&.{ 1, 1, ci(hidden) }, .float16), w, eps_arr, &.{suh}, 1, 8));
    try testing.expectError(error.BadExl3Dtype, addnorm.addNormPrepare(s, h, h, w, try g.scalar(eps, .bfloat16), &.{suh}, 1, 8));
    try testing.expectError(error.BadExl3Shape, addnorm.addNormPrepare(s, h, h, w, eps_arr, &.{try g.zeros(&.{2048}, .float16)}, 1, 8));
    try testing.expectError(error.BadExl3Shape, addnorm.addNormPrepare(s, h, h, w, eps_arr, &.{suh}, 1, 4));
}
