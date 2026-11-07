//! The shared expert's gate | up in one DEC-ONE launch (`shared_gu.gateUp`, the gate's and the up's outputs) against
//! `root.linear` of each member, word for word (TURBOEXL3_GPU=1, else skipped): GLM-5.3-Flash's shared expert (4096 -> 2048 twice) at the trunk's K6 and the
//! MTP layer's K4 on random codewords, suh and svh, decode rows 1 to 8 and 16, bf16 x (the served activations) and f32.
//! A row count it does not serve (17) returns error.Exl3EngineNotImplemented. Lines: TURBOEXL3_GLM_SHARED_GU_BYTES per
//! rate, rows and dtype, and a total.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const A = mlx.mlx_array;
const format = @import("../../core/format.zig");
const root = @import("../../root.zig");
const sgu = @import("shared_gu.zig");

const testing = std.testing;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn bf16Exact(v: f32) f32 {
    return @bitCast(@as(u32, @bitCast(v)) & 0xffff0000);
}

/// A projection's random operands on the device: codewords, suh in [0.5, 1.5], svh [0.5, 1.5] / sqrt(in)
fn randProj(a: std.mem.Allocator, g: *G, rnd: std.Random, in: u32, out: u32, n: u32) !root.Proj {
    const t = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(t);
    rnd.bytes(std.mem.sliceAsBytes(t));
    const su = try a.alloc(u16, in);
    defer a.free(su);
    for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, out);
    defer a.free(sv);
    for (sv) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    return .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(t), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(out)}, .float16),
    };
}

fn randX(a: std.mem.Allocator, g: *G, rnd: std.Random, rows: u32, in: u32, dt: mlx.mlx_dtype) !A {
    const xh = try a.alloc(f32, @as(usize, rows) * in);
    defer a.free(xh);
    for (xh) |*v| v.* = bf16Exact(rnd.floatNorm(f32));
    const x = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(in) }, .float32);
    return if (dt == .float32) x else g.astype(x, dt);
}

/// the words of `x` and `y` (any float dtype, read as f32: exact for bf16 and f16) that differ, of `n`
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

test "turboexl3 gpu glm53f shared gu: the shared expert's gate | up in one launch gives each member's linear words at the trunk's K6 and the MTP layer's K4, decode rows 1 to 8 and 16, bf16 and f32 (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x5A1E);
    const rnd = prng.random();
    const in: u32 = 4096;
    const out: u32 = 2048;
    var lines: usize = 0;
    var fails: usize = 0;
    for ([_]u32{ 6, 4 }) |k| {
        const n = 16 * k;
        const gate = try randProj(a, &g, rnd, in, out, n);
        const up = try randProj(a, &g, rnd, in, out, n);
        const svh = try g.adopt(try sgu.joinSvh(s, gate.svh, up.svh));
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
            for ([_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 16 }) |rows| {
                const rm = g.mark();
                defer g.resetTo(rm);
                const x = try randX(a, &g, rnd, rows, in, dt);
                const gu = try sgu.gateUp(s, x, gate, up, svh, .mul1);
                const yg = try g.adopt(gu[0]);
                const yu = try g.adopt(gu[1]);
                const y0 = try g.adopt(try root.linear(s, x, gate.trellis, gate.suh, gate.svh, .mul1));
                const y1 = try g.adopt(try root.linear(s, x, up.trellis, up.suh, up.svh, .mul1));
                const dtype_ok = mlx.mlx_array_dtype(yg) == dt and mlx.mlx_array_dtype(yu) == dt;
                const d0 = try wordsDiffer(a, &g, yg, y0, rows * out);
                const d1 = try wordsDiffer(a, &g, yu, y1, rows * out);
                const ok = dtype_ok and d0 + d1 == 0;
                lines += 1;
                fails += @intFromBool(!ok);
                std.debug.print("TURBOEXL3_GLM_SHARED_GU_BYTES k={d} rows={d} x={s} gate_differing={d}/{d} up_differing={d}/{d} dtype_ok={} {s}\n", .{ k, rows, @tagName(dt), d0, rows * out, d1, rows * out, dtype_ok, if (ok) "PASS" else "FAIL" });
            }
            // a row count it does not serve: declined before any launch
            const rm = g.mark();
            defer g.resetTo(rm);
            const x = try randX(a, &g, rnd, 17, in, dt);
            try testing.expectError(error.Exl3EngineNotImplemented, sgu.gateUp(s, x, gate, up, svh, .mul1));
        }
    }
    std.debug.print("TURBOEXL3_GLM_SHARED_GU_BYTES lines={d} fails={d} {s}\n", .{ lines, fails, if (fails == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 0), fails);
}
