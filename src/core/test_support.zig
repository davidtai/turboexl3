//! Test-only support for turboexl3's tests: `Graph`, a scope of tracked MLX arrays with the graph ops the tests build
//! their inputs and host chains from (each the mlx-c call of the same name, nothing of its own), `bf16Bits` (round to
//! nearest even, as MLX rounds an f32 to bf16) and Sushi's EXL3 module, the second reference the expert tests compare
//! against where the host exports it (`mlx_host.sushi_exl3`; `has_sushi` false skips those comparisons). The tests' only
//! reach to the host besides `sdk`; nothing but the tests imports this file.
const std = @import("std");
const mlx = @import("sdk").mlx;
const host = @import("mlx_host");

pub const has_sushi = @hasDecl(host, "sushi_exl3");
pub const sushi = if (has_sushi) host.sushi_exl3 else struct {};

/// f32 to bf16 bits, round to nearest even (a NaN stays a quiet NaN).
pub fn bf16Bits(f: f32) u16 {
    const u: u32 = @bitCast(f);
    if (std.math.isNan(f)) return @intCast((u >> 16) | 0x40);
    const rounding: u32 = 0x7FFF + ((u >> 16) & 1);
    return @intCast((u +% rounding) >> 16);
}

/// A test's pin file (src/kernels/test/pins/*.metal): the Metal statements the test searches the kernel texts for, one per
/// line, exactly as it searches for them (a final newline ends the last line).
pub fn pins(comptime text: []const u8) []const []const u8 {
    comptime {
        @setEvalBranchQuota(1_000_000);
        const body = if (text.len > 0 and text[text.len - 1] == '\n') text[0 .. text.len - 1] else text;
        var out: [std.mem.count(u8, body, "\n") + 1][]const u8 = undefined;
        var it = std.mem.splitScalar(u8, body, '\n');
        for (&out) |*o| o.* = it.next().?;
        const final = out;
        return &final;
    }
}

/// A point in a graph's scope (`mark`): `resetTo` frees every array tracked after it.
pub const Mark = struct { n: usize };

/// MLX arrays built on one stream, each tracked until `resetTo` / `deinit` frees it; `keep` makes a handle that outlives
/// the scope (freed by `release`).
pub const Graph = struct {
    pub const T = mlx.mlx_array;

    gpa: std.mem.Allocator,
    s: mlx.mlx_stream,
    live: std.ArrayList(mlx.mlx_array) = .empty,

    pub fn init(gpa: std.mem.Allocator, s: mlx.mlx_stream) !Graph {
        return .{ .gpa = gpa, .s = s };
    }

    pub fn deinit(g: *Graph) void {
        g.resetTo(.{ .n = 0 });
        g.live.deinit(g.gpa);
    }

    pub fn mark(g: *const Graph) Mark {
        return .{ .n = g.live.items.len };
    }

    pub fn resetTo(g: *Graph, m: Mark) void {
        std.debug.assert(m.n <= g.live.items.len);
        for (g.live.items[m.n..]) |a| _ = mlx.mlx_array_free(a);
        g.live.shrinkRetainingCapacity(m.n);
    }

    pub fn keep(_: *Graph, x: T) T {
        var out = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&out, x);
        return out;
    }

    pub fn release(_: *Graph, x: T) void {
        _ = mlx.mlx_array_free(x);
    }

    /// Take ownership of an array built outside the graph (freed with the scope).
    pub fn adopt(g: *Graph, x: T) !T {
        return g.track(x);
    }

    /// A copy of host bytes as an array.
    pub fn hostArray(g: *Graph, bytes: []const u8, shape: []const c_int, dt: mlx.mlx_dtype) !T {
        const a = mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), dt);
        if (a.ctx == null) return error.MlxError;
        return g.track(a);
    }

    pub fn evalAll(_: *Graph, xs: []const T) !void {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
    }

    /// `x` evaluated and its values copied row-major into `out` (x's size; a view is read through its strides).
    pub fn hostF32(_: *Graph, x: T, out: []f32) ![]const f32 {
        try mlx.check(mlx.mlx_array_eval(x));
        if (mlx.mlx_array_size(x) != out.len) return error.HostReadSize;
        std.debug.assert(mlx.mlx_array_dtype(x) == .float32);
        return readInPlace(f32, x, mlx.mlx_array_data_float32(x), out);
    }

    pub fn hostU32(_: *Graph, x: T, out: []u32) ![]const u32 {
        try mlx.check(mlx.mlx_array_eval(x));
        if (mlx.mlx_array_size(x) != out.len) return error.HostReadSize;
        std.debug.assert(mlx.mlx_array_dtype(x) == .uint32);
        return readInPlace(u32, x, mlx.mlx_array_data_uint32(x), out);
    }

    fn readInPlace(comptime E: type, x: T, data: ?[*]const E, out: []E) ![]const E {
        const src = data orelse return error.MlxNoData;
        const nd = mlx.mlx_array_ndim(x);
        const shape = mlx.mlx_array_shape(x)[0..nd];
        const strides = mlx.mlx_array_strides(x)[0..nd];
        var row_major = true;
        var n: usize = 1;
        var d = nd;
        while (d > 0) {
            d -= 1;
            if (shape[d] != 1 and strides[d] != n) row_major = false;
            n *= @intCast(shape[d]);
        }
        if (row_major) {
            @memcpy(out, src[0..out.len]);
            return out;
        }
        var idx: [8]usize = @splat(0);
        std.debug.assert(nd <= idx.len);
        for (out) |*o| {
            var off: usize = 0;
            for (idx[0..nd], strides) |i, st| off += i * st;
            o.* = src[off];
            var k = nd;
            while (k > 0) {
                k -= 1;
                idx[k] += 1;
                if (idx[k] < @as(usize, @intCast(shape[k]))) break;
                idx[k] = 0;
            }
        }
        return out;
    }

    fn track(g: *Graph, a: T) !T {
        g.live.append(g.gpa, a) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        return a;
    }

    /// A 0-d constant of `dt` holding `v` rounded as MLX rounds a Python scalar (f64 -> f32 -> dt).
    pub fn scalar(g: *Graph, v: f64, dt: mlx.mlx_dtype) !T {
        const shape = [_]c_int{};
        const a = switch (dt) {
            .float32 => mlx.mlx_array_new_float(@floatCast(v)),
            .bool_ => mlx.mlx_array_new_bool(v != 0),
            .int32 => mlx.mlx_array_new_int(@intFromFloat(v)),
            .bfloat16 => blk: {
                const bits = bf16Bits(@floatCast(v));
                break :blk mlx.mlx_array_new_data(&bits, &shape, 0, .bfloat16);
            },
            .float16 => blk: {
                const h: f16 = @floatCast(@as(f32, @floatCast(v)));
                break :blk mlx.mlx_array_new_data(&h, &shape, 0, .float16);
            },
            else => return error.UnsupportedScalar,
        };
        if (a.ctx == null) return error.MlxError;
        return g.track(a);
    }

    pub fn arange(g: *Graph, start: f64, stop: f64, step: f64, dt: mlx.mlx_dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_arange(&r, start, stop, step, dt, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn zeros(g: *Graph, shape: []const c_int, dt: mlx.mlx_dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_zeros(&r, shape.ptr, shape.len, dt, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn astype(g: *Graph, x: T, dt: mlx.mlx_dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_astype(&r, x, dt, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    fn op1(g: *Graph, comptime f: anytype, x: T) !T {
        var r = mlx.mlx_array_new();
        mlx.check(f(&r, x, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    fn op2(g: *Graph, comptime f: anytype, a: T, b: T) !T {
        var r = mlx.mlx_array_new();
        mlx.check(f(&r, a, b, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn add(g: *Graph, a: T, b: T) !T {
        return g.op2(mlx.mlx_add, a, b);
    }
    pub fn sub(g: *Graph, a: T, b: T) !T {
        return g.op2(mlx.mlx_subtract, a, b);
    }
    pub fn mul(g: *Graph, a: T, b: T) !T {
        return g.op2(mlx.mlx_multiply, a, b);
    }
    pub fn matmul(g: *Graph, a: T, b: T) !T {
        return g.op2(mlx.mlx_matmul, a, b);
    }
    pub fn less(g: *Graph, a: T, b: T) !T {
        return g.op2(mlx.mlx_less, a, b);
    }
    pub fn greater(g: *Graph, a: T, b: T) !T {
        return g.op2(mlx.mlx_greater, a, b);
    }
    pub fn logicalAnd(g: *Graph, a: T, b: T) !T {
        return g.op2(mlx.mlx_logical_and, a, b);
    }
    pub fn neg(g: *Graph, x: T) !T {
        return g.op1(mlx.mlx_negative, x);
    }
    pub fn abs(g: *Graph, x: T) !T {
        return g.op1(mlx.mlx_abs, x);
    }
    pub fn sigmoid(g: *Graph, x: T) !T {
        return g.op1(mlx.mlx_sigmoid, x);
    }

    pub fn where(g: *Graph, c: T, x: T, y: T) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_where(&r, c, x, y, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn mean(g: *Graph, x: T, axis: c_int, keepdims: bool) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_mean_axis(&r, x, axis, keepdims, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn reshape(g: *Graph, x: T, shape: []const c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_reshape(&r, x, shape.ptr, shape.len, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn transposeAxes(g: *Graph, x: T, axes: []const c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_transpose_axes(&r, x, axes.ptr, axes.len, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn slice(g: *Graph, x: T, start: []const c_int, stop: []const c_int, strides: []const c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_slice(&r, x, start.ptr, start.len, stop.ptr, stop.len, strides.ptr, strides.len, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn take(g: *Graph, x: T, idx: T, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_take_axis(&r, x, idx, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn concat(g: *Graph, xs: []const T, axis: c_int) !T {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_concatenate_axis(&r, vec, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }
};
