//! The host reference of one routed MoE layer and the synthetic banks the GPU tests and benches use. The reference is
//! Sushi's `moeSwigluHost` over this library's `format.project`: gate and up through the full projection (f16 rounding at
//! the input, the inner product and the output), SwiGLU in f32, the down projection on that f32 h (its prepare rounds h
//! to f16), and the score-weighted sum over the slots in slot order. Every kernel is held to this within a tolerance;
//! the clamped SwiGLU's (`root.moeClamped`) to the same with gate and up clamped first (`moeRowClamped`).
const std = @import("std");
const format = @import("format.zig");

pub const Bank = struct {
    experts: usize,
    hidden: usize,
    inter: usize,
    rate: format.Rate,
    gate_t: []u16,
    gate_suh: []u16,
    gate_svh: []u16,
    up_t: []u16,
    up_suh: []u16,
    up_svh: []u16,
    down_t: []u16,
    down_suh: []u16,
    down_svh: []u16,

    pub fn deinit(self: *Bank, a: std.mem.Allocator) void {
        a.free(self.gate_t);
        a.free(self.gate_suh);
        a.free(self.gate_svh);
        a.free(self.up_t);
        a.free(self.up_suh);
        a.free(self.up_svh);
        a.free(self.down_t);
        a.free(self.down_suh);
        a.free(self.down_svh);
    }

    /// halfwords of one expert's gate or up trellis
    pub fn guStride(self: Bank) usize {
        return (self.hidden / 16) * (self.inter / 16) * self.rate.n;
    }
};

/// Random trellis halfwords (every 16-bit pattern is a valid codeword), suh in [0.5, 1.5] and svh in [0.5, 1.5] times a
/// scale that keeps the outputs O(1): the decoded weights are O(1), a 128-rotated input is O(1), so an inner product over
/// `in` terms is O(sqrt(in)).
pub fn synthBank(a: std.mem.Allocator, rnd: std.Random, experts: usize, hidden: usize, inter: usize, rate: format.Rate) !Bank {
    const n = rate.n;
    const gu = experts * (hidden / 16) * (inter / 16) * n;
    const b: Bank = .{
        .experts = experts,
        .hidden = hidden,
        .inter = inter,
        .rate = rate,
        .gate_t = try a.alloc(u16, gu),
        .gate_suh = try a.alloc(u16, experts * hidden),
        .gate_svh = try a.alloc(u16, experts * inter),
        .up_t = try a.alloc(u16, gu),
        .up_suh = try a.alloc(u16, experts * hidden),
        .up_svh = try a.alloc(u16, experts * inter),
        .down_t = try a.alloc(u16, gu),
        .down_suh = try a.alloc(u16, experts * inter),
        .down_svh = try a.alloc(u16, experts * hidden),
    };
    for (b.gate_t) |*w| w.* = rnd.int(u16);
    for (b.up_t) |*w| w.* = rnd.int(u16);
    for (b.down_t) |*w| w.* = rnd.int(u16);
    const out_scale_gu: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden)));
    const out_scale_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(inter)));
    for (b.gate_suh) |*s| s.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    for (b.up_suh) |*s| s.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    for (b.down_suh) |*s| s.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    for (b.gate_svh) |*s| s.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * out_scale_gu);
    for (b.up_svh) |*s| s.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * out_scale_gu);
    for (b.down_svh) |*s| s.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * out_scale_d);
    return b;
}

/// y [hidden] for one row x [hidden] over its slots (expert ids) and scores, in slot order.
pub fn moeRow(a: std.mem.Allocator, b: *const Bank, x: []const f32, slots: []const u32, scores: []const f32, y: []f32) !void {
    const hidden = b.hidden;
    const inter = b.inter;
    @memset(y, 0);
    const transformed = try a.alloc(f32, hidden);
    defer a.free(transformed);
    const inner = try a.alloc(f32, @max(hidden, inter));
    defer a.free(inner);
    const gate_y = try a.alloc(f32, inter);
    defer a.free(gate_y);
    const up_y = try a.alloc(f32, inter);
    defer a.free(up_y);
    const h = try a.alloc(f32, inter);
    defer a.free(h);
    const down_y = try a.alloc(f32, hidden);
    defer a.free(down_y);
    const stride = b.guStride();
    for (slots, scores) |e_u, w| {
        const e: usize = e_u;
        const off = e * stride;
        format.project(x, b.gate_t[off..][0..stride], b.gate_suh[e * hidden ..][0..hidden], b.gate_svh[e * inter ..][0..inter], hidden, inter, b.rate, .{ .codebook = .mul1 }, transformed, inner[0..inter], gate_y);
        _ = &gate_y;
        format.project(x, b.up_t[off..][0..stride], b.up_suh[e * hidden ..][0..hidden], b.up_svh[e * inter ..][0..inter], hidden, inter, b.rate, .{ .codebook = .mul1 }, transformed, inner[0..inter], up_y);
        for (0..inter) |i| {
            const g = gate_y[i];
            h[i] = (g / (1.0 + @exp(-g))) * up_y[i];
        }
        format.project(h, b.down_t[off..][0..stride], b.down_suh[e * inter ..][0..inter], b.down_svh[e * hidden ..][0..hidden], inter, hidden, b.rate, .{ .codebook = .mul1 }, inner[0..inter], transformed, down_y);
        for (0..hidden) |i| y[i] += w * down_y[i];
    }
}

/// `moeRow` under a decode (codebook, window) other than MUL1 w16.
pub fn moeRowDec(a: std.mem.Allocator, b: *const Bank, dec: format.Decode, x: []const f32, slots: []const u32, scores: []const f32, y: []f32) !void {
    _ = try moeRowClamped(a, b, dec, 0, x, slots, scores, y);
}

/// The elements `moeRowClamped` clamped over a row's slots: gates above the limit and ups outside [-limit, limit], of
/// `total` each (the slots times the intermediate width).
pub const Hits = struct { gate: usize = 0, up: usize = 0, total: usize = 0 };

/// One element's SwiGLU in f32 with DeepSeek-V4's and GLM's clamp (`limit` > 0; 0 none): the gate clamped from above only,
/// up to [-limit, limit], then silu(gate) * up. Comparisons, so a NaN passes through as torch.clamp passes it (the Metal
/// clamp's rule).
pub fn swigluClamped(g: f32, u: f32, limit: f32) f32 {
    var gc = g;
    var uc = u;
    if (limit > 0) {
        gc = if (g > limit) limit else g;
        uc = if (u > limit) limit else if (u < -limit) -limit else u;
    }
    return (gc / (1.0 + @exp(-gc))) * uc;
}

/// `moeRowDec` with the clamped SwiGLU (`root.moeClamped`'s reference; DeepSeek-V4.1's Expert at swiglu_limit `limit`):
/// gate and up through the full projection, both clamped (`swigluClamped`) on their f16-rounded values, SwiGLU in f32, the
/// down projection and the score-weighted sum in slot order. The reference model scales the f32 activation by the slot's
/// weight before w2 and rounds it to x's dtype; here, as in the kernels, the weight scales the down projection's
/// f16-rounded output (the same product, the down projection being linear; the rounding points differ). `limit` 0 is
/// `moeRowDec`. Returns what it clamped.
pub fn moeRowClamped(a: std.mem.Allocator, b: *const Bank, dec: format.Decode, limit: f32, x: []const f32, slots: []const u32, scores: []const f32, y: []f32) !Hits {
    const hidden = b.hidden;
    const inter = b.inter;
    @memset(y, 0);
    const transformed = try a.alloc(f32, hidden);
    defer a.free(transformed);
    const inner = try a.alloc(f32, @max(hidden, inter));
    defer a.free(inner);
    const gate_y = try a.alloc(f32, inter);
    defer a.free(gate_y);
    const up_y = try a.alloc(f32, inter);
    defer a.free(up_y);
    const h = try a.alloc(f32, inter);
    defer a.free(h);
    const down_y = try a.alloc(f32, hidden);
    defer a.free(down_y);
    const stride = b.guStride();
    var hits: Hits = .{};
    for (slots, scores) |e_u, w| {
        const e: usize = e_u;
        const off = e * stride;
        format.project(x, b.gate_t[off..][0..stride], b.gate_suh[e * hidden ..][0..hidden], b.gate_svh[e * inter ..][0..inter], hidden, inter, b.rate, dec, transformed, inner[0..inter], gate_y);
        format.project(x, b.up_t[off..][0..stride], b.up_suh[e * hidden ..][0..hidden], b.up_svh[e * inter ..][0..inter], hidden, inter, b.rate, dec, transformed, inner[0..inter], up_y);
        for (0..inter) |i| {
            h[i] = swigluClamped(gate_y[i], up_y[i], limit);
            if (limit > 0) {
                hits.gate += @intFromBool(gate_y[i] > limit);
                hits.up += @intFromBool(@abs(up_y[i]) > limit);
            }
        }
        hits.total += inter;
        format.project(h, b.down_t[off..][0..stride], b.down_suh[e * inter ..][0..inter], b.down_svh[e * hidden ..][0..hidden], inter, hidden, b.rate, dec, inner[0..inter], transformed, down_y);
        for (0..hidden) |i| y[i] += w * down_y[i];
    }
    return hits;
}

/// max |a - b| and the rms of b.
pub const Err = struct { max_abs: f64, rms_ref: f64, max_ratio: f64 };

pub fn compare(got: []const f32, ref: []const f32) Err {
    var max_abs: f64 = 0;
    var ss: f64 = 0;
    for (got, ref) |g, r| {
        max_abs = @max(max_abs, @abs(@as(f64, g) - @as(f64, r)));
        ss += @as(f64, r) * @as(f64, r);
    }
    const rms = @sqrt(ss / @as(f64, @floatFromInt(ref.len)));
    return .{ .max_abs = max_abs, .rms_ref = rms, .max_ratio = if (rms > 0) max_abs / rms else max_abs };
}

/// A dense EXL3 linear in -> out on the host, or `groups` stacked slices of one (a grouped linear's), drawn as `synthBank`
/// draws a projection: trellis [groups, in/16, out/16, n] (every pattern a valid codeword), suh [groups, in] in [0.5, 1.5],
/// svh [groups, out] in [0.5, 1.5] / sqrt(in) (outputs O(1)). `w` holds each slice's W_hat [in, out] (f16 bits,
/// `format.reconstructInner`), decoded once, so a row's reference (`row`) is a GEMV, not a decode of every tile.
pub const Linear = struct {
    groups: usize,
    in: usize,
    out: usize,
    rate: format.Rate,
    trellis: []u16,
    suh: []u16,
    svh: []u16,
    w: []u16,

    pub fn deinit(self: *Linear, a: std.mem.Allocator) void {
        a.free(self.trellis);
        a.free(self.suh);
        a.free(self.svh);
        a.free(self.w);
    }

    /// slice g's trellis halfwords, [in/16, out/16, n]
    pub fn trellisOf(self: *const Linear, g: usize) []const u16 {
        const n = (self.in / 16) * (self.out / 16) * self.rate.n;
        return self.trellis[g * n ..][0..n];
    }

    /// y [out], slice g of one row x [in]: `format.project`'s stages on the decoded W_hat, each output summed over the input
    /// in ascending order as `format.innerGemv` sums it, so `format.project`'s values. `transformed` [in] and `inner` [out]
    /// are scratch.
    pub fn row(self: *const Linear, g: usize, x: []const f32, transformed: []f32, inner: []f32, y: []f32) void {
        const w = self.w[g * self.in * self.out ..][0 .. self.in * self.out];
        format.prepareInput(x, self.suh[g * self.in ..][0..self.in], transformed);
        @memset(inner, 0);
        for (0..self.in) |k| {
            const xv = transformed[k];
            for (inner, w[k * self.out ..][0..self.out]) |*o, bits| o.* += xv * format.f16BitsToF32(bits);
        }
        for (inner) |*v| v.* = format.f16BitsToF32(format.f32ToF16Bits(v.*));
        format.finishOutput(inner, self.svh[g * self.out ..][0..self.out], y);
    }
};

/// `Linear` with random weights at `rate`, its W_hat decoded under `dec`.
pub fn synthLinear(a: std.mem.Allocator, rnd: std.Random, groups: usize, in: usize, out: usize, rate: format.Rate, dec: format.Decode) !Linear {
    const per = (in / 16) * (out / 16) * rate.n;
    const trellis = try a.alloc(u16, groups * per);
    errdefer a.free(trellis);
    const suh = try a.alloc(u16, groups * in);
    errdefer a.free(suh);
    const svh = try a.alloc(u16, groups * out);
    errdefer a.free(svh);
    const w = try a.alloc(u16, groups * in * out);
    errdefer a.free(w);
    for (trellis) |*v| v.* = rnd.int(u16);
    const out_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(in)));
    for (suh) |*s| s.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    for (svh) |*s| s.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * out_scale);
    for (0..groups) |g| format.reconstructInner(trellis[g * per ..][0..per], in, out, rate, dec, w[g * in * out ..][0 .. in * out]);
    return .{ .groups = groups, .in = in, .out = out, .rate = rate, .trellis = trellis, .suh = suh, .svh = svh, .w = w };
}

const testing = std.testing;

test "turboexl3 ref: a 256x128 K4 bank's row through the reference is finite and O(1); slot order is the summation order" {
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rnd = prng.random();
    const a = testing.allocator;
    var b = try synthBank(a, rnd, 4, 256, 128, format.Rate.fromK(4));
    defer b.deinit(a);
    var x: [256]f32 = undefined;
    for (&x) |*v| v.* = rnd.floatNorm(f32);
    var y: [256]f32 = undefined;
    try moeRow(a, &b, &x, &.{ 1, 3 }, &.{ 0.7, 0.3 }, &y);
    var ss: f64 = 0;
    for (y) |v| {
        try testing.expect(std.math.isFinite(v));
        ss += v * v;
    }
    const rms = @sqrt(ss / 256.0);
    try testing.expect(rms > 1e-3 and rms < 100.0);
    // the same slots in the other order differ only by f32 summation order
    var y2: [256]f32 = undefined;
    try moeRow(a, &b, &x, &.{ 3, 1 }, &.{ 0.3, 0.7 }, &y2);
    const e = compare(&y2, &y);
    try testing.expect(e.max_ratio < 1e-5);
}
