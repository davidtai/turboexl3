//! The int8 GEMMs' weight center (`Config.pf_int8_c2`, `prefill.Options.int8_c2`). The routed int8 GEMMs
//! (prefill.zig's `text_gemm_i8`, `text_gemm_i8d`) and trunk int8's (trunk_q8.zig) multiply each MUL1 weight as the int8
//! q = round((t - 510) / 4) of its codeword's byte sum t, halves rounded up (`exl3pf_q8`: (t - 508) >> 2), and their
//! epilogues rebuild it as 4 A q + (c A + C), A = f16 0x1EEE and C = 1024 A + f16 0xC931 (MUL1's map: w = f16((1024 + t)
//! A + f16 0xC931), `format.decodeMul1`). With c = 510 the rebuilt t, 4 q + 510, is t - 1 .. t + 2: half a unit of t
//! high on average, so every rebuilt weight runs A / 2 high and a row's product by sum(x) A / 2 off. c = 509.5 centers
//! it. The texts take the center doubled as a template argument (QC2): 1020 is the texts GLM-5.3-Flash serves (and
//! validated), 1019 their `_c` variants. MLX-free: std and format.zig only (`zig test src/core/int8_center.zig`).
const std = @import("std");
const format = @import("format.zig");

/// The codewords' count: every 16-bit codeword
pub const codewords: usize = 1 << 16;

/// MUL1's byte sum of codeword `cw` (`format.decodeMul1`'s, `exl3pf_q8`'s t): 0 .. 1020 by its type, 0 .. 1005 over the
/// codewords
pub fn byteSum(cw: u16) i32 {
    const mixed = @as(u32, cw) *% format.MUL1_MULT;
    var t: i32 = 0;
    for (std.mem.toBytes(mixed)) |b| t += b;
    return t;
}

/// `exl3pf_q8` on a byte sum: (t - 508) >> 2 (round((t - 510) / 4), halves up), clamped to int8
pub fn q8(t: i32) i32 {
    return std.math.clamp((t - 508) >> 2, -128, 127);
}

/// An epilogue's rebuild of int8 weight `q` at the center c2 / 2, in f32 as the GEMMs compute it: fma(q, 4 A,
/// c2 / 2 A + C) (the epilogue's fma over a row's sum, at one input of 1)
pub fn rebuild(q: i32, c2: u16) f32 {
    const ma = format.f16BitsToF32(0x1EEE);
    const mc = 1024.0 * ma + format.f16BitsToF32(0xC931);
    const center = 0.5 * @as(f32, @floatFromInt(c2)) * ma + mc;
    return @mulAdd(f32, @as(f32, @floatFromInt(q)), 4.0 * ma, center);
}

/// Over every codeword: the rebuild's error against MUL1's weight (w - rebuild, f64), its mean and rms, and the ranges of
/// t and q
pub const Stats = struct { mean: f64, rms: f64, t_min: i32, t_max: i32, q_min: i32, q_max: i32 };

pub fn stats(c2: u16) Stats {
    var sum: f64 = 0;
    var sq: f64 = 0;
    var st: Stats = .{ .mean = 0, .rms = 0, .t_min = std.math.maxInt(i32), .t_max = std.math.minInt(i32), .q_min = 127, .q_max = -128 };
    for (0..codewords) |i| {
        const cw: u16 = @intCast(i);
        const t = byteSum(cw);
        const q = q8(t);
        const w = format.f16BitsToF32(format.decodeMul1(cw));
        const e = @as(f64, w) - @as(f64, rebuild(q, c2));
        sum += e;
        sq += e * e;
        st.t_min = @min(st.t_min, t);
        st.t_max = @max(st.t_max, t);
        st.q_min = @min(st.q_min, q);
        st.q_max = @max(st.q_max, q);
    }
    const n: f64 = @floatFromInt(codewords);
    st.mean = sum / n;
    st.rms = @sqrt(sq / n);
    return st;
}

const testing = std.testing;

test "turboexl3 int8 center: over all 65,536 MUL1 codewords q fits int8 unclamped; the 510 rebuild runs half a unit of t high (mean -0.003380, rms 0.008290), the 509.5 one is centered (mean +0.000004, rms 0.007570)" {
    const a = stats(1020);
    const b = stats(1019);
    // t spans 0 .. 1005 over the codewords, so q stays inside int8 and the clamp never acts
    try testing.expectEqual(@as(i32, 0), a.t_min);
    try testing.expectEqual(@as(i32, 1005), a.t_max);
    try testing.expectEqual(@as(i32, -127), a.q_min);
    try testing.expectEqual(@as(i32, 124), a.q_max);
    try testing.expectEqual(q8(a.t_min), (a.t_min - 508) >> 2);
    try testing.expectEqual(q8(a.t_max), (a.t_max - 508) >> 2);
    // both centers' error, pinned
    try testing.expectApproxEqAbs(@as(f64, -0.003380), a.mean, 0.0000005);
    try testing.expectApproxEqAbs(@as(f64, 0.008290), a.rms, 0.0000005);
    try testing.expectApproxEqAbs(@as(f64, 0.000004), b.mean, 0.0000005);
    try testing.expectApproxEqAbs(@as(f64, 0.007570), b.rms, 0.0000005);
    // 510's bias is half a unit of t, A / 2, within the f16 weights' rounding; 509.5 removes it and with it the rms the
    // bias adds (rms^2 = var + mean^2: the variance is the same at both centers, the rebuilds a constant apart)
    const half_a = 0.5 * @as(f64, format.f16BitsToF32(0x1EEE));
    try testing.expectApproxEqAbs(-half_a, a.mean, 0.00001);
    try testing.expect(@abs(b.mean) < 0.01 * @abs(a.mean));
    try testing.expectApproxEqAbs(a.rms * a.rms - a.mean * a.mean, b.rms * b.rms - b.mean * b.mean, 1e-7);
}

test "turboexl3 int8 center: q is round((t - 510) / 4) with halves rounded up, the rebuilt t (4 q + 510) is t - 1 .. t + 2, and 509.5 is 4 q + 509.5's center" {
    var counts: [4]u32 = @splat(0);
    var t: i32 = 0;
    // the byte sums the codewords reach (the test above pins 0 .. 1005); past 1017 the clamp would act
    while (t <= 1005) : (t += 1) {
        const q = q8(t);
        // round half up of (t - 510) / 4: floor((t - 510) / 4 + 1/2) = floor((t - 508) / 4)
        try testing.expectEqual(@divFloor(t - 508, 4), q);
        const d = 4 * q + 510 - t;
        try testing.expect(d >= -1 and d <= 2);
        counts[@intCast(d + 1)] += 1;
    }
    // every residue of t mod 4 once in four: the rebuilt t's offsets -1, 0, 1, 2 equally often (mean +1/2; at 509.5, 0)
    for (counts[1..]) |c| try testing.expect(c == counts[0] or c + 1 == counts[0] or c == counts[0] + 1);
    // the byte sum is decodeMul1's: its f16 is 1024 + t, mapped
    for ([_]u16{ 0, 1, 7, 0x1234, 0xFFFF }) |cw| {
        const t2 = byteSum(cw);
        const h = format.f16BitsToF32(@truncate(0x6400 + @as(u32, @intCast(t2))));
        try testing.expectEqual(@as(f32, @floatFromInt(1024 + t2)), h);
        const w = @mulAdd(f32, h, format.f16BitsToF32(0x1EEE), format.f16BitsToF32(0xC931));
        try testing.expectEqual(format.f32ToF16Bits(w), format.decodeMul1(cw));
    }
}
