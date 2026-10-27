//! MLX steel's NAX tile headers for the prefill GEMMs (`prefill.zig`): steel/defines.h, utils/type_traits.h,
//! utils/integral_constant.h and gemm/nax.h, in include order (each uses only the ones before it), with their
//! `#pragma once` and `#include "mlx/..."` lines dropped so they sit inline in one kernel header. The consumer's texts
//! when its `sdk` declares `steel` (`defines`, `type_traits`, `integral_constant`, `nax`: its pinned MLX's own files,
//! so an MLX bump that changes them reaches the kernels), else the vendored copy (`steel_nax.metal`, MLX 0.32.2's).
const std = @import("std");
const sdk = @import("sdk");

/// The four headers as one text.
pub const nax: []const u8 = if (@hasDecl(sdk, "steel"))
    inlined(&.{ sdk.steel.defines, sdk.steel.type_traits, sdk.steel.integral_constant, sdk.steel.nax })
else
    @embedFile("kernels/steel_nax.metal");

fn kept(line: []const u8) bool {
    return !std.mem.startsWith(u8, line, "#pragma once") and !std.mem.startsWith(u8, line, "#include \"mlx/");
}

/// `texts` in order, one line at a time, without the lines `kept` drops.
fn inlined(comptime texts: []const []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(20_000_000);
        var n: usize = 0;
        for (texts) |text| {
            var it = std.mem.splitScalar(u8, text, '\n');
            while (it.next()) |line| {
                if (kept(line)) n += line.len + 1;
            }
        }
        var out: [n]u8 = undefined;
        var at: usize = 0;
        for (texts) |text| {
            var it = std.mem.splitScalar(u8, text, '\n');
            while (it.next()) |line| {
                if (!kept(line)) continue;
                @memcpy(out[at..][0..line.len], line);
                out[at + line.len] = '\n';
                at += line.len + 1;
            }
        }
        const final = out;
        return &final;
    }
}

test "steel: the header texts keep every line but the include guards and MLX's own includes, in order" {
    const pin = comptime @import("test_support.zig").pins(@embedFile("kernels/test/pins/steel.metal"));
    const t = std.testing;
    const got = comptime inlined(&.{ @embedFile("kernels/test/steel_inline_1.metal"), @embedFile("kernels/test/steel_inline_2.metal") });
    try t.expectEqualStrings(@embedFile("kernels/test/steel_inlined.metal"), got);
    try t.expect(std.mem.indexOf(u8, nax, pin[0]) != null);
    try t.expect(std.mem.indexOf(u8, nax, pin[1]) == null);
}
