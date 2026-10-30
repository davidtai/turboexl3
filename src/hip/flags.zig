//! The HIP kernel groups and how hipcc compiles them: one code object a group, for one exact gfx name. TensorFold
//! vendors this file with the kernels, so both builds compile the same code. No kernel text here: the groups are files
//! under src/kernels/hip/.
const std = @import("std");

/// A group: its source (under src/kernels/hip/) and its code object's name.
pub const Group = struct { source: []const u8, name: []const u8 };

/// Every group, in the order the objects module embeds them.
pub const groups = [_]Group{
    .{ .source = "test/lane.hip", .name = "test_lane" },
    .{ .source = "dense/dense.hip", .name = "dense" },
    .{ .source = "decode/decode.hip", .name = "decode" },
};

/// Every header the groups include (under src/kernels/hip/), so an edit rebuilds them.
pub const headers = [_][]const u8{
    "lane.hpp",
    "common.hpp",
};

/// The chips the tests run on; the build refuses any other -Dhip-arch.
pub const arches = [_][]const u8{"gfx1151"};

/// hipcc's flags for every group: wave32, no FP contraction (each multiply and add rounds as written), C++20.
pub const flags = [_][]const u8{ "--genco", "-O3", "-std=c++20", "-ffp-contract=off", "-mno-wavefrontsize64", "-fno-gpu-rdc" };

pub fn admitted(arch: []const u8) bool {
    for (arches) |a| if (std.mem.eql(u8, a, arch)) return true;
    return false;
}
