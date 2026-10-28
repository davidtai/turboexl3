//! MLX's NAX gate (`is_nax_available`: an M5-class GPU and macOS 26.2 or later), read once and MLX-free so the claim
//! can ask it: the chip from `machdep.cpu.brand_string` (an Apple M5 or later carries the GPU's neural accelerators),
//! the OS from `kern.osproductversion`. The server's QA switch FORCE_GPU_FAMILY_FALLBACK=1 pretends they are absent, as
//! the server's own gate does. Below the gate turboexl3's NAX GEMMs (the routed prefill chain, the dense register path)
//! compute wrong, so they stay off (`Config.nax`).

const std = @import("std");
const Global = @import("global.zig").Global;
const builtin = @import("builtin");

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

const cached = Global(?bool, "kit_nax_cached", null);

/// The gate on this machine, probed once.
pub fn available() bool {
    if (cached.get().*) |v| return v;
    var brand: [128]u8 = undefined;
    var os: [64]u8 = undefined;
    const v = from(forcedOff(), sysctl("machdep.cpu.brand_string", &brand), sysctl("kern.osproductversion", &os));
    cached.get().* = v;
    return v;
}

/// The gate over its inputs: not forced off, an Apple M5 or later, macOS 26.2 or later.
pub fn from(forced_off: bool, brand: []const u8, os: []const u8) bool {
    if (forced_off) return false;
    const gen = chipGeneration(brand) orelse return false;
    return gen >= 5 and osAtLeast(os, 26, 2);
}

/// `N` of an "Apple M<N>..." brand (M5, M5 Max, M10); null for any other.
pub fn chipGeneration(brand: []const u8) ?u32 {
    const prefix = "Apple M";
    if (!std.mem.startsWith(u8, brand, prefix)) return null;
    var end: usize = prefix.len;
    while (end < brand.len and std.ascii.isDigit(brand[end])) end += 1;
    return std.fmt.parseInt(u32, brand[prefix.len..end], 10) catch null;
}

/// A "26.4" / "26.4.1" version at least `major`.`minor`; an unparsable part reads 0.
pub fn osAtLeast(ver: []const u8, major: u32, minor: u32) bool {
    var it = std.mem.splitScalar(u8, ver, '.');
    const ma = std.fmt.parseInt(u32, it.first(), 10) catch 0;
    const mi: u32 = if (it.next()) |m| (std.fmt.parseInt(u32, m, 10) catch 0) else 0;
    return ma > major or (ma == major and mi >= minor);
}

fn forcedOff() bool {
    const v = std.c.getenv("FORCE_GPU_FAMILY_FALLBACK") orelse return false;
    return v[0] == '1';
}

fn sysctl(name: [*:0]const u8, buf: []u8) []const u8 {
    if (comptime !builtin.os.tag.isDarwin()) return "";
    var len: usize = buf.len;
    if (sysctlbyname(name, buf.ptr, &len, null, 0) != 0) return "";
    var n = @min(len, buf.len);
    while (n > 0 and buf[n - 1] == 0) n -= 1;
    return buf[0..n];
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{cached};

// ── tests ──

const testing = std.testing;

test "nax: an Apple M5 or later on macOS 26.2 or later passes; an older chip, OS or the forced fallback does not" {
    try testing.expectEqual(@as(?u32, 5), chipGeneration("Apple M5 Max"));
    try testing.expectEqual(@as(?u32, 10), chipGeneration("Apple M10"));
    try testing.expectEqual(@as(?u32, 4), chipGeneration("Apple M4 Pro"));
    try testing.expectEqual(@as(?u32, null), chipGeneration("Intel(R) Core(TM) i9"));
    try testing.expectEqual(@as(?u32, null), chipGeneration("Apple M"));
    try testing.expect(from(false, "Apple M5 Ultra", "26.2"));
    try testing.expect(from(false, "Apple M5", "27.0.1"));
    try testing.expect(!from(false, "Apple M5", "26.1"));
    try testing.expect(!from(false, "Apple M4 Max", "27.0.1"));
    try testing.expect(!from(true, "Apple M5 Max", "26.4"));
    try testing.expect(!from(false, "", ""));
}
