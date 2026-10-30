//! turboexl3's AMD (HIP) kernels on a real GPU against the host reference (format.zig): `zig build hip-test -Dhip=true`.
//! The tests need libamdhip64 and a GPU of a -Dhip-arch the build admits (src/hip/flags.zig); they import no MLX.
const std = @import("std");
const format = @import("format.zig");
const lane = @import("lane.zig");
const rt = @import("hip/runtime.zig");
const objects = @import("turboexl3_hip_objects");

const gpa = std.testing.allocator;

fn expect(ok: bool, comptime fmt: []const u8, args: anytype) !void {
    if (ok) return;
    std.debug.print(fmt ++ "\n", args);
    return error.TestUnexpectedResult;
}

fn read(comptime T: type, b: rt.Buffer, n: usize) ![]T {
    const out = try gpa.alloc(T, n);
    errdefer gpa.free(out);
    try b.read(std.mem.sliceAsBytes(out));
    return out;
}

test "HIP codebooks: every 16-bit codeword's MCG and MUL1 values and MUL1's byte sum equal format.zig's" {
    var h = try rt.Hip.open();
    defer h.close();
    var m = try h.load(&objects.test_lane);
    defer m.unload();
    const f = try m.function("t3_test_codebooks");
    var mcg = try h.alloc(65536 * 2);
    defer mcg.free();
    var mul1 = try h.alloc(65536 * 2);
    defer mul1.free();
    var t = try h.alloc(65536 * 2);
    defer t.free();
    var a: rt.Args = .{};
    try a.add(mcg.addr());
    try a.add(mul1.addr());
    try a.add(t.addr());
    try f.launch(.{ .x = 256 }, .{ .x = 256 }, 0, &a);
    const gm = try read(u16, mcg, 65536);
    defer gpa.free(gm);
    const gu = try read(u16, mul1, 65536);
    defer gpa.free(gu);
    const gt = try read(u16, t, 65536);
    defer gpa.free(gt);
    for (0..65536) |i| {
        const cw: u16 = @intCast(i);
        try expect(gm[i] == format.decodeMcg(cw), "MCG {x}: {x} for {x}", .{ cw, gm[i], format.decodeMcg(cw) });
        try expect(gu[i] == format.decodeMul1(cw), "MUL1 {x}: {x} for {x}", .{ cw, gu[i], format.decodeMul1(cw) });
        const mixed = @as(u32, cw) *% format.MUL1_MULT;
        var sum: u32 = 0;
        for (std.mem.toBytes(mixed)) |byte| sum += byte;
        try expect(gt[i] == sum, "MUL1 byte sum {x}: {d} for {d}", .{ cw, gt[i], sum });
    }
    std.debug.print("RESULT HIP codebooks: 65,536 codewords, MCG and MUL1 values and MUL1 byte sums equal format.zig\n", .{});
}

test "HIP lane reader: every lane's codewords and decoded values equal format.unpackTile at every even n and window" {
    var h = try rt.Hip.open();
    defer h.close();
    var m = try h.load(&objects.test_lane);
    defer m.unload();
    const f = try m.function("t3_test_lane");
    var prng = std.Random.DefaultPrng.init(0x7E3A);
    const rnd = prng.random();
    const tiles = 64;
    var checked: usize = 0;
    var n: u32 = format.Rate.min_n;
    while (n <= format.Rate.max_n) : (n += 2) {
        const rate = format.Rate.fromN(n).?;
        const words = try gpa.alloc(u16, tiles * n);
        defer gpa.free(words);
        for (words) |*w| w.* = rnd.int(u16);
        var tr = try h.upload(std.mem.sliceAsBytes(words));
        defer tr.free();
        var cw_b = try h.alloc(tiles * 256 * 2);
        defer cw_b.free();
        var mcg_b = try h.alloc(tiles * 256 * 2);
        defer mcg_b.free();
        var mul1_b = try h.alloc(tiles * 256 * 2);
        defer mul1_b.free();
        for (0..format.Window.count) |wi| {
            const window: format.Window = @fromBackingInt(@intCast(wi));
            const mask: u32 = window.mask();
            var a: rt.Args = .{};
            try a.add(tr.addr());
            try a.add(n);
            try a.add(mask);
            try a.add(cw_b.addr());
            try a.add(mcg_b.addr());
            try a.add(mul1_b.addr());
            try f.launch(.{ .x = tiles }, .{ .x = 32 }, 0, &a);
            const gc = try read(u16, cw_b, tiles * 256);
            defer gpa.free(gc);
            const gm = try read(u16, mcg_b, tiles * 256);
            defer gpa.free(gm);
            const gu = try read(u16, mul1_b, tiles * 256);
            defer gpa.free(gu);
            for (0..tiles) |ti| {
                var want: [format.TILE_VALUES]u16 = undefined;
                format.unpackTile(words[ti * n ..][0..n], rate, &want);
                for (0..256) |p| {
                    const cw = want[p] & window.mask();
                    const at = ti * 256 + p;
                    try expect(gc[at] == cw, "n {d} w{d} tile {d} position {d}: codeword {x} for {x}", .{ n, window.bits(), ti, p, gc[at], cw });
                    try expect(gm[at] == format.decodeMcg(cw), "n {d} w{d}: MCG of {x}", .{ n, window.bits(), cw });
                    try expect(gu[at] == format.decodeMul1(cw), "n {d} w{d}: MUL1 of {x}", .{ n, window.bits(), cw });
                }
                checked += 1;
            }
        }
        // the host lane plan (lane.zig, the Metal arithmetic) gives the same codewords as the HIP reader
        for (0..32) |l| {
            var host: [8]u16 = undefined;
            lane.laneCodewords(words[0..n], rate, @intCast(l), &host);
            var want: [format.TILE_VALUES]u16 = undefined;
            format.unpackTile(words[0..n], rate, &want);
            for (host, 0..) |v, mm| try expect(v == want[8 * l + mm], "n {d}: lane.zig lane {d} window {d}", .{ n, l, mm });
        }
    }
    std.debug.print("RESULT HIP lane reader: {d} tiles over every even n in [16, 128] and every window 8-16 equal format.unpackTile, MCG and MUL1 values equal format.zig\n", .{checked});
}
