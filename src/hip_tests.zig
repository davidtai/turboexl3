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

// ---- dense projections ------------------------------------------------------------------------------------------

const plan = @import("hip/plan.zig");
const ref = @import("ref.zig");

/// One projection's weights on the host and the device.
const Proj = struct {
    in: u32,
    out: u32,
    rate: format.Rate,
    trellis: []u16,
    suh: []u16,
    svh: []u16,
    dt: rt.Buffer,
    dsuh: rt.Buffer,
    dsvh: rt.Buffer,

    fn init(h: *const rt.Hip, rnd: std.Random, in: u32, out: u32, rate: format.Rate) !Proj {
        const trellis = try gpa.alloc(u16, (in / 16) * (out / 16) * rate.n);
        errdefer gpa.free(trellis);
        for (trellis) |*w| w.* = rnd.int(u16);
        const suh = try gpa.alloc(u16, in);
        errdefer gpa.free(suh);
        for (suh) |*s| s.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
        const svh = try gpa.alloc(u16, out);
        errdefer gpa.free(svh);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(in)));
        for (svh) |*s| s.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * scale);
        return .{ .in = in, .out = out, .rate = rate, .trellis = trellis, .suh = suh, .svh = svh, .dt = try h.upload(std.mem.sliceAsBytes(trellis)), .dsuh = try h.upload(std.mem.sliceAsBytes(suh)), .dsvh = try h.upload(std.mem.sliceAsBytes(svh)) };
    }

    fn deinit(p: *Proj) void {
        p.dt.free();
        p.dsuh.free();
        p.dsvh.free();
        gpa.free(p.trellis);
        gpa.free(p.suh);
        gpa.free(p.svh);
    }
};

/// The kernels' groups, loaded once a test.
const Groups = struct {
    dense: rt.Module,

    fn load(h: *const rt.Hip) !Groups {
        return .{ .dense = try h.load(&objects.dense) };
    }

    fn unload(g: *Groups) void {
        g.dense.unload();
    }

    fn module(g: *const Groups, name: []const u8) !rt.Module {
        if (std.mem.eql(u8, name, "dense")) return g.dense;
        return error.UnknownGroup;
    }
};

/// Runs a plan's launches with each role bound to `bind`'s address.
fn runLaunches(g: *const Groups, launches: []const plan.Launch, bind: std.EnumArray(plan.Role, u64)) !void {
    for (launches) |*l| {
        const f = try (try g.module(l.group)).function(l.symbol);
        var a: rt.Args = .{};
        for (l.argList()) |arg| switch (arg) {
            .buf => |r| try a.add(bind.get(r)),
            .u => |v| try a.add(v),
            .i => |v| try a.add(v),
        };
        try f.launch(.{ .x = l.grid[0], .y = l.grid[1], .z = l.grid[2] }, .{ .x = l.block }, l.shared, &a);
    }
}

/// x [rows][in] in `dtype` (bytes) and its values as f32.
const Input = struct {
    bytes: []u8,
    values: []f32,

    fn init(rnd: std.Random, n: usize, dtype: plan.Dtype) !Input {
        const values = try gpa.alloc(f32, n);
        errdefer gpa.free(values);
        const bytes = try gpa.alloc(u8, n * @as(usize, if (dtype == .f32) 4 else 2));
        for (values, 0..) |*v, i| {
            const f = rnd.floatNorm(f32);
            switch (dtype) {
                .f32 => {
                    v.* = f;
                    std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(f), .little);
                },
                .f16 => {
                    const b = format.f32ToF16Bits(f);
                    v.* = format.f16BitsToF32(b);
                    std.mem.writeInt(u16, bytes[i * 2 ..][0..2], b, .little);
                },
                .bf16 => {
                    const b: u16 = @intCast(@as(u32, @bitCast(f)) >> 16);
                    v.* = @bitCast(@as(u32, b) << 16);
                    std.mem.writeInt(u16, bytes[i * 2 ..][0..2], b, .little);
                },
            }
        }
        return .{ .bytes = bytes, .values = values };
    }

    fn deinit(x: *Input) void {
        gpa.free(x.bytes);
        gpa.free(x.values);
    }
};

fn toF32(bytes: []const u8, dtype: plan.Dtype, out: []f32) void {
    for (out, 0..) |*v, i| v.* = switch (dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little)),
        .f16 => format.f16BitsToF32(std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16),
    };
}

/// The chain over `rows` rows of x (from row `first`): y's bytes. The caller frees.
fn denseRun(h: *const rt.Hip, g: *const Groups, p: *const Proj, x: Input, first: usize, d: plan.Dense) ![]u8 {
    const launches = try plan.denseChain(d);
    const xsize: usize = if (d.dtype == .f32) 4 else 2;
    var xb = try h.upload(x.bytes[first * p.in * xsize ..][0 .. d.rows * p.in * xsize]);
    defer xb.free();
    var xp = try h.alloc(d.xpBytes());
    defer xp.free();
    var xs = try h.alloc(d.xsBytes());
    defer xs.free();
    var z = try h.alloc(d.zBytes());
    defer z.free();
    var y = try h.alloc(@as(usize, d.rows) * p.out * xsize);
    defer y.free();
    var bind = std.EnumArray(plan.Role, u64).initFill(0);
    bind.set(.x, xb.addr());
    bind.set(.suh, p.dsuh.addr());
    bind.set(.svh, p.dsvh.addr());
    bind.set(.trellis, p.dt.addr());
    bind.set(.xp, xp.addr());
    bind.set(.xs, xs.addr());
    bind.set(.z, z.addr());
    bind.set(.y, y.addr());
    try runLaunches(g, &launches, bind);
    const out = try gpa.alloc(u8, y.len);
    errdefer gpa.free(out);
    try y.read(out);
    return out;
}

const DenseCase = struct { in: u32, out: u32, n: u32, dec: format.Decode, dtype: plan.Dtype, defer_map: bool };

test "HIP dense chain: rows 1-16 within format.project's bar per row, each row alone, the lean text's bytes" {
    var h = try rt.Hip.open();
    defer h.close();
    var g = try Groups.load(&h);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(0xDE45);
    const rnd = prng.random();
    const cases = [_]DenseCase{
        .{ .in = 512, .out = 256, .n = 64, .dec = .mcg, .dtype = .f32, .defer_map = false },
        .{ .in = 512, .out = 256, .n = 64, .dec = .mul1, .dtype = .f32, .defer_map = true },
        .{ .in = 512, .out = 256, .n = 64, .dec = .mul1, .dtype = .f32, .defer_map = false },
        .{ .in = 512, .out = 384, .n = 32, .dec = .mcg, .dtype = .f32, .defer_map = false },
        .{ .in = 512, .out = 256, .n = 48, .dec = .mul1, .dtype = .f32, .defer_map = true },
        .{ .in = 512, .out = 256, .n = 36, .dec = .mcg, .dtype = .f32, .defer_map = false },
        .{ .in = 512, .out = 256, .n = 128, .dec = .{ .codebook = .mcg, .window = .w12 }, .dtype = .f32, .defer_map = false },
        .{ .in = 2560, .out = 640, .n = 80, .dec = .mul1, .dtype = .bf16, .defer_map = true },
        .{ .in = 4096, .out = 2048, .n = 64, .dec = .mcg, .dtype = .bf16, .defer_map = false },
    };
    var worst_f32: f64 = 0;
    var worst_bf16: f64 = 0;
    for (cases) |c| {
        const rate = format.Rate.fromN(c.n).?;
        var p = try Proj.init(&h, rnd, c.in, c.out, rate);
        defer p.deinit();
        const max_rows = 16;
        var x = try Input.init(rnd, max_rows * c.in, c.dtype);
        defer x.deinit();
        const yref = try gpa.alloc(f32, max_rows * c.out);
        defer gpa.free(yref);
        {
            const tr = try gpa.alloc(f32, c.in);
            defer gpa.free(tr);
            const inner = try gpa.alloc(f32, c.out);
            defer gpa.free(inner);
            for (0..max_rows) |r| format.project(x.values[r * c.in ..][0..c.in], p.trellis, p.suh, p.svh, c.in, c.out, rate, c.dec, tr, inner, yref[r * c.out ..][0..c.out]);
        }
        const got = try gpa.alloc(f32, max_rows * c.out);
        defer gpa.free(got);
        const ysize: usize = if (c.dtype == .f32) 4 else 2;
        const bar: f64 = if (c.dtype == .f32) 5e-3 else 2.5e-2;
        for ([_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16 }) |rows| {
            for ([_]bool{ true, false }) |lean| {
                const d: plan.Dense = .{ .in = c.in, .out = c.out, .rows = rows, .rate = rate, .dec = c.dec, .dtype = c.dtype, .defer_map = c.defer_map, .lean = lean };
                const y = try denseRun(&h, &g, &p, x, 0, d);
                defer gpa.free(y);
                toF32(y, c.dtype, got[0 .. rows * c.out]);
                for (0..rows) |r| {
                    const e = ref.compare(got[r * c.out ..][0..c.out], yref[r * c.out ..][0..c.out]);
                    try expect(e.max_ratio <= bar, "dense {d}x{d} n {d} {s} rows {d} row {d}: {e:.3} of rms over the bar {e}", .{ c.in, c.out, c.n, @tagName(c.dec.codebook), rows, r, e.max_ratio, bar });
                    if (c.dtype == .f32) worst_f32 = @max(worst_f32, e.max_ratio) else worst_bf16 = @max(worst_bf16, e.max_ratio);
                }
                // each row alone (one-row chain) gives the row's bytes, whatever its chunk and text
                if (rows > 1 and (rows == 16 or rows == 7)) for ([_]usize{ 0, rows / 2, rows - 1 }) |r| {
                    var d1 = d;
                    d1.rows = 1;
                    const alone = try denseRun(&h, &g, &p, x, r, d1);
                    defer gpa.free(alone);
                    try expect(std.mem.eql(u8, alone, y[r * c.out * ysize ..][0 .. c.out * ysize]), "dense {d}x{d} rows {d} lean {}: row {d} differs from the row alone", .{ c.in, c.out, rows, lean, r });
                };
            }
        }
    }
    std.debug.print("RESULT HIP dense chain: rows 1-16 within {e:.3} (f32, bar 5e-3) and {e:.3} (bf16, bar 2.5e-2) of format.project, rows alone and lean on/off byte-equal\n", .{ worst_f32, worst_bf16 });
}
