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
    decode: rt.Module,
    prefill: rt.Module,

    fn load(h: *const rt.Hip) !Groups {
        var g: Groups = .{ .dense = try h.load(&objects.dense), .decode = undefined, .prefill = undefined };
        errdefer g.dense.unload();
        g.decode = try h.load(&objects.decode);
        errdefer g.decode.unload();
        g.prefill = try h.load(&objects.prefill);
        return g;
    }

    fn unload(g: *Groups) void {
        g.dense.unload();
        g.decode.unload();
        g.prefill.unload();
    }

    fn module(g: *const Groups, name: []const u8) !rt.Module {
        if (std.mem.eql(u8, name, "dense")) return g.dense;
        if (std.mem.eql(u8, name, "decode")) return g.decode;
        if (std.mem.eql(u8, name, "prefill")) return g.prefill;
        return error.UnknownGroup;
    }
};

/// Runs a plan's launches with each role bound to `bind`'s address.
fn runLaunches(g: *const Groups, launches: []const plan.Launch, bind: std.EnumArray(plan.Role, u64)) !void {
    try runLaunchesOn(g, null, launches, bind);
}

fn runLaunchesOn(g: *const Groups, stream: ?rt.Stream, launches: []const plan.Launch, bind: std.EnumArray(plan.Role, u64)) !void {
    for (launches) |*l| {
        const f = try (try g.module(l.group)).function(l.symbol);
        var a: rt.Args = .{};
        for (l.argList()) |arg| switch (arg) {
            .buf => |r| try a.add(bind.get(r)),
            .u => |v| try a.add(v),
            .i => |v| try a.add(v),
            .f => |v| try a.add(v),
        };
        try f.launchOn(stream, .{ .x = l.grid[0], .y = l.grid[1], .z = l.grid[2] }, .{ .x = l.block }, l.shared, &a);
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

// ---- routed experts at decode rows ------------------------------------------------------------------------------

/// A bank (ref.synthBank's layout: [E, in/16, out/16, n], suh [E, in], svh [E, out]) on the device.
const DevBank = struct {
    bufs: [9]rt.Buffer,

    fn init(h: *const rt.Hip, b: *const ref.Bank) !DevBank {
        var d: DevBank = undefined;
        const host = [9][]const u16{ b.gate_t, b.up_t, b.down_t, b.gate_suh, b.up_suh, b.gate_svh, b.up_svh, b.down_suh, b.down_svh };
        var made: usize = 0;
        errdefer for (d.bufs[0..made]) |*x| x.free();
        for (host, 0..) |arr, i| {
            d.bufs[i] = try h.upload(std.mem.sliceAsBytes(arr));
            made += 1;
        }
        return d;
    }

    fn deinit(d: *DevBank) void {
        for (&d.bufs) |*x| x.free();
    }

    fn bindTo(d: *const DevBank, bind: *std.EnumArray(plan.Role, u64)) void {
        const roles = [9]plan.Role{ .gate_t, .up_t, .down_t, .gate_suh, .up_suh, .gate_svh, .up_svh, .down_suh, .down_svh };
        for (roles, 0..) |r, i| bind.set(r, d.bufs[i].addr());
    }
};

/// Form 3 over `d.rows` rows of x from row `first` (slots and scores from row `first` too): y's bytes (fill 0xff first).
fn decodeRun(h: *const rt.Hip, g: *const Groups, bank: *const DevBank, x: Input, slots: []const i32, scores: []const f32, first: usize, d: plan.Decode) ![]u8 {
    return decodeRunForm(h, g, bank, x, slots, scores, first, d, 3);
}

/// `decodeRun` on form 3 or form 9.
fn decodeRunForm(h: *const rt.Hip, g: *const Groups, bank: *const DevBank, x: Input, slots: []const i32, scores: []const f32, first: usize, d: plan.Decode, form: u8) ![]u8 {
    const f3 = try plan.decodeForm3(d);
    const f9 = try plan.decodeForm9(d);
    const launches: []const plan.Launch = if (form == 9) f9.launches[0..f9.len] else &f3;
    const xsize: usize = if (d.dtype == .f32) 4 else 2;
    var xb = try h.upload(x.bytes[first * d.hidden * xsize ..][0 .. d.rows * d.hidden * xsize]);
    defer xb.free();
    var sb = try h.upload(std.mem.sliceAsBytes(slots[first * d.topk ..][0 .. d.rows * d.topk]));
    defer sb.free();
    var cb = try h.upload(std.mem.sliceAsBytes(scores[first * d.topk ..][0 .. d.rows * d.topk]));
    defer cb.free();
    var bind = std.EnumArray(plan.Role, u64).initFill(0);
    bank.bindTo(&bind);
    bind.set(.x, xb.addr());
    bind.set(.slots, sb.addr());
    bind.set(.scores, cb.addr());
    const scratch = [_]plan.Role{ .xg, .xu, .sg, .su, .zg, .zu, .hd, .hs, .zd, .y };
    var bufs: [scratch.len]rt.Buffer = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for (scratch, 0..) |r, i| {
        bufs[i] = try h.alloc(d.bytes(r));
        made += 1;
        bind.set(r, bufs[i].addr());
    }
    try bufs[scratch.len - 1].fill(0xff);
    try runLaunches(g, launches, bind);
    const out = try gpa.alloc(u8, d.bytes(.y));
    errdefer gpa.free(out);
    try bufs[scratch.len - 1].read(out);
    return out;
}

/// Each row's `topk` distinct slots below `experts` (then, with `shared`, the id `experts` last) and softmax-like scores.
fn routing(rnd: std.Random, rows: usize, topk: usize, experts: usize, shared: bool) !struct { slots: []i32, scores: []f32 } {
    const k = topk + @intFromBool(shared);
    const slots = try gpa.alloc(i32, rows * k);
    errdefer gpa.free(slots);
    const scores = try gpa.alloc(f32, rows * k);
    for (0..rows) |r| {
        var seen: [512]bool = @splat(false);
        for (0..topk) |j| {
            var e = rnd.uintLessThan(usize, experts);
            while (seen[e]) e = rnd.uintLessThan(usize, experts);
            seen[e] = true;
            slots[r * k + j] = @intCast(e);
            scores[r * k + j] = 0.05 + rnd.float(f32) * 0.2;
        }
        if (shared) {
            slots[r * k + topk] = @intCast(experts);
            scores[r * k + topk] = 1.0;
        }
    }
    return .{ .slots = slots, .scores = scores };
}

const DecodeCase = struct { hidden: u32, inter: u32, experts: u32, topk: u32, n: u32, dec: format.Decode, dtype: plan.Dtype, defer_map: bool };

test "HIP routed decode, form 3: rows 1-16 within format.project's bar per row (ref.moeRowDec), each row alone" {
    var h = try rt.Hip.open();
    defer h.close();
    var g = try Groups.load(&h);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(0xF03);
    const rnd = prng.random();
    const cases = [_]DecodeCase{
        .{ .hidden = 512, .inter = 256, .experts = 12, .topk = 4, .n = 64, .dec = .mcg, .dtype = .f32, .defer_map = false },
        .{ .hidden = 512, .inter = 256, .experts = 12, .topk = 4, .n = 64, .dec = .mul1, .dtype = .f32, .defer_map = true },
        .{ .hidden = 512, .inter = 256, .experts = 12, .topk = 4, .n = 64, .dec = .mul1, .dtype = .f32, .defer_map = false },
        .{ .hidden = 512, .inter = 256, .experts = 12, .topk = 8, .n = 36, .dec = .mcg, .dtype = .f32, .defer_map = false },
        .{ .hidden = 512, .inter = 384, .experts = 12, .topk = 4, .n = 128, .dec = .{ .codebook = .mul1, .window = .w12 }, .dtype = .f32, .defer_map = true },
        .{ .hidden = 2560, .inter = 640, .experts = 16, .topk = 10, .n = 48, .dec = .mul1, .dtype = .bf16, .defer_map = true },
    };
    var worst_f32: f64 = 0;
    var worst_bf16: f64 = 0;
    for (cases) |c| {
        const rate = format.Rate.fromN(c.n).?;
        var bank = try ref.synthBank(gpa, rnd, c.experts, c.hidden, c.inter, rate);
        defer bank.deinit(gpa);
        var dbank = try DevBank.init(&h, &bank);
        defer dbank.deinit();
        const max_rows = 16;
        var x = try Input.init(rnd, max_rows * c.hidden, c.dtype);
        defer x.deinit();
        const rt_ = try routing(rnd, max_rows, c.topk, c.experts, false);
        defer gpa.free(rt_.slots);
        defer gpa.free(rt_.scores);
        const yref = try gpa.alloc(f32, max_rows * c.hidden);
        defer gpa.free(yref);
        for (0..max_rows) |r| {
            var sl: [32]u32 = undefined;
            for (0..c.topk) |j| sl[j] = @intCast(rt_.slots[r * c.topk + j]);
            try ref.moeRowDec(gpa, &bank, c.dec, x.values[r * c.hidden ..][0..c.hidden], sl[0..c.topk], rt_.scores[r * c.topk ..][0..c.topk], yref[r * c.hidden ..][0..c.hidden]);
        }
        const got = try gpa.alloc(f32, max_rows * c.hidden);
        defer gpa.free(got);
        const ysize: usize = if (c.dtype == .f32) 4 else 2;
        const bar: f64 = if (c.dtype == .f32) 5e-3 else 1.5e-2;
        for ([_]u32{ 1, 2, 3, 4, 5, 8, 16 }) |rows| {
            const d: plan.Decode = .{ .hidden = c.hidden, .inter = c.inter, .topk = c.topk, .rows = rows, .experts = c.experts, .rate = rate, .dec = c.dec, .dtype = c.dtype, .defer_map = c.defer_map };
            const y = try decodeRun(&h, &g, &dbank, x, rt_.slots, rt_.scores, 0, d);
            defer gpa.free(y);
            toF32(y, c.dtype, got[0 .. rows * c.hidden]);
            for (0..rows) |r| {
                const e = ref.compare(got[r * c.hidden ..][0..c.hidden], yref[r * c.hidden ..][0..c.hidden]);
                try expect(e.max_ratio <= bar, "decode {d}x{d} n {d} {s} rows {d} row {d}: {e:.3} over the bar {e}", .{ c.hidden, c.inter, c.n, @tagName(c.dec.codebook), rows, r, e.max_ratio, bar });
                if (c.dtype == .f32) worst_f32 = @max(worst_f32, e.max_ratio) else worst_bf16 = @max(worst_bf16, e.max_ratio);
            }
            if (rows > 1) for ([_]usize{ 0, rows - 1 }) |r| {
                var d1 = d;
                d1.rows = 1;
                const alone = try decodeRun(&h, &g, &dbank, x, rt_.slots, rt_.scores, r, d1);
                defer gpa.free(alone);
                try expect(std.mem.eql(u8, alone, y[r * c.hidden * ysize ..][0 .. c.hidden * ysize]), "decode {d}x{d} rows {d}: row {d} differs from the row alone", .{ c.hidden, c.inter, rows, r });
            };
        }
    }
    std.debug.print("RESULT HIP routed decode form 3 (format.project's profile): within {e:.3} (f32, bar 5e-3) and {e:.3} (bf16, bar 1.5e-2), rows alone byte-equal\n", .{ worst_f32, worst_bf16 });
}

// ---- the GLM profile against float64 --------------------------------------------------------------------------

/// The 128-blocks of a row-major [rows][cols] f64 matrix along `axis` through H / sqrt(128), a butterfly in f64.
fn rotate64(x: []f64, rows: usize, cols: usize, axis: usize) void {
    const inv = 1.0 / @sqrt(128.0);
    if (axis == 0) {
        var b: usize = 0;
        while (b < rows) : (b += 128) {
            var hh: usize = 1;
            while (hh < 128) : (hh *= 2) {
                var i: usize = 0;
                while (i < 128) : (i += 2 * hh) for (i..i + hh) |r| {
                    const p = x[(b + r) * cols ..][0..cols];
                    const q = x[(b + r + hh) * cols ..][0..cols];
                    for (p, q) |*u, *v| {
                        const t = u.*;
                        u.* = t + v.*;
                        v.* = t - v.*;
                    }
                };
            }
        }
    } else for (0..rows) |r| {
        var b: usize = 0;
        while (b < cols) : (b += 128) {
            const v = x[r * cols + b ..][0..128];
            var hh: usize = 1;
            while (hh < 128) : (hh *= 2) {
                var i: usize = 0;
                while (i < 128) : (i += 2 * hh) for (i..i + hh) |j| {
                    const t = v[j];
                    v[j] = t + v[j + hh];
                    v[j + hh] = t - v[j + hh];
                };
            }
        }
    }
    for (x) |*v| v.* *= inv;
}

/// W = diag(suh) H W_q H diag(svh) [in][out] in f64 (W_q from format.reconstructInner; f16 scratch `wq`).
fn dequant64(trellis: []const u16, suh: []const u16, svh: []const u16, in: usize, out: usize, rate: format.Rate, dec: format.Decode, w: []f64, wq: []u16) void {
    format.reconstructInner(trellis, in, out, rate, dec, wq[0 .. in * out]);
    for (w[0 .. in * out], wq[0 .. in * out]) |*d, b| d.* = format.f16BitsToF32(b);
    rotate64(w[0 .. in * out], in, out, 0);
    for (w[0 .. in * out], 0..) |*v, i| v.* *= format.f16BitsToF32(suh[i / out]);
    rotate64(w[0 .. in * out], in, out, 1);
    for (w[0 .. in * out], 0..) |*v, i| v.* *= format.f16BitsToF32(svh[i % out]);
}

fn bf16Round(v: f64) f64 {
    const f: f32 = @floatCast(v);
    if (std.math.isNan(f)) return f;
    const u: u32 = @bitCast(f);
    return @as(f32, @bitCast((u +% 0x7fff +% ((u >> 16) & 1)) & 0xffff0000));
}

/// GLM's SwiGLU: gate and up rounded to bf16 and limited, silu rounded to bf16, the product rounded to bf16.
fn glmAct(gv: f64, uv: f64, limit: f64) f64 {
    var gg = bf16Round(gv);
    var uu = bf16Round(uv);
    if (limit > 0) {
        gg = @min(gg, limit);
        uu = @max(@min(uu, limit), -limit);
    }
    return bf16Round(bf16Round(gg / (1 + @exp(-gg))) * uu);
}

/// The routed slots of rows `which` (every row of `rows` when null) against float64 under GLM's rounding: each slot's f32
/// output y [row][k][hidden] within the CUDA test's tolerance (sum |a w| / 64 + 1e-6); returns the worst ratio.
fn glmExpect(bank: *const ref.Bank, rate: format.Rate, x: Input, slots: []const i32, k: usize, topk: usize, limit: f32, rows: usize, which: ?[]const usize, y: []const f32) !f64 {
    const D = bank.hidden;
    const N = bank.inter;
    var all: [64]usize = undefined;
    const list = which orelse blk: {
        for (all[0..rows], 0..) |*r, i| r.* = i;
        break :blk all[0..rows];
    };
    var worst: f64 = 0;
    var w: [3][]f64 = undefined;
    for (&w) |*m| m.* = try gpa.alloc(f64, D * N);
    defer for (w) |m| gpa.free(m);
    const wq = try gpa.alloc(u16, D * N);
    defer gpa.free(wq);
    const gu = try gpa.alloc(f64, 2 * N);
    defer gpa.free(gu);
    const yy = try gpa.alloc(f64, 2 * D);
    defer gpa.free(yy);
    const stride = bank.guStride();
    for (0..bank.experts) |e| {
        var used = false;
        for (list) |r| for (0..topk) |j| {
            used = used or slots[r * k + j] == @as(i32, @intCast(e));
        };
        if (!used) continue;
        dequant64(bank.gate_t[e * stride ..][0..stride], bank.gate_suh[e * D ..][0..D], bank.gate_svh[e * N ..][0..N], D, N, rate, .mcg, w[0], wq);
        dequant64(bank.up_t[e * stride ..][0..stride], bank.up_suh[e * D ..][0..D], bank.up_svh[e * N ..][0..N], D, N, rate, .mcg, w[1], wq);
        dequant64(bank.down_t[e * stride ..][0..stride], bank.down_suh[e * N ..][0..N], bank.down_svh[e * D ..][0..D], N, D, rate, .mcg, w[2], wq);
        for (list) |r| for (0..topk) |j| {
            if (slots[r * k + j] != @as(i32, @intCast(e))) continue;
            @memset(gu, 0);
            for (0..D) |kk| {
                const xv: f64 = x.values[r * D + kk];
                for (gu[0..N], gu[N..], w[0][kk * N ..][0..N], w[1][kk * N ..][0..N]) |*a, *b, w0, w1| {
                    a.* += xv * w0;
                    b.* += xv * w1;
                }
            }
            for (0..N) |nn| gu[nn] = glmAct(gu[nn], gu[N + nn], limit);
            @memset(yy, 0);
            for (0..N) |kk| for (yy[0..D], yy[D..], w[2][kk * D ..][0..D]) |*o, *mag, wd| {
                o.* += gu[kk] * wd;
                mag.* += @abs(gu[kk] * wd);
            };
            for (0..D) |cc| {
                const tol = yy[D + cc] / 64.0 + 1e-6;
                const q = @abs(@as(f64, y[(r * k + j) * D + cc]) - yy[cc]) / tol;
                try expect(q <= 1, "GLM {d}x{d} row {d} slot {d} column {d}: {d:.3} of its tolerance", .{ D, N, r, j, cc, q });
                worst = @max(worst, q);
            }
        };
    }
    return worst;
}

test "HIP routed decode, GLM profile: each routed slot's f32 output within the CUDA test's float64 tolerance" {
    var h = try rt.Hip.open();
    defer h.close();
    var g = try Groups.load(&h);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(0x61A);
    const rnd = prng.random();
    const limit: f32 = 7.0;
    const Shape = struct { hidden: u32, inter: u32, experts: u32 };
    var worst: f64 = 0;
    for ([_]Shape{ .{ .hidden = 512, .inter = 256, .experts = 12 }, .{ .hidden = 4096, .inter = 2048, .experts = 9 } }) |sh| {
        const rate = format.Rate.fromK(4);
        var bank = try ref.synthBank(gpa, rnd, sh.experts, sh.hidden, sh.inter, rate);
        defer bank.deinit(gpa);
        var dbank = try DevBank.init(&h, &bank);
        defer dbank.deinit();
        const topk = 8;
        const k = topk + 1; // the shared slot last, skipped
        const max_rows = 8;
        var x = try Input.init(rnd, max_rows * sh.hidden, .bf16);
        defer x.deinit();
        const rt_ = try routing(rnd, max_rows, topk, sh.experts, true);
        defer gpa.free(rt_.slots);
        defer gpa.free(rt_.scores);
        const D = sh.hidden;
        const N = sh.inter;
        for ([_]u32{ 1, 4, 8 }) |rows| {
            const d: plan.Decode = .{ .hidden = D, .inter = N, .topk = k, .rows = rows, .experts = sh.experts, .rate = rate, .dec = .mcg, .dtype = .bf16, .defer_map = false, .profile = .glm, .limit = limit };
            const yb = try decodeRun(&h, &g, &dbank, x, rt_.slots, rt_.scores, 0, d);
            defer gpa.free(yb);
            const y: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, yb));
            // the shared slot untouched (the fill's bytes)
            for (0..rows) |r| for (y[(r * k + topk) * D ..][0..D]) |v| try expect(@as(u32, @bitCast(v)) == 0xffffffff, "GLM decode: the shared slot left alone", .{});
            worst = @max(worst, try glmExpect(&bank, rate, x, rt_.slots, k, topk, limit, rows, null, y));
        }
    }
    std.debug.print("RESULT HIP routed decode form 3 (GLM profile): every routed slot within {d:.3} of the float64 tolerance, the shared slot untouched\n", .{worst});
}

// ---- routed experts at prefill rows, dense rows above 16 ---------------------------------------------------------

/// The prefill chain over `d.rows` rows of x from row `first`: y's bytes (fill 0xff first).
fn prefillRun(h: *const rt.Hip, g: *const Groups, bank: *const DevBank, x: Input, slots: []const i32, scores: []const f32, first: usize, d: plan.Prefill) ![]u8 {
    const launches = try plan.prefillChain(d);
    const xsize: usize = if (d.dtype == .f32) 4 else 2;
    var xb = try h.upload(x.bytes[first * d.hidden * xsize ..][0 .. d.rows * d.hidden * xsize]);
    defer xb.free();
    var sb = try h.upload(std.mem.sliceAsBytes(slots[first * d.topk ..][0 .. d.rows * d.topk]));
    defer sb.free();
    var cb = try h.upload(std.mem.sliceAsBytes(scores[first * d.topk ..][0 .. d.rows * d.topk]));
    defer cb.free();
    var bind = std.EnumArray(plan.Role, u64).initFill(0);
    bank.bindTo(&bind);
    bind.set(.x, xb.addr());
    bind.set(.slots, sb.addr());
    bind.set(.scores, cb.addr());
    const scratch = [_]plan.Role{ .order, .inv, .items, .counts, .xg, .xu, .sg, .su, .zg, .hd, .hs, .zd, .y };
    var bufs: [scratch.len]rt.Buffer = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for (scratch, 0..) |r, i| {
        bufs[i] = try h.alloc(d.bytes(r));
        made += 1;
        bind.set(r, bufs[i].addr());
    }
    try bufs[scratch.len - 1].fill(0xff);
    try runLaunches(g, &launches, bind);
    const out = try gpa.alloc(u8, d.bytes(.y));
    errdefer gpa.free(out);
    try bufs[scratch.len - 1].read(out);
    return out;
}

test "HIP routed prefill: sampled rows within format.project's bar (ref.moeRowDec); a row's bytes ignore its placement" {
    var h = try rt.Hip.open();
    defer h.close();
    var g = try Groups.load(&h);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(0x9F1);
    const rnd = prng.random();
    const cases = [_]DecodeCase{
        .{ .hidden = 512, .inter = 256, .experts = 12, .topk = 4, .n = 64, .dec = .mcg, .dtype = .f32, .defer_map = false },
        .{ .hidden = 512, .inter = 256, .experts = 12, .topk = 4, .n = 64, .dec = .mul1, .dtype = .f32, .defer_map = true },
        .{ .hidden = 512, .inter = 256, .experts = 12, .topk = 8, .n = 36, .dec = .{ .codebook = .mcg, .window = .w12 }, .dtype = .f32, .defer_map = false },
        .{ .hidden = 2560, .inter = 640, .experts = 16, .topk = 10, .n = 48, .dec = .mul1, .dtype = .bf16, .defer_map = true },
    };
    var worst_f32: f64 = 0;
    var worst_bf16: f64 = 0;
    for (cases) |c| {
        const rate = format.Rate.fromN(c.n).?;
        var bank = try ref.synthBank(gpa, rnd, c.experts, c.hidden, c.inter, rate);
        defer bank.deinit(gpa);
        var dbank = try DevBank.init(&h, &bank);
        defer dbank.deinit();
        const max_rows: usize = if (c.hidden > 512) 200 else 520;
        var x = try Input.init(rnd, max_rows * c.hidden, c.dtype);
        defer x.deinit();
        const rt_ = try routing(rnd, max_rows, c.topk, c.experts, false);
        defer gpa.free(rt_.slots);
        defer gpa.free(rt_.scores);
        const ysize: usize = if (c.dtype == .f32) 4 else 2;
        const bar: f64 = if (c.dtype == .f32) 5e-3 else 1.5e-2;
        const yr = try gpa.alloc(f32, c.hidden);
        defer gpa.free(yr);
        const got = try gpa.alloc(f32, c.hidden);
        defer gpa.free(got);
        var y17: ?[]u8 = null;
        defer if (y17) |b| gpa.free(b);
        for ([_]usize{ 17, 40, 200, 520 }) |rows| {
            if (rows > max_rows) continue;
            const d: plan.Prefill = .{ .hidden = c.hidden, .inter = c.inter, .topk = c.topk, .rows = @intCast(rows), .experts = c.experts, .rate = rate, .dec = c.dec, .dtype = c.dtype, .defer_map = c.defer_map };
            const y = try prefillRun(&h, &g, &dbank, x, rt_.slots, rt_.scores, 0, d);
            defer gpa.free(y);
            for ([_]usize{ 0, 1, rows / 2, rows - 1 }) |r| {
                var sl: [32]u32 = undefined;
                for (0..c.topk) |j| sl[j] = @intCast(rt_.slots[r * c.topk + j]);
                try ref.moeRowDec(gpa, &bank, c.dec, x.values[r * c.hidden ..][0..c.hidden], sl[0..c.topk], rt_.scores[r * c.topk ..][0..c.topk], yr);
                toF32(y[r * c.hidden * ysize ..][0 .. c.hidden * ysize], c.dtype, got);
                const e = ref.compare(got, yr);
                try expect(e.max_ratio <= bar, "prefill {d}x{d} n {d} rows {d} row {d}: {e:.3} over the bar {e}", .{ c.hidden, c.inter, c.n, rows, r, e.max_ratio, bar });
                if (c.dtype == .f32) worst_f32 = @max(worst_f32, e.max_ratio) else worst_bf16 = @max(worst_bf16, e.max_ratio);
            }
            // the first 17 rows' bytes do not depend on the rows after them (their slots land elsewhere in the order)
            if (y17) |first| {
                try expect(std.mem.eql(u8, first, y[0 .. 17 * c.hidden * ysize]), "prefill {d}x{d}: rows 0-16 of {d} rows differ from the 17-row call", .{ c.hidden, c.inter, rows });
            } else y17 = try gpa.dupe(u8, y[0 .. 17 * c.hidden * ysize]);
        }
    }
    std.debug.print("RESULT HIP routed prefill (format.project's profile): sampled rows within {e:.3} (f32, bar 5e-3) and {e:.3} (bf16, bar 1.5e-2); rows 0-16 byte-equal at 17-520 rows\n", .{ worst_f32, worst_bf16 });
}

test "HIP routed prefill, GLM profile: sampled rows' routed slots within the CUDA test's float64 tolerance" {
    var h = try rt.Hip.open();
    defer h.close();
    var g = try Groups.load(&h);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(0x61B);
    const rnd = prng.random();
    const limit: f32 = 7.0;
    var worst: f64 = 0;
    const Shape = struct { hidden: u32, inter: u32, experts: u32 };
    for ([_]Shape{ .{ .hidden = 512, .inter = 256, .experts = 12 }, .{ .hidden = 4096, .inter = 2048, .experts = 9 } }) |sh| {
        const rate = format.Rate.fromK(4);
        var bank = try ref.synthBank(gpa, rnd, sh.experts, sh.hidden, sh.inter, rate);
        defer bank.deinit(gpa);
        var dbank = try DevBank.init(&h, &bank);
        defer dbank.deinit();
        const topk = 8;
        const k = topk + 1;
        const rows = 40;
        var x = try Input.init(rnd, rows * sh.hidden, .bf16);
        defer x.deinit();
        const rt_ = try routing(rnd, rows, topk, sh.experts, true);
        defer gpa.free(rt_.slots);
        defer gpa.free(rt_.scores);
        const d: plan.Prefill = .{ .hidden = sh.hidden, .inter = sh.inter, .topk = k, .rows = rows, .experts = sh.experts, .rate = rate, .dec = .mcg, .dtype = .bf16, .defer_map = false, .profile = .glm, .limit = limit };
        const yb = try prefillRun(&h, &g, &dbank, x, rt_.slots, rt_.scores, 0, d);
        defer gpa.free(yb);
        const y: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, yb));
        for (0..rows) |r| for (y[(r * k + topk) * sh.hidden ..][0..sh.hidden]) |v| try expect(@as(u32, @bitCast(v)) == 0xffffffff, "GLM prefill: the shared slot left alone", .{});
        worst = @max(worst, try glmExpect(&bank, rate, x, rt_.slots, k, topk, limit, rows, &.{ 0, 1, 20, 39 }, y));
    }
    std.debug.print("RESULT HIP routed prefill (GLM profile): sampled rows' routed slots within {d:.3} of the float64 tolerance, the shared slot untouched\n", .{worst});
}

/// The dense GEMM path over `rows` rows of x from row `first`: y's bytes.
fn denseGemmRun(h: *const rt.Hip, g: *const Groups, p: *const Proj, x: Input, first: usize, d: plan.Dense) ![]u8 {
    const launches = try plan.denseGemm(d);
    const xsize: usize = if (d.dtype == .f32) 4 else 2;
    var xb = try h.upload(x.bytes[first * p.in * xsize ..][0 .. d.rows * p.in * xsize]);
    defer xb.free();
    const items = try gpa.alloc(i32, 3 * plan.denseTiles(d.rows));
    defer gpa.free(items);
    plan.denseItems(d.rows, items);
    var ib = try h.upload(std.mem.sliceAsBytes(items));
    defer ib.free();
    var xp = try h.alloc(@as(usize, d.rows) * d.in * 2);
    defer xp.free();
    var xs = try h.alloc(@as(usize, d.rows) * (d.in / 128) * 4);
    defer xs.free();
    var z = try h.alloc(@as(usize, d.rows) * d.out * 4);
    defer z.free();
    var y = try h.alloc(@as(usize, d.rows) * d.out * xsize);
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
    bind.set(.items, ib.addr());
    try runLaunches(g, &launches, bind);
    const out = try gpa.alloc(u8, y.len);
    errdefer gpa.free(out);
    try y.read(out);
    return out;
}

test "HIP dense GEMM, rows above 16: within format.project's bar per row; a row's bytes ignore the rows after it" {
    var h = try rt.Hip.open();
    defer h.close();
    var g = try Groups.load(&h);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(0xD6E);
    const rnd = prng.random();
    const cases = [_]DenseCase{
        .{ .in = 512, .out = 256, .n = 64, .dec = .mcg, .dtype = .f32, .defer_map = false },
        .{ .in = 512, .out = 256, .n = 64, .dec = .mul1, .dtype = .f32, .defer_map = true },
        .{ .in = 512, .out = 384, .n = 36, .dec = .{ .codebook = .mul1, .window = .w12 }, .dtype = .f32, .defer_map = false },
        .{ .in = 2560, .out = 640, .n = 80, .dec = .mul1, .dtype = .bf16, .defer_map = true },
    };
    var worst_f32: f64 = 0;
    var worst_bf16: f64 = 0;
    for (cases) |c| {
        const rate = format.Rate.fromN(c.n).?;
        var p = try Proj.init(&h, rnd, c.in, c.out, rate);
        defer p.deinit();
        const max_rows = 200;
        var x = try Input.init(rnd, max_rows * c.in, c.dtype);
        defer x.deinit();
        const tr = try gpa.alloc(f32, c.in);
        defer gpa.free(tr);
        const inner = try gpa.alloc(f32, c.out);
        defer gpa.free(inner);
        const yr = try gpa.alloc(f32, c.out);
        defer gpa.free(yr);
        const got = try gpa.alloc(f32, c.out);
        defer gpa.free(got);
        const ysize: usize = if (c.dtype == .f32) 4 else 2;
        const bar: f64 = if (c.dtype == .f32) 5e-3 else 2.5e-2;
        var y17: ?[]u8 = null;
        defer if (y17) |b| gpa.free(b);
        for ([_]u32{ 17, 40, 129, 200 }) |rows| {
            const d: plan.Dense = .{ .in = c.in, .out = c.out, .rows = rows, .rate = rate, .dec = c.dec, .dtype = c.dtype, .defer_map = c.defer_map };
            const y = try denseGemmRun(&h, &g, &p, x, 0, d);
            defer gpa.free(y);
            for ([_]usize{ 0, 16, rows / 2, rows - 1 }) |r| {
                format.project(x.values[r * c.in ..][0..c.in], p.trellis, p.suh, p.svh, c.in, c.out, rate, c.dec, tr, inner, yr);
                toF32(y[r * c.out * ysize ..][0 .. c.out * ysize], c.dtype, got);
                const e = ref.compare(got, yr);
                try expect(e.max_ratio <= bar, "dense GEMM {d}x{d} n {d} rows {d} row {d}: {e:.3} over the bar {e}", .{ c.in, c.out, c.n, rows, r, e.max_ratio, bar });
                if (c.dtype == .f32) worst_f32 = @max(worst_f32, e.max_ratio) else worst_bf16 = @max(worst_bf16, e.max_ratio);
            }
            if (y17) |first| {
                try expect(std.mem.eql(u8, first, y[0 .. 17 * c.out * ysize]), "dense GEMM {d}x{d}: rows 0-16 of {d} rows differ from the 17-row call", .{ c.in, c.out, rows });
            } else y17 = try gpa.dupe(u8, y[0 .. 17 * c.out * ysize]);
        }
    }
    std.debug.print("RESULT HIP dense GEMM (17-200 rows): sampled rows within {e:.3} (f32, bar 5e-3) and {e:.3} (bf16, bar 2.5e-2) of format.project; rows 0-16 byte-equal\n", .{ worst_f32, worst_bf16 });
}

test "HIP routed decode, form 9: the bytes of form 3 at rows 1-5, 8 and 16, both profiles, every codebook form" {
    var h = try rt.Hip.open();
    defer h.close();
    var g = try Groups.load(&h);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(0xF09);
    const rnd = prng.random();
    const Case = struct { c: DecodeCase, profile: plan.Profile };
    const cases = [_]Case{
        .{ .c = .{ .hidden = 512, .inter = 256, .experts = 12, .topk = 4, .n = 64, .dec = .mcg, .dtype = .f32, .defer_map = false }, .profile = .exl3 },
        .{ .c = .{ .hidden = 512, .inter = 256, .experts = 12, .topk = 4, .n = 64, .dec = .mul1, .dtype = .f32, .defer_map = true }, .profile = .exl3 },
        .{ .c = .{ .hidden = 512, .inter = 256, .experts = 12, .topk = 4, .n = 36, .dec = .mul1, .dtype = .bf16, .defer_map = false }, .profile = .exl3 },
        .{ .c = .{ .hidden = 2560, .inter = 640, .experts = 16, .topk = 10, .n = 48, .dec = .mul1, .dtype = .bf16, .defer_map = true }, .profile = .exl3 },
        .{ .c = .{ .hidden = 4096, .inter = 2048, .experts = 9, .topk = 9, .n = 64, .dec = .mcg, .dtype = .bf16, .defer_map = false }, .profile = .glm },
    };
    var compared: usize = 0;
    var with_gu8: usize = 0;
    for (cases) |cs| {
        const c = cs.c;
        const rate = format.Rate.fromN(c.n).?;
        var bank = try ref.synthBank(gpa, rnd, c.experts, c.hidden, c.inter, rate);
        defer bank.deinit(gpa);
        var dbank = try DevBank.init(&h, &bank);
        defer dbank.deinit();
        const max_rows = 16;
        var x = try Input.init(rnd, max_rows * c.hidden, c.dtype);
        defer x.deinit();
        const glm = cs.profile == .glm;
        const rt_ = try routing(rnd, max_rows, if (glm) c.topk - 1 else c.topk, c.experts, glm);
        defer gpa.free(rt_.slots);
        defer gpa.free(rt_.scores);
        for ([_]u32{ 1, 2, 3, 4, 5, 8, 16 }) |rows| {
            const d: plan.Decode = .{ .hidden = c.hidden, .inter = c.inter, .topk = c.topk, .rows = rows, .experts = c.experts, .rate = rate, .dec = c.dec, .dtype = c.dtype, .defer_map = c.defer_map, .profile = cs.profile, .limit = if (glm) 7.0 else 0 };
            const y3 = try decodeRunForm(&h, &g, &dbank, x, rt_.slots, rt_.scores, 0, d, 3);
            defer gpa.free(y3);
            const y9 = try decodeRunForm(&h, &g, &dbank, x, rt_.slots, rt_.scores, 0, d, 9);
            defer gpa.free(y9);
            try expect(std.mem.eql(u8, y3, y9), "form 9 {d}x{d} {s} {s} rows {d}: bytes differ from form 3", .{ c.hidden, c.inter, @tagName(cs.profile), @tagName(c.dec.codebook), rows });
            compared += 1;
            if (plan.gu8Admitted(d)) with_gu8 += 1;
        }
    }
    std.debug.print("RESULT HIP routed decode form 9: {d} calls byte-identical to form 3 ({d} of them through gu8)\n", .{ compared, with_gu8 });
}

// ---- DEC-ONE ------------------------------------------------------------------------------------------------------

test "HIP DEC-ONE: the bytes of the dense chain at rows 1-9, 11 and 16; a captured graph replayed twice gives them again" {
    var h = try rt.Hip.open();
    defer h.close();
    var g = try Groups.load(&h);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(0xDEC1);
    const rnd = prng.random();
    const cases = [_]DenseCase{
        .{ .in = 512, .out = 256, .n = 64, .dec = .mcg, .dtype = .f32, .defer_map = false },
        .{ .in = 512, .out = 256, .n = 64, .dec = .mul1, .dtype = .f32, .defer_map = true },
        .{ .in = 512, .out = 384, .n = 36, .dec = .mul1, .dtype = .bf16, .defer_map = false },
        .{ .in = 2560, .out = 640, .n = 80, .dec = .mul1, .dtype = .bf16, .defer_map = true },
        .{ .in = 4096, .out = 2048, .n = 64, .dec = .{ .codebook = .mcg, .window = .w12 }, .dtype = .bf16, .defer_map = false },
    };
    var compared: usize = 0;
    for (cases) |c| {
        const rate = format.Rate.fromN(c.n).?;
        var p = try Proj.init(&h, rnd, c.in, c.out, rate);
        defer p.deinit();
        var x = try Input.init(rnd, 16 * c.in, c.dtype);
        defer x.deinit();
        const xsize: usize = if (c.dtype == .f32) 4 else 2;
        for ([_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16 }) |rows| {
            const d: plan.Dense = .{ .in = c.in, .out = c.out, .rows = rows, .rate = rate, .dec = c.dec, .dtype = c.dtype, .defer_map = c.defer_map, .lean = false };
            const chain = try denseRun(&h, &g, &p, x, 0, d);
            defer gpa.free(chain);
            const launches = try plan.denseDecOne(d);
            var xb = try h.upload(x.bytes[0 .. rows * c.in * xsize]);
            defer xb.free();
            var xp = try h.alloc(d.xpBytes());
            defer xp.free();
            var xs = try h.alloc(d.xsBytes());
            defer xs.free();
            var z = try h.alloc(d.zBytes());
            defer z.free();
            var y = try h.alloc(rows * c.out * xsize);
            defer y.free();
            var cnt = try h.alloc(@as(usize, plan.decOneCounters(d)) * 4);
            defer cnt.free();
            try cnt.fill(0x5a); // prep_sc zeroes them
            try y.fill(0x7f); // an element nothing writes reads as a NaN pattern, never as a reused buffer's bytes
            var bind = std.EnumArray(plan.Role, u64).initFill(0);
            for ([_]plan.Role{ .x, .suh, .svh, .trellis, .xp, .xs, .z, .y, .cnt }, [_]u64{ xb.addr(), p.dsuh.addr(), p.dsvh.addr(), p.dt.addr(), xp.addr(), xs.addr(), z.addr(), y.addr(), cnt.addr() }) |r, a| bind.set(r, a);
            try runLaunches(&g, &launches, bind);
            const got = try gpa.alloc(u8, y.len);
            defer gpa.free(got);
            try y.read(got);
            if (!std.mem.eql(u8, got, chain)) {
                var nd: usize = 0;
                var first: usize = 0;
                for (got, chain, 0..) |a_, b_, i| if (a_ != b_) {
                    if (nd == 0) first = i;
                    nd += 1;
                };
                // a second run: does DEC-ONE agree with itself?
                const again = try gpa.alloc(u8, y.len);
                defer gpa.free(again);
                try runLaunches(&g, &launches, bind);
                try y.read(again);
                std.debug.print("DEC-ONE {d}x{d} {s} rows {d}: {d} of {d} bytes differ from the chain (first at byte {d}, row {d}); a second run {s} the first\n", .{ c.in, c.out, @tagName(c.dec.codebook), rows, nd, got.len, first, first / (c.out * xsize), if (std.mem.eql(u8, again, got)) "equals" else "differs from" });
                // the chain again: is it the same each run?
                const chain2 = try denseRun(&h, &g, &p, x, 0, d);
                defer gpa.free(chain2);
                std.debug.print("  the chain run again {s} the first chain run\n", .{if (std.mem.eql(u8, chain2, chain)) "equals" else "differs from"});
                if (c.dtype == .f32) {
                    const gf: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, got));
                    const cf: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, chain));
                    var shown: usize = 0;
                    for (gf, cf, 0..) |a_, b_, i| if (@as(u32, @bitCast(a_)) != @as(u32, @bitCast(b_)) and shown < 8) {
                        std.debug.print("  row {d} column {d}: DEC-ONE {e} ({x}), chain {e} ({x})\n", .{ i / c.out, i % c.out, a_, @as(u32, @bitCast(a_)), b_, @as(u32, @bitCast(b_)) });
                        shown += 1;
                    };
                }
                return error.TestUnexpectedResult;
            }
            compared += 1;
            // a captured graph, replayed twice: the counters start at 0 each time (prep_sc), so the same bytes
            if (rows == 1 or rows == 9) {
                var s = try h.stream();
                defer s.destroy();
                try s.beginCapture();
                runLaunchesOn(&g, s, &launches, bind) catch |err| {
                    _ = s.endCapture() catch {};
                    return err;
                };
                var graph = try s.endCapture();
                defer graph.destroy();
                for (0..2) |_| {
                    try y.fill(0);
                    try graph.launch(s);
                    try s.sync();
                    try y.read(got);
                    try expect(std.mem.eql(u8, got, chain), "DEC-ONE {d}x{d} rows {d}: a graph replay differs from the chain", .{ c.in, c.out, rows });
                }
            }
        }
    }
    std.debug.print("RESULT HIP DEC-ONE: {d} calls byte-identical to the dense chain; graph replays (twice) byte-identical\n", .{compared});
}
