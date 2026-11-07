//! The dense decode GEMV's schedules (DEC-ONE: `Config.dense_variant`, `Config.dec_bt`) at GLM-5.3-Flash's dense shapes.
//! - CPU (always): the variant texts are their pieces (ku's tile loop in place of today's, nothing else), each variant
//!   has kernel names of its own, and today's texts are today's pieces.
//! - GPU (TURBOEXL3_GPU=1, else skipped): every variant and every output-tile count against the served configuration,
//!   byte for byte, at each sc_3.10bpw dense shape (`served`: the K6 trunk and head, the MTP layer's K4 linears), decode
//!   rows 1 to 4, bf16 x (the served activations) and f32. Each configuration's engines are freed before the next is
//!   made (`root.releaseEngines`), so each is built from its configuration. TURBOEXL3_GLM_DEC_K (a comma list of K)
//!   narrows the shapes. Lines: TURBOEXL3_GLM_DEC_BYTES per shape, rows, dtype and schedule, and a total.
//! - TURBOEXL3_GLM_TIME=1 (with TURBOEXL3_GPU=1, else skipped): the trunk's dense linears at a verify's rows
//!   (TURBOEXL3_GLM_DEC_ROWS, default 3: 1 + the pinned depth 2) as one forward's work, each class a chain over as many
//!   distinct weights as a forward reads (34 KDA layers' qkv then o, 11 DSA layers' q_a, q_b, kv_a, indexer wq_b and o, 42
//!   shared experts' gate, up and down, the 3 dense layers' MLP, the head), held in device memory as a verify reads them
//!   (5.9 GB, past every cache), each linear's input the previous one's output where the forward chains them; per
//!   schedule (today's, each variant, each output-tile count), the best of `time_rounds` evaluations of each chain after
//!   one that builds its kernels, and its trellis bytes over that time. A microbenchmark: it steers, the standard test
//!   decides. Lines: TURBOEXL3_GLM_DEC_TIME per class and schedule, TURBOEXL3_GLM_DEC_TIME_TOTAL per schedule.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const format = @import("../../core/format.zig");
const dense = @import("../../core/dense.zig");
const config = @import("../../core/config.zig");
const root = @import("../../root.zig");
const routed = @import("routed_test.zig");

const testing = std.testing;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// A dense linear of sc_3.10bpw at its stored K.
pub const DecCase = struct { name: []const u8, in: u32, out: u32, k: u32 };

/// sc_3.10bpw's dense linears (glm53f_test.zig's `cases` at the pack's rates): the trunk's attention, shared expert, dense
/// MLP and head at K6, the MTP layer's attention, shared expert and `eh_proj` at K4.
pub const served = [_]DecCase{
    .{ .name = "kda.qkv_proj", .in = 4096, .out = 24576, .k = 6 },
    .{ .name = "kda.o_proj", .in = 8192, .out = 4096, .k = 6 },
    .{ .name = "dsa.q_a_proj", .in = 4096, .out = 1536, .k = 6 },
    .{ .name = "dsa.q_b_proj", .in = 1536, .out = 16384, .k = 6 },
    .{ .name = "dsa.kv_a_proj_with_mqa", .in = 4096, .out = 512, .k = 6 },
    .{ .name = "dsa.o_proj", .in = 16384, .out = 4096, .k = 6 },
    .{ .name = "indexer.wq_b", .in = 1536, .out = 4096, .k = 6 },
    .{ .name = "shared.gate_proj/up_proj", .in = 4096, .out = 2048, .k = 6 },
    .{ .name = "shared.down_proj", .in = 2048, .out = 4096, .k = 6 },
    .{ .name = "dense.gate_proj/up_proj", .in = 4096, .out = 12288, .k = 6 },
    .{ .name = "dense.down_proj", .in = 12288, .out = 4096, .k = 6 },
    .{ .name = "lm_head", .in = 4096, .out = 154880, .k = 6 },
    .{ .name = "mtp.dsa.q_a_proj", .in = 4096, .out = 1536, .k = 4 },
    .{ .name = "mtp.dsa.q_b_proj", .in = 1536, .out = 16384, .k = 4 },
    .{ .name = "mtp.dsa.kv_a_proj_with_mqa", .in = 4096, .out = 512, .k = 4 },
    .{ .name = "mtp.dsa.o_proj", .in = 16384, .out = 4096, .k = 4 },
    .{ .name = "mtp.indexer.wq_b", .in = 1536, .out = 4096, .k = 4 },
    .{ .name = "mtp.shared.gate_proj/up_proj", .in = 4096, .out = 2048, .k = 4 },
    .{ .name = "mtp.shared.down_proj", .in = 2048, .out = 4096, .k = 4 },
    .{ .name = "mtp.eh_proj", .in = 8192, .out = 4096, .k = 4 },
};

/// A schedule: a DEC-ONE variant at the default output tiles, or today's text at one output-tile count for every decode
/// row count (`bt` 0: the default).
pub const Sched = struct { variant: u8 = 0, bt: u8 = 0 };

/// Every schedule the switch can name: each variant past today's, then today's text at each output-tile count.
pub const scheds = blk: {
    var s: [dense.dec_variants.len - 1 + 4]Sched = undefined;
    for (1..dense.dec_variants.len) |v| s[v - 1] = .{ .variant = v };
    for ([_]u8{ 1, 2, 4, 8 }, 0..) |b, i| s[dense.dec_variants.len - 1 + i] = .{ .bt = b };
    break :blk s;
};

fn schedName(sc: Sched, buf: []u8) []const u8 {
    if (sc.variant == 0 and sc.bt == 0) return "served";
    if (sc.bt == 0) return dense.dec_variants[sc.variant].name;
    return std.fmt.bufPrint(buf, "bt{d}", .{sc.bt}) catch "bt?";
}

/// turboexl3's configuration with `sc` (every other field as it was), its engines freed so the next call builds them.
fn configureSched(base: config.Config, sc: Sched) !void {
    var c = base;
    c.dense_variant = sc.variant;
    c.dec_bt = .{ 0, sc.bt, sc.bt, sc.bt, sc.bt };
    try config.configure(c);
    root.releaseEngines();
}

test "turboexl3 glm53f dense decode texts: each variant is today's texts with ku's tile loop in place of today's and its own kernel names; today's texts are today's pieces" {
    // today's texts, piece by piece (the split left their bytes)
    try testing.expectEqualStrings(dense.dec_gemv0_head ++ dense.dec_gemv_loop ++ dense.dec_gemv0_x, dense.dec_gemv0);
    try testing.expectEqualStrings(dense.dec_gemv1_head ++ dense.dec_gemv_loop ++ dense.dec_gemv1_x, dense.dec_gemv1);
    try testing.expectEqualStrings(dense.dec_gemv_arrive_2a ++ dense.dec_gemv_arrive_2b ++ dense.dec_gemv_arrive_2c, dense.dec_gemv_arrive_2);
    // ku: the tile loop and the tile's words swapped (no tile address, no load in the output-tile loop), the FMAs' piece
    // once, nothing else
    for ([_][:0]const u8{ dense.text_dec0_ku, dense.text_dec1_ku }, [_][:0]const u8{ dense.text_dec0, dense.text_dec1 }) |ku, base| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, ku, dense.dec_gemv_arrive_2b));
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, ku, dense.dec_tile));
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, ku, dense.dec_gemv_arrive_2a));
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, ku, dense.dec_gemv_loop));
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, ku, dense.dec_ku_loop));
        try testing.expectEqual(base.len - dense.dec_gemv_loop.len - dense.dec_tile.len - dense.dec_gemv_arrive_2a.len + dense.dec_ku_loop.len + dense.dec_ku_ww.len + dense.dec_ku_close.len, ku.len);
    }
    // names: today's unchanged, every variant's its own
    try testing.expectEqualStrings("turboexl3_dense_dec0", dense.dec_variants[0].dec0);
    try testing.expectEqualStrings("turboexl3_dense_dec1", dense.dec_variants[0].dec1);
    try testing.expectEqualStrings(dense.text_dec0, dense.dec_variants[0].text0);
    for (dense.dec_variants, 0..) |a, i| for (dense.dec_variants[i + 1 ..]) |b| {
        try testing.expect(!std.mem.eql(u8, a.dec0, b.dec0) and !std.mem.eql(u8, a.dec1, b.dec1) and !std.mem.eql(u8, a.name, b.name));
    };
    for (dense.dec_variants, 0..) |v, i| try testing.expectEqual(@as(?u8, @intCast(i)), dense.decVariantOf(v.name));
    try testing.expectEqual(@as(?u8, null), dense.decVariantOf("sh"));
    // every schedule the test and the switch name is a configuration turboexl3 serves
    for (scheds) |sc| try testing.expect((config.Config{ .dense_variant = sc.variant, .dec_bt = .{ 0, sc.bt, sc.bt, sc.bt, sc.bt } }).valid());
}

/// `x`'s 32-bit words (an f32 array's bits, or a bf16 array's bits through its exact f32 widening).
fn bitsOf(g: *G, x: mlx.mlx_array, out: []f32) ![]const u32 {
    const f = if (mlx.mlx_array_dtype(x) == .float32) x else try g.astype(x, .float32);
    _ = try g.hostF32(f, out);
    const p: [*]const u32 = @ptrCast(out.ptr);
    return p[0..out.len];
}

/// A K-rate trellis of random words (every pattern a codeword), suh in [0.5, 1.5], svh in [0.5, 1.5] / sqrt(in).
fn randomProj(g: *G, a: std.mem.Allocator, rnd: std.Random, in: u32, out: u32, k: u32) !root.Proj {
    const n = 16 * k;
    const tw = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(tw);
    rnd.bytes(std.mem.sliceAsBytes(tw));
    return projOf(g, a, rnd, tw, in, out, n);
}

fn projOf(g: *G, a: std.mem.Allocator, rnd: std.Random, tw: []const u16, in: u32, out: u32, n: u32) !root.Proj {
    const su = try a.alloc(u16, in);
    defer a.free(su);
    for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, out);
    defer a.free(sv);
    for (sv) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    return .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(tw), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(out)}, .float16),
    };
}

test "turboexl3 gpu glm53f dense decode schedules: every DEC-ONE variant and output-tile count gives the served configuration's y byte for byte at each sc_3.10bpw dense shape, decode rows 1 to 4, bf16 and f32 x (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    const base = config.get().*;
    defer {
        config.configure(base) catch unreachable;
        root.releaseEngines();
    }
    var prng = std.Random.DefaultPrng.init(0xDEC0E);
    const rnd = prng.random();
    var kbuf: [8]u32 = undefined;
    const ks = routed.envList("TURBOEXL3_GLM_DEC_K", &.{ 4, 6 }, &kbuf);
    var failed = false;
    var lines: usize = 0;
    var words: usize = 0;
    for (served) |c| {
        if (std.mem.indexOfScalar(u32, ks, c.k) == null) continue;
        const mark = g.mark();
        defer g.resetTo(mark);
        const p = try randomProj(&g, a, rnd, c.in, c.out, c.k);
        try g.evalAll(&.{ p.trellis, p.suh, p.svh });
        for (1..5) |rows| {
            const xh = try a.alloc(f32, rows * c.in);
            defer a.free(xh);
            for (xh) |*v| v.* = routed.bfExact(rnd.floatNorm(f32));
            const xf = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(c.in) }, .float32);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const rm = g.mark();
                defer g.resetTo(rm);
                const x = if (dt == .float32) xf else try g.astype(xf, dt);
                try g.evalAll(&.{x});
                const n = rows * c.out;
                const want_buf = try a.alloc(f32, n);
                defer a.free(want_buf);
                const got_buf = try a.alloc(f32, n);
                defer a.free(got_buf);
                try configureSched(base, .{});
                const want = try bitsOf(&g, try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1)), want_buf);
                for (scheds) |sc| {
                    try configureSched(base, sc);
                    const got = try bitsOf(&g, try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1)), got_buf);
                    var differing: usize = 0;
                    for (want, got) |w, v| {
                        if (w != v) differing += 1;
                    }
                    const ok = differing == 0;
                    if (!ok) failed = true;
                    lines += 1;
                    words += n;
                    var nb: [16]u8 = undefined;
                    std.debug.print("TURBOEXL3_GLM_DEC_BYTES name={s} k={d} rows={d} x={s} sched={s} differing={d}/{d} {s}\n", .{ c.name, c.k, rows, @tagName(dt), schedName(sc, &nb), differing, n, if (ok) "PASS" else "FAIL" });
                }
            }
        }
    }
    std.debug.print("TURBOEXL3_GLM_DEC_BYTES lines={d} words={d} {s}\n", .{ lines, words, if (failed or lines == 0) "FAIL" else "PASS" });
    try testing.expect(!failed and lines > 0);
}

/// The best of this many timed evaluations of each chain (after one that builds its kernels).
pub const time_rounds: usize = 3;

/// One forward's trunk linears, in the classes the chains run: per layer, the linears in forward order.
const Class = struct { name: []const u8, layers: u32, shapes: []const [2]u32 };
const classes = [_]Class{
    .{ .name = "kda", .layers = 34, .shapes = &.{ .{ 4096, 24576 }, .{ 8192, 4096 } } },
    .{ .name = "dsa", .layers = 11, .shapes = &.{ .{ 4096, 1536 }, .{ 1536, 16384 }, .{ 4096, 512 }, .{ 1536, 4096 }, .{ 16384, 4096 } } },
    .{ .name = "shared", .layers = 42, .shapes = &.{ .{ 4096, 2048 }, .{ 4096, 2048 }, .{ 2048, 4096 } } },
    .{ .name = "dense", .layers = 3, .shapes = &.{ .{ 4096, 12288 }, .{ 4096, 12288 }, .{ 12288, 4096 } } },
    .{ .name = "head", .layers = 1, .shapes = &.{.{ 4096, 154880 }} },
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// One class's chain over its layers' projections (`ps`, layer-major in `Class.shapes` order) from `x` [rows, 4096] bf16:
/// the outputs to evaluate (the chain's last x and every side branch).
fn chain(g: *G, s: mlx.mlx_stream, cls: Class, ps: []const root.Proj, x0: mlx.mlx_array, rows: c_int, outs: *std.ArrayList(mlx.mlx_array), a: std.mem.Allocator) !void {
    var x = x0;
    const k = cls.shapes.len;
    for (0..cls.layers) |l| {
        const p = ps[l * k ..][0..k];
        if (std.mem.eql(u8, cls.name, "kda")) {
            // qkv, its first 8192 columns (the value heads' width) into o
            const qkv = try g.adopt(try root.linear(s, x, p[0].trellis, p[0].suh, p[0].svh, .mul1));
            const v = try g.slice(qkv, &.{ 0, 0 }, &.{ rows, 8192 }, &.{ 1, 1 });
            x = try g.adopt(try root.linear(s, v, p[1].trellis, p[1].suh, p[1].svh, .mul1));
        } else if (std.mem.eql(u8, cls.name, "dsa")) {
            // q_a, then q_b and the indexer's wq_b from it; kv_a beside them from x; o from q_b
            const qa = try g.adopt(try root.linear(s, x, p[0].trellis, p[0].suh, p[0].svh, .mul1));
            const q = try g.adopt(try root.linear(s, qa, p[1].trellis, p[1].suh, p[1].svh, .mul1));
            try outs.append(a, try g.adopt(try root.linear(s, x, p[2].trellis, p[2].suh, p[2].svh, .mul1)));
            try outs.append(a, try g.adopt(try root.linear(s, qa, p[3].trellis, p[3].suh, p[3].svh, .mul1)));
            x = try g.adopt(try root.linear(s, q, p[4].trellis, p[4].suh, p[4].svh, .mul1));
        } else if (std.mem.eql(u8, cls.name, "head")) {
            try outs.append(a, try g.adopt(try root.linear(s, x, p[0].trellis, p[0].suh, p[0].svh, .mul1)));
        } else {
            // a SwiGLU MLP's linears: gate and up from x, down from their product
            const gate = try g.adopt(try root.linear(s, x, p[0].trellis, p[0].suh, p[0].svh, .mul1));
            const up = try g.adopt(try root.linear(s, x, p[1].trellis, p[1].suh, p[1].svh, .mul1));
            x = try g.adopt(try root.linear(s, try g.mul(gate, up), p[2].trellis, p[2].suh, p[2].svh, .mul1));
        }
    }
    try outs.append(a, x);
}

test "turboexl3 gpu glm53f dense decode timing: each DEC-ONE schedule over one forward's trunk linears at a verify's rows, each class a chain over distinct weights in device memory (TURBOEXL3_GLM_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_GLM_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer _ = mlx.mlx_clear_cache();
    const base = config.get().*;
    defer {
        config.configure(base) catch unreachable;
        root.releaseEngines();
    }
    var prng = std.Random.DefaultPrng.init(0x7DEC);
    const rnd = prng.random();
    var rbuf: [1]u32 = undefined;
    const rows_l = routed.envList("TURBOEXL3_GLM_DEC_ROWS", &.{3}, &rbuf);
    const rows: c_int = @intCast(rows_l[0]);
    const k: u32 = 6;
    const n: u32 = 16 * k;
    // a pool of random words each trellis copies from (every pattern a codeword; the timing does not depend on them):
    // each projection its own array, so a forward's 5.9 GB stream from device memory as a verify's do
    const pool = try a.alloc(u16, 64 << 20);
    defer a.free(pool);
    rnd.bytes(std.mem.sliceAsBytes(pool));
    var projs: [classes.len][]root.Proj = undefined;
    var class_bytes: [classes.len]f64 = undefined;
    for (classes, 0..) |cls, ci_| {
        projs[ci_] = try a.alloc(root.Proj, cls.layers * cls.shapes.len);
        var bytes: f64 = 0;
        for (0..cls.layers) |l| for (cls.shapes, 0..) |sh, j| {
            const words = (sh[0] / 16) * (sh[1] / 16) * n;
            const tw = try a.alloc(u16, words);
            defer a.free(tw);
            var off: usize = (l * 7919 + j * 104729) % pool.len;
            var done: usize = 0;
            while (done < words) {
                const take = @min(words - done, pool.len - off);
                @memcpy(tw[done..][0..take], pool[off..][0..take]);
                done += take;
                off = 0;
            }
            const p = try projOf(&g, a, rnd, tw, sh[0], sh[1], n);
            try g.evalAll(&.{ p.trellis, p.suh, p.svh });
            projs[ci_][l * cls.shapes.len + j] = p;
            bytes += @floatFromInt(words * 2);
        };
        class_bytes[ci_] = bytes;
    }
    defer for (projs) |p| a.free(p);
    const xh = try a.alloc(f32, @as(usize, @intCast(rows)) * 4096);
    defer a.free(xh);
    for (xh) |*v| v.* = routed.bfExact(rnd.floatNorm(f32));
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ rows, 4096 }, .float32), .bfloat16);
    try g.evalAll(&.{x});
    var all: [1 + scheds.len]Sched = undefined;
    all[0] = .{};
    for (scheds, 1..) |sc, i| all[i] = sc;
    for (all) |sc| {
        try configureSched(base, sc);
        var nb: [16]u8 = undefined;
        const name = schedName(sc, &nb);
        var total_ms: f64 = 0;
        var total_bytes: f64 = 0;
        for (classes, 0..) |cls, ci_| {
            var best: f64 = std.math.inf(f64);
            for (0..time_rounds + 1) |round| {
                const m = g.mark();
                defer g.resetTo(m);
                var outs: std.ArrayList(mlx.mlx_array) = .empty;
                defer outs.deinit(a);
                // the graph is lazy: built first, then its evaluation (the encode and the GPU's run) timed
                try chain(&g, s, cls, projs[ci_], x, rows, &outs, a);
                const t0 = nowNs();
                try g.evalAll(outs.items);
                const ms = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
                if (round > 0) best = @min(best, ms);
            }
            total_ms += best;
            total_bytes += class_bytes[ci_];
            std.debug.print("TURBOEXL3_GLM_DEC_TIME class={s} sched={s} rows={d} linears={d} ms={d:.3} gb={d:.3} gbps={d:.1}\n", .{ cls.name, name, rows, cls.layers * cls.shapes.len, best, class_bytes[ci_] / 1e9, class_bytes[ci_] / 1e6 / best });
        }
        std.debug.print("TURBOEXL3_GLM_DEC_TIME_TOTAL sched={s} rows={d} ms={d:.3} gb={d:.3} gbps={d:.1}\n", .{ name, rows, total_ms, total_bytes / 1e9, total_bytes / 1e6 / total_ms });
    }
}
