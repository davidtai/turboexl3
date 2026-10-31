//! DeepSeek-V4.1's trunk at decode rows: the dense decode GEMV's launches. The grouped linear (`wo_a`) on dense's
//! multi-row GEMV (`grouped.text_gemv_d`): its texts are dense's GEMV statements (CPU), and it gives the routed down
//! GEMV's bytes at every decode row count and launch shape (TURBOEXL3_GPU=1). TURBOEXL3_DS_GEMV=1 (with
//! TURBOEXL3_GPU=1) sweeps the dense decode GEMV's launch shapes on deepseek_test.zig's 40-layer trunk chain (asserting
//! bytes only).
//! Lines: TURBOEXL3_DS_GROUPED_BYTES, TURBOEXL3_DS_GEMV*.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ks_launch = sdk.kernel_launch;
const support = @import("test_support.zig");
const G = support.Graph;
const format = @import("format.zig");
const root = @import("root.zig");
const dense = @import("dense.zig");
const decode = @import("decode.zig");
const grouped = @import("grouped.zig");
const ds = @import("deepseek_test.zig");

const testing = std.testing;

fn has(text: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, text, needle) != null;
}

test "turboexl3 deepseek gemv: the grouped dense-GEMV texts are dense's GEMV statements from the input loads to the fold, read every operand and constant they bind" {
    // dense's own pieces, verbatim and in dense's order
    try testing.expect(std.mem.eql(u8, grouped.text_gemv_d, grouped.gemv_head ++ dense.dgemv_x ++ dense.dgemv_2 ++ dense.dgemv_m ++ dense.dgemv_fold ++ grouped.gemv_store));
    try testing.expect(std.mem.eql(u8, grouped.text_gemv_dl, grouped.gemv_head ++ dense.dgemv_2 ++ dense.dgemv_j ++ dense.dgemv_fold ++ grouped.gemv_store));
    // dense's GEMV texts end in the fold, then the store
    try testing.expect(std.mem.endsWith(u8, dense.text_dgemv, dense.dgemv_fold ++ dense.dgemv_store));
    try testing.expect(std.mem.endsWith(u8, dense.text_dgemv_lean, dense.dgemv_fold ++ dense.dgemv_store));
    for ([_][]const u8{ dense.dgemv_x, dense.dgemv_2, dense.dgemv_m, dense.dgemv_fold }) |piece| try testing.expect(has(dense.text_dgemv, piece));
    for ([_][]const u8{ dense.dgemv_2, dense.dgemv_j, dense.dgemv_fold }) |piece| try testing.expect(has(dense.text_dgemv_lean, piece));
    // the head's K partition and the store's sum are text_dgemv's statements; the store's row-slot is the routed planes'
    const pin = comptime support.pins(@embedFile("kernels/test/pins/grouped_gemv_d.metal"));
    try testing.expect(has(dense.text_dgemv, pin[0]) and has(grouped.gemv_head, pin[0]));
    try testing.expect(has(dense.text_dgemv, pin[1]) and has(grouped.gemv_head, pin[1]));
    try testing.expect(has(dense.dgemv_store, pin[2]) and has(grouped.gemv_store, pin[2]));
    try testing.expect(has(grouped.gemv_store, pin[3]));
    try testing.expect(has(decode.text_dfin3, pin[4]));
    for ([_][:0]const u8{ grouped.text_gemv_d, grouped.text_gemv_dl }) |txt| {
        for (grouped.gemv_d_in ++ grouped.gemv_d_out) |n| try testing.expect(has(txt, std.mem.span(n)));
        for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "IN", "OUT", "KS", "BT", "RB", "ROWS", "GROUPS", "DEFER" }) |n| try testing.expect(has(txt, n));
        // MLX adds a shape / strides / ndim buffer for an input name followed by those suffixes
        for (grouped.gemv_d_in ++ grouped.gemv_d_out) |n| for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| {
            var buf: [16]u8 = undefined;
            try testing.expect(!has(txt, std.fmt.bufPrint(&buf, "{s}{s}", .{ std.mem.span(n), suf }) catch unreachable));
        };
    }
}

test "turboexl3 deepseek gemv: lean chunks take BT 2 in launches of 2048 threadgroups or more at BT 1 (DeepSeek-V4.1's wq_b and wo_a), btFor's elsewhere" {
    const D = dense.Dense;
    for (1..9) |i| {
        const rb: u32 = @intCast(i);
        try testing.expectEqual(D.btFor(rb), D.btLaunch(rb, D.lean_bt2_threadgroups - 1));
        try testing.expectEqual(if (rb > D.max_rb) @as(u32, 2) else D.btFor(rb), D.btLaunch(rb, D.lean_bt2_threadgroups));
    }
    // the trunk's classes at one lean chunk (rows 6 to 8): threadgroups at BT 1 = output tiles x K slices (x slices)
    const Want = struct { c: GemvClass, tgs: u32, bt: u32 };
    const want = [_]Want{
        .{ .c = .wq_a, .tgs = 640, .bt = 1 },  .{ .c = .wq_b, .tgs = 2048, .bt = 2 }, .{ .c = .wkv, .tgs = 640, .bt = 1 },
        .{ .c = .iwq_b, .tgs = 1024, .bt = 1 }, .{ .c = .wo_b, .tgs = 640, .bt = 1 },  .{ .c = .w13, .tgs = 720, .bt = 1 },
        .{ .c = .w2, .tgs = 640, .bt = 1 },    .{ .c = .wo_a, .tgs = 8192, .bt = 2 },
    };
    for (want) |w| {
        const io = classShape(w.c);
        const tgs = if (w.c == .wo_a) (io[1] / 16) * ds.wo_a.groups * grouped.planFor(io[0], io[1]).ks else (io[1] / 16) * D.ksFor(io[0], io[1]);
        try testing.expectEqual(w.tgs, tgs);
        for (6..9) |rows| try testing.expectEqual(w.bt, D.btLaunch(D.rbCap(@intCast(rows), D.lean_rb), tgs));
        if (w.c == .wo_a) for (1..17) |rows| {
            const rb = D.rbCap(@intCast(rows), D.lean_rb);
            try testing.expectEqual(if (rb > D.max_rb) @as(u32, 2) else D.btFor(rb), groupedBt(@intCast(rows)));
        };
    }
}

// The grouped linear's GEMV on dense's multi-row GEMV (`grouped.text_gemv_d`, the default) against the routed down GEMV it
// replaces (`Grouped.dense_gemv` false), byte for byte: wo_a's grouping (8 slices of 4096 -> 1024, K6, K split 16) and a
// smaller one (4 slices of 1024 -> 512 at K4), every decode row count 1..16 (one chunk to 8 rows, the lean text past 5,
// two chunks past 8), bf16 and f32 x, at the served tiles per threadgroup and at 1, 2, 4 and 8. The planes a row-slot sums
// do not depend on the launch, so every arm must give the routed GEMV's bytes.
test "turboexl3 gpu deepseek gemv: the grouped linear's dense GEMV gives the routed GEMV's bytes, rows 1..16, at any tiles per threadgroup (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const Geo = struct { groups: u32, in: u32, out: u32, n: u32 };
    const geos = [_]Geo{ .{ .groups = 8, .in = 4096, .out = 1024, .n = ds.N }, .{ .groups = 4, .in = 1024, .out = 512, .n = 64 } };
    var failed = false;
    var checks: usize = 0;
    for (geos) |geo| {
        var prng = std.Random.DefaultPrng.init(0x6D0A ^ geo.in ^ (geo.out << 12));
        const rnd = prng.random();
        const mark = g.mark();
        defer g.resetTo(mark);
        const p = try ds.randProj(a, &g, rnd, geo.groups, geo.in, geo.out, geo.n);
        const max_rows: u32 = 16;
        const xs = try a.alloc(f32, max_rows * geo.groups * geo.in);
        defer a.free(xs);
        for (xs) |*v| v.* = ds.bf16Exact(rnd.floatNorm(f32));
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
            const sh: grouped.Shape = .{ .groups = geo.groups, .in = geo.in, .out = geo.out, .rate = format.Rate.fromN(geo.n).?, .dec = .mul1, .out_dtype = dt };
            var routed = try grouped.Grouped.init(sh);
            defer routed.deinit();
            routed.dense_gemv = false;
            var dn = try grouped.Grouped.init(sh);
            defer dn.deinit();
            for (1..max_rows + 1) |ri| {
                const rows: u32 = @intCast(ri);
                const rm = g.mark();
                defer g.resetTo(rm);
                const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. rows * geo.groups * geo.in]), &.{ ds.ci(rows), ds.ci(geo.groups * geo.in) }, .float32), dt);
                const n_out = rows * geo.groups * geo.out;
                const want = try a.alloc(f32, n_out);
                defer a.free(want);
                const got = try a.alloc(f32, n_out);
                defer a.free(got);
                _ = try g.hostF32(try g.astype(try g.adopt(try routed.decodeRows(s, x, p.trellis, p.suh, p.svh, rows)), .float32), want);
                for ([_]u8{ 0, 1, 2, 4, 8 }) |bt| {
                    // a lane's 2 BT RB accumulators: up to 64 floats (past that the compile, not the bytes, is the cost)
                    if (2 * @as(u32, bt) * dense.Dense.rbCap(rows, dense.Dense.lean_rb) > 64) continue;
                    dn.bt_set[rows] = bt;
                    dn.resetCfgs();
                    const c = try dn.cfgs(rows);
                    _ = try g.hostF32(try g.astype(try g.adopt(try dn.decodeRows(s, x, p.trellis, p.suh, p.svh, rows)), .float32), got);
                    var differing: usize = 0;
                    for (got, want) |u, w| differing += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
                    checks += 1;
                    const ok = c.dense and differing == 0;
                    std.debug.print("TURBOEXL3_DS_GROUPED_BYTES groups={d} in={d} out={d} n={d} x={s} rows={d} bt={d} lean={} differing={d}/{d} {s}\n", .{ geo.groups, geo.in, geo.out, geo.n, @tagName(dt), rows, bt, c.lean, differing, n_out, if (ok) "PASS" else "FAIL" });
                    if (!ok) failed = true;
                }
                dn.bt_set[rows] = 0;
            }
        }
    }
    std.debug.print("TURBOEXL3_DS_GROUPED_BYTES_DONE checks={d} {s}\n", .{ checks, if (failed) "FAIL" else "PASS" });
    try testing.expect(!failed);
}

// ---- TURBOEXL3_DS_GEMV: the dense decode GEMV's launch shapes on DeepSeek-V4.1's trunk (a sweep; asserts bytes only) ----

/// A trunk linear's shape class: the linears deepseek_test.zig's `chainForward` runs at that (in, out) (wkv with the
/// compressor's two, w1 with w3), and the grouped wo_a
const GemvClass = enum { wq_a, wq_b, wkv, iwq_b, wo_b, w13, w2, wo_a };
const gemv_classes = [_]GemvClass{ .wq_a, .wq_b, .wkv, .iwq_b, .wo_b, .w13, .w2, .wo_a };

fn classShape(c: GemvClass) [2]u32 {
    return switch (c) {
        .wq_a => .{ 5120, 1280 },
        .wq_b => .{ 1280, 32768 },
        .wkv => .{ 5120, 512 },
        .iwq_b => .{ 1280, 4096 },
        .wo_b => .{ 8192, 5120 },
        .w13 => .{ 5120, 2304 },
        .w2 => .{ 2304, 5120 },
        .wo_a => .{ ds.wo_a.in, ds.wo_a.out },
    };
}

/// One launch-shape choice for a class at a row count; null fields keep the served value. `routed`: the grouped wo_a on
/// the routed down GEMV.
const GemvConf = struct {
    bt: u8 = 0,
    dec_on: ?bool = null,
    dec_rows: ?u32 = null,
    prep_in: ?bool = null,
    lean: ?bool = null,
    routed: bool = false,

    fn isDefault(c: GemvConf) bool {
        return c.bt == 0 and c.dec_on == null and c.dec_rows == null and c.prep_in == null and c.lean == null and !c.routed;
    }

    fn name(c: GemvConf, buf: []u8) []const u8 {
        if (c.isDefault()) return "served";
        const tri = struct {
            fn f(v: ?bool) []const u8 {
                return if (v) |b| (if (b) "1" else "0") else "-";
            }
        }.f;
        return std.fmt.bufPrint(buf, "bt{d}_dec{s}_decrows{d}_prepin{s}_lean{s}{s}", .{ c.bt, tri(c.dec_on), c.dec_rows orelse 0, tri(c.prep_in), tri(c.lean), if (c.routed) "_routed" else "" }) catch "?";
    }
};

/// What `applyDense` changed, to put back
const DenseSaved = struct { bt: u8, dec_on: bool, dec_rows: u32, dec_prep_rows: u32, dec_prep_out: u32, lean: bool };

fn denseShape(in: u32, out: u32) dense.Shape {
    return .{ .in = in, .out = out, .rate = format.Rate.fromN(ds.N).?, .dec = .mul1, .out_dtype = .bfloat16 };
}

fn groupedShape() grouped.Shape {
    return .{ .groups = ds.wo_a.groups, .in = ds.wo_a.in, .out = ds.wo_a.out, .rate = format.Rate.fromN(ds.N).?, .dec = .mul1, .out_dtype = .bfloat16 };
}

fn applyDense(d: *dense.Dense, rows: u32, c: GemvConf) DenseSaved {
    const sv: DenseSaved = .{ .bt = d.bt_set[rows], .dec_on = d.dec_on, .dec_rows = d.dec_rows, .dec_prep_rows = d.dec_prep_rows, .dec_prep_out = d.dec_prep_out, .lean = d.lean };
    d.bt_set[rows] = c.bt;
    if (c.dec_on) |v| d.dec_on = v;
    if (c.dec_rows) |v| d.dec_rows = v;
    if (c.lean) |v| d.lean = v;
    if (c.prep_in) |v| {
        // inside: every row count and width that fits; outside: none
        d.dec_prep_rows = if (v) @intCast(decode_rows_max) else 0;
        d.dec_prep_out = if (v) std.math.maxInt(u32) else 0;
    }
    d.resetCfgs();
    return sv;
}

fn restoreDense(d: *dense.Dense, rows: u32, sv: DenseSaved) void {
    d.bt_set[rows] = sv.bt;
    d.dec_on = sv.dec_on;
    d.dec_rows = sv.dec_rows;
    d.dec_prep_rows = sv.dec_prep_rows;
    d.dec_prep_out = sv.dec_prep_out;
    d.lean = sv.lean;
    d.resetCfgs();
}

const decode_rows_max: usize = root.kernels.DECODE_ROWS_MAX;

/// The grouped wo_a's served tiles per threadgroup at `rows` (`Grouped.denseGemvCfg`'s: `dense.Dense.btLaunch` over its
/// threadgroups at BT 1)
fn groupedBt(rows: u32) u32 {
    const rb = dense.Dense.rbCap(rows, dense.Dense.lean_rb);
    const chunks = (rows + rb - 1) / rb;
    return dense.Dense.btLaunch(rb, (ds.wo_a.out / 16) * chunks * ds.wo_a.groups * grouped.planFor(ds.wo_a.in, ds.wo_a.out).ks);
}

/// The launch-shape choices tried for `class` at `rows` (the served one first): tiles per threadgroup, DEC-ONE's prepare
/// inside or in its own launch, DEC-ONE against the three-launch chain, the lean text's one chunk against text_dgemv's
/// chunks of up to 5; for wo_a the dense GEMV's tiles and the routed down GEMV.
fn gemvConfs(d: ?*dense.Dense, class: GemvClass, rows: u32, buf: []GemvConf) ![]const GemvConf {
    var n: usize = 0;
    buf[n] = .{};
    n += 1;
    if (class == .wo_a) {
        const grb = dense.Dense.rbCap(rows, dense.Dense.lean_rb);
        for ([_]u8{ 1, 2, 4 }) |bt| {
            if (bt == groupedBt(rows) or 2 * @as(u32, bt) * grb > 32) continue;
            buf[n] = .{ .bt = bt };
            n += 1;
        }
        buf[n] = .{ .routed = true };
        n += 1;
        return buf[0..n];
    }
    const dd = d.?;
    const dec = dd.decTakes(rows);
    const rb = if (dec) dense.Dense.rbFor(rows) else dd.chainRb(rows);
    for ([_]u8{ 1, 2, 4, 8 }) |bt| {
        if (bt == dd.btAt(rows, rb)) continue;
        // a lane's 2 BT RB accumulators: at most 32 floats
        if (2 * @as(u32, bt) * rb > 32) continue;
        buf[n] = .{ .bt = bt };
        n += 1;
    }
    if (dec) {
        const inside = try dd.decPrepInside(rows);
        const fits = dense.Dense.decPrepFits(dd.shape, dd.ks_split, rb, dd.btAt(rows, rb));
        if (inside or fits) {
            buf[n] = .{ .prep_in = !inside };
            n += 1;
        }
        buf[n] = .{ .dec_on = false };
        n += 1;
    } else {
        buf[n] = .{ .lean = false };
        n += 1;
        buf[n] = .{ .dec_rows = 8 };
        n += 1;
    }
    return buf[0..n];
}

/// The trunk chain (deepseek_test.zig's `chainForward`, its trunk arm, 40 layers) at `rows`: min ms of 5 evaluations
/// after 2, and its output's bytes (as f32) in `out`
fn trunkMin(g: *G, s: mlx.mlx_stream, layers: []const ds.ChainLayer, x: mlx.mlx_array, rows: u32, out: []f32) !f64 {
    var total: [5]f64 = undefined;
    for (0..total.len + 2) |round| {
        const fm = g.mark();
        defer g.resetTo(fm);
        const t0 = ds.nowNs();
        const y = try ds.chainForward(g, s, layers, &.{}, &.{}, x, rows, .trunk, null);
        try g.evalAll(&.{y});
        const dt = ds.nowNs() - t0;
        if (round >= 2) total[round - 2] = @as(f64, @floatFromInt(dt)) / 1e6;
        if (round == 0) _ = try g.hostF32(try g.astype(y, .float32), out);
    }
    std.mem.sort(f64, &total, {}, std.sort.asc(f64));
    return total[0];
}

fn bytesDiffer(p: []const f32, q: []const f32) usize {
    var n: usize = 0;
    for (p, q) |u, w| n += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return n;
}

/// deepseek_test.zig's 40-layer decode chain, its trunk weights only (each layer's in its own buffers, the routed bank a
/// placeholder the trunk arm never reads), 40 of the indexer's wq_b for the per-class costs (the chain has 8), and the
/// cached `Dense` of each shape class and the grouped wo_a's `Grouped` (the objects `root.linear` / `linearGrouped` run)
const TrunkChain = struct {
    const nl = 40;
    layers: [nl]ds.ChainLayer,
    iwq: [nl]root.Proj,
    dense_of: [gemv_classes.len]?*dense.Dense,
    grp: *grouped.Grouped,

    fn init(a: std.mem.Allocator, g: *G, rnd: std.Random) !TrunkChain {
        var hp = [_]ds.HostProj{
            try ds.HostProj.draw(a, rnd, null, 5120, 1280, ds.N), // 0 wq_a
            try ds.HostProj.draw(a, rnd, null, 1280, 32768, ds.N), // 1 wq_b
            try ds.HostProj.draw(a, rnd, null, 5120, 512, ds.N), // 2 wkv, the compressor's wkv and wgate
            try ds.HostProj.draw(a, rnd, null, 1280, 4096, ds.N), // 3 the indexer's wq_b
            try ds.HostProj.draw(a, rnd, ds.wo_a.groups, ds.wo_a.in, ds.wo_a.out, ds.N), // 4 wo_a
            try ds.HostProj.draw(a, rnd, null, 8192, 5120, ds.N), // 5 wo_b
            try ds.HostProj.draw(a, rnd, null, 5120, 2304, ds.N), // 6 the shared w1 and w3
            try ds.HostProj.draw(a, rnd, null, 2304, 5120, ds.N), // 7 the shared w2
        };
        defer for (&hp) |*h| h.deinit(a);
        var tc: TrunkChain = undefined;
        for (&tc.layers, 0..) |*ly, li| {
            const L: u32 = @intCast(li);
            const comp = std.mem.indexOfScalar(u32, &ds.comp_layers, L) != null;
            const idx = std.mem.indexOfScalar(u32, &ds.index_layers, L) != null;
            const wq_a = try hp[0].upload(g);
            ly.* = .{
                .wq_a = wq_a,
                .wq_b = try hp[1].upload(g),
                .wkv = try hp[2].upload(g),
                .cwkv = if (comp) try hp[2].upload(g) else null,
                .cwgate = if (comp) try hp[2].upload(g) else null,
                .iwq_b = if (idx) try hp[3].upload(g) else null,
                .wo_a = try hp[4].upload(g),
                .wo_b = try hp[5].upload(g),
                .sw1 = try hp[6].upload(g),
                .sw3 = try hp[6].upload(g),
                .sw2 = try hp[7].upload(g),
                .bank = .{ .gate = wq_a, .up = wq_a, .down = wq_a },
            };
        }
        for (&tc.iwq) |*p| p.* = try hp[3].upload(g);
        tc.dense_of = @splat(null);
        for (gemv_classes, 0..) |c, k| {
            if (c == .wo_a) continue;
            const io = classShape(c);
            tc.dense_of[k] = try root.denseCached(denseShape(io[0], io[1]));
        }
        tc.grp = try root.groupedCached(groupedShape());
        return tc;
    }
};

// TURBOEXL3_DS_GEMV=1 with TURBOEXL3_GPU=1: the dense decode GEMV's launch shapes swept on the 40-layer trunk chain
// (deepseek_test.zig's `chainForward`, its trunk arm, the served kernels around it), rows 1..8. Per row count: the chain
// as served, then for each shape class each launch-shape choice on that class's linears alone (`gemvConfs`), the chain's
// ms and its delta from the served chain's (measured before and after the row count's sweep), and the chain output's
// bytes against the served chain's (every choice is a launch shape: the bytes must not move). Then the per-class GEMV
// costs: 40 calls of one class (serialized by their launches' own dependencies) against 40 of its GEMV launches alone on
// one prepared input (no barrier between them). Then the best choice per (class, row count) together, against the served
// chain, in interleaved rounds.
test "turboexl3 gpu deepseek gemv sweep: the dense decode GEMV's launch shapes on the 40-layer trunk chain, rows 1..8 (TURBOEXL3_DS_GEMV=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_DS_GEMV") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xD5C4A2);
    const rnd = prng.random();
    const nl = TrunkChain.nl;
    const tc = try a.create(TrunkChain);
    defer a.destroy(tc);
    tc.* = try TrunkChain.init(a, &g, rnd);
    const layers = &tc.layers;
    const iwq = &tc.iwq;
    const dense_of = tc.dense_of;
    const grp = tc.grp;
    // the chosen choice per (row count, class): the fastest past 0.3% of the served chain
    var best: [9][gemv_classes.len]GemvConf = @splat(@splat(.{}));
    var name_buf: [96]u8 = undefined;
    for (1..9) |ri| {
        const rows: u32 = @intCast(ri);
        const rm = g.mark();
        defer g.resetTo(rm);
        const x = try ds.randBf16(a, &g, rnd, &.{ ds.ci(rows), 5120 });
        const want = try a.alloc(f32, rows * 5120);
        defer a.free(want);
        const got = try a.alloc(f32, rows * 5120);
        defer a.free(got);
        const base0 = try trunkMin(&g, s, layers, x, rows, want);
        std.debug.print("TURBOEXL3_DS_GEMV rows={d} served chain_ms={d:.3}\n", .{ rows, base0 });
        var results: [gemv_classes.len][12]f64 = undefined;
        var confs_of: [gemv_classes.len][12]GemvConf = undefined;
        var nconf: [gemv_classes.len]usize = @splat(0);
        for (gemv_classes, 0..) |c, k| {
            var cbuf: [12]GemvConf = undefined;
            const confs = try gemvConfs(dense_of[k], c, rows, &cbuf);
            nconf[k] = confs.len;
            for (confs, 0..) |cf, j| {
                confs_of[k][j] = cf;
                if (j == 0) {
                    results[k][0] = base0;
                    continue;
                }
                var dsv: ?DenseSaved = null;
                if (dense_of[k]) |d| dsv = applyDense(d, rows, cf) else {
                    grp.dense_gemv = !cf.routed;
                    grp.bt_set[rows] = cf.bt;
                    grp.resetCfgs();
                }
                const t = trunkMin(&g, s, layers, x, rows, got) catch |e| blk: {
                    std.debug.print("TURBOEXL3_DS_GEMV rows={d} class={s} conf={s} error={s}\n", .{ rows, @tagName(c), cf.name(&name_buf), @errorName(e) });
                    break :blk std.math.inf(f64);
                };
                if (dense_of[k]) |d| restoreDense(d, rows, dsv.?) else {
                    grp.dense_gemv = true;
                    grp.bt_set[rows] = 0;
                    grp.resetCfgs();
                }
                results[k][j] = t;
                const diff = if (std.math.isFinite(t)) bytesDiffer(got, want) else got.len;
                std.debug.print("TURBOEXL3_DS_GEMV rows={d} class={s} conf={s} chain_ms={d:.3} delta_ms={d:.3} differing={d}/{d} {s}\n", .{ rows, @tagName(c), cf.name(&name_buf), t, t - base0, diff, got.len, if (diff == 0) "BYTES_OK" else "BYTES_DIFFER" });
                if (diff != 0) results[k][j] = std.math.inf(f64);
            }
        }
        const base1 = try trunkMin(&g, s, layers, x, rows, got);
        const base = @min(base0, base1);
        std.debug.print("TURBOEXL3_DS_GEMV rows={d} served_again chain_ms={d:.3} (drift {d:.3}) differing={d}\n", .{ rows, base1, base1 - base0, bytesDiffer(got, want) });
        for (gemv_classes, 0..) |c, k| {
            var bj: usize = 0;
            for (1..nconf[k]) |j| if (results[k][j] < results[k][bj]) {
                bj = j;
            };
            if (bj != 0 and results[k][bj] < base * (1.0 - 0.003)) best[ri][k] = confs_of[k][bj];
            std.debug.print("TURBOEXL3_DS_GEMV_BEST rows={d} class={s} conf={s} chain_ms={d:.3} vs_served_ms={d:.3}\n", .{ rows, @tagName(c), best[ri][k].name(&name_buf), if (best[ri][k].isDefault()) base else results[k][bj], base });
        }
    }
    // the per-class GEMV costs: 40 linears of one class back to back vs 40 GEMV launches on one prepared input
    for ([_]u32{ 1, 4, 6, 8 }) |rows| for (gemv_classes, 0..) |c, k| {
        const d = dense_of[k] orelse continue;
        const rm = g.mark();
        defer g.resetTo(rm);
        const io = classShape(c);
        const x = try ds.randBf16(a, &g, rnd, &.{ ds.ci(rows), ds.ci(io[0]) });
        var projs: [nl]root.Proj = undefined;
        for (&projs, 0..) |*p, li| p.* = switch (c) {
            .wq_a => layers[li].wq_a,
            .wq_b => layers[li].wq_b,
            .wkv => layers[li].wkv,
            .iwq_b => iwq[li],
            .wo_b => layers[li].wo_b,
            .w13 => layers[li].sw1,
            .w2 => layers[li].sw2,
            .wo_a => unreachable,
        };
        var t_lin: [5]f64 = undefined;
        var t_gemv: [5]f64 = undefined;
        const cc = try d.cfgs(rows);
        for (0..t_lin.len + 2) |round| {
            {
                const fm = g.mark();
                defer g.resetTo(fm);
                var ys: [nl]mlx.mlx_array = undefined;
                for (projs, 0..) |p, li| ys[li] = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
                const t0 = ds.nowNs();
                try g.evalAll(&ys);
                if (round >= 2) t_lin[round - 2] = @as(f64, @floatFromInt(ds.nowNs() - t0)) / 1e3 / nl;
            }
            {
                const fm = g.mark();
                defer g.resetTo(fm);
                var po: [2]mlx.mlx_array = undefined;
                try ks_launch.applyConfig(d.prep_s, cc.prep, &.{ x, projs[0].suh }, &po, s);
                _ = try g.adopt(po[0]);
                _ = try g.adopt(po[1]);
                try g.evalAll(&.{po[0]});
                var zs: [nl]mlx.mlx_array = undefined;
                for (projs, 0..) |p, li| {
                    var o: [1]mlx.mlx_array = undefined;
                    try ks_launch.applyConfig(if (cc.lean) d.dgemv_lean else d.dgemv, cc.gemv, &.{ po[0], p.trellis }, &o, s);
                    zs[li] = try g.adopt(o[0]);
                }
                const t0 = ds.nowNs();
                try g.evalAll(&zs);
                if (round >= 2) t_gemv[round - 2] = @as(f64, @floatFromInt(ds.nowNs() - t0)) / 1e3 / nl;
            }
        }
        std.mem.sort(f64, &t_lin, {}, std.sort.asc(f64));
        std.mem.sort(f64, &t_gemv, {}, std.sort.asc(f64));
        const io_w: f64 = @floatFromInt(@as(u64, io[0]) * io[1]);
        std.debug.print("TURBOEXL3_DS_GEMV_COST class={s} {d}x{d} rows={d} linear_us={d:.2} gemv_alone_us={d:.2} gemv_gweights_per_s={d:.1} chain_rb={d} bt={d} ks={d}\n", .{ @tagName(c), io[0], io[1], rows, t_lin[0], t_gemv[0], io_w / (t_gemv[0] * 1e3), d.chainRb(rows), d.btAt(rows, d.chainRb(rows)), d.ks_split });
    };
    // the chosen choices together against the served chain, interleaved (served, chosen) x 3
    for (1..9) |ri| {
        const rows: u32 = @intCast(ri);
        const rm = g.mark();
        defer g.resetTo(rm);
        const x = try ds.randBf16(a, &g, rnd, &.{ ds.ci(rows), 5120 });
        const want = try a.alloc(f32, rows * 5120);
        defer a.free(want);
        const got = try a.alloc(f32, rows * 5120);
        defer a.free(got);
        var t_served: f64 = std.math.inf(f64);
        var t_chosen: f64 = std.math.inf(f64);
        var diff: usize = 0;
        for (0..3) |_| {
            t_served = @min(t_served, try trunkMin(&g, s, layers, x, rows, want));
            var saved: [gemv_classes.len]?DenseSaved = @splat(null);
            for (gemv_classes, 0..) |_, k| {
                const cf = best[ri][k];
                if (cf.isDefault()) continue;
                if (dense_of[k]) |d| saved[k] = applyDense(d, rows, cf) else {
                    grp.dense_gemv = !cf.routed;
                    grp.bt_set[rows] = cf.bt;
                    grp.resetCfgs();
                }
            }
            t_chosen = @min(t_chosen, try trunkMin(&g, s, layers, x, rows, got));
            diff += bytesDiffer(got, want);
            for (gemv_classes, 0..) |_, k| {
                if (dense_of[k]) |d| {
                    if (saved[k]) |sv| restoreDense(d, rows, sv);
                } else {
                    grp.dense_gemv = true;
                    grp.bt_set[rows] = 0;
                    grp.resetCfgs();
                }
            }
        }
        std.debug.print("TURBOEXL3_DS_GEMV_CHOSEN rows={d} served_ms={d:.3} chosen_ms={d:.3} gain={d:.2}% differing={d} {s}\n", .{ rows, t_served, t_chosen, 100.0 * (t_served - t_chosen) / t_served, diff, if (diff == 0) "BYTES_OK" else "BYTES_DIFFER" });
    }
}

/// An arm of the A/B on the trunk chain: the served launches, the lean chunks at BT 1 (btFor's, before btLaunch), or
/// window 12's per-class choices at rows 1 to 4 (`ab_table`)
const AbArm = enum { served, bt1, table };

/// Window 12's fastest per-class choices at rows 1 to 4 (TURBOEXL3_DS_GEMV_BEST), in `gemv_classes`' order
const ab_table = [4][gemv_classes.len]GemvConf{
    .{ .{ .bt = 2 }, .{ .bt = 2 }, .{ .dec_on = false }, .{}, .{ .bt = 2 }, .{}, .{}, .{} },
    .{ .{}, .{ .bt = 1 }, .{ .dec_on = false }, .{}, .{}, .{ .dec_on = false }, .{}, .{} },
    .{ .{}, .{}, .{ .dec_on = false }, .{}, .{}, .{ .dec_on = false }, .{}, .{} },
    .{ .{}, .{}, .{ .dec_on = false }, .{}, .{}, .{ .dec_on = false }, .{}, .{} },
};

/// Sets `arm`'s launches at `rows` on the chain's linears (`saved`: what to put back with `clearArm`)
fn setArm(tc: *TrunkChain, arm: AbArm, rows: u32, saved: *[gemv_classes.len]?DenseSaved) void {
    saved.* = @splat(null);
    for (gemv_classes, 0..) |c, k| {
        const cf: GemvConf = switch (arm) {
            .served => .{},
            .table => if (rows >= 1 and rows <= ab_table.len) ab_table[rows - 1][k] else .{},
            .bt1 => blk: {
                if (c == .wo_a) break :blk if (dense.Dense.rbCap(rows, dense.Dense.lean_rb) > dense.Dense.max_rb) GemvConf{ .bt = 1 } else GemvConf{};
                const d = tc.dense_of[k].?;
                break :blk if (!d.decTakes(rows) and d.leanFor(d.chainRb(rows))) GemvConf{ .bt = 1 } else GemvConf{};
            },
        };
        if (cf.isDefault()) continue;
        if (tc.dense_of[k]) |d| saved[k] = applyDense(d, rows, cf) else {
            tc.grp.bt_set[rows] = cf.bt;
            tc.grp.resetCfgs();
        }
    }
}

fn clearArm(tc: *TrunkChain, rows: u32, saved: *[gemv_classes.len]?DenseSaved) void {
    for (gemv_classes, 0..) |_, k| {
        if (tc.dense_of[k]) |d| {
            if (saved[k]) |sv| restoreDense(d, rows, sv);
        } else {
            tc.grp.bt_set[rows] = 0;
            tc.grp.resetCfgs();
        }
    }
    saved.* = @splat(null);
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return if (xs.len % 2 == 1) xs[xs.len / 2] else 0.5 * (xs[xs.len / 2 - 1] + xs[xs.len / 2]);
}

// TURBOEXL3_DS_GEMV_AB=1 with TURBOEXL3_GPU=1: the served launches against their alternatives on the 40-layer trunk chain,
// A B B A in 3 rounds (each run the min of 5 evaluations after 2): at rows 6, 7, 8, 12 and 16 the lean chunks at BT 1 (A,
// the launches before `btLaunch`) against the served BT (B); at rows 1 to 4 the served launches (A) against window 12's
// per-class table (B, `ab_table`); at row 5, where neither applies, the served launches twice (the noise floor). Every
// run's chain output is held to the first run's bytes. Lines: TURBOEXL3_DS_GEMV_AB per run, TURBOEXL3_DS_GEMV_AB_SUM per
// row count.
test "turboexl3 gpu deepseek gemv ab: the lean chunks' BT rule and window 12's table against the launches before them on the trunk chain, A B B A (TURBOEXL3_DS_GEMV_AB=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_DS_GEMV_AB") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xD5C4A3);
    const rnd = prng.random();
    const tc = try a.create(TrunkChain);
    defer a.destroy(tc);
    tc.* = try TrunkChain.init(a, &g, rnd);
    var saved: [gemv_classes.len]?DenseSaved = @splat(null);
    var failed = false;
    for ([_]u32{ 6, 7, 8, 12, 16, 1, 2, 3, 4, 5 }) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        const x = try ds.randBf16(a, &g, rnd, &.{ ds.ci(rows), 5120 });
        const want = try a.alloc(f32, rows * 5120);
        defer a.free(want);
        const got = try a.alloc(f32, rows * 5120);
        defer a.free(got);
        const arms: [2]AbArm = if (rows <= 4) .{ .served, .table } else if (rows == 5) .{ .served, .served } else .{ .bt1, .served };
        var t: [2][6]f64 = undefined;
        var n: [2]usize = .{ 0, 0 };
        var diff: usize = 0;
        var first = true;
        for (0..3) |round| for ([_]u1{ 0, 1, 1, 0 }, 0..) |ai, pos| {
            setArm(tc, arms[ai], rows, &saved);
            const ms = try trunkMin(&g, s, &tc.layers, x, rows, if (first) want else got);
            clearArm(tc, rows, &saved);
            const d = if (first) 0 else bytesDiffer(got, want);
            first = false;
            diff += d;
            t[ai][n[ai]] = ms;
            n[ai] += 1;
            std.debug.print("TURBOEXL3_DS_GEMV_AB rows={d} round={d} pos={d} arm={s}{s} chain_ms={d:.3} differing={d}\n", .{ rows, round, pos, if (ai == 0) "A:" else "B:", @tagName(arms[ai]), ms, d });
        };
        var amin: f64 = std.math.inf(f64);
        var bmin: f64 = std.math.inf(f64);
        for (t[0]) |v| amin = @min(amin, v);
        for (t[1]) |v| bmin = @min(bmin, v);
        const amed = median(&t[0]);
        const bmed = median(&t[1]);
        std.debug.print("TURBOEXL3_DS_GEMV_AB_SUM rows={d} A={s} B={s} a_min_ms={d:.3} a_med_ms={d:.3} b_min_ms={d:.3} b_med_ms={d:.3} b_minus_a_min={d:.3} b_minus_a_med={d:.3} differing={d} {s}\n", .{ rows, @tagName(arms[0]), @tagName(arms[1]), amin, amed, bmin, bmed, bmin - amin, bmed - amed, diff, if (diff == 0) "BYTES_OK" else "BYTES_DIFFER" });
        if (diff != 0) failed = true;
    }
    try testing.expect(!failed);
}
