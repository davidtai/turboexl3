//! Ling-3.1-flash's routed combine (`models/ling/combine.zig`) on the GPU (TURBOEXL3_GPU=1, else skipped), at Ling's
//! shape (top-8, hidden 4096), against the host's composed combine, MLX's `sum(multiply(astype(routed, f32),
//! expand_dims(weights, -1)), -2)` (turboexl3_ling's streamed source), word for word:
//! - `combineRouted` at rows 1, 7, 4096 and 4097, over rows that hold signed zeros, products that overflow, tiny and
//!   denormal products and inputs, one normal product among denormal ones, and exact cancellations;
//! - `combineParts` over a wide call's unjoined outputs laid out as the streamer's JOINLESS plan lays them (up to 24 read
//!   in place; above that the smallest merged, in join order, into the last source: `planJoinless`), against the
//!   composed combine on the array the join builds (`take(concatenate(outputs), inverse(positions))`), at 5 and 40
//!   outputs;
//! - the refusals: K past 8, a shape or dtype off, no sources or more than 24.
//! Lines: TURBOEXL3_LING_COMBINE.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const root = @import("../../root.zig");
const G = support.Graph;
const combine = @import("combine.zig");

const testing = std.testing;
const hidden: usize = 4096;
const top_k: usize = 8;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// MLX's sum over one axis (the host combine's `mlx_sum_axis`): the consumer's binding where its `sdk.mlx` declares one,
/// else libmlxc's own symbol
const sumAxis = if (@hasDecl(mlx, "mlx_sum_axis")) mlx.mlx_sum_axis else struct {
    extern "c" fn mlx_sum_axis(res: *mlx.mlx_array, a: mlx.mlx_array, axis: c_int, keepdims: bool, s: mlx.mlx_stream) c_int;
}.mlx_sum_axis;
/// MLX's expand_dims (the host combine's): likewise
const expandDims = if (@hasDecl(mlx, "mlx_expand_dims")) mlx.mlx_expand_dims else struct {
    extern "c" fn mlx_expand_dims(res: *mlx.mlx_array, a: mlx.mlx_array, axis: c_int, s: mlx.mlx_stream) c_int;
}.mlx_expand_dims;

/// The host's combine: `sum(multiply(astype(routed, f32), expand_dims(weights, -1)), -2)`, op for op
fn hostCombine(g: *G, s: mlx.mlx_stream, routed: mlx.mlx_array, weights: mlx.mlx_array) !mlx.mlx_array {
    const rf = try g.astype(routed, .float32);
    var we = mlx.mlx_array_new();
    try mlx.check(expandDims(&we, weights, -1, s));
    const wt = try g.adopt(we);
    const p = try g.mul(rf, wt);
    var y = mlx.mlx_array_new();
    try mlx.check(sumAxis(&y, p, -2, false, s));
    return g.adopt(y);
}

fn bfExact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
}

/// A random f32 subnormal (either sign)
fn denormal(rnd: std.Random) f32 {
    const m: u32 = 1 + rnd.uintLessThan(u32, 0x007F_FFFF);
    const sign: u32 = if (rnd.boolean()) 0x8000_0000 else 0;
    return @bitCast(sign | m);
}

/// The patterns a row takes (by row % 8)
const patterns = [_][]const u8{ "plain", "signed_zeros", "all_negative_zero", "overflow", "tiny", "denormal_in", "one_normal", "cancel" };

/// One row's routed values [K * D] (pair-major) and weights [K]: pattern 0 normal values and bf16-exact weights in
/// (0, 1) (a router's); 1 signed zeros among both; 2 every product -0; 3 products past f32's range (one sign a column);
/// 4 tiny products (denormal sums); 5 denormal routed values; 6 one normal product among denormal ones; 7 pairs of
/// products that cancel exactly.
fn fillRow(rnd: std.Random, pattern: usize, routed: []f32, w: []f32) void {
    switch (pattern) {
        0 => {
            for (routed) |*v| v.* = rnd.floatNorm(f32) * 2;
            for (w) |*v| v.* = bfExact(0.01 + 0.98 * rnd.float(f32));
        },
        1 => {
            for (routed, 0..) |*v, i| v.* = if (i % 3 == 0) (if (rnd.boolean()) @as(f32, -0.0) else 0.0) else rnd.floatNorm(f32);
            for (w, 0..) |*v, k| v.* = switch (k % 4) {
                0 => 0.0,
                1 => -0.0,
                2 => bfExact(rnd.float(f32)),
                else => -bfExact(rnd.float(f32)),
            };
        },
        2 => {
            for (routed) |*v| v.* = -0.0;
            for (w) |*v| v.* = bfExact(0.01 + rnd.float(f32));
        },
        3 => {
            for (routed) |*v| v.* = 1.0e38 * (1 + rnd.float(f32));
            for (w) |*v| v.* = 2 + 2 * rnd.float(f32);
        },
        4 => {
            for (routed) |*v| v.* = rnd.floatNorm(f32) * 1.0e-36;
            for (w) |*v| v.* = 1.0e-3 * (0.5 + rnd.float(f32));
        },
        5 => {
            for (routed) |*v| v.* = denormal(rnd);
            const ws = [_]f32{ 1, 0.5, 0.25, 0.75, 1, 0.125, 0.5, 1 };
            for (w, 0..) |*v, k| v.* = ws[k % ws.len];
        },
        6 => {
            const d = routed.len / w.len;
            const big = rnd.uintLessThan(usize, w.len);
            for (0..w.len) |k| for (routed[k * d ..][0..d]) |*v| {
                v.* = if (k == big) rnd.floatNorm(f32) else denormal(rnd);
            };
            for (w) |*v| v.* = bfExact(0.01 + 0.98 * rnd.float(f32));
        },
        else => {
            const d = routed.len / w.len;
            for (0..w.len / 2) |j| {
                const wv = bfExact(0.01 + 0.98 * rnd.float(f32));
                w[2 * j] = wv;
                w[2 * j + 1] = wv;
                for (0..d) |c| {
                    const x = rnd.floatNorm(f32);
                    routed[(2 * j) * d + c] = x;
                    routed[(2 * j + 1) * d + c] = -x;
                }
            }
        },
    }
}

/// Rows `n` of pattern-filled routed values [n, K, D] and weights [n, K]
fn fillRows(rnd: std.Random, n: usize, routed: []f32, w: []f32) void {
    for (0..n) |r| fillRow(rnd, r % patterns.len, routed[r * top_k * hidden ..][0 .. top_k * hidden], w[r * top_k ..][0..top_k]);
}

/// The words of `got` that differ from `want`, a line per row pattern (with what its expected words hold: infinities,
/// subnormals, +0 and -0), and the first few differences
fn compareWords(label: []const u8, n: usize, got: []const f32, want: []const f32) usize {
    var by_pattern: [patterns.len]usize = @splat(0);
    var held: [patterns.len][4]usize = @splat(@splat(0));
    var shown: usize = 0;
    var total: usize = 0;
    for (got, want, 0..) |u, v, i| {
        const bu: u32 = @bitCast(u);
        const bv: u32 = @bitCast(v);
        const pat = (i / hidden) % patterns.len;
        if (std.math.isInf(v)) held[pat][0] += 1;
        if (v != 0 and @abs(v) < std.math.floatMin(f32)) held[pat][1] += 1;
        if (bv == 0) held[pat][2] += 1;
        if (bv == 0x8000_0000) held[pat][3] += 1;
        if (bu == bv) continue;
        total += 1;
        const row = i / hidden;
        by_pattern[row % patterns.len] += 1;
        if (shown < 8) {
            shown += 1;
            std.debug.print("TURBOEXL3_LING_COMBINE {s} differs at row={d} col={d} pattern={s}: got 0x{x:0>8} want 0x{x:0>8}\n", .{ label, row, i % hidden, patterns[row % patterns.len], bu, bv });
        }
    }
    for (patterns, by_pattern, held, 0..) |name, d, h, p| if (p < n) std.debug.print("TURBOEXL3_LING_COMBINE {s} rows={d} pattern={s} inf={d} subnormal={d} pos_zero={d} neg_zero={d} differing={d}\n", .{ label, n, name, h[0], h[1], h[2], h[3], d });
    return total;
}

test "turboexl3 gpu ling combine: combineRouted gives MLX's sum(multiply(astype(routed, f32), expand_dims(weights, -1)), -2) word for word at rows 1, 7, 4096 and 4097 over signed zeros, overflowing, tiny and denormal products and cancellations (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    // a GPU test past others in one process starts and ends with no engine cached (`root.releaseEngines`)
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x11C0_4B1E);
    const rnd = prng.random();
    var failed = false;
    for ([_]usize{ 1, 7, 4096, 4097 }) |n| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const routed = try a.alloc(f32, n * top_k * hidden);
        defer a.free(routed);
        const w = try a.alloc(f32, n * top_k);
        defer a.free(w);
        fillRows(rnd, n, routed, w);
        const ra = try g.hostArray(std.mem.sliceAsBytes(routed), &.{ ci(n), ci(top_k), ci(hidden) }, .float32);
        const wa = try g.hostArray(std.mem.sliceAsBytes(w), &.{ ci(n), ci(top_k) }, .float32);
        const want = try a.alloc(f32, n * hidden);
        defer a.free(want);
        const got = try a.alloc(f32, n * hidden);
        defer a.free(got);
        _ = try g.hostF32(try hostCombine(&g, s, ra, wa), want);
        const y = try g.adopt(try combine.combineRouted(s, ra, wa));
        try testing.expectEqualSlices(c_int, &.{ ci(n), ci(hidden) }, mlx.getShape(y));
        _ = try g.hostF32(y, got);
        const d = compareWords("routed", n, got, want);
        std.debug.print("TURBOEXL3_LING_COMBINE routed rows={d} k={d} hidden={d} differing={d}/{d} {s}\n", .{ n, top_k, hidden, d, n * hidden, if (d == 0) "PASS" else "FAIL" });
        if (d != 0) failed = true;
    }
    try testing.expect(!failed);
}

// ── the streamer's JOINLESS plan (sdk/weights/streamed/experts.zig `planJoinless` / `fillJoinlessLoc`), as the host makes it ──

const joinless_sources = combine.max_sources;

/// Up to `joinless_sources` outputs each read in place; above, the smallest n - 23 (earlier first among equals) merged
/// in join order into the last source, the rest sources 0.. in join order. Per output: `src[i]`, `off[i]` (its first
/// row there); returns the sources.
pub fn planJoinless(rows: []const u32, order: []u32, src: []u8, off: []u32) usize {
    const n = rows.len;
    @memset(off, 0);
    if (n <= joinless_sources) {
        for (src, 0..) |*sv, i| sv.* = @intCast(i);
        return n;
    }
    for (order, 0..) |*x, i| x.* = @intCast(i);
    std.sort.pdq(u32, order, rows, struct {
        fn lt(r: []const u32, x: u32, y: u32) bool {
            return if (r[x] != r[y]) r[x] < r[y] else x < y;
        }
    }.lt);
    const merged: u8 = joinless_sources - 1;
    @memset(src, 0);
    for (order[0 .. n - merged]) |i| src[i] = merged;
    var in_place: u8 = 0;
    var at: u32 = 0;
    for (rows, src, off) |r, *sv, *o| {
        if (sv.* == merged) {
            o.* = at;
            at += r;
        } else {
            sv.* = in_place;
            in_place += 1;
        }
    }
    return joinless_sources;
}

/// Each assignment's (source, row): output i's join-ordered rows (`pos`, the outputs' rows back to back) are rows
/// `off[i]..` of source `src[i]`
pub fn fillJoinlessLoc(rows: []const u32, src: []const u8, off: []const u32, pos: []const u32, loc: []i32) void {
    var j: usize = 0;
    for (rows, src, off) |r, sv, o| {
        for (pos[j..][0..r], o..) |p, row| {
            loc[2 * p] = sv;
            loc[2 * p + 1] = @intCast(row);
        }
        j += r;
    }
}

test "turboexl3 gpu ling combine: combineParts over a wide call's unjoined outputs (5 read in place; 40 through the streamer's merge rule) gives the composed combine's bytes on the array the join builds (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    // a GPU test past others in one process starts and ends with no engine cached (`root.releaseEngines`)
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x11C0_4B1F);
    const rnd = prng.random();
    var failed = false;
    for ([_]usize{ 7, 4096 }) |n| for ([_]usize{ 5, 40 }) |n_out| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const n_ids = n * top_k;
        // each assignment's expert output (by routed position) and the weights
        const by_pos = try a.alloc(f32, n_ids * hidden);
        defer a.free(by_pos);
        const w = try a.alloc(f32, n_ids);
        defer a.free(w);
        fillRows(rnd, n, by_pos, w);
        // the waves' outputs: a random permutation of the positions cut into n_out runs (each at least one row)
        const pos = try a.alloc(u32, n_ids);
        defer a.free(pos);
        for (pos, 0..) |*p, i| p.* = @intCast(i);
        rnd.shuffle(u32, pos);
        const rows = try a.alloc(u32, n_out);
        defer a.free(rows);
        {
            @memset(rows, 1);
            var left = n_ids - n_out;
            for (rows[0 .. n_out - 1]) |*r| {
                const take_n = if (left == 0) 0 else rnd.uintLessThan(usize, @min(left, 3 * n_ids / n_out) + 1);
                r.* += @intCast(take_n);
                left -= take_n;
            }
            rows[n_out - 1] += @intCast(left);
        }
        var outs: std.ArrayList(mlx.mlx_array) = .empty;
        defer outs.deinit(a);
        {
            var j: usize = 0;
            for (rows) |r| {
                const buf = try a.alloc(f32, r * hidden);
                defer a.free(buf);
                for (0..r) |q| @memcpy(buf[q * hidden ..][0..hidden], by_pos[pos[j + q] * hidden ..][0..hidden]);
                try outs.append(a, try g.hostArray(std.mem.sliceAsBytes(buf), &.{ ci(r), ci(hidden) }, .float32));
                j += r;
            }
        }
        const wa = try g.hostArray(std.mem.sliceAsBytes(w), &.{ ci(n), ci(top_k) }, .float32);
        // the join: take(concatenate(outputs), inverse(positions)), then the composed combine
        const inv = try a.alloc(u32, n_ids);
        defer a.free(inv);
        for (pos, 0..) |p, j| inv[p] = @intCast(j);
        const joined = try g.concat(outs.items, 0);
        const ord = try g.hostArray(std.mem.sliceAsBytes(inv), &.{ci(n_ids)}, .uint32);
        const ro = try g.reshape(try g.take(joined, ord, 0), &.{ ci(n), ci(top_k), ci(hidden) });
        const want = try a.alloc(f32, n * hidden);
        defer a.free(want);
        _ = try g.hostF32(try hostCombine(&g, s, ro, wa), want);
        // the JOINLESS plan: the sources (in place, then the merged one) and each assignment's (source, row)
        const order = try a.alloc(u32, n_out);
        defer a.free(order);
        const src = try a.alloc(u8, n_out);
        defer a.free(src);
        const off = try a.alloc(u32, n_out);
        defer a.free(off);
        const n_src = planJoinless(rows, order, src, off);
        const loc = try a.alloc(i32, 2 * n_ids);
        defer a.free(loc);
        fillJoinlessLoc(rows, src, off, pos, loc);
        var sources: std.ArrayList(mlx.mlx_array) = .empty;
        defer sources.deinit(a);
        var merge_in: std.ArrayList(mlx.mlx_array) = .empty;
        defer merge_in.deinit(a);
        for (outs.items, src) |x, sv| {
            if (n_src < n_out and sv == n_src - 1) try merge_in.append(a, x) else try sources.append(a, x);
        }
        if (merge_in.items.len > 0) try sources.append(a, try g.concat(merge_in.items, 0));
        try testing.expectEqual(n_src, sources.items.len);
        const la = try g.hostArray(std.mem.sliceAsBytes(loc), &.{ ci(n), ci(top_k), 2 }, .int32);
        const y = try g.adopt(try combine.combineParts(s, sources.items, la, wa));
        const got = try a.alloc(f32, n * hidden);
        defer a.free(got);
        _ = try g.hostF32(y, got);
        var label_buf: [32]u8 = undefined;
        const label = std.fmt.bufPrint(&label_buf, "parts_{d}", .{n_out}) catch "parts";
        const d = compareWords(label, n, got, want);
        std.debug.print("TURBOEXL3_LING_COMBINE parts rows={d} outputs={d} sources={d} merged={d} differing={d}/{d} {s}\n", .{ n, n_out, n_src, merge_in.items.len, d, n * hidden, if (d == 0) "PASS" else "FAIL" });
        if (d != 0) failed = true;
    };
    try testing.expect(!failed);
}

test "turboexl3 gpu ling combine: the refusals (K past 8, a shape or dtype off, no sources or more than 24) come before any launch (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    // a GPU test past others in one process starts and ends with no engine cached (`root.releaseEngines`)
    root.releaseEngines();
    defer root.releaseEngines();
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const z = try a.alloc(f32, 2 * 9 * 8);
    defer a.free(z);
    @memset(z, 0);
    const zi = try a.alloc(i32, 2 * 9 * 2);
    defer a.free(zi);
    @memset(zi, 0);
    const r9 = try g.hostArray(std.mem.sliceAsBytes(z[0 .. 2 * 9 * 8]), &.{ 2, 9, 8 }, .float32);
    const w9 = try g.hostArray(std.mem.sliceAsBytes(z[0 .. 2 * 9]), &.{ 2, 9 }, .float32);
    try testing.expectError(error.BadExl3Shape, combine.combineRouted(s, r9, w9));
    const r8 = try g.hostArray(std.mem.sliceAsBytes(z[0 .. 2 * 8 * 8]), &.{ 2, 8, 8 }, .float32);
    const w8 = try g.hostArray(std.mem.sliceAsBytes(z[0 .. 2 * 8]), &.{ 2, 8 }, .float32);
    const w7 = try g.hostArray(std.mem.sliceAsBytes(z[0 .. 2 * 7]), &.{ 2, 7 }, .float32);
    try testing.expectError(error.BadExl3Shape, combine.combineRouted(s, r8, w7));
    try testing.expectError(error.BadExl3Dtype, combine.combineRouted(s, try g.astype(r8, .bfloat16), w8));
    const l8 = try g.hostArray(std.mem.sliceAsBytes(zi[0 .. 2 * 8 * 2]), &.{ 2, 8, 2 }, .int32);
    const src = try g.hostArray(std.mem.sliceAsBytes(z[0 .. 4 * 8]), &.{ 4, 8 }, .float32);
    try testing.expectError(error.BadExl3Shape, combine.combineParts(s, &.{}, l8, w8));
    var many: [combine.max_sources + 1]mlx.mlx_array = @splat(src);
    try testing.expectError(error.BadExl3Shape, combine.combineParts(s, &many, l8, w8));
    try testing.expectError(error.BadExl3Dtype, combine.combineParts(s, &.{src}, try g.astype(l8, .float32), w8));
    try testing.expectError(error.BadExl3Dtype, combine.combineParts(s, &.{try g.astype(src, .bfloat16)}, l8, w8));
    const src4 = try g.hostArray(std.mem.sliceAsBytes(z[0 .. 4 * 4]), &.{ 4, 4 }, .float32);
    try testing.expectError(error.BadExl3Shape, combine.combineParts(s, &.{ src, src4 }, l8, w8));
}
