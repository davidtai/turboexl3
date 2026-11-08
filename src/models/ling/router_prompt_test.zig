//! Ling-3.1-flash's MoE router at prompt widths in one launch (`models/ling/router_prompt.zig`) against the host's
//! (turboexl3_ling's) served path word for word, every operand built the way the host builds it:
//! - the router weight: f16 [512, hidden] row-major as the loader reads it from the pack (here every finite f16 word,
//!   -0.0 and the subnormals among them), and the loader's `transposed` (MLX's transpose: a [hidden, 512] view,
//!   strides (1, hidden)): what Linear holds with MLX_SERVE_LING_ROUTER_F16=1; the loader's f32 table over that view
//!   (MLX's astype, then + a 0-dim f32 zero, column-major kept): what it holds without;
//! - x: the MoE block's input [1, S, hidden] bf16, a layer-major span's normed rows (an array of its own);
//! - the served logits: the host's stock call, `router.tableF32` over the view (stockTable: the astype, then + 0) and
//!   Linear's matmul(x, table) (MLX casts x to f32 and runs its fused NAX GEMM over the column-major table), at the
//!   span widths where MLX runs its regular kernel: 4096 (a prompt chunk), 1366 (the fewest; 1365 and fewer take MLX's
//!   NAX split K), the standard request's 1965-row span, the tails of 1996 and 2253, 3063, and 4095 and 4097 (a row
//!   off the tile).
//! GPU tests (TURBOEXL3_GPU=1, else skipped): those words over the f16 weight and over the f32 table, and the refusals.
//! Timing (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; asserts nothing, guidance only): the stock call against `logits`
//! at 4096 and 2253 rows, min and median us over 8 rounds after 2 warm. Lines: TURBOEXL3_LING_ROUTER_PROMPT,
//! TURBOEXL3_LING_ROUTER_PROMPT_TIME.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const T = mlx.mlx_array;
const router = @import("router.zig");
const rp = @import("router_prompt.zig");
const pgemm = @import("../../core/pgemm.zig");

const testing = std.testing;
const hidden: usize = 4096;
const ne: usize = 512;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Every finite f16 word, shuffled into the weight (-0.0 and every subnormal among them)
fn fillAllWords(rnd: std.Random, words: []u16) void {
    var i: usize = 0;
    var v: u32 = 0;
    while (i < words.len) : (v = (v + 1) & 0xFFFF) {
        const w: u16 = @intCast(v);
        if ((w & 0x7C00) == 0x7C00) continue;
        words[i] = w;
        i += 1;
    }
    rnd.shuffle(u16, words);
}

/// x's rows by pattern: normal values at a scale from small to saturating, 0 a row of zeros
const row_scales = [_]f32{ 1.0, 8.0, 0.0, 0.25, 32.0, 2.0, 64.0, 0.5 };

fn fillX(rnd: std.Random, bits: []u16) void {
    for (bits, 0..) |*b, i| b.* = support.bf16Bits(rnd.floatNorm(f32) * row_scales[(i / hidden) % row_scales.len]);
}

/// The words of `got` that differ from `want`, the first few printed
fn differing(label: []const u8, rows: usize, got: []const f32, want: []const f32) usize {
    var n: usize = 0;
    for (got, want, 0..) |u, v, i| {
        const bu: u32 = @bitCast(u);
        const bv: u32 = @bitCast(v);
        if (bu == bv) continue;
        if (n < 6) std.debug.print("TURBOEXL3_LING_ROUTER_PROMPT {s} rows={d} differs at row={d} col={d}: got 0x{x:0>8} want 0x{x:0>8}\n", .{ label, rows, i / ne, i % ne, bu, bv });
        n += 1;
    }
    return n;
}

/// The loader's weight, its view and its f32 table, and the stock call's table rebuilt over the view
const Operands = struct { view: T, table: T, rebuilt: T };

fn operands(g: *G, s: mlx.mlx_stream, words: []const u16) !Operands {
    const wl = try g.hostArray(std.mem.sliceAsBytes(words), &.{ ci(ne), ci(hidden) }, .float16);
    const view = try g.transposeAxes(wl, &.{ 1, 0 });
    const table = try g.add(try g.astype(view, .float32), try g.zeros(&.{}, .float32));
    const rebuilt = try g.adopt(try router.tableF32(s, view));
    try g.evalAll(&.{ table, rebuilt });
    return .{ .view = view, .table = table, .rebuilt = rebuilt };
}

/// A gemm_loop text's statements, from its first (`first`, TM's) through its last (`last`, the return)
fn loopStatements(text: []const u8, first: []const u8, last: []const u8) []const u8 {
    const from = std.mem.indexOf(u8, text, first).?;
    const end = std.mem.lastIndexOf(u8, text, last).? + last.len;
    return text[from..end];
}

fn columnMajor(t: T) bool {
    const sh = mlx.getShape(t);
    const st = mlx.mlx_array_strides(t);
    return sh.len == 2 and sh[0] == ci(hidden) and sh[1] == ci(ne) and st[0] == 1 and st[1] == hidden;
}

test "turboexl3 ling router_prompt texts: the loop is MLX's gemm_loop (pg2's verbatim copy) but for its f32 tiles loaded from each operand's own type and B's -0.0 stored +0.0; the kernel attribute's macro closes the header and the body undefines it" {
    const pin = comptime support.pins(@embedFile("kernels/test/pins/router_prompt.metal"));
    const has = struct {
        fn f(hay: []const u8, needle: []const u8) bool {
            return std.mem.indexOf(u8, hay, needle) != null;
        }
    }.f;
    // the loop's statements from its first to its return: pg2's (MLX's), with the tiles' type and the + 0 after B's load
    const mine = loopStatements(rp.text_loop, pin[12], pin[13]);
    const theirs = loopStatements(pgemm.text_loop, pin[12], pin[13]);
    const a = testing.allocator;
    var norm: []u8 = try a.dupe(u8, mine);
    defer a.free(norm);
    const swaps = [_][2][]const u8{ .{ pin[0], pin[1] }, .{ pin[2], pin[3] }, .{ pin[4], pin[5] }, .{ pin[6] ++ "\n", "" } };
    for (swaps) |sw| {
        const n = try std.mem.replaceOwned(u8, a, norm, sw[0], sw[1]);
        a.free(norm);
        norm = n;
    }
    try testing.expectEqualStrings(theirs, norm);
    // two B loads, each followed by the + 0; two MMAs
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rp.text_loop, pin[6]));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rp.text_loop, pin[7]));
    // the header ends with the attribute's macro, the body's last line undefines it, and the body runs MLX's store
    try testing.expect(std.mem.endsWith(u8, rp.text_loop, pin[8] ++ "\n"));
    try testing.expect(std.mem.endsWith(u8, rp.text_gemm, pin[9] ++ "\n"));
    try testing.expect(has(rp.text_gemm, pin[10]) and has(rp.text_gemm, pin[11]));
}

test "turboexl3 gpu ling router_prompt: logits over the loaded f16 weight's view and over the loader's f32 table gives the stock call's words (tableF32, then MLX's matmul) at every span width MLX runs its regular kernel at (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x11C0_9A0B);
    const rnd = prng.random();
    const words = try a.alloc(u16, ne * hidden);
    defer a.free(words);
    fillAllWords(rnd, words);
    const op = try operands(&g, s, words);
    try testing.expect(columnMajor(op.view) and columnMajor(op.table) and columnMajor(op.rebuilt));
    const widths = [_]usize{ 4096, 1366, 1965, 1996, 2253, 3063, 4095, 4097 };
    const max_rows: usize = 4097;
    const xbits = try a.alloc(u16, max_rows * hidden);
    defer a.free(xbits);
    fillX(rnd, xbits);
    const want = try a.alloc(f32, max_rows * ne);
    defer a.free(want);
    const got = try a.alloc(f32, max_rows * ne);
    defer a.free(got);
    var failed = false;
    for (widths) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const x3 = try g.hostArray(std.mem.sliceAsBytes(xbits[0 .. rows * hidden]), &.{ 1, ci(rows), ci(hidden) }, .bfloat16);
        const n = rows * ne;
        const ref = try g.matmul(x3, op.rebuilt);
        _ = try g.hostF32(ref, want[0..n]);
        for ([_]bool{ true, false }) |f16w| {
            const w = if (f16w) op.view else op.table;
            const call = rp.takes(x3, w);
            const y = if (try rp.logits(s, x3, w)) |yy| try g.adopt(yy) else {
                std.debug.print("TURBOEXL3_LING_ROUTER_PROMPT rows={d} w={s} arch={c} declined FAIL\n", .{ rows, if (f16w) "f16" else "f32", rp.archLetter() orelse '?' });
                failed = true;
                continue;
            };
            const ysh = mlx.getShape(y);
            const shaped = ysh.len == 3 and ysh[0] == 1 and ysh[1] == ci(rows) and ysh[2] == ci(ne);
            _ = try g.hostF32(y, got[0..n]);
            const d = differing(if (f16w) "f16" else "f32", rows, got[0..n], want[0..n]);
            const t = call.?.tile;
            std.debug.print("TURBOEXL3_LING_ROUTER_PROMPT rows={d} w={s} arch={c} tile={d}x{d}x{d} wm={d} wn={d} swz={d} shape_ok={} differing={d}/{d} {s}\n", .{ rows, if (f16w) "f16" else "f32", rp.archLetter() orelse '?', t.bm, t.bn, t.bk, t.wm, t.wn, t.swz, shaped, d, n, if (d == 0 and shaped) "PASS" else "FAIL" });
            if (d != 0 or !shaped) failed = true;
        }
    }
    try testing.expect(!failed);
}

test "turboexl3 gpu ling router_prompt refusals: one row and every span MLX runs on its NAX split K (1365 rows and fewer), x not bf16, a row-major or bf16 weight (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const view = try g.transposeAxes(try g.zeros(&.{ ci(ne), ci(hidden) }, .float16), &.{ 1, 0 });
    const w_rows = try g.zeros(&.{ ci(hidden), ci(ne) }, .float16);
    const w_bf16 = try g.transposeAxes(try g.zeros(&.{ ci(ne), ci(hidden) }, .bfloat16), &.{ 1, 0 });
    // Evaluated first, as the served weights are (the loader's evalTrunk): MLX gives an array it has not evaluated
    // row-contiguous strides, so a view shows its own layout only once evaluated.
    try g.evalAll(&.{ view, w_rows, w_bf16 });
    const x4k = try g.zeros(&.{ 1, 4096, ci(hidden) }, .bfloat16);
    try testing.expect(rp.takes(x4k, view) != null);
    for ([_]usize{ 1, 2, 31, 1024, 1365 }) |rows| {
        const x3 = try g.zeros(&.{ 1, ci(rows), ci(hidden) }, .bfloat16);
        try testing.expect((try rp.logits(s, x3, view)) == null);
    }
    try testing.expect((try rp.logits(s, try g.zeros(&.{ 1, 4096, ci(hidden) }, .float32), view)) == null);
    try testing.expect((try rp.logits(s, x4k, w_rows)) == null);
    try testing.expect((try rp.logits(s, x4k, w_bf16)) == null);
    std.debug.print("TURBOEXL3_LING_ROUTER_PROMPT refusals arch={c} PASS\n", .{rp.archLetter() orelse '?'});
}

const TimeCtx = struct { s: mlx.mlx_stream, x: T, view: T };

/// `f(g, ctx)` timed on the device: per round its graph built and evaluated alone (the round's arrays freed after), 2
/// warm-up rounds, then the min and median us over 8
fn timeCall(g: *G, ctx: TimeCtx, comptime f: fn (*G, TimeCtx) anyerror!T) ![2]f64 {
    var t: [8]f64 = undefined;
    for (0..t.len + 2) |round| {
        const m = g.mark();
        defer g.resetTo(m);
        const t0 = nowNs();
        const y = try f(g, ctx);
        try g.evalAll(&.{y});
        const dt = nowNs() - t0;
        if (round >= 2) t[round - 2] = @as(f64, @floatFromInt(dt)) / 1000.0;
    }
    std.mem.sort(f64, &t, {}, std.sort.asc(f64));
    return .{ t[0], t[t.len / 2] };
}

// Timing (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; it asserts nothing): guidance for the serving plan, never a speed
// verdict (the standard test through the outside plugin decides speed). The stock call (the table rebuilt over the
// f16 weight's view, then Linear's matmul) against `logits` over the view, at a prompt chunk's 4096 rows and a tail's
// 2253.
test "turboexl3 gpu ling timing: router_prompt's logits in one launch against the router's stock call at prompt widths (tableF32, then MLX's matmul) (TURBOEXL3_LING_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_LING_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x11C0_71AE);
    const rnd = prng.random();
    const words = try a.alloc(u16, ne * hidden);
    defer a.free(words);
    for (words) |*v| v.* = @bitCast(@as(f16, @floatCast(rnd.floatNorm(f32) * 0.02)));
    const op = try operands(&g, s, words);
    const xbits = try a.alloc(u16, 4096 * hidden);
    defer a.free(xbits);
    fillX(rnd, xbits);
    for ([_]usize{ 4096, 2253 }) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        defer _ = mlx.mlx_clear_cache();
        const x3 = try g.hostArray(std.mem.sliceAsBytes(xbits[0 .. rows * hidden]), &.{ 1, ci(rows), ci(hidden) }, .bfloat16);
        try g.evalAll(&.{x3});
        const ctx: TimeCtx = .{ .s = s, .x = x3, .view = op.view };
        const stock = try timeCall(&g, ctx, struct {
            fn f(gg: *G, cc: TimeCtx) anyerror!T {
                return gg.matmul(cc.x, try gg.adopt(try router.tableF32(cc.s, cc.view)));
            }
        }.f);
        const one = try timeCall(&g, ctx, struct {
            fn f(gg: *G, cc: TimeCtx) anyerror!T {
                return gg.adopt((try rp.logits(cc.s, cc.x, cc.view)).?);
            }
        }.f);
        const flops = 2.0 * @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(ne));
        std.debug.print("TURBOEXL3_LING_ROUTER_PROMPT_TIME rows={d} stock_min_us={d:.1} stock_med_us={d:.1} one_min_us={d:.1} one_med_us={d:.1} saved_min_us={d:.1} saved_med_us={d:.1} stock_tflops={d:.2} one_tflops={d:.2}\n", .{ rows, stock[0], stock[1], one[0], one[1], stock[0] - one[0], stock[1] - one[1], flops / (stock[0] * 1e6), flops / (one[0] * 1e6) });
    }
}
