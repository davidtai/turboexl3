//! `kdaProjPrompt` (P4, the outside plugin turboexl3_ling's MLX_SERVE_LING_KDA_PROJ_PROMPT): a KDA layer's q|k|v, f
//! and g at prompt rows in one launch past their prepares and W_hats, held word for word to each member's `linear`.
//! - GPU (TURBOEXL3_GPU=1, else skipped), at Ling-3.1-flash's KDA shapes (q|k|v 4096 -> 24576, f and g 4096 -> 8192 at
//!   K 6, the loader's I16 trellis words) with x the layer's normed input [1, S, 4096] bf16: at S 33 (the first rows past
//!   the register path), 100, 1000 and 1965 (rows that end inside a tile; 1965 the standard prompt's last layer-major
//!   call), 512, 4096 and 16384 (the plugin's chunk and layer-major calls), each member's output against `root.linear`'s
//!   (the served chain: prepare, W_hat, MLX's matmul, finish), every word, both finite; and the refusal at 32 rows.
//!   Lines: TURBOEXL3_LING_KDA_PROMPT.
//! - GPU, over given W_hats (`kdaProjPromptW`): each member's W_hat decoded once by `Dense.dequantize` and read by every
//!   call, at rows 33, 1965 and 4096 each member's y against `kdaProjPrompt`'s word for word; the declines at 16, 17 and
//!   32 rows, a W_hat of another width refused, and a red check (g's call over f's W_hat gives other words). Lines:
//!   TURBOEXL3_LING_KDA_PROMPT_W (then TURBOEXL3_LING_KDA_PROMPT_W_DONE, 10 lines).
//! - Timing (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; asserts nothing, guidance only): the served chain's three
//!   `linear` calls against `kdaProjPrompt` at each tile of a short sweep, interleaved round by round, at rows 64 to
//!   16384 (`time_rows`); min and median ms over 8 rounds. Lines: TURBOEXL3_LING_TIME kdaproj_prompt.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const T = mlx.mlx_array;
const config = @import("../../core/config.zig");
const format = @import("../../core/format.zig");
const pgemm = @import("../../core/pgemm.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const ds_test = @import("../deepseek_v41/deepseek_test.zig");
const kdaproj_pf = @import("kdaproj_pf.zig");

const testing = std.testing;

const in: u32 = 4096;
const outs = [3]u32{ 24576, 8192, 8192 };

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// A synthetic projection and the operands `linear` takes for it
const Served = struct { lin: ref.Linear, p: root.Proj };

/// A projection `linear` takes at (in, out, K 6): its synthetic trellis as the loader hands it (the pack's I16 words)
fn servedProj(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, rnd: std.Random, o: u32) !Served {
    var lin = try ref.synthLinear(a, rnd, 1, in, o, format.Rate.fromN(96).?, .mul1);
    errdefer lin.deinit(a);
    const u = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(in / 16), ci(o / 16), 96 }, .uint16);
    var v = mlx.mlx_array_new();
    mlx.check(mlx.mlx_view(&v, u, .int16, s)) catch |e| {
        _ = mlx.mlx_array_free(v);
        return e;
    };
    return .{ .lin = lin, .p = .{
        .trellis = try g.adopt(v),
        .suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(o)}, .float16),
    } };
}

/// Words of `got` and `want` (one size; compared as f32) that differ, and the values of either that are not finite
fn compare(a: std.mem.Allocator, g: *G, got: T, want: T) !struct { differ: usize, nonfinite: usize } {
    const n = mlx.mlx_array_size(want);
    if (mlx.mlx_array_size(got) != n) return error.SizeMismatch;
    const p = try a.alloc(f32, n);
    defer a.free(p);
    const q = try a.alloc(f32, n);
    defer a.free(q);
    _ = try g.hostF32(try g.astype(got, .float32), p);
    _ = try g.hostF32(try g.astype(want, .float32), q);
    var d: usize = 0;
    var nf: usize = 0;
    for (p, q) |u, w| {
        d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
        nf += @as(usize, @intFromBool(!std.math.isFinite(u))) + @intFromBool(!std.math.isFinite(w));
    }
    return .{ .differ = d, .nonfinite = nf };
}

fn sameShape(x: T, y: T) bool {
    return std.mem.eql(c_int, mlx.getShape(x), mlx.getShape(y));
}

test "turboexl3 gpu ling kdaproj prompt: kdaProjPrompt's q|k|v, f and g are each member's `linear` word for word at prompt rows, operands as the loader keeps them (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s) or !config.get().nax) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x9F4A11);
    const rnd = prng.random();
    var ms: [3]Served = undefined;
    var made: usize = 0;
    defer for (ms[0..made]) |*m| m.lin.deinit(a);
    for (&ms, outs) |*m, o| {
        m.* = try servedProj(a, &g, s, rnd, o);
        made += 1;
    }
    const p = [3]root.Proj{ ms[0].p, ms[1].p, ms[2].p };
    var failed: usize = 0;
    for ([_]u32{ 33, 100, 512, 1000, 1965, 4096, 16384 }) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const x = try ds_test.randBf16(a, &g, rnd, &.{ 1, ci(rows), ci(in) });
        const got = try kdaproj_pf.kdaProjPrompt(s, x, p, .mul1);
        var gys: [3]T = undefined;
        for (&gys, got) |*d, y| d.* = try g.adopt(y);
        for (p, gys, outs, 0..) |m, gy, o, i| {
            const want = try g.adopt(try root.linear(s, x, m.trellis, m.suh, m.svh, .mul1));
            const c = try compare(a, &g, gy, want);
            const ok = c.differ == 0 and c.nonfinite == 0 and sameShape(gy, want);
            failed += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_LING_KDA_PROMPT rows={d} member={d} out={d} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, i, o, c.differ, mlx.mlx_array_size(want), c.nonfinite, if (ok) "PASS" else "FAIL" });
        }
    }
    {
        const mark = g.mark();
        defer g.resetTo(mark);
        const x = try ds_test.randBf16(a, &g, rnd, &.{ 1, 32, ci(in) });
        try testing.expectError(error.Exl3EngineNotImplemented, kdaproj_pf.kdaProjPrompt(s, x, p, .mul1));
        std.debug.print("TURBOEXL3_LING_KDA_PROMPT rows=32 refused (the register path's rows) PASS\n", .{});
    }
    std.debug.print("TURBOEXL3_LING_KDA_PROMPT_DONE failed={d} {s}\n", .{ failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 0), failed);
}

test "turboexl3 gpu ling kdaproj prompt over given W_hats: kdaProjPromptW over each member's W_hat, decoded once by Dense.dequantize and read by every call, gives kdaProjPrompt's words at prompt rows 33, 1965 and 4096, operands as the loader keeps them; its declines at 16, 17 and 32 rows, its refusal, and a swapped W_hat caught (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s) or !config.get().nax) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x9F4A13);
    const rnd = prng.random();
    var ms: [3]Served = undefined;
    var made: usize = 0;
    defer for (ms[0..made]) |*m| m.lin.deinit(a);
    for (&ms, outs) |*m, o| {
        m.* = try servedProj(a, &g, s, rnd, o);
        made += 1;
    }
    const p = [3]root.Proj{ ms[0].p, ms[1].p, ms[2].p };
    // each member's W_hat by its `linear`'s decode launch, evaluated once: every call below reads it
    var w_hat: [3]T = undefined;
    for (&w_hat, p, outs) |*wm, m, o| {
        const dn = try root.denseCached(.{ .in = in, .out = o, .rate = format.Rate.fromN(96).?, .dec = .mul1, .out_dtype = .bfloat16 });
        wm.* = try g.adopt(try dn.dequantize(s, m.trellis));
    }
    try g.evalAll(&w_hat);
    var lines: usize = 0;
    var failed: usize = 0;
    for ([_]u32{ 33, 1965, 4096 }) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const x = try ds_test.randBf16(a, &g, rnd, &.{ 1, ci(rows), ci(in) });
        const want = try kdaproj_pf.kdaProjPrompt(s, x, p, .mul1);
        var wys: [3]T = undefined;
        for (&wys, want) |*dst, y| dst.* = try g.adopt(y);
        const got = try kdaproj_pf.kdaProjPromptW(s, x, p, w_hat, .mul1);
        var gys: [3]T = undefined;
        for (&gys, got) |*dst, y| dst.* = try g.adopt(y);
        for (gys, wys, outs, 0..) |gy, wy, o, i| {
            const c = try compare(a, &g, gy, wy);
            const ok = c.differ == 0 and c.nonfinite == 0 and sameShape(gy, wy);
            lines += 1;
            failed += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_LING_KDA_PROMPT_W rows={d} member={d} out={d} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, i, o, c.differ, mlx.mlx_array_size(wy), c.nonfinite, if (ok) "PASS" else "FAIL" });
        }
    }
    {
        const mark = g.mark();
        defer g.resetTo(mark);
        // the declines, before any launch: the GEMV's and the register path's rows, where kdaProjPrompt declines too
        for ([_]u32{ 16, 17, 32 }) |rows| {
            const x = try ds_test.randBf16(a, &g, rnd, &.{ 1, ci(rows), ci(in) });
            try testing.expectError(error.Exl3EngineNotImplemented, kdaproj_pf.kdaProjPromptW(s, x, p, w_hat, .mul1));
        }
        // the refusal, before any launch: a W_hat of another member's width (q|k|v's in f's place)
        const x33 = try ds_test.randBf16(a, &g, rnd, &.{ 1, 33, ci(in) });
        var wide = w_hat;
        wide[1] = w_hat[0];
        try testing.expectError(error.BadExl3Shape, kdaproj_pf.kdaProjPromptW(s, x33, p, wide, .mul1));
        // red: g's call over f's W_hat (the same width) does not give g's words, so a wrong W_hat would be caught
        const want = try kdaproj_pf.kdaProjPrompt(s, x33, p, .mul1);
        var wys: [3]T = undefined;
        for (&wys, want) |*dst, y| dst.* = try g.adopt(y);
        var swapped = w_hat;
        swapped[2] = w_hat[1];
        const got = try kdaproj_pf.kdaProjPromptW(s, x33, p, swapped, .mul1);
        var gys: [3]T = undefined;
        for (&gys, got) |*dst, y| dst.* = try g.adopt(y);
        const c = try compare(a, &g, gys[2], wys[2]);
        const caught = c.differ > 0;
        lines += 1;
        failed += @intFromBool(!caught);
        std.debug.print("TURBOEXL3_LING_KDA_PROMPT_W red: g over f's W_hat at 33 rows differing={d}/{d} (want > 0) {s}\n", .{ c.differ, mlx.mlx_array_size(wys[2]), if (caught) "PASS" else "FAIL" });
        std.debug.print("TURBOEXL3_LING_KDA_PROMPT_W rows=16,17,32 declined, a W_hat of another width refused PASS\n", .{});
    }
    std.debug.print("TURBOEXL3_LING_KDA_PROMPT_W_DONE lines={d} failed={d} {s}\n", .{ lines, failed, if (failed == 0) "BYTES_OK" else "BYTES_DIFFER" });
    try testing.expectEqual(@as(usize, 10), lines);
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── timing (TURBOEXL3_LING_TIME=1): guidance for the tile, never a speed verdict ──

/// The sweep: pg2's shapes for its K 5120 and 8192 entries and their neighbours, each admitted
const sweep = [_]pgemm.Tile{
    .{ .bm = 32, .bn = 256, .bk = 512, .wm = 1, .wn = 4, .swz = 2 },
    .{ .bm = 32, .bn = 256, .bk = 256, .wm = 1, .wn = 4, .swz = 2 },
    .{ .bm = 32, .bn = 128, .bk = 512, .wm = 1, .wn = 4, .swz = 2 },
    .{ .bm = 64, .bn = 256, .bk = 512, .wm = 2, .wn = 4, .swz = 2 },
    .{ .bm = 64, .bn = 256, .bk = 256, .wm = 2, .wn = 4, .swz = 2 },
    .{ .bm = 64, .bn = 128, .bk = 512, .wm = 2, .wn = 2, .swz = 2 },
    .{ .bm = 128, .bn = 128, .bk = 512, .wm = 4, .wn = 2, .swz = 2 },
};

/// The sweep's rows: powers of two from 64 to 16384 and three sizes that end inside a tile, so a tile is chosen by a
/// rule over prompt widths (one tile, or a threshold on rows), never by one prompt's call sizes
const time_rows = [_]u32{ 64, 256, 777, 1024, 3001, 4096, 8192, 12345, 16384 };

const Arms = struct {
    s: mlx.mlx_stream,
    x: T,
    p: [3]root.Proj,

    /// arm 0: the served chain's three `linear` calls; arm 1 + i: `kdaProjPrompt` at sweep[i]
    fn call(cx: Arms, arm: usize) ![3]T {
        if (arm == 0) {
            var ys: [3]T = undefined;
            var made: usize = 0;
            errdefer for (ys[0..made]) |y| {
                _ = mlx.mlx_array_free(y);
            };
            for (&ys, cx.p) |*y, m| {
                y.* = try root.linear(cx.s, cx.x, m.trellis, m.suh, m.svh, .mul1);
                made += 1;
            }
            return ys;
        }
        kdaproj_pf.tile_override.get().* = sweep[arm - 1];
        defer kdaproj_pf.tile_override.get().* = null;
        return kdaproj_pf.kdaProjPrompt(cx.s, cx.x, cx.p, .mul1);
    }
};

test "turboexl3 gpu ling timing: kdaProjPrompt against the served chain's three linear calls at prompt rows, a short tile sweep (TURBOEXL3_LING_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_LING_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s) or !config.get().nax) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x9F4A12);
    const rnd = prng.random();
    var p: [3]root.Proj = undefined;
    for (&p, outs) |*m, o| m.* = try ds_test.randProj(a, &g, rnd, null, in, o, 96);
    const n_arms = 1 + sweep.len;
    const rounds = 10;
    for (time_rows) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        defer _ = mlx.mlx_clear_cache();
        const cx: Arms = .{ .s = s, .x = try ds_test.randBf16(a, &g, rnd, &.{ 1, ci(rows), ci(in) }), .p = p };
        var samples: [n_arms][rounds - 2]f64 = undefined;
        for (0..rounds) |round| for (0..n_arms) |arm| {
            const ys = try cx.call(arm);
            defer for (ys) |y| {
                _ = mlx.mlx_array_free(y);
            };
            const t0 = ds_test.nowNs();
            try g.evalAll(&ys);
            const dt = ds_test.nowNs() - t0;
            if (round >= 2) samples[arm][round - 2] = @as(f64, @floatFromInt(dt)) / 1e6;
        };
        var chain_min: f64 = 0;
        for (&samples, 0..) |*sm, arm| {
            std.mem.sort(f64, sm, {}, std.sort.asc(f64));
            const min = sm[0];
            const med = sm[sm.len / 2];
            if (arm == 0) {
                chain_min = min;
                std.debug.print("TURBOEXL3_LING_TIME kdaproj_prompt rows={d} arm=chain min_ms={d:.3} med_ms={d:.3}\n", .{ rows, min, med });
            } else {
                const t = sweep[arm - 1];
                std.debug.print("TURBOEXL3_LING_TIME kdaproj_prompt rows={d} arm=pf tile={d}x{d}x{d} wm={d} wn={d} swz={d} min_ms={d:.3} med_ms={d:.3} of_chain={d:.3}\n", .{ rows, t.bm, t.bn, t.bk, t.wm, t.wn, t.swz, min, med, min / chain_min });
            }
        }
    }
}
