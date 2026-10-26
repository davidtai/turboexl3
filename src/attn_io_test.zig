//! The fused attention projections' GPU tests (`attn_io.zig`; TURBOEXL3_GPU=1, else skipped): ATTN-IN and ATTN-OUT against
//! the chain they replace, byte for byte, on synthetic trellises at Qwen3.8-Flash-Next's attention geometry (hidden 2560, 24
//! query heads [query | gate], 2 kv heads, head 256, RoPE on 64 dims, theta 1e7). The chain is the host's: `root.linear` per
//! projection, then MLX's reshape, split, rms_norm, transpose and rope (the host takes its hd-256 QK-norm+RoPE kernel at
//! <= 32 rows, pinned to this chain by its own parity test), and for the tail MLX's sigmoid, multiply, reshape and
//! `root.linear`. Row counts: a 17.7K-token prompt's prefill chunks (8192 at offsets 0 and 8192, 1328 at 16384, the 31-row
//! tail at 17712) and the path boundaries (17 and 32: the register-path GEMM; 33: dequantize + matmul).
//! TURBOEXL3_ATTN_ROWS=<r,r,...> keeps only those row counts. Lines: TURBOEXL3_ATTN_IN and TURBOEXL3_ATTN_OUT per case
//! (differing elements / total, PASS when 0), then the four-member prepare's TURBOEXL3_IDXPREP_* lines.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("test_support.zig");
const G = support.Graph;
const bf16Bits = support.bf16Bits;
const format = @import("format.zig");
const root = @import("root.zig");
const dense = @import("dense.zig");

const testing = std.testing;

const HIDDEN: u32 = 2560;
const HQ: u32 = 24;
const HK: u32 = 2;
const HD: u32 = 256;
const RD: c_int = 64;
const THETA: f32 = 10_000_000.0;
const EPS: f32 = 1.0e-6;

const Cell = struct { rows: u32, offset: c_int };
const cells = [_]Cell{
    .{ .rows = 8192, .offset = 0 },
    .{ .rows = 8192, .offset = 8192 },
    .{ .rows = 1328, .offset = 16384 },
    .{ .rows = 31, .offset = 17712 },
    .{ .rows = 17, .offset = 3 },
    .{ .rows = 32, .offset = 77 },
    .{ .rows = 33, .offset = 1000 },
};

/// The members' rates: the full pack's attention projections are K7 (n 112); the second set gives each member its own rate.
const Rates = struct { q: u32, k: u32, v: u32, o: u32 };
const rate_sets = [_]Rates{ .{ .q = 112, .k = 112, .v = 112, .o = 112 }, .{ .q = 96, .k = 80, .v = 128, .o = 64 } };

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn rowsWanted(rows: u32) bool {
    const only = std.c.getenv("TURBOEXL3_ATTN_ROWS") orelse return true;
    var it = std.mem.splitScalar(u8, std.mem.span(only), ',');
    while (it.next()) |tok| {
        if ((std.fmt.parseInt(u32, tok, 10) catch continue) == rows) return true;
    }
    return false;
}

/// A dense EXL3 projection in -> out at rate n as the host holds it: trellis U16 [in/16, out/16, n] (every pattern is a valid
/// codeword), suh F16 [in] in [0.5, 1.5], svh F16 [out] in [0.5, 1.5] / sqrt(in) (outputs O(1)).
fn synthProj(a: std.mem.Allocator, g: *G, rnd: std.Random, in: u32, out: u32, n: u32) !root.Proj {
    const words = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(words);
    for (words) |*w| w.* = rnd.int(u16);
    const suh = try a.alloc(u16, in);
    defer a.free(suh);
    for (suh) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const svh = try a.alloc(u16, out);
    defer a.free(svh);
    const sc: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(in)));
    for (svh) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * sc);
    return .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(words), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(suh), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(svh), &.{ci(out)}, .float16),
    };
}

/// A bf16 array of `shape` with bf16-exact values lo + span * N(0,1) (or lo + span * U[0,1) when `uniform`).
fn randBf16(a: std.mem.Allocator, g: *G, rnd: std.Random, shape: []const c_int, lo: f32, span: f32, uniform: bool) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try a.alloc(f32, n);
    defer a.free(buf);
    for (buf) |*v| {
        const r = if (uniform) rnd.float(f32) else rnd.floatNorm(f32);
        v.* = @bitCast(@as(u32, bf16Bits(lo + span * r)) << 16);
    }
    return g.astype(try g.hostArray(std.mem.sliceAsBytes(buf), shape, .float32), .bfloat16);
}

/// The host's `ropeAngleRows`: MLX's rope over a [1, 1, rows, RD] probe holding 1 in each pair's first half, so row p is
/// cos | sin at position offset + p, MLX's own floats; flattened to [rows * RD].
fn angleRows(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, rows: u32, offset: c_int) !mlx.mlx_array {
    const n: usize = @intCast(RD);
    const nrows: usize = rows;
    const buf = try a.alloc(f32, n * nrows);
    defer a.free(buf);
    for (buf, 0..) |*v, i| v.* = if ((i % n) < n / 2) 1.0 else 0.0;
    const probe = try g.hostArray(std.mem.sliceAsBytes(buf), &.{ 1, 1, ci(rows), RD }, .float32);
    var rot = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(&rot, probe, RD, false, .{ .value = THETA, .has_value = true }, 1.0, offset, .{ .ctx = null }, s));
    return g.reshape(try g.adopt(rot), &.{ci(nrows * n)});
}

/// The number of elements whose bits differ (bf16 / f16 -> f32 is one-to-one on bit patterns), after a shape check.
fn bitsDiff(a: std.mem.Allocator, g: *G, want: mlx.mlx_array, got: mlx.mlx_array) !usize {
    if (!std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got)) or mlx.mlx_array_dtype(want) != mlx.mlx_array_dtype(got)) return std.math.maxInt(usize);
    const n = mlx.mlx_array_size(want);
    const wb = try a.alloc(f32, n);
    defer a.free(wb);
    const gb = try a.alloc(f32, n);
    defer a.free(gb);
    _ = try g.hostF32(try g.astype(want, .float32), wb);
    _ = try g.hostF32(try g.astype(got, .float32), gb);
    var diff: usize = 0;
    for (wb, gb) |p, q| {
        if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) diff += 1;
    }
    return diff;
}

const ChainIn = struct { qp: mlx.mlx_array, q: mlx.mlx_array, gate: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array };

/// The host's composed chain for q|gate, k and v at prefill rows (gatedFullAttnProjected's > 32-row arm).
fn chainIn(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, q: root.Proj, k: root.Proj, v: root.Proj, qw: mlx.mlx_array, kw: mlx.mlx_array, rows: u32, offset: c_int) !ChainIn {
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const base = mlx.mlx_optional_float{ .value = THETA, .has_value = true };
    const qp = try g.adopt(try root.linear(s, x, q.trellis, q.suh, q.svh, .mul1));
    const qg = try g.reshape(qp, &.{ 1, ci(rows), ci(HQ), ci(2 * HD) });
    var split = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(split);
    try mlx.check(mlx.mlx_split(&split, qg, 2, -1, s));
    var queries = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&queries, split, 0));
    _ = try g.adopt(queries);
    var gate = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&gate, split, 1));
    _ = try g.adopt(gate);
    var outs: [2]mlx.mlx_array = undefined;
    const kp = try g.adopt(try root.linear(s, x, k.trellis, k.suh, k.svh, .mul1));
    const kr = try g.reshape(kp, &.{ 1, ci(rows), ci(HK), ci(HD) });
    for ([_]mlx.mlx_array{ queries, kr }, [_]mlx.mlx_array{ qw, kw }, &outs) |src, w, *o| {
        var n = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&n, src, w, EPS, s));
        const t = try g.transposeAxes(try g.adopt(n), &perm);
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rope(&r, t, RD, false, base, 1.0, offset, .{ .ctx = null }, s));
        o.* = try g.adopt(r);
    }
    const vp = try g.adopt(try root.linear(s, x, v.trellis, v.suh, v.svh, .mul1));
    const vt = try g.transposeAxes(try g.reshape(vp, &.{ 1, ci(rows), ci(HK), ci(HD) }), &perm);
    return .{ .qp = qp, .q = outs[0], .gate = gate, .k = outs[1], .v = vt };
}

/// The host's composed tail at prefill rows (gatedAttnTail's > 16-row arm): sigmoid(gate), attn^T * it, flatten, o_proj.
fn chainOut(g: *G, s: mlx.mlx_stream, attn: mlx.mlx_array, gate4: mlx.mlx_array, o: root.Proj, rows: u32) !mlx.mlx_array {
    const at = try g.transposeAxes(attn, &.{ 0, 2, 1, 3 });
    var sig = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&sig, gate4, s));
    var prod = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&prod, at, try g.adopt(sig), s));
    const flat = try g.reshape(try g.adopt(prod), &.{ 1, ci(rows), ci(HQ * HD) });
    return g.adopt(try root.linear(s, flat, o.trellis, o.suh, o.svh, .mul1));
}

/// MLX's sigmoid of every bf16 pattern (the host's `swigluSigTable`).
fn sigTable(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream) !mlx.mlx_array {
    const pats = try a.alloc(f32, 1 << 16);
    defer a.free(pats);
    for (pats, 0..) |*v, i| v.* = @bitCast(@as(u32, @intCast(i)) << 16);
    const vals = try g.astype(try g.hostArray(std.mem.sliceAsBytes(pats), &.{1 << 16}, .float32), .bfloat16);
    var t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&t, vals, s));
    const tab = try g.adopt(t);
    try g.evalAll(&.{tab});
    return tab;
}

fn pathOf(rows: u32) []const u8 {
    return if (rows <= 16) "gemv" else if (rows <= 32) "regpath" else "dequant+matmul";
}

test "turboexl3 gpu attn_io: the fused q|gate, k, v projections with the norm + RoPE finish, and o_proj over the gated prepare, equal the host's chain byte for byte (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var failed = false;
    const eps = mlx.mlx_array_new_float(EPS);
    defer _ = mlx.mlx_array_free(eps);
    const top = g.mark();
    defer g.resetTo(top);
    const sigtab = try sigTable(a, &g, s);
    for (rate_sets, 0..) |rs, ri| {
        var prng = std.Random.DefaultPrng.init(0xA77E10 + ri);
        const rnd = prng.random();
        const set_mark = g.mark();
        defer g.resetTo(set_mark);
        const q = try synthProj(a, &g, rnd, HIDDEN, 2 * HQ * HD, rs.q);
        const k = try synthProj(a, &g, rnd, HIDDEN, HK * HD, rs.k);
        const v = try synthProj(a, &g, rnd, HIDDEN, HK * HD, rs.v);
        const o = try synthProj(a, &g, rnd, HQ * HD, HIDDEN, rs.o);
        const qw = try randBf16(a, &g, rnd, &.{ci(HD)}, 0.5, 1.0, true);
        const kw = try randBf16(a, &g, rnd, &.{ci(HD)}, 0.5, 1.0, true);
        for (cells) |c| {
            if (!rowsWanted(c.rows)) continue;
            // the second rate set at the shorter row counts only
            if (ri != 0 and c.rows > 1328) continue;
            const mark = g.mark();
            defer g.resetTo(mark);
            const x = try randBf16(a, &g, rnd, &.{ 1, ci(c.rows), ci(HIDDEN) }, 0.0, 1.0, false);
            const angles = try angleRows(a, &g, s, c.rows, c.offset);
            const want = try chainIn(&g, s, x, q, k, v, qw, kw, c.rows, c.offset);
            const ep: root.attn_io.Epilogue = .{ .heads = HQ, .kv_heads = HK, .rope_dims = @intCast(RD), .q_norm = qw, .k_norm = kw, .eps = eps, .angles = angles };
            var got = try root.attnIn(s, x, q, k, v, ep, .mul1);
            defer got.deinit();
            const dq = try bitsDiff(a, &g, want.q, got.q);
            const dg = try bitsDiff(a, &g, want.gate, got.gate);
            const dk = try bitsDiff(a, &g, want.k, got.k);
            const dv = try bitsDiff(a, &g, want.v, got.v);
            const ok_in = dq == 0 and dg == 0 and dk == 0 and dv == 0;
            const nq = c.rows * HQ * HD;
            const nk = c.rows * HK * HD;
            std.debug.print("TURBOEXL3_ATTN_IN rates=q{d}/k{d}/v{d} rows={d} offset={d} path={s} q={d}/{d} gate={d}/{d} k={d}/{d} v={d}/{d} {s}\n", .{ rs.q, rs.k, rs.v, c.rows, c.offset, pathOf(c.rows), dq, nq, dg, nq, dk, nk, dv, nk, if (ok_in) "PASS" else "FAIL" });
            if (!ok_in) failed = true;

            // ATTN-OUT: the gate read in place from q_proj (ATTN-IN off) and from ATTN-IN's gate (ATTN-IN on)
            const attn = try randBf16(a, &g, rnd, &.{ 1, ci(HQ), ci(c.rows), ci(HD) }, 0.0, 1.0, false);
            const y_want = try chainOut(&g, s, attn, want.gate, o, c.rows);
            const layouts = [_]struct { name: []const u8, gate: root.attn_io.Gate }{
                .{ .name = "qproj", .gate = .{ .base = want.qp, .row_stride = 2 * HQ * HD, .head_stride = 2 * HD, .off = HD } },
                .{ .name = "attn_in", .gate = .{ .base = got.gate, .row_stride = HQ * HD, .head_stride = HD, .off = 0 } },
            };
            for (layouts) |l| {
                const y_got = try g.adopt(try root.linearGated(s, attn, l.gate, sigtab, o, .mul1));
                const dy = try bitsDiff(a, &g, y_want, y_got);
                std.debug.print("TURBOEXL3_ATTN_OUT rate=o{d} rows={d} gate={s} path={s} y={d}/{d} {s}\n", .{ rs.o, c.rows, l.name, pathOf(c.rows), dy, c.rows * HIDDEN, if (dy == 0) "PASS" else "FAIL" });
                if (dy != 0) failed = true;
            }
        }
    }
    try testing.expect(!failed);
}

test "turboexl3 gpu attn_io: what it does not serve is refused before any launch (GEMV rows, an unserved rotary width, f32 norms) (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const mark = g.mark();
    defer g.resetTo(mark);
    var prng = std.Random.DefaultPrng.init(0xDEC1);
    const rnd = prng.random();
    const q = try synthProj(a, &g, rnd, HIDDEN, 2 * HQ * HD, 112);
    const k = try synthProj(a, &g, rnd, HIDDEN, HK * HD, 112);
    const v = try synthProj(a, &g, rnd, HIDDEN, HK * HD, 112);
    const qw = try randBf16(a, &g, rnd, &.{ci(HD)}, 0.5, 1.0, true);
    const eps = mlx.mlx_array_new_float(EPS);
    defer _ = mlx.mlx_array_free(eps);
    // 16 rows: the GEMV path
    const x16 = try randBf16(a, &g, rnd, &.{ 1, 16, ci(HIDDEN) }, 0.0, 1.0, false);
    const ang16 = try angleRows(a, &g, s, 16, 0);
    const ep16: root.attn_io.Epilogue = .{ .heads = HQ, .kv_heads = HK, .rope_dims = @intCast(RD), .q_norm = qw, .k_norm = qw, .eps = eps, .angles = ang16 };
    try testing.expectError(error.Exl3EngineNotImplemented, root.attnIn(s, x16, q, k, v, ep16, .mul1));
    // 64 rows: a rotary width the staged rotation does not serve, then f32 norms
    const x64 = try randBf16(a, &g, rnd, &.{ 1, 64, ci(HIDDEN) }, 0.0, 1.0, false);
    const ang64 = try angleRows(a, &g, s, 64, 0);
    var ep: root.attn_io.Epilogue = .{ .heads = HQ, .kv_heads = HK, .rope_dims = 48, .q_norm = qw, .k_norm = qw, .eps = eps, .angles = ang64 };
    try testing.expectError(error.Exl3EngineNotImplemented, root.attnIn(s, x64, q, k, v, ep, .mul1));
    ep.rope_dims = @intCast(RD);
    ep.q_norm = try g.astype(qw, .float32);
    try testing.expectError(error.Exl3EngineNotImplemented, root.attnIn(s, x64, q, k, v, ep, .mul1));
    std.debug.print("TURBOEXL3_ATTN_REFUSALS PASS\n", .{});
}

// ---- the four-member prepare (`attn_io.prepare4`) against each member's own prepare, the indexer's `linear` over its copy
// against `linear`, and ATTN-IN over the given copies against ATTN-IN, bit for bit ----

/// The QSA indexer projection at Qwen3.8-Flash-Next's geometry: 4 index heads of 128 and the key head (640 wide).
const IDX_OUT: u32 = 5 * 128;

/// The four-member prepare's cases: the register path (17, 31, 32), dequantize + matmul from 33, a 17.7K-token prompt's
/// prefill chunks (8192 at 0 and 8192, 1328 at 16384, 31 at 17712), and a 2100-row call (the offset moves only ATTN-IN's
/// RoPE rows).
const idxprep_cells = [_]Cell{
    .{ .rows = 17, .offset = 3 },
    .{ .rows = 31, .offset = 17712 },
    .{ .rows = 32, .offset = 77 },
    .{ .rows = 33, .offset = 1000 },
    .{ .rows = 1328, .offset = 16384 },
    .{ .rows = 2100, .offset = 5000 },
    .{ .rows = 8192, .offset = 0 },
    .{ .rows = 8192, .offset = 8192 },
};

/// The members' rates: the full pack's (q|gate, k, v at K7, the indexer at K5), and a set giving each member its own.
const IdxRates = struct { q: u32, k: u32, v: u32, i: u32 };
const idx_rate_sets = [_]IdxRates{ .{ .q = 112, .k = 112, .v = 112, .i = 80 }, .{ .q = 96, .k = 80, .v = 128, .i = 64 } };

test "turboexl3 gpu attn_io: the four-member prepare writes each member's own prepared rows, the indexer's linear over its copy gives linear's qk, and ATTN-IN over the other three gives ATTN-IN's q, gate, k and v, bit for bit (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var failed = false;
    const eps = mlx.mlx_array_new_float(EPS);
    defer _ = mlx.mlx_array_free(eps);
    const top = g.mark();
    defer g.resetTo(top);
    for (idx_rate_sets, 0..) |rs, ri| {
        var prng = std.Random.DefaultPrng.init(0x1D_0001 + ri);
        const rnd = prng.random();
        const set_mark = g.mark();
        defer g.resetTo(set_mark);
        const q = try synthProj(a, &g, rnd, HIDDEN, 2 * HQ * HD, rs.q);
        const k = try synthProj(a, &g, rnd, HIDDEN, HK * HD, rs.k);
        const v = try synthProj(a, &g, rnd, HIDDEN, HK * HD, rs.v);
        const ix = try synthProj(a, &g, rnd, HIDDEN, IDX_OUT, rs.i);
        const qw = try randBf16(a, &g, rnd, &.{ci(HD)}, 0.5, 1.0, true);
        const kw = try randBf16(a, &g, rnd, &.{ci(HD)}, 0.5, 1.0, true);
        // each member's own prepare as `linear` launches it at prefill rows (`Dense.prepare`, natural order)
        const outs = [4]u32{ 2 * HQ * HD, HK * HD, HK * HD, IDX_OUT };
        const rates = [4]u32{ rs.q, rs.k, rs.v, rs.i };
        const suhs = [4]mlx.mlx_array{ q.suh, k.suh, v.suh, ix.suh };
        var ds: [4]dense.Dense = undefined;
        var n_ds: usize = 0;
        defer for (ds[0..n_ds]) |*d| d.deinit();
        for (0..4) |m| {
            ds[m] = try dense.Dense.init(.{ .in = HIDDEN, .out = outs[m], .rate = format.Rate.fromN(rates[m]).?, .dec = .mul1, .out_dtype = .bfloat16 });
            n_ds += 1;
        }
        for (idxprep_cells) |c| {
            if (!rowsWanted(c.rows)) continue;
            // the second rate set at the shorter row counts (as the fused projections' test)
            if (ri != 0 and c.rows > 2100) continue;
            const mark = g.mark();
            defer g.resetTo(mark);
            const x = try randBf16(a, &g, rnd, &.{ 1, ci(c.rows), ci(HIDDEN) }, 0.0, 1.0, false);
            const x2 = try g.reshape(x, &.{ ci(c.rows), ci(HIDDEN) });
            const before = root.attn_io.prep4_calls;
            const xp = try root.attnPrep4(s, x, q, k, v, ix, .mul1);
            for (xp) |arr| _ = try g.adopt(arr);
            var dp: [4]usize = undefined;
            for (0..4) |m| dp[m] = try bitsDiff(a, &g, try g.adopt(try ds[m].prepare(s, x2, suhs[m], c.rows)), xp[m]);
            // the indexer projection: `linear` over x against its launches over the fourth copy
            const qk_want = try g.adopt(try root.linear(s, x, ix.trellis, ix.suh, ix.svh, .mul1));
            const qk_got = try g.reshape(try g.adopt(try root.linearPreparedIdx(s, xp[3], ix, .bfloat16, .mul1)), &.{ 1, ci(c.rows), ci(IDX_OUT) });
            const dqk = try bitsDiff(a, &g, qk_want, qk_got);
            // ATTN-IN over the three given copies against ATTN-IN's own prepare
            const angles = try angleRows(a, &g, s, c.rows, c.offset);
            var ep: root.attn_io.Epilogue = .{ .heads = HQ, .kv_heads = HK, .rope_dims = @intCast(RD), .q_norm = qw, .k_norm = kw, .eps = eps, .angles = angles };
            var want = try root.attnIn(s, x, q, k, v, ep, .mul1);
            defer want.deinit();
            ep.prep = .{ xp[0], xp[1], xp[2] };
            var got = try root.attnIn(s, x, q, k, v, ep, .mul1);
            defer got.deinit();
            const d = [4]usize{ try bitsDiff(a, &g, want.q, got.q), try bitsDiff(a, &g, want.gate, got.gate), try bitsDiff(a, &g, want.k, got.k), try bitsDiff(a, &g, want.v, got.v) };
            const ran = root.attn_io.prep4_calls == before + 1;
            const ok_prep = dp[0] == 0 and dp[1] == 0 and dp[2] == 0 and dp[3] == 0 and ran;
            const ok_qk = dqk == 0;
            const ok_in = d[0] == 0 and d[1] == 0 and d[2] == 0 and d[3] == 0;
            const n_in = c.rows * HIDDEN;
            std.debug.print("TURBOEXL3_IDXPREP_PREP rates=q{d}/k{d}/v{d}/i{d} rows={d} q={d}/{d} k={d}/{d} v={d}/{d} idx={d}/{d} {s}\n", .{ rs.q, rs.k, rs.v, rs.i, c.rows, dp[0], n_in, dp[1], n_in, dp[2], n_in, dp[3], n_in, if (ok_prep) "PASS" else "FAIL" });
            std.debug.print("TURBOEXL3_IDXPREP_QK rate=i{d} rows={d} path={s} qk={d}/{d} {s}\n", .{ rs.i, c.rows, pathOf(c.rows), dqk, c.rows * IDX_OUT, if (ok_qk) "PASS" else "FAIL" });
            std.debug.print("TURBOEXL3_IDXPREP_IN rates=q{d}/k{d}/v{d} rows={d} offset={d} q={d}/{d} gate={d}/{d} k={d}/{d} v={d}/{d} {s}\n", .{ rs.q, rs.k, rs.v, c.rows, c.offset, d[0], c.rows * HQ * HD, d[1], c.rows * HQ * HD, d[2], c.rows * HK * HD, d[3], c.rows * HK * HD, if (ok_in) "PASS" else "FAIL" });
            if (!ok_prep or !ok_qk or !ok_in) failed = true;
        }
    }
    // refused before any launch: the GEMV rows (16), an indexer of another input width
    {
        var prng = std.Random.DefaultPrng.init(0x1D_0003);
        const rnd = prng.random();
        const mark = g.mark();
        defer g.resetTo(mark);
        const q = try synthProj(a, &g, rnd, HIDDEN, 2 * HQ * HD, 112);
        const k = try synthProj(a, &g, rnd, HIDDEN, HK * HD, 112);
        const v = try synthProj(a, &g, rnd, HIDDEN, HK * HD, 112);
        const ix = try synthProj(a, &g, rnd, HIDDEN, IDX_OUT, 80);
        const ix_narrow = try synthProj(a, &g, rnd, 2048, IDX_OUT, 80);
        const x16 = try randBf16(a, &g, rnd, &.{ 1, 16, ci(HIDDEN) }, 0.0, 1.0, false);
        const x64 = try randBf16(a, &g, rnd, &.{ 1, 64, ci(HIDDEN) }, 0.0, 1.0, false);
        const before = root.attn_io.prep4_calls;
        try testing.expectError(error.Exl3EngineNotImplemented, root.attnPrep4(s, x16, q, k, v, ix, .mul1));
        try testing.expectError(error.BadExl3Shape, root.attnPrep4(s, x64, q, k, v, ix_narrow, .mul1));
        try testing.expectEqual(before, root.attn_io.prep4_calls);
        std.debug.print("TURBOEXL3_IDXPREP_REFUSALS PASS\n", .{});
    }
    try testing.expect(!failed);
}
