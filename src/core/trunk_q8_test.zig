//! The int8 options' checks at a host's shapes, the batteries the model files run (models/qwen38/qwen38_int8_test.zig,
//! models/deepseek_v41/deepseek_int8_test.zig) on synthetic weights (every 16-bit pattern a codeword, suh in [0.5, 1.5],
//! svh in [0.5, 1.5] / sqrt(in)) and bf16 inputs:
//! - `linearBattery`: one trunk linear on trunk int8 (`Config.pf_trunk_int8`, trunk_q8.zig) through root's entry points:
//!   OFF, AGREE and SPLIT as GLM-5.3-Flash's trunk int8 test holds them (models/glm53f/fused_test.zig), then INNER, the
//!   inner products a host splits off `linear` (`Config.pf_trunk_int8_inner`) against `linear`'s int8 words, and INNER_OFF;
//! - `routedAgree`: the routed prefill's int8 GEMMs (`Config.pf_int8`) against the exact chain at both weight centers
//!   (`Config.pf_int8_c2`), and the centered one's error below 510's.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("test_support.zig");
const G = support.Graph;
const format = @import("format.zig");
const ref = @import("ref.zig");
const root = @import("../root.zig");

const A = mlx.mlx_array;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// A dense EXL3 projection in -> out at rate n: random words (every pattern a codeword), suh in [0.5, 1.5], svh in
/// [0.5, 1.5] / sqrt(in) (outputs O(1) for O(1) inputs)
pub fn synthProj(g: *G, a: std.mem.Allocator, rnd: std.Random, in: u32, out: u32, n: u32) !root.Proj {
    const tw = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(tw);
    rnd.bytes(std.mem.sliceAsBytes(tw));
    const su = try a.alloc(u16, in);
    defer a.free(su);
    for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, out);
    defer a.free(sv);
    for (sv) |*v| v.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    const p: root.Proj = .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(tw), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(out)}, .float16),
    };
    try g.evalAll(&.{ p.trellis, p.suh, p.svh });
    return p;
}

/// bf16-exact standard normals, the pool `poolX` cycles through
pub fn normalPool(rnd: std.Random, pool: []f32) void {
    for (pool) |*v| v.* = @bitCast(@as(u32, support.bf16Bits(rnd.floatNorm(f32))) << 16);
}

/// x bf16 [rows, in] from `pool`'s values in turn
pub fn poolX(g: *G, a: std.mem.Allocator, pool: []const f32, rows: u32, in: u32) !A {
    const xh = try a.alloc(f32, @as(usize, rows) * in);
    defer a.free(xh);
    for (xh, 0..) |*v, i| v.* = pool[i % pool.len];
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(in) }, .float32), .bfloat16);
    try g.evalAll(&.{x});
    return x;
}

/// `x`'s values as f32 in `out`, and their 32-bit words (bf16 and f16 widen to f32 one to one)
pub fn bitsOf(g: *G, x: A, out: []f32) ![]const u32 {
    const f = if (mlx.mlx_array_dtype(x) == .float32) x else try g.astype(x, .float32);
    _ = try g.hostF32(f, out);
    const p: [*]const u32 = @ptrCast(out.ptr);
    return p[0..out.len];
}

/// The words of `got` that differ from `want`, and the non-finite values of `vals` (got's values)
pub fn countDiff(got: []const u32, want: []const u32, vals: []const f32) [2]usize {
    var d: [2]usize = .{ 0, 0 };
    for (got, want, vals) |u, v, f| {
        d[0] += @intFromBool(u != v);
        d[1] += @intFromBool(!std.math.isFinite(f));
    }
    return d;
}

/// `got` against `want`: the cosine, the error's rms over want's rms, and the non-finite values (left out of both)
pub const Agree = struct {
    cos: f64,
    rms: f64,
    nonfinite: usize,

    /// GLM-5.3-Flash's int8 bar: cosine >= 0.99, error rms <= 0.1 of y's, every value finite
    pub fn ok(x: Agree) bool {
        return x.nonfinite == 0 and x.cos >= 0.99 and x.rms <= 0.1;
    }
};

pub fn agreeOf(got: []const f32, want: []const f32) Agree {
    var dot: f64 = 0;
    var gg: f64 = 0;
    var ww: f64 = 0;
    var ee: f64 = 0;
    var nonfinite: usize = 0;
    for (got, want) |gv, wv| {
        if (!std.math.isFinite(gv) or !std.math.isFinite(wv)) {
            nonfinite += 1;
            continue;
        }
        dot += @as(f64, gv) * wv;
        gg += @as(f64, gv) * gv;
        ww += @as(f64, wv) * wv;
        ee += (@as(f64, gv) - wv) * (@as(f64, gv) - wv);
    }
    return .{ .cos = dot / @sqrt(gg * ww), .rms = @sqrt(ee / ww), .nonfinite = nonfinite };
}

/// The configuration and trunk int8 shapes in force, restored by `restore` (the engines released first)
pub const Saved = struct { cfg: root.config.Config, shapes: ?[]const root.trunk_q8.Shape };

pub fn save() Saved {
    return .{ .cfg = root.config.get().*, .shapes = root.trunk_q8.shapes.get().* };
}

pub fn restore(sv: Saved) void {
    root.releaseEngines();
    root.configure(sv.cfg) catch unreachable;
    root.trunk_q8.shapes.get().* = sv.shapes;
}

/// `base` with trunk int8 `on` (its inner products with `inner`) at center `c2`, and the host's shapes `list`
pub fn setTrunk(base: root.config.Config, on: bool, inner: bool, c2: u16, list: ?[]const root.trunk_q8.Shape) !void {
    var c = base;
    c.pf_trunk_int8 = on;
    c.pf_trunk_int8_inner = on and inner;
    c.pf_int8_c2 = c2;
    try root.configure(c);
    root.trunk_q8.shapes.get().* = list;
}

/// trunk int8's GEMM launches so far (a check's "int8" is whether they moved)
pub fn int8Calls() u64 {
    return root.trunk_q8.calls.get().*.load(.monotonic);
}

/// A battery's host buffers: the chain's values, root.linear's int8 values, and a scratch, each of the most words a call
/// writes
pub const Bufs = struct {
    want: []f32,
    ref8: []f32,
    got: []f32,

    pub fn init(a: std.mem.Allocator, words: usize) !Bufs {
        const w = try a.alloc(f32, words);
        errdefer a.free(w);
        const r = try a.alloc(f32, words);
        errdefer a.free(r);
        return .{ .want = w, .ref8 = r, .got = try a.alloc(f32, words) };
    }

    pub fn deinit(b: Bufs, a: std.mem.Allocator) void {
        a.free(b.want);
        a.free(b.ref8);
        a.free(b.got);
    }
};

fn linearY(g: *G, s: mlx.mlx_stream, x: A, p: root.Proj) !A {
    const y = try g.adopt(try root.linear(s, x, p.trellis, p.suh, p.svh, .mul1));
    try g.evalAll(&.{y});
    return y;
}

fn verdict(ok: bool) []const u8 {
    return if (ok) "PASS" else "FAIL";
}

/// One trunk linear (k -> n, `p`) at each of `rows_list` rows (x bf16 from `pool`; at least 66 rows, so a half of them
/// takes the matmul path), every int8 arm at center `c2`; each check printed as `<label>_<CHECK> ... PASS|FAIL`:
/// - OFF: the switch off with the shape named gives the chain's words, no int8 launch;
/// - AGREE: root.linear on int8 against the chain (`Agree.ok`);
/// - SPLIT: the rows in two calls give the one call's int8 words;
/// - INNER: with the inner switch, the inner products a host splits off give root.linear's int8 words: linearInnerSplit +
///   linearFinish (split), linearPrep + linearPrepared (prepared), the two halves as row pieces from x (pieces) and from
///   prepared rows (pieces_prepared) against linearPiecesBegin's int8 wq;
/// - INNER_OFF: the inner switch off (trunk int8 on): linearInnerSplit + linearFinish give the chain's words, no int8 launch.
/// Returns whether every check passed. The configuration and shapes are restored after.
pub fn linearBattery(g: *G, a: std.mem.Allocator, s: mlx.mlx_stream, label: []const u8, name: []const u8, p: root.Proj, k: u32, n: u32, rows_list: []const u32, pool: []const f32, c2: u16, bufs: Bufs) !bool {
    const sv = save();
    defer restore(sv);
    const one = [1]root.trunk_q8.Shape{.{ .k = k, .n = n }};
    var ok_all = true;
    for (rows_list) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        const x = try poolX(g, a, pool, rows, k);
        const xd = mlx.mlx_array_dtype(x);
        const words = @as(usize, rows) * n;
        const r1: u32 = if (rows >= 4096) 2048 else rows / 2;
        const xa = try g.slice(x, &.{ 0, 0 }, &.{ ci(r1), ci(k) }, &.{ 1, 1 });
        const xb = try g.slice(x, &.{ ci(r1), 0 }, &.{ ci(rows), ci(k) }, &.{ 1, 1 });

        // the chain: the switch off, no shape named
        try setTrunk(sv.cfg, false, false, c2, null);
        const wb = try bitsOf(g, try linearY(g, s, x, p), bufs.want[0..words]);

        // OFF: the switch off with the shape named
        try setTrunk(sv.cfg, false, false, c2, &one);
        var before = int8Calls();
        const yoff = try linearY(g, s, x, p);
        const off_int8 = int8Calls() != before;
        const od = countDiff(try bitsOf(g, yoff, bufs.got[0..words]), wb, bufs.got[0..words]);
        const off_ok = !off_int8 and od[0] == 0 and od[1] == 0;
        std.debug.print("{s}_OFF name={s} k={d} n={d} rows={d} int8={} differing={d}/{d} nonfinite={d} {s}\n", .{ label, name, k, n, rows, off_int8, od[0], words, od[1], verdict(off_ok) });
        ok_all = ok_all and off_ok;

        // AGREE: trunk int8 on (its inner products off)
        try setTrunk(sv.cfg, true, false, c2, &one);
        before = int8Calls();
        const y8 = try linearY(g, s, x, p);
        const on_int8 = int8Calls() != before;
        const rb = try bitsOf(g, y8, bufs.ref8[0..words]);
        const ag = agreeOf(bufs.ref8[0..words], bufs.want[0..words]);
        const agree_ok = on_int8 and ag.ok();
        std.debug.print("{s}_AGREE name={s} k={d} n={d} rows={d} c2={d} int8={} cos={d:.6} rms_ratio={d:.5} nonfinite={d} {s}\n", .{ label, name, k, n, rows, c2, on_int8, ag.cos, ag.rms, ag.nonfinite, verdict(agree_ok) });
        ok_all = ok_all and agree_ok;

        // SPLIT: the rows in two calls
        const yab = try g.concat(&.{ try linearY(g, s, xa, p), try linearY(g, s, xb, p) }, 0);
        const sd = countDiff(try bitsOf(g, yab, bufs.got[0..words]), rb, bufs.got[0..words]);
        const split_ok = sd[0] == 0 and sd[1] == 0;
        std.debug.print("{s}_SPLIT name={s} k={d} n={d} rows={d} split={d}+{d} differing={d}/{d} nonfinite={d} {s}\n", .{ label, name, k, n, rows, r1, rows - r1, sd[0], words, sd[1], verdict(split_ok) });
        ok_all = ok_all and split_ok;

        // INNER: the inner products a host splits off, each against root.linear's int8 words
        try setTrunk(sv.cfg, true, true, c2, &one);
        const ys = blk: {
            before = int8Calls();
            const z = try g.adopt(try root.linearInnerSplit(s, s, x, p, .mul1));
            const y = try g.adopt(try root.linearFinish(s, z, p, .mul1, xd));
            try g.evalAll(&.{y});
            break :blk .{ y, int8Calls() != before };
        };
        const yp = blk: {
            const xp = try g.adopt(try root.linearPrep(s, x, p, .mul1));
            before = int8Calls();
            const y = try g.adopt(try root.linearPrepared(s, xp, p, .mul1, xd));
            try g.evalAll(&.{y});
            break :blk .{ y, int8Calls() != before };
        };
        const W = try g.adopt(try root.linearPiecesBegin(s, p, &[_]usize{ r1, rows - r1 }, xd, .mul1));
        const w_int8 = mlx.mlx_array_dtype(W) == .int8;
        const ypc = blk: {
            before = int8Calls();
            const za = try g.adopt(try root.linearPieceInner(s, xa, W, p, .mul1));
            const zb = try g.adopt(try root.linearPieceInner(s, xb, W, p, .mul1));
            const y = try g.adopt(try root.linearPiecesFinish(s, &.{ za, zb }, p, .mul1, xd));
            try g.evalAll(&.{y});
            break :blk .{ y, int8Calls() != before };
        };
        const ypp = blk: {
            const xpa = try g.adopt(try root.linearPrep(s, xa, p, .mul1));
            const xpb = try g.adopt(try root.linearPrep(s, xb, p, .mul1));
            before = int8Calls();
            const za = try g.adopt(try root.linearPiecePrepared(s, xpa, W, p, .mul1, xd));
            const zb = try g.adopt(try root.linearPiecePrepared(s, xpb, W, p, .mul1, xd));
            const y = try g.adopt(try root.linearPiecesFinish(s, &.{ za, zb }, p, .mul1, xd));
            try g.evalAll(&.{y});
            break :blk .{ y, int8Calls() != before };
        };
        const paths = [_]struct { name: []const u8, y: A, int8: bool }{
            .{ .name = "split", .y = ys[0], .int8 = ys[1] },
            .{ .name = "prepared", .y = yp[0], .int8 = yp[1] },
            .{ .name = "pieces", .y = ypc[0], .int8 = ypc[1] and w_int8 },
            .{ .name = "pieces_prepared", .y = ypp[0], .int8 = ypp[1] and w_int8 },
        };
        for (paths) |ph| {
            const d = countDiff(try bitsOf(g, ph.y, bufs.got[0..words]), rb, bufs.got[0..words]);
            const inner_ok = ph.int8 and d[0] == 0 and d[1] == 0;
            std.debug.print("{s}_INNER name={s} k={d} n={d} rows={d} path={s} int8={} differing={d}/{d} nonfinite={d} {s}\n", .{ label, name, k, n, rows, ph.name, ph.int8, d[0], words, d[1], verdict(inner_ok) });
            ok_all = ok_all and inner_ok;
        }

        // INNER_OFF: trunk int8 on, its inner products off: the split-off chain
        try setTrunk(sv.cfg, true, false, c2, &one);
        before = int8Calls();
        const zo = try g.adopt(try root.linearInnerSplit(s, s, x, p, .mul1));
        const yo = try g.adopt(try root.linearFinish(s, zo, p, .mul1, xd));
        try g.evalAll(&.{yo});
        const io_int8 = int8Calls() != before;
        const iod = countDiff(try bitsOf(g, yo, bufs.got[0..words]), wb, bufs.got[0..words]);
        const io_ok = !io_int8 and iod[0] == 0 and iod[1] == 0;
        std.debug.print("{s}_INNER_OFF name={s} k={d} n={d} rows={d} int8={} differing={d}/{d} nonfinite={d} {s}\n", .{ label, name, k, n, rows, io_int8, iod[0], words, iod[1], verdict(io_ok) });
        ok_all = ok_all and io_ok;
    }
    return ok_all;
}

/// A routed prefill call at a host's geometry: a synthetic bank (ref.synthBank: `experts` experts, hidden -> inter at
/// rate n), x bf16 [1, rows, hidden] of standard normals, each row's `topk` distinct experts drawn uniformly with positive
/// scores summing to one
pub const RoutedCase = struct {
    bank: root.Bank,
    x3: A,
    inds3: A,
    sc3: A,
    rows: u32,
    hidden: u32,
};

pub fn routedCase(g: *G, a: std.mem.Allocator, rnd: std.Random, experts: u32, hidden: u32, inter: u32, n: u32, rows: u32, topk: u32) !RoutedCase {
    const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
    var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
    defer bank.deinit(a);
    const E = ci(experts);
    const b: root.Bank = .{
        .gate = .{
            .trellis = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ E, ci(hidden / 16), ci(inter / 16), ci(n) }, .uint16),
            .suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ E, ci(hidden) }, .float16),
            .svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ E, ci(inter) }, .float16),
        },
        .up = .{
            .trellis = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ E, ci(hidden / 16), ci(inter / 16), ci(n) }, .uint16),
            .suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ E, ci(hidden) }, .float16),
            .svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ E, ci(inter) }, .float16),
        },
        .down = .{
            .trellis = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ E, ci(inter / 16), ci(hidden / 16), ci(n) }, .uint16),
            .suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ E, ci(inter) }, .float16),
            .svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ E, ci(hidden) }, .float16),
        },
    };
    const S = @as(usize, rows) * topk;
    const slots = try a.alloc(u32, S);
    defer a.free(slots);
    const scores = try a.alloc(f32, S);
    defer a.free(scores);
    for (0..rows) |r| {
        const row = slots[r * topk ..][0..topk];
        for (row, 0..) |*sl, j| {
            while (true) {
                const e = rnd.uintLessThan(u32, experts);
                if (std.mem.indexOfScalar(u32, row[0..j], e) == null) {
                    sl.* = e;
                    break;
                }
            }
        }
        var sum: f32 = 0;
        for (scores[r * topk ..][0..topk]) |*w| {
            w.* = 0.05 + rnd.float(f32);
            sum += w.*;
        }
        for (scores[r * topk ..][0..topk]) |*w| w.* /= sum;
    }
    const xh = try a.alloc(f32, @as(usize, rows) * hidden);
    defer a.free(xh);
    for (xh) |*v| v.* = rnd.floatNorm(f32);
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ 1, ci(rows), ci(hidden) }, .float32), .bfloat16);
    const c: RoutedCase = .{
        .bank = b,
        .x3 = x,
        .inds3 = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ 1, ci(rows), ci(topk) }, .uint32),
        .sc3 = try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, ci(rows), ci(topk) }, .float32),
        .rows = rows,
        .hidden = hidden,
    };
    try g.evalAll(&.{ b.gate.trellis, b.gate.suh, b.gate.svh, b.up.trellis, b.up.suh, b.up.svh, b.down.trellis, b.down.suh, b.down.svh, c.x3, c.inds3, c.sc3 });
    return c;
}

/// The routed call (root.moe, or root.moeClamped at `limit`) at the configuration in force, evaluated
pub fn routedRun(g: *G, s: mlx.mlx_stream, c: RoutedCase, limit: c_int) !A {
    const y = if (limit == 0)
        try root.moe(s, c.x3, c.bank, c.inds3, c.sc3, .mul1, false)
    else
        try root.moeClamped(s, c.x3, c.bank, c.inds3, c.sc3, .mul1, limit);
    const ya = try g.adopt(y);
    try g.evalAll(&.{ya});
    return ya;
}

/// `base` with the routed int8 scheme `scheme` (0 the exact chain) at center `c2`, every engine released first (an
/// engine keeps the options it was made with)
pub fn setRouted(base: root.config.Config, scheme: u8, c2: u16) !void {
    root.releaseEngines();
    var c = base;
    c.pf_int8 = scheme;
    c.pf_int8_c2 = c2;
    try root.configure(c);
}

/// The routed int8's checks at one call (`c`, clamp `limit`): the exact chain, then each scheme of `schemes` at both
/// centers, each printed as `<label>_AGREE ... PASS|FAIL` (int8 ran: y is not the chain's words; `Agree.ok`), then
/// `<label>_CENTER` (the two centers' y differ, and 509.5's error rms is below 510's: the rebuild's bias of half a unit of
/// t, removed). Returns whether every check passed. The configuration is restored after.
pub fn routedAgree(g: *G, a: std.mem.Allocator, s: mlx.mlx_stream, label: []const u8, c: RoutedCase, limit: c_int, n: u32, schemes: []const u8) !bool {
    const sv = save();
    defer restore(sv);
    const words = @as(usize, c.rows) * c.hidden;
    const want = try a.alloc(f32, words);
    defer a.free(want);
    const g1020 = try a.alloc(f32, words);
    defer a.free(g1020);
    const g1019 = try a.alloc(f32, words);
    defer a.free(g1019);
    const mark = g.mark();
    defer g.resetTo(mark);
    try setRouted(sv.cfg, 0, 1020);
    const wb = try bitsOf(g, try routedRun(g, s, c, limit), want);
    var ok_all = true;
    for (schemes) |v| {
        var rms: [2]f64 = undefined;
        var bits: [2][]const u32 = undefined;
        for ([_]u16{ 1020, 1019 }, [_][]f32{ g1020, g1019 }, 0..) |c2, buf, i| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            try setRouted(sv.cfg, v, c2);
            bits[i] = try bitsOf(g, try routedRun(g, s, c, limit), buf);
            const d = countDiff(bits[i], wb, buf);
            const ag = agreeOf(buf, want);
            rms[i] = ag.rms;
            const ok = d[0] != 0 and ag.ok();
            std.debug.print("{s}_AGREE n={d} rows={d} int8={d} c2={d} differing={d}/{d} cos={d:.6} rms_ratio={d:.5} nonfinite={d} {s}\n", .{ label, n, c.rows, v, c2, d[0], words, ag.cos, ag.rms, ag.nonfinite, verdict(ok) });
            ok_all = ok_all and ok;
        }
        const cd = countDiff(bits[1], bits[0], g1019);
        const center_ok = cd[0] != 0 and rms[1] < rms[0];
        std.debug.print("{s}_CENTER n={d} rows={d} int8={d} differing={d}/{d} rms_ratio_510={d:.5} rms_ratio_509.5={d:.5} {s}\n", .{ label, n, c.rows, v, cd[0], words, rms[0], rms[1], verdict(center_ok) });
        ok_all = ok_all and center_ok;
    }
    return ok_all;
}
