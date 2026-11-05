//! Ricursive Ripple-65B on turboexl3's linears (TURBOEXL3_GPU=1, else skipped), on synthetic EXL3 weights at its shapes,
//! K3 MUL1 for every layer linear and K6 for the head (the pack's rates):
//! - `root.linear` at every Ripple shape (q and g 5120 -> 10240, k and v 5120 -> 1024, o 10240 -> 5120, gate and up
//!   5120 -> 40960, down 40960 -> 5120; the head 5120 -> 100352) against the host reference (`ref.Linear`:
//!   `format.project`'s values) at rows 1, 4, 16, 32 and 64 (the head 1 and 4): DEC-ONE, the GEMV chain, the register path
//!   and dequantize + matmul. f32 x and y to 5e-3 of the reference rms (the dense chain's bar), bf16 (rows 4 and 64) to
//!   2.5e-2. Lines: TURBOEXL3_RIC_PARITY.
//! - `ricursive.jointLinear` (q | g, gate | up) and `jointLinearBlend` (k | v, v's input the value residual) against each
//!   member's `linear` (v's on the host's blend: astype f32, subtract, multiply, add, astype), word for word, at every
//!   decode row count the joined launch serves. Lines: TURBOEXL3_RIC_JOINT.
//! - `ricursive.valueLinear` at prefill rows against the host's blend then `linear`, word for word.
//! - `ricursive.mlp` (with and without gate | up joined) against `linear(gate)`, `linear(up)`, the host's clamped SwiGLU
//!   in the activation dtype (MLX's minimum, clip, sigmoid and two multiplies) and `linear(down)`, word for word, at rows
//!   1 to 5, 8 and 16 (decode) and 17, 32, 33 and 64 (prefill), the gate and up scaled so the clamp at 20 is reached.
//!   Lines: TURBOEXL3_RIC_MLP.
//! The joined launches' and the fused prepares' values against `format.project` follow: their words are `linear`'s, and
//! `linear` is held to `format.project` by the first test at the same shapes.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const A = mlx.mlx_array;
const format = @import("../../core/format.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const ric = @import("ricursive.zig");

const testing = std.testing;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn bf16Exact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
}

const Case = struct { name: []const u8, in: u32, out: u32, n: u32, head: bool = false };

const cases = [_]Case{
    .{ .name = "q/g", .in = ric.hidden, .out = ric.q_out, .n = ric.n_linear },
    .{ .name = "k/v", .in = ric.hidden, .out = ric.kv_out, .n = ric.n_linear },
    .{ .name = "o", .in = ric.q_out, .out = ric.hidden, .n = ric.n_linear },
    .{ .name = "gate/up", .in = ric.hidden, .out = ric.ffn, .n = ric.n_linear },
    .{ .name = "down", .in = ric.ffn, .out = ric.hidden, .n = ric.n_linear },
    .{ .name = "head", .in = ric.hidden, .out = ric.vocab, .n = ric.n_head, .head = true },
};

const row_counts = [_]u32{ 1, 4, 16, 32, 64 };
const head_rows = [_]u32{ 1, 4 };

fn pickRows(rows: u32, buf: *[16]u32) []const u32 {
    if (rows <= 16) {
        for (buf[0..rows], 0..) |*r, i| r.* = @intCast(i);
        return buf[0..rows];
    }
    buf[0..6].* = .{ 0, 1, rows / 2 - 1, rows / 2, rows - 2, rows - 1 };
    return buf[0..6];
}

fn report(name: []const u8, rows: u32, xd: []const u8, width: usize, got: []const f32, pick: []const u32, want: []const f32, sel: []f32, bar: f64) usize {
    for (pick, 0..) |r, i| @memcpy(sel[i * width ..][0..width], got[r * width ..][0..width]);
    const e = ref.compare(sel[0 .. pick.len * width], want[0 .. pick.len * width]);
    const ok = std.math.isFinite(e.max_abs) and e.max_ratio <= bar;
    std.debug.print("TURBOEXL3_RIC_PARITY {s} rows={d} x={s} compared_rows={d} max_abs={e:.3} rms={e:.3} ratio={e:.3} bar={e:.1} {s}\n", .{ name, rows, xd, pick.len, e.max_abs, e.rms_ref, e.max_ratio, bar, if (ok) "PASS" else "FAIL" });
    return @intFromBool(!ok);
}

fn linearCase(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, c: Case) !usize {
    var prng = std.Random.DefaultPrng.init(0x81C ^ (@as(u64, c.in) << 24) ^ c.out);
    const rnd = prng.random();
    var lin = try ref.synthLinear(a, rnd, 1, c.in, c.out, format.Rate.fromN(c.n).?, .mul1);
    defer lin.deinit(a);
    const mark = g.mark();
    defer g.resetTo(mark);
    const trellis = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(c.in / 16), ci(c.out / 16), ci(c.n) }, .uint16);
    const suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(c.in)}, .float16);
    const svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(c.out)}, .float16);
    const transformed = try a.alloc(f32, c.in);
    defer a.free(transformed);
    const inner = try a.alloc(f32, c.out);
    defer a.free(inner);
    var failed: usize = 0;
    for (if (c.head) head_rows[0..] else row_counts[0..]) |rows| {
        const rmark = g.mark();
        defer g.resetTo(rmark);
        const x = try a.alloc(f32, rows * c.in);
        defer a.free(x);
        for (x) |*v| v.* = bf16Exact(rnd.floatNorm(f32));
        var pick_buf: [16]u32 = undefined;
        const pick = pickRows(rows, &pick_buf);
        const want = try a.alloc(f32, pick.len * c.out);
        defer a.free(want);
        for (pick, 0..) |r, i| lin.row(0, x[r * c.in ..][0..c.in], transformed, inner, want[i * c.out ..][0..c.out]);
        const got = try a.alloc(f32, rows * c.out);
        defer a.free(got);
        const sel = try a.alloc(f32, pick.len * c.out);
        defer a.free(sel);
        const x3 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(c.in) }, .float32);
        const y = try g.adopt(try root.linear(s, x3, trellis, suh, svh, .mul1));
        _ = try g.hostF32(y, got);
        failed += report(c.name, rows, "f32", c.out, got, pick, want, sel, 5e-3);
        if (rows == 4 or rows == 64) {
            const yb = try g.adopt(try root.linear(s, try g.astype(x3, .bfloat16), trellis, suh, svh, .mul1));
            _ = try g.hostF32(try g.astype(yb, .float32), got);
            failed += report(c.name, rows, "bf16", c.out, got, pick, want, sel, 2.5e-2);
        }
    }
    return failed;
}

test "turboexl3 gpu ricursive: root.linear equals the host reference at every Ripple-65B shape, K3 linears and a K6 head (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var failed: usize = 0;
    for (cases) |c| failed += try linearCase(a, &g, s, c);
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── word-for-word tests ──

/// A projection's random operands on the device: codewords, suh in [0.5, 1.5], svh `scale` x [0.5, 1.5] / sqrt(in)
fn randProj(a: std.mem.Allocator, g: *G, rnd: std.Random, in: u32, out: u32, n: u32, scale: f32) !root.Proj {
    const t = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(t);
    rnd.bytes(std.mem.sliceAsBytes(t));
    const su = try a.alloc(u16, in);
    defer a.free(su);
    for (su) |*v| v.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, out);
    defer a.free(sv);
    for (sv) |*v| v.* = format.f32ToF16Bits(scale * (0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    return .{
        .trellis = try g.hostArray(std.mem.sliceAsBytes(t), &.{ ci(in / 16), ci(out / 16), ci(n) }, .uint16),
        .suh = try g.hostArray(std.mem.sliceAsBytes(su), &.{ci(in)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(sv), &.{ci(out)}, .float16),
    };
}

fn randX(a: std.mem.Allocator, g: *G, rnd: std.Random, rows: u32, in: u32, dt: mlx.mlx_dtype) !A {
    const xh = try a.alloc(f32, @as(usize, rows) * in);
    defer a.free(xh);
    for (xh) |*v| v.* = bf16Exact(rnd.floatNorm(f32));
    const x = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(in) }, .float32);
    return if (dt == .float32) x else g.astype(x, dt);
}

/// the words of `x` (any float dtype) and `y` that differ, both read as f32 (exact for bf16 and f16), of `n`
fn wordsDiffer(a: std.mem.Allocator, g: *G, x: A, y: A, n: usize) !usize {
    const p = try a.alloc(f32, n);
    defer a.free(p);
    const q = try a.alloc(f32, n);
    defer a.free(q);
    _ = try g.hostF32(try g.astype(x, .float32), p);
    _ = try g.hostF32(try g.astype(y, .float32), q);
    var d: usize = 0;
    for (p, q) |u, w| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(w)));
    return d;
}

fn op2(g: *G, comptime f: anytype, a: A, b: A) !A {
    var r = mlx.mlx_array_new();
    mlx.check(f(&r, a, b, g.s)) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
}

/// columns [c0, c1) of y [rows, w]
fn cols(g: *G, y: A, rows: u32, c0: u32, c1: u32) !A {
    return g.slice(y, &.{ 0, ci(c0) }, &.{ ci(rows), ci(c1) }, &.{ 1, 1 });
}

/// the host's value residual (mlx-serve's LagunaAttention.valueInput): x0 + blend (x - x0) in f32, back in x's dtype
fn hostBlend(g: *G, x: A, x0: A, blend: A) !A {
    const base = try g.astype(x0, .float32);
    const mixed = try g.add(base, try g.mul(blend, try g.sub(try g.astype(x, .float32), base)));
    return g.astype(mixed, mlx.mlx_array_dtype(x));
}

/// MLX's sigmoid of every 16-bit pattern of `dt` (bf16 or f16): the host's table
fn sigTable(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, dt: mlx.mlx_dtype) !A {
    const pats = try a.alloc(u16, 1 << 16);
    defer a.free(pats);
    for (pats, 0..) |*v, i| v.* = @intCast(i);
    const vals = try g.hostArray(std.mem.sliceAsBytes(pats), &.{1 << 16}, dt);
    var t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&t, vals, s));
    const tab = try g.adopt(t);
    try g.evalAll(&.{tab});
    return tab;
}

test "turboexl3 gpu ricursive joint: q | g and gate | up in one launch, and k | v with v's value residual in its prepare, give each member's linear words (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x81C0);
    const rnd = prng.random();
    const n = ric.n_linear;
    const q = try randProj(a, &g, rnd, ric.hidden, ric.q_out, n, 1);
    const gp = try randProj(a, &g, rnd, ric.hidden, ric.q_out, n, 1);
    const k = try randProj(a, &g, rnd, ric.hidden, ric.kv_out, n, 1);
    const v = try randProj(a, &g, rnd, ric.hidden, ric.kv_out, n, 1);
    const gate = try randProj(a, &g, rnd, ric.hidden, ric.ffn, n, 1);
    const up = try randProj(a, &g, rnd, ric.hidden, ric.ffn, n, 1);
    var qg = try ric.join(s, q, gp);
    defer qg.deinit();
    var kv = try ric.join(s, k, v);
    defer kv.deinit();
    var gu = try ric.join(s, gate, up);
    defer gu.deinit();
    const blend = try g.hostArray(std.mem.sliceAsBytes(&[_]f32{0.3775406688}), &.{1}, .float32);
    var bad: usize = 0;
    var lines: usize = 0;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| for ([_]u32{ 1, 2, 3, 4, 5, 8, 16 }) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        const x = try randX(a, &g, rnd, rows, ric.hidden, dt);
        const x0 = try randX(a, &g, rnd, rows, ric.hidden, dt);
        const Pair = struct { name: []const u8, j: *const ric.Joint, m0: root.Proj, m1: root.Proj, blend: bool };
        for ([_]Pair{
            .{ .name = "q,g", .j = &qg, .m0 = q, .m1 = gp, .blend = false },
            .{ .name = "gate,up", .j = &gu, .m0 = gate, .m1 = up, .blend = false },
            .{ .name = "k,v", .j = &kv, .m0 = k, .m1 = v, .blend = true },
        }) |p| {
            const serves = ric.jointPlan(p.j.*, .mul1, rows) != null;
            if (!serves) {
                const r = if (p.blend) ric.jointLinearBlend(s, x, x0, blend, p.j.*, .mul1) else ric.jointLinear(s, x, p.j.*, .mul1);
                try testing.expectError(error.Exl3EngineNotImplemented, r);
                continue;
            }
            const y = try g.adopt(if (p.blend) try ric.jointLinearBlend(s, x, x0, blend, p.j.*, .mul1) else try ric.jointLinear(s, x, p.j.*, .mul1));
            const w0 = try g.adopt(try root.linear(s, x, p.m0.trellis, p.m0.suh, p.m0.svh, .mul1));
            const x1 = if (p.blend) try hostBlend(&g, x, x0, blend) else x;
            const w1 = try g.adopt(try root.linear(s, x1, p.m1.trellis, p.m1.suh, p.m1.svh, .mul1));
            const o0 = p.j.out0;
            const o1 = p.j.out1;
            const d0 = try wordsDiffer(a, &g, try cols(&g, y, rows, 0, o0), w0, rows * o0);
            const d1 = try wordsDiffer(a, &g, try cols(&g, y, rows, o0, o0 + o1), w1, rows * o1);
            lines += 1;
            bad += @intFromBool(d0 + d1 != 0);
            std.debug.print("TURBOEXL3_RIC_JOINT {s} rows={d} x={s} differing={d}/{d} {d}/{d} {s}\n", .{ p.name, rows, @tagName(dt), d0, rows * o0, d1, rows * o1, if (d0 + d1 == 0) "OK" else "DIFFER" });
        }
    };
    // v alone at prefill rows: the blend in its prepare
    for ([_]u32{ 17, 32, 33, 64 }) |rows| {
        const rm = g.mark();
        defer g.resetTo(rm);
        const x = try randX(a, &g, rnd, rows, ric.hidden, .bfloat16);
        const x0 = try randX(a, &g, rnd, rows, ric.hidden, .bfloat16);
        const y = try g.adopt(try ric.valueLinear(s, x, x0, blend, v, .mul1));
        const w = try g.adopt(try root.linear(s, try hostBlend(&g, x, x0, blend), v.trellis, v.suh, v.svh, .mul1));
        const d = try wordsDiffer(a, &g, y, w, rows * ric.kv_out);
        lines += 1;
        bad += @intFromBool(d != 0);
        std.debug.print("TURBOEXL3_RIC_JOINT v rows={d} x={s} differing={d}/{d} {s}\n", .{ rows, @tagName(mlx.mlx_dtype.bfloat16), d, rows * ric.kv_out, if (d == 0) "OK" else "DIFFER" });
    }
    std.debug.print("TURBOEXL3_RIC_JOINT_DONE lines={d} differing_lines={d} {s}\n", .{ lines, bad, if (bad == 0) "BYTES_OK" else "BYTES_DIFFER" });
    try testing.expectEqual(@as(usize, 0), bad);
}

/// The host's MLP (mlx-serve's GatedMlp with Ripple's clamp, sdk/blocks/mlp/activations.zig `clampedSwiGLU` in the
/// activation dtype), op for op on turboexl3's linears
fn hostMlp(g: *G, s: mlx.mlx_stream, x: A, gate: root.Proj, up: root.Proj, down: root.Proj) !A {
    const dt = mlx.mlx_array_dtype(x);
    const gv = try g.adopt(try root.linear(s, x, gate.trellis, gate.suh, gate.svh, .mul1));
    const uv = try g.adopt(try root.linear(s, x, up.trellis, up.suh, up.svh, .mul1));
    const hi = try g.scalar(ric.clamp, dt);
    const lo = try g.scalar(-ric.clamp, dt);
    const gm = try op2(g, mlx.mlx_minimum, gv, hi);
    const um = try op2(g, mlx.mlx_minimum, try op2(g, mlx.mlx_maximum, uv, lo), hi);
    const act = try g.mul(gm, try g.sigmoid(gm));
    return g.adopt(try root.linear(s, try g.mul(act, um), down.trellis, down.suh, down.svh, .mul1));
}

test "turboexl3 gpu ricursive mlp: gate | up, the clamped SwiGLU in the down's prepare and the down give the host's MLP words, decode and prefill rows (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x81C1);
    const rnd = prng.random();
    const n = ric.n_linear;
    // gate and up scaled so their outputs pass the limit (a few percent beyond 20 in magnitude)
    const gate = try randProj(a, &g, rnd, ric.hidden, ric.ffn, n, 12);
    const up = try randProj(a, &g, rnd, ric.hidden, ric.ffn, n, 12);
    const down = try randProj(a, &g, rnd, ric.ffn, ric.hidden, n, 1);
    var gu = try ric.join(s, gate, up);
    defer gu.deinit();
    var bad: usize = 0;
    var lines: usize = 0;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
        const sigtab = try sigTable(a, &g, s, dt);
        const lim = try g.scalar(ric.clamp, dt);
        for ([_]u32{ 1, 2, 3, 4, 5, 8, 16, 17, 32, 33, 64 }) |rows| {
            if (dt == .float16 and rows != 1 and rows != 33) continue;
            const rm = g.mark();
            defer g.resetTo(rm);
            const x = try randX(a, &g, rnd, rows, ric.hidden, dt);
            const want = try hostMlp(&g, s, x, gate, up, down);
            for ([_]bool{ true, false }) |joined| {
                const y = try g.adopt(try ric.mlp(s, x, gate, up, down, if (joined) gu else null, lim, sigtab, .mul1));
                const d = try wordsDiffer(a, &g, y, want, rows * ric.hidden);
                lines += 1;
                bad += @intFromBool(d != 0);
                std.debug.print("TURBOEXL3_RIC_MLP rows={d} x={s} gate_up={s} differing={d}/{d} {s}\n", .{ rows, @tagName(dt), if (joined and ric.jointPlan(gu, .mul1, rows) != null) "joined" else "two linears", d, rows * ric.hidden, if (d == 0) "OK" else "DIFFER" });
            }
        }
    }
    std.debug.print("TURBOEXL3_RIC_MLP_DONE lines={d} differing_lines={d} {s}\n", .{ lines, bad, if (bad == 0) "BYTES_OK" else "BYTES_DIFFER" });
    try testing.expectEqual(@as(usize, 0), bad);
}
