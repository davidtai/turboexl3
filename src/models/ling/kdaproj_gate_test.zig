//! L6, `kdaProjPromptGate` (the outside plugin turboexl3_ling's KDA_GATE_EPI): P4's launch with member 1's (f's) output
//! the KDA layer's bounded gate, made in its store loop, held word for word to the served chain: `kdaProjPrompt`, then
//! the plugin's one-launch bounded gate (`ling_kda_gate`, the ops of mlx-serve's `kdaGateChain`) over its y_1.
//! - GPU (TURBOEXL3_GPU=1, else skipped), at Ling-3.1-flash's KDA shapes (q|k|v 4096 -> 24576, f and g 4096 -> 8192 at
//!   K 6, the loader's I16 trellis words), x the layer's normed input [1, S, 4096] bf16, the gate's A_log and dt_bias
//!   f32 [8192] and the bound -5 (the config's kda_gate_lower_bound): at S 33 (the first rows past the register path),
//!   100, 1000, 1965 and 1996 (rows that end inside a tile), 512, 4096 and 16384 (the plugin's chunk and layer-major
//!   calls), for G0 and G1, y_0 and y_2 against `kdaProjPrompt`'s and g against the served gate (its test-only copy,
//!   kernels/test/copy/ling_kda_gate.metal) over `kdaProjPrompt`'s y_1, every word, both finite. Lines:
//!   TURBOEXL3_LING_KDA_GATE.
//! - GPU, over given W_hats (`kdaProjPromptWGate`): at rows 33, 1965 and 4096, both forms, every output against
//!   `kdaProjPromptGate`'s word for word. Lines: TURBOEXL3_LING_KDA_GATE_W.
//! - GPU, the declines at 16, 17 and 32 rows and the gate's refusals (A_log bf16, another length, the bound 0), before
//!   any launch; then the planted faults, each in the gate's store loop's place for one call, its g against the served
//!   gate's (each must differ): member 1's word not rounded as P4 stores it, dt_bias of the next channel, the sigmoid's
//!   exp on the fast path, the gate on member 2's tiles, and G1's exp(A_log) of the next channel. Lines:
//!   TURBOEXL3_LING_KDA_GATE_RED, then TURBOEXL3_LING_KDA_GATE_DONE.
//! - Timing (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; asserts nothing, guidance only): the served chain
//!   (`kdaProjPrompt`, then the served gate) against G0 and G1, and the served gate's launch alone, at rows 512, 1996,
//!   4096, 8192 and 16384; the W pair (`kdaProjPromptW` and the gate against `kdaProjPromptWGate`) at 4096; interleaved
//!   round by round, min and median ms over 8 rounds. Lines: TURBOEXL3_LING_TIME kdaproj_gate.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ks = sdk.kernel_launch;
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
/// Ling-3.1-flash's kda_lower_bound
const bound: f32 = -5.0;

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

/// f32 [n] host values uniform in [lo, hi)
fn uniformF32(a: std.mem.Allocator, g: *G, rnd: std.Random, n: u32, lo: f32, hi: f32) !T {
    const v = try a.alloc(f32, n);
    defer a.free(v);
    for (v) |*x| x.* = lo + (hi - lo) * rnd.float(f32);
    return g.hostArray(std.mem.sliceAsBytes(v), &.{ci(n)}, .float32);
}

// ── the served gate: its test-only copy, launched as the plugin launches it ──

const gate_in = [_][*:0]const u8{ "A_log", "a", "dt_bias", "lb", "n_total" };
const gate_out = [_][*:0]const u8{"g"};
var served_gate_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The plugin's one-launch bounded gate (overlay/mlx_serve/sdk/blocks/kda/ling.zig `gate`) over `a` [..., W] in the
/// activation dtype: g f32 of a's shape, one thread an element, threadgroups of 256, W a template int. Owned.
fn servedGate(s: mlx.mlx_stream, A_log: T, a: T, dt_bias: T, lb_v: f32) !T {
    const k = served_gate_kernel orelse blk: {
        const nk = ks.newKernel("turboexl3_test_ling_kda_gate", &gate_in, &gate_out, @embedFile("kernels/test/copy/ling_kda_gate.metal"), "", true) orelse return error.KernelCreateFailed;
        served_gate_kernel = nk;
        break :blk nk;
    };
    const ash = mlx.getShape(a);
    const w = ash[ash.len - 1];
    const n_total: c_int = @intCast(mlx.mlx_array_size(a));
    const tg: c_int = 256;
    const cfg = try ks.buildConfig(.{
        .grid = .{ @intCast(@divTrunc(n_total + tg - 1, tg) * tg), 1, 1 },
        .threadgroup = .{ @intCast(tg), 1, 1 },
        .out_shapes = &.{ash},
        .out_dtypes = &.{.float32},
        .template = &.{.{ .name = "W", .value = .{ .int = @intCast(w) } }},
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const lb = mlx.mlx_array_new_float(lb_v);
    defer _ = mlx.mlx_array_free(lb);
    const nt = mlx.mlx_array_new_int(n_total);
    defer _ = mlx.mlx_array_free(nt);
    var o: [1]T = undefined;
    try ks.applyConfig(k, cfg, &.{ A_log, a, dt_bias, lb, nt }, &o, s);
    return o[0];
}

/// The layer's gate operands for `form`: G0 A_log itself, G1 its exp from `gateEa`
fn gateOf(form: kdaproj_pf.GateForm, A_log: T, ea: T, dt_bias: T) kdaproj_pf.Gate {
    return .{ .form = form, .a = if (form == .g0) A_log else ea, .dt_bias = dt_bias, .bound = bound };
}

fn sentinel(comptime s: []const u8) [:0]const u8 {
    comptime {
        var buf: [s.len:0]u8 = undefined;
        @memcpy(buf[0..s.len], s);
        const final = buf;
        return &final;
    }
}

/// A gate launch's text with `piece` in its store loop's place (a planted fault's)
fn faultText(comptime piece: []const u8) [:0]const u8 {
    return sentinel(kdaproj_pf.kpf_1 ++ pgemm.fused_2 ++ kdaproj_pf.kpf_3_1 ++ kdaproj_pf.kpf_3_2g ++ pgemm.fused_4 ++ kdaproj_pf.kpf_5 ++ pgemm.fused_6_1 ++ piece ++ kdaproj_pf.kpf_undef);
}

const Fault = struct { name: []const u8, form: kdaproj_pf.GateForm, rows: u32, k: mlx.mlx_fast_metal_kernel };

test "turboexl3 gpu ling kdaproj gate: kdaProjPromptGate's y_0 and y_2 are kdaProjPrompt's and its g the served one-launch gate of kdaProjPrompt's y_1, word for word, G0 and G1, at prompt rows 33 to 16384; kdaProjPromptWGate's the same over given W_hats; its declines and refusals; planted faults caught (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s) or !config.get().nax) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x16A7E5);
    const rnd = prng.random();
    var ms: [3]Served = undefined;
    var made: usize = 0;
    defer for (ms[0..made]) |*m| m.lin.deinit(a);
    for (&ms, outs) |*m, o| {
        m.* = try servedProj(a, &g, s, rnd, o);
        made += 1;
    }
    const p = [3]root.Proj{ ms[0].p, ms[1].p, ms[2].p };
    const A_log = try uniformF32(a, &g, rnd, outs[1], -3.0, 1.0);
    const dt_bias = try uniformF32(a, &g, rnd, outs[1], -4.0, 2.0);
    const ea = try g.adopt(try kdaproj_pf.gateEa(s, A_log));
    try g.evalAll(&.{ A_log, dt_bias, ea });
    const forms = [_]kdaproj_pf.GateForm{ .g0, .g1 };
    var lines: usize = 0;
    var failed: usize = 0;
    for ([_]u32{ 33, 100, 512, 1000, 1965, 1996, 4096, 16384 }) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const x = try ds_test.randBf16(a, &g, rnd, &.{ 1, ci(rows), ci(in) });
        const want = try kdaproj_pf.kdaProjPrompt(s, x, p, .mul1);
        var wys: [3]T = undefined;
        for (&wys, want) |*d, y| d.* = try g.adopt(y);
        const g_want = try g.adopt(try servedGate(s, A_log, wys[1], dt_bias, bound));
        for (forms) |form| {
            const got = try kdaproj_pf.kdaProjPromptGate(s, x, p, gateOf(form, A_log, ea, dt_bias), .mul1);
            var gys: [3]T = undefined;
            for (&gys, got) |*d, y| d.* = try g.adopt(y);
            const refs = [3]T{ wys[0], g_want, wys[2] };
            for (gys, refs, outs, 0..) |gy, wy, o, i| {
                const c = try compare(a, &g, gy, wy);
                const dt_ok = i != 1 or mlx.mlx_array_dtype(gy) == .float32;
                const ok = c.differ == 0 and c.nonfinite == 0 and sameShape(gy, wy) and dt_ok;
                lines += 1;
                failed += @intFromBool(!ok);
                std.debug.print("TURBOEXL3_LING_KDA_GATE rows={d} form={s} member={d} out={d} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, @tagName(form), i, o, c.differ, mlx.mlx_array_size(wy), c.nonfinite, if (ok) "PASS" else "FAIL" });
            }
        }
    }
    // over given W_hats: each member's W_hat by its `linear`'s decode launch, evaluated once
    {
        var w_hat: [3]T = undefined;
        for (&w_hat, p, outs) |*wm, m, o| {
            const dn = try root.denseCached(.{ .in = in, .out = o, .rate = format.Rate.fromN(96).?, .dec = .mul1, .out_dtype = .bfloat16 });
            wm.* = try g.adopt(try dn.dequantize(s, m.trellis));
        }
        try g.evalAll(&w_hat);
        for ([_]u32{ 33, 1965, 4096 }) |rows| {
            const mark = g.mark();
            defer g.resetTo(mark);
            const x = try ds_test.randBf16(a, &g, rnd, &.{ 1, ci(rows), ci(in) });
            for (forms) |form| {
                const gate = gateOf(form, A_log, ea, dt_bias);
                const want = try kdaproj_pf.kdaProjPromptGate(s, x, p, gate, .mul1);
                var wys: [3]T = undefined;
                for (&wys, want) |*d, y| d.* = try g.adopt(y);
                const got = try kdaproj_pf.kdaProjPromptWGate(s, x, p, w_hat, gate, .mul1);
                var gys: [3]T = undefined;
                for (&gys, got) |*d, y| d.* = try g.adopt(y);
                for (gys, wys, outs, 0..) |gy, wy, o, i| {
                    const c = try compare(a, &g, gy, wy);
                    const ok = c.differ == 0 and c.nonfinite == 0 and sameShape(gy, wy);
                    lines += 1;
                    failed += @intFromBool(!ok);
                    std.debug.print("TURBOEXL3_LING_KDA_GATE_W rows={d} form={s} member={d} out={d} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, @tagName(form), i, o, c.differ, mlx.mlx_array_size(wy), c.nonfinite, if (ok) "PASS" else "FAIL" });
                }
            }
        }
        // the declines at the GEMV's and the register path's rows, both calls, before any launch
        {
            const mark = g.mark();
            defer g.resetTo(mark);
            for ([_]u32{ 16, 17, 32 }) |rows| {
                const x = try ds_test.randBf16(a, &g, rnd, &.{ 1, ci(rows), ci(in) });
                for (forms) |form| {
                    const gate = gateOf(form, A_log, ea, dt_bias);
                    try testing.expectError(error.Exl3EngineNotImplemented, kdaproj_pf.kdaProjPromptGate(s, x, p, gate, .mul1));
                    try testing.expectError(error.Exl3EngineNotImplemented, kdaproj_pf.kdaProjPromptWGate(s, x, p, w_hat, gate, .mul1));
                }
            }
        }
        std.debug.print("TURBOEXL3_LING_KDA_GATE rows=16,17,32 declined (both calls, both forms) PASS\n", .{});
    }
    // the gate's refusals, before any launch: A_log bf16, a length other than member 1's, the bound 0
    {
        const mark = g.mark();
        defer g.resetTo(mark);
        const x = try ds_test.randBf16(a, &g, rnd, &.{ 1, 64, ci(in) });
        const bf = try g.astype(A_log, .bfloat16);
        const short = try uniformF32(a, &g, rnd, outs[1] / 2, -3.0, 1.0);
        const refused = [_]kdaproj_pf.Gate{
            .{ .form = .g0, .a = bf, .dt_bias = dt_bias, .bound = bound },
            .{ .form = .g0, .a = short, .dt_bias = dt_bias, .bound = bound },
            .{ .form = .g1, .a = ea, .dt_bias = short, .bound = bound },
            .{ .form = .g0, .a = A_log, .dt_bias = dt_bias, .bound = 0.0 },
        };
        for (refused) |gate| try testing.expectError(error.Exl3EngineNotImplemented, kdaproj_pf.kdaProjPromptGate(s, x, p, gate, .mul1));
        std.debug.print("TURBOEXL3_LING_KDA_GATE refused: A_log bf16, a length of 4096, dt_bias of 4096, the bound 0 PASS\n", .{});
    }
    // red: each planted fault in the store loop's place for one call; its g must differ from the served gate's
    {
        const faults = [_]Fault{
            .{ .name = "member1_not_rounded", .form = .g0, .rows = 512, .k = ks.newKernel("turboexl3_test_ling_kdaproj_pfg_unrounded_fault", &kdaproj_pf.pfg0_in, &kdaproj_pf.pfg_out, comptime faultText(@embedFile("kernels/test/fault/kdagate_unrounded.metal")), pgemm.header, true) orelse return error.KernelCreateFailed },
            .{ .name = "dt_bias_next_channel", .form = .g0, .rows = 512, .k = ks.newKernel("turboexl3_test_ling_kdaproj_pfg_dtb_shift_fault", &kdaproj_pf.pfg0_in, &kdaproj_pf.pfg_out, comptime faultText(@embedFile("kernels/test/fault/kdagate_dtb_shift.metal")), pgemm.header, true) orelse return error.KernelCreateFailed },
            .{ .name = "sigmoid_fast_exp", .form = .g0, .rows = 4096, .k = ks.newKernel("turboexl3_test_ling_kdaproj_pfg_fastexp_fault", &kdaproj_pf.pfg0_in, &kdaproj_pf.pfg_out, comptime faultText(@embedFile("kernels/test/fault/kdagate_fastexp.metal")), pgemm.header, true) orelse return error.KernelCreateFailed },
            .{ .name = "gate_on_member2", .form = .g0, .rows = 512, .k = ks.newKernel("turboexl3_test_ling_kdaproj_pfg_member2_fault", &kdaproj_pf.pfg0_in, &kdaproj_pf.pfg_out, comptime faultText(@embedFile("kernels/test/fault/kdagate_member2.metal")), pgemm.header, true) orelse return error.KernelCreateFailed },
            .{ .name = "g1_ea_next_channel", .form = .g1, .rows = 512, .k = ks.newKernel("turboexl3_test_ling_kdaproj_pfg1_ea_shift_fault", &kdaproj_pf.pfg1_in, &kdaproj_pf.pfg_out, comptime faultText(@embedFile("kernels/test/fault/kdagate_g1_ea_shift.metal")), pgemm.header, true) orelse return error.KernelCreateFailed },
        };
        defer kdaproj_pf.gate_kernel_override.get().* = null;
        for (faults) |f| {
            const mark = g.mark();
            defer g.resetTo(mark);
            const x = try ds_test.randBf16(a, &g, rnd, &.{ 1, ci(f.rows), ci(in) });
            const want = try kdaproj_pf.kdaProjPrompt(s, x, p, .mul1);
            var wys: [3]T = undefined;
            for (&wys, want) |*d, y| d.* = try g.adopt(y);
            const g_want = try g.adopt(try servedGate(s, A_log, wys[1], dt_bias, bound));
            kdaproj_pf.gate_kernel_override.get().* = f.k;
            const got = try kdaproj_pf.kdaProjPromptGate(s, x, p, gateOf(f.form, A_log, ea, dt_bias), .mul1);
            kdaproj_pf.gate_kernel_override.get().* = null;
            var gys: [3]T = undefined;
            for (&gys, got) |*d, y| d.* = try g.adopt(y);
            const c = try compare(a, &g, gys[1], g_want);
            const caught = c.differ > 0;
            lines += 1;
            failed += @intFromBool(!caught);
            std.debug.print("TURBOEXL3_LING_KDA_GATE_RED fault={s} form={s} rows={d} differing={d}/{d} (want > 0) {s}\n", .{ f.name, @tagName(f.form), f.rows, c.differ, mlx.mlx_array_size(g_want), if (caught) "PASS" else "FAIL" });
        }
    }
    std.debug.print("TURBOEXL3_LING_KDA_GATE_DONE lines={d} failed={d} {s}\n", .{ lines, failed, if (failed == 0) "PASS" else "FAIL" });
    // 8 row counts x 2 forms x 3 members, 3 W row counts x 2 forms x 3 members, 5 faults
    try testing.expectEqual(@as(usize, 8 * 2 * 3 + 3 * 2 * 3 + 5), lines);
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── timing (TURBOEXL3_LING_TIME=1): guidance, never a speed verdict ──

const time_rows = [_]u32{ 512, 1996, 4096, 8192, 16384 };
const arm_names = [_][]const u8{ "served", "g0", "g1", "gate" };

const Arms = struct {
    s: mlx.mlx_stream,
    x: T,
    p: [3]root.Proj,
    A_log: T,
    ea: T,
    dt_bias: T,
    /// y_1 of `kdaProjPrompt` at these rows, evaluated: the gate-alone arm's input
    y1: T,
    /// the W pair: each member's W_hat, or null ctx for the per-call arms
    w: ?[3]T,

    /// arm 0: the served chain (`kdaProjPrompt` or `kdaProjPromptW`, then the served gate over y_1); 1, 2: G0, G1; 3: the
    /// served gate's launch alone over `y1`. The results the caller frees (up to 4).
    fn call(cx: Arms, arm: usize, out: *[4]T) !usize {
        switch (arm) {
            0 => {
                const ys = if (cx.w) |w| try kdaproj_pf.kdaProjPromptW(cx.s, cx.x, cx.p, w, .mul1) else try kdaproj_pf.kdaProjPrompt(cx.s, cx.x, cx.p, .mul1);
                out[0] = ys[0];
                out[1] = ys[1];
                out[2] = ys[2];
                out[3] = try servedGate(cx.s, cx.A_log, ys[1], cx.dt_bias, bound);
                return 4;
            },
            1, 2 => {
                const form: kdaproj_pf.GateForm = if (arm == 1) .g0 else .g1;
                const gate = gateOf(form, cx.A_log, cx.ea, cx.dt_bias);
                const ys = if (cx.w) |w| try kdaproj_pf.kdaProjPromptWGate(cx.s, cx.x, cx.p, w, gate, .mul1) else try kdaproj_pf.kdaProjPromptGate(cx.s, cx.x, cx.p, gate, .mul1);
                out[0] = ys[0];
                out[1] = ys[1];
                out[2] = ys[2];
                return 3;
            },
            else => {
                out[0] = try servedGate(cx.s, cx.A_log, cx.y1, cx.dt_bias, bound);
                return 1;
            },
        }
    }
};

fn timeRows(a: std.mem.Allocator, g: *G, rnd: std.Random, cx0: Arms, rows: u32, label: []const u8) !void {
    const mark = g.mark();
    defer g.resetTo(mark);
    defer _ = mlx.mlx_clear_cache();
    var cx = cx0;
    cx.x = try ds_test.randBf16(a, g, rnd, &.{ 1, ci(rows), ci(in) });
    const ys = try kdaproj_pf.kdaProjPrompt(cx.s, cx.x, cx.p, .mul1);
    var keep: [3]T = undefined;
    for (&keep, ys) |*d, y| d.* = try g.adopt(y);
    try g.evalAll(&keep);
    cx.y1 = keep[1];
    const n_arms = arm_names.len;
    const rounds = 10;
    var samples: [n_arms][rounds - 2]f64 = undefined;
    for (0..rounds) |round| for (0..n_arms) |arm| {
        var out: [4]T = undefined;
        const n = try cx.call(arm, &out);
        defer for (out[0..n]) |y| {
            _ = mlx.mlx_array_free(y);
        };
        const t0 = ds_test.nowNs();
        try g.evalAll(out[0..n]);
        const dt = ds_test.nowNs() - t0;
        if (round >= 2) samples[arm][round - 2] = @as(f64, @floatFromInt(dt)) / 1e6;
    };
    var served_min: f64 = 0;
    for (&samples, 0..) |*sm, arm| {
        std.mem.sort(f64, sm, {}, std.sort.asc(f64));
        const min = sm[0];
        const med = sm[sm.len / 2];
        if (arm == 0) served_min = min;
        std.debug.print("TURBOEXL3_LING_TIME kdaproj_gate {s} rows={d} arm={s} min_ms={d:.3} med_ms={d:.3} of_served={d:.3}\n", .{ label, rows, arm_names[arm], min, med, min / served_min });
    }
}

test "turboexl3 gpu ling timing: kdaproj gate: the served chain (kdaProjPrompt, then the served gate) against G0 and G1 and the gate alone at prompt rows, and the W pair at 4096 (TURBOEXL3_LING_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_LING_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s) or !config.get().nax) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x16A7E6);
    const rnd = prng.random();
    var p: [3]root.Proj = undefined;
    for (&p, outs) |*m, o| m.* = try ds_test.randProj(a, &g, rnd, null, in, o, 96);
    const A_log = try uniformF32(a, &g, rnd, outs[1], -3.0, 1.0);
    const dt_bias = try uniformF32(a, &g, rnd, outs[1], -4.0, 2.0);
    const ea = try g.adopt(try kdaproj_pf.gateEa(s, A_log));
    try g.evalAll(&.{ A_log, dt_bias, ea });
    const cx: Arms = .{ .s = s, .x = .{ .ctx = null }, .p = p, .A_log = A_log, .ea = ea, .dt_bias = dt_bias, .y1 = .{ .ctx = null }, .w = null };
    for (time_rows) |rows| try timeRows(a, &g, rnd, cx, rows, "per_call");
    // the W pair at 4096: each member's W_hat decoded once, outside the timing
    var w_hat: [3]T = undefined;
    for (&w_hat, p, outs) |*wm, m, o| {
        const dn = try root.denseCached(.{ .in = in, .out = o, .rate = format.Rate.fromN(96).?, .dec = .mul1, .out_dtype = .bfloat16 });
        wm.* = try g.adopt(try dn.dequantize(s, m.trellis));
    }
    try g.evalAll(&w_hat);
    var cxw = cx;
    cxw.w = w_hat;
    try timeRows(a, &g, rnd, cxw, 4096, "w_hats");
}
