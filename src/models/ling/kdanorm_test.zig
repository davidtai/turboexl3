//! L5, `kdanorm.kdaNormGatePrep` (the outside plugin turboexl3_ling's KDA_NORM_PREP): a KDA layer's gated norm written
//! straight into o_proj's prepared prompt rows, held word for word to the served chain: the host's fused norm-gate (its
//! test-only copy, kernels/test/copy/gdn_normgate.metal, at TAB 1 with the sigmoid gate, as the plugin serves it with
//! its prework), then o_proj's `linear`.
//! - GPU (TURBOEXL3_GPU=1, else skipped), at Ling-3.1-flash's KDA shapes (64 heads of 128; o_proj 8192 -> 4096 at K 6,
//!   the loader's I16 trellis words), y [1, S, 64, 128], z [1, S, 8192] (z_off 0) and norm_w [128] in bf16, eps 1e-6
//!   and the host's sigmoid table (`shared.sigmoidTable`, MLX's sigmoid of every bf16 pattern): at S 33 (the first rows
//!   past the register path), 512 and 4096 (the plugin's chunk calls), 1965 and 1996 (rows that end inside a block of
//!   the matmul), and 16384 (a layer-major call): xp against `Dense.prepare` of the served norm-gate's output (what
//!   o_proj's `linear` prepares), and o_proj's y (`Dense.piecePrepared` over the W_hat `Dense.dequantize` decodes once,
//!   then `Dense.finishOf`) against `root.linear` of that output, every word, both finite. Lines: TURBOEXL3_LING_KDA_NORM.
//! - GPU, the refusals before any launch (16 and 32 rows, the GEMV's and the register path's; y f32; z_off past the
//!   row; suh bf16); then the planted faults, each in the gate's place for one call at 512 rows, its xp against the
//!   served chain's (each must differ): norm_w of the next channel, z of the next head, the sum of squares over each
//!   lane's strided channels, and suh of the next block. Lines: TURBOEXL3_LING_KDA_NORM_RED, then
//!   TURBOEXL3_LING_KDA_NORM_DONE.
//! - Both on NAX (the served Ling path's; off it skipped), where `linear` takes rows 17 to 32 on its register path.
//! - Timing (TURBOEXL3_LING_TIME=1 with TURBOEXL3_GPU=1; asserts nothing, guidance only): the served pair (the host's
//!   norm-gate, then o_proj's prepare) against the fused launch at rows 512, 1996, 4096, 8192 and 16384, and o_proj's
//!   chain at 4096 (the norm-gate and `linearOver`'s prepare, matmul and finish against the fused launch, the matmul
//!   and the finish, both over one W_hat); interleaved round by round, min and median ms over 8 rounds. Lines:
//!   TURBOEXL3_LING_TIME kdanorm.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ks = sdk.kernel_launch;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const T = mlx.mlx_array;
const config = @import("../../core/config.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const format = @import("../../core/format.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const ds_test = @import("../deepseek_v41/deepseek_test.zig");
const shared = @import("shared.zig");
const kdanorm = @import("kdanorm.zig");

const testing = std.testing;

const hv: u32 = 64;
const dv: u32 = 128;
const width: u32 = hv * dv;
const o_out: u32 = 4096;
const eps_v: f32 = 1e-6;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

// ── the served chain ──

const ng_in = [_][*:0]const u8{ "y", "z", "norm_w", "eps", "sigtab" };
const ng_out = [_][*:0]const u8{"out"};
var served_ng_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The host's fused norm-gate as the plugin serves it at prompt rows (mlx-serve `gdnNormGateFusedTab(.., tab true)`,
/// the sigmoid gate), its test-only copy (the host's text under its `//! copied_from:` comment line): out bf16 [1, S,
/// HV·DV] of y [1, S, HV, DV] and z [1, S, 8192] at 0; its grid (32, S, HV), its threadgroup 32, its template
/// arguments in its order. Owned.
fn servedNormGate(s: mlx.mlx_stream, y: T, z: T, norm_w: T, eps: T, sigtab: T, sq: u32) !T {
    const k = served_ng_kernel orelse made: {
        const nk = ks.newKernel("turboexl3_test_ling_gdn_normgate", &ng_in, &ng_out, @embedFile("kernels/test/copy/gdn_normgate.metal"), "", true) orelse return error.KernelCreateFailed;
        served_ng_kernel = nk;
        break :made nk;
    };
    const cfg = try ks.buildConfig(.{
        .grid = .{ 32, sq, hv },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{&[_]c_int{ 1, ci(sq), ci(width) }},
        .out_dtypes = &.{.bfloat16},
        .template = &.{
            .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
            .{ .name = "HV", .value = .{ .int = @intCast(hv) } },
            .{ .name = "DV", .value = .{ .int = @intCast(dv) } },
            .{ .name = "ZSTRIDE", .value = .{ .int = @intCast(width) } },
            .{ .name = "ZOFF", .value = .{ .int = 0 } },
            .{ .name = "TAB", .value = .{ .int = 1 } },
            .{ .name = "SWISH", .value = .{ .int = 0 } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var o: [1]T = undefined;
    try ks.applyConfig(k, cfg, &.{ y, z, norm_w, eps, sigtab }, &o, s);
    return o[0];
}

/// o_proj as the loader hands it (8192 -> 4096 at K 6, the pack's I16 words)
const OProj = struct { lin: ref.Linear, p: root.Proj };

fn oProj(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, rnd: std.Random) !OProj {
    var lin = try ref.synthLinear(a, rnd, 1, width, o_out, format.Rate.fromN(96).?, .mul1);
    errdefer lin.deinit(a);
    const u = try g.hostArray(std.mem.sliceAsBytes(lin.trellis), &.{ ci(width / 16), ci(o_out / 16), 96 }, .uint16);
    var v = mlx.mlx_array_new();
    mlx.check(mlx.mlx_view(&v, u, .int16, s)) catch |e| {
        _ = mlx.mlx_array_free(v);
        return e;
    };
    return .{ .lin = lin, .p = .{
        .trellis = try g.adopt(v),
        .suh = try g.hostArray(std.mem.sliceAsBytes(lin.suh), &.{ci(width)}, .float16),
        .svh = try g.hostArray(std.mem.sliceAsBytes(lin.svh), &.{ci(o_out)}, .float16),
    } };
}

fn oDense() !*dense.Dense {
    return root.denseCached(.{ .in = width, .out = o_out, .rate = format.Rate.fromN(96).?, .dec = .mul1, .out_dtype = .bfloat16 });
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

/// A call's operands at S rows: y, z and norm_w drawn as bf16, the layer's eps, the table and o_proj's suh
const Inputs = struct { y: T, z: T, norm_w: T, eps: T, sigtab: T, suh: T };

fn opOf(in: Inputs) kdanorm.Op {
    return .{ .y = in.y, .z = in.z, .z_off = 0, .z_stride = width, .norm_w = in.norm_w, .eps = in.eps, .sigtab = in.sigtab, .suh = in.suh };
}

fn draw(a: std.mem.Allocator, g: *G, rnd: std.Random, sq: u32, base: Inputs) !Inputs {
    var in = base;
    in.y = try ds_test.randBf16(a, g, rnd, &.{ 1, ci(sq), ci(hv), ci(dv) });
    in.z = try ds_test.randBf16(a, g, rnd, &.{ 1, ci(sq), ci(width) });
    return in;
}

fn sentinel(comptime s: []const u8) [:0]const u8 {
    comptime {
        var buf: [s.len:0]u8 = undefined;
        @memcpy(buf[0..s.len], s);
        const final = buf;
        return &final;
    }
}

/// The fused launch's text with `piece` in the gate's place (a planted fault's)
fn faultText(comptime piece: []const u8) [:0]const u8 {
    return sentinel(dense.prep_1 ++ piece ++ dense.prep_3);
}

test "turboexl3 gpu ling kdanorm: kdaNormGatePrep's xp is o_proj's prepare of the served norm-gate's output and o_proj's y over it is linear's, word for word, at prompt rows 33 to 16384; its refusals before any launch; planted faults caught (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s) or !config.get().nax) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x15A0D1);
    const rnd = prng.random();
    var o = try oProj(a, &g, s, rnd);
    defer o.lin.deinit(a);
    const dn = try oDense();
    // o_proj's W_hat by its `linear`'s decode launch, evaluated once: every call below reads it
    const w_hat = try g.adopt(try dn.dequantize(s, o.p.trellis));
    try g.evalAll(&.{w_hat});
    const base: Inputs = .{
        .y = undefined,
        .z = undefined,
        .norm_w = try ds_test.randBf16(a, &g, rnd, &.{ci(dv)}),
        .eps = try g.adopt(mlx.mlx_array_new_float(eps_v)),
        .sigtab = try g.adopt(try shared.sigmoidTable(s, .bfloat16, false)),
        .suh = o.p.suh,
    };
    var lines: usize = 0;
    var failed: usize = 0;
    for ([_]u32{ 33, 512, 1965, 1996, 4096, 16384 }) |sq| {
        const mark = g.mark();
        defer g.resetTo(mark);
        defer _ = mlx.mlx_clear_cache();
        try testing.expectEqual(dense.Dense.Path.matmul, dn.pathFor(sq));
        const in = try draw(a, &g, rnd, sq, base);
        // the served chain: the host's norm-gate, then o_proj's `linear` (its prepare, for xp)
        const flat = try g.adopt(try servedNormGate(s, in.y, in.z, in.norm_w, in.eps, in.sigtab, sq));
        const xp_want = try g.adopt(try dn.prepare(s, flat, o.p.suh, sq));
        const y_want = try g.adopt(try root.linear(s, flat, o.p.trellis, o.p.suh, o.p.svh, .mul1));
        // the fused launch, then o_proj's matmul and finish over its rows
        const xp = try g.adopt(try kdanorm.kdaNormGatePrep(s, dn, opOf(in)));
        const zi = try g.adopt(try dn.piecePrepared(s, xp, w_hat, sq));
        const y = try g.adopt(try dn.finishOf(s, zi, o.p.svh, sq));
        inline for (.{ .{ "xp", xp, xp_want }, .{ "y", y, y_want } }) |c| {
            const r = try compare(a, &g, c[1], c[2]);
            const ok = r.differ == 0 and r.nonfinite == 0;
            lines += 1;
            failed += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_LING_KDA_NORM rows={d} what={s} differing={d}/{d} nonfinite={d} {s}\n", .{ sq, c[0], r.differ, mlx.mlx_array_size(c[2]), r.nonfinite, if (ok) "PASS" else "FAIL" });
        }
    }
    {
        const mark = g.mark();
        defer g.resetTo(mark);
        // the GEMV's rows and the register path's, y f32, z_off past z's row, suh bf16: refused before any launch
        for ([_]u32{ 16, 32 }) |sq| {
            const few = try draw(a, &g, rnd, sq, base);
            try testing.expectError(error.Exl3EngineNotImplemented, kdanorm.kdaNormGatePrep(s, dn, opOf(few)));
        }
        const in = try draw(a, &g, rnd, 512, base);
        var bad = opOf(in);
        bad.y = try g.astype(in.y, .float32);
        try testing.expectError(error.Exl3EngineNotImplemented, kdanorm.kdaNormGatePrep(s, dn, bad));
        bad = opOf(in);
        bad.z_off = 1;
        try testing.expectError(error.Exl3EngineNotImplemented, kdanorm.kdaNormGatePrep(s, dn, bad));
        bad = opOf(in);
        bad.suh = try g.astype(in.suh, .bfloat16);
        try testing.expectError(error.Exl3EngineNotImplemented, kdanorm.kdaNormGatePrep(s, dn, bad));
        std.debug.print("TURBOEXL3_LING_KDA_NORM refused: rows=16,32 (the GEMV's and the register path's), y f32, z_off past the row, suh bf16 PASS\n", .{});
    }
    {
        const mark = g.mark();
        defer g.resetTo(mark);
        const Fault = struct { name: []const u8, k: mlx.mlx_fast_metal_kernel };
        const faults = [_]Fault{
            .{ .name = "norm_w_next_channel", .k = ks.newKernel("turboexl3_test_ling_kdanorm_nw_shift_fault", &kdanorm.normprep_in, &kdanorm.normprep_out, comptime faultText(@embedFile("kernels/test/fault/kdanorm_nw_shift.metal")), decode.header, true) orelse return error.KernelCreateFailed },
            .{ .name = "z_next_head", .k = ks.newKernel("turboexl3_test_ling_kdanorm_z_next_head_fault", &kdanorm.normprep_in, &kdanorm.normprep_out, comptime faultText(@embedFile("kernels/test/fault/kdanorm_z_next_head.metal")), decode.header, true) orelse return error.KernelCreateFailed },
            .{ .name = "sum_strided", .k = ks.newKernel("turboexl3_test_ling_kdanorm_sum_strided_fault", &kdanorm.normprep_in, &kdanorm.normprep_out, comptime faultText(@embedFile("kernels/test/fault/kdanorm_sum_strided.metal")), decode.header, true) orelse return error.KernelCreateFailed },
            .{ .name = "suh_next_block", .k = ks.newKernel("turboexl3_test_ling_kdanorm_suh_next_block_fault", &kdanorm.normprep_in, &kdanorm.normprep_out, comptime faultText(@embedFile("kernels/test/fault/kdanorm_suh_next_block.metal")), decode.header, true) orelse return error.KernelCreateFailed },
        };
        defer for (faults) |f| decode.freeKernel(f.k);
        const in = try draw(a, &g, rnd, 512, base);
        const flat = try g.adopt(try servedNormGate(s, in.y, in.z, in.norm_w, in.eps, in.sigtab, 512));
        const xp_want = try g.adopt(try dn.prepare(s, flat, o.p.suh, 512));
        for (faults) |f| {
            kdanorm.kernel_override.get().* = f.k;
            defer kdanorm.kernel_override.get().* = null;
            const xp = try g.adopt(try kdanorm.kdaNormGatePrep(s, dn, opOf(in)));
            const r = try compare(a, &g, xp, xp_want);
            const caught = r.differ > 0;
            lines += 1;
            failed += @intFromBool(!caught);
            std.debug.print("TURBOEXL3_LING_KDA_NORM_RED fault={s} rows=512 differing={d}/{d} (want > 0) {s}\n", .{ f.name, r.differ, mlx.mlx_array_size(xp_want), if (caught) "PASS" else "FAIL" });
        }
    }
    // 6 row counts x (xp, y), 4 faults
    std.debug.print("TURBOEXL3_LING_KDA_NORM_DONE lines={d} failed={d} {s}\n", .{ lines, failed, if (failed == 0) "PASS" else "FAIL" });
    try testing.expectEqual(@as(usize, 16), lines);
    try testing.expectEqual(@as(usize, 0), failed);
}

// ── timing (TURBOEXL3_LING_TIME=1): guidance, never a speed verdict ──

const time_rows = [_]u32{ 512, 1996, 4096, 8192, 16384 };
const arm_names = [_][]const u8{ "served", "fused" };

const Arms = struct {
    s: mlx.mlx_stream,
    dn: *dense.Dense,
    p: root.Proj,
    in: Inputs,
    sq: u32,
    /// the o_proj chain: o_proj's W_hat (null: the pair alone)
    w: ?T,

    /// arm 0: the served chain (the host's norm-gate, then o_proj's prepare, or with `w` `linearOver`'s prepare,
    /// matmul and finish); 1: the fused launch (with `w`, then the matmul and the finish). One result, the caller's.
    fn call(cx: Arms, arm: usize) !T {
        if (arm == 0) {
            const flat = try servedNormGate(cx.s, cx.in.y, cx.in.z, cx.in.norm_w, cx.in.eps, cx.in.sigtab, cx.sq);
            defer _ = mlx.mlx_array_free(flat);
            const w = cx.w orelse return cx.dn.prepare(cx.s, flat, cx.p.suh, cx.sq);
            const zi = try cx.dn.pieceInner(cx.s, flat, w, cx.p.suh, cx.sq);
            defer _ = mlx.mlx_array_free(zi);
            return cx.dn.finishOf(cx.s, zi, cx.p.svh, cx.sq);
        }
        const xp = try kdanorm.kdaNormGatePrep(cx.s, cx.dn, opOf(cx.in));
        const w = cx.w orelse return xp;
        defer _ = mlx.mlx_array_free(xp);
        const zi = try cx.dn.piecePrepared(cx.s, xp, w, cx.sq);
        defer _ = mlx.mlx_array_free(zi);
        return cx.dn.finishOf(cx.s, zi, cx.p.svh, cx.sq);
    }
};

fn timeRows(a: std.mem.Allocator, g: *G, rnd: std.Random, cx0: Arms, sq: u32, label: []const u8) !void {
    const mark = g.mark();
    defer g.resetTo(mark);
    defer _ = mlx.mlx_clear_cache();
    var cx = cx0;
    cx.sq = sq;
    cx.in = try draw(a, g, rnd, sq, cx0.in);
    try g.evalAll(&.{ cx.in.y, cx.in.z });
    const rounds = 10;
    var samples: [arm_names.len][rounds - 2]f64 = undefined;
    for (0..rounds) |round| for (0..arm_names.len) |arm| {
        const out = try cx.call(arm);
        defer _ = mlx.mlx_array_free(out);
        const t0 = ds_test.nowNs();
        try g.evalAll(&.{out});
        const dt = ds_test.nowNs() - t0;
        if (round >= 2) samples[arm][round - 2] = @as(f64, @floatFromInt(dt)) / 1e6;
    };
    var served_min: f64 = 0;
    for (&samples, 0..) |*sm, arm| {
        std.mem.sort(f64, sm, {}, std.sort.asc(f64));
        const min = sm[0];
        const med = sm[sm.len / 2];
        if (arm == 0) served_min = min;
        std.debug.print("TURBOEXL3_LING_TIME kdanorm {s} rows={d} arm={s} min_ms={d:.3} med_ms={d:.3} of_served={d:.3}\n", .{ label, sq, arm_names[arm], min, med, min / served_min });
    }
}

test "turboexl3 gpu ling timing: kdanorm: the served pair (the host's norm-gate, then o_proj's prepare) against kdaNormGatePrep at prompt rows, and o_proj's chain at 4096 (TURBOEXL3_LING_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_LING_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s) or !config.get().nax) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x15A0D3);
    const rnd = prng.random();
    var o = try oProj(a, &g, s, rnd);
    defer o.lin.deinit(a);
    const dn = try oDense();
    const w_hat = try g.adopt(try dn.dequantize(s, o.p.trellis));
    try g.evalAll(&.{w_hat});
    const base: Inputs = .{
        .y = undefined,
        .z = undefined,
        .norm_w = try ds_test.randBf16(a, &g, rnd, &.{ci(dv)}),
        .eps = try g.adopt(mlx.mlx_array_new_float(eps_v)),
        .sigtab = try g.adopt(try shared.sigmoidTable(s, .bfloat16, false)),
        .suh = o.p.suh,
    };
    const cx: Arms = .{ .s = s, .dn = dn, .p = o.p, .in = base, .sq = 0, .w = null };
    for (time_rows) |sq| try timeRows(a, &g, rnd, cx, sq, "pair");
    var cw = cx;
    cw.w = w_hat;
    try timeRows(a, &g, rnd, cw, 4096, "o_proj");
}
