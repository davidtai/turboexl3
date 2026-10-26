//! The decode-chain benchmark (TURBOEXL3_GPU=1 TURBOEXL3_BENCH=1, else skipped): form-3 variants and Sushi on the same synthetic banks at
//! every integer rate K2..K8 (n 32..128) plus the fractional n40, Qwen3.8-Flash-Next's geometry (2560 x 640, top-10),
//! rows 1 / 4 / 8 (decode and MTP verify widths), 48 synthetic experts with rows sharing few of them (Flash-Next's
//! 512-expert routing shares few slots across verify rows, so weight reuse between rows is not what the bench rewards):
//!   bNkMg   - turboexl3's form 3 (split-K planes in both GEMVs, finish launches fuse SwiGLU, the down prepare and the
//!             reduce) with N output tiles per GEMV threadgroup (2N f32 accumulators per lane), a K partition of M slices
//!             and expert-grouped row-slots; `_full` puts every slice in its own threadgroup at every row count
//!   sushi   - the host's Sushi module chain (`sushi_exl3.moe`, through `test_support.zig`), the shipped control
//! TURBOEXL3_BENCH_N / _ROWS / _ARMS (comma lists) and _ROUNDS narrow a run.
//! Parity of every arm against the host reference first (an arm failing parity is reported, not timed); then rounds
//! interleaved over the arms with a rotated start, 3 warm, `rounds` timed; median / p10 us per layer and the effective
//! GB/s over the slots' trellis bytes. Lines: TURBOEXL3_BENCH_PARITY, TURBOEXL3_BENCH per (n, rows, arm). Relative numbers
//! only on a laptop GPU; the box's window gives the ones that count.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("test_support.zig");
const G = support.Graph;
const format = @import("format.zig");
const decode = @import("decode.zig");
const unfused = @import("unfused.zig");
const ref = @import("ref.zig");

const testing = std.testing;
const has_sushi = support.has_sushi;
const sushi = support.sushi;

fn ci(x: u32) c_int {
    return @intCast(x);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const Arm = enum { b2k4g, b1k4g, b1k4g_full, b1k2g, b1k1g, sushi };
const arms = [_]Arm{ .b2k4g, .b1k4g, .b1k4g_full, .b1k2g, .b1k1g, .sushi };
/// form-3 variants: bN = N output tiles per GEMV threadgroup (2N accumulators per lane), kM = the K partition (M slices
/// for gate/up; the down's follows from its width), g = expert-grouped row-slots (two slots that chose the same expert
/// share one lead threadgroup and one decode); the K threadgroups per row-slot follow the row count (adaptive), or `_full`
/// one per slice at every row count. A row's bytes depend on kM only.
const variants = [_]struct { bt: u32, ks: u32, group: u32, kpar: decode.Kernels3.KPar }{
    .{ .bt = 2, .ks = 4, .group = 2, .kpar = .adaptive },
    .{ .bt = 1, .ks = 4, .group = 2, .kpar = .adaptive },
    .{ .bt = 1, .ks = 4, .group = 2, .kpar = .full },
    .{ .bt = 1, .ks = 2, .group = 2, .kpar = .adaptive },
    .{ .bt = 1, .ks = 1, .group = 2, .kpar = .adaptive },
};
const rates = [_]u32{ 32, 40, 48, 64, 80, 96, 112, 128 };
const row_counts = [_]u32{ 1, 4, 8 };
const hidden: u32 = 2560;
const inter: u32 = 640;
const topk: u32 = 10;
const experts: u32 = 48;
/// timed rounds per arm (TURBOEXL3_BENCH_ROUNDS overrides, up to max_rounds)
const default_rounds: usize = 20;
const max_rounds: usize = 200;
const tolerance: f64 = 5e-3;

const Ctx = struct {
    s: mlx.mlx_stream,
    rows: u32,
    x2: mlx.mlx_array, // [rows, hidden] f32
    x3: mlx.mlx_array, // [1, rows, hidden] f32
    slots1: mlx.mlx_array, // [rows*topk] u32
    slots3: mlx.mlx_array, // [1, rows, topk] u32
    sc1: mlx.mlx_array,
    sc3: mlx.mlx_array,
    bank: decode.Kernels.Bank,
    forms: []*decode.Kernels3,
    dec: format.Decode,

    fn run(c: *const Ctx, arm: Arm) !mlx.mlx_array {
        switch (arm) {
            .b2k4g, .b1k4g, .b1k4g_full, .b1k2g, .b1k1g => return c.forms[@backingInt(arm)].moe(c.s, c.x2, c.slots1, c.sc1, c.bank, c.rows),
            .sushi => {
                if (!has_sushi) return error.SkipZigTest;
                const b: sushi.Bank = .{
                    .gate = .{ .trellis = c.bank.gate_t, .suh = c.bank.gate_suh, .svh = c.bank.gate_svh },
                    .up = .{ .trellis = c.bank.up_t, .suh = c.bank.up_suh, .svh = c.bank.up_svh },
                    .down = .{ .trellis = c.bank.down_t, .suh = c.bank.down_suh, .svh = c.bank.down_svh },
                };
                const dec: sushi.format.Decode = .{ .codebook = @fromBackingInt(@intCast(@backingInt(c.dec.codebook))), .window = @fromBackingInt(@intCast(@backingInt(c.dec.window))) };
                return sushi.moe(c.s, c.x3, b, c.slots3, c.sc3, dec, false);
            },
        }
    }
};

/// Whether `v` is in the comma list of the env var `name` (unset: every value is).
fn listed(name: [*:0]const u8, v: u32) bool {
    const raw = std.mem.span(std.c.getenv(name) orelse return true);
    var it = std.mem.splitScalar(u8, raw, ',');
    while (it.next()) |tok| if ((std.fmt.parseInt(u32, tok, 10) catch continue) == v) return true;
    return false;
}

/// Whether arm `arm` is in TURBOEXL3_BENCH_ARMS (a comma list of arm names; unset: every arm).
fn armListed(arm: Arm) bool {
    const raw = std.mem.span(std.c.getenv("TURBOEXL3_BENCH_ARMS") orelse return true);
    var it = std.mem.splitScalar(u8, raw, ',');
    while (it.next()) |tok| if (std.mem.eql(u8, tok, @tagName(arm))) return true;
    return false;
}

fn benchCase(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, n: u32, rows: u32) !void {
    var prng = std.Random.DefaultPrng.init(0xBE4C ^ n ^ (rows << 8));
    const rnd = prng.random();
    const rate = format.Rate.fromN(n).?;
    const dec: format.Decode = .mul1;
    var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
    defer bank.deinit(a);
    const nslots = rows * topk;
    const x = try a.alloc(f32, rows * hidden);
    defer a.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const slots = try a.alloc(u32, nslots);
    defer a.free(slots);
    const scores = try a.alloc(f32, nslots);
    defer a.free(scores);
    for (0..rows) |r| {
        var sum: f32 = 0;
        for (0..topk) |k| {
            slots[r * topk + k] = @intCast((k * 5 + r * 11) % experts); // rows share few experts, as Flash-Next's 512-expert routing does
            scores[r * topk + k] = 0.2 + rnd.float(f32);
            sum += scores[r * topk + k];
        }
        for (0..topk) |k| scores[r * topk + k] /= sum;
    }
    const y_ref = try a.alloc(f32, rows * hidden);
    defer a.free(y_ref);
    for (0..rows) |r| try ref.moeRowDec(a, &bank, dec, x[r * hidden ..][0..hidden], slots[r * topk ..][0..topk], scores[r * topk ..][0..topk], y_ref[r * hidden ..][0..hidden]);
    const mark = g.mark();
    defer g.resetTo(mark);
    const ht = ci(hidden / 16);
    const it = ci(inter / 16);
    const bank_arrays: decode.Kernels.Bank = .{
        .gate_t = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(experts), ht, it, ci(n) }, .uint16),
        .gate_suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ ci(experts), ci(hidden) }, .float16),
        .gate_svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ ci(experts), ci(inter) }, .float16),
        .up_t = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ ci(experts), ht, it, ci(n) }, .uint16),
        .up_suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ ci(experts), ci(hidden) }, .float16),
        .up_svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ ci(experts), ci(inter) }, .float16),
        .down_t = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ ci(experts), it, ht, ci(n) }, .uint16),
        .down_suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ ci(experts), ci(inter) }, .float16),
        .down_svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ ci(experts), ci(hidden) }, .float16),
    };
    const geo: decode.Geometry = .{ .hidden = hidden, .inter = inter, .topk = topk, .rate = rate, .dec = dec };
    var forms_store: [variants.len]decode.Kernels3 = undefined;
    var forms: [variants.len]*decode.Kernels3 = undefined;
    for (variants, 0..) |v, i| {
        forms_store[i] = try decode.Kernels3.initWith(geo, v.bt, v.ks, v.group);
        forms_store[i].kpar = v.kpar;
        forms[i] = &forms_store[i];
    }
    defer for (&forms_store) |*f| f.deinit();
    const c: Ctx = .{
        .s = s,
        .rows = rows,
        .x2 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ ci(rows), ci(hidden) }, .float32),
        .x3 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(hidden) }, .float32),
        .slots1 = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(nslots)}, .uint32),
        .slots3 = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ 1, ci(rows), ci(topk) }, .uint32),
        .sc1 = try g.hostArray(std.mem.sliceAsBytes(scores), &.{ci(nslots)}, .float32),
        .sc3 = try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, ci(rows), ci(topk) }, .float32),
        .bank = bank_arrays,
        .forms = &forms,
        .dec = dec,
    };
    // parity
    var timed: [arms.len]bool = @splat(false);
    const got = try a.alloc(f32, rows * hidden);
    defer a.free(got);
    for (arms, 0..) |arm, ai| {
        if (!armListed(arm)) continue;
        const y = c.run(arm) catch |err| {
            if (err == error.SkipZigTest) continue;
            std.debug.print("TURBOEXL3_BENCH_PARITY n={d} rows={d} arm={s} ERROR {s}\n", .{ n, rows, @tagName(arm), @errorName(err) });
            continue;
        };
        const y2 = try g.adopt(y);
        var yr = y2;
        if (mlx.mlx_array_dtype(y2) != .float32) yr = try g.astype(y2, .float32);
        var flat = yr;
        if (mlx.mlx_array_ndim(yr) != 2) flat = try g.reshape(yr, &.{ ci(rows), ci(hidden) });
        _ = try g.hostF32(flat, got);
        const e = ref.compare(got, y_ref);
        timed[ai] = e.max_ratio <= tolerance;
        std.debug.print("TURBOEXL3_BENCH_PARITY n={d} rows={d} arm={s} max_abs={e:.3} ratio={e:.3} {s}\n", .{ n, rows, @tagName(arm), e.max_abs, e.max_ratio, if (timed[ai]) "PASS" else "FAIL" });
    }
    // timing: interleaved rounds, rotated start
    const rounds: usize = blk: {
        const v = std.c.getenv("TURBOEXL3_BENCH_ROUNDS") orelse break :blk default_rounds;
        const k = std.fmt.parseInt(usize, std.mem.span(v), 10) catch break :blk default_rounds;
        break :blk @max(5, @min(k, max_rounds));
    };
    var samples_store: [arms.len][max_rounds]u64 = undefined;
    var samples: [arms.len][]u64 = undefined;
    for (&samples, 0..) |*sm, i| sm.* = samples_store[i][0..rounds];
    var r: usize = 0;
    while (r < rounds + 3) : (r += 1) {
        var ai: usize = 0;
        while (ai < arms.len) : (ai += 1) {
            const i = (ai + r) % arms.len;
            if (!timed[i]) continue;
            const t0 = nowNs();
            const y = try c.run(arms[i]);
            try mlx.check(mlx.mlx_array_eval(y));
            const dt = nowNs() - t0;
            _ = mlx.mlx_array_free(y);
            if (r >= 3) samples[i][r - 3] = dt;
        }
        _ = mlx.mlx_clear_cache();
    }
    const bytes = @as(f64, @floatFromInt(topk)) * 3.0 * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter)) * @as(f64, @floatFromInt(n)) / 128.0 * @as(f64, @floatFromInt(rows));
    for (arms, 0..) |arm, ai| {
        if (!timed[ai]) continue;
        std.mem.sort(u64, samples[ai], {}, std.sort.asc(u64));
        const med = @as(f64, @floatFromInt(samples[ai][rounds / 2])) / 1000.0;
        const p10 = @as(f64, @floatFromInt(samples[ai][rounds / 10])) / 1000.0;
        const min = @as(f64, @floatFromInt(samples[ai][0])) / 1000.0;
        std.debug.print("TURBOEXL3_BENCH n={d} k={d}.{d:0>2} rows={d} arm={s} min_us={d:.1} median_us={d:.1} p10_us={d:.1} gbps={d:.1}\n", .{ n, n / 16, (n % 16) * 100 / 16, rows, @tagName(arm), min, med, p10, bytes / (med * 1e-6) / 1e9 });
    }
}

test "turboexl3 bench: fused vs separately launched steps vs Sushi's chain, K2..K8 (+ n40), Qwen geometry, rows 1/4/8 (TURBOEXL3_BENCH=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_BENCH") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    std.debug.print("TURBOEXL3_BENCH_START geo={d}x{d} topk={d} experts={d} sushi={}\n", .{ hidden, inter, topk, experts, has_sushi });
    // TURBOEXL3_BENCH_N=80[,48...] restricts the rates (default: all of them)
    const only = if (std.c.getenv("TURBOEXL3_BENCH_N")) |v| std.mem.span(v) else "";
    for (rates) |n| {
        if (only.len > 0) {
            var it = std.mem.splitScalar(u8, only, ',');
            var hit = false;
            while (it.next()) |tok| hit = hit or (std.fmt.parseInt(u32, tok, 10) catch 0) == n;
            if (!hit) continue;
        }
        for (row_counts) |rows| {
            if (!listed("TURBOEXL3_BENCH_ROWS", rows)) continue;
            try benchCase(a, &g, s, n, rows);
        }
    }
    std.debug.print("TURBOEXL3_BENCH_END\n", .{});
}
