//! `mlaRows` (the outside plugin turboexl3_ling's MLX_SERVE_LING_MLA_ROWS): the MLA chain's kv_b and head gate at decode
//! and verify rows in one GEMV launch, held word for word to the host's served launches.
//! - CPU (always): each planted fault's text is the launch's text with one piece swapped.
//! - GPU (TURBOEXL3_GPU=1, else skipped), at Ling-3.1-flash's two shapes (kv_b bf16 512 -> 16384, the head gate bf16
//!   4096 -> 64), each weight as the plugin's loader keeps it (the pack's f16 [out, in] viewed [in, out] and cast to
//!   bf16, its column-major layout kept) and transposed back for the launch, x bf16 [rows, in]:
//!   1. random rows at rows 1 to 8, 12 and 16 (serial decode's one row, an exact verify's rows, the widest a call
//!      takes) and caps 1, 2, 4 and 8 rows a threadgroup: every word of y against the served path, `servedCall`: MLX's
//!      matmul at one row, turboexl3_ling's denseRows at 2 and more (each row stored twice and the first viewed, one
//!      batched matmul: a one-row gemv a row);
//!   2. two built rows over an all-ones weight whose sums an order change moves (row 0: a lane's four K words 2^26,
//!      -2^26, 1, 1, which sum to 2 in MLX's order and to 0 in reverse; row 1: the same four words a K block apart at
//!      BN 1, a simdgroup apart at BN 8): the served words are 2 (the premise), and the launch gives them at every cap;
//!   3. red checks, each of which must differ from the served words: two planted faults on the built rows, each a pure
//!      reordering (`kernels/test/fault/`: the four products of a lane's K block added in reverse,
//!      `turboexl3_test_ling_mla_rows_krev_fault`; at BN 8 the simdgroups' sums added in reverse simdgroup order,
//!      `turboexl3_test_ling_mla_rows_sgrev_fault`), one weight word changed and two rows of x swapped;
//!   4. `mlaRows` under `Config.mla_rows` 8 against `mlaRowsWith` at 8, its refusal at 0, and the refusals of x, w and
//!      rows the launch does not take.
//! Lines: TURBOEXL3_LING_MLA_ROWS (_BUILT, _CONFIG, _RED, _DONE).
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ks = sdk.kernel_launch;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const T = mlx.mlx_array;
const config = @import("../../core/config.zig");
const decode = @import("../../core/decode.zig");
const ml = @import("mla_rows.zig");
const mla_latent_test = @import("mla_latent_test.zig");

const testing = std.testing;

fn ci(v: anytype) c_int {
    return @intCast(v);
}

/// Ling-3.1-flash's two plain projections of the MLA chain: kv_b (the latent's 512 to 64 heads of 128 nope keys and
/// 128 values) and the head gate (hidden 4096 to 64 heads)
pub const Shape = struct { name: []const u8, in: u32, out: u32 };
pub const shapes = [_]Shape{
    .{ .name = "kv_b", .in = 512, .out = 16384 },
    .{ .name = "gate", .in = 4096, .out = 64 },
};

/// A weight as the plugin's loader keeps it (`served`: the pack's f16 [out, in] words viewed [in, out] and cast to
/// bf16, its column-major layout kept, evaluated) and as the launch reads it (`rows`: its transpose back, [out, in]
/// row-major, a view)
pub const Weight = struct { served: T, rows: T };

pub fn loaderWeight(g: *G, raw: []const f16, in: u32, out: u32) !Weight {
    const w16 = try g.hostArray(std.mem.sliceAsBytes(raw), &.{ ci(out), ci(in) }, .float16);
    const served = try g.astype(try g.transposeAxes(w16, &.{ 1, 0 }), .bfloat16);
    try g.evalAll(&.{served});
    return .{ .served = served, .rows = try g.transposeAxes(served, &.{ 1, 0 }) };
}

/// MLX's expand_dims (denseRows): the consumer's binding where its `sdk.mlx` declares it, else libmlxc's own symbol
const expandDims = if (@hasDecl(mlx, "mlx_expand_dims")) mlx.mlx_expand_dims else struct {
    extern "c" fn mlx_expand_dims(res: *mlx.mlx_array, a: mlx.mlx_array, axis: c_int, s: mlx.mlx_stream) c_int;
}.mlx_expand_dims;

/// The host's served words of x [1, rows, in] over the loaded weight w [in, out], a new [rows, out] the caller frees: at
/// one row the trunk's `mm` (MLX's matmul, its gemv); at 2 and more rows turboexl3_ling's `denseRows`
/// (plugins/turboexl3_ling/src/ling_rows.zig), its statements: x as [rows, 1, in] stored twice on axis 1 and the first
/// viewed, w as [1, in, out], one batched matmul (a one-row gemv a row)
pub fn servedCall(s: mlx.mlx_stream, x3: T, w: T) !T {
    const xs = mlx.getShape(x3);
    if (xs.len != 3 or xs[0] != 1) return error.BadExl3Shape;
    const rows = xs[1];
    const k = xs[2];
    const n = mlx.getShape(w)[1];
    var tmp: [5]T = @splat(.{ .ctx = null });
    defer for (tmp) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (rows == 1) {
        tmp[0] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&tmp[0], x3, w, s));
        try mlx.check(mlx.mlx_reshape(&out, tmp[0], &[_]c_int{ 1, n }, 2, s));
        return out;
    }
    tmp[0] = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&tmp[0], x3, &[_]c_int{ rows, 1, k }, 3, s));
    {
        const parts = [_]T{ tmp[0], tmp[0] };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        tmp[1] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&tmp[1], vec, 1, s));
    }
    tmp[2] = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&tmp[2], tmp[1], &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ rows, 1, k }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    tmp[3] = mlx.mlx_array_new();
    try mlx.check(expandDims(&tmp[3], w, 0, s));
    tmp[4] = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&tmp[4], tmp[2], tmp[3], s));
    try mlx.check(mlx.mlx_reshape(&out, tmp[4], &[_]c_int{ rows, n }, 2, s));
    return out;
}

/// x bf16 [1, rows, in]: the first rows of `vals` (f32, rounded to bf16 by the cast)
pub fn xRows(g: *G, vals: []const f32, rows: u32, in: u32) !T {
    const n = @as(usize, rows) * in;
    return g.astype(try g.hostArray(std.mem.sliceAsBytes(vals[0..n]), &.{ 1, ci(rows), ci(in) }, .float32), .bfloat16);
}

/// The words of `got` that differ from `want`'s (bf16 [rows, out] both; another shape or dtype counts every word)
fn differ(a: std.mem.Allocator, g: *G, got: T, want: T) !usize {
    const n = mlx.mlx_array_size(want);
    if (!std.mem.eql(c_int, mlx.getShape(got), mlx.getShape(want)) or mlx.mlx_array_dtype(got) != .bfloat16 or mlx.mlx_array_dtype(want) != .bfloat16) return n;
    const p = try a.alloc(f32, n);
    defer a.free(p);
    const q = try a.alloc(f32, n);
    defer a.free(q);
    _ = try g.hostF32(try g.astype(got, .float32), p);
    _ = try g.hostF32(try g.astype(want, .float32), q);
    var d: usize = 0;
    for (p, q) |u, v| d += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
    return d;
}

/// The words of `y` equal to 2
fn twos(a: std.mem.Allocator, g: *G, y: T) !usize {
    const n = mlx.mlx_array_size(y);
    const p = try a.alloc(f32, n);
    defer a.free(p);
    _ = try g.hostF32(try g.astype(y, .float32), p);
    var c: usize = 0;
    for (p) |v| c += @intFromBool(v == 2.0);
    return c;
}

/// The built rows (f32, each value a bf16 word), [2, in]: row 0 one lane's four K words 2^26, -2^26, 1, 1 (MLX's order
/// sums them to 2: 2^26 - 2^26 + 1 + 1; in reverse 1 + 1 - 2^26 rounds to -2^26, then 0), row 1 the same words at K 0,
/// 128, 256 and 384 (lane 0's first word in four K blocks at BN 1; in simdgroups 0 to 3 at BN 8), every other word 0
fn builtRows(a: std.mem.Allocator, in: u32) ![]f32 {
    const v = try a.alloc(f32, 2 * @as(usize, in));
    @memset(v, 0);
    const big: f32 = 67108864.0;
    for ([_]f32{ big, -big, 1, 1 }, 0..) |word, j| {
        v[j] = word;
        v[in + 128 * j] = word;
    }
    return v;
}

/// The planted faults: the launch's text with one piece swapped for a pure reordering of its sums
/// (`kernels/test/fault/`): a lane's four products of a K block added in reverse; the simdgroups' sums added in reverse
/// simdgroup order
const fault_acc_krev = @embedFile("kernels/test/fault/mla_rows_acc_krev.metal");
const fault_tgsum_rev = @embedFile("kernels/test/fault/mla_rows_tgsum_rev.metal");
const text_krev: [:0]const u8 = ml.rows_head ++ fault_acc_krev ++ ml.rows_simd ++ ml.rows_tgsum ++ ml.rows_store;
const text_sgrev: [:0]const u8 = ml.rows_head ++ ml.rows_acc ++ ml.rows_simd ++ fault_tgsum_rev ++ ml.rows_store;

test "turboexl3 ling mla rows faults: each planted fault is the launch's text with one piece swapped, the accumulate for its reverse K order, the threadgroup sum for its reverse simdgroup order" {
    try testing.expect(!std.mem.eql(u8, fault_acc_krev, ml.rows_acc));
    try testing.expect(!std.mem.eql(u8, fault_tgsum_rev, ml.rows_tgsum));
    try testing.expectEqual(ml.text_rows.len - ml.rows_acc.len + fault_acc_krev.len, text_krev.len);
    try testing.expectEqualStrings(ml.text_rows[0..ml.rows_head.len], text_krev[0..ml.rows_head.len]);
    try testing.expectEqualStrings(ml.text_rows[ml.rows_head.len + ml.rows_acc.len ..], text_krev[ml.rows_head.len + fault_acc_krev.len ..]);
    const before_tgsum = ml.rows_head.len + ml.rows_acc.len + ml.rows_simd.len;
    try testing.expectEqualStrings(ml.text_rows[0..before_tgsum], text_sgrev[0..before_tgsum]);
    try testing.expectEqualStrings(ml.rows_store, text_sgrev[before_tgsum + fault_tgsum_rev.len ..]);
}

fn passFail(ok: bool) []const u8 {
    return if (ok) "PASS" else "FAIL";
}

/// One red check's line: its words differing from the served ones, which must be some
fn red(name: []const u8, fault: []const u8, d: usize, n: usize) bool {
    std.debug.print("TURBOEXL3_LING_MLA_ROWS_RED shape={s} fault={s} differing={d}/{d} (a fault must differ) {s}\n", .{ name, fault, d, n, passFail(d > 0) });
    return d > 0;
}

test "turboexl3 gpu ling mla rows: kv_b and the head gate in one GEMV launch give the served words (the one-row matmul, the exact verify's denseRows) at rows 1 to 8, 12 and 16 and caps 1 to 8, on random rows and on built rows an order change moves, weights as the loader keeps them; red checks (two planted reorderings, a changed weight word, swapped rows) differ; the configuration's path and the refusals (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    // a launch MLX refuses prints its error and fails its own line; the run goes on
    mla_latent_test.reportMlxErrors(true);
    defer mla_latent_test.reportMlxErrors(false);
    const krev = ks.newKernel("turboexl3_test_ling_mla_rows_krev_fault", &ml.rows_in, &ml.rows_out, text_krev, "", true) orelse return error.KernelCreateFailed;
    defer decode.freeKernel(krev);
    const sgrev = ks.newKernel("turboexl3_test_ling_mla_rows_sgrev_fault", &ml.rows_in, &ml.rows_out, text_sgrev, "", true) orelse return error.KernelCreateFailed;
    defer decode.freeKernel(sgrev);
    var prng = std.Random.DefaultPrng.init(0x4D1A_2085);
    const rnd = prng.random();
    const row_counts = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16 };
    const caps = [_]u32{ 1, 2, 4, 8 };
    var failed: usize = 0;
    var lines: usize = 0;
    var red_failed: usize = 0;
    for (shapes) |sh| {
        const sm = g.mark();
        defer g.resetTo(sm);
        const nw = @as(usize, sh.in) * sh.out;
        const raw = try a.alloc(f16, nw);
        defer a.free(raw);
        const ws: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(sh.in)));
        for (raw) |*v| v.* = @floatCast(rnd.floatNorm(f32) * ws);
        const w = try loaderWeight(&g, raw, sh.in, sh.out);
        const ones_raw = try a.alloc(f16, nw);
        defer a.free(ones_raw);
        @memset(ones_raw, @as(f16, 1.0));
        const ones = try loaderWeight(&g, ones_raw, sh.in, sh.out);
        const xs = try a.alloc(f32, @as(usize, ml.max_rows) * sh.in);
        defer a.free(xs);
        for (xs) |*v| v.* = rnd.floatNorm(f32);
        const built = try builtRows(a, sh.in);
        defer a.free(built);
        // 1. random rows at every cap
        for (row_counts) |rows| {
            const rm = g.mark();
            defer g.resetTo(rm);
            const x3 = try xRows(&g, xs, rows, sh.in);
            const x2 = try g.reshape(x3, &.{ ci(rows), ci(sh.in) });
            const want = try g.adopt(try servedCall(s, x3, w.served));
            for (caps) |cap| {
                lines += 1;
                const got = ml.mlaRowsWith(s, x2, w.rows, cap) catch |e| {
                    std.debug.print("TURBOEXL3_LING_MLA_ROWS shape={s} rows={d} cap={d}: {s} FAIL\n", .{ sh.name, rows, cap, @errorName(e) });
                    failed += 1;
                    continue;
                };
                const d = try differ(a, &g, try g.adopt(got), want);
                std.debug.print("TURBOEXL3_LING_MLA_ROWS shape={s} in={d} out={d} rows={d} cap={d} served={s} differing={d}/{d} {s}\n", .{ sh.name, sh.in, sh.out, rows, cap, if (rows == 1) "matmul" else "denseRows", d, @as(usize, rows) * sh.out, passFail(d == 0) });
                failed += @intFromBool(d != 0);
            }
        }
        // 2. the built rows over the all-ones weight, and 3. the planted reorderings on them
        {
            const rm = g.mark();
            defer g.resetTo(rm);
            const x3 = try xRows(&g, built, 2, sh.in);
            const x2 = try g.reshape(x3, &.{ 2, ci(sh.in) });
            const want = try g.adopt(try servedCall(s, x3, ones.served));
            // the premise: MLX's order sums both rows to 2 in every word, what each reordering below changes
            const n2 = try twos(a, &g, want);
            std.debug.print("TURBOEXL3_LING_MLA_ROWS_BUILT shape={s} served words equal to 2: {d}/{d} {s}\n", .{ sh.name, n2, 2 * sh.out, passFail(n2 == 2 * sh.out) });
            failed += @intFromBool(n2 != 2 * sh.out);
            for (caps) |cap| {
                lines += 1;
                const got = ml.mlaRowsWith(s, x2, ones.rows, cap) catch |e| {
                    std.debug.print("TURBOEXL3_LING_MLA_ROWS_BUILT shape={s} cap={d}: {s} FAIL\n", .{ sh.name, cap, @errorName(e) });
                    failed += 1;
                    continue;
                };
                const d = try differ(a, &g, try g.adopt(got), want);
                std.debug.print("TURBOEXL3_LING_MLA_ROWS_BUILT shape={s} rows=2 cap={d} differing={d}/{d} {s}\n", .{ sh.name, cap, d, 2 * sh.out, passFail(d == 0) });
                failed += @intFromBool(d != 0);
            }
            {
                const got = try g.adopt(try ml.mlaRowsOn(s, krev, x2, ones.rows, 8));
                red_failed += @intFromBool(!red(sh.name, "k_order_reversed", try differ(a, &g, got, want), 2 * sh.out));
            }
            if (ml.tileFor(sh.in, sh.out).?.bn > 1) {
                const got = try g.adopt(try ml.mlaRowsOn(s, sgrev, x2, ones.rows, 8));
                red_failed += @intFromBool(!red(sh.name, "group_order_reversed", try differ(a, &g, got, want), 2 * sh.out));
            }
        }
        // 3. a value fault and a route fault at 4 rows
        {
            const rm = g.mark();
            defer g.resetTo(rm);
            const rows: u32 = 4;
            const x3 = try xRows(&g, xs, rows, sh.in);
            const x2 = try g.reshape(x3, &.{ ci(rows), ci(sh.in) });
            const want = try g.adopt(try servedCall(s, x3, w.served));
            // one weight word every row reads (output 3, input 5) plus 4
            {
                const bad = try a.dupe(f16, raw);
                defer a.free(bad);
                const at = 3 * @as(usize, sh.in) + 5;
                bad[at] = @floatCast(@as(f32, bad[at]) + 4.0);
                const wb = try loaderWeight(&g, bad, sh.in, sh.out);
                const got = try g.adopt(try ml.mlaRowsWith(s, x2, wb.rows, 8));
                red_failed += @intFromBool(!red(sh.name, "weight_word_changed", try differ(a, &g, got, want), rows * sh.out));
            }
            // rows 0 and 1 of x swapped
            {
                const sw = try a.dupe(f32, xs[0 .. @as(usize, rows) * sh.in]);
                defer a.free(sw);
                for (0..sh.in) |j| std.mem.swap(f32, &sw[j], &sw[sh.in + j]);
                const xw = try g.reshape(try xRows(&g, sw, rows, sh.in), &.{ ci(rows), ci(sh.in) });
                const got = try g.adopt(try ml.mlaRowsWith(s, xw, w.rows, 8));
                red_failed += @intFromBool(!red(sh.name, "rows_swapped", try differ(a, &g, got, want), rows * sh.out));
            }
        }
        // 4. the configuration's path: Config.mla_rows 8 gives mlaRowsWith's words at 8; 0 refuses
        {
            const rm = g.mark();
            defer g.resetTo(rm);
            const saved = config.get().*;
            defer config.configure(saved) catch {};
            var cfg = saved;
            cfg.mla_rows = 8;
            try config.configure(cfg);
            const rows: u32 = 3;
            const x2 = try g.reshape(try xRows(&g, xs, rows, sh.in), &.{ ci(rows), ci(sh.in) });
            try testing.expect(ml.takes(rows, sh.in, sh.out));
            const by_cfg = try g.adopt(try ml.mlaRows(s, x2, w.rows));
            const by_cap = try g.adopt(try ml.mlaRowsWith(s, x2, w.rows, 8));
            const d = try differ(a, &g, by_cfg, by_cap);
            cfg.mla_rows = 0;
            try config.configure(cfg);
            try testing.expect(!ml.takes(rows, sh.in, sh.out));
            const refused = if (ml.mlaRows(s, x2, w.rows)) |y| blk: {
                _ = mlx.mlx_array_free(y);
                break :blk false;
            } else |e| e == error.Exl3EngineNotImplemented;
            std.debug.print("TURBOEXL3_LING_MLA_ROWS_CONFIG shape={s} rows={d} mla_rows 8 against cap 8 differing={d}/{d}, mla_rows 0 refused={} {s}\n", .{ sh.name, rows, d, rows * sh.out, refused, passFail(d == 0 and refused) });
            failed += @intFromBool(d != 0 or !refused);
        }
        // the refusals, before any launch: f32 x, an f16 weight, rows past 16, inputs that differ
        {
            const rm = g.mark();
            defer g.resetTo(rm);
            const x32 = try g.zeros(&.{ 1, ci(sh.in) }, .float32);
            try testing.expectError(error.Exl3EngineNotImplemented, ml.mlaRowsWith(s, x32, w.rows, 8));
            const xb = try g.zeros(&.{ 1, ci(sh.in) }, .bfloat16);
            try testing.expectError(error.Exl3EngineNotImplemented, ml.mlaRowsWith(s, xb, try g.astype(w.rows, .float16), 8));
            const wide = try g.zeros(&.{ ci(ml.max_rows + 1), ci(sh.in) }, .bfloat16);
            try testing.expectError(error.Exl3EngineNotImplemented, ml.mlaRowsWith(s, wide, w.rows, 8));
            const short = try g.zeros(&.{ 1, ci(sh.in - 128) }, .bfloat16);
            try testing.expectError(error.BadExl3Shape, ml.mlaRowsWith(s, short, w.rows, 8));
        }
    }
    // a weight MLX's matmul runs another tile at (K 64: SM 8, SN 4) is refused before any launch
    {
        const xk = try g.zeros(&.{ 1, 64 }, .bfloat16);
        const wk = try g.zeros(&.{ 4096, 64 }, .bfloat16);
        try testing.expectError(error.Exl3EngineNotImplemented, ml.mlaRowsWith(s, xk, wk, 8));
    }
    std.debug.print("TURBOEXL3_LING_MLA_ROWS_DONE {d} lines, {d} failed; red checks failed {d}\n", .{ lines, failed, red_failed });
    try testing.expectEqual(@as(usize, 0), failed);
    try testing.expectEqual(@as(usize, 0), red_failed);
}
