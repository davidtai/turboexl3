//! hw2, the hit wave's ILP retune: the grouped banked GEMVs' variants (`decode.group_variants`; `Kernels3.bankedGateUpV`,
//! `bankedDownV`; `Config.banked_variant_gu`, `banked_variant_dn`): gu3bg and down3bg on today's texts with Metal
//! defines before their header (gemv_group.metal's EXL3_GPF, EXL3_GXH, EXL3_GM2, EXL3_GSH; the MUL1 helper's EXL3_T1V
//! and EXL3_KU), each another schedule of the same reads, decodes and sums. CPU (always): the variant table, its names
//! and the config's count agree. GPU (TURBOEXL3_GPU=1, else skipped), at DeepSeek-V4.1's routed geometry (hidden 5120,
//! expert width 2304, swiglu_limit 10, MUL1) over three synthetic banks of 20, 12 and 8 experts at K3 and K2
//! (TURBOEXL3_BANKED_N):
//! - every variant's gate|up outputs (hd f16, hs f32) and down rows (y f32), on today's grouped texts (g4) and on hw4b's
//!   (g4m3: the M = 3 member path), are today's (g4 at variant 0), byte for byte (differing words 0, max_ulps 0; a
//!   skipped row-slot's y row 0), at rows 1, 2, 3, 4, 6, 7, 12, 16, 24, 36 and 48
//!   (TURBOEXL3_BANKED_ROWS), x bf16 and f32, the trellis I16 and U16, the pairs as banked_fold's byte test draws them
//!   (up to eight row-slots of one id from 6 rows: two leads of four, then of three; skipped row-slots);
//! - every route from a codeword to t (EXL3_T1V 0 to 4; 3 today's) gives route 3's t and the byte sum's, word for word,
//!   for all 65536 codeword states read from I16 and from U16 trellis words (a probe kernel per route);
//! With TURBOEXL3_BANKED_ILP_TIME=1 a further test times the whole banked call at a verify's 36 pairs (34 hits by the
//! recorded p10 / median / p90 profiles, 2 misses skipped), K3 and K2, x bf16, trellis I16, on banks of 48, 16 and 8
//! experts: per variant v the call with both GEMVs at v, with the gate|up GEMV alone at v and with the down GEMV alone at
//! v, and today's (0, 0); one call an evaluation, and a chain of 40 dependent calls an evaluation (call k + 1's x is call
//! k's y), its time a call; 2 warm then 9 timed rounds, the arms interleaved with a rotated start. With
//! TURBOEXL3_BANKED_ILP_M3=1 the timed calls run on hw4b's texts (`banked_m3`: variant 0 is g4m3, every variant on it).
//! TURBOEXL3_BANKED_ILP_ONLY_TIME skips the byte test (a window's timing run, after its cool gate).
//! Lines: TURBOEXL3_BANKED_ILP_BYTES, TURBOEXL3_BANKED_ILP_T, TURBOEXL3_BANKED_ILP_TIME.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("test_support.zig");
const G = support.Graph;
const format = @import("format.zig");
const ref = @import("ref.zig");
const root = @import("root.zig");
const decode = @import("decode.zig");
const config = @import("config.zig");

const testing = std.testing;

const hidden: u32 = 5120;
const inter: u32 = 2304;
const limit: c_int = 10;
const sigma: f32 = 8;
/// the byte test's banks (base, ext, transient), and the timing test's (72 experts)
const bank_experts = [3]u32{ 20, 12, 8 };
const time_experts = [3]u32{ 48, 16, 8 };
const n_var = decode.group_variants.len;
const Sel = decode.Kernels3.BankedSel;
/// the byte test's texts: today's grouped texts and hw4b's (every variant on each)
const sels = [_]Sel{ .g4, .g4m3 };

fn ci(x: anytype) c_int {
    return @intCast(x);
}

/// `v` rounded to bf16 (as MLX rounds), back in f32: a value both sides read alike.
fn bfExact(v: f32) f32 {
    return @bitCast(@as(u32, support.bf16Bits(v)) << 16);
}

/// A comma list of numbers from the environment (at most buf.len of them), else `default`.
fn envList(name: [*:0]const u8, default: []const u32, buf: []u32) []const u32 {
    const v = std.c.getenv(name) orelse return default;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
    while (it.next()) |tok| {
        if (n == buf.len) break;
        buf[n] = std.fmt.parseInt(u32, std.mem.trim(u8, tok, " "), 10) catch continue;
        n += 1;
    }
    return if (n == 0) default else buf[0..n];
}

test "turboexl3 banked ilp: variant 0 is today's (no defines, the stock kernel names), every other variant a distinct name with its defines, and the config counts them" {
    try testing.expectEqual(@as(usize, config.banked_variants), n_var);
    try testing.expectEqualStrings("", decode.group_variants[0].name);
    try testing.expectEqualStrings("", decode.group_variants[0].defines);
    try testing.expectEqual(@as(?u8, 0), decode.groupVariantOf(""));
    for (decode.group_variants[1..], 1..) |v, i| {
        try testing.expect(v.name.len > 0 and v.defines.len > 0);
        try testing.expect(std.mem.startsWith(u8, v.defines, "#define EXL3_") and std.mem.endsWith(u8, v.defines, "\n"));
        try testing.expectEqual(@as(?u8, @as(u8, @intCast(i))), decode.groupVariantOf(v.name));
        for (decode.group_variants[i + 1 ..]) |w| {
            try testing.expect(!std.mem.eql(u8, v.name, w.name));
            try testing.expect(!std.mem.eql(u8, v.defines, w.defines));
        }
    }
    try testing.expectEqual(@as(?u8, null), decode.groupVariantOf("nope"));
    // every grouped GEMV kernel a variant or a text set makes has a name of its own (MLX keys a library by its name)
    const names = decode.group_kernel_names;
    for (names, 0..) |a, i| for (names[i + 1 ..]) |b| try testing.expect(!std.mem.eql(u8, a, b));
    // every define a variant sets has its default in the header the defines go before
    for (decode.group_variants[1..]) |v| {
        var it = std.mem.splitScalar(u8, v.defines, '\n');
        while (it.next()) |line| {
            if (line.len == 0) continue;
            var parts = std.mem.splitScalar(u8, line, ' ');
            _ = parts.next();
            const name = parts.next().?;
            var buf: [64]u8 = undefined;
            try testing.expect(std.mem.indexOf(u8, decode.header_group, try std.fmt.bufPrint(&buf, "#ifndef {s}\n#define {s} ", .{ name, name })) != null);
        }
    }
}

fn rootBank(g: *G, b: *const ref.Bank, td: mlx.mlx_dtype) !root.Bank {
    const E = ci(b.experts);
    const ht = ci(b.hidden / 16);
    const it = ci(b.inter / 16);
    const n = ci(b.rate.n);
    return .{
        .gate = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(b.gate_t), &.{ E, ht, it, n }, td), .suh = try g.hostArray(std.mem.sliceAsBytes(b.gate_suh), &.{ E, ci(b.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.gate_svh), &.{ E, ci(b.inter) }, .float16) },
        .up = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(b.up_t), &.{ E, ht, it, n }, td), .suh = try g.hostArray(std.mem.sliceAsBytes(b.up_suh), &.{ E, ci(b.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.up_svh), &.{ E, ci(b.inter) }, .float16) },
        .down = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(b.down_t), &.{ E, it, ht, n }, td), .suh = try g.hostArray(std.mem.sliceAsBytes(b.down_suh), &.{ E, ci(b.inter) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(b.down_svh), &.{ E, ci(b.hidden) }, .float16) },
    };
}

fn evalBank(g: *G, b: root.Bank) !void {
    try g.evalAll(&.{ b.gate.trellis, b.gate.suh, b.gate.svh, b.up.trellis, b.up.suh, b.up.svh, b.down.trellis, b.down.suh, b.down.svh });
}

/// The banked launches' banks of three root banks.
fn kernelBanks(rb: *const [3]root.Bank) [3]decode.Kernels.Bank {
    var kb: [3]decode.Kernels.Bank = undefined;
    for (rb, &kb) |b, *o| o.* = .{ .gate_t = b.gate.trellis, .gate_suh = b.gate.suh, .gate_svh = b.gate.svh, .up_t = b.up.trellis, .up_suh = b.up.suh, .up_svh = b.up.svh, .down_t = b.down.trellis, .down_suh = b.down.suh, .down_svh = b.down.svh };
    return kb;
}

/// A Kernels3 at DeepSeek-V4.1's routed geometry (top-1, clamped, MUL1) as `root` builds it for a banked call, its
/// banked GEMVs grouped (4 row-slots a lead).
fn groupedKernels3(rate: format.Rate) !decode.Kernels3 {
    var k = try decode.Kernels3.initWith(.{ .hidden = hidden, .inter = inter, .topk = 1, .rate = rate, .dec = .mul1, .limit = @intCast(limit) }, 1, 4, 2);
    k.out_dtype = .float32;
    k.banked_group = 4;
    return k;
}

/// Two arrays' f32 words compared (`row_w` words a row): the compared words, the differing words, the largest ulp
/// distance (|bits(a) - bits(b)| as integers), the non-finite words of `got`; a skipped row's words are 0 in both
/// (`zero_skipped`: y's, counted as skipped_nonzero) or not compared (hd's and hs's: the launches leave them unwritten).
const Diff = struct { words: usize = 0, differing: usize = 0, max_ulps: u64 = 0, nonfinite: usize = 0, skipped_nonzero: usize = 0 };
fn wordDiff(got: []const f32, want: []const f32, row_w: usize, skipped: []const usize, zero_skipped: bool) Diff {
    var r: Diff = .{};
    for (got, want, 0..) |u, v, i| {
        const bu: u32 = @bitCast(u);
        const bv: u32 = @bitCast(v);
        const skip = std.mem.indexOfScalar(usize, skipped, i / row_w) != null;
        if (skip and !zero_skipped) continue;
        r.words += 1;
        if (bu != bv) {
            r.differing += 1;
            const iu: i64 = @as(i32, @bitCast(bu));
            const iv: i64 = @as(i32, @bitCast(bv));
            r.max_ulps = @max(r.max_ulps, @abs(iu - iv));
        }
        r.nonfinite += @intFromBool(!std.math.isFinite(u));
        if (skip) r.skipped_nonzero += @as(usize, @intFromBool(bu != 0)) + @as(usize, @intFromBool(bv != 0));
    }
    return r;
}

/// A byte test call's pairs at `rows` row-slots (ids packed bank << 24 | slot row; skipped row-slots `banked_skip_id`):
/// from 6 rows every token's expert base row 1 (so up to eight row-slots of one id: two leads), one of three ext rows in
/// each token, the rest drawn, and two skipped row-slots; below that drawn, row-slot 1 repeating row-slot 0 from 4 rows
/// and the last row-slot skipped from 3. Returns the skipped row-slots.
fn drawIds(rnd: std.Random, rows: usize, ids: []u32, skipped: *[2]usize) []const usize {
    for (0..rows) |p| {
        var bk: u32 = rnd.uintLessThan(u32, 3);
        var sl: u32 = rnd.uintLessThan(u32, bank_experts[bk]);
        if (rows >= 6) {
            if (p % 6 == 0) {
                bk = 0;
                sl = 1;
            } else if (p % 6 == 1) {
                bk = 1;
                sl = @intCast((p / 6) % 3);
            }
        }
        ids[p] = (bk << 24) | sl;
    }
    if (rows >= 4 and rows < 6) ids[1] = ids[0];
    if (rows >= 6) {
        skipped.* = .{ rows - 1, rows - 4 };
        for (skipped) |p| ids[p] = root.banked_skip_id;
        return skipped[0..2];
    }
    if (rows >= 3) {
        skipped[0] = rows - 1;
        ids[rows - 1] = root.banked_skip_id;
        return skipped[0..1];
    }
    return skipped[0..0];
}

/// The banked call's three outputs on `sel`'s texts at variants (`vgu`, `vdn`): hd (f16, as f32), hs and y; KEPT by `g`.
const Outs = struct { hd: mlx.mlx_array, hs: mlx.mlx_array, y: mlx.mlx_array };
fn callV(g: *G, k: *decode.Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, ids: mlx.mlx_array, kb: *const [3]decode.Kernels.Bank, rows: usize, sel: Sel, vgu: u8, vdn: u8) !Outs {
    const h = try k.bankedGateUpSelV(s, x, ids, kb, rows, sel, vgu);
    const hd = try g.adopt(h[0]);
    const hs = try g.adopt(h[1]);
    const y = try g.adopt(try k.bankedDownSelV(s, hd, hs, ids, kb, rows, sel, vdn));
    return .{ .hd = try g.astype(hd, .float32), .hs = hs, .y = y };
}

test "turboexl3 gpu banked ilp: every grouped GEMV variant's gate|up outputs and rows, on today's grouped texts and on hw4b's, are today's at variant 0, byte for byte, a skipped row-slot's row 0, at DeepSeek-V4.1's routed geometry (K3, K2) (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    // a timing run (its window's second binary run) skips the bytes
    if (std.c.getenv("TURBOEXL3_BANKED_ILP_ONLY_TIME") != null) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_BANKED_N", &.{ 48, 32 }, &nbuf);
    var rbuf: [16]u32 = undefined;
    const row_counts = envList("TURBOEXL3_BANKED_ROWS", &.{ 1, 2, 3, 4, 6, 7, 12, 16, 24, 36, 48 }, &rbuf);
    var failed = false;
    var checked: usize = 0;
    const max_w = root.banked_rows_max * hidden;
    const want = try a.alloc(f32, max_w);
    defer a.free(want);
    const got = try a.alloc(f32, max_w);
    defer a.free(got);
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x11F2 ^ @as(u64, n));
        const rnd = prng.random();
        var rb_u: [3]root.Bank = undefined;
        var rb_i: [3]root.Bank = undefined;
        for (&rb_u, &rb_i, bank_experts) |*u, *i, e| {
            var hb = try ref.synthBank(a, rnd, e, hidden, inter, rate);
            defer hb.deinit(a);
            u.* = try rootBank(&g, &hb, .uint16);
            i.* = try rootBank(&g, &hb, .int16);
            try evalBank(&g, u.*);
            try evalBank(&g, i.*);
        }
        const kb_u = kernelBanks(&rb_u);
        const kb_i = kernelBanks(&rb_i);
        var k = try groupedKernels3(rate);
        defer k.deinit();
        for (row_counts) |rows| {
            if (rows == 0 or rows > root.banked_rows_max) continue;
            const m2 = g.mark();
            defer g.resetTo(m2);
            var ids: [root.banked_rows_max]u32 = undefined;
            var sbuf: [2]usize = undefined;
            const skipped = drawIds(rnd, rows, ids[0..rows], &sbuf);
            const xh = try a.alloc(f32, rows * hidden);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
            const x32 = try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(hidden) }, .float32);
            const xbf = try g.astype(x32, .bfloat16);
            const ids_a = try g.hostArray(std.mem.sliceAsBytes(ids[0..rows]), &.{ci(rows)}, .uint32);
            const Arm = struct { x: mlx.mlx_array, kb: *const [3]decode.Kernels.Bank, name: []const u8 };
            for ([_]Arm{
                .{ .x = xbf, .kb = &kb_i, .name = "x=bf16 trellis=i16" },
                .{ .x = x32, .kb = &kb_u, .name = "x=f32 trellis=u16" },
                .{ .x = xbf, .kb = &kb_u, .name = "x=bf16 trellis=u16" },
                .{ .x = x32, .kb = &kb_i, .name = "x=f32 trellis=i16" },
            }) |arm| {
                const m3 = g.mark();
                defer g.resetTo(m3);
                const o0 = try callV(&g, &k, s, arm.x, ids_a, arm.kb, rows, .g4, 0, 0);
                try g.evalAll(&.{ o0.hd, o0.hs, o0.y });
                for (sels) |sel| for (0..n_var) |vi| {
                    if (sel == .g4 and vi == 0) continue;
                    const m4 = g.mark();
                    defer g.resetTo(m4);
                    const v: u8 = @intCast(vi);
                    const o = try callV(&g, &k, s, arm.x, ids_a, arm.kb, rows, sel, v, v);
                    try g.evalAll(&.{ o.hd, o.hs, o.y });
                    var line: Diff = .{};
                    var ok = true;
                    for ([_]struct { w: mlx.mlx_array, v: mlx.mlx_array, row_w: usize, y: bool }{
                        .{ .w = o0.hd, .v = o.hd, .row_w = inter, .y = false },
                        .{ .w = o0.hs, .v = o.hs, .row_w = inter / 128, .y = false },
                        .{ .w = o0.y, .v = o.y, .row_w = hidden, .y = true },
                    }) |c| {
                        const nw = rows * c.row_w;
                        const w = try g.hostF32(c.w, want[0..nw]);
                        const u = try g.hostF32(c.v, got[0..nw]);
                        const df = wordDiff(u, w, c.row_w, skipped, c.y);
                        line.words += df.words;
                        line.differing += df.differing;
                        line.max_ulps = @max(line.max_ulps, df.max_ulps);
                        line.skipped_nonzero += df.skipped_nonzero;
                        line.nonfinite += df.nonfinite;
                        if (df.differing != 0) ok = false;
                    }
                    if (line.nonfinite != 0 or line.skipped_nonzero != 0) ok = false;
                    if (!ok) failed = true;
                    checked += line.words;
                    std.debug.print("TURBOEXL3_BANKED_ILP_BYTES n={d} rows={d} {s} texts={t} variant={s} words={d} differing={d} max_ulps={d} nonfinite={d} skipped={d} skipped_nonzero={d} {s}\n", .{ n, rows, arm.name, sel, varName(v), line.words, line.differing, line.max_ulps, line.nonfinite, skipped.len, line.skipped_nonzero, if (ok) "OK" else "FAIL" });
                };
            }
        }
    }
    std.debug.print("TURBOEXL3_BANKED_ILP_BYTES total_words={d} {s}\n", .{ checked, if (failed) "FAIL" else "OK" });
    try testing.expect(!failed);
}

/// The routes from a MUL1 codeword to t (EXL3_T1V; 3 today's), each probed alone on every codeword state.
const t_routes = [_]u8{ 0, 1, 2, 3, 4 };
const t_dtypes = [_][]const u8{ "i16", "u16" };
/// The probe: thread i reads codeword state i from the code array's memory as the GEMVs read a trellis (its words as
/// uint, halfword i & 1 of word i >> 1) and writes exl3_mul1_t of it, under one route's define.
const t_probe_body: [:0]const u8 =
    \\    const uint i = thread_position_in_grid.x;
    \\    if (i >= 65536u) return;
    \\    const device uint* w = reinterpret_cast<const device uint*>(code);
    \\    const uint cw = (w[i >> 1u] >> (16u * (i & 1u))) & 0xffffu;
    \\    t[i] = exl3_mul1_t(cw);
    \\
;
const t_probe_headers = blk: {
    var h: [t_routes.len][:0]const u8 = undefined;
    for (t_routes, 0..) |r, i| h[i] = std.fmt.comptimePrint("#define EXL3_T1V {d}\n", .{r}) ++ decode.header;
    break :blk h;
};
const t_probe_names = blk: {
    var n: [t_routes.len][t_dtypes.len][:0]const u8 = undefined;
    for (t_routes, 0..) |r, i| {
        for (t_dtypes, 0..) |dt, j| n[i][j] = std.fmt.comptimePrint("turboexl3_t1v_probe_{d}_{s}", .{ r, dt });
    }
    break :blk n;
};

test "turboexl3 gpu banked ilp t routes: every EXL3_T1V route gives today's route's t, word for word, and the byte sum's, for all 65536 codeword states read from I16 and from U16 trellis words (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    if (std.c.getenv("TURBOEXL3_BANKED_ILP_ONLY_TIME") != null) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const ks = @import("sdk").kernel_launch;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const n_states: usize = 65536;
    // the reference: t = the byte sum of cw * 0x83DCD12D (mod 2^32), exact in f32
    const ref_t = try a.alloc(f32, n_states);
    defer a.free(ref_t);
    for (ref_t, 0..) |*r, cw| {
        const rr: u32 = @as(u32, @intCast(cw)) *% 0x83DCD12D;
        r.* = @floatFromInt((rr & 0xff) + ((rr >> 8) & 0xff) + ((rr >> 16) & 0xff) + (rr >> 24));
    }
    // every state once, in order, as U16 and as I16 (the same bits)
    const codes = try a.alloc(u16, n_states);
    defer a.free(codes);
    for (codes, 0..) |*c, i| c.* = @intCast(i);
    const mark = g.mark();
    defer g.resetTo(mark);
    const arr = [t_dtypes.len]mlx.mlx_array{
        try g.hostArray(std.mem.sliceAsBytes(codes), &.{ci(n_states)}, .int16),
        try g.hostArray(std.mem.sliceAsBytes(codes), &.{ci(n_states)}, .uint16),
    };
    try g.evalAll(&arr);
    const shp = [_]c_int{ci(n_states)};
    const cfg = try ks.buildConfig(.{ .grid = .{ @intCast(n_states), 1, 1 }, .threadgroup = .{ 256, 1, 1 }, .out_shapes = &.{&shp}, .out_dtypes = &.{.float32}, .template = &.{} });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const today = try a.alloc(f32, n_states);
    defer a.free(today);
    const got = try a.alloc(f32, n_states);
    defer a.free(got);
    var failed = false;
    for (t_dtypes, 0..) |dt, j| {
        // today's route first (route 3), then every route against it and against the reference
        const order = [_]usize{ 3, 0, 1, 2, 4 };
        for (order) |ri| {
            const r = t_routes[ri];
            const k = ks.newKernel(t_probe_names[ri][j].ptr, &.{"code"}, &.{"t"}, t_probe_body.ptr, t_probe_headers[ri].ptr, true) orelse return error.KernelCreateFailed;
            defer decode.freeKernel(k);
            var out: [1]mlx.mlx_array = undefined;
            try ks.applyConfig(k, cfg, &.{arr[j]}, &out, s);
            const o = try g.adopt(out[0]);
            try g.evalAll(&.{o});
            const v = try g.hostF32(o, if (r == 3) today else got);
            var d_today: usize = 0;
            var d_ref: usize = 0;
            for (v, today, ref_t) |x, y, z| {
                d_today += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
                d_ref += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(z)));
            }
            const ok = d_today == 0 and d_ref == 0;
            if (!ok) failed = true;
            std.debug.print("TURBOEXL3_BANKED_ILP_T route={d} trellis={s} states={d} differing_vs_route3={d} differing_vs_bytesum={d} {s}\n", .{ r, dt, v.len, d_today, d_ref, if (ok) "OK" else "FAIL" });
        }
    }
    try testing.expect(!failed);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// A timed call's hit multiplicities: `count[c]` experts named by c row-slots each (34 hit row-slots), as a 36-pair
/// verify call of DeepSeek-V4.1's names them (banked_test.zig's recorded p10, median and p90 profiles).
const Profile = struct { name: []const u8, count: [7]u32 };
const profiles = [_]Profile{
    .{ .name = "d18", .count = .{ 0, 10, 4, 2, 1, 0, 1 } },
    .{ .name = "d22", .count = .{ 0, 14, 5, 2, 1, 0, 0 } },
    .{ .name = "d29", .count = .{ 0, 25, 3, 1, 0, 0, 0 } },
};

/// The timed arms: today's (0, 0), then per variant v both GEMVs at v, the gate|up GEMV alone and the down GEMV alone.
const TArm = struct { vgu: u8, vdn: u8 };
const t_arms = blk: {
    var t: [1 + 3 * (n_var - 1)]TArm = undefined;
    t[0] = .{ .vgu = 0, .vdn = 0 };
    for (1..n_var) |v| {
        t[3 * v - 2] = .{ .vgu = @intCast(v), .vdn = @intCast(v) };
        t[3 * v - 1] = .{ .vgu = @intCast(v), .vdn = 0 };
        t[3 * v] = .{ .vgu = 0, .vdn = @intCast(v) };
    }
    break :blk t;
};

fn timedCall(k: *decode.Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, ids: mlx.mlx_array, kb: *const [3]decode.Kernels.Bank, rows: usize, arm: TArm) !mlx.mlx_array {
    const h = try k.bankedGateUpV(s, x, ids, kb, rows, arm.vgu);
    defer _ = mlx.mlx_array_free(h[0]);
    defer _ = mlx.mlx_array_free(h[1]);
    return k.bankedDownV(s, h[0], h[1], ids, kb, rows, arm.vdn);
}

test "turboexl3 gpu banked ilp time: the grouped GEMV variants against today's at a verify's 36 pairs (TURBOEXL3_GPU=1 TURBOEXL3_BANKED_ILP_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_BANKED_ILP_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_BANKED_N", &.{ 48, 32 }, &nbuf);
    const rows: usize = 36;
    const warm: usize = 2;
    const timed: usize = 9;
    const chain: usize = 40;
    const n_arms = t_arms.len;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0x7F01D ^ @as(u64, n));
        const rnd = prng.random();
        var rb: [3]root.Bank = undefined;
        for (&rb, time_experts) |*b, e| {
            var hb = try ref.synthBank(a, rnd, e, hidden, inter, rate);
            defer hb.deinit(a);
            b.* = try rootBank(&g, &hb, .int16);
            try evalBank(&g, b.*);
        }
        const kb = kernelBanks(&rb);
        var k = try groupedKernels3(rate);
        defer k.deinit();
        // TURBOEXL3_BANKED_ILP_M3: the calls on hw4b's texts (variant 0 g4m3, every variant on them)
        k.banked_m3 = std.c.getenv("TURBOEXL3_BANKED_ILP_M3") != null;
        for (profiles) |pf| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            // the hit experts: distinct global indexes over base rows 1..47 and the ext bank, never base row 0
            var order: [63]u32 = undefined;
            for (&order, 0..) |*o, i| o.* = @intCast(i + 1);
            rnd.shuffle(u32, &order);
            var hit: [rows]u32 = undefined;
            var nh: usize = 0;
            var d: usize = 0;
            for (pf.count, 0..) |cnt, c| for (0..cnt) |_| {
                const e = order[d];
                d += 1;
                const id: u32 = if (e < 48) e else (1 << 24) | (e - 48);
                for (0..c) |_| {
                    hit[nh] = id;
                    nh += 1;
                }
            };
            std.debug.assert(nh == rows - 2);
            rnd.shuffle(u32, hit[0..nh]);
            var ids: [rows]u32 = undefined;
            const miss = [2]usize{ 7, 29 };
            var hi: usize = 0;
            for (0..rows) |p| {
                if (p == miss[0] or p == miss[1]) {
                    ids[p] = root.banked_skip_id;
                } else {
                    ids[p] = hit[hi];
                    hi += 1;
                }
            }
            const xh = try a.alloc(f32, rows * hidden);
            defer a.free(xh);
            for (xh) |*v| v.* = bfExact(sigma * rnd.floatNorm(f32));
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xh), &.{ ci(rows), ci(hidden) }, .float32), .bfloat16);
            const ids_a = try g.hostArray(std.mem.sliceAsBytes(&ids), &.{ci(rows)}, .uint32);
            try g.evalAll(&.{ x, ids_a });
            var single: [n_arms][timed]u64 = undefined;
            var chained: [n_arms][timed]u64 = undefined;
            var ys: [chain]mlx.mlx_array = undefined;
            for (0..warm + timed) |round| {
                for (0..n_arms) |kk| {
                    const ai = (kk + round) % n_arms;
                    const arm = t_arms[ai];
                    const t0 = nowNs();
                    {
                        const y = try timedCall(&k, s, x, ids_a, &kb, rows, arm);
                        defer _ = mlx.mlx_array_free(y);
                        try mlx.check(mlx.mlx_array_eval(y));
                    }
                    const t1 = nowNs();
                    var made_y: usize = 0;
                    defer for (ys[0..made_y]) |y| {
                        _ = mlx.mlx_array_free(y);
                    };
                    var xc = x;
                    for (0..chain) |c| {
                        ys[c] = try timedCall(&k, s, xc, ids_a, &kb, rows, arm);
                        made_y += 1;
                        xc = ys[c];
                    }
                    try mlx.check(mlx.mlx_array_eval(ys[chain - 1]));
                    const t2 = nowNs();
                    if (round >= warm) {
                        single[ai][round - warm] = t1 - t0;
                        chained[ai][round - warm] = (t2 - t1) / chain;
                    }
                }
            }
            for (0..n_arms) |ai| {
                std.mem.sort(u64, &single[ai], {}, std.sort.asc(u64));
                std.mem.sort(u64, &chained[ai], {}, std.sort.asc(u64));
                const us = struct {
                    fn of(v: u64) f64 {
                        return @as(f64, @floatFromInt(v)) / 1000.0;
                    }
                };
                const arm = t_arms[ai];
                std.debug.print("TURBOEXL3_BANKED_ILP_TIME n={d} profile={s} distinct={d} texts={s} gu={s} dn={s} single_min_us={d:.1} single_med_us={d:.1} chain_min_us={d:.1} chain_med_us={d:.1}\n", .{ n, pf.name, d, if (k.banked_m3) "g4m3" else "g4", varName(arm.vgu), varName(arm.vdn), us.of(single[ai][0]), us.of(single[ai][timed / 2]), us.of(chained[ai][0]), us.of(chained[ai][timed / 2]) });
            }
        }
    }
}

/// A variant's name in the lines ("0" for today's).
fn varName(v: u8) []const u8 {
    return if (v == 0) "0" else decode.group_variants[v].name;
}
