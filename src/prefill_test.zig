//! The prefill chain's GPU parity and timing (TURBOEXL3_GPU=1, else skipped). Parity: `prefill.Prefill.moe` against the host
//! reference (`ref.moeRowDec`) on sampled rows and against Sushi's module (`sushi_exl3.moe`, its prefill arm above 16 rows)
//! on every row; byte for byte: every option set that only moves work against the same set without it, and the 128-row
//! tiles (`moeT128`, call and stages) against the mixed chain. Timing (TURBOEXL3_PF_TIME=1; TURBOEXL3_DEC_TIME=1 for the
//! decode chain): both engines on the same synthetic bank and routing, rounds interleaved with a rotated start, the minimum
//! and median over rounds; TURBOEXL3_PF_E (experts, default 128) and TURBOEXL3_PF_ROWS (comma list, default 2048,340) set the
//! case: 128 experts x 2048 rows give Qwen's ~160 rows per expert of an 8192-token chunk on 512 experts, 340 rows its ~27-row
//! tail chunk. Lines: TURBOEXL3_PF_PARITY, TURBOEXL3_PF_TIME.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("test_support.zig");
const G = support.Graph;
const format = @import("format.zig");
const ref = @import("ref.zig");
const prefill = @import("prefill.zig");

const testing = std.testing;
const has_sushi = support.has_sushi;
const sushi = support.sushi;

const hidden: u32 = 2560;
const inter: u32 = 640;
const topk: u32 = 10;

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// One synthetic layer on the GPU: the bank as both engines take it, a routing and the inputs.
const Case = struct {
    rows: u32,
    experts: u32,
    rate: format.Rate,
    bank: ref.Bank,
    ops: prefill.Operands,
    x_f32: []f32, // the bf16-rounded inputs as f32 (what both engines read)
    slots: []u32,
    scores: []f32,
    x_bf: mlx.mlx_array, // [rows, hidden] bf16
    x3: mlx.mlx_array, // [1, rows, hidden] bf16
    inds1: mlx.mlx_array, // [rows * topk] u32
    inds3: mlx.mlx_array,
    sc1: mlx.mlx_array, // [rows * topk] f32
    sc3: mlx.mlx_array,

    fn deinit(c: *Case, a: std.mem.Allocator) void {
        c.bank.deinit(a);
        a.free(c.x_f32);
        a.free(c.slots);
        a.free(c.scores);
    }
};

fn makeCase(a: std.mem.Allocator, g: *G, rnd: std.Random, rows: u32, experts: u32, n: u32, skew: f64, real: ?[]const u32) !Case {
    if (experts < topk) return error.TooFewExperts;
    const rate = format.Rate.fromN(n).?;
    var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
    errdefer bank.deinit(a);
    const E = ci(experts);
    const ops: prefill.Operands = .{
        .gate_t = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ E, ci(hidden / 16), ci(inter / 16), ci(n) }, .uint16),
        .gate_suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ E, ci(hidden) }, .float16),
        .gate_svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ E, ci(inter) }, .float16),
        .up_t = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ E, ci(hidden / 16), ci(inter / 16), ci(n) }, .uint16),
        .up_suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ E, ci(hidden) }, .float16),
        .up_svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ E, ci(inter) }, .float16),
        .down_t = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ E, ci(inter / 16), ci(hidden / 16), ci(n) }, .uint16),
        .down_suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ E, ci(inter) }, .float16),
        .down_svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ E, ci(hidden) }, .float16),
    };
    const S = rows * topk;
    const slots = try a.alloc(u32, S);
    errdefer a.free(slots);
    const scores = try a.alloc(f32, S);
    errdefer a.free(scores);
    // distinct experts per row, scores positive and summing to one (a softmax's top-k renormalized). Popularity: uniform, or
    // lognormal with sigma `skew` (Gumbel top-k: each row draws k distinct experts with probability ~ popularity)
    const logw = try a.alloc(f64, experts);
    defer a.free(logw);
    for (logw) |*w| w.* = skew * rnd.floatNorm(f64);
    const keys = try a.alloc(f64, experts);
    defer a.free(keys);
    for (0..rows) |r| {
        if (real) |rs| {
            for (0..topk) |k| slots[r * topk + k] = rs[r * topk + k];
        } else for (keys, logw) |*kv, lw| kv.* = lw - @log(-@log(@max(rnd.float(f64), 1e-300)));
        if (real == null) for (0..topk) |k| {
            var best: usize = 0;
            for (keys, 0..) |kv, e| {
                if (kv > keys[best]) best = e;
            }
            slots[r * topk + k] = @intCast(best);
            keys[best] = -std.math.inf(f64);
        };
        var sum: f32 = 0;
        for (scores[r * topk ..][0..topk]) |*w| {
            w.* = 0.05 + rnd.float(f32);
            sum += w.*;
        }
        for (scores[r * topk ..][0..topk]) |*w| w.* /= sum;
    }
    const x_f32 = try a.alloc(f32, rows * hidden);
    errdefer a.free(x_f32);
    for (x_f32) |*v| v.* = rnd.floatNorm(f32);
    const xf = try g.hostArray(std.mem.sliceAsBytes(x_f32), &.{ ci(rows), ci(hidden) }, .float32);
    const x_bf = try g.astype(xf, .bfloat16);
    // the host values both engines read: x rounded to bf16
    const back = try g.astype(x_bf, .float32);
    _ = try g.hostF32(back, x_f32);
    const inds1 = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(S)}, .uint32);
    const sc1 = try g.hostArray(std.mem.sliceAsBytes(scores), &.{ci(S)}, .float32);
    const c: Case = .{
        .rows = rows,
        .experts = experts,
        .rate = rate,
        .bank = bank,
        .ops = ops,
        .x_f32 = x_f32,
        .slots = slots,
        .scores = scores,
        .x_bf = x_bf,
        .x3 = try g.reshape(x_bf, &.{ 1, ci(rows), ci(hidden) }),
        .inds1 = inds1,
        .inds3 = try g.reshape(inds1, &.{ 1, ci(rows), ci(topk) }),
        .sc1 = sc1,
        .sc3 = try g.reshape(sc1, &.{ 1, ci(rows), ci(topk) }),
    };
    try g.evalAll(&.{ ops.gate_t, ops.up_t, ops.down_t, ops.gate_suh, ops.up_suh, ops.down_suh, ops.gate_svh, ops.up_svh, ops.down_svh, c.x_bf, c.x3, c.inds1, c.inds3, c.sc1, c.sc3 });
    return c;
}

fn sushiBank(c: *const Case) sushi.Bank {
    return .{
        .gate = .{ .trellis = c.ops.gate_t, .suh = c.ops.gate_suh, .svh = c.ops.gate_svh },
        .up = .{ .trellis = c.ops.up_t, .suh = c.ops.up_suh, .svh = c.ops.up_svh },
        .down = .{ .trellis = c.ops.down_t, .suh = c.ops.down_suh, .svh = c.ops.down_svh },
    };
}

fn sushiDec() sushi.format.Decode {
    return .{ .codebook = .mul1 };
}

fn readF32(a: std.mem.Allocator, g: *G, y: mlx.mlx_array) ![]f32 {
    const yf = try g.astype(y, .float32);
    const out = try a.alloc(f32, mlx.mlx_array_size(yf));
    _ = try g.hostF32(yf, out);
    return out;
}

test "turboexl3 gpu prefill: the fused chain equals the host reference and Sushi's prefill arm (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const Spec = struct { rows: u32, experts: u32, n: u32 };
    // 12 experts x 400 rows: ~333 rows per expert (two full M tiles and a partial one); 32 x 64: 20 per expert (partial tiles
    // only, simdgroups with no rows); 24 x 17: the first width past the decode chain (experts >= topk: rows pick distinct experts);
    // 10 x 40: every row takes all 10 experts, so each has exactly 40 rows: at bm 64 / 128 a shared-decode tile whose second
    // simdgroup row holds 8 (half_rows' first-fragment-row MMA), on the all-register path a 32-row and an 8-row item
    for ([_]Spec{ .{ .rows = 400, .experts = 12, .n = 80 }, .{ .rows = 64, .experts = 32, .n = 80 }, .{ .rows = 17, .experts = 24, .n = 80 }, .{ .rows = 96, .experts = 16, .n = 48 }, .{ .rows = 40, .experts = 10, .n = 80 } }) |sp| {
        var prng = std.Random.DefaultPrng.init(0xF00D ^ sp.rows ^ (sp.experts << 16));
        const rnd = prng.random();
        const mark = g.mark();
        defer g.resetTo(mark);
        var c = try makeCase(a, &g, rnd, sp.rows, sp.experts, sp.n, 0, null);
        defer c.deinit(a);
        // the reference on a few rows (it decodes every weight on the CPU), once per case: every option set and engine is
        // measured against the same rows
        const sample = [_]u32{ 0, sp.rows / 2, sp.rows - 1 };
        const refs = try a.alloc(f32, sample.len * hidden);
        defer a.free(refs);
        for (sample, 0..) |r, i| try ref.moeRowDec(a, &c.bank, .mul1, c.x_f32[r * hidden ..][0..hidden], c.slots[r * topk ..][0..topk], c.scores[r * topk ..][0..topk], refs[i * hidden ..][0..hidden]);
      // the down-GEMM reshapes (128-deep stages, several output blocks per threadgroup; one single-buffered in the padded layout)
      // and the register path's integer-rate reader (sintr: all-register, mixed at bm 128, the served launch) are checked byte
      // for byte against the same option set without them as well
      for ([_]prefill.Options{ .{}, .{ .small = .all }, .{ .small = .mixed, .dbuf = true, .unroll = true }, .{ .dbuf = true, .fragb = true }, .{ .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0 }, .{ .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0 }, .{ .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .defer_affine = true }, .{ .defer_affine = true }, .{ .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .defer_affine = true, .split = 2 },
          .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .defer_affine = true, .bk128 = true },
          .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .defer_affine = true, .ncb = 4 },
          .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .defer_affine = true, .bk128 = true, .ncb = 2 },
          .{ .bk128 = true, .ncb = 2 },
          .{ .small = .all, .intr = sp.n % 16 == 0, .sintr = sp.n % 16 == 0 },
          .{ .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .sintr = sp.n % 16 == 0, .defer_affine = true },
          .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .sintr = sp.n % 16 == 0, .defer_affine = true, .ncb = 4 },
          .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .sintr = sp.n % 16 == 0, .defer_affine = true, .ncb = 4, .froute = true },
          .{ .small = .all, .froute = true },
          .{ .froute = true },
          .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .sintr = sp.n % 16 == 0, .defer_affine = true, .ncb = 4, .froute = true, .aux = 8 },
          .{ .small = .all, .aux = 32 },
          // the A layouts (alay 1: 32-blocks, 16-byte loads; 2: 16-blocks, 8-byte loads, the served layout) and the
          // first-fragment-row MMA (half_rows) on the served launch, each alone and with half_rows; both at bm 128 mixed, on the
          // shared-decode body alone (no deferred map) and on the all-register path
          .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .sintr = sp.n % 16 == 0, .defer_affine = true, .ncb = 4, .froute = true, .alay = 1 },
          .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .sintr = sp.n % 16 == 0, .defer_affine = true, .ncb = 4, .froute = true, .alay = 2 },
          .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .sintr = sp.n % 16 == 0, .defer_affine = true, .ncb = 4, .froute = true, .half_rows = true },
          .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .sintr = sp.n % 16 == 0, .defer_affine = true, .ncb = 4, .froute = true, .alay = 1, .half_rows = true },
          .{ .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = sp.n % 16 == 0, .defer_affine = true, .alay = 2, .half_rows = true },
          .{ .dbuf = true, .fragb = true, .alay = 1, .half_rows = true },
          .{ .small = .all, .intr = sp.n % 16 == 0, .sintr = sp.n % 16 == 0, .alay = 2, .half_rows = true } }) |opts| {
        var pf = try prefill.Prefill.init(.{ .hidden = hidden, .inter = inter, .topk = topk, .experts = sp.experts, .rate = c.rate, .dec = .mul1 }, opts);
        defer pf.deinit();
        const y32 = try g.adopt(try pf.moe(s, c.x_bf, c.inds1, c.sc1, c.ops, sp.rows, .float32));
        const got = try readF32(a, &g, y32);
        defer a.free(got);
        const worstVsRef = struct {
            fn run(y: []const f32, rf: []const f32, rows: []const u32) f64 {
                var w: f64 = 0;
                for (rows, 0..) |r, i| w = @max(w, ref.compare(y[r * hidden ..][0..hidden], rf[i * hidden ..][0..hidden]).max_ratio);
                return w;
            }
        }.run;
        const lib_f32 = worstVsRef(got, refs, &sample);
        // the served dtype: both engines in bf16 (one bf16 ulp is 2^-8 of a value, so up to ~3e-2 of the rms at the outliers)
        const yb = try g.adopt(try pf.moe(s, c.x_bf, c.inds1, c.sc1, c.ops, sp.rows, .bfloat16));
        const gb = try readF32(a, &g, yb);
        defer a.free(gb);
        const lib_bf16 = worstVsRef(gb, refs, &sample);
        var sushi_bf16: f64 = -1;
        var vs_sushi: f64 = -1;
        if (has_sushi) {
            const ys = try g.adopt(try sushi.moe(s, c.x3, sushiBank(&c), c.inds3, c.sc3, sushiDec(), false));
            const gs = try readF32(a, &g, ys);
            defer a.free(gs);
            sushi_bf16 = worstVsRef(gs, refs, &sample);
            vs_sushi = ref.compare(gb, gs).max_ratio;
        }
        const ok = lib_f32 <= 5e-3 and (sushi_bf16 < 0 or lib_bf16 <= 1.5 * sushi_bf16 + 2e-3);
        std.debug.print("TURBOEXL3_PF_PARITY small={s} sgt={d} fragb={} intr={} sintr={} unroll={} defer={} split={d} bm={d} bk128={} ncb={d} alay={d} h16={} rows={d} experts={d} n={d} lib_f32_vs_ref={e:.3} lib_bf16_vs_ref={e:.3} sushi_bf16_vs_ref={e:.3} lib_vs_sushi={e:.3} {s}\n", .{ @tagName(opts.small), opts.sgt, opts.fragb, opts.intr, opts.sintr, opts.unroll, opts.defer_affine, opts.split, opts.bm, opts.bk128, opts.ncb, opts.alay, opts.half_rows, sp.rows, sp.experts, sp.n, lib_f32, lib_bf16, sushi_bf16, vs_sushi, if (ok) "PASS" else "FAIL" });
        // the down-GEMM reshapes keep every output's MMA order (same 16-deep steps, k ascending, one block's accumulator at a
        // time), sintr reads the same codewords, alay feeds the same A values to the same fragment slots and half_rows drops only
        // MMAs of rows past the tile: the bytes of the option set without them
        var same = true;
        if (opts.bk128 or opts.ncb != 1 or opts.sintr or opts.froute or opts.aux != 0 or opts.alay != 0 or opts.half_rows) {
            var base = opts;
            base.bk128 = false;
            base.ncb = 1;
            base.sintr = false;
            base.froute = false;
            base.aux = 0;
            base.alay = 0;
            base.half_rows = false;
            var pb = try prefill.Prefill.init(.{ .hidden = hidden, .inter = inter, .topk = topk, .experts = sp.experts, .rate = c.rate, .dec = .mul1 }, base);
            defer pb.deinit();
            const want = try readF32(a, &g, try g.adopt(try pb.moe(s, c.x_bf, c.inds1, c.sc1, c.ops, sp.rows, .float32)));
            defer a.free(want);
            var diff: usize = 0;
            for (got, want) |p, q| diff += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
            same = diff == 0;
            std.debug.print("TURBOEXL3_PF_PARITY_BYTES bm={d} bk128={} ncb={d} sintr={} froute={} aux={d} alay={d} h16={} dbuf={} small={s} rows={d} experts={d} n={d} differing={d}/{d} {s}\n", .{ opts.bm, opts.bk128, opts.ncb, opts.sintr, opts.froute, opts.aux, opts.alay, opts.half_rows, opts.dbuf, @tagName(opts.small), sp.rows, sp.experts, sp.n, diff, got.len, if (same) "PASS" else "FAIL" });
        }
        try testing.expect(ok and same);
      }
    }
}

fn envU32(name: [*:0]const u8, default: u32) u32 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch default;
}

fn envF64(name: [*:0]const u8, default: f64) f64 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch default;
}

/// A benchmark arm: Sushi's prefill arm, or the fused chain at one option set ("b<rows>[d]": M tile rows, d = double-buffered;
/// "m<rows>": the mixed launch; the other suffixes are `Options` fields, e.g. k = bk128, n<c> = ncb, j = sintr (needs i),
/// l = alay 1 (l2 = alay 2), e = half_rows (both need f), h = t128 (the 128-row tiles beside the mixed launch, timed through
/// `Prefill.moeT128` as root.zig's `moePrefill` serves them; needs m64 and q); "r32[s<sgt>][j][q][l|l2][e]": the all-register
/// path, j = its integer-rate reader).
const Arm = union(enum) {
    sushi,
    lib: prefill.Options,

    fn parse(tok: []const u8) ?Arm {
        if (std.mem.eql(u8, tok, "sushi")) return .sushi;
        if (std.mem.startsWith(u8, tok, "r32")) {
            // the suffixes peeled from the end in the order `name` prints them
            const e = std.mem.endsWith(u8, tok, "e");
            const te = tok[0 .. tok.len - @intFromBool(e)];
            const l: u8 = if (std.mem.endsWith(u8, te, "l2")) 2 else if (std.mem.endsWith(u8, te, "l")) 1 else 0;
            const tl = te[0 .. te.len - @as(usize, if (l == 2) 2 else l)];
            const q = std.mem.endsWith(u8, tl, "q");
            const tq = tl[0 .. tl.len - @intFromBool(q)];
            const j = std.mem.endsWith(u8, tq, "j");
            const t = tq[0 .. tq.len - @intFromBool(j)];
            const sgt: u32 = if (t.len > 4 and t[3] == 's') std.fmt.parseInt(u32, t[4..], 10) catch return null else 4;
            const o: prefill.Options = .{ .small = .all, .sgt = sgt, .intr = j, .sintr = j, .froute = q, .alay = l, .half_rows = e };
            return if (o.valid()) .{ .lib = o } else null;
        }
        if (tok.len < 2 or (tok[0] != 'b' and tok[0] != 'm')) return null;
        var i: usize = 1;
        while (i < tok.len and std.ascii.isDigit(tok[i])) i += 1;
        var o: prefill.Options = .{ .bm = std.fmt.parseInt(u32, tok[1..i], 10) catch return null };
        if (tok[0] == 'm') o.small = .mixed;
        // suffixes: d = double-buffered stage, p<k> = timing probe k
        while (i < tok.len) : (i += 1) switch (tok[i]) {
            'd' => o.dbuf = true,
            'r' => o.mma_first = true,
            'u' => o.unroll = true,
            'f' => o.fragb = true,
            'i' => o.intr = true,
            'a' => o.defer_affine = true,
            'j' => o.sintr = true,
            'k' => o.bk128 = true,
            'q' => o.froute = true,
            'l' => {
                // l = the 32-block layout (16-byte loads), l2 = the 16-block one (8-byte loads)
                var j = i + 1;
                while (j < tok.len and std.ascii.isDigit(tok[j])) j += 1;
                o.alay = if (j == i + 1) 1 else std.fmt.parseInt(u8, tok[i + 1 .. j], 10) catch return null;
                i = j - 1;
            },
            'e' => o.half_rows = true,
            'h' => o.t128 = true,
            'o' => o.aux_lane_sums = true,
            'v' => {
                var j = i + 1;
                while (j < tok.len and std.ascii.isDigit(tok[j])) j += 1;
                o.aux = std.fmt.parseInt(u8, tok[i + 1 .. j], 10) catch return null;
                i = j - 1;
            },
            'n' => {
                var j = i + 1;
                while (j < tok.len and std.ascii.isDigit(tok[j])) j += 1;
                o.ncb = std.fmt.parseInt(u8, tok[i + 1 .. j], 10) catch return null;
                i = j - 1;
            },
            't' => {
                var j = i + 1;
                while (j < tok.len and std.ascii.isDigit(tok[j])) j += 1;
                o.split = std.fmt.parseInt(u8, tok[i + 1 .. j], 10) catch return null;
                i = j - 1;
            },
            'w' => {
                var j = i + 1;
                while (j < tok.len and std.ascii.isDigit(tok[j])) j += 1;
                o.wsp = std.fmt.parseInt(u8, tok[i + 1 .. j], 10) catch return null;
                i = j - 1;
            },
            'p' => {
                var j = i + 1;
                while (j < tok.len and std.ascii.isDigit(tok[j])) j += 1;
                o.probe = std.fmt.parseInt(u8, tok[i + 1 .. j], 10) catch return null;
                i = j - 1;
            },
            else => return null,
        };
        return if (o.valid()) .{ .lib = o } else null;
    }

    fn name(self: Arm, buf: []u8) []const u8 {
        return switch (self) {
            .sushi => "sushi",
            .lib => |o| blk: {
                if (o.small == .all) break :blk std.fmt.bufPrint(buf, "r32s{d}{s}{s}{s}{s}", .{ o.sgt, if (o.sintr) "j" else "", if (o.froute) "q" else "", switch (o.alay) {
                    0 => "",
                    1 => "l",
                    else => "l2",
                }, if (o.half_rows) "e" else "" }) catch "?";
                var w: usize = 0;
                w += (std.fmt.bufPrint(buf[w..], "{s}{d}", .{ if (o.small == .mixed) "m" else "b", o.bm }) catch break :blk "?").len;
                if (o.dbuf) w += (std.fmt.bufPrint(buf[w..], "d", .{}) catch break :blk "?").len;
                if (o.mma_first) w += (std.fmt.bufPrint(buf[w..], "r", .{}) catch break :blk "?").len;
                if (o.unroll) w += (std.fmt.bufPrint(buf[w..], "u", .{}) catch break :blk "?").len;
                if (o.fragb) w += (std.fmt.bufPrint(buf[w..], "f", .{}) catch break :blk "?").len;
                if (o.intr) w += (std.fmt.bufPrint(buf[w..], "i", .{}) catch break :blk "?").len;
                if (o.defer_affine) w += (std.fmt.bufPrint(buf[w..], "a", .{}) catch break :blk "?").len;
                if (o.sintr) w += (std.fmt.bufPrint(buf[w..], "j", .{}) catch break :blk "?").len;
                if (o.bk128) w += (std.fmt.bufPrint(buf[w..], "k", .{}) catch break :blk "?").len;
                if (o.ncb != 1) w += (std.fmt.bufPrint(buf[w..], "n{d}", .{o.ncb}) catch break :blk "?").len;
                if (o.froute) w += (std.fmt.bufPrint(buf[w..], "q", .{}) catch break :blk "?").len;
                if (o.alay != 0) w += (std.fmt.bufPrint(buf[w..], "{s}", .{if (o.alay == 1) "l" else "l2"}) catch break :blk "?").len;
                if (o.half_rows) w += (std.fmt.bufPrint(buf[w..], "e", .{}) catch break :blk "?").len;
                if (o.t128) w += (std.fmt.bufPrint(buf[w..], "h", .{}) catch break :blk "?").len;
                if (o.aux != 0) w += (std.fmt.bufPrint(buf[w..], "v{d}", .{o.aux}) catch break :blk "?").len;
                if (o.aux_lane_sums) w += (std.fmt.bufPrint(buf[w..], "o", .{}) catch break :blk "?").len;
                if (o.split != 1) w += (std.fmt.bufPrint(buf[w..], "t{d}", .{o.split}) catch break :blk "?").len;
                if (o.wsp != 0) w += (std.fmt.bufPrint(buf[w..], "w{d}", .{o.wsp}) catch break :blk "?").len;
                if (o.probe != 0) w += (std.fmt.bufPrint(buf[w..], "p{d}", .{o.probe}) catch break :blk "?").len;
                break :blk buf[0..w];
            },
        };
    }
};

test "turboexl3 prefill arms: the down-GEMM and register-reader suffixes parse and print back (k = bk128, n<c> = ncb, j = sintr)" {
    for ([_][]const u8{ "m64dufia", "m64dufiak", "m64dufian2", "m64dufian4", "m64dufiakn2", "m64dufiakn4", "b128fkn2", "b128k", "m64dufiaj", "m64dufiajn4", "m128dufiaj", "m64dufiajt2", "r32s4", "r32s4j", "r32s8j" }) |tok| {
        const arm = Arm.parse(tok) orelse return error.ArmRefused;
        var buf: [32]u8 = undefined;
        try testing.expectEqualStrings(tok, arm.name(&buf));
    }
    const o = Arm.parse("m64dufiakn4").?.lib;
    try testing.expect(o.bm == 64 and o.small == .mixed and o.dbuf and o.fragb and o.defer_affine and o.bk128 and o.ncb == 4 and !o.sintr);
    try testing.expect(Arm.parse("m64duiak") == null); // the padded layout's double-buffered 128-deep stage: 36 KB
    try testing.expect(Arm.parse("m64dufian3") == null);
    // j: the register path through intr's reader; the served arm without it keeps today's general reader
    const oj = Arm.parse("m64dufiajn4").?.lib;
    try testing.expect(oj.small == .mixed and oj.intr and oj.sintr and oj.ncb == 4);
    try testing.expect(!Arm.parse("m64dufia").?.lib.sintr);
    const rj = Arm.parse("r32s4j").?.lib;
    try testing.expect(rj.small == .all and rj.sgt == 4 and rj.intr and rj.sintr and !Arm.parse("r32s4").?.lib.sintr);
    try testing.expect(Arm.parse("m64dufaj") == null); // sintr needs intr
    try testing.expect(Arm.parse("b64dufiaj") == null); // no register path in the shared-decode-only launch
}

test "turboexl3 prefill arms: the A-layout and first-fragment-row suffixes parse and print back (l = alay, e = half_rows; both need f where a shared-decode body runs)" {
    for ([_][]const u8{ "m64dufiajn4ql", "m64dufiajn4ql2", "m64dufiajn4qe", "m64dufiajn4qle", "m64dufiajn4ql2e", "m128dufiale", "b128dfle", "r32s4l", "r32s4jle", "r32s4jl2e", "r32s8jqle" }) |tok| {
        const arm = Arm.parse(tok) orelse return error.ArmRefused;
        var buf: [32]u8 = undefined;
        try testing.expectEqualStrings(tok, arm.name(&buf));
    }
    const o = Arm.parse("m64dufiajn4qle").?.lib;
    try testing.expect(o.bm == 64 and o.small == .mixed and o.fragb and o.sintr and o.ncb == 4 and o.froute and o.alay == 1 and o.half_rows);
    const plain = Arm.parse("m64dufiajn4q").?.lib;
    try testing.expect(plain.alay == 0 and !plain.half_rows);
    const r = Arm.parse("r32s4jle").?.lib;
    try testing.expect(r.small == .all and r.sintr and r.alay == 1 and r.half_rows and !r.froute);
    try testing.expect(Arm.parse("m64dufiajn4ql2e").?.lib.alay == 2 and Arm.parse("r32s4jl2e").?.lib.alay == 2);
    try testing.expect(Arm.parse("m64dufiajn4ql3") == null); // two layouts
    try testing.expect(Arm.parse("m64duiajl") == null); // the padded body loads A in natural order
    try testing.expect(Arm.parse("m64duiaje") == null);
    try testing.expect(Arm.parse("m64dufiav8l") == null); // the vector-layout prep / mid write natural order
    try testing.expect(Arm.parse("m64dufiav8e") != null);
}

test "turboexl3 prefill timing arms: h (t128, the 128-row tiles) parses and prints back; the served arm is root.zig's pfOptions" {
    for ([_][]const u8{ "m64dufiajn4ql2h", "m64dufiajn4qh" }) |tok| {
        const arm = Arm.parse(tok) orelse return error.ArmRefused;
        var buf: [32]u8 = undefined;
        try testing.expectEqualStrings(tok, arm.name(&buf));
    }
    // root.zig's pfOptions at the default Config (pf_alay 2, pf_t128), as prefillFor keeps it at every rate n % 16 == 0 (bm 64
    // mixed, dbuf, unroll, fragb, intr, defer, sintr, ncb 4, froute, alay 2, t128)
    const served: prefill.Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .defer_affine = true, .ncb = 4, .sintr = true, .froute = true, .alay = 2, .t128 = true };
    try testing.expect(std.meta.eql(served, Arm.parse("m64dufiajn4ql2h").?.lib));
    try testing.expect(Arm.parse("m64dufiajn4l2h") == null); // the 128-row tiles need froute's route
    try testing.expect(Arm.parse("m128dufiajn4ql2h") == null); // and bm 64
    try testing.expect(Arm.parse("b64dfh") == null); // and the mixed launch
}

/// A library arm's call as root.zig's `moePrefill` makes it: `moeT128` with the 128-row tiles (t128), else `moe`.
fn pfCall(pf: *prefill.Prefill, s: mlx.mlx_stream, c: *const Case, rows: u32) !mlx.mlx_array {
    if (pf.opts.t128) return pf.moeT128(s, c.x_bf, c.inds1, c.sc1, c.ops, rows, .bfloat16);
    return pf.moe(s, c.x_bf, c.inds1, c.sc1, c.ops, rows, .bfloat16);
}

fn timeIt(g: *G, comptime F: type, f: F, rounds: u32, calls: u32) !f64 {
    var best: f64 = std.math.inf(f64);
    for (0..rounds + 2) |round| {
        var ys: [16]mlx.mlx_array = undefined;
        const nc = @min(calls, ys.len);
        for (ys[0..nc]) |*y| y.* = try f.call();
        const t0 = nowNs();
        try g.evalAll(ys[0..nc]);
        const dt = nowNs() - t0;
        for (ys[0..nc]) |y| _ = mlx.mlx_array_free(y);
        if (round >= 2) best = @min(best, @as(f64, @floatFromInt(dt)) / 1000.0 / @as(f64, @floatFromInt(nc)));
    }
    return best;
}

// Timing (TURBOEXL3_GPU=1 TURBOEXL3_PF_TIME=1): every arm on the same bank, routing and inputs, rounds interleaved with a rotated
// start; min and median us per call. TURBOEXL3_PF_ARMS (default "b128,sushi"), TURBOEXL3_PF_E (128), TURBOEXL3_PF_N (80),
// TURBOEXL3_PF_ROWS (default 4096,1359: production's 4096-token chunk and the standard cell's tail), TURBOEXL3_PF_SKEW (lognormal
// sigma of expert popularity, 0 = uniform), TURBOEXL3_PF_STAGES=1 adds each library stage timed alone. TURBOEXL3_PF_INDS (comma
// list of u32 [rows, 10] routing files, `Config.pf_capture_dir`'s output) replays served routing instead of
// TURBOEXL3_PF_ROWS: one case per file, rows from the file's size, experts must cover its largest index. An arm with the
// 128-row tiles (h) runs `Prefill.moeT128` (`pfCall`), as root.zig's `moePrefill` serves it; TURBOEXL3_PF_STAGES skips it.
test "turboexl3 gpu prefill timing: the fused chain vs Sushi's prefill arm (TURBOEXL3_PF_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_PF_TIME") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const experts = envU32("TURBOEXL3_PF_E", 128);
    const n = envU32("TURBOEXL3_PF_N", 80);
    const skew = envF64("TURBOEXL3_PF_SKEW", 0);
    const rounds = envU32("TURBOEXL3_PF_ROUNDS", 9);
    const calls = envU32("TURBOEXL3_PF_CALLS", 4);
    const stages = std.c.getenv("TURBOEXL3_PF_STAGES") != null;
    var arms_buf: [8]Arm = undefined;
    var narms: usize = 0;
    {
        const raw = if (std.c.getenv("TURBOEXL3_PF_ARMS")) |v| std.mem.span(v) else "b128,sushi";
        var it = std.mem.splitScalar(u8, raw, ',');
        while (it.next()) |tok| {
            const arm = Arm.parse(tok) orelse continue;
            if (arm == .sushi and !has_sushi) continue;
            if (narms == arms_buf.len) break;
            arms_buf[narms] = arm;
            narms += 1;
        }
    }
    const arms = arms_buf[0..narms];
    var rows_list: [16]u32 = undefined;
    var real_list: [16]?[]u32 = @splat(null);
    defer for (real_list) |r| if (r) |rs| a.free(rs);
    var name_list: [16][]const u8 = @splat("");
    var nrows: usize = 0;
    if (std.c.getenv("TURBOEXL3_PF_INDS")) |v| {
        var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
        while (it.next()) |path| {
            if (nrows == rows_list.len) break;
            const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(64 << 20));
            defer a.free(bytes);
            if (bytes.len == 0 or bytes.len % (4 * topk) != 0) return error.BadIndsFile;
            const rs = try a.alloc(u32, bytes.len / 4);
            @memcpy(std.mem.sliceAsBytes(rs), bytes);
            for (rs) |e| if (e >= experts) return error.IndsBeyondExperts;
            real_list[nrows] = rs;
            rows_list[nrows] = @intCast(rs.len / topk);
            name_list[nrows] = std.fs.path.basename(path);
            nrows += 1;
        }
    } else {
        const raw = if (std.c.getenv("TURBOEXL3_PF_ROWS")) |v| std.mem.span(v) else "4096,1359";
        var it = std.mem.splitScalar(u8, raw, ',');
        while (it.next()) |tok| {
            if (nrows == rows_list.len) break;
            rows_list[nrows] = std.fmt.parseInt(u32, tok, 10) catch continue;
            nrows += 1;
        }
    }
    var prng = std.Random.DefaultPrng.init(0xBEEF ^ experts);
    const rnd = prng.random();
    std.debug.print("TURBOEXL3_PF_START experts={d} n={d} skew={d:.2} rounds={d} calls={d} arms={d}\n", .{ experts, n, skew, rounds, calls, narms });
    for (rows_list[0..nrows], real_list[0..nrows], name_list[0..nrows]) |rows, real, src| {
        const mark = g.mark();
        defer g.resetTo(mark);
        var c = try makeCase(a, &g, rnd, rows, experts, n, skew, real);
        if (real) |rs| {
            // the replayed routing's shape: rows per expert (max, and how many experts land in each tile-size class)
            var cnt: [1024]u32 = @splat(0);
            for (rs) |e| cnt[e] += 1;
            var mx: u32 = 0;
            var cls: [5]u32 = @splat(0); // 0, 1-32, 33-64, 65-128, > 128
            for (cnt[0..experts]) |k| {
                mx = @max(mx, k);
                cls[if (k == 0) 0 else if (k <= 32) 1 else if (k <= 64) 2 else if (k <= 128) 3 else 4] += 1;
            }
            std.debug.print("TURBOEXL3_PF_REAL src={s} rows={d} max_rows_per_expert={d} experts_0={d} 1-32={d} 33-64={d} 65-128={d} gt128={d}\n", .{ src, rows, mx, cls[0], cls[1], cls[2], cls[3], cls[4] });
        }
        defer c.deinit(a);
        const geo: prefill.Geometry = .{ .hidden = hidden, .inter = inter, .topk = topk, .experts = experts, .rate = c.rate, .dec = .mul1 };
        var pfs: [8]?prefill.Prefill = @splat(null);
        defer for (&pfs) |*p| if (p.*) |*pp| pp.deinit();
        for (arms, 0..) |arm, i| switch (arm) {
            .lib => |o| pfs[i] = try prefill.Prefill.init(geo, o),
            .sushi => {},
        };
        // every library arm must give the first library arm's bytes (the options move work, not arithmetic)
        var first: ?[]f32 = null;
        defer if (first) |f| a.free(f);
        // each engine's first call, timed alone: in the process's first case it includes compiling that engine's kernels (what
        // a server's first prefill pays), afterwards it is one plain call
        for (arms, 0..) |arm, i| if (arm == .sushi) {
            const t0 = nowNs();
            var ys = [_]mlx.mlx_array{try sushi.moe(s, c.x3, sushiBank(&c), c.inds3, c.sc3, sushiDec(), false)};
            try g.evalAll(&ys);
            std.debug.print("TURBOEXL3_PF_FIRSTCALL rows={d} arm={d} engine=sushi us={d:.0}\n", .{ rows, i, @as(f64, @floatFromInt(nowNs() - t0)) / 1000.0 });
            _ = mlx.mlx_array_free(ys[0]);
        };
        for (arms, 0..) |arm, i| if (arm == .lib) {
            const t0 = nowNs();
            const y = try g.adopt(try pfCall(&pfs[i].?, s, &c, rows));
            var ys = [_]mlx.mlx_array{y};
            try g.evalAll(&ys);
            std.debug.print("TURBOEXL3_PF_FIRSTCALL rows={d} arm={d} engine=turboexl3 us={d:.0}\n", .{ rows, i, @as(f64, @floatFromInt(nowNs() - t0)) / 1000.0 });
            const got = try readF32(a, &g, y);
            if (first) |f| {
                var diff: usize = 0;
                for (got, f) |p, q| diff += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
                std.debug.print("TURBOEXL3_PF_ARMBYTES rows={d} arm={d} differing={d}/{d}\n", .{ rows, i, diff, got.len });
                a.free(got);
            } else first = got;
        };
        var samples: [8][64]f64 = undefined;
        for (0..rounds + 2) |round| {
            for (0..narms) |ai| {
                const i = (ai + round) % narms;
                var ys: [16]mlx.mlx_array = undefined;
                const nc = @min(calls, ys.len);
                for (ys[0..nc]) |*y| {
                    y.* = switch (arms[i]) {
                        .lib => try pfCall(&pfs[i].?, s, &c, rows),
                        .sushi => try sushi.moe(s, c.x3, sushiBank(&c), c.inds3, c.sc3, sushiDec(), false),
                    };
                }
                const t0 = nowNs();
                try g.evalAll(ys[0..nc]);
                const dt = nowNs() - t0;
                for (ys[0..nc]) |y| _ = mlx.mlx_array_free(y);
                if (round >= 2) samples[i][round - 2] = @as(f64, @floatFromInt(dt)) / 1000.0 / @as(f64, @floatFromInt(nc));
            }
        }
        const flops = 2.0 * @as(f64, @floatFromInt(rows * topk)) * 3.0 * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter));
        var sushi_min: f64 = 0;
        var mins: [8]f64 = undefined;
        var meds: [8]f64 = undefined;
        for (0..narms) |i| {
            const sm = samples[i][0..rounds];
            std.mem.sort(f64, sm, {}, std.sort.asc(f64));
            mins[i] = sm[0];
            meds[i] = sm[rounds / 2];
            if (arms[i] == .sushi) sushi_min = mins[i];
        }
        for (0..narms) |i| {
            var name_buf: [32]u8 = undefined;
            const name = arms[i].name(&name_buf);
            std.debug.print("TURBOEXL3_PF_TIME src={s} rows={d} experts={d} skew={d:.2} rows_per_expert={d:.1} arm={s} us_min={d:.0} us_med={d:.0} tflops={d:.2} vs_sushi_min={d:.3}x\n", .{
                if (real != null) src else "synthetic", rows, experts, skew, @as(f64, @floatFromInt(rows * topk)) / @as(f64, @floatFromInt(experts)), name, mins[i], meds[i], flops / (mins[i] * 1e6), if (sushi_min > 0) sushi_min / mins[i] else 0,
            });
        }
        if (!stages) continue;
        {
            // the same FLOPs as dense MLX GEMMs (one weight shared by every row): gate|up [S, hidden] x [hidden, 2 inter], down
            // [S, inter] x [inter, hidden], f16: what this GPU's own GEMM does at these shapes
            const S: c_int = @intCast(rows * topk);
            var ka = mlx.mlx_array_new();
            var kb = mlx.mlx_array_new();
            var kc = mlx.mlx_array_new();
            var kd = mlx.mlx_array_new();
            const nokey = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&ka, &[_]c_int{ S, ci(hidden) }, 2, .float16, 0, 1, nokey, s));
            try mlx.check(mlx.mlx_random_normal(&kb, &[_]c_int{ ci(hidden), ci(2 * inter) }, 2, .float16, 0, 0.02, nokey, s));
            try mlx.check(mlx.mlx_random_normal(&kc, &[_]c_int{ S, ci(inter) }, 2, .float16, 0, 1, nokey, s));
            try mlx.check(mlx.mlx_random_normal(&kd, &[_]c_int{ ci(inter), ci(hidden) }, 2, .float16, 0, 0.02, nokey, s));
            ka = try g.adopt(ka);
            kb = try g.adopt(kb);
            kc = try g.adopt(kc);
            kd = try g.adopt(kd);
            try g.evalAll(&.{ ka, kb, kc, kd });
            const Mm = struct {
                x: mlx.mlx_array,
                w: mlx.mlx_array,
                s: mlx.mlx_stream,
                fn call(m: @This()) !mlx.mlx_array {
                    var o = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_matmul(&o, m.x, m.w, m.s));
                    return o;
                }
            };
            const t_gu = try timeIt(&g, Mm, .{ .x = ka, .w = kb, .s = s }, rounds, calls);
            const t_dn = try timeIt(&g, Mm, .{ .x = kc, .w = kd, .s = s }, rounds, calls);
            const f_gu = 2.0 * @as(f64, @floatFromInt(S)) * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(2 * inter));
            const f_dn = 2.0 * @as(f64, @floatFromInt(S)) * @as(f64, @floatFromInt(inter)) * @as(f64, @floatFromInt(hidden));
            std.debug.print("TURBOEXL3_PF_REF rows={d} mlx_dense_gemm_gu_us={d:.0} ({d:.2} TFLOPS) mlx_dense_gemm_down_us={d:.0} ({d:.2} TFLOPS)\n", .{ rows, t_gu, f_gu / (t_gu * 1e6), t_dn, f_dn / (t_dn * 1e6) });
        }
        // the mixed chain's stages (a t128 arm's moeT128 stages are not these: skipped)
        for (arms, 0..) |arm, i| if (arm == .lib and !arm.lib.t128) {
            const pf = &pfs[i].?;
            const S: u32 = rows * topk;
            const r = try pf.route(s, c.inds1, S);
            defer r.deinit();
            const po = try pf.prep(s, c.x_bf, c.inds1, r.order, c.ops, S);
            defer for (po) |x| {
                _ = mlx.mlx_array_free(x);
            };
            const zgu = try pf.gemmGU(s, r, po[0], po[1], po[3], po[4], c.ops, S);
            defer for (zgu) |x| {
                _ = mlx.mlx_array_free(x);
            };
            const xd = try pf.mid(s, zgu[0], zgu[1], c.inds1, r.order, c.ops, S);
            defer for (xd) |x| {
                _ = mlx.mlx_array_free(x);
            };
            const zd = try pf.gemmD(s, r, xd[0], xd[1], c.ops, S);
            defer _ = mlx.mlx_array_free(zd);
            try g.evalAll(&.{ r.order, r.tiles, po[0], po[1], po[2], po[3], po[4], zgu[0], zgu[1], xd[0], xd[1], zd });
            const Ctx = struct {
                pf: *prefill.Prefill,
                s: mlx.mlx_stream,
                c: *const Case,
                r: prefill.Route,
                po: [5]mlx.mlx_array,
                zgu: [2]mlx.mlx_array,
                xd: [2]mlx.mlx_array,
                zd: mlx.mlx_array,
                rows: u32,
                which: u8,
                fn call(x: @This()) !mlx.mlx_array {
                    const S2: u32 = x.rows * topk;
                    return switch (x.which) {
                        0 => blk: {
                            const rr = try x.pf.route(x.s, x.c.inds1, S2);
                            _ = mlx.mlx_array_free(rr.order);
                            break :blk rr.tiles;
                        },
                        1 => blk: {
                            const p = try x.pf.prep(x.s, x.c.x_bf, x.c.inds1, x.r.order, x.c.ops, S2);
                            for (p[1..]) |extra| _ = mlx.mlx_array_free(extra);
                            break :blk p[0];
                        },
                        2 => blk: {
                            const z = try x.pf.gemmGU(x.s, x.r, x.po[0], x.po[1], x.po[3], x.po[4], x.c.ops, S2);
                            _ = mlx.mlx_array_free(z[1]);
                            break :blk z[0];
                        },
                        3 => blk: {
                            const m = try x.pf.mid(x.s, x.zgu[0], x.zgu[1], x.c.inds1, x.r.order, x.c.ops, S2);
                            _ = mlx.mlx_array_free(m[1]);
                            break :blk m[0];
                        },
                        4 => try x.pf.gemmD(x.s, x.r, x.xd[0], x.xd[1], x.c.ops, S2),
                        else => try x.pf.finish(x.s, x.zd, x.c.inds1, x.po[2], x.c.sc1, x.c.ops, x.rows, .bfloat16),
                    };
                }
            };
            const names = [_][]const u8{ "route", "prep", "gemm_gu", "mid", "gemm_down", "finish" };
            var t: [6]f64 = undefined;
            var total: f64 = 0;
            for (0..6) |w| {
                t[w] = try timeIt(&g, Ctx, .{ .pf = pf, .s = s, .c = &c, .r = r, .po = po, .zgu = zgu, .xd = xd, .zd = zd, .rows = rows, .which = @intCast(w) }, rounds, calls);
                total += t[w];
            }
            for (0..6) |w| {
                var nb: [32]u8 = undefined;
                std.debug.print("TURBOEXL3_PF_STAGE rows={d} arm={s} stage={s} us_min={d:.0} share={d:.1}%\n", .{ rows, arm.name(&nb), names[w], t[w], 100.0 * t[w] / total });
            }
        };
    }
}

// Decode timing (TURBOEXL3_GPU=1 TURBOEXL3_DEC_TIME=1): turboexl3's decode chain (`root.moe` at <= 16 rows) against Sushi's decode
// arm on the same 512-expert bank, inputs and routing, as the served decode runs it, at each trellis rate of TURBOEXL3_DEC_NS
// (default 80 = K5; e.g. 32,48,64,80,96,112,128 = K2..K8; one bank per rate): a chain of TURBOEXL3_DEC_LAYERS (48) MoE calls
// where each call's output is the next call's input (serialized like layers) and each call routes to fresh experts (a layer's
// experts come from DRAM, not the cache). Rows TURBOEXL3_DEC_ROWS (default 1,2,3,4: the MTP verify widths); routing synthetic
// (each row 10 distinct uniform experts), then with TURBOEXL3_DEC_INDS (a captured u32 [rows, 10] prefill call: call k reads
// consecutive rows of it, i.e. consecutive tokens' experts) captured. Arms, interleaved with a rotated start: turboexl3 forms 3,
// 7 and 9 (root.zig's served form) and Sushi. Min / median us per call over TURBOEXL3_DEC_ROUNDS (at most 62) rounds, and
// TURBOEXL3_DEC_SAMPLES lines with every round's us per call per arm (for paired stats).
test "turboexl3 gpu decode timing: the decode chain vs Sushi's decode arm (TURBOEXL3_DEC_TIME=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("TURBOEXL3_DEC_TIME") orelse return error.SkipZigTest;
    if (!has_sushi) return error.SkipZigTest;
    const root = @import("root.zig");
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const experts = envU32("TURBOEXL3_DEC_E", 512);
    const layers = envU32("TURBOEXL3_DEC_LAYERS", 48);
    const rounds = envU32("TURBOEXL3_DEC_ROUNDS", 9);
    if (rounds > 62) return error.TooManyRounds;
    var rows_list: [8]u32 = undefined;
    var nrows: usize = 0;
    {
        const raw = if (std.c.getenv("TURBOEXL3_DEC_ROWS")) |v| std.mem.span(v) else "1,2,3,4";
        var it = std.mem.splitScalar(u8, raw, ',');
        while (it.next()) |tok| {
            if (nrows == rows_list.len) break;
            rows_list[nrows] = std.fmt.parseInt(u32, tok, 10) catch continue;
            nrows += 1;
        }
    }
    var n_list: [8]u32 = undefined;
    var nn: usize = 0;
    {
        const raw = if (std.c.getenv("TURBOEXL3_DEC_NS")) |v| std.mem.span(v) else "80";
        var it = std.mem.splitScalar(u8, raw, ',');
        while (it.next()) |tok| {
            if (nn == n_list.len) break;
            const v = std.fmt.parseInt(u32, tok, 10) catch continue;
            if (format.Rate.fromN(v) == null) continue;
            n_list[nn] = v;
            nn += 1;
        }
    }
    var real: ?[]u32 = null;
    defer if (real) |r| a.free(r);
    // (the loop below swaps `real` between null and the capture; it is the capture again at the end, so the defer frees it)
    if (std.c.getenv("TURBOEXL3_DEC_INDS")) |v| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, std.mem.span(v), a, .limited(64 << 20));
        defer a.free(bytes);
        const rs = try a.alloc(u32, bytes.len / 4);
        @memcpy(std.mem.sliceAsBytes(rs), bytes[0 .. rs.len * 4]);
        real = rs;
    }
    var prng = std.Random.DefaultPrng.init(0xDEC0DE ^ experts);
    const rnd = prng.random();
    std.debug.print("TURBOEXL3_DEC_START experts={d} layers={d} rounds={d} captured={} rates={d}\n", .{ experts, layers, rounds, real != null, nn });
    const captured_all = real;
    for (n_list[0..nn]) |n| {
        const bank_mark = g.mark();
        defer g.resetTo(bank_mark);
        var c = try makeCase(a, &g, rnd, 16, experts, n, 0, null);
        defer c.deinit(a);
        const rbank: root.Bank = .{
            .gate = .{ .trellis = c.ops.gate_t, .suh = c.ops.gate_suh, .svh = c.ops.gate_svh },
            .up = .{ .trellis = c.ops.up_t, .suh = c.ops.up_suh, .svh = c.ops.up_svh },
            .down = .{ .trellis = c.ops.down_t, .suh = c.ops.down_suh, .svh = c.ops.down_svh },
        };
        for (0..@as(usize, if (captured_all != null) 2 else 1)) |src_i| for (rows_list[0..nrows]) |rows| {
            real = if (src_i == 1) captured_all else null;
            const src_name: []const u8 = if (src_i == 1) "captured" else "synthetic";
            const mark = g.mark();
            defer g.resetTo(mark);
            // per layer: rows x topk slots (u32) and bf16 scores
            const S = rows * topk;
            var inds: [64]mlx.mlx_array = undefined;
            var scs: [64]mlx.mlx_array = undefined;
            const L = @min(layers, inds.len);
            const slots = try a.alloc(u32, S);
            defer a.free(slots);
            const sc = try a.alloc(f32, S);
            defer a.free(sc);
            for (0..L) |k| {
                for (0..rows) |r| {
                    if (real) |rs| {
                        const nrow = rs.len / topk;
                        const base = (k * 97 + r) % nrow;
                        for (0..topk) |j| slots[r * topk + j] = rs[base * topk + j] % experts;
                    } else {
                        var j: usize = 0;
                        while (j < topk) {
                            const e: u32 = rnd.uintLessThan(u32, experts);
                            var dup = false;
                            for (slots[r * topk ..][0..j]) |q| dup = dup or q == e;
                            if (dup) continue;
                            slots[r * topk + j] = e;
                            j += 1;
                        }
                    }
                    var sum: f32 = 0;
                    for (sc[r * topk ..][0..topk]) |*w| {
                        w.* = 0.05 + rnd.float(f32);
                        sum += w.*;
                    }
                    for (sc[r * topk ..][0..topk]) |*w| w.* /= sum;
                }
                inds[k] = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ 1, ci(rows), ci(topk) }, .uint32);
                scs[k] = try g.astype(try g.hostArray(std.mem.sliceAsBytes(sc), &.{ 1, ci(rows), ci(topk) }, .float32), .bfloat16);
            }
            const x0 = try g.slice(c.x3, &.{ 0, 0, 0 }, &.{ 1, ci(rows), ci(hidden) }, &.{ 1, 1, 1 });
            try g.evalAll(inds[0..L]);
            try g.evalAll(scs[0..L]);
            try g.evalAll(&.{x0});
            // one layer, every engine: the decode chains agree to the bf16 tolerance (turboexl3 and Sushi round in different
            // orders); turboexl3's forms 3, 7 and 9 agree byte for byte
            defer root.decode_form_override = null;
            {
                root.decode_form_override = 3;
                const yl = try g.adopt(try root.moe(s, x0, rbank, inds[0], scs[0], .mul1, false));
                root.decode_form_override = 7;
                const y7 = try g.adopt(try root.moe(s, x0, rbank, inds[0], scs[0], .mul1, false));
                root.decode_form_override = 9;
                const y9 = try g.adopt(try root.moe(s, x0, rbank, inds[0], scs[0], .mul1, false));
                const ys = try g.adopt(try sushi.moe(s, x0, sushiBank(&c), inds[0], scs[0], sushiDec(), false));
                const gl = try readF32(a, &g, yl);
                defer a.free(gl);
                const g7 = try readF32(a, &g, y7);
                defer a.free(g7);
                const g9 = try readF32(a, &g, y9);
                defer a.free(g9);
                const gs = try readF32(a, &g, ys);
                defer a.free(gs);
                var diff7: usize = 0;
                var diff9: usize = 0;
                for (g7, g9, gl) |p7, p9, q| {
                    if (@as(u32, @bitCast(p7)) != @as(u32, @bitCast(q))) diff7 += 1;
                    if (@as(u32, @bitCast(p9)) != @as(u32, @bitCast(q))) diff9 += 1;
                }
                std.debug.print("TURBOEXL3_DEC_PARITY src={s} n={d} k={d} rows={d} lib_vs_sushi_ratio={e:.3} form7_vs_form3_differing={d}/{d} form9_vs_form3_differing={d}/{d}\n", .{ src_name, n, n / 16, rows, ref.compare(gl, gs).max_ratio, diff7, gl.len, diff9, gl.len });
            }
            var samples: [4][64]f64 = undefined;
            for (0..rounds + 2) |round| {
                for (0..4) |ai| {
                    const arm = (ai + round) % 4; // 0 turboexl3 form 3, 1 Sushi, 2 turboexl3 form 7, 3 turboexl3 form 9
                    root.decode_form_override = if (arm == 2) 7 else if (arm == 3) 9 else 3;
                    var x = x0;
                    var held: [64]mlx.mlx_array = undefined;
                    for (0..L) |k| {
                        const y = if (arm != 1) try root.moe(s, x, rbank, inds[k], scs[k], .mul1, false) else try sushi.moe(s, x, sushiBank(&c), inds[k], scs[k], sushiDec(), false);
                        held[k] = y;
                        x = y;
                    }
                    const t0 = nowNs();
                    try g.evalAll(held[L - 1 .. L]);
                    const dt = nowNs() - t0;
                    for (held[0..L]) |y| _ = mlx.mlx_array_free(y);
                    if (round >= 2) samples[arm][round - 2] = @as(f64, @floatFromInt(dt)) / 1000.0 / @as(f64, @floatFromInt(L));
                }
            }
            // every round's us per call per arm, in round order (rounds interleave the arms, so a round is a paired sample);
            // printed before the sort below reorders them
            std.debug.print("TURBOEXL3_DEC_SAMPLES src={s} n={d} rows={d}", .{ src_name, n, rows });
            for ([_][]const u8{ "f3", "sushi", "f7", "f9" }, 0..) |an, arm| {
                std.debug.print(" {s}=", .{an});
                for (samples[arm][0..rounds], 0..) |v, i| std.debug.print("{s}{d:.2}", .{ if (i == 0) "" else ",", v });
            }
            std.debug.print("\n", .{});
            var mins: [4]f64 = undefined;
            var meds: [4]f64 = undefined;
            for (0..4) |arm| {
                const sm = samples[arm][0..rounds];
                std.mem.sort(f64, sm, {}, std.sort.asc(f64));
                mins[arm] = sm[0];
                meds[arm] = sm[rounds / 2];
            }
            const bytes_per_expert: f64 = 3.0 * @as(f64, @floatFromInt(hidden * inter)) * @as(f64, @floatFromInt(n)) / 16.0 / 8.0;
            std.debug.print("TURBOEXL3_DEC_TIME src={s} n={d} rows={d} experts={d} lib_us_min={d:.1} lib_us_med={d:.1} sushi_us_min={d:.1} sushi_us_med={d:.1} lib_vs_sushi={d:.3}x lib_expert_gbps={d:.0} form7_us_min={d:.1} form7_us_med={d:.1} form7_vs_sushi={d:.3}x form7_vs_form3={d:.3}x form9_us_min={d:.1} form9_us_med={d:.1} form9_vs_sushi={d:.3}x form9_vs_form3={d:.3}x\n", .{
                src_name,                                                                            n,       rows,    experts, mins[0], meds[0], mins[1], meds[1], mins[1] / mins[0],
                @as(f64, @floatFromInt(S)) * bytes_per_expert / (mins[0] * 1000.0), mins[2], meds[2], mins[1] / mins[2], mins[0] / mins[2],
                mins[3],                                                                meds[3], mins[1] / mins[3], mins[0] / mins[3],
            });
        };
    }
    real = captured_all;
}

// ── The 128-row tiles (`prefill.Options.t128`, `Prefill.moeT128`: root.zig's served prefill path) ──
// Per routing case and base option set, the mixed chain (`moe`, then stage by stage) and the 128-row tiles' chain
// (`moeT128`, then stage by stage) on the same inputs. Each chain has its own route (froute's atomic scatter orders an
// expert's slots by arrival), so the intermediates compare in slot order: zg / zu, xd / xsd, zd, the finish's y (bf16, f32),
// then the whole calls. bm 128 and bm 64 run the same per-simdgroup K loop, so every comparison is bit for bit. The route's
// tables are checked against `prefill.t128Tables` (the CPU rule) on the case's per-expert counts.

/// Elements of `x` and `y` (one shape and dtype) whose bit patterns differ, counted on the GPU (unsigned views, so NaNs and
/// signed zeros compare by their bits).
fn bitDiff(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, y: mlx.mlx_array) !usize {
    if (!std.mem.eql(c_int, mlx.getShape(x), mlx.getShape(y)) or mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(y)) return error.ShapeMismatch;
    const word: mlx.mlx_dtype = switch (mlx.mlx_array_itemsize(x)) {
        2 => .uint16,
        4 => .uint32,
        else => return error.UnsupportedDtype,
    };
    var xv = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_view(&xv, x, word, s));
    xv = try g.adopt(xv);
    var yv = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_view(&yv, y, word, s));
    yv = try g.adopt(yv);
    var eq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_equal(&eq, xv, yv, s));
    eq = try g.adopt(eq);
    const eqi = try g.astype(eq, .int32);
    var same_a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum(&same_a, eqi, false, s));
    same_a = try g.adopt(same_a);
    try g.evalAll(&.{same_a});
    var same: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&same, same_a));
    return mlx.mlx_array_size(x) - @as(usize, @intCast(same));
}

/// A routing whose experts hold exactly `counts` slots (sum rows * topk, each <= rows): row r takes the topk experts with the
/// most slots left (ties to the lower id), which keeps a row's experts distinct and never strands a count; then each row's
/// slot order and the row order are shuffled (slot order is a routing's score order, and served rows interleave experts).
fn countedRouting(a: std.mem.Allocator, rnd: std.Random, counts: []const u32, rows: u32) ![]u32 {
    var total: u64 = 0;
    for (counts) |c| {
        if (c > rows) return error.CountPastRows;
        total += c;
    }
    if (total != @as(u64, rows) * topk) return error.CountSum;
    const left = try a.dupe(u32, counts);
    defer a.free(left);
    const by_row = try a.alloc(u32, rows * topk);
    defer a.free(by_row);
    for (0..rows) |r| {
        const row = by_row[r * topk ..][0..topk];
        for (0..topk) |k| {
            var best: ?usize = null;
            for (left, 0..) |c, e| {
                if (c == 0) continue;
                var taken = false;
                for (row[0..k]) |t| taken = taken or t == e;
                if (taken) continue;
                if (best == null or c > left[best.?]) best = e;
            }
            const e = best orelse return error.CountInfeasible;
            row[k] = @intCast(e);
            left[e] -= 1;
        }
        var m: usize = topk;
        while (m > 1) : (m -= 1) std.mem.swap(u32, &row[m - 1], &row[rnd.uintLessThan(u32, @intCast(m))]);
    }
    const perm = try a.alloc(u32, rows);
    defer a.free(perm);
    for (perm, 0..) |*p, pi| p.* = @intCast(pi);
    var m: usize = rows;
    while (m > 1) : (m -= 1) std.mem.swap(u32, &perm[m - 1], &perm[rnd.uintLessThan(u32, @intCast(m))]);
    const out = try a.alloc(u32, rows * topk);
    for (perm, 0..) |src, r| @memcpy(out[r * topk ..][0..topk], by_row[src * topk ..][0..topk]);
    return out;
}

/// One GEMM op's rows of a `prefill.ZT128` as one array [S, N] in sorted order, each from the launch that computed it (the
/// 128-row tiles' output before the expert's xbase position, the mixed launch's from it), with MLX ops: the tests' view.
fn t128Merged(g: *G, zt: prefill.ZT128, op: usize, rt: prefill.RouteT128, inds: mlx.mlx_array, S: u32) !mlx.mlx_array {
    const e_sorted = try g.take(inds, rt.order, 0);
    const first = try g.take(rt.xbase, e_sorted, 0);
    const pos = try g.arange(0, @floatFromInt(S), 1, .int32);
    const in_t128 = try g.reshape(try g.less(pos, first), &.{ @intCast(S), 1 });
    return g.where(in_t128, zt.t[op], zt.m[op]);
}

/// The route kernel's tables against `prefill.t128Tables` on the case's per-expert counts: tiles128's rows in order then
/// zero rows; the mixed table's tiles from 0 and its register-path tiles from TBIG (`t128MixedRows`), each then zero rows;
/// both tables at the sizes `t128TileRows` / `t128MixedRows` give (the kernel's TQ / TBIG must agree with them); xbase: each
/// expert's first sorted position plus its 128-row tiles' rows.
fn t128TablesMatch(a: std.mem.Allocator, g: *G, rt: prefill.RouteT128, counts: []const u32, o: prefill.Options) !bool {
    var tb = try prefill.t128Tables(a, counts, o.tileRows(), prefill.t128_cut, true);
    defer tb.deinit(a);
    const q = try readF32(a, g, rt.tiles128);
    defer a.free(q);
    const m = try readF32(a, g, rt.tiles);
    defer a.free(m);
    var S: u32 = 0;
    for (counts) |cn| S += cn;
    const E: u32 = @intCast(counts.len);
    const tbig: usize = prefill.t128MixedRows(S, E, o.tileRows(), o.split, prefill.t128_cut, true);
    if (rt.tmax != tbig or m.len != 4 * (tbig + o.split * E)) return false;
    if (rt.tq != prefill.t128TileRows(S, E, prefill.t128_cut, true) or q.len != 4 * @as(usize, rt.tq)) return false;
    const rows = struct {
        fn f(got: []const f32, want: []const [3]u32, from: usize, to: usize) bool {
            if (to * 4 > got.len or want.len > to - from) return false;
            for (from..to) |i| {
                const w: [3]u32 = if (i - from < want.len) want[i - from] else .{ 0, 0, 0 };
                for (0..3) |j| if (got[4 * i + j] != @as(f32, @floatFromInt(w[j]))) return false;
                if (got[4 * i + 3] != 0) return false;
            }
            return true;
        }
    }.f;
    if (!(rows(q, tb.t128.items, 0, q.len / 4) and rows(m, tb.big.items, 0, tbig) and rows(m, tb.small.items, tbig, m.len / 4))) return false;
    const xb = try readF32(a, g, rt.xbase);
    defer a.free(xb);
    if (xb.len != counts.len) return false;
    const want = try a.alloc(u32, counts.len);
    defer a.free(want);
    var lo: u32 = 0;
    for (counts, 0..) |cn, e| {
        want[e] = lo;
        lo += cn;
    }
    for (tb.t128.items) |t| want[t[0]] += t[2];
    for (xb, want) |v, w| if (v != @as(f32, @floatFromInt(w))) return false;
    return true;
}

/// The base option sets of a case (each run as the mixed chain and with the 128-row tiles beside it): the served launch (bm 64,
/// mixed, A layout 2, sintr, froute, ncb 4), its natural A layout, no deferred map, one down column block, half_rows.
/// `served_only`: the served launch.
fn t128BaseSets(n: u32, served_only: bool, buf: *[5]prefill.Options) []const prefill.Options {
    const ir = n % 16 == 0;
    const served: prefill.Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = ir, .defer_affine = true, .ncb = 4, .sintr = ir, .froute = true, .alay = 2 };
    buf[0] = served;
    if (served_only) return buf[0..1];
    var k: usize = 1;
    for (0..4) |v| {
        var o = served;
        switch (v) {
            0 => o.alay = 0,
            1 => o.defer_affine = false,
            2 => o.ncb = 1,
            else => o.half_rows = true,
        }
        buf[k] = o;
        k += 1;
    }
    return buf[0..k];
}

test "turboexl3 gpu prefill 128-row tiles: the call and every stage give the mixed chain's bytes; the route's tables are the CPU rule's (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    // the counted case's experts: every count 1..32 (register-path tiles), then counts on both sides of each 32-, 64- and
    // 128-row edge and of the cut (64 / 65 past a 128-row tile: 193, 192), two empty experts; 640 rows
    var counts: [64]u32 = undefined;
    for (0..32) |i| counts[i] = @intCast(i + 1);
    const more = [_]u32{ 33, 63, 64, 65, 96, 97, 127, 128, 129, 160, 191, 192, 193, 255, 256, 257, 300, 320, 383, 384, 385, 511, 512, 513, 0, 0, 1, 31, 32, 33, 64, 97 };
    @memcpy(counts[32..], &more);
    const Spec = struct { name: []const u8, rows: u32, experts: u32, n: u32, skew: f64 = 0, counted: bool = false, served_only: bool = false };
    var specs: std.ArrayList(Spec) = .empty;
    defer specs.deinit(a);
    // 10 experts: every row takes all of them, so each expert holds exactly `rows` slots: every 128-row tile shape, both
    // sides of the cut and of every tile edge
    for ([_]u32{ 1, 2, 7, 16, 17, 31, 32, 33, 47, 63, 64, 65, 96, 97, 127, 128, 129, 160, 191, 192, 193, 255, 256, 257, 300, 383, 384, 385 }) |r| {
        try specs.append(a, .{ .name = "all10", .rows = r, .experts = 10, .n = 80 });
    }
    try specs.append(a, .{ .name = "counted64", .rows = 640, .experts = 64, .n = 80, .counted = true });
    // a 48-codeword rate (K3) on skewed routing
    try specs.append(a, .{ .name = "k3skew", .rows = 400, .experts = 24, .n = 48, .skew = 0.5 });
    // the served bank's 512 experts, skewed (lognormal sigma 1.5) and uniform, at widths of every kind: the first prefill
    // widths, a ~1,328-row tail, half and whole chunks
    for ([_]u32{ 17, 31, 129, 1328, 1359, 2048, 4096, 8192 }) |r| try specs.append(a, .{ .name = "skew512", .rows = r, .experts = 512, .n = 80, .skew = 1.5, .served_only = true });
    try specs.append(a, .{ .name = "uni512", .rows = 4096, .experts = 512, .n = 80, .served_only = true });
    var failed = false;
    var checks: usize = 0;
    for (specs.items, 0..) |sp, ci_| {
        var prng = std.Random.DefaultPrng.init(0xE96D ^ sp.rows ^ (sp.experts << 16) ^ (@as(u64, ci_) << 32));
        const rnd = prng.random();
        const mark = g.mark();
        defer g.resetTo(mark);
        const routing: ?[]u32 = if (sp.counted) try countedRouting(a, rnd, &counts, sp.rows) else null;
        defer if (routing) |r| a.free(r);
        var c = try makeCase(a, &g, rnd, sp.rows, sp.experts, sp.n, sp.skew, routing);
        defer c.deinit(a);
        const cnt = try a.alloc(u32, sp.experts);
        defer a.free(cnt);
        @memset(cnt, 0);
        for (c.slots) |e| cnt[e] += 1;
        {
            var mx: u32 = 0;
            var cls: [6]u32 = @splat(0); // 0, 1-32, 33-64, 65-127, 128-255, >= 256
            var full128: u64 = 0;
            for (cnt) |k| {
                mx = @max(mx, k);
                full128 += k / 128 * 128;
                cls[if (k == 0) 0 else if (k <= 32) 1 else if (k <= 64) 2 else if (k < 128) 3 else if (k < 256) 4 else 5] += 1;
            }
            std.debug.print("TURBOEXL3_PF_T128_CASE case={s} rows={d} experts={d} n={d} max_rows_per_expert={d} experts_0={d} 1-32={d} 33-64={d} 65-127={d} 128-255={d} ge256={d} slots_in_full_128={d}/{d}\n", .{ sp.name, sp.rows, sp.experts, sp.n, mx, cls[0], cls[1], cls[2], cls[3], cls[4], cls[5], full128, c.slots.len });
        }
        const geo: prefill.Geometry = .{ .hidden = hidden, .inter = inter, .topk = topk, .experts = sp.experts, .rate = c.rate, .dec = .mul1 };
        const S = sp.rows * topk;
        var bbuf: [5]prefill.Options = undefined;
        for (t128BaseSets(sp.n, sp.served_only, &bbuf)) |base| {
            const bmark = g.mark();
            defer g.resetTo(bmark);
            var ps = try prefill.Prefill.init(geo, base);
            defer ps.deinit();
            var bnb: [32]u8 = undefined;
            const bname = (Arm{ .lib = base }).name(&bnb);
            // the mixed chain's whole call (bf16 and f32 out), each in an eval of its own as a layer runs it
            const ym = [_]mlx.mlx_array{
                try g.adopt(try ps.moe(s, c.x_bf, c.inds1, c.sc1, c.ops, sp.rows, .bfloat16)),
                try g.adopt(try ps.moe(s, c.x_bf, c.inds1, c.sc1, c.ops, sp.rows, .float32)),
            };
            for (ym) |y| try g.evalAll(&.{y});
            // the mixed chain, stage by stage
            const r = try ps.route(s, c.inds1, S);
            defer r.deinit();
            const po = try ps.prep(s, c.x_bf, c.inds1, r.order, c.ops, S);
            for (po) |x| _ = try g.adopt(x);
            const zgu = try ps.gemmGU(s, r, po[0], po[1], po[3], po[4], c.ops, S);
            for (zgu) |x| _ = try g.adopt(x);
            const xd = try ps.mid(s, zgu[0], zgu[1], c.inds1, r.order, c.ops, S);
            for (xd) |x| _ = try g.adopt(x);
            const zd = try g.adopt(try ps.gemmD(s, r, xd[0], xd[1], c.ops, S));
            const ys = [_]mlx.mlx_array{
                try g.adopt(try ps.finish(s, zd, c.inds1, po[2], c.sc1, c.ops, sp.rows, .bfloat16)),
                try g.adopt(try ps.finish(s, zd, c.inds1, po[2], c.sc1, c.ops, sp.rows, .float32)),
            };
            // slot order: zg, zu, xd, xsd, zd
            const sl = [_]mlx.mlx_array{ try g.take(zgu[0], po[2], 0), try g.take(zgu[1], po[2], 0), try g.take(xd[0], po[2], 0), try g.take(xd[1], po[2], 0), try g.take(zd, po[2], 0) };
            try g.evalAll(&sl);
            try g.evalAll(&ys);
            // the mixed chain against itself: its whole call is its stages' bytes (a row's bytes do not depend on its slots'
            // places), so a whole-call mismatch below is the 128-row tiles'
            const base_d = [_]usize{ try bitDiff(&g, s, ym[0], ys[0]), try bitDiff(&g, s, ym[1], ys[1]) };
            const base_ok = base_d[0] == 0 and base_d[1] == 0;
            if (!base_ok) failed = true;
            checks += 2;
            std.debug.print("TURBOEXL3_PF_T128_BASE case={s} rows={d} experts={d} n={d} base={s} moe_vs_stages_bf16={d}/{d} moe_vs_stages_f32={d}/{d} {s}\n", .{ sp.name, sp.rows, sp.experts, sp.n, bname, base_d[0], mlx.mlx_array_size(ym[0]), base_d[1], mlx.mlx_array_size(ym[1]), if (base_ok) "PASS" else "FAIL" });
            var tiled = base;
            tiled.t128 = true;
            var pt = try prefill.Prefill.init(geo, tiled);
            defer pt.deinit();
            // FIRST the whole call through the 128-row tiles (bf16 and f32 out, each in an eval of its own) against the mixed
            // chain's whole call
            const yw = [_]mlx.mlx_array{
                try g.adopt(try pt.moeT128(s, c.x_bf, c.inds1, c.sc1, c.ops, sp.rows, .bfloat16)),
                try g.adopt(try pt.moeT128(s, c.x_bf, c.inds1, c.sc1, c.ops, sp.rows, .float32)),
            };
            for (yw) |y| try g.evalAll(&.{y});
            const whole_d = [_]usize{ try bitDiff(&g, s, yw[0], ym[0]), try bitDiff(&g, s, yw[1], ym[1]) };
            checks += 2;
            // then the 128-row tiles' chain stage by stage
            const rt = try pt.routeT128(s, c.inds1, S);
            defer rt.deinit();
            const pot = try pt.prep(s, c.x_bf, c.inds1, rt.order, c.ops, S);
            for (pot) |x| _ = try g.adopt(x);
            const zt = try pt.gemmGUT128(s, rt, pot[0], pot[1], pot[3], pot[4], c.ops, S);
            defer zt.deinit();
            const xdt = try pt.midT128(s, zt, rt, c.inds1, c.ops, S);
            for (xdt) |x| _ = try g.adopt(x);
            const zdt = try pt.gemmDT128(s, rt, xdt[0], xdt[1], c.ops, S);
            defer zdt.deinit();
            const yt = [_]mlx.mlx_array{
                try g.adopt(try pt.finishT128(s, zdt, rt.xbase, c.inds1, pot[2], c.sc1, c.ops, sp.rows, .bfloat16)),
                try g.adopt(try pt.finishT128(s, zdt, rt.xbase, c.inds1, pot[2], c.sc1, c.ops, sp.rows, .float32)),
            };
            // each GEMM's rows as one array (every row from the launch that computed it), then slot order
            const slt = [_]mlx.mlx_array{
                try g.take(try t128Merged(&g, zt, 0, rt, c.inds1, S), pot[2], 0),
                try g.take(try t128Merged(&g, zt, 1, rt, c.inds1, S), pot[2], 0),
                try g.take(xdt[0], pot[2], 0),
                try g.take(xdt[1], pot[2], 0),
                try g.take(try t128Merged(&g, zdt, 0, rt, c.inds1, S), pot[2], 0),
            };
            try g.evalAll(&slt);
            try g.evalAll(&yt);
            const tables_ok = try t128TablesMatch(a, &g, rt, cnt, tiled);
            // the whole call against its own stages: the same kernels on the same rows
            const self_d = [_]usize{ try bitDiff(&g, s, yw[0], yt[0]), try bitDiff(&g, s, yw[1], yt[1]) };
            checks += 2;
            const names = [_][]const u8{ "zg", "zu", "xd", "xsd", "zd", "y_bf16", "y_f32" };
            var stage_d: [7]usize = undefined;
            for (0..7) |i| {
                stage_d[i] = if (i < 5) try bitDiff(&g, s, slt[i], sl[i]) else try bitDiff(&g, s, yt[i - 5], ys[i - 5]);
                checks += 1;
            }
            var ok = tables_ok and whole_d[0] == 0 and whole_d[1] == 0 and self_d[0] == 0 and self_d[1] == 0;
            for (stage_d) |d| ok = ok and d == 0;
            if (!ok) failed = true;
            std.debug.print("TURBOEXL3_PF_T128 case={s} rows={d} experts={d} n={d} base={s} moe_bf16={d}/{d} moe_f32={d}/{d} whole_vs_stages={d}/{d} tables={s}", .{ sp.name, sp.rows, sp.experts, sp.n, bname, whole_d[0], mlx.mlx_array_size(yw[0]), whole_d[1], mlx.mlx_array_size(yw[1]), self_d[0], self_d[1], if (tables_ok) "match" else "MISMATCH" });
            for (names, stage_d) |nm, d| std.debug.print(" {s}={d}", .{ nm, d });
            std.debug.print(" {s}\n", .{if (ok) "PASS" else "FAIL"});
        }
    }
    std.debug.print("TURBOEXL3_PF_T128_DONE checks={d} {s}\n", .{ checks, if (failed) "FAIL" else "PASS" });
    try testing.expect(!failed);
}
