//! DEVROUTE's device wave in one call (`root.moeLingBankedLut`, the outside plugin turboexl3_ling's
//! MLX_SERVE_LING_DEVWAVE_LUT=1), held byte for byte to the wave it replaces: the copied streamer's experts.zig
//! `devWaveIn`, statement for statement (`servedWave`): the layer's LUT taken at the call's routed experts (reshaped to
//! [tokens · topk]), x's rows taken per pair (pair p at token p / topk), then `moeLingBanked` over them.
//! - CPU (always): prep2bgl's text is prep2bg's with its skip and expert lines swapped for the LUT's (each reads `sid`
//!   where prep2bg reads the row-slot's id) and the id line before them, nothing else.
//! - GPU (TURBOEXL3_GPU=1, else skipped), at Ling's shapes (hidden 4096, expert width 1280, MUL1 at n 40 = K 2.5 and
//!   n 32 = K 2; TURBOEXL3_LING_N narrows), limits 0 and 10 (the clamp after the SiLU), on operands as the plugin serves
//!   them: the banks three slot banks of 40, 16 and 8 rows, each code the pack's I16 words as the streamer holds them
//!   (the plugin's `t3Banks` hands them over as they are) and also viewed as U16, rin and rout f16; x the call's normed
//!   MoE input [tokens, hidden] bf16 (and f32), x ~ N(0, 8^2) so the clamp acts; `idx` the router's ids [tokens, 8] i32
//!   (the source casts them to i32), each token's top-8 distinct of 512 experts by Gumbel top-k over lognormal
//!   popularity; `lut` u32 [512] as the streamer writes it (`lutOf`: the 64 most popular experts resident, each its
//!   (bank << 24 | row), every other 0); tokens 1 to 6 (DEVROUTE's decode lane) and 8 (64 row-slots, the most a banked
//!   call takes). Arms: the served texts (the grouped texts with the M = 3 path at variant t4), and variant 0 of the
//!   M = 3 and today's grouped texts. Each line: y's words against the served wave's, and the prepare's ids against
//!   the LUT's take. Lines: TURBOEXL3_LING_DEVLUT.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("../../core/test_support.zig");
const G = support.Graph;
const config = @import("../../core/config.zig");
const decode = @import("../../core/decode.zig");
const format = @import("../../core/format.zig");
const ref = @import("../../core/ref.zig");
const root = @import("../../root.zig");
const banked = @import("../deepseek_v41/banked.zig");
const ds = @import("../deepseek_v41/api.zig");

const testing = std.testing;

const hidden: u32 = 4096;
const inter: u32 = 1280;
const experts: u32 = 512;
const top_k: u32 = 8;
const limit: c_int = 10;
const sigma: f32 = 8;
/// The banks' slot rows: base, ext, transient (64 resident experts)
const bank_rows = [3]u32{ 40, 16, 8 };

fn ci(x: anytype) c_int {
    return @intCast(x);
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

test "turboexl3 ling devlut texts: prep2bgl is prep2bg with its skip and expert lines swapped for the LUT's and the id line before them, nothing else" {
    const pin = comptime support.pins(@embedFile("kernels/test/pins/devlut_texts.metal"));
    const skip_rk = @embedFile("../deepseek_v41/kernels/banked/skip_rk.metal");
    const expert = @embedFile("../deepseek_v41/kernels/banked/prep2b_expert.metal");
    const slot = @embedFile("../deepseek_v41/kernels/banked/prep2bl_slot.metal");
    const skip_sid = @embedFile("../deepseek_v41/kernels/banked/skip_sid.metal");
    const expert_l = @embedFile("../deepseek_v41/kernels/banked/prep2bl_expert.metal");
    try testing.expectEqualStrings(decode.prep2_1 ++ skip_rk ++ expert ++ decode.prep2_row ++ decode.prep2_body, banked.text_prep2bg);
    try testing.expectEqualStrings(decode.prep2_1 ++ slot ++ skip_sid ++ expert_l ++ decode.prep2_row ++ decode.prep2_body, banked.text_prep2bgl);
    // each swapped line is prep2bg's with the row-slot's id read as `sid`; the id line binds sid and is the only one that
    // reads the ids' buffer (it writes it)
    var buf: [1024]u8 = undefined;
    for ([_][2][]const u8{ .{ skip_rk, skip_sid }, .{ expert, expert_l } }) |p| {
        const n = std.mem.replacementSize(u8, p[0], pin[0], pin[1]);
        _ = std.mem.replace(u8, p[0], pin[0], pin[1], buf[0..n]);
        try testing.expectEqualStrings(buf[0..n], p[1]);
        try testing.expect(std.mem.indexOf(u8, p[1], pin[0]) == null);
    }
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, slot, "\n"));
    try testing.expect(std.mem.indexOf(u8, banked.text_prep2bgl, pin[0]) == null);
}

/// Each token's top-8 distinct experts by Gumbel top-k over the experts' popularity (`logw`), token by token.
fn drawRoutes(rnd: std.Random, logw: []const f64, tokens: u32, idx: []i32) void {
    var keys: [experts]f64 = undefined;
    for (0..tokens) |t| {
        for (&keys, logw) |*kv, lw| kv.* = lw - @log(-@log(@max(rnd.float(f64), 1e-300)));
        for (0..top_k) |j| {
            var best: usize = 0;
            for (keys, 0..) |kv, e| {
                if (kv > keys[best]) best = e;
            }
            idx[t * top_k + j] = @intCast(best);
            keys[best] = -std.math.inf(f64);
        }
    }
}

/// The layer's LUT as the streamer writes it (`lutOf`): the `bank_rows` total most popular experts resident, each its
/// own (bank << 24 | row), filled bank by bank in popularity order; every other expert 0.
fn residentLut(logw: []const f64, lut: *[experts]u32, resident: *[experts]bool) void {
    var order: [experts]u16 = undefined;
    for (&order, 0..) |*o, e| o.* = @intCast(e);
    std.mem.sort(u16, &order, logw, struct {
        fn lt(w: []const f64, x: u16, y: u16) bool {
            return w[x] > w[y];
        }
    }.lt);
    @memset(lut, 0);
    @memset(resident, false);
    var k: usize = 0;
    for (bank_rows, 0..) |rows, b| for (0..rows) |r| {
        lut[order[k]] = (@as(u32, @intCast(b)) << 24) | @as(u32, @intCast(r));
        resident[order[k]] = true;
        k += 1;
    };
}

/// The bank as the plugin hands it to turboexl3 (`t3Banks`: code as it is, rin as suh, rout as svh), its code in `td`
/// (the streamer's I16, or the same words as U16).
fn servedBank(g: *G, b: *const ref.Bank, td: mlx.mlx_dtype) !root.Bank {
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

/// The wave DEVROUTE serves today (experts.zig `devWaveIn`): the LUT taken at the experts, x's rows taken per pair, then
/// `moeLingBanked` on `sel`'s texts over them; also the taken ids.
fn servedWave(g: *G, s: mlx.mlx_stream, x: mlx.mlx_array, idx: mlx.mlx_array, lut: mlx.mlx_array, banks: *const [3]root.Bank, lim: c_int, sel: root.BankedSel) ![2]mlx.mlx_array {
    const nk = mlx.mlx_array_size(idx);
    const ids = try g.take(lut, try g.reshape(idx, &.{ci(nk)}), 0);
    var tok: [banked.banked_rows_max]i32 = undefined;
    for (tok[0..nk], 0..) |*t, p| t.* = @intCast(p / top_k);
    const xr = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(tok[0..nk]), &.{ci(nk)}, .int32), 0);
    return .{ try g.adopt(try root.moeLingBankedSel(s, xr, banks, ids, .mul1, lim, sel)), ids };
}

/// The words of `got` and `want` (same size and dtype: f32 or u32) that differ, and `got`'s non-finite f32 words
fn compare(a: std.mem.Allocator, g: *G, got: mlx.mlx_array, want: mlx.mlx_array) !struct { differing: usize, nonfinite: usize, of: usize } {
    const n = mlx.mlx_array_size(want);
    const dt = mlx.mlx_array_dtype(want);
    if (mlx.mlx_array_size(got) != n or mlx.mlx_array_dtype(got) != dt or (dt != .float32 and dt != .uint32)) return error.SizeMismatch;
    const p = try a.alloc(u32, n);
    defer a.free(p);
    const q = try a.alloc(u32, n);
    defer a.free(q);
    if (dt == .float32) {
        _ = try g.hostF32(got, @ptrCast(p));
        _ = try g.hostF32(want, @ptrCast(q));
    } else {
        _ = try g.hostU32(got, p);
        _ = try g.hostU32(want, q);
    }
    var d: usize = 0;
    var nf: usize = 0;
    for (p, q) |u, v| {
        d += @intFromBool(u != v);
        nf += @intFromBool(dt == .float32 and !std.math.isFinite(@as(f32, @bitCast(u))));
    }
    return .{ .differing = d, .nonfinite = nf, .of = n };
}

// The GPU byte test (see the file's doc): every arm's y against the served wave's on the same texts and variant, and the
// prepare's ids against the LUT's take, at each rate, limit, code dtype, x dtype and token count.
test "turboexl3 gpu ling devlut: moeLingBankedLut is DEVROUTE's device wave (the LUT taken at the routed experts, x's rows taken per pair, moeLingBanked) byte for byte, tokens 1 to 6 and 8 at top-8, limits 0 and 10, x bf16 and f32, on the streamer's banks (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = std.heap.page_allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const saved = config.get().*;
    defer config.configure(saved) catch {};
    const t4 = root.groupVariantOf("t4") orelse return error.NoVariantT4;
    var nbuf: [8]u32 = undefined;
    const rates = envList("TURBOEXL3_LING_N", &.{ 40, 32 }, &nbuf);
    const Arm = struct { name: []const u8, sel: root.BankedSel, variant: u8 };
    const arms = [_]Arm{ .{ .name = "served(g4m3,t4)", .sel = .g4m3, .variant = t4 }, .{ .name = "g4m3,v0", .sel = .g4m3, .variant = 0 }, .{ .name = "g4,v0", .sel = .g4, .variant = 0 } };
    var failed = false;
    var lines: usize = 0;
    for (rates) |n| {
        defer _ = mlx.mlx_clear_cache();
        const rate = format.Rate.fromN(n) orelse return error.Exl3RateUnsupported;
        const mark = g.mark();
        defer g.resetTo(mark);
        var prng = std.Random.DefaultPrng.init(0xDE71A7 ^ @as(u64, n));
        const rnd = prng.random();
        var hb: [3]ref.Bank = undefined;
        var made: usize = 0;
        defer for (hb[0..made]) |*b| b.deinit(a);
        for (&hb, bank_rows) |*b, e| {
            b.* = try ref.synthBank(a, rnd, e, hidden, inter, rate);
            made += 1;
        }
        var rb_i: [3]root.Bank = undefined;
        var rb_u: [3]root.Bank = undefined;
        for (&hb, &rb_i, &rb_u) |*b, *i, *u| {
            i.* = try servedBank(&g, b, .int16);
            u.* = try servedBank(&g, b, .uint16);
        }
        var logw: [experts]f64 = undefined;
        for (&logw) |*w| w.* = 1.5 * rnd.floatNorm(f64);
        var lut_h: [experts]u32 = undefined;
        var resident: [experts]bool = undefined;
        residentLut(&logw, &lut_h, &resident);
        const lut = try g.hostArray(std.mem.sliceAsBytes(&lut_h), &.{ci(experts)}, .uint32);
        for ([_]u32{ 1, 2, 3, 4, 5, 6, 8 }) |tokens| {
            const m2 = g.mark();
            defer g.resetTo(m2);
            const nk = tokens * top_k;
            var idx_h: [banked.banked_rows_max]i32 = undefined;
            drawRoutes(rnd, &logw, tokens, idx_h[0..nk]);
            var hits: usize = 0;
            for (idx_h[0..nk]) |e| hits += @intFromBool(resident[@intCast(e)]);
            const idx = try g.hostArray(std.mem.sliceAsBytes(idx_h[0..nk]), &.{ ci(tokens), ci(top_k) }, .int32);
            const x = try a.alloc(f32, tokens * hidden);
            defer a.free(x);
            for (x) |*v| v.* = sigma * rnd.floatNorm(f32);
            const x32 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ ci(tokens), ci(hidden) }, .float32);
            const xbf = try g.astype(x32, .bfloat16);
            for ([_]c_int{ 0, limit }) |lim| for ([_]struct { banks: *const [3]root.Bank, name: []const u8 }{ .{ .banks = &rb_i, .name = "i16" }, .{ .banks = &rb_u, .name = "u16" } }) |code| for ([_]struct { x: mlx.mlx_array, name: []const u8 }{ .{ .x = xbf, .name = "bf16" }, .{ .x = x32, .name = "f32" } }) |xa| {
                // the served operands are i16 codes and bf16 x; the others once a rate, at one token count
                if ((std.mem.eql(u8, code.name, "u16") or std.mem.eql(u8, xa.name, "f32")) and tokens != 3) continue;
                for (arms) |arm| {
                    const m3 = g.mark();
                    defer g.resetTo(m3);
                    // the plugin's configuration (ling_math.configure): the grouped texts with the M = 3 path, a variant needs them
                    var c = config.get().*;
                    c.banked_group = 4;
                    c.banked_m3 = true;
                    c.banked_variant_gu = arm.variant;
                    c.banked_variant_dn = arm.variant;
                    try config.configure(c);
                    const want = try servedWave(&g, s, xa.x, idx, lut, code.banks, lim, arm.sel);
                    const got = try g.adopt(try root.moeLingBankedLutSel(s, xa.x, idx, lut, code.banks, .mul1, lim, arm.sel));
                    // the prepare's ids (the down's), from the gate|up half alone
                    const h = try ds.bankedGateUpLutAct(s, xa.x, idx, lut, code.banks, .mul1, .{ .limit = @intCast(lim), .clamp = .silu_out }, arm.sel, arm.variant);
                    _ = try g.adopt(h.hd);
                    _ = try g.adopt(h.hs);
                    const slots = try g.adopt(h.slots);
                    try testing.expectEqualSlices(c_int, &.{ ci(nk), ci(hidden) }, mlx.getShape(got));
                    const dy = try compare(a, &g, got, want[0]);
                    const di = try compare(a, &g, slots, want[1]);
                    const ok = dy.differing == 0 and dy.nonfinite == 0 and di.differing == 0;
                    std.debug.print("TURBOEXL3_LING_DEVLUT n={d} tokens={d} pairs={d} hits={d} limit={d} code={s} x={s} arm={s} differing={d}/{d} ids_differing={d}/{d} nonfinite={d} {s}\n", .{ n, tokens, nk, hits, lim, code.name, xa.name, arm.name, dy.differing, dy.of, di.differing, di.of, dy.nonfinite, if (ok) "PASS" else "FAIL" });
                    lines += 1;
                    if (!ok) failed = true;
                }
            };
        }
    }
    try testing.expect(lines > 0);
    try testing.expect(!failed);
}
