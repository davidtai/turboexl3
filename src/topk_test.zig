//! The routed decode chain below a top-k of 4: a shared expert served as a one-expert bank (top-1) and the DSpark stages of
//! DeepSeek-V4.1 (top-3). dfin7 (forms 7, 8 and 9) writes each block's 128 outputs from its first 128 threads, 32 per routed
//! slot, so below top-4 form 3's dfin finishes the down projection (`Kernels3.moeForm`): every form must then give form 3's
//! bytes, and form 3 must equal the host reference within the chain's tolerance.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const support = @import("test_support.zig");
const G = support.Graph;
const format = @import("format.zig");
const ref = @import("ref.zig");
const root = @import("root.zig");

const testing = std.testing;

fn ci(x: u32) c_int {
    return @intCast(x);
}

test "turboexl3 gpu topk: root.moe at top-1, 2 and 3 gives form 3's bytes under decode forms 7, 8 and 9, and form 3 equals the reference (TURBOEXL3_GPU=1)" {
    _ = std.c.getenv("TURBOEXL3_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    defer root.decode_form_override.get().* = null;
    const hidden: u32 = 2560;
    const inter: u32 = 640;
    const n: u32 = 80;
    const dec: format.Decode = .mul1;
    const bf = support.bf16Bits;
    var failed = false;
    // (top-k, experts): the one-expert bank of a shared expert, then two and three of twelve
    for ([_][2]u32{ .{ 1, 1 }, .{ 2, 12 }, .{ 3, 12 } }) |kc| {
        const topk = kc[0];
        const experts = kc[1];
        var prng = std.Random.DefaultPrng.init(0x70C0 + topk);
        const rnd = prng.random();
        var bank = try ref.synthBank(a, rnd, experts, hidden, inter, format.Rate.fromN(n).?);
        defer bank.deinit(a);
        const ht = ci(hidden / 16);
        const it = ci(inter / 16);
        for ([_]u32{ 1, 3 }) |rows| {
            const mark = g.mark();
            defer g.resetTo(mark);
            const nslots = rows * topk;
            const x = try a.alloc(f32, rows * hidden);
            defer a.free(x);
            for (x) |*v| v.* = @bitCast(@as(u32, bf(rnd.floatNorm(f32))) << 16); // bf16-exact inputs
            const slots = try a.alloc(u32, nslots);
            defer a.free(slots);
            const inds = try a.alloc(i32, nslots);
            defer a.free(inds);
            const scores = try a.alloc(f32, nslots);
            defer a.free(scores);
            for (0..rows) |r| for (0..topk) |k| {
                const i = r * topk + k;
                slots[i] = @intCast((k * 5 + r) % experts);
                inds[i] = @intCast(slots[i]);
                scores[i] = @bitCast(@as(u32, bf(0.05 + 0.1 * rnd.float(f32))) << 16); // bf16-exact scores
            };
            const y_ref = try a.alloc(f32, rows * hidden);
            defer a.free(y_ref);
            for (0..rows) |r| try ref.moeRowDec(a, &bank, dec, x[r * hidden ..][0..hidden], slots[r * topk ..][0..topk], scores[r * topk ..][0..topk], y_ref[r * hidden ..][0..hidden]);
            const bank_arrays: root.Bank = .{
                .gate = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(experts), ht, it, ci(n) }, .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ ci(experts), ci(hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ ci(experts), ci(inter) }, .float16) },
                .up = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ ci(experts), ht, it, ci(n) }, .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ ci(experts), ci(hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ ci(experts), ci(inter) }, .float16) },
                .down = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ ci(experts), it, ht, ci(n) }, .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ ci(experts), ci(inter) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ ci(experts), ci(hidden) }, .float16) },
            };
            const xf = try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(hidden) }, .float32);
            const inds3 = try g.hostArray(std.mem.sliceAsBytes(inds), &.{ 1, ci(rows), ci(topk) }, .int32);
            const sc3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, ci(rows), ci(topk) }, .float32), .bfloat16);
            const got3 = try a.alloc(f32, rows * hidden);
            defer a.free(got3);
            const got = try a.alloc(f32, rows * hidden);
            defer a.free(got);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const x3 = if (dt == .float32) xf else try g.astype(xf, dt);
                for ([_]u8{ 3, 7, 8, 9 }) |form| {
                    root.decode_form_override.get().* = form;
                    const y = try g.adopt(try root.moe(s, x3, bank_arrays, inds3, sc3, dec, false));
                    try testing.expectEqual(dt, mlx.mlx_array_dtype(y));
                    _ = try g.hostF32(try g.astype(y, .float32), if (form == 3) got3 else got);
                    if (form == 3) {
                        // the chain's tolerance beyond its rounding: each slot's down output is stored in f16 before the
                        // weighted sum (half an f16 ulp, 2^-11 of a value; with 1 to 3 terms little of it averages out), and
                        // a bf16 output's store adds 2^-8
                        const e = ref.compareRounded(got3, y_ref, if (dt == .float32) 0x1p-11 else 0x1p-8 + 0x1p-11);
                        const ok = e.max_ratio <= 5e-3;
                        std.debug.print("TURBOEXL3_TOPK ref form=3 topk={d} rows={d} out={s} max_abs={e:.3} ratio={e:.3} {s}\n", .{ topk, rows, @tagName(dt), e.max_abs, e.max_ratio, if (ok) "PASS" else "FAIL" });
                        if (!ok) failed = true;
                        continue;
                    }
                    var differing: usize = 0;
                    for (got, got3) |p, q| differing += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
                    std.debug.print("TURBOEXL3_TOPK bytes form={d} topk={d} rows={d} out={s} differing={d}/{d} {s}\n", .{ form, topk, rows, @tagName(dt), differing, got.len, if (differing == 0) "PASS" else "FAIL" });
                    if (differing != 0) failed = true;
                }
            }
        }
    }
    try testing.expect(!failed);
}
