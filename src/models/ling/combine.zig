//! Ling-3.1-flash's routed combine at prompt rows: a wide call's routed rows (each pair's expert output at weight 1, f32)
//! times their routing weights, summed over the call's top-k, in one launch. The host's combine is three MLX ops,
//! `sum(multiply(astype(routed, f32), expand_dims(weights, -1)), -2)` (turboexl3_ling's streamed source); this gives its
//! bytes word for word: MLX's Multiply rounds each product to f32, and its sum over K <= 8 rows is col_reduce_small with
//! one thread per row (threadgroup y = min(8, K)), folded as v_k = p_k + 0 (each thread's op(v, Sum::init)), then
//! t = v_0 and t = v_k + t in k order (thread 0 over the threadgroup's values).
//! - `combineRouted`: over the joined routed array [n, K, hidden];
//! - `combineParts`: over the call's unjoined outputs (JOINLESS: up to `max_sources` f32 sources [r_i, hidden], each
//!   pair's (source, row) in loc int32 [n, K, 2]), so the joined array is never built.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const decode = @import("../../core/decode.zig");

/// The fold's rounding points: no product contracted into a sum (the pragma), so each product rounds to f32 alone
pub const contract = @embedFile("kernels/combine/contract.metal");
/// JOINLESS's source choice: source s of the `max_sources` a launch binds
pub const pick = @embedFile("kernels/combine/pick.metal");
/// The thread's row and columns, then the head of the fold's k loop (pair (row, k)'s weight)
pub const combine_1 = @embedFile("kernels/combine/combine_1.metal");
/// Pair (row, k)'s output row in the joined routed array ...
pub const src_routed = @embedFile("kernels/combine/src_routed.metal");
/// ... or read in place from its source (JOINLESS)
pub const src_parts = @embedFile("kernels/combine/src_parts.metal");
/// The product, MLX's col_reduce_small fold, and the store
pub const combine_2 = @embedFile("kernels/combine/combine_2.metal");

pub const header_routed: [:0]const u8 = contract;
pub const header_parts: [:0]const u8 = contract ++ pick;
/// The combine over routed f32 [n, K, D] and weights f32 [n, K]: out f32 [n, D]
pub const text_routed: [:0]const u8 = combine_1 ++ src_routed ++ combine_2;
/// The same over sources s0..s23 f32 [r_i, D] and loc int32 [n, K, 2]: out f32 [n, D]
pub const text_parts: [:0]const u8 = combine_1 ++ src_parts ++ combine_2;

/// The sources a parts launch binds (MLX binds at most 31 buffers: 24 sources, loc, weights and the output)
pub const max_sources: usize = 24;
/// The widest top-k the fold's order holds for: col_reduce_small's one thread per row (threadgroup y = min(8, K))
pub const max_topk: c_int = 8;
/// A thread's adjacent outputs (`C` in the text)
const cols: u32 = 4;
const threads: u32 = 256;

const routed_in = [_][*:0]const u8{ "routed", "weights" };
const parts_in = [_][*:0]const u8{ "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "loc", "weights" };
const out_names = [_][*:0]const u8{"out"};

comptime {
    if (parts_in.len != max_sources + 2) @compileError("parts_in must name max_sources sources, loc and weights");
}

const Kernels = struct { routed: mlx.mlx_fast_metal_kernel, parts: mlx.mlx_fast_metal_kernel };
const lock = Global(std.atomic.Value(u32), "ling_combine_lock", std.atomic.Value(u32).init(0));
const kernels = Global(?Kernels, "ling_combine_kernels", null);

/// This file's module state (`Global`): what a host that shares its units' state registers (root's `shared_state`).
pub const shared_state = .{ lock, kernels };

/// The two kernels, made at the first call
fn kernelsFor() !Kernels {
    while (lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer lock.get().*.store(0, .release);
    if (kernels.get().*) |k| return k;
    const r = ks.newKernel("turboexl3_ling_combine", &routed_in, &out_names, text_routed, header_routed, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(r);
    const p = ks.newKernel("turboexl3_ling_combine_parts", &parts_in, &out_names, text_parts, header_parts, true) orelse return error.KernelCreateFailed;
    kernels.get().* = .{ .routed = r, .parts = p };
    return kernels.get().*.?;
}

/// One launch of `k` over `ins` for n rows of `topk` pairs into f32 [n, hidden]
fn launch(s: mlx.mlx_stream, k: mlx.mlx_fast_metal_kernel, ins: []const mlx.mlx_array, n: c_int, topk: c_int, hidden: c_int) !mlx.mlx_array {
    const dc: u64 = (@as(u64, @intCast(hidden)) + cols - 1) / cols;
    const total: u64 = @as(u64, @intCast(n)) * dc;
    if (total > std.math.maxInt(u32)) return error.BadExl3Shape;
    const c = try ks.buildConfig(.{
        .grid = .{ @intCast(total), 1, 1 },
        .threadgroup = .{ threads, 1, 1 },
        .out_shapes = &.{&[_]c_int{ n, hidden }},
        .out_dtypes = &.{.float32},
        .template = &.{
            .{ .name = "K", .value = .{ .int = topk } },
            .{ .name = "D", .value = .{ .int = hidden } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    var o: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(k, c, ins, &o, s);
    return o[0];
}

/// The combine of the joined routed array: routed f32 [n, K, hidden] (each pair's expert output at weight 1) times
/// weights f32 [n, K], summed over K -> f32 [n, hidden], the bytes of MLX's `sum(multiply(astype(routed, f32),
/// expand_dims(weights, -1)), -2)`. K from 1 to `max_topk`; error.BadExl3Shape / error.BadExl3Dtype otherwise. The
/// caller frees the result.
pub fn combineRouted(s: mlx.mlx_stream, routed: mlx.mlx_array, weights: mlx.mlx_array) !mlx.mlx_array {
    const rs = mlx.getShape(routed);
    const ws = mlx.getShape(weights);
    if (rs.len != 3 or ws.len != 2 or rs[0] < 1 or rs[1] < 1 or rs[1] > max_topk or rs[2] < 1 or ws[0] != rs[0] or ws[1] != rs[1]) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(routed) != .float32 or mlx.mlx_array_dtype(weights) != .float32) return error.BadExl3Dtype;
    const k = try kernelsFor();
    return launch(s, k.routed, &.{ routed, weights }, rs[0], rs[1], rs[2]);
}

/// `combineRouted` over a wide call's unjoined outputs (JOINLESS): `outs` 1 to `max_sources` sources f32 [r_i, hidden],
/// `loc` int32 [n, K, 2] each pair's (source, row) there (the streamer's `planJoinless` / `fillJoinlessLoc`: every
/// pair's row lies in its source), weights f32 [n, K] -> f32 [n, hidden], the bytes of `combineRouted` over the array
/// the join would build (each pair's row copied from its source). Unused source slots bind source 0. The caller frees
/// the result.
pub fn combineParts(s: mlx.mlx_stream, outs: []const mlx.mlx_array, loc: mlx.mlx_array, weights: mlx.mlx_array) !mlx.mlx_array {
    if (outs.len == 0 or outs.len > max_sources) return error.BadExl3Shape;
    const ls = mlx.getShape(loc);
    const ws = mlx.getShape(weights);
    if (ls.len != 3 or ws.len != 2 or ls[0] < 1 or ls[1] < 1 or ls[1] > max_topk or ls[2] != 2 or ws[0] != ls[0] or ws[1] != ls[1]) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(loc) != .int32 or mlx.mlx_array_dtype(weights) != .float32) return error.BadExl3Dtype;
    const hidden = blk: {
        const s0 = mlx.getShape(outs[0]);
        if (s0.len != 2 or s0[1] < 1) return error.BadExl3Shape;
        break :blk s0[1];
    };
    for (outs) |x| {
        const sh = mlx.getShape(x);
        if (sh.len != 2 or sh[0] < 1 or sh[1] != hidden) return error.BadExl3Shape;
        if (mlx.mlx_array_dtype(x) != .float32) return error.BadExl3Dtype;
    }
    var ins: [max_sources + 2]mlx.mlx_array = undefined;
    for (ins[0..max_sources], 0..) |*x, i| x.* = outs[if (i < outs.len) i else 0];
    ins[max_sources] = loc;
    ins[max_sources + 1] = weights;
    const k = try kernelsFor();
    return launch(s, k.parts, &ins, ls[0], ls[1], hidden);
}

test "turboexl3 ling combine texts: the joined and JOINLESS texts are one composition with their row read swapped; the fold keeps MLX's rounding points (no contraction, each product alone, p + 0 then v + t)" {
    const t = std.testing;
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/combine.metal"));
    const has = struct {
        fn f(hay: []const u8, needle: []const u8) bool {
            return std.mem.indexOf(u8, hay, needle) != null;
        }
    }.f;
    try t.expect(std.mem.eql(u8, text_routed, combine_1 ++ src_routed ++ combine_2));
    try t.expect(std.mem.eql(u8, text_parts, combine_1 ++ src_parts ++ combine_2));
    try t.expect(has(contract, pin[0]) and std.mem.startsWith(u8, header_parts, contract) and std.mem.endsWith(u8, header_parts, pick));
    // the product its own statement, then p + 0, then thread 0's fold in k order
    for (pin[1..4]) |line| try t.expect(has(combine_2, line));
    // only the JOINLESS text picks its source
    try t.expect(has(src_parts, pin[4]) and !has(text_routed, pin[4]));
    try t.expect(has(combine_2, pin[5]));
    // every operand a launch binds is a name its text reads; no name takes MLX's shape / strides suffixes
    for ([_][:0]const u8{ text_routed, text_parts }, [_][]const [*:0]const u8{ &routed_in, &parts_in }) |txt, ins| {
        for (ins) |n| try t.expect(has(txt, std.mem.span(n)) or has(header_parts, std.mem.span(n)));
        for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| try t.expect(!has(txt, suf));
    }
}
