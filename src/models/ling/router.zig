//! Ling-3.1-flash's MoE router at decode rows 1 to 8 in two launches, with the read-ahead predictor's scores (the next
//! MoE layer's router on this call's rows), the host's words. The host (turboexl3_ling) spends nine launches a MoE
//! layer: MLX's astype of x to f32 and its f32 gemv over the router table; the served fused router kernel
//! (`sigmoid_bias_grouped`: sigmoid + expert bias, the top-8 of the 4 best of 8 groups of 64 by the sum of each group's
//! two best, the weights normalized and scaled by routed_scaling_factor) and the astype of its f32 weights to bf16; and
//! the predictor's bf16 gemv over the next layer's bf16 router, its astype to f32, logaddexp(z, 0), sqrt and the bias
//! add (the streamer's `nextScores`). The host's loader builds the router table as the f16 weight's transposed view
//! widened to f32, then + 0 (its materialized copy): MLX's astype and add keep that view's layout, so the table is
//! [hidden, 512] column-major and MLX's matmul runs gemv over its memory, the weight's own [512, hidden] row-major
//! order (b transposed), not gemv_t. Here:
//! - `routerLogits`: one launch. The router's threadgroups each run MLX's gemv statements for 16 logits of a row over
//!   `w`, the table's memory as [512, hidden] row-major (x widened to f32 in registers: the astype's values): f32 (the
//!   table transposed back, a view) or f16 (the weight as loaded, each word widened, then + 0: the table's word). With a
//!   `Predictor`, the predictor's threadgroups run MLX's gemv statements at one row and gemv_wide's at 2 to 8 (MLX's
//!   route there on M3 and later), round each word to bf16, widen it and score it: DeepSeek's sqrt(softplus(z)) + bias
//!   (`turboexl3_ling_router_pred`, `_pred_h`) or Ling's sigmoid(z) + bias (`turboexl3_ling_router_pred_sig`,
//!   `_pred_sig_h`, candidate C1). The last MoE layer has no next router (`turboexl3_ling_router_logits`, `_logits_h`).
//! - `select`: the served kernel's text verbatim over those logits, its weights stored in bf16 (TOUT bfloat16: the
//!   astype folded into its store; `turboexl3_ling_router_select`).
//! - `route`: both.
//! - `tableF32`: the loader's f32 table from the f16 weight's transposed view (the astype, then + 0), for a host that
//!   keeps the weight in f16 and runs its own matmul where these launches do not serve.
//! Every row of every call is that row's own: the logits are MLX's one-row gemv words at any row count (at 2 to 8 rows
//! the words of the host's exact verify, turboexl3_ling's MLX_SERVE_LING_EXACT_VERIFY=1 denseRows; MLX's own matmul of
//! several rows takes a steel GEMM there, another order), and the predictor's are MLX's own at the call's row count.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const decode = @import("../../core/decode.zig");

/// The launch's threadgroups (the router's, then the predictor's) ...
pub const head_1 = @embedFile("kernels/router/head_1.metal");
/// ... and the router branch's row and tile
pub const head_2 = @embedFile("kernels/router/head_2.metal");
pub const head: [:0]const u8 = head_1 ++ head_2;
/// MLX's gemv statements for 16 logits of one row up to the table's read (x widened in registers): up to x's row ...
pub const logits_1a = @embedFile("kernels/router/logits_1a.metal");
/// ... x's row (the reads of x go through `in_vec`) ...
pub const logits_x = @embedFile("kernels/router/logits_x.metal");
/// ... then the K loop up to the table's read
pub const logits_1b = @embedFile("kernels/router/logits_1b.metal");
pub const logits_1: [:0]const u8 = logits_1a ++ logits_x ++ logits_1b;
/// ... the read of an f32 table word (MLX's load) ...
pub const logits_w = @embedFile("kernels/router/logits_w.metal");
/// ... or of an f16 weight word, widened, then + 0 (the loader's table word) ...
pub const logits_wh = @embedFile("kernels/router/logits_wh.metal");
/// ... then the accumulation, the shuffle tree and the store
pub const logits_2 = @embedFile("kernels/router/logits_2.metal");
/// The router's GEMV over an f32 table (MLX's gemv statements in order)
pub const logits_gemv: [:0]const u8 = logits_1 ++ logits_w ++ logits_2;
/// The same over the f16 weight, each word widened, then + 0
pub const logits_gemv_h: [:0]const u8 = logits_1 ++ logits_wh ++ logits_2;
/// The predictor branch's opening: the threadgroup's index among the predictor's
pub const pred_open = @embedFile("kernels/router/pred_open.metal");
/// One row: MLX's gemv statements for 16 rows of the next router, each word rounded to bf16 and scored: up to x ...
pub const pred_one_a = @embedFile("kernels/router/pred_one_a.metal");
/// ... x (its reads go through `in_vec`) ...
pub const pred_one_x = @embedFile("kernels/router/pred_one_x.metal");
/// ... the rest
pub const pred_one_b = @embedFile("kernels/router/pred_one_b.metal");
pub const pred_one: [:0]const u8 = pred_one_a ++ pred_one_x ++ pred_one_b;
/// 2 to 8 rows: MLX's gemv_wide statements (k_lanes 32 to 5 rows, 16 past), each word rounded to bf16 and scored: up
/// to x ...
pub const pred_wide_a = @embedFile("kernels/router/pred_wide_a.metal");
/// ... x as rows of 4-vectors (its reads go through `x4`) ...
pub const pred_wide_x = @embedFile("kernels/router/pred_wide_x.metal");
/// ... the rest
pub const pred_wide_b = @embedFile("kernels/router/pred_wide_b.metal");
pub const pred_wide: [:0]const u8 = pred_wide_a ++ pred_wide_x ++ pred_wide_b;
/// The branch's close
pub const close = @embedFile("kernels/router/close.metal");
/// The predictor's score of each word, a header each: DeepSeek's sqrt(softplus(z)) + bias (the host's today) ...
pub const score_softplus_sqrt = @embedFile("kernels/router/score_softplus_sqrt.metal");
/// ... or Ling's own sigmoid(z) + bias (candidate C1)
pub const score_sigmoid = @embedFile("kernels/router/score_sigmoid.metal");

/// The router's logits alone (the last MoE layer): x bf16 [R, HK], w f32 [NE, HK] -> logits f32 [R, NE]
pub const text_logits: [:0]const u8 = head ++ logits_gemv ++ close;
/// The router's logits and the predictor's scores (its score the header's): x, w, pw bf16 [NE, HK], pb f32 [NE] ->
/// logits, scores f32 [R, NE]
pub const text_pred: [:0]const u8 = head ++ logits_gemv ++ pred_open ++ pred_one ++ pred_wide ++ close;
/// `text_logits` over the f16 weight w [NE, HK]
pub const text_logits_h: [:0]const u8 = head ++ logits_gemv_h ++ close;
/// `text_pred` over the f16 weight w [NE, HK]
pub const text_pred_h: [:0]const u8 = head ++ logits_gemv_h ++ pred_open ++ pred_one ++ pred_wide ++ close;
/// The host's fused router kernel at `sigmoid_bias_grouped` (sdk/blocks/moe/router_kernel.zig `moeRouterSource`, the
/// MLX kernel "mlxserve_moe_router_sigmoid_grouped"), byte for byte; turboexl3_ling's build pins this copy to the
/// host's source. Inputs logits, bias, scale (0-dim); outputs inds, scores; template TOUT, NE, NK, TG, NORM, NG, GSZ,
/// TOPKG.
pub const text_select: [:0]const u8 = @embedFile("kernels/router/select.metal");

/// Ling-3.1-flash's routing, the only one these launches take: 512 experts, top-8 of the 4 best of 8 groups, the
/// weights normalized (norm_topk_prob)
pub const n_experts: c_int = 512;
pub const top_k: c_int = 8;
pub const n_group: c_int = 8;
pub const topk_group: c_int = 4;
/// The rows a call takes: decode's one and a verify's 2 to 8 (the streamer's read-ahead `max_rows`)
pub const max_rows: c_int = 8;
/// The select's threads a row: the host's `moeRouterThreadGroup` at 512 experts
pub const select_threads: c_int = 256;
/// Launch 1's threads a threadgroup: MLX's gemv and gemv_wide (k_lanes 32) tiles' 128
const threads: u32 = 128;
const max_rows_u: usize = @intCast(max_rows);
const n_experts_u: u32 = @intCast(n_experts);

const logits_in = [_][*:0]const u8{ "x", "w" };
const logits_out = [_][*:0]const u8{"logits"};
const pred_in = [_][*:0]const u8{ "x", "w", "pw", "pb" };
const pred_out = [_][*:0]const u8{ "logits", "scores" };
/// The select's operands (the host's)
pub const select_in = [_][*:0]const u8{ "logits", "bias", "scale" };
pub const select_out = [_][*:0]const u8{ "inds", "scores" };

/// The predictor's score of each word z of x times the next router (f32, after the bf16 rounding)
pub const Score = enum {
    /// DeepSeek's sqrt(softplus(z)) + bias: MLX's add(sqrt(logaddexp(z, 0)), bias), the host streamer's `nextScores`
    softplus_sqrt,
    /// Ling's sigmoid(z) + bias: MLX's add(sigmoid(z), bias) (candidate C1, turboexl3_ling's MLX_SERVE_LING_PREDICTOR=1)
    sigmoid,
};

/// The read-ahead predictor of a MoE layer's call: the next MoE layer's router and bias
pub const Predictor = struct {
    /// bf16 [512, hidden] row-major (the host's `router_raw`: the gate weight in bf16), the product's b transposed
    w: mlx.mlx_array,
    /// f32 [512], the next layer's expert bias
    bias: mlx.mlx_array,
    score: Score = .softplus_sqrt,
};

/// `routerLogits`' outputs: the logits f32 [R, 512], and with a predictor its scores f32 [R, 512]. The caller frees them.
pub const Logits = struct { logits: mlx.mlx_array, scores: ?mlx.mlx_array };
/// `select`'s outputs: the ids u32 [R, 8] (in the selection's order) and their weights bf16 [R, 8]. The caller frees them.
pub const Selected = struct { inds: mlx.mlx_array, weights: mlx.mlx_array };
/// `route`'s outputs: `Selected`'s and the predictor's scores. The caller frees them.
pub const Routes = struct { inds: mlx.mlx_array, weights: mlx.mlx_array, scores: ?mlx.mlx_array };

const Kind = enum(u3) { logits, pred, pred_sig, logits_h, pred_h, pred_sig_h, select };
const n_kinds = @typeInfo(Kind).@"enum".field_names.len;
const Kernels = struct {
    logits: mlx.mlx_fast_metal_kernel,
    pred: mlx.mlx_fast_metal_kernel,
    pred_sig: mlx.mlx_fast_metal_kernel,
    logits_h: mlx.mlx_fast_metal_kernel,
    pred_h: mlx.mlx_fast_metal_kernel,
    pred_sig_h: mlx.mlx_fast_metal_kernel,
    select: mlx.mlx_fast_metal_kernel,
};
/// A cached launch config (it bakes the row count's output shapes and grid, and the hidden size)
const Slot = struct { hidden: c_int, cfg: mlx.mlx_fast_metal_kernel_config };
const lock = Global(std.atomic.Value(u32), "ling_router_lock", std.atomic.Value(u32).init(0));
const kernels = Global(?Kernels, "ling_router_kernels", null);
const cfgs = Global([n_kinds][max_rows_u]?Slot, "ling_router_cfgs", @splat(@splat(null)));

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ lock, kernels, cfgs };

/// Whether the launches take x [rows, hidden]: rows 1 to `max_rows`; hidden a multiple of 1024 below 8192, where MLX's
/// tiles are the texts' (gemv at BM 4 for 512 outputs and the predictor's gemv_wide) and none has a leftover block
/// (gemv's K step 128, gemv_wide's 4 * k_lanes * 8).
pub fn takes(rows: c_int, hidden: c_int) bool {
    return rows >= 1 and rows <= max_rows and hidden >= 1024 and hidden < 8192 and @rem(hidden, 1024) == 0;
}

/// Launch 1's threadgroups at `rows` rows: the router's 512 / 16 a row (MLX's gemv tiles, 16 logits each), then the
/// predictor's: 512 / 16 at one row (gemv's tiles, 16 rows of the next router each), 512 / 4 at 2 to 5 rows and 512 / 8
/// at 6 to 8 (gemv_wide at k_lanes 32 or 16: 1 or 2 rows of the next router a simdgroup, 4 simdgroups each).
pub fn threadgroups(rows: u32, pred: bool) u32 {
    const router = rows * (n_experts_u / 16);
    if (!pred) return router;
    const per: u32 = if (rows == 1) 16 else if ((rows + 4) / 5 == 1) 4 else 8;
    return router + n_experts_u / per;
}

/// The seven kernels, made at the first call (the caller holds the lock)
fn kernelsFor() !Kernels {
    if (kernels.get().*) |k| return k;
    const l = ks.newKernel("turboexl3_ling_router_logits", &logits_in, &logits_out, text_logits, "", true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(l);
    const p = ks.newKernel("turboexl3_ling_router_pred", &pred_in, &pred_out, text_pred, score_softplus_sqrt, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(p);
    const ps = ks.newKernel("turboexl3_ling_router_pred_sig", &pred_in, &pred_out, text_pred, score_sigmoid, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(ps);
    const lh = ks.newKernel("turboexl3_ling_router_logits_h", &logits_in, &logits_out, text_logits_h, "", true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(lh);
    const ph = ks.newKernel("turboexl3_ling_router_pred_h", &pred_in, &pred_out, text_pred_h, score_softplus_sqrt, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(ph);
    const psh = ks.newKernel("turboexl3_ling_router_pred_sig_h", &pred_in, &pred_out, text_pred_h, score_sigmoid, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(psh);
    const sel = ks.newKernel("turboexl3_ling_router_select", &select_in, &select_out, text_select, "", true) orelse return error.KernelCreateFailed;
    kernels.get().* = .{ .logits = l, .pred = p, .pred_sig = ps, .logits_h = lh, .pred_h = ph, .pred_sig_h = psh, .select = sel };
    return kernels.get().*.?;
}

/// The select's launch: the host's config (a threadgroup of 256 a row; TOUT bfloat16 in place of its f32)
fn selectCfg(rows: c_int) !mlx.mlx_fast_metal_kernel_config {
    const out = [_]c_int{ rows, top_k };
    return ks.buildConfig(.{
        .grid = .{ @intCast(select_threads), @intCast(rows), 1 },
        .threadgroup = .{ @intCast(select_threads), 1, 1 },
        .out_shapes = &.{ &out, &out },
        .out_dtypes = &.{ .uint32, .bfloat16 },
        .template = &.{
            .{ .name = "TOUT", .value = .{ .dtype = .bfloat16 } },
            .{ .name = "NE", .value = .{ .int = n_experts } },
            .{ .name = "NK", .value = .{ .int = top_k } },
            .{ .name = "TG", .value = .{ .int = select_threads } },
            .{ .name = "NORM", .value = .{ .int = 1 } },
            .{ .name = "NG", .value = .{ .int = n_group } },
            .{ .name = "GSZ", .value = .{ .int = @divExact(n_experts, n_group) } },
            .{ .name = "TOPKG", .value = .{ .int = topk_group } },
        },
    });
}

/// Launch 1's: `threadgroups` of 128 on x, its outputs f32 [rows, 512] (the logits, and the scores with a predictor)
fn gemvCfg(rows: c_int, hidden: c_int, pred: bool) !mlx.mlx_fast_metal_kernel_config {
    const out = [_]c_int{ rows, n_experts };
    const tmpl = [_]ks.TemplateArg{
        .{ .name = "R", .value = .{ .int = rows } },
        .{ .name = "NE", .value = .{ .int = n_experts } },
        .{ .name = "HK", .value = .{ .int = hidden } },
    };
    const grid: [3]u32 = .{ threadgroups(@intCast(rows), pred) * threads, 1, 1 };
    if (pred) return ks.buildConfig(.{ .grid = grid, .threadgroup = .{ threads, 1, 1 }, .out_shapes = &.{ &out, &out }, .out_dtypes = &.{ .float32, .float32 }, .template = &tmpl });
    return ks.buildConfig(.{ .grid = grid, .threadgroup = .{ threads, 1, 1 }, .out_shapes = &.{&out}, .out_dtypes = &.{.float32}, .template = &tmpl });
}

const Launch = struct { k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config };

/// The kernel of `kind` and its config at `rows` (and `hidden`: 0 for the select), each made once
fn launchFor(kind: Kind, rows: c_int, hidden: c_int) !Launch {
    while (lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer lock.get().*.store(0, .release);
    const k = try kernelsFor();
    const kern = switch (kind) {
        .logits => k.logits,
        .pred => k.pred,
        .pred_sig => k.pred_sig,
        .logits_h => k.logits_h,
        .pred_h => k.pred_h,
        .pred_sig_h => k.pred_sig_h,
        .select => k.select,
    };
    const slot = &cfgs.get().*[@intFromEnum(kind)][@intCast(rows - 1)];
    if (slot.*) |cached| {
        if (cached.hidden == hidden) return .{ .k = kern, .c = cached.cfg };
        _ = mlx.mlx_fast_metal_kernel_config_free(cached.cfg);
        slot.* = null;
    }
    const c = switch (kind) {
        .select => try selectCfg(rows),
        .logits, .logits_h => try gemvCfg(rows, hidden, false),
        .pred, .pred_sig, .pred_h, .pred_sig_h => try gemvCfg(rows, hidden, true),
    };
    slot.* = .{ .hidden = hidden, .cfg = c };
    return .{ .k = kern, .c = c };
}

/// Launch 1: the router's logits of x bf16 [rows, hidden] (`takes`) over `w`, the router table's memory as [512,
/// hidden] row-major: f32 (the host loader's [hidden, 512] column-major table, transposed back: a view) or f16 (the
/// weight as loaded; each word widened, then + 0, the table's word). The logits f32 [rows, 512], each row MLX's one-row
/// `matmul(astype(x_r, f32), table)` words (gemv over the table's memory); with `pred`, the predictor's scores f32
/// [rows, 512], the words of MLX's `matmul(x, transpose(pred.w))` (bf16: gemv at one row, gemv_wide at 2 to 8, M3 and
/// later), widened to f32 and scored (`Score`) with `pred.bias`. error.BadExl3Shape / error.BadExl3Dtype on anything
/// else. The caller frees the outputs.
pub fn routerLogits(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, pred: ?Predictor) !Logits {
    const xs = mlx.getShape(x);
    if (xs.len != 2 or !takes(xs[0], xs[1])) return error.BadExl3Shape;
    const rows = xs[0];
    const hidden = xs[1];
    const ws = mlx.getShape(w);
    if (ws.len != 2 or ws[0] != n_experts or ws[1] != hidden) return error.BadExl3Shape;
    const wdt = mlx.mlx_array_dtype(w);
    if (mlx.mlx_array_dtype(x) != .bfloat16 or (wdt != .float32 and wdt != .float16)) return error.BadExl3Dtype;
    const half = wdt == .float16;
    const p = pred orelse {
        const lo = try launchFor(if (half) .logits_h else .logits, rows, hidden);
        var o1: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(lo.k, lo.c, &.{ x, w }, &o1, s);
        return .{ .logits = o1[0], .scores = null };
    };
    const ps = mlx.getShape(p.w);
    const bs = mlx.getShape(p.bias);
    if (ps.len != 2 or ps[0] != n_experts or ps[1] != hidden or bs.len != 1 or bs[0] != n_experts) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(p.w) != .bfloat16 or mlx.mlx_array_dtype(p.bias) != .float32) return error.BadExl3Dtype;
    const kind: Kind = if (p.score == .sigmoid) (if (half) .pred_sig_h else .pred_sig) else (if (half) .pred_h else .pred);
    const lp = try launchFor(kind, rows, hidden);
    var o2: [2]mlx.mlx_array = undefined;
    try ks.applyConfig(lp.k, lp.c, &.{ x, w, p.w, p.bias }, &o2, s);
    return .{ .logits = o2[0], .scores = o2[1] };
}

/// Launch 2: the host's fused router (`sigmoid_bias_grouped`, its text and config) over logits f32 [rows, 512] (rows 1
/// to `max_rows`) with the expert bias f32 [512] and routed_scaling_factor `scale`: the ids u32 [rows, 8] and their
/// weights in bf16 [rows, 8], the words of the host's f32 weights after MLX's astype to bf16. error.BadExl3Shape /
/// error.BadExl3Dtype on anything else. The caller frees the outputs.
pub fn select(s: mlx.mlx_stream, logits: mlx.mlx_array, bias: mlx.mlx_array, scale: f32) !Selected {
    const ls = mlx.getShape(logits);
    const bs = mlx.getShape(bias);
    if (ls.len != 2 or ls[0] < 1 or ls[0] > max_rows or ls[1] != n_experts or bs.len != 1 or bs[0] != n_experts) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(logits) != .float32 or mlx.mlx_array_dtype(bias) != .float32) return error.BadExl3Dtype;
    const l = try launchFor(.select, ls[0], 0);
    // a 0-dim input: MLX binds it as a scalar reference (the text's `scale`)
    const sc = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(sc);
    var o: [2]mlx.mlx_array = undefined;
    try ks.applyConfig(l.k, l.c, &.{ logits, bias, sc }, &o, s);
    return .{ .inds = o[0], .weights = o[1] };
}

/// The router of x [rows, hidden] in two launches: `routerLogits` then `select` (`bias` the layer's expert bias, `scale`
/// its routed_scaling_factor, `pred` the next MoE layer's router or null). The caller frees the outputs.
pub fn route(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, scale: f32, pred: ?Predictor) !Routes {
    const l = try routerLogits(s, x, w, pred);
    defer _ = mlx.mlx_array_free(l.logits);
    errdefer if (l.scores) |sc| {
        _ = mlx.mlx_array_free(sc);
    };
    const sel = try select(s, l.logits, bias, scale);
    return .{ .inds = sel.inds, .weights = sel.weights, .scores = l.scores };
}

/// The host loader's f32 router table from its f16 source `w` (the weight's transposed view, [hidden, 512]): MLX's
/// astype to f32, then + a 0-dim f32 zero (the loader's materialized copy), each keeping `w`'s layout, so the result is
/// the loader's table word for word and stride for stride (-0.0 becomes +0.0, every other word the f16 one widened).
/// error.BadExl3Dtype unless `w` is f16. The caller frees the result.
pub fn tableF32(s: mlx.mlx_stream, w: mlx.mlx_array) !mlx.mlx_array {
    if (mlx.mlx_array_dtype(w) != .float16) return error.BadExl3Dtype;
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, w, .float32, s));
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{}, 0, .float32, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, f, zero, s));
    return out;
}

test "turboexl3 ling router texts: the logits and predictor texts are one composition per weight dtype, the table's read the only piece the f16 texts swap and the predictor's branch spliced before the close; MLX's GEMV statements carry no contraction pragma and the scores' headers do; the select text is the host's statements; every operand a name its text reads" {
    const t = std.testing;
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/router.metal"));
    const has = struct {
        fn f(hay: []const u8, needle: []const u8) bool {
            return std.mem.indexOf(u8, hay, needle) != null;
        }
    }.f;
    try t.expect(std.mem.eql(u8, text_logits, head ++ logits_1 ++ logits_w ++ logits_2 ++ close));
    try t.expect(std.mem.eql(u8, text_pred, head ++ logits_1 ++ logits_w ++ logits_2 ++ pred_open ++ pred_one ++ pred_wide ++ close));
    try t.expect(std.mem.eql(u8, text_logits_h, head ++ logits_1 ++ logits_wh ++ logits_2 ++ close));
    try t.expect(std.mem.eql(u8, text_pred_h, head ++ logits_1 ++ logits_wh ++ logits_2 ++ pred_open ++ pred_one ++ pred_wide ++ close));
    // MLX's GEMV statements compile as MLX's own (no pragma); each score's sums keep their roundings (the pragma)
    try t.expect(!has(text_pred, pin[0]) and !has(text_pred_h, pin[0]));
    try t.expect(has(score_softplus_sqrt, pin[0]) and has(score_sigmoid, pin[0]));
    // the table's read: an f32 word as MLX loads it, or an f16 word widened, then + 0 (only in the f16 texts)
    try t.expect(has(text_logits, pin[1]) and !has(text_logits, pin[2]));
    try t.expect(has(text_logits_h, pin[2]) and !has(text_logits_h, pin[1]));
    // gemv's accumulate (the router's and the one-row predictor's) and the logits' store
    try t.expect(has(logits_2, pin[3]) and has(pred_one, pin[3]) and has(logits_2, pin[4]));
    // the predictor: its bf16 word scored (one row), gemv_wide's k_lanes, dot and its bf16 word scored (2 to 8 rows)
    try t.expect(has(pred_one, pin[5]));
    for (pin[6..9]) |line| try t.expect(has(pred_wide, line));
    // DeepSeek's score: MLX's log1p product on the precise log, LogAddExp, the precise sqrt; Ling's: MLX's sigmoid
    for (pin[9..12]) |line| try t.expect(has(score_softplus_sqrt, line));
    try t.expect(has(score_sigmoid, pin[12]) and !has(score_softplus_sqrt, pin[12]));
    // the host's sigmoid (MLX's two-sided form, precise exp) and its TOUT store
    try t.expect(has(text_select, pin[13]) and has(text_select, pin[14]));
    for ([_][:0]const u8{ text_logits, text_pred, text_logits_h, text_pred_h, text_select }, [_][]const [*:0]const u8{ &(logits_in ++ logits_out), &(pred_in ++ pred_out), &(logits_in ++ logits_out), &(pred_in ++ pred_out), &(select_in ++ select_out) }) |txt, names| {
        for (names) |n| try t.expect(has(txt, std.mem.span(n)));
        for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| try t.expect(!has(txt, suf));
    }
}

test "turboexl3 ling router plan: launch 1's threadgroups per row count (the router's 32 a row; the predictor's 32 at one row, 128 at 2 to 5 rows, 64 at 6 to 8) and the shapes the launches take (rows 1 to 8, hidden a multiple of 1024 below 8192)" {
    const t = std.testing;
    try t.expectEqual(@as(u32, 32), threadgroups(1, false));
    try t.expectEqual(@as(u32, 256), threadgroups(8, false));
    try t.expectEqual(@as(u32, 32 + 32), threadgroups(1, true));
    var r: u32 = 2;
    while (r <= 5) : (r += 1) try t.expectEqual(32 * r + 128, threadgroups(r, true));
    while (r <= 8) : (r += 1) try t.expectEqual(32 * r + 64, threadgroups(r, true));
    for ([_]c_int{ 1, 2, 5, 6, 8 }) |rows| for ([_]c_int{ 1024, 4096, 7168 }) |h| try t.expect(takes(rows, h));
    for ([_][2]c_int{ .{ 0, 4096 }, .{ 9, 4096 }, .{ 1, 8192 }, .{ 1, 4000 }, .{ 1, 512 }, .{ 1, 0 }, .{ -1, 4096 } }) |c| try t.expect(!takes(c[0], c[1]));
}
