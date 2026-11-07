//! Ling-3.1-flash's post-attention add_norm folded into the MoE router's launch 1 (`addNormLogits`): at decode rows 1
//! to 8, one launch makes a layer's residual add of its mixer output, the RMSNorm of that residual with the
//! post-attention norm (mlx-serve's add_norm, sdk/blocks/norms/add_norm.zig, the plain arm), and over the normed rows
//! the router's logits with the read-ahead predictor's scores (`router.routerLogits`, its threadgroups and statements).
//! The host serves these as two launches: the add_norm, then launch 1 over its normed rows.
//! - Each threadgroup first makes the add_norm's sum of squares of every row it reads (the router's threadgroups their
//!   row, the predictor's every row; `an_open`): the add_norm's own statements, its 1024 threads' sums run as 8 of its
//!   simdgroups in turn on each of the 4 simdgroups here, lane for lane, so each sum is the add_norm's word.
//! - Each read of x then makes the normed word where it is read (`an_head`'s readers in place of x's row, `an_*_x`):
//!   the add rounded to bf16, then w * bf16(h * inv) rounded to bf16, the add_norm's statements.
//! - The router's threadgroups store their columns of the residual and the normed rows, the add_norm's words, for the
//!   MoE block's other readers (the shared expert, the routed call) and the next layer.
//! So the logits and the scores are launch 1's words of the add_norm's normed rows, and the residual and the normed
//! rows are the add_norm's.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const decode = @import("../../core/decode.zig");
const router = @import("router.zig");

// ── texts ──

/// The readers of the normed rows (a header)
pub const an_head = @embedFile("kernels/router/an_head.metal");
/// The add_norm's sums of squares of the rows the threadgroup reads, and the router's threadgroups' columns of the
/// residual and the normed rows
pub const an_open = @embedFile("kernels/router/an_open.metal");
/// The router's x row: the normed row's reader
pub const an_logits_x = @embedFile("kernels/router/an_logits_x.metal");
/// The predictor's x at one row: the normed row's reader
pub const an_pred_one_x = @embedFile("kernels/router/an_pred_one_x.metal");
/// The predictor's x at 2 to 8 rows: the normed rows' 4-vector reader
pub const an_pred_wide_x = @embedFile("kernels/router/an_pred_wide_x.metal");

const open: [:0]const u8 = router.head_1 ++ an_open ++ router.head_2;
const gemv_1: [:0]const u8 = router.logits_1a ++ an_logits_x ++ router.logits_1b;
const pred: [:0]const u8 = router.pred_open ++ router.pred_one_a ++ an_pred_one_x ++ router.pred_one_b ++ router.pred_wide_a ++ an_pred_wide_x ++ router.pred_wide_b;

/// The router's logits alone (the last MoE layer): H, XD bf16 [R, HK], NW bf16 [HK], eps (0-dim f32), w f32 [NE, HK]
/// -> logits f32 [R, NE], H_NEW, HN bf16 [R, HK]
pub const text_logits: [:0]const u8 = open ++ gemv_1 ++ router.logits_w ++ router.logits_2 ++ router.close;
/// The logits and the predictor's scores (its score the header's): + pw bf16 [NE, HK], pb f32 [NE] -> logits, scores
/// f32 [R, NE], H_NEW, HN
pub const text_pred: [:0]const u8 = open ++ gemv_1 ++ router.logits_w ++ router.logits_2 ++ pred ++ router.close;
/// `text_logits` over the f16 weight w [NE, HK]
pub const text_logits_h: [:0]const u8 = open ++ gemv_1 ++ router.logits_wh ++ router.logits_2 ++ router.close;
/// `text_pred` over the f16 weight w [NE, HK]
pub const text_pred_h: [:0]const u8 = open ++ gemv_1 ++ router.logits_wh ++ router.logits_2 ++ pred ++ router.close;

/// The widest row: the add_norm's single-row kernel (HK / 4 threads, at most 1024)
pub const max_hidden: c_int = 4096;

const threads: u32 = 128;
const logits_in = [_][*:0]const u8{ "H", "XD", "NW", "eps", "w" };
const logits_out = [_][*:0]const u8{ "logits", "H_NEW", "HN" };
const pred_in = [_][*:0]const u8{ "H", "XD", "NW", "eps", "w", "pw", "pb" };
const pred_out = [_][*:0]const u8{ "logits", "scores", "H_NEW", "HN" };

/// The predictor kernels' headers: the score's (DeepSeek's or Ling's), then the normed rows' readers
pub const header_pred: [:0]const u8 = router.score_softplus_sqrt ++ an_head;
pub const header_pred_sig: [:0]const u8 = router.score_sigmoid ++ an_head;

const Kind = enum(u3) { logits, pred, pred_sig, logits_h, pred_h, pred_sig_h };
const n_kinds = @typeInfo(Kind).@"enum".field_names.len;
const Kernels = struct {
    logits: mlx.mlx_fast_metal_kernel,
    pred: mlx.mlx_fast_metal_kernel,
    pred_sig: mlx.mlx_fast_metal_kernel,
    logits_h: mlx.mlx_fast_metal_kernel,
    pred_h: mlx.mlx_fast_metal_kernel,
    pred_sig_h: mlx.mlx_fast_metal_kernel,
};
/// A cached launch config (it bakes the row count's output shapes and grid, and the hidden size)
const Slot = struct { hidden: c_int, cfg: mlx.mlx_fast_metal_kernel_config };
const max_rows_u: usize = @intCast(router.max_rows);
const lock = Global(std.atomic.Value(u32), "ling_anroute_lock", std.atomic.Value(u32).init(0));
const kernels = Global(?Kernels, "ling_anroute_kernels", null);
const cfgs = Global([n_kinds][max_rows_u]?Slot, "ling_anroute_cfgs", @splat(@splat(null)));

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ lock, kernels, cfgs };

/// Whether the launch takes rows of width `hidden`: launch 1's rows and widths (`router.takes`) up to the add_norm's
/// single-row width
pub fn takes(rows: c_int, hidden: c_int) bool {
    return router.takes(rows, hidden) and hidden <= max_hidden;
}

/// The six kernels, made at the first call (the caller holds the lock)
fn kernelsFor() !Kernels {
    if (kernels.get().*) |k| return k;
    const l = ks.newKernel("turboexl3_ling_router_an_logits", &logits_in, &logits_out, text_logits, an_head, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(l);
    const p = ks.newKernel("turboexl3_ling_router_an_pred", &pred_in, &pred_out, text_pred, header_pred, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(p);
    const ps = ks.newKernel("turboexl3_ling_router_an_pred_sig", &pred_in, &pred_out, text_pred, header_pred_sig, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(ps);
    const lh = ks.newKernel("turboexl3_ling_router_an_logits_h", &logits_in, &logits_out, text_logits_h, an_head, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(lh);
    const ph = ks.newKernel("turboexl3_ling_router_an_pred_h", &pred_in, &pred_out, text_pred_h, header_pred, true) orelse return error.KernelCreateFailed;
    errdefer decode.freeKernel(ph);
    const psh = ks.newKernel("turboexl3_ling_router_an_pred_sig_h", &pred_in, &pred_out, text_pred_h, header_pred_sig, true) orelse return error.KernelCreateFailed;
    kernels.get().* = .{ .logits = l, .pred = p, .pred_sig = ps, .logits_h = lh, .pred_h = ph, .pred_sig_h = psh };
    return kernels.get().*.?;
}

/// The launch's config: launch 1's `router.threadgroups` of 128 threads, its outputs (the logits, and the scores with
/// a predictor, f32 [rows, 512]) then the residual and the normed rows (bf16 [rows, hidden])
fn gemvCfg(rows: c_int, hidden: c_int, with_pred: bool) !mlx.mlx_fast_metal_kernel_config {
    const out = [_]c_int{ rows, router.n_experts };
    const hsh = [_]c_int{ rows, hidden };
    const tmpl = [_]ks.TemplateArg{
        .{ .name = "R", .value = .{ .int = rows } },
        .{ .name = "NE", .value = .{ .int = router.n_experts } },
        .{ .name = "HK", .value = .{ .int = hidden } },
    };
    const grid: [3]u32 = .{ router.threadgroups(@intCast(rows), with_pred) * threads, 1, 1 };
    if (with_pred) return ks.buildConfig(.{ .grid = grid, .threadgroup = .{ threads, 1, 1 }, .out_shapes = &.{ &out, &out, &hsh, &hsh }, .out_dtypes = &.{ .float32, .float32, .bfloat16, .bfloat16 }, .template = &tmpl });
    return ks.buildConfig(.{ .grid = grid, .threadgroup = .{ threads, 1, 1 }, .out_shapes = &.{ &out, &hsh, &hsh }, .out_dtypes = &.{ .float32, .bfloat16, .bfloat16 }, .template = &tmpl });
}

const Launch = struct { k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config };

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
    };
    const slot = &cfgs.get().*[@intFromEnum(kind)][@intCast(rows - 1)];
    if (slot.*) |cached| {
        if (cached.hidden == hidden) return .{ .k = kern, .c = cached.cfg };
        _ = mlx.mlx_fast_metal_kernel_config_free(cached.cfg);
        slot.* = null;
    }
    const with_pred = switch (kind) {
        .logits, .logits_h => false,
        .pred, .pred_sig, .pred_h, .pred_sig_h => true,
    };
    const c = try gemvCfg(rows, hidden, with_pred);
    slot.* = .{ .hidden = hidden, .cfg = c };
    return .{ .k = kern, .c = c };
}

/// `addNormLogits`' outputs: launch 1's (the logits f32 [rows, 512], with a predictor its scores f32 [rows, 512]), then
/// the add_norm's (the residual and the normed rows, bf16 [rows, hidden]). The caller frees them.
pub const Out = struct { logits: mlx.mlx_array, scores: ?mlx.mlx_array, h: mlx.mlx_array, normed: mlx.mlx_array };

/// mlx-serve's add_norm of `h` + `delta` (bf16 [rows, hidden], `takes`) with the norm weight `nw` (bf16 [hidden]) and
/// `eps` (0-dim f32), and launch 1 (`router.routerLogits`) of its normed rows over `w` (the router table's memory as
/// [512, hidden] row-major: f32, or the f16 weight) with `pred`, in one launch: launch 1's words of the add_norm's
/// normed rows, and the add_norm's residual and normed rows. error.BadExl3Shape / error.BadExl3Dtype on anything else,
/// before anything is built. The caller frees the outputs.
pub fn addNormLogits(s: mlx.mlx_stream, h: mlx.mlx_array, delta: mlx.mlx_array, nw: mlx.mlx_array, eps: mlx.mlx_array, w: mlx.mlx_array, p: ?router.Predictor) !Out {
    const hs = mlx.getShape(h);
    if (hs.len != 2 or !takes(hs[0], hs[1])) return error.BadExl3Shape;
    const rows = hs[0];
    const hidden = hs[1];
    const ds = mlx.getShape(delta);
    if (ds.len != 2 or ds[0] != rows or ds[1] != hidden) return error.BadExl3Shape;
    const ns = mlx.getShape(nw);
    if (ns.len != 1 or ns[0] != hidden) return error.BadExl3Shape;
    const ws = mlx.getShape(w);
    if (ws.len != 2 or ws[0] != router.n_experts or ws[1] != hidden) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(h) != .bfloat16 or mlx.mlx_array_dtype(delta) != .bfloat16 or mlx.mlx_array_dtype(nw) != .bfloat16) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(eps) != .float32 or mlx.mlx_array_ndim(eps) != 0) return error.BadExl3Dtype;
    const wdt = mlx.mlx_array_dtype(w);
    if (wdt != .float32 and wdt != .float16) return error.BadExl3Dtype;
    const half = wdt == .float16;
    const pr = p orelse {
        const lo = try launchFor(if (half) .logits_h else .logits, rows, hidden);
        var o3: [3]mlx.mlx_array = undefined;
        try ks.applyConfig(lo.k, lo.c, &.{ h, delta, nw, eps, w }, &o3, s);
        return .{ .logits = o3[0], .scores = null, .h = o3[1], .normed = o3[2] };
    };
    const ps = mlx.getShape(pr.w);
    const bs = mlx.getShape(pr.bias);
    if (ps.len != 2 or ps[0] != router.n_experts or ps[1] != hidden or bs.len != 1 or bs[0] != router.n_experts) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(pr.w) != .bfloat16 or mlx.mlx_array_dtype(pr.bias) != .float32) return error.BadExl3Dtype;
    const kind: Kind = if (pr.score == .sigmoid) (if (half) .pred_sig_h else .pred_sig) else (if (half) .pred_h else .pred);
    const lp = try launchFor(kind, rows, hidden);
    var o4: [4]mlx.mlx_array = undefined;
    try ks.applyConfig(lp.k, lp.c, &.{ h, delta, nw, eps, w, pr.w, pr.bias }, &o4, s);
    return .{ .logits = o4[0], .scores = o4[1], .h = o4[2], .normed = o4[3] };
}

const testing = std.testing;

test "turboexl3 ling anroute texts: each text is launch 1's composition with the add_norm's sums after the head, x's reads swapped for the normed rows' readers and nothing else; launch 1's texts are their pieces' compositions" {
    const t = testing;
    try t.expect(std.mem.eql(u8, router.text_logits, router.head_1 ++ router.head_2 ++ router.logits_1a ++ router.logits_x ++ router.logits_1b ++ router.logits_w ++ router.logits_2 ++ router.close));
    try t.expect(std.mem.eql(u8, router.text_pred_h, router.head_1 ++ router.head_2 ++ router.logits_1a ++ router.logits_x ++ router.logits_1b ++ router.logits_wh ++ router.logits_2 ++ router.pred_open ++ router.pred_one_a ++ router.pred_one_x ++ router.pred_one_b ++ router.pred_wide_a ++ router.pred_wide_x ++ router.pred_wide_b ++ router.close));
    try t.expect(std.mem.eql(u8, text_pred_h, router.head_1 ++ an_open ++ router.head_2 ++ router.logits_1a ++ an_logits_x ++ router.logits_1b ++ router.logits_wh ++ router.logits_2 ++ router.pred_open ++ router.pred_one_a ++ an_pred_one_x ++ router.pred_one_b ++ router.pred_wide_a ++ an_pred_wide_x ++ router.pred_wide_b ++ router.close));
    // the swapped pieces are one line each, and no text here keeps one of launch 1's lines of x
    for ([_][]const u8{ router.logits_x, router.pred_one_x, router.pred_wide_x, an_logits_x, an_pred_one_x, an_pred_wide_x }) |line| {
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, line, "\n"));
    }
    for ([_][]const u8{ router.logits_x, router.pred_one_x, router.pred_wide_x }) |line| {
        for ([_][]const u8{ text_logits, text_pred, text_logits_h, text_pred_h }) |text| try t.expect(std.mem.indexOf(u8, text, line) == null);
    }
    try t.expect(!takes(1, 8192 - 1024) and takes(1, 4096) and takes(8, 4096) and !takes(9, 4096));
}
