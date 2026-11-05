//! Qwen3.8-Flash-Next's entry points (root re-exports each under its own name): the decode-row merges at a gated
//! attention layer's linears (`decb.zig`), the fused attention projections (`attn_io.zig`), the GDN tail (`gdn_out.zig`)
//! and the MoE with its gated shared expert in the routed launches (`shared.zig`). Each runs on root's caches of kernels
//! and dense shapes (`root.kernelsFor`, `root.denseOf`).
const std = @import("std");
const mlx = @import("sdk").mlx;
const root = @import("../../root.zig");
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const attn_io = @import("attn_io.zig");
const decode_merge = @import("decb.zig");
const gdn_out = @import("gdn_out.zig");
const shared_heads = @import("shared.zig");
const Bank = root.Bank;
const Proj = root.Proj;
const SlotStats = root.SlotStats;
const bankGeometry = root.bankGeometry;
const denseOf = root.denseOf;
const freeArr = root.freeArr;
const kernels = root.kernels;
const kernelsFor = root.kernelsFor;
const rowsOf = root.rowsOf;
const validateCall = root.validateCall;

// ── Decode-row merges at a gated attention layer's EXL3 linears (`decb.zig`) ──
// Rows 1..kernels.DECODE_ROWS_MAX (the GEMV path), each the bytes of the chain it replaces, with DEC-ONE taking the call or
// not; error.Exl3EngineNotImplemented where a call is not served (the caller keeps its chain).

/// `decode_merge.gatedDecode`: y [1, rows, out] = o_proj(T(attn * sigmoid(gate))) at decode rows for the attention
/// core's attn [1, heads, rows, hd] (bf16 or f16), the gate read in place, `sigtab` attn's dtype [65536]: the bytes of the
/// host's strided sigmoid-gate kernel and `linear`, with the product formed inside o_proj's prepare (DEC-ONE's dec0 after it).
pub fn decodeMergeGated(s: mlx.mlx_stream, attn: mlx.mlx_array, gate: decode_merge.Gate, sigtab: mlx.mlx_array, o: Proj, dec: format.Decode) !mlx.mlx_array {
    const ash = mlx.getShape(attn);
    if (ash.len != 4 or ash[0] != 1 or ash[1] < 1 or ash[2] < 1 or ash[3] < 1) return error.BadExl3Shape;
    const heads: u32 = @intCast(ash[1]);
    const rows: usize = @intCast(ash[2]);
    const hd: u32 = @intCast(ash[3]);
    const ad = mlx.mlx_array_dtype(attn);
    if (ad != .bfloat16 and ad != .float16) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(gate.base) != ad or mlx.mlx_array_dtype(sigtab) != ad or mlx.mlx_array_size(sigtab) != 65536) return error.BadExl3Dtype;
    const d = try denseOf(o, heads * hd, ad, dec);
    const y2 = try decode_merge.gatedDecode(s, attn, gate, sigtab, .{ .d = d, .trellis = o.trellis, .suh = o.suh, .svh = o.svh }, heads, hd, rows);
    defer freeArr(y2);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &[_]c_int{ 1, @intCast(rows), @intCast(d.shape.out) }, 3, s));
    return y;
}

/// The qmerge form (`decode_merge.mergeGatedDecode`): y [1, rows, out] = o_proj(T(merge(pacc, pml) * sigmoid(gate)))
/// for the host's fused QSA core's split-K partials (pacc f32 [heads, rows, NS, 256], pml f32 [heads, rows, NS, 2]), the gate
/// in place and `sigtab` (the gate's dtype): the bytes of the host's merge launch, its strided gate kernel and `linear`.
pub fn decodeMergeQmergeGated(s: mlx.mlx_stream, pacc: mlx.mlx_array, pml: mlx.mlx_array, gate: decode_merge.Gate, sigtab: mlx.mlx_array, o: Proj, dec: format.Decode) !mlx.mlx_array {
    const psh = mlx.getShape(pacc);
    if (psh.len != 4 or psh[0] < 1 or psh[1] < 1 or psh[3] != 256) return error.BadExl3Shape;
    const heads: u32 = @intCast(psh[0]);
    const rows: usize = @intCast(psh[1]);
    const gd = mlx.mlx_array_dtype(gate.base);
    if (gd != .bfloat16 and gd != .float16) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(sigtab) != gd or mlx.mlx_array_size(sigtab) != 65536) return error.BadExl3Dtype;
    const d = try denseOf(o, heads * 256, gd, dec);
    const y2 = try decode_merge.mergeGatedDecode(s, pacc, pml, gate, sigtab, .{ .d = d, .trellis = o.trellis, .suh = o.suh, .svh = o.svh }, heads, rows);
    defer freeArr(y2);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &[_]c_int{ 1, @intCast(rows), @intCast(d.shape.out) }, 3, s));
    return y;
}

/// `decode_merge.qPlanes`: q_proj of x [..., in] at decode rows as two planes [1, rows, heads, hd] (each head's query
/// half, then its gate half): `linear`'s bytes, relaid out, so neither half is a strided view.
pub fn decodeMergeQPlanes(s: mlx.mlx_stream, x: mlx.mlx_array, q: Proj, heads: u32, hd: u32, dec: format.Decode) ![2]mlx.mlx_array {
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return error.BadExl3Dtype;
    const xsh = mlx.getShape(x);
    if (xsh.len < 1) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const d = try denseOf(q, in_dim, xd, dec);
    const v = try rowsOf(x, in_dim, s);
    defer freeArr(v.x2);
    const planes = try decode_merge.qPlanes(s, v.x2, v.rows, .{ .d = d, .trellis = q.trellis, .suh = q.suh, .svh = q.svh }, heads, hd);
    defer for (planes) |p| freeArr(p);
    var out: [2]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null } };
    errdefer for (out) |p| if (p.ctx != null) freeArr(p);
    for (planes, 0..) |p, i| {
        out[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out[i], p, &[_]c_int{ 1, @intCast(v.rows), @intCast(heads), @intCast(hd) }, 4, s));
    }
    return out;
}

/// `decode_merge.indexerQk`: the indexer projection of x [..., in] at decode rows with its finish's epilogue: qk
/// [1, rows, out] (`linear`'s bytes), the index queries normed and rotated [1, heads, rows, 128] and the one pooled key the
/// call completes [1, 1, 128] (each where asked; the caller frees every non-null result).
pub fn decodeMergeIndexer(s: mlx.mlx_stream, x: mlx.mlx_array, idx: Proj, ep: decode_merge.IdxEpilogue, dec: format.Decode) !decode_merge.IdxOut {
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16) return error.BadExl3Dtype;
    const xsh = mlx.getShape(x);
    if (xsh.len < 1) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const d = try denseOf(idx, in_dim, xd, dec);
    const v = try rowsOf(x, in_dim, s);
    defer freeArr(v.x2);
    var r = try decode_merge.indexerQk(s, v.x2, v.rows, .{ .d = d, .trellis = idx.trellis, .suh = idx.suh, .svh = idx.svh }, ep);
    errdefer r.deinit();
    var qk = mlx.mlx_array_new();
    mlx.check(mlx.mlx_reshape(&qk, r.qk, &[_]c_int{ 1, @intCast(v.rows), @intCast(d.shape.out) }, 3, s)) catch |e| {
        freeArr(qk);
        return e;
    };
    freeArr(r.qk);
    r.qk = qk;
    return r;
}

/// ATTN-IN (`attn_io.attnIn`): an attention layer's q (each head [query | gate]), k and v projections of x [..., in]
/// (bf16 or f16) at prefill rows, with the q / k RMSNorm and partial RoPE in the finish: q, k, v [1, heads, rows, 256]
/// (normed and rotated q and k), the gate [1, rows, heads, 256], each the bytes of `linear` + MLX's split / rms_norm /
/// transpose / rope. error.Exl3EngineNotImplemented where it does not serve (the caller keeps its chain).
pub fn attnIn(s: mlx.mlx_stream, x: mlx.mlx_array, q: Proj, k: Proj, v: Proj, ep: attn_io.Epilogue, dec: format.Decode) !attn_io.InOut {
    const xsh = mlx.getShape(x);
    if (xsh.len < 2) return error.BadExl3Shape;
    var rows_c: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows_c *= d;
    if (rows_c < 1) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return error.BadExl3Dtype;
    const dq = try denseOf(q, in_dim, xd, dec);
    const dk = try denseOf(k, in_dim, xd, dec);
    const dv = try denseOf(v, in_dim, xd, dec);
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows_c, @intCast(in_dim) }, 2, s));
    return attn_io.attnIn(s, x2, @intCast(rows_c), .{ .d = dq, .trellis = q.trellis, .suh = q.suh, .svh = q.svh }, .{ .d = dk, .trellis = k.trellis, .suh = k.suh, .svh = k.svh }, .{ .d = dv, .trellis = v.trellis, .suh = v.suh, .svh = v.svh }, ep);
}

/// `attn_io.prepare4`: ATTN-IN's prepare launch with the QSA indexer projection `idx` as a fourth member, over
/// x [..., in] (bf16 or f16): xp[0..3] q's, k's and v's rotated copies (for `attnIn`'s `Epilogue.prep`), xp[3] the
/// indexer's (for `linearPrepared`), each f16 [rows, in], the caller's. error.Exl3EngineNotImplemented where it does not
/// serve (the caller keeps its launches).
pub fn attnPrep4(s: mlx.mlx_stream, x: mlx.mlx_array, q: Proj, k: Proj, v: Proj, idx: Proj, dec: format.Decode) ![4]mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh.len < 2) return error.BadExl3Shape;
    var rows_c: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows_c *= d;
    if (rows_c < 1) return error.BadExl3Shape;
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return error.BadExl3Dtype;
    const dq = try denseOf(q, in_dim, xd, dec);
    const dk = try denseOf(k, in_dim, xd, dec);
    const dv = try denseOf(v, in_dim, xd, dec);
    const di = try denseOf(idx, in_dim, xd, dec);
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows_c, @intCast(in_dim) }, 2, s));
    return attn_io.prepare4(s, x2, @intCast(rows_c), .{ .d = dq, .trellis = q.trellis, .suh = q.suh, .svh = q.svh }, .{ .d = dk, .trellis = k.trellis, .suh = k.suh, .svh = k.svh }, .{ .d = dv, .trellis = v.trellis, .suh = v.suh, .svh = v.svh }, .{ .d = di, .trellis = idx.trellis, .suh = idx.suh, .svh = idx.svh });
}

/// `attn_io.linearPrepared` (named apart from the GDN tail's `linearPrepared`): `linear` of an x of dtype `xd` over its rows
/// already prepared (xp f16 [rows, in], `attnPrep4`'s fourth copy): y [rows, out] in `xd` (the caller reshapes), by
/// `linear`'s inner product and finish launches on that Dense (`denseFor` keys it as `linear` does), so `linear`'s bytes.
/// error.Exl3EngineNotImplemented where `linear`'s path at these rows reads no prepared rows (the GEMV rows, the fused GEMM).
pub fn linearPreparedIdx(s: mlx.mlx_stream, xp: mlx.mlx_array, p: Proj, xd: mlx.mlx_dtype, dec: format.Decode) !mlx.mlx_array {
    const xsh = mlx.getShape(xp);
    if (xsh.len != 2 or xsh[0] < 1 or mlx.mlx_array_dtype(xp) != .float16) return error.BadExl3Shape;
    if (xd != .bfloat16 and xd != .float32 and xd != .float16) return error.BadExl3Dtype;
    const d = try denseOf(p, @intCast(xsh[1]), xd, dec);
    return attn_io.linearPrepared(s, xp, .{ .d = d, .trellis = p.trellis, .suh = p.suh, .svh = p.svh }, @intCast(xsh[0]));
}

/// ATTN-OUT (`attn_io.linearGated`): y [1, rows, out] = o_proj(T(attn * sigmoid(gate))) for the attention core's
/// attn [1, heads, rows, hd] (bf16 or f16), the gate read in place (`attn_io.Gate`) and `sigtab` (MLX's sigmoid of every
/// 16-bit pattern, attn's dtype): the bytes of MLX's Sigmoid, Multiply and `linear`. error.Exl3EngineNotImplemented where
/// it does not serve.
pub fn linearGated(s: mlx.mlx_stream, attn: mlx.mlx_array, gate: attn_io.Gate, sigtab: mlx.mlx_array, o: Proj, dec: format.Decode) !mlx.mlx_array {
    const ash = mlx.getShape(attn);
    if (ash.len != 4 or ash[0] != 1 or ash[1] < 1 or ash[2] < 1 or ash[3] < 1) return error.BadExl3Shape;
    const heads: u32 = @intCast(ash[1]);
    const rows: usize = @intCast(ash[2]);
    const hd: u32 = @intCast(ash[3]);
    const ad = mlx.mlx_array_dtype(attn);
    if (ad != .bfloat16 and ad != .float16) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(gate.base) != ad or mlx.mlx_array_dtype(sigtab) != ad or mlx.mlx_array_size(sigtab) != 65536) return error.BadExl3Dtype;
    const d = try denseOf(o, heads * hd, ad, dec);
    const y2 = try attn_io.linearGated(s, attn, gate, sigtab, .{ .d = d, .trellis = o.trellis, .suh = o.suh, .svh = o.svh }, heads, hd, rows);
    defer freeArr(y2);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &[_]c_int{ 1, @intCast(rows), @intCast(d.shape.out) }, 3, s));
    return y;
}

/// ATTN-IN over the un-rotated trunk (`attn_io.attnInStd`): an attention layer's q (each head [query | gate]), k and v from their
/// standard-basis products in x's dtype, `srcs[m]` [..., stride_m] holding member m at columns [offs[m], ...) (a joined
/// product passed three times, or each member's own plain matmul), the outputs `attnIn`'s. error.Exl3EngineNotImplemented
/// where it does not serve (the caller keeps its chain).
pub fn attnInStd(s: mlx.mlx_stream, srcs: [3]mlx.mlx_array, offs: [3]u32, ep: attn_io.Epilogue) !attn_io.InOut {
    var flat: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (flat) |a| freeArr(a);
    var ins: [3]attn_io.StdIn = undefined;
    var rows: usize = 0;
    for (srcs, offs, &flat, &ins, 0..) |a, off, *f, *in, i| {
        const sh = mlx.getShape(a);
        if (sh.len < 2) return error.BadExl3Shape;
        var r: c_int = 1;
        for (sh[0 .. sh.len - 1]) |d| r *= d;
        if (r < 1) return error.BadExl3Shape;
        if (i == 0) rows = @intCast(r) else if (rows != @as(usize, @intCast(r))) return error.BadExl3Shape;
        try mlx.check(mlx.mlx_reshape(f, a, &[_]c_int{ r, sh[sh.len - 1] }, 2, s));
        in.* = .{ .a = f.*, .stride = @intCast(sh[sh.len - 1]), .off = off };
    }
    return attn_io.attnInStd(s, rows, ins[0], ins[1], ins[2], ep, mlx.mlx_array_dtype(srcs[0]));
}

/// ATTN-OUT over the un-rotated trunk (`attn_io.linearGatedStd`): y [1, rows, out] = T(attn * sigmoid(gate)) W_std for the attention
/// core's attn [1, heads, rows, hd], the gate read in place and `sigtab` (`linearGated`'s operands), o_proj's W_std from the
/// resident table. error.Exl3EngineNotImplemented where o_proj is not un-rotated or the call is not served.
pub fn linearGatedStd(s: mlx.mlx_stream, attn: mlx.mlx_array, gate: attn_io.Gate, sigtab: mlx.mlx_array, o: Proj) !mlx.mlx_array {
    const ash = mlx.getShape(attn);
    if (ash.len != 4 or ash[0] != 1 or ash[1] < 1 or ash[2] < 1 or ash[3] < 1) return error.BadExl3Shape;
    const w = dense.residentStd(o.trellis) orelse return error.Exl3EngineNotImplemented;
    defer freeArr(w);
    const rows: usize = @intCast(ash[2]);
    const y2 = try attn_io.linearGatedStd(s, attn, gate, sigtab, w, @intCast(ash[1]), @intCast(ash[3]), rows);
    defer freeArr(y2);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &[_]c_int{ 1, @intCast(rows), mlx.getShape(y2)[1] }, 3, s));
    return y;
}

/// The GDN layer's tail at prefill rows (`gdn_out.zig`). It returns out_proj's prepared rows xp F16 [rows, hv 128] (natural
/// order, as `linearPrepared` and `linearPiecePrepared` take them). The inputs are the recurrence's output y
/// [B, S, hv, 128] (bf16 or f16) and in_proj_z's inner product zq F16 [B S, hv 128] (`linearInnerSplit`'s), with
/// rows = B S. One launch runs three steps of the chain:
/// - z's finish (`linearFinish` with z's `svh`);
/// - the gated norm: MLX's rms_norm over each 128-wide head with `norm.w` and `norm.eps`, times MLX's sigmoid of z by
///   `norm.sigtab`, in y's dtype;
/// - out_proj's prepare (`linearPrep` with o's `suh`).
/// Every element is the chain's bytes. It refuses with error.Exl3EngineNotImplemented, before any launch, at decode rows
/// (<= `kernels.DECODE_ROWS_MAX`, where `linear` takes the GEMV), for another head width and off the GPU.
pub fn gdnOutPrep(s: mlx.mlx_stream, y: mlx.mlx_array, zq: mlx.mlx_array, z: Proj, norm: gdn_out.Norm, o: Proj) !mlx.mlx_array {
    const ysh = mlx.getShape(y);
    if (ysh.len != 4 or ysh[0] < 1 or ysh[1] < 1 or ysh[2] < 1) return error.BadExl3Shape;
    const dt = mlx.mlx_array_dtype(y);
    if (dt != .bfloat16 and dt != .float16) return error.BadExl3Dtype;
    const rows: usize = @intCast(ysh[0] * ysh[1]);
    if (ysh[3] != gdn_out.head_dim or rows <= kernels.DECODE_ROWS_MAX or !mlx.streamIsGpu(s)) return error.Exl3EngineNotImplemented;
    const hv: u32 = @intCast(ysh[2]);
    const vd: c_int = @intCast(hv * gdn_out.head_dim);
    const zsh = mlx.getShape(zq);
    if (zsh.len != 2 or zsh[0] != ysh[0] * ysh[1] or zsh[1] != vd or mlx.mlx_array_dtype(zq) != .float16) return error.BadExl3Shape;
    for ([_]mlx.mlx_array{ z.svh, o.suh }) |vec| {
        const sh = mlx.getShape(vec);
        if (sh.len != 1 or sh[0] != vd or mlx.mlx_array_dtype(vec) != .float16) return error.BadExl3Shape;
    }
    if (mlx.mlx_array_dtype(norm.w) != dt or mlx.mlx_array_size(norm.w) != gdn_out.head_dim) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(norm.sigtab) != dt or mlx.mlx_array_size(norm.sigtab) != 65536) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(norm.eps) != .float32 or mlx.mlx_array_ndim(norm.eps) != 0) return error.BadExl3Dtype;
    return gdn_out.prepare(s, y, zq, z.svh, norm, o.suh, hv, rows);
}

/// Decode / verify rows (up to `shared_heads.shared_rows_max`) of a MoE with a gated shared expert, both in form 9's
/// launches (`shared_heads.SharedHeads.moeShared`): y [x's shape, x's dtype] = routed + sigmoid(logit) * shared(x), the bytes of the
/// host's chain over `moe` (form 3), three `linear`s (dense.zig), its table SwiGLU (`sigtab` OUT_T [65536]: MLX's sigmoid of
/// every 16-bit pattern) and MLX's Sigmoid, Multiply and Add. `shared`: the dense linears (trellis [in/16, out/16, n], suh,
/// svh), `logit` the shared gate's logit per row ([rows] or [.., rows, 1]). error.Exl3EngineNotImplemented for a call the
/// fused launches do not carry (the caller keeps its chain). The shared expert's heads run on the routed kernels
/// (`moeShared`); `shared_heads.SharedHeads`' side launches and v2 heads give the same bytes.
pub fn moeFusedShared(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, shared: Bank, sigtab: mlx.mlx_array, logit: mlx.mlx_array) !mlx.mlx_array {
    const geo = try validateCall(x, bank, inds, scores);
    if (geo.gate_up.n != geo.down.n) return error.Exl3EngineNotImplemented;
    const xsh = mlx.getShape(x);
    const rows_c: c_int = xsh[0] * xsh[1];
    const rows: usize = @intCast(rows_c);
    if (rows > shared_heads.shared_rows_max) return error.Exl3EngineNotImplemented;
    const K: c_int = mlx.getShape(inds)[2];
    const xd = mlx.mlx_array_dtype(x);
    if (mlx.mlx_array_dtype(sigtab) != xd or mlx.mlx_array_size(sigtab) != 65536 or mlx.mlx_array_dtype(logit) != xd or mlx.mlx_array_size(logit) != rows) return error.Exl3EngineNotImplemented;
    const gsh = mlx.getShape(shared.gate.trellis);
    const dsh = mlx.getShape(shared.down.trellis);
    if (gsh.len != 3 or dsh.len != 3) return error.Exl3EngineNotImplemented;
    const inter_s: u32 = @intCast(gsh[1] * 16);
    const sb: shared_heads.SharedBank = .{
        .gate_t = shared.gate.trellis,
        .gate_suh = shared.gate.suh,
        .gate_svh = shared.gate.svh,
        .up_t = shared.up.trellis,
        .up_suh = shared.up.suh,
        .up_svh = shared.up.svh,
        .down_t = shared.down.trellis,
        .down_suh = shared.down.suh,
        .down_svh = shared.down.svh,
        .n_gu = @intCast(gsh[2]),
        .n_dn = @intCast(dsh[2]),
        .ks_gu = dense.Dense.ksFor(@intCast(geo.hidden), inter_s),
        .ks_dn = dense.Dense.ksFor(inter_s, @intCast(geo.hidden)),
        .defer_affine = dense.Dense.deferFor(dec),
        .sigtab = sigtab,
        .logit = logit,
    };
    const dgeo: decode.Geometry = .{ .hidden = @intCast(geo.hidden), .inter = @intCast(geo.width), .topk = @intCast(K), .rate = geo.gate_up, .dec = dec };
    const k3 = try kernelsFor(dgeo, xd);
    if (!shared_heads.SharedHeads.sharedAdmitted(k3, rows, sb)) return error.Exl3EngineNotImplemented;
    const heads = try shared_heads.forKernels(k3);
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows_c, xsh[2] }, 2, s));
    var inds1 = mlx.mlx_array_new();
    defer freeArr(inds1);
    try mlx.check(mlx.mlx_reshape(&inds1, inds, &[_]c_int{rows_c * K}, 1, s));
    var s1 = mlx.mlx_array_new();
    defer freeArr(s1);
    try mlx.check(mlx.mlx_reshape(&s1, scores, &[_]c_int{rows_c * K}, 1, s));
    var l1 = mlx.mlx_array_new();
    defer freeArr(l1);
    try mlx.check(mlx.mlx_reshape(&l1, logit, &[_]c_int{rows_c}, 1, s));
    var sbr = sb;
    sbr.logit = l1;
    const kb: decode.Kernels.Bank = .{
        .gate_t = bank.gate.trellis,
        .gate_suh = bank.gate.suh,
        .gate_svh = bank.gate.svh,
        .up_t = bank.up.trellis,
        .up_suh = bank.up.suh,
        .up_svh = bank.up.svh,
        .down_t = bank.down.trellis,
        .down_suh = bank.down.suh,
        .down_svh = bank.down.svh,
    };
    const y0 = try heads.moeShared(k3, s, x2, inds1, s1, kb, rows, sbr);
    defer freeArr(y0);
    if (!fused_shared_logged.get().*) {
        fused_shared_logged.get().* = true;
        std.debug.print("[turboexl3] shared expert fused into the routed decode launches (gate|up K{d} split {d}, down K{d} split {d})\n", .{ sb.n_gu / 16, sb.ks_gu, sb.n_dn / 16, sb.ks_dn });
    }
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y0, xsh.ptr, 3, s));
    return y;
}
const fused_shared_logged = Global(bool, "root_fused_shared_logged", false);

/// `moeFusedShared` with the routing folded into form 9's routed prepare launch (`shared_heads.SharedHeads.moeSharedRouted`).
/// `logits` [.., rows, E] is the router
/// gemv's output in x's dtype and `k` the experts per token: the top-k and the renormalized softmax weights are the host's
/// fused router kernel's (softmax mode, `mlxserve_moe_router_softmax`, `shared_heads.text_router_softmax`), computed inside
/// the launch. `logit` is the shared gate's logit per row as `moeFusedShared` takes it, or null with `gate_vec` the gate's
/// weight vector [hidden] (the logit computed in the same launch in MLX's gemv order, bf16 at 2..4 rows). The bytes of the
/// router kernel + `moeFusedShared`, one dispatch and one dependent level fewer. error.Exl3EngineNotImplemented, before any
/// launch, for a call the routed launch does not carry (the caller runs its chain): rows the gate|up launch prepares itself
/// (gu8, one row by default), the slot statistics, a geometry or dtype outside the kernel.
pub fn moeFusedSharedRouted(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, logits: mlx.mlx_array, k: u32, dec: format.Decode, shared: Bank, sigtab: mlx.mlx_array, logit: ?mlx.mlx_array, gate_vec: ?mlx.mlx_array) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const lsh = mlx.getShape(logits);
    if (xsh.len != 3 or lsh.len != 3 or xsh[0] <= 0 or xsh[1] <= 0 or xsh[0] != lsh[0] or xsh[1] != lsh[1]) return error.BadExl3Shape;
    if ((logit == null) == (gate_vec == null)) return error.BadExl3Shape;
    const rows_c: c_int = xsh[0] * xsh[1];
    const rows: usize = @intCast(rows_c);
    if (rows > shared_heads.shared_rows_max or k == 0 or k > 31) return error.Exl3EngineNotImplemented;
    const xd = mlx.mlx_array_dtype(x);
    if (mlx.mlx_array_dtype(logits) != xd) return error.Exl3EngineNotImplemented;
    const geo = try bankGeometry(bank, xsh[2]);
    if (geo.gate_up.n != geo.down.n) return error.Exl3EngineNotImplemented;
    if (lsh[2] != geo.experts or lsh[2] > 2048 or lsh[2] < @as(c_int, @intCast(k))) return error.Exl3EngineNotImplemented;
    if (mlx.mlx_array_dtype(sigtab) != xd or mlx.mlx_array_size(sigtab) != 65536) return error.Exl3EngineNotImplemented;
    if (logit) |l| {
        if (mlx.mlx_array_dtype(l) != xd or mlx.mlx_array_size(l) != rows) return error.Exl3EngineNotImplemented;
    }
    if (gate_vec) |gv| {
        // MLX's matmul of x [R, D] and [D, 1] runs gemv_bfloat16_bm1_bn8_sm1_sn32_tm{1,4}_tn4 only at R = 2..4, D > 64, D >= 16 R
        const d = xsh[2];
        if (xd != .bfloat16 or mlx.mlx_array_dtype(gv) != .bfloat16 or mlx.mlx_array_size(gv) != @as(usize, @intCast(d)) or rows < 2 or rows > 4 or d <= 64 or d < 16 * rows_c) return error.Exl3EngineNotImplemented;
    }
    const gsh = mlx.getShape(shared.gate.trellis);
    const dsh = mlx.getShape(shared.down.trellis);
    if (gsh.len != 3 or dsh.len != 3) return error.Exl3EngineNotImplemented;
    const inter_s: u32 = @intCast(gsh[1] * 16);
    const sb: shared_heads.SharedBank = .{
        .gate_t = shared.gate.trellis,
        .gate_suh = shared.gate.suh,
        .gate_svh = shared.gate.svh,
        .up_t = shared.up.trellis,
        .up_suh = shared.up.suh,
        .up_svh = shared.up.svh,
        .down_t = shared.down.trellis,
        .down_suh = shared.down.suh,
        .down_svh = shared.down.svh,
        .n_gu = @intCast(gsh[2]),
        .n_dn = @intCast(dsh[2]),
        .ks_gu = dense.Dense.ksFor(@intCast(geo.hidden), inter_s),
        .ks_dn = dense.Dense.ksFor(inter_s, @intCast(geo.hidden)),
        .defer_affine = dense.Dense.deferFor(dec),
        .sigtab = sigtab,
        // with a gate vector the merged launch's logit takes this one's place in dfin7s (never read; any array of the dtype)
        .logit = logit orelse sigtab,
    };
    const dgeo: decode.Geometry = .{ .hidden = @intCast(geo.hidden), .inter = @intCast(geo.width), .topk = k, .rate = geo.gate_up, .dec = dec };
    const k3 = try kernelsFor(dgeo, xd);
    if (!shared_heads.SharedHeads.sharedAdmitted(k3, rows, sb)) return error.Exl3EngineNotImplemented;
    const heads = try shared_heads.forKernels(k3);
    // the prepare in a launch of its own (form 9 above gu8_rows), no slot stats
    if (rows <= k3.gu8_rows and k3.gu8Admitted(rows, k3.kdiv9)) return error.Exl3EngineNotImplemented;
    if (SlotStats.enabled()) return error.Exl3EngineNotImplemented;
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows_c, xsh[2] }, 2, s));
    var l2 = mlx.mlx_array_new();
    defer freeArr(l2);
    try mlx.check(mlx.mlx_reshape(&l2, logits, &[_]c_int{ rows_c, lsh[2] }, 2, s));
    var l1 = mlx.mlx_array_new();
    defer freeArr(l1);
    var sbr = sb;
    if (logit) |l| {
        try mlx.check(mlx.mlx_reshape(&l1, l, &[_]c_int{rows_c}, 1, s));
        sbr.logit = l1;
    }
    const kb: decode.Kernels.Bank = .{
        .gate_t = bank.gate.trellis,
        .gate_suh = bank.gate.suh,
        .gate_svh = bank.gate.svh,
        .up_t = bank.up.trellis,
        .up_suh = bank.up.suh,
        .up_svh = bank.up.svh,
        .down_t = bank.down.trellis,
        .down_suh = bank.down.suh,
        .down_svh = bank.down.svh,
    };
    const y0 = try heads.moeSharedRouted(k3, s, x2, l2, kb, rows, sbr, gate_vec);
    defer freeArr(y0);
    if (!routed_logged.get().*[@intFromBool(gate_vec != null)]) {
        routed_logged.get().*[@intFromBool(gate_vec != null)] = true;
        std.debug.print("[turboexl3] router folded into form 9's prepare launch ({d} rows, {d} experts, top-{d}){s}\n", .{ rows, lsh[2], k, if (gate_vec != null) ", the shared gate's logit with it" else "" });
    }
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y0, xsh.ptr, 3, s));
    return y;
}
const routed_logged = Global([2]bool, "root_routed_logged", .{ false, false });

/// The process-wide state here (root's `shared_state` lists it)
pub const shared_state = .{ fused_shared_logged, routed_logged };
