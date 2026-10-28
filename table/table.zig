//! `turboexl3_table`: turboexl3 as an mlx-serve quant table (`abi/quant.zig`), EXL3 trellis weights on turboexl3's
//! Metal kernels. Two consumers build it in: the quant plugin in plugin/ (a dylib from the plugin kit, the server's
//! table) and a model plugin that runs the table in process behind its own format seam. Each hands it its `sdk`
//! (`mlx`, `kernel_launch`, and `quant_abi`: the quant ABI its table type comes from) and the kit's root
//! (`mlx_serve`). It serves the blocks it claims (config.zig): a fully EXL3
//! pack's `trunk_quant`, the trunk's dense linears (`load_tensor`; `matmul` on turboexl3's dense `linear`), and in the
//! reordered numerics where MLX's NAX gate holds, `expert_quant`, the routed banks (`load_bank`; `swiglu` on `moe`).
//! The server's built-in Sushi table keeps the banks otherwise. Below the gate turboexl3's NAX GEMMs stay off
//! (`Config.nax`): a dense linear's prefill rows dequantize + run MLX's matmul. The fused paths join the table as
//! optional members, as `swiglu` did (the GDN tail's `linear_*` parts, a gated attention layer's linears, the
//! decode-row merges, the shared expert inside the routed launches, the resident and un-rotated trunk), each refusing
//! by name what it does not carry: the host then runs its chain.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = sdk.quant_abi;
const sdk = @import("sdk");
const mlx = sdk.mlx;
const t3 = @import("turboexl3");
const tkit = @import("turboexl3_kit");
const nax = tkit.nax;
pub const config = @import("config.zig");

/// Shown at install and in Settings.
/// The table's name: the quant plugin's, and the format a model plugin's seam reports.
pub const table_name = "turboexl3";

pub const table = abi.QuantTable{
    .name = abi.Str.of(table_name),
    .claims = &config.claims,
    .validate = &config.validate,
    .bill = &config.bill,
    .open = &open,
    .close = &close,
    .load_tensor = &loadTensor,
    .load_bank = &loadBank,
    .release = &release,
    .matmul = &matmul,
    .moe = &moe,
    .swiglu = &swiglu,
    .linear_split = &linearSplit,
    .linear_finish = &linearFinish,
    .linear_prepared = &linearPrepared,
    .linear_pieces = &linearPieces,
    .linear_piece = &linearPiece,
    .attn_in = &attnIn,
    .attn_prep = &attnPrep,
    .linear_prepared_idx = &linearPreparedIdx,
    .gated_out = &gatedOut,
    .merged_gated_out = &mergedGatedOut,
    .q_planes = &qPlanes,
    .indexer = &indexer,
    .swiglu_shared = &swigluShared,
    .resident = &resident,
    .resident_pending = &residentPending,
    .unrotate = &unrotate,
    .linear_joined = &linearJoined,
};

const Verdict = abi.Verdict;
const Str = abi.Str;
const gpa = std.heap.c_allocator;

/// What `open` read from the claimed block: its kind and decode, a bank block's rate.
const State = struct {
    kind: config.Kind,
    dec: t3.format.Decode,
    rate: ?t3.format.Rate = null,
};

fn stateOf(ctx: *const abi.LoadCtx) ?*const State {
    return @ptrCast(@alignCast(ctx.state orelse return null));
}

fn open(ctx: *abi.LoadCtx) callconv(.c) Verdict {
    const h = kit.host.Host.of(ctx.host) catch |e| return Verdict.named(.unsupported, @errorName(e));
    tkit.bind(h.t);
    configureOnce(h);
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const b = config.parseBlock(arena.allocator(), ctx.block_json.slice()) catch |e| return Verdict.named(.unsupported, @errorName(e));
    const st = gpa.create(State) catch return Verdict.refuse(.failed, "turboexl3: out of memory");
    st.* = .{
        .kind = b.kind,
        .dec = .{
            .codebook = switch (b.codebook) {
                .mul1 => .mul1,
                .mcg => .mcg,
            },
            .window = t3.format.Window.fromBits(b.window_bits) orelse .w16,
        },
        .rate = if (b.kind == .bank) t3.format.Rate.fromN(b.n) else null,
    };
    ctx.state = st;
    return .ok;
}

fn close(ctx: *abi.LoadCtx) callconv(.c) void {
    if (ctx.state) |p| gpa.destroy(@as(*State, @ptrCast(@alignCast(p))));
    ctx.state = null;
}

/// turboexl3's configuration, once per process: the served defaults and this machine's NAX gate.
var configured = false;

fn configureOnce(h: kit.host.Host) void {
    if (configured) return;
    configured = true;
    const on = nax.available();
    t3.configure(.{ .nax = on }) catch {};
    if (!on) h.log(.info, "[turboexl3] MLX's NAX gate is off here (an M5-class GPU and macOS 26.2+): dense prefill rows dequantize + run MLX's matmul, and the routed banks stay the built-in table's\n", .{});
}

fn loadTensor(ctx: *const abi.LoadCtx, name: Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    const st = stateOf(ctx) orelse return Verdict.refuse(.failed, "turboexl3: the format is not open");
    if (st.kind != .trunk) return Verdict.refuse(.unsupported, "turboexl3: dense linears load under the trunk_quant block");
    linearInto(ctx, name.slice(), in_dim, st.dec, out) catch |e| return failed(e);
    return .ok;
}

/// `<base>.weight` (the trellis U16/I16 [in/16, out/16, n]) beside `<base>.scales` (suh f16 [in]) and `<base>.biases`
/// (svh f16 [out]): a trunk linear as the pack stores it under the affine names. The weight carries its decode:
/// `layout` the codebook, `flags` the window's bits, `block` the rate (halfwords per tile).
fn linearInto(ctx: *const abi.LoadCtx, base: []const u8, in_dim: u32, dec: t3.format.Decode, out: *abi.Weight) !void {
    const t = tensor(ctx, base, "weight");
    const suh = tensor(ctx, base, "scales");
    const svh = tensor(ctx, base, "biases");
    if (t.ctx == null or suh.ctx == null or svh.ctx == null) return error.MissingWeight;
    const sh = mlx.getShape(t);
    const dt = mlx.mlx_array_dtype(t);
    if (sh.len != 3 or (dt != .uint16 and dt != .int16) or sh[0] <= 0 or sh[1] <= 0) return error.Exl3TrellisGeometry;
    if (t3.format.Rate.fromN(@intCast(@max(sh[2], 0))) == null) return error.Exl3RateUnsupported;
    const in = sh[0] * 16;
    const width = sh[1] * 16;
    if (@rem(in, 128) != 0 or @rem(width, 128) != 0) return error.Exl3WidthNotMultipleOf128;
    if (in_dim != 0 and in_dim != @as(u32, @intCast(in))) return error.Exl3TrellisGeometry;
    if (!std.mem.eql(c_int, mlx.getShape(suh), &.{in}) or !std.mem.eql(c_int, mlx.getShape(svh), &.{width})) return error.Exl3SideGeometry;
    if (mlx.mlx_array_dtype(suh) != .float16 or mlx.mlx_array_dtype(svh) != .float16) return error.Exl3SideDtype;
    try fill(out, t, suh, svh, dec, @intCast(sh[2]));
}

fn loadBank(ctx: *const abi.LoadCtx, name: Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    const st = stateOf(ctx) orelse return Verdict.refuse(.failed, "turboexl3: the format is not open");
    if (st.kind != .bank) return Verdict.refuse(.unsupported, "turboexl3: routed banks load under the expert_quant block");
    bankInto(ctx, name.slice(), in_dim, st, out) catch |e| return failed(e);
    return .ok;
}

/// `<base>.{trellis,suh,svh}`: trellis U16 [E, in/16, out/16, n] at or under the block's rate, suh f16 [E, in], svh f16
/// [E, out] (Sushi's layout, the server's EXL3 bank names).
fn bankInto(ctx: *const abi.LoadCtx, base: []const u8, in_dim: u32, st: *const State, out: *abi.Weight) !void {
    const t = tensor(ctx, base, "trellis");
    const suh = tensor(ctx, base, "suh");
    const svh = tensor(ctx, base, "svh");
    if (t.ctx == null or suh.ctx == null or svh.ctx == null) return error.MissingWeight;
    const sh = mlx.getShape(t);
    if (sh.len != 4 or sh[0] <= 0 or sh[1] <= 0 or sh[2] <= 0) return error.Exl3TrellisGeometry;
    const experts: u32 = @intCast(sh[0]);
    const in: u32 = @intCast(sh[1] * 16);
    const width: u32 = @intCast(sh[2] * 16);
    if (in_dim != 0 and in_dim != in) return error.Exl3TrellisGeometry;
    const rate = st.rate orelse return error.Exl3RateUnsupported;
    if (mlx.mlx_array_dtype(t) != .uint16 or !t3.spec.trellisAdmitted(sh, experts, in, width, rate)) return error.Exl3TrellisGeometry;
    if (!std.mem.eql(c_int, mlx.getShape(suh), &.{ sh[0], sh[1] * 16 }) or !std.mem.eql(c_int, mlx.getShape(svh), &.{ sh[0], sh[2] * 16 })) return error.Exl3SideGeometry;
    if (mlx.mlx_array_dtype(suh) != .float16 or mlx.mlx_array_dtype(svh) != .float16) return error.Exl3SideDtype;
    try fill(out, t, suh, svh, st.dec, @intCast(sh[3]));
}

/// `out` over new handles on the three arrays (the source keeps its own), with their decode.
fn fill(out: *abi.Weight, t: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, dec: t3.format.Decode, n: u32) !void {
    out.* = .{};
    errdefer release(out);
    out.w = try share(t);
    out.scales = try share(suh);
    out.biases = try share(svh);
    out.layout = switch (dec.codebook) {
        .mul1 => 0,
        .mcg => 1,
    };
    out.flags = dec.window.bits();
    out.block = n;
}

fn release(w: *abi.Weight) callconv(.c) void {
    for ([_]abi.Array{ w.w, w.scales, w.biases, w.side }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(.{ .ctx = a.ctx });
    };
    w.* = .{};
}

fn matmul(w: *const abi.Weight, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const dec = decodeOf(w) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
    const y = t3.linear(.{ .ctx = s.ctx }, .{ .ctx = x.ctx }, arr(w.w), arr(w.scales), arr(w.biases), dec) catch |e| return failed(e);
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

fn moe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "turboexl3: a layer's banks run as one routed SwiGLU (swiglu)");
}

fn swiglu(gate: *const abi.Weight, up: *const abi.Weight, down: *const abi.Weight, x: abi.Array, inds: abi.Array, scores: abi.Array, opts: *const abi.Swiglu, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    if (opts.limit > 0) return Verdict.refuse(.unsupported, "turboexl3: no clamped SwiGLU kernels");
    const dec = decodeOf(gate) orelse return Verdict.refuse(.failed, "turboexl3: a bank without its decode");
    if (!std.meta.eql(decodeOf(up), dec) or !std.meta.eql(decodeOf(down), dec)) return Verdict.refuse(.unsupported, "turboexl3: a layer's banks decode differently");
    const bank: t3.Bank = .{ .gate = projOf(gate), .up = projOf(up), .down = projOf(down) };
    const y = t3.moe(.{ .ctx = s.ctx }, arr(x), bank, arr(inds), arr(scores), dec, opts.verify_rows) catch |e| return failed(e);
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

// ── The GDN tail's parts (the table's `linear_*` entries): turboexl3's `linear` in the pieces the host's fused GDN tail
// runs between, each one `linear`'s own launches; an un-rotated linear has no rotated parts and refuses. ──

fn linearSplit(w: *const abi.Weight, x: abi.Array, opts: *const abi.LinearParts, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const dec = decodeOf(w) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
    if (t3.dense.residentUnrot(arr(w.w))) return Verdict.refuse(.unsupported, "turboexl3: an un-rotated linear has no rotated parts");
    const main: mlx.mlx_stream = .{ .ctx = s.ctx };
    const side: mlx.mlx_stream = if (opts.side.ctx != null) .{ .ctx = opts.side.ctx } else main;
    const y = (if (opts.inner)
        t3.linearInnerSplit(main, side, arr(x), projOf(w), dec)
    else
        t3.linearSplit(main, side, arr(x), arr(w.w), arr(w.scales), arr(w.biases), dec)) catch |e| return failed(e);
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

fn linearFinish(w: *const abi.Weight, opts: *const abi.LinearParts, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const dec = decodeOf(w) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
    const od = dtypeOf(opts.dtype) orelse return Verdict.refuse(.unsupported, "turboexl3: an output dtype it does not write");
    const parts = opts.parts.slice();
    var inners: [t3.dense.Dense.max_pieces]mlx.mlx_array = undefined;
    if (parts.len < 1 or parts.len > inners.len) return Verdict.refuse(.unsupported, "turboexl3: a finish over 1..4 inner products");
    for (parts, inners[0..parts.len]) |p, *o| o.* = arr(p);
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const y = (if (parts.len == 1)
        t3.linearFinish(st, inners[0], projOf(w), dec, od)
    else
        t3.linearPiecesFinish(st, inners[0..parts.len], projOf(w), dec, od)) catch |e| return failed(e);
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

fn linearPrepared(w: *const abi.Weight, xp: abi.Array, opts: *const abi.LinearParts, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const dec = decodeOf(w) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
    if (t3.dense.residentUnrot(arr(w.w))) return Verdict.refuse(.unsupported, "turboexl3: an un-rotated linear has no rotated parts");
    const od = dtypeOf(opts.dtype) orelse return Verdict.refuse(.unsupported, "turboexl3: an output dtype it does not write");
    const y = t3.linearPrepared(.{ .ctx = s.ctx }, arr(xp), projOf(w), dec, od) catch |e| return failed(e);
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

fn linearPieces(w: *const abi.Weight, opts: *const abi.LinearParts, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const dec = decodeOf(w) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
    if (t3.dense.residentUnrot(arr(w.w))) return Verdict.refuse(.unsupported, "turboexl3: an un-rotated linear has no rotated parts");
    const xd = dtypeOf(opts.dtype) orelse return Verdict.refuse(.unsupported, "turboexl3: an activation dtype it does not read");
    const rows = opts.rows.slice();
    var buf: [t3.dense.Dense.max_pieces]usize = undefined;
    if (rows.len > buf.len) return Verdict.refuse(.unsupported, "turboexl3: more pieces than its finish reads");
    for (rows, buf[0..rows.len]) |r, *o| o.* = r;
    const W = t3.linearPiecesBegin(.{ .ctx = s.ctx }, projOf(w), buf[0..rows.len], xd, dec) catch |e| return failed(e);
    out.* = .{ .ctx = W.ctx };
    return .ok;
}

fn linearPiece(w: *const abi.Weight, x: abi.Array, opts: *const abi.LinearParts, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const dec = decodeOf(w) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
    if (opts.held.ctx == null) return Verdict.refuse(.failed, "turboexl3: a piece without the pieces' weight");
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const y = (if (opts.prepared) blk: {
        const xd = dtypeOf(opts.dtype) orelse return Verdict.refuse(.unsupported, "turboexl3: an activation dtype it does not read");
        break :blk t3.linearPiecePrepared(st, arr(x), arr(opts.held), projOf(w), dec, xd);
    } else t3.linearPieceInner(st, arr(x), arr(opts.held), projOf(w), dec)) catch |e| return failed(e);
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

/// mlx-c's dtype value `v` (the table's i32), where turboexl3 reads or writes it.
fn dtypeOf(v: i32) ?mlx.mlx_dtype {
    const d = std.enums.fromInt(mlx.mlx_dtype, v) orelse return null;
    return switch (d) {
        .bfloat16, .float16, .float32 => d,
        else => null,
    };
}

// ── the fused entries ──

fn attnIn(q: *const abi.Weight, k: *const abi.Weight, v: *const abi.Weight, x: abi.Array, opts: *const abi.AttnIn, s: abi.Stream, out: *abi.AttnInOut) callconv(.c) Verdict {
    if (opts.size < @sizeOf(abi.AttnIn)) return Verdict.refuse(.unsupported, "turboexl3: an AttnIn without the fields it reads");
    const ep: t3.attn_io.Epilogue = .{
        .heads = opts.heads,
        .kv_heads = opts.kv_heads,
        .rope_dims = opts.rope_dims,
        .q_norm = arr(opts.q_norm),
        .k_norm = arr(opts.k_norm),
        .eps = arr(opts.eps),
        .angles = arr(opts.angles),
        .prep = if (opts.prep[0].ctx != null) .{ arr(opts.prep[0]), arr(opts.prep[1]), arr(opts.prep[2]) } else null,
    };
    const r = attnInOf(q, k, v, arr(x), opts, ep, .{ .ctx = s.ctx }) catch |e| return failed(e);
    out.* = .{ .q = .{ .ctx = r.q.ctx }, .gate = .{ .ctx = r.gate.ctx }, .k = .{ .ctx = r.k.ctx }, .v = .{ .ctx = r.v.ctx } };
    return .ok;
}

/// The un-rotated trunk's epilogue over the joined product the host made, or over each member's own plain matmul where
/// all three are un-rotated (past the rows the trellis paths keep); else ATTN-IN over the rotated linears.
fn attnInOf(q: *const abi.Weight, k: *const abi.Weight, v: *const abi.Weight, x: mlx.mlx_array, opts: *const abi.AttnIn, ep: t3.attn_io.Epilogue, s: mlx.mlx_stream) !t3.attn_io.InOut {
    if (opts.std.ctx != null) {
        const j = arr(opts.std);
        return t3.attnInStd(s, .{ j, j, j }, opts.std_offs, ep);
    }
    const dec = sameDecode(&.{ q, k, v }) orelse return error.Exl3DecodesDiffer;
    if (rowsOf(x) > t3.dense.unrot_min_rows and unrotated(q) and unrotated(k) and unrotated(v)) {
        var own: [3]mlx.mlx_array = @splat(.{});
        defer for (own) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        for (&own, [_]*const abi.Weight{ q, k, v }) |*o, w| o.* = try t3.linear(s, x, arr(w.w), arr(w.scales), arr(w.biases), dec);
        return t3.attnInStd(s, own, .{ 0, 0, 0 }, ep);
    }
    return t3.attnIn(s, x, projOf(q), projOf(k), projOf(v), ep, dec);
}

fn attnPrep(q: *const abi.Weight, k: *const abi.Weight, v: *const abi.Weight, w4: *const abi.Weight, x: abi.Array, s: abi.Stream, out: [*]abi.Array) callconv(.c) Verdict {
    // an un-rotated fourth linear has no rotated copy to read (its plain matmul serves it)
    if (unrotated(w4)) return Verdict.refuse(.unsupported, "turboexl3: the fourth linear is un-rotated");
    const dec = sameDecode(&.{ q, k, v, w4 }) orelse return Verdict.refuse(.unsupported, "turboexl3: the four linears decode differently");
    const xp = t3.attnPrep4(.{ .ctx = s.ctx }, arr(x), projOf(q), projOf(k), projOf(v), projOf(w4), dec) catch |e| return failed(e);
    for (out[0..4], xp) |*o, p| o.* = .{ .ctx = p.ctx };
    return .ok;
}

fn linearPreparedIdx(w: *const abi.Weight, xp: abi.Array, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const dec = decodeOf(w) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const y2 = t3.linearPreparedIdx(st, arr(xp), projOf(w), mlx.mlx_array_dtype(arr(x)), dec) catch |e| return failed(e);
    defer _ = mlx.mlx_array_free(y2);
    // y [rows, out] back to x's leading dims
    const xsh = mlx.getShape(arr(x));
    var sh: [8]c_int = undefined;
    if (xsh.len < 1 or xsh.len > sh.len) return Verdict.refuse(.unsupported, "turboexl3: x's rank");
    @memcpy(sh[0..xsh.len], xsh);
    sh[xsh.len - 1] = mlx.getShape(y2)[1];
    var y = mlx.mlx_array_new();
    mlx.check(mlx.mlx_reshape(&y, y2, &sh, xsh.len, st)) catch |e| {
        _ = mlx.mlx_array_free(y);
        return failed(e);
    };
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

/// Decode rows: the merge with o_proj's GEMV; wider: ATTN-OUT (an un-rotated o_proj's: its plain matmul over the product).
fn gatedOut(o: *const abi.Weight, attn: abi.Array, gate: *const abi.Gate, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    if (gate.size < @sizeOf(abi.Gate)) return Verdict.refuse(.unsupported, "turboexl3: a Gate without the fields it reads");
    const sh = mlx.getShape(arr(attn));
    if (sh.len != 4) return Verdict.refuse(.unsupported, "turboexl3: attn is not [1, heads, rows, head_dim]");
    const rows: usize = @intCast(@max(sh[2], 0));
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const y = blk: {
        if (rows <= t3.kernels.DECODE_ROWS_MAX) {
            const dec = decodeOf(o) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
            break :blk t3.decodeMergeGated(st, arr(attn), mergeGate(gate), arr(gate.sigtab), projOf(o), dec) catch |e| return failed(e);
        }
        if (rows > t3.dense.unrot_min_rows and unrotated(o)) break :blk t3.linearGatedStd(st, arr(attn), ioGate(gate), arr(gate.sigtab), projOf(o)) catch |e| return failed(e);
        const dec = decodeOf(o) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
        break :blk t3.linearGated(st, arr(attn), ioGate(gate), arr(gate.sigtab), projOf(o), dec) catch |e| return failed(e);
    };
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

fn mergedGatedOut(o: *const abi.Weight, pacc: abi.Array, pml: abi.Array, gate: *const abi.Gate, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    if (gate.size < @sizeOf(abi.Gate)) return Verdict.refuse(.unsupported, "turboexl3: a Gate without the fields it reads");
    const dec = decodeOf(o) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
    const y = t3.decodeMergeQmergeGated(.{ .ctx = s.ctx }, arr(pacc), arr(pml), mergeGate(gate), arr(gate.sigtab), projOf(o), dec) catch |e| return failed(e);
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

fn qPlanes(q: *const abi.Weight, x: abi.Array, opts: *const abi.QPlanes, s: abi.Stream, out: [*]abi.Array) callconv(.c) Verdict {
    if (opts.size < @sizeOf(abi.QPlanes)) return Verdict.refuse(.unsupported, "turboexl3: a QPlanes without the fields it reads");
    const dec = decodeOf(q) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
    const planes = t3.decodeMergeQPlanes(.{ .ctx = s.ctx }, arr(x), projOf(q), opts.heads, opts.head_dim, dec) catch |e| return failed(e);
    for (out[0..2], planes) |*o, p| o.* = .{ .ctx = p.ctx };
    return .ok;
}

fn indexer(idx: *const abi.Weight, x: abi.Array, opts: *const abi.Indexer, s: abi.Stream, out: [*]abi.Array) callconv(.c) Verdict {
    if (opts.size < @sizeOf(abi.Indexer)) return Verdict.refuse(.unsupported, "turboexl3: an Indexer without the fields it reads");
    const dec = decodeOf(idx) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
    // the inputs a variant does not read stay bound (any array of their dtype): the norm weights stand in
    const r = t3.decodeMergeIndexer(.{ .ctx = s.ctx }, arr(x), projOf(idx), .{
        .heads = opts.heads,
        .rope_dims = opts.rope_dims,
        .ratio = opts.ratio,
        .qepi = opts.q_rope,
        .q_norm = arr(opts.q_norm),
        .angles = arr(if (opts.angles.ctx != null) opts.angles else opts.q_norm),
        .pool = opts.pool,
        .k_norm = arr(opts.k_norm),
        .ring = arr(if (opts.ring.ctx != null) opts.ring else opts.k_norm),
        .ln = opts.ln,
        .pcos = arr(if (opts.pcos.ctx != null) opts.pcos else opts.k_norm),
        .psin = arr(if (opts.psin.ctx != null) opts.psin else opts.k_norm),
        .eps = opts.eps,
    }, dec) catch |e| return failed(e);
    out[0..3].* = .{ .{ .ctx = r.qk.ctx }, .{ .ctx = r.q_rope.ctx }, .{ .ctx = r.pooled.ctx } };
    return .ok;
}

fn swigluShared(gate: *const abi.Weight, up: *const abi.Weight, down: *const abi.Weight, x: abi.Array, inds: abi.Array, scores: abi.Array, opts: *const abi.SharedSwiglu, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    if (opts.size < @sizeOf(abi.SharedSwiglu)) return Verdict.refuse(.unsupported, "turboexl3: a SharedSwiglu without the fields it reads");
    const dec = sameDecode(&.{ gate, up, down, opts.gate, opts.up, opts.down }) orelse return Verdict.refuse(.unsupported, "turboexl3: the banks and the shared expert decode differently");
    const bank: t3.Bank = .{ .gate = projOf(gate), .up = projOf(up), .down = projOf(down) };
    const shared: t3.Bank = .{ .gate = projOf(opts.gate), .up = projOf(opts.up), .down = projOf(opts.down) };
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const logit: ?mlx.mlx_array = if (opts.logit.ctx != null) arr(opts.logit) else null;
    const y = if (opts.logits.ctx != null)
        t3.moeFusedSharedRouted(st, arr(x), bank, arr(opts.logits), opts.k, dec, shared, arr(opts.sigtab), logit, if (opts.gate_vec.ctx != null) arr(opts.gate_vec) else null) catch |e| return failed(e)
    else
        t3.moeFusedShared(st, arr(x), bank, arr(inds), arr(scores), dec, shared, arr(opts.sigtab), logit orelse return Verdict.refuse(.unsupported, "turboexl3: the shared gate's logit is not given")) catch |e| return failed(e);
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

fn mergeGate(g: *const abi.Gate) t3.decode_merge.Gate {
    return .{ .base = arr(g.base), .row_stride = g.row_stride, .head_stride = g.head_stride, .off = g.off };
}

fn ioGate(g: *const abi.Gate) t3.attn_io.Gate {
    return .{ .base = arr(g.base), .row_stride = g.row_stride, .head_stride = g.head_stride, .off = g.off };
}

/// The one decode the weights share, or null.
fn sameDecode(ws: []const *const abi.Weight) ?t3.format.Decode {
    const dec = decodeOf(ws[0]) orelse return null;
    for (ws[1..]) |w| if (!std.meta.eql(decodeOf(w), dec)) return null;
    return dec;
}

/// The resident table holds the linear in the standard basis (the un-rotated trunk).
fn unrotated(w: *const abi.Weight) bool {
    return t3.dense.residentUnrot(arr(w.w));
}

fn rowsOf(x: mlx.mlx_array) usize {
    const sh = mlx.getShape(x);
    var rows: usize = 1;
    if (sh.len > 1) for (sh[0 .. sh.len - 1]) |d| {
        rows *= @intCast(@max(d, 0));
    };
    return rows;
}

// ── The resident trunk and its standard-basis copy (turboexl3's dense.zig `resident*`, root.zig `trunkUnrot`) ──

/// The resident table over `ws`' trellises (each W_hat kept at its first prefill call); an empty `ws` drops it. One table
/// for the process: a second model's arm replaces the first's.
fn resident(ws: abi.Span(abi.Weight), room: u64, _: abi.Stream, bytes: *u64) callconv(.c) Verdict {
    bytes.* = 0;
    const list = ws.slice();
    if (list.len == 0) {
        t3.dense.residentRelease();
        return .ok;
    }
    const ts = gpa.alloc(mlx.mlx_array, list.len) catch return Verdict.refuse(.failed, "turboexl3: out of memory");
    defer gpa.free(ts);
    const bs = gpa.alloc(u64, list.len) catch return Verdict.refuse(.failed, "turboexl3: out of memory");
    defer gpa.free(bs);
    var total: u64 = 0;
    for (list, ts, bs) |w, *t, *b| {
        t.* = arr(w.w);
        b.* = t3.dense.residentBytesOf(mlx.getShape(t.*));
        if (b.* == 0) return Verdict.refuse(.unsupported, "turboexl3: a resident weight that is not a trunk trellis");
        total += b.*;
    }
    bytes.* = total;
    if (room != 0 and total > room) return Verdict.refuse(.too_large, "turboexl3: the decoded matrices do not fit the room");
    t3.dense.residentArm(ts, bs) catch |e| return failed(e);
    return .ok;
}

fn residentPending() callconv(.c) u64 {
    return t3.dense.residentPending();
}

/// The resident groups out of the Hadamard basis, joined per group (`t3.trunkUnrot`), in bf16 or f16.
fn unrotate(ws: abi.Span(abi.Weight), sizes: abi.Span(u8), act: i32, s: abi.Stream, took: [*]bool) callconv(.c) Verdict {
    const list = ws.slice();
    const n = sizes.slice();
    if (list.len == 0) return .ok;
    const dt: mlx.mlx_dtype = switch (act) {
        @intFromEnum(mlx.mlx_dtype.bfloat16) => .bfloat16,
        @intFromEnum(mlx.mlx_dtype.float16) => .float16,
        else => return Verdict.refuse(.unsupported, "turboexl3: the standard-basis copy is bf16 or f16"),
    };
    const dec = decodeOf(&list[0]) orelse return Verdict.refuse(.failed, "turboexl3: a weight without its decode");
    const projs = gpa.alloc(t3.Proj, list.len) catch return Verdict.refuse(.failed, "turboexl3: out of memory");
    defer gpa.free(projs);
    for (list, projs) |*w, *p| {
        if (!std.meta.eql(decodeOf(w), dec)) return Verdict.refuse(.unsupported, "turboexl3: the trunk's linears decode differently");
        p.* = projOf(w);
    }
    _ = t3.trunkUnrot(.{ .ctx = s.ctx }, projs, n, dec, dt, took[0..n.len]) catch |e| return failed(e);
    return .ok;
}

/// `x` times the joined W_std of the group whose members are `ws`, in its column order; refused past the group's members,
/// out of order, on another dtype, and at decode widths (the trellis decode path serves those).
fn linearJoined(ws: abi.Span(abi.Weight), x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const list = ws.slice();
    if (list.len < 2) return Verdict.refuse(.unsupported, "turboexl3: a join has two members or more");
    const xs = arr(x);
    const sh = mlx.getShape(xs);
    var rows: usize = 1;
    if (sh.len > 1) for (sh[0 .. sh.len - 1]) |d| {
        rows *= @intCast(d);
    };
    if (rows <= t3.dense.unrot_min_rows) return Verdict.refuse(.unsupported, "turboexl3: decode widths keep the trellis path");
    var jw: mlx.mlx_array = .{ .ctx = null };
    defer if (jw.ctx != null) {
        _ = mlx.mlx_array_free(jw);
    };
    var group: u32 = 0;
    var col: u32 = 0;
    for (list, 0..) |*w, i| {
        const j = t3.dense.residentJoined(arr(w.w)) orelse return Verdict.refuse(.unsupported, "turboexl3: a member is not un-rotated");
        if (i == 0) {
            jw = j.w;
            group = j.group;
        } else _ = mlx.mlx_array_free(j.w);
        if (j.group != group or j.col != col) return Verdict.refuse(.unsupported, "turboexl3: the members are joined otherwise");
        col += j.width;
    }
    const jsh = mlx.getShape(jw);
    if (jsh.len != 2 or jsh[1] != @as(c_int, @intCast(col))) return Verdict.refuse(.unsupported, "turboexl3: the members are joined otherwise");
    if (mlx.mlx_array_dtype(xs) != mlx.mlx_array_dtype(jw)) return Verdict.refuse(.unsupported, "turboexl3: x is not the joined matrix's dtype");
    var y = mlx.mlx_array_new();
    mlx.check(mlx.mlx_matmul(&y, xs, jw, .{ .ctx = s.ctx })) catch |e| {
        _ = mlx.mlx_array_free(y);
        return failed(e);
    };
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

/// The decode a loaded weight carries (`fill`).
fn decodeOf(w: *const abi.Weight) ?t3.format.Decode {
    const codebook: t3.format.Codebook = switch (w.layout) {
        0 => .mul1,
        1 => .mcg,
        else => return null,
    };
    return .{ .codebook = codebook, .window = t3.format.Window.fromBits(w.flags) orelse return null };
}

fn projOf(w: *const abi.Weight) t3.Proj {
    return .{ .trellis = arr(w.w), .suh = arr(w.scales), .svh = arr(w.biases) };
}

fn tensor(ctx: *const abi.LoadCtx, base: []const u8, leaf: []const u8) mlx.mlx_array {
    var nb: [320]u8 = undefined;
    const name = std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, leaf }) catch return .{};
    return arr(ctx.source.tensor(name));
}

fn share(a: mlx.mlx_array) !abi.Array {
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_array_set(&h, a));
    return .{ .ctx = h.ctx };
}

fn arr(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}

fn failed(e: anyerror) Verdict {
    return Verdict.named(if (e == error.MissingWeight) .failed else .unsupported, @errorName(e));
}
