//! Ling-3.1-flash's MLA attention at decode rows in the latent (`decodeLatent`, root's `ling.mlaLatentDecode`): the
//! absorbed form of DeepSeek-V3's MLA over a cache of each token's normed KV latent (DL 512) and roped shared key
//! (DR 64), in three launches a layer, every value in f32 from the exact operands:
//! 1. `turboexl3_ling_mla_qabs`: the query's non-positional half through kv_b's key half into the latent,
//!    ql[s, h] = q_nope[h, s] . Wk[h] (DN 128 terms, d ascending);
//! 2. `turboexl3_ling_mla_attn`: the attention's partials, T cut into key groups of `kpt` keys counted from position 0,
//!    one threadgroup a (key group, 32 heads of a query row): each sub-chunk of `ck` keys scores the 32 rows on f32
//!    simdgroup matrices (ql . latent, then q_rope . rope key, times the scale), updates each row's online softmax and
//!    adds its weights times the latents into the row's 512 sums;
//! 3. `turboexl3_ling_mla_vabs`: a row's key-group partials merged in group order, then the latent output through kv_b's
//!    value half (DV 128 outputs).
//! A query row's values depend on its own operands and its key position only (the key tiles and groups are counted
//! from position 0, a key past the row's position reads as 0 and scores -inf, the merge takes the groups holding a key
//! it sees, in order): a verify call's row has the bits of a one-row call at its position, so an exact verify can run
//! its rows in one call. Against the host's composed latent attention (MLX's matmuls and softmax) the sums run in other
//! orders: within tolerance, not byte for byte (`mla_latent_test.zig`).
//!
//! Launch 2 has three variants (`Attn`), each a host's opt-in (`Config.mla_latent_rows`, `mla_latent_stage`,
//! `mla_latent_block`) and each launch 2's words, every tile at its place and every product and sum in its order
//! (`mla_latent_test.zig`, word for word):
//! - `turboexl3_ling_mla_attn_rows`: one threadgroup a (key group, 32 heads) for 2 or 4 rows of a call (`rows2`,
//!   `rows4`), the group's latents and rope keys read once for those rows, each row's softmax and sums its own;
//! - `turboexl3_ling_mla_attn_stage`: each sub-chunk's latents and rope keys converted to f32 once a threadgroup, in
//!   threadgroup memory, and every simdgroup's tiles read there;
//! - `turboexl3_ling_mla_attn_block`: 2 x 2 score tiles a simdgroup over 4 sub-chunks at a time and 4 x 4 output tiles,
//!   so each word a simdgroup reads serves more tiles.
//! One more host opt-in (`Config.mla_latent_nax`) runs launch 2 on the matrix units, not with launch 2's words
//! (`turboexl3_ling_mla_attn_nax`, `Attn.nax_*`): MLX steel's NAX tiles (16 x 32 x 16 multiply-accumulates into f32)
//! over the query rounded to the cache's dtype and the cache's bf16 or f16 words, 16 heads of a row an M tile, each M
//! tile's 4 simdgroups a quarter of the latent's columns, the online softmax in f32. Its partials are launch 2's
//! (pm, pl, po, the same key groups from position 0), so launch 3 merges them unchanged. A tolerance against the f64
//! reference, not launch 2's words (`mla_latent_test.zig`): the plugin serves it only to the MTP head, whose words
//! move the drafts and never the verified text.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const decode = @import("../../core/decode.zig");
const config = @import("../../core/config.zig");
const steel_nax = @import("../../core/steel.zig").nax;

fn freeArr(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

/// Ling-3.1-flash's MLA widths: the query's and the expanded key's non-positional half (DN), the rope half (DR), the
/// value (DV), the latent (DL, kv_lora_rank). kv_b [H (DN + DV), DL]: head h's key half rows h (DN + DV) + d, d < DN, its
/// value half the next DV.
pub const dn: u32 = 128;
pub const dr: u32 = 64;
pub const dv: u32 = 128;
pub const dl: u32 = 512;
/// Keys a launch-2 threadgroup takes (a key group, counted from position 0) and a sub-chunk's keys (4 simdgroups' 8-key
/// tiles; the softmax update's 16 lanes of 2 keys): part of the arithmetic, so fixed
pub const kpt: u32 = 256;
pub const ck: u32 = 32;
/// Heads a launch-2 threadgroup takes: 4 M tiles of 8 rows, each across 4 simdgroups (16 simdgroups, 512 threads)
pub const heads_per_group: u32 = 32;
pub const attn_threads: u32 = 512;
/// The most query rows a call takes (the host's widest verify)
pub const max_rows: u32 = 16;

comptime {
    if (kpt % ck != 0 or ck != 32 or dl != 4 * 128 or dr % 8 != 0 or dv % 4 != 0 or dl % 32 != 0) @compileError("mla_latent: the widths the texts assume");
}

comptime {
    // the variants' texts: the stage text stages the values a column a thread (DL threads), the scores' keys in slabs of
    // 64 dims (the rope key one slab, CK keys of 64 dims 4 words a thread) in a buffer that holds 8 keys of DL + 4 too;
    // the block text runs 16 simdgroups of 32 columns, 2 x 2 score tiles over 4 sub-chunks (16 key tiles) and 4 M tiles
    if (dl != attn_threads or dr != 64 or dl % 64 != 0 or (ck * 64) % attn_threads != 0 or 2 * ck * 68 < 8 * (dl + 4) or
        attn_threads != 16 * 32 or dl != 16 * 32 or heads_per_group != 4 * 8 or 4 * ck != 16 * 8) @compileError("mla_latent: the widths the variants' texts assume");
}

// ── texts ──

/// The header: the simdgroup matrices launch 2 multiplies on
pub const text_header: [:0]const u8 = @embedFile("kernels/mla_latent/header.metal");
/// Launch 1, the query absorbed: ql f32 [S, H, DL]
pub const text_qabs: [:0]const u8 = @embedFile("kernels/mla_latent/qabs.metal");
/// Launch 2's pieces: its head and each sub-chunk's scores (A), the online softmax update of each row (B, the variants'
/// too), the weights times the latents (C) and the loop's end, the outputs (the stage variant's too)
pub const attn_1 = @embedFile("kernels/mla_latent/attn_1.metal");
pub const attn_2 = @embedFile("kernels/mla_latent/attn_2.metal");
pub const attn_3 = @embedFile("kernels/mla_latent/attn_3.metal");
pub const attn_4 = @embedFile("kernels/mla_latent/attn_4.metal");
/// Launch 2, the attention's partials: pm, pl f32 [S H NJ], po f32 [S H NJ, DL]
pub const text_attn: [:0]const u8 = attn_1 ++ attn_2 ++ attn_3 ++ attn_4;
/// The rows variant's pieces around launch 2's softmax update: its head and every row's scores; each row's weights times
/// the latents, the loop's end and every row's outputs
pub const rows_1 = @embedFile("kernels/mla_latent/rows_1.metal");
pub const rows_2 = @embedFile("kernels/mla_latent/rows_2.metal");
/// Launch 2 for RPT rows a threadgroup (`Attn.rows2`, `rows4`): launch 2's partials, each row's words
pub const text_attn_rows: [:0]const u8 = rows_1 ++ attn_2 ++ rows_2;
/// The stage variant's pieces around launch 2's softmax update and outputs: its head and the scores from staged slabs;
/// the weights times the staged latents and the loop's end
pub const stage_1 = @embedFile("kernels/mla_latent/stage_1.metal");
pub const stage_2 = @embedFile("kernels/mla_latent/stage_2.metal");
/// Launch 2 over f32 staged in threadgroup memory (`Attn.stage`): launch 2's partials, word for word
pub const text_attn_stage: [:0]const u8 = stage_1 ++ attn_2 ++ stage_2 ++ attn_4;
/// The block variant's pieces around launch 2's softmax update: its head and a block's scores on 2 x 2 tiles; the 4 x 4
/// output tiles, the loops' ends and the outputs
pub const block_1 = @embedFile("kernels/mla_latent/block_1.metal");
pub const block_2 = @embedFile("kernels/mla_latent/block_2.metal");
/// Launch 2 on more tiles a simdgroup (`Attn.block`): launch 2's partials, word for word
pub const text_attn_block: [:0]const u8 = block_1 ++ attn_2 ++ block_2;
/// Launch 3, the merge and the value half: out OT [1, H, S, DV]
pub const text_vabs: [:0]const u8 = @embedFile("kernels/mla_latent/vabs.metal");
/// Launch 2 on the matrix units (`Attn.nax_*`): its header piece after MLX steel's NAX tiles (the tile row ops in
/// natural units, the finite minimum, the threads-per-threadgroup attribute), its body, and the close that undefines the
/// attribute
pub const nax_head = @embedFile("kernels/mla_latent/nax_head.metal");
pub const nax_body = @embedFile("kernels/mla_latent/nax.metal");
pub const nax_close = @embedFile("kernels/mla_latent/nax_close.metal");
/// `turboexl3_ling_mla_attn_nax`'s header: MLX steel's NAX tiles (core/steel.zig), then its own piece
pub const header_nax: [:0]const u8 = sentinel(steel_nax ++ nax_head);
/// Launch 2 on the matrix units: launch 2's partials, pm, pl f32 [S H NJ], po f32 [S H NJ, DL]
pub const text_attn_nax: [:0]const u8 = nax_body ++ nax_close;

fn sentinel(comptime s: []const u8) [:0]const u8 {
    comptime {
        var buf: [s.len:0]u8 = undefined;
        @memcpy(buf[0..s.len], s);
        const final = buf;
        return &final;
    }
}

pub const qabs_in = [_][*:0]const u8{ "qn", "w" };
pub const qabs_out = [_][*:0]const u8{"ql"};
pub const attn_in = [_][*:0]const u8{ "ql", "qr", "lat", "rot", "scale", "dims" };
pub const attn_out = [_][*:0]const u8{ "pm", "pl", "po" };
/// The NAX launch's operands: launch 2's, the query halves in the cache's dtype (qlb, qrb)
pub const attn_nax_in = [_][*:0]const u8{ "qlb", "qrb", "lat", "rot", "scale", "dims" };
pub const vabs_in = [_][*:0]const u8{ "pm", "pl", "po", "w", "dims" };
pub const vabs_out = [_][*:0]const u8{"out"};

// ── kernels and configs: made at their first call ──

const Kind = enum(u8) { qabs, attn, vabs };
const kernel_lock = Global(std.atomic.Value(u32), "ling_mla_latent_lock", std.atomic.Value(u32).init(0));
const kernels = Global([3]?mlx.mlx_fast_metal_kernel, "ling_mla_latent_kernels", @splat(null));

fn lock() void {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}
fn unlock() void {
    kernel_lock.get().*.store(0, .release);
}

fn kernelOf(k: Kind) !mlx.mlx_fast_metal_kernel {
    lock();
    defer unlock();
    const slot = &kernels.get().*[@intFromEnum(k)];
    if (slot.*) |made| return made;
    // launch 2 reads the cache views through their strides (ql and qr are made contiguous before it)
    const made = switch (k) {
        .qabs => ks.newKernel("turboexl3_ling_mla_qabs", &qabs_in, &qabs_out, text_qabs, text_header, true),
        .attn => ks.newKernel("turboexl3_ling_mla_attn", &attn_in, &attn_out, text_attn, text_header, false),
        .vabs => ks.newKernel("turboexl3_ling_mla_vabs", &vabs_in, &vabs_out, text_vabs, text_header, true),
    } orelse return error.KernelCreateFailed;
    slot.* = made;
    return made;
}

/// Launch 2's text a call runs (`attnFor`, or a test's choice): launch 2 as it is, or a variant with its words: 2 or 4
/// rows a threadgroup (`text_attn_rows` at RPT 2 or 4; a call of fewer rows takes them all), its latents and rope keys
/// staged in f32 (`text_attn_stage`), more tiles a simdgroup (`text_attn_block`); or launch 2 on the matrix units, not
/// with its words (`text_attn_nax`, each `nax_*` a launch shape: `naxOf`)
pub const Attn = enum(u8) { base, rows2, rows4, stage, block, nax_m1_k32, nax_m1_k64, nax_m2_k32, nax_m2_k64, nax_m4_k32, nax_m4_k64, nax_sd_m1_k32, nax_sd_m1_k64, nax_sd_m2_k32 };

/// A NAX launch shape: M tiles of 16 heads a threadgroup (`mt`, each M tile's 4 simdgroups a quarter of the latent's
/// columns), the keys a step (`bk`), and whether an M tile's 4 simdgroups split the scores' 36 slices and sum their
/// partial tiles in threadgroup memory (`sd`) or each score every slice
pub const Nax = struct { mt: u32, bk: u32, sd: bool };

/// `a`'s NAX launch shape (null for launch 2 and its variants with its words). A split shape's partial tiles take
/// 4 x 16 x (bk + 4) floats an M tile of threadgroup memory, so the split shapes stay under the 32 KiB a threadgroup
/// holds (at most 18 KiB).
pub fn naxOf(a: Attn) ?Nax {
    return switch (a) {
        .base, .rows2, .rows4, .stage, .block => null,
        .nax_m1_k32 => .{ .mt = 1, .bk = 32, .sd = false },
        .nax_m1_k64 => .{ .mt = 1, .bk = 64, .sd = false },
        .nax_m2_k32 => .{ .mt = 2, .bk = 32, .sd = false },
        .nax_m2_k64 => .{ .mt = 2, .bk = 64, .sd = false },
        .nax_m4_k32 => .{ .mt = 4, .bk = 32, .sd = false },
        .nax_m4_k64 => .{ .mt = 4, .bk = 64, .sd = false },
        .nax_sd_m1_k32 => .{ .mt = 1, .bk = 32, .sd = true },
        .nax_sd_m1_k64 => .{ .mt = 1, .bk = 64, .sd = true },
        .nax_sd_m2_k32 => .{ .mt = 2, .bk = 32, .sd = true },
    };
}

/// Every NAX launch shape (TIME's arms)
pub const nax_shapes = [_]Attn{ .nax_m1_k32, .nax_m1_k64, .nax_m2_k32, .nax_m2_k64, .nax_m4_k32, .nax_m4_k64, .nax_sd_m1_k32, .nax_sd_m1_k64, .nax_sd_m2_k32 };

/// The shape `Config.mla_latent_nax` runs: TIME's fastest at every row count of the MTP head's calls (fork D1,
/// 2026-11-09, ~/lingc/logs/t3time-d1time.log: 18,380 keys, rows 1 to 5, 229 to 938 us, 0.161 to 0.189 of the stage
/// form's 1,209 to 5,825; the split scores at 1 M tile next, 0.170 to 0.192; the whole scores 0.23 to 0.27; 64 keys a
/// step 0.39 to 0.43)
pub const nax_served: Attn = .nax_sd_m2_k32;

const Variant = enum(u8) { rows, stage, block, nax };
const variant_kernels = Global([4]?mlx.mlx_fast_metal_kernel, "ling_mla_latent_variant_kernels", @splat(null));
/// The variants whose engagement line has printed (bit `@intFromEnum(Attn)`)
const variant_said = Global(u16, "ling_mla_latent_variant_said", 0);
/// The text the last call's launch 2 ran (`lastAttn`): a host's engagement line reads it after its call
const last_attn = Global(u8, "ling_mla_latent_last_attn", 0);

fn variantKernel(v: Variant) !mlx.mlx_fast_metal_kernel {
    lock();
    defer unlock();
    const slot = &variant_kernels.get().*[@intFromEnum(v)];
    if (slot.*) |made| return made;
    // as launch 2: the cache views read through their strides
    const made = switch (v) {
        .rows => ks.newKernel("turboexl3_ling_mla_attn_rows", &attn_in, &attn_out, text_attn_rows, text_header, false),
        .stage => ks.newKernel("turboexl3_ling_mla_attn_stage", &attn_in, &attn_out, text_attn_stage, text_header, false),
        .block => ks.newKernel("turboexl3_ling_mla_attn_block", &attn_in, &attn_out, text_attn_block, text_header, false),
        .nax => ks.newKernel("turboexl3_ling_mla_attn_nax", &attn_nax_in, &attn_out, text_attn_nax, header_nax, false),
    } orelse return error.KernelCreateFailed;
    slot.* = made;
    return made;
}

/// Launch 2's kernel for `a`
fn attnKernel(a: Attn) !mlx.mlx_fast_metal_kernel {
    return switch (a) {
        .base => kernelOf(.attn),
        .rows2, .rows4 => variantKernel(.rows),
        .stage => variantKernel(.stage),
        .block => variantKernel(.block),
        .nax_m1_k32, .nax_m1_k64, .nax_m2_k32, .nax_m2_k64, .nax_m4_k32, .nax_m4_k64, .nax_sd_m1_k32, .nax_sd_m1_k64, .nax_sd_m2_k32 => variantKernel(.nax),
    };
}

/// A variant's engagement line, at its first call (where the call runs it)
fn sayEngaged(a: Attn, p: Plan) void {
    if (a == .base) return;
    const bit = @as(u16, 1) << @intCast(@intFromEnum(a));
    if ((variant_said.get().* & bit) != 0) return;
    variant_said.get().* |= bit;
    if (naxOf(a)) |n| {
        std.debug.print("[turboexl3] mla latent attn engaged: {s} (launch 2 on the matrix units, {d} heads a group, {d} keys a step, {s} scores; first call {d} rows over {d} keys)\n", .{ @tagName(a), 16 * n.mt, n.bk, if (n.sd) "split" else "whole", p.s, p.t });
        return;
    }
    std.debug.print("[turboexl3] mla latent attn engaged: {s} (launch 2's variant, {d} rows a group; first call {d} rows over {d} keys)\n", .{ @tagName(a), rowsPerGroup(a), p.s, p.t });
}

/// The text the last `decodeLatent` call's launch 2 ran (`.base` before any call)
pub fn lastAttn() Attn {
    return @enumFromInt(last_attn.get().*);
}

/// The process's kernels dropped (a test's or a host's teardown), through the consumer (`decode.freeKernel`)
pub fn freeKernels() void {
    lock();
    defer unlock();
    for (&kernels.get().*) |*slot| if (slot.*) |k| {
        decode.freeKernel(k);
        slot.* = null;
    };
    for (&variant_kernels.get().*) |*slot| if (slot.*) |k| {
        decode.freeKernel(k);
        slot.* = null;
    };
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernels, variant_kernels, variant_said, last_attn };

// ── the plan ──

/// A call's geometry: heads, query rows, cached keys (the rows the last S of them), key groups; the operands' dtypes
pub const Plan = struct {
    h: u32,
    s: u32,
    t: u32,
    nj: u32,
    q: mlx.mlx_dtype,
    c: mlx.mlx_dtype,
    w: mlx.mlx_dtype,

    /// launch 1's grid: a thread for 4 columns of a head
    pub fn qabsGrid(p: Plan) [3]u32 {
        return .{ p.h * (dl / 4), 1, 1 };
    }
    /// launch 2's grid: a threadgroup for each (key group, 32 heads of a row)
    pub fn attnGrid(p: Plan) [3]u32 {
        return .{ attn_threads, p.nj, p.s * (p.h / heads_per_group) };
    }
    /// launch 3's grid: a threadgroup of 128 for each (head, row)
    pub fn vabsGrid(p: Plan) [3]u32 {
        return .{ 128, p.h, p.s };
    }
};

fn floatType(d: mlx.mlx_dtype) bool {
    return d == .bfloat16 or d == .float16 or d == .float32;
}

/// The key groups of `t` keys
pub fn groupsOf(t: u32) u32 {
    return (t + kpt - 1) / kpt;
}

/// The query rows a launch-2 threadgroup takes under `a` (the rows text's RPT): 2 or 4 for the rows variants, else 1
pub fn rowsPerGroup(a: Attn) u32 {
    return switch (a) {
        .rows2 => 2,
        .rows4 => 4,
        else => 1,
    };
}

/// Launch 2's threads a threadgroup under `a`: 512, or 128 an M tile of a NAX shape
pub fn attnThreadsOf(a: Attn) u32 {
    return if (naxOf(a)) |n| 128 * n.mt else attn_threads;
}

/// Launch 2's grid under `a` (in threads along x, as MLX takes it): a threadgroup for each (key group, 32 heads of
/// `rowsPerGroup(a)` rows; a call's last group of rows may hold fewer); launch 2's own (`Plan.attnGrid`) at one row a
/// threadgroup; a NAX shape's, one for each (key group, 16 mt heads of a row)
pub fn attnGridOf(p: Plan, a: Attn) [3]u32 {
    if (naxOf(a)) |n| return .{ 128 * n.mt, p.nj, p.s * (p.h / (16 * n.mt)) };
    const rpt = rowsPerGroup(a);
    return .{ attn_threads, p.nj, ((p.s + rpt - 1) / rpt) * (p.h / heads_per_group) };
}

/// Whether NAX shape `n` takes a call of plan `p`: the cache in bf16 or f16 (the matrix units' operands) and the heads a
/// whole number of its threadgroups. The cache's strides are the launch's (a view's strides exist once it is
/// evaluated): the text reads a key's dims as one run where they are contiguous, else word by word. A call it does not
/// take runs the configuration's text without it (`attnServedFor`).
pub fn naxTakes(p: Plan, n: Nax) bool {
    return (p.c == .bfloat16 or p.c == .float16) and p.h % (16 * n.mt) == 0;
}

/// Launch 2's text for a call of `rows` rows under `c`: `mla_latent_nax` (`nax_served`, where `naxTakes` the call),
/// else `mla_latent_rows` (2 or 4) at 2 rows or more, else `mla_latent_block`, else `mla_latent_stage`, else launch 2 as
/// it is (the defaults)
pub fn attnFor(c: config.Config, rows: u32) Attn {
    if (c.mla_latent_nax) return nax_served;
    return attnServedFor(c, rows);
}

/// `attnFor` without `mla_latent_nax`: the text a call runs where the NAX shape does not take it
pub fn attnServedFor(c: config.Config, rows: u32) Attn {
    if (rows >= 2 and c.mla_latent_rows == 2) return .rows2;
    if (rows >= 2 and c.mla_latent_rows == 4) return .rows4;
    if (c.mla_latent_block) return .block;
    if (c.mla_latent_stage) return .stage;
    return .base;
}

/// `decodeLatent`'s plan from the operands' shapes and dtypes: q_nope [1, H, S, DN], q_rot [1, H, S, DR] (one dtype),
/// lat [1, 1, T, DL], rot [1, 1, T, DR] (one dtype), kv_b [H (DN + DV), DL]; bf16, f16 or f32 each. A shape the texts
/// do not take (H not a multiple of 32, S past `max_rows`, a batch) is error.Exl3EngineNotImplemented; operands that do
/// not agree, error.BadExl3Shape or error.BadExl3Dtype.
pub fn planOf(qn: []const c_int, qr: []const c_int, lat: []const c_int, rot: []const c_int, w: []const c_int, dq: [2]mlx.mlx_dtype, dc: [2]mlx.mlx_dtype, dw: mlx.mlx_dtype) !Plan {
    if (qn.len != 4 or qr.len != 4 or lat.len != 4 or rot.len != 4 or w.len != 2) return error.BadExl3Shape;
    if (qn[0] != 1 or qr[0] != 1 or lat[0] != 1 or rot[0] != 1) return error.Exl3EngineNotImplemented;
    if (lat[1] != 1 or rot[1] != 1) return error.BadExl3Shape;
    if (qn[3] != dn or qr[3] != dr or lat[3] != dl or rot[3] != dr or w[1] != dl) return error.BadExl3Shape;
    if (qr[1] != qn[1] or qr[2] != qn[2] or rot[2] != lat[2]) return error.BadExl3Shape;
    if (qn[1] < 1 or qn[2] < 1) return error.BadExl3Shape;
    const h: u32 = @intCast(qn[1]);
    const s: u32 = @intCast(qn[2]);
    if (w[0] != @as(c_int, @intCast(h * (dn + dv)))) return error.BadExl3Shape;
    if (lat[2] < qn[2]) return error.BadExl3Shape;
    if (h % heads_per_group != 0 or s > max_rows) return error.Exl3EngineNotImplemented;
    if (dq[0] != dq[1] or dc[0] != dc[1] or !floatType(dq[0]) or !floatType(dc[0]) or !floatType(dw)) return error.BadExl3Dtype;
    const t: u32 = @intCast(lat[2]);
    return .{ .h = h, .s = s, .t = t, .nj = groupsOf(t), .q = dq[0], .c = dc[0], .w = dw };
}

/// Whether `decodeLatent` serves these operands (`planOf` without error)
pub fn takes(q_nope: mlx.mlx_array, q_rot: mlx.mlx_array, lat: mlx.mlx_array, rot: mlx.mlx_array, kv_b: mlx.mlx_array) bool {
    _ = planOf(mlx.getShape(q_nope), mlx.getShape(q_rot), mlx.getShape(lat), mlx.getShape(rot), mlx.getShape(kv_b), .{ mlx.mlx_array_dtype(q_nope), mlx.mlx_array_dtype(q_rot) }, .{ mlx.mlx_array_dtype(lat), mlx.mlx_array_dtype(rot) }, mlx.mlx_array_dtype(kv_b)) catch return false;
    return true;
}

fn ci(x: u32) c_int {
    return @intCast(x);
}

fn launch(s: mlx.mlx_stream, k: Kind, l: ks.Launch, inputs: []const mlx.mlx_array, outs: []mlx.mlx_array) !void {
    try launchKernel(s, try kernelOf(k), l, inputs, outs);
}

fn launchKernel(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, l: ks.Launch, inputs: []const mlx.mlx_array, outs: []mlx.mlx_array) !void {
    const cfg = try ks.buildConfig(l);
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try ks.applyConfig(kern, cfg, inputs, outs, s);
}

/// Ling's MLA attention at decode rows in the latent: out [1, H, S, DV] in q_nope's dtype (the attention's output per
/// head, before the host's transpose), for q_nope [1, H, S, DN] and q_rot [1, H, S, DR] (roped; any strides), the
/// latent cache lat [1, 1, T, DL] (the normed latents) and rot [1, 1, T, DR] (the roped shared keys) after this step's
/// rows are appended (the call's rows the last S of the T; read through their strides), kv_b [H (DN + DV), DL] and the
/// softmax scale. Query row s sits at key position T - S + s and attends to the keys up to it. Shapes and refusals:
/// `planOf` (before any launch). Launch 2 runs the text `attnFor` picks from the configuration (each variant launch 2's
/// words).
pub fn decodeLatent(s: mlx.mlx_stream, q_nope: mlx.mlx_array, q_rot: mlx.mlx_array, lat: mlx.mlx_array, rot: mlx.mlx_array, kv_b: mlx.mlx_array, scale: f32) !mlx.mlx_array {
    return decodeLatentWith(s, q_nope, q_rot, lat, rot, kv_b, scale, null);
}

/// `decodeLatent` with launch 2 on `attn` (null: `attnFor` the configuration and the call's rows), the tests' and the
/// timing's choice of text
pub fn decodeLatentWith(s: mlx.mlx_stream, q_nope: mlx.mlx_array, q_rot: mlx.mlx_array, lat: mlx.mlx_array, rot: mlx.mlx_array, kv_b: mlx.mlx_array, scale: f32, attn: ?Attn) !mlx.mlx_array {
    const r = try decodeLatentParts(s, q_nope, q_rot, lat, rot, kv_b, scale, attn);
    freeArr(r.pm);
    freeArr(r.pl);
    freeArr(r.po);
    return r.out;
}

/// `decodeLatentWith`'s output and launch 2's partials (pm, pl f32 [S H NJ], po f32 [S H NJ, DL]), which a variant must
/// give word for word; the caller frees all four
pub const Parts = struct { out: mlx.mlx_array, pm: mlx.mlx_array, pl: mlx.mlx_array, po: mlx.mlx_array };

/// `decodeLatentWith`, launch 2's partials kept (`Parts`)
pub fn decodeLatentParts(s: mlx.mlx_stream, q_nope: mlx.mlx_array, q_rot: mlx.mlx_array, lat: mlx.mlx_array, rot: mlx.mlx_array, kv_b: mlx.mlx_array, scale: f32, attn: ?Attn) !Parts {
    const p = try planOf(mlx.getShape(q_nope), mlx.getShape(q_rot), mlx.getShape(lat), mlx.getShape(rot), mlx.getShape(kv_b), .{ mlx.mlx_array_dtype(q_nope), mlx.mlx_array_dtype(q_rot) }, .{ mlx.mlx_array_dtype(lat), mlx.mlx_array_dtype(rot) }, mlx.mlx_array_dtype(kv_b));
    // a NAX shape the call's operands do not take: a test's choice is refused, the configuration's runs its text without
    // it (`attnServedFor`)
    const a = blk: {
        const want = attn orelse attnFor(config.get().*, p.s);
        if (naxOf(want)) |n| if (!naxTakes(p, n)) {
            if (attn != null) return error.Exl3EngineNotImplemented;
            break :blk attnServedFor(config.get().*, p.s);
        };
        break :blk want;
    };
    // the queries' rows contiguous (a no-op where they are): launch 2 indexes qr flat
    var qn = mlx.mlx_array_new();
    defer freeArr(qn);
    try mlx.check(mlx.mlx_contiguous(&qn, q_nope, false, s));
    var qr = mlx.mlx_array_new();
    defer freeArr(qr);
    try mlx.check(mlx.mlx_contiguous(&qr, q_rot, false, s));
    var ql: [1]mlx.mlx_array = undefined;
    try launch(s, .qabs, .{
        .grid = p.qabsGrid(),
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{&[_]c_int{ ci(p.s), ci(p.h), ci(dl) }},
        .out_dtypes = &.{.float32},
        .template = &.{
            .{ .name = "H", .value = .{ .int = ci(p.h) } },
            .{ .name = "S", .value = .{ .int = ci(p.s) } },
            .{ .name = "DN", .value = .{ .int = ci(dn) } },
            .{ .name = "DV", .value = .{ .int = ci(dv) } },
            .{ .name = "DL", .value = .{ .int = ci(dl) } },
            .{ .name = "QT", .value = .{ .dtype = p.q } },
            .{ .name = "WT", .value = .{ .dtype = p.w } },
        },
    }, &.{ qn, kv_b }, &ql);
    defer freeArr(ql[0]);
    const scale_arr = mlx.mlx_array_new_data(&scale, &[_]c_int{1}, 1, .float32);
    defer freeArr(scale_arr);
    const dims_v = [2]i32{ @intCast(p.t), @intCast(p.nj) };
    const dims = mlx.mlx_array_new_data(&dims_v, &[_]c_int{2}, 1, .int32);
    defer freeArr(dims);
    const slots: c_int = ci(p.s * p.h * p.nj);
    var parts: [3]mlx.mlx_array = undefined;
    // launch 2's constants; the rows text's RPT after them
    const attn_t = [_]ks.TemplateArg{
        .{ .name = "H", .value = .{ .int = ci(p.h) } },
        .{ .name = "S", .value = .{ .int = ci(p.s) } },
        .{ .name = "DL", .value = .{ .int = ci(dl) } },
        .{ .name = "DR", .value = .{ .int = ci(dr) } },
        .{ .name = "KPT", .value = .{ .int = ci(kpt) } },
        .{ .name = "CK", .value = .{ .int = ci(ck) } },
        .{ .name = "QT", .value = .{ .dtype = p.q } },
        .{ .name = "CT", .value = .{ .dtype = p.c } },
        .{ .name = "RPT", .value = .{ .int = ci(rowsPerGroup(a)) } },
    };
    const attn_tl: []const ks.TemplateArg = if (a == .rows2 or a == .rows4) attn_t[0..9] else attn_t[0..8];
    if (naxOf(a)) |n| {
        // the NAX launch's query halves in the cache's dtype: launch 1's f32 rounded once, the rope half converted where
        // its dtype differs (contiguous already: the text indexes them flat)
        var qlb = mlx.mlx_array_new();
        defer freeArr(qlb);
        try mlx.check(mlx.mlx_astype(&qlb, ql[0], p.c, s));
        var qrb = mlx.mlx_array_new();
        defer freeArr(qrb);
        try mlx.check(mlx.mlx_astype(&qrb, qr, p.c, s));
        try launchKernel(s, try attnKernel(a), .{
            .grid = attnGridOf(p, a),
            .threadgroup = .{ attnThreadsOf(a), 1, 1 },
            .out_shapes = &.{ &[_]c_int{slots}, &[_]c_int{slots}, &[_]c_int{ slots, ci(dl) } },
            .out_dtypes = &.{ .float32, .float32, .float32 },
            .template = &.{
                .{ .name = "H", .value = .{ .int = ci(p.h) } },
                .{ .name = "S", .value = .{ .int = ci(p.s) } },
                .{ .name = "DL", .value = .{ .int = ci(dl) } },
                .{ .name = "DR", .value = .{ .int = ci(dr) } },
                .{ .name = "KPT", .value = .{ .int = ci(kpt) } },
                .{ .name = "MT", .value = .{ .int = ci(n.mt) } },
                .{ .name = "BKN", .value = .{ .int = ci(n.bk) } },
                .{ .name = "SD", .value = .{ .int = @intFromBool(n.sd) } },
                .{ .name = "CT", .value = .{ .dtype = p.c } },
            },
        }, &.{ qlb, qrb, lat, rot, scale_arr, dims }, &parts);
    } else {
        try launchKernel(s, try attnKernel(a), .{
            .grid = attnGridOf(p, a),
            .threadgroup = .{ attn_threads, 1, 1 },
            .out_shapes = &.{ &[_]c_int{slots}, &[_]c_int{slots}, &[_]c_int{ slots, ci(dl) } },
            .out_dtypes = &.{ .float32, .float32, .float32 },
            .template = attn_tl,
        }, &.{ ql[0], qr, lat, rot, scale_arr, dims }, &parts);
    }
    errdefer for (parts) |x| freeArr(x);
    sayEngaged(a, p);
    last_attn.get().* = @intFromEnum(a);
    var out: [1]mlx.mlx_array = undefined;
    try launch(s, .vabs, .{
        .grid = p.vabsGrid(),
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = &.{&[_]c_int{ 1, ci(p.h), ci(p.s), ci(dv) }},
        .out_dtypes = &.{p.q},
        .template = &.{
            .{ .name = "H", .value = .{ .int = ci(p.h) } },
            .{ .name = "S", .value = .{ .int = ci(p.s) } },
            .{ .name = "DL", .value = .{ .int = ci(dl) } },
            .{ .name = "DN", .value = .{ .int = ci(dn) } },
            .{ .name = "DV", .value = .{ .int = ci(dv) } },
            .{ .name = "KPT", .value = .{ .int = ci(kpt) } },
            .{ .name = "WT", .value = .{ .dtype = p.w } },
            .{ .name = "OT", .value = .{ .dtype = p.q } },
        },
    }, &.{ parts[0], parts[1], parts[2], kv_b, dims }, &out);
    return .{ .out = out[0], .pm = parts[0], .pl = parts[1], .po = parts[2] };
}

// ── CPU tests: the texts and the plan ──

const testing = std.testing;
const support = @import("../../core/test_support.zig");

fn has(text: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, text, needle) != null;
}

test "turboexl3 ling mla latent texts: each text reads every operand it binds and the constants its launch passes; only launch 2 reads strides; key positions count from 0 in fixed groups and sub-chunks" {
    const pin = comptime support.pins(@embedFile("kernels/test/pins/mla_latent_texts.metal"));
    for ([_][]const u8{ text_qabs, text_attn, text_vabs }, [_][]const [*:0]const u8{ &qabs_in, &attn_in, &vabs_in }, [_][]const [*:0]const u8{ &qabs_out, &attn_out, &vabs_out }) |txt, ins, outs| {
        for (ins) |n| try testing.expect(has(txt, std.mem.span(n)));
        for (outs) |n| try testing.expect(has(txt, std.mem.span(n)));
    }
    for ([_][]const u8{ "H", "S", "DN", "DV", "DL", "QT", "WT" }) |n| try testing.expect(has(text_qabs, n));
    for ([_][]const u8{ "H", "S", "DL", "DR", "KPT", "CK", "QT", "CT" }) |n| try testing.expect(has(text_attn, n));
    for ([_][]const u8{ "H", "S", "DL", "DN", "DV", "KPT", "WT", "OT" }) |n| try testing.expect(has(text_vabs, n));
    // the strided cache views (launch 2 only; MLX binds a <name>_strides buffer where a text names one)
    try testing.expect(has(text_attn, "lat_strides") and has(text_attn, "rot_strides"));
    for ([_][]const u8{ text_qabs, text_vabs }) |txt| try testing.expect(!has(txt, "_strides"));
    for ([_][]const u8{ text_qabs, text_attn, text_vabs }) |txt| try testing.expect(!has(txt, "_shape"));
    // the arithmetic a row's bits rest on: its position (lim), the key group and sub-chunk from position 0, a key past
    // it read as 0 and scored -inf, the merge over the groups up to it in order
    for (pin[0..5]) |line| try testing.expect(has(text_attn, line));
    for (pin[5..7]) |line| try testing.expect(has(text_vabs, line));
    try testing.expect(has(text_header, pin[7]));
}

test "turboexl3 ling mla latent plan: Ling's shapes are taken, a batch, an odd head count or a wide call refused before any launch; the key groups and grids" {
    const bf: mlx.mlx_dtype = .bfloat16;
    const p = try planOf(&.{ 1, 64, 1, 128 }, &.{ 1, 64, 1, 64 }, &.{ 1, 1, 18380, 512 }, &.{ 1, 1, 18380, 64 }, &.{ 16384, 512 }, .{ bf, bf }, .{ bf, bf }, bf);
    try testing.expect(p.h == 64 and p.s == 1 and p.t == 18380 and p.nj == 72);
    try testing.expectEqual([3]u32{ 8192, 1, 1 }, p.qabsGrid());
    try testing.expectEqual([3]u32{ 512, 72, 2 }, p.attnGrid());
    try testing.expectEqual([3]u32{ 128, 64, 1 }, p.vabsGrid());
    const v = try planOf(&.{ 1, 64, 6, 128 }, &.{ 1, 64, 6, 64 }, &.{ 1, 1, 256, 512 }, &.{ 1, 1, 256, 64 }, &.{ 16384, 512 }, .{ bf, bf }, .{ bf, bf }, bf);
    try testing.expect(v.s == 6 and v.nj == 1);
    try testing.expectEqual([3]u32{ 512, 1, 12 }, v.attnGrid());
    try testing.expectEqual(@as(u32, 2), groupsOf(257));
    try testing.expectEqual(@as(u32, 1), groupsOf(1));
    const E = error.Exl3EngineNotImplemented;
    try testing.expectError(E, planOf(&.{ 2, 64, 1, 128 }, &.{ 2, 64, 1, 64 }, &.{ 2, 1, 8, 512 }, &.{ 2, 1, 8, 64 }, &.{ 16384, 512 }, .{ bf, bf }, .{ bf, bf }, bf));
    try testing.expectError(E, planOf(&.{ 1, 48, 1, 128 }, &.{ 1, 48, 1, 64 }, &.{ 1, 1, 8, 512 }, &.{ 1, 1, 8, 64 }, &.{ 12288, 512 }, .{ bf, bf }, .{ bf, bf }, bf));
    try testing.expectError(E, planOf(&.{ 1, 64, 17, 128 }, &.{ 1, 64, 17, 64 }, &.{ 1, 1, 40, 512 }, &.{ 1, 1, 40, 64 }, &.{ 16384, 512 }, .{ bf, bf }, .{ bf, bf }, bf));
    const B = error.BadExl3Shape;
    try testing.expectError(B, planOf(&.{ 1, 64, 4, 128 }, &.{ 1, 64, 4, 64 }, &.{ 1, 1, 3, 512 }, &.{ 1, 1, 3, 64 }, &.{ 16384, 512 }, .{ bf, bf }, .{ bf, bf }, bf));
    try testing.expectError(B, planOf(&.{ 1, 64, 1, 192 }, &.{ 1, 64, 1, 64 }, &.{ 1, 1, 8, 512 }, &.{ 1, 1, 8, 64 }, &.{ 16384, 512 }, .{ bf, bf }, .{ bf, bf }, bf));
    try testing.expectError(B, planOf(&.{ 1, 64, 1, 128 }, &.{ 1, 64, 1, 64 }, &.{ 1, 1, 8, 512 }, &.{ 1, 1, 8, 64 }, &.{ 8192, 512 }, .{ bf, bf }, .{ bf, bf }, bf));
    try testing.expectError(error.BadExl3Dtype, planOf(&.{ 1, 64, 1, 128 }, &.{ 1, 64, 1, 64 }, &.{ 1, 1, 8, 512 }, &.{ 1, 1, 8, 64 }, &.{ 16384, 512 }, .{ bf, .float32 }, .{ bf, bf }, bf));
    try testing.expectError(error.BadExl3Dtype, planOf(&.{ 1, 64, 1, 128 }, &.{ 1, 64, 1, 64 }, &.{ 1, 1, 8, 512 }, &.{ 1, 1, 8, 64 }, &.{ 16384, 512 }, .{ bf, bf }, .{ bf, bf }, .int32));
}

test "turboexl3 ling mla latent variant texts: each is launch 2's online softmax update (attn_2) between pieces of its own, the stage text launch 2's outputs (attn_4) too; each reads every operand launch 2 binds, its constants and the cache's strides; the statements a row's words rest on" {
    const pin = comptime support.pins(@embedFile("kernels/test/pins/mla_latent_variants.metal"));
    for ([_][]const u8{ text_attn, text_attn_rows, text_attn_stage, text_attn_block }) |txt| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, txt, attn_2));
        for (attn_in) |n| try testing.expect(has(txt, std.mem.span(n)));
        for (attn_out) |n| try testing.expect(has(txt, std.mem.span(n)));
        for ([_][]const u8{ "H", "S", "DL", "DR", "KPT", "CK", "QT" }) |n| try testing.expect(has(txt, n));
        try testing.expect(has(txt, "lat_strides") and has(txt, "rot_strides") and !has(txt, "_shape"));
        // a key group counted from position 0
        try testing.expect(has(txt, pin[0]));
    }
    // launch 2, rows and block read the cache through CT pointers. The stage text reads `lat` and `rot` themselves,
    // whose element type MLX writes into the signature from the cache's dtype (the type CT names), so it names no CT.
    for ([_][]const u8{ text_attn, text_attn_rows, text_attn_block }) |txt| try testing.expect(has(txt, "CT"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, text_attn_stage, attn_4));
    try testing.expect(has(text_attn_rows, "RPT"));
    // rows: each row's bound, the sub-chunks it runs, a key it does not see read as 0 and scored -inf
    for (pin[1..6]) |line| try testing.expect(has(text_attn_rows, line));
    // stage: the staged word is launch 2's (the converted word, 0 past the bound), launch 2's score statement
    for (pin[6..9]) |line| try testing.expect(has(text_attn_stage, line));
    // block: launch 2's operand and score statements on each tile, the rescale of each output tile
    for (pin[9..13]) |line| try testing.expect(has(text_attn_block, line));
    for ([_][]const u8{ text_attn, text_attn_stage, text_attn_block }) |txt| try testing.expect(has(txt, pin[13]));
    try testing.expect(has(text_attn, pin[8]));
    // MLX writes a kernel's signature from its text's words: a variant names no thread attribute launch 2 does not
    for ([_][]const u8{ text_attn, text_attn_rows, text_attn_stage, text_attn_block }) |txt| {
        try testing.expect(!has(txt, pin[14]) and !has(txt, pin[15]));
    }
}

test "turboexl3 ling mla latent variants plan: the configuration's choice of launch 2's text per row count, each text's grid, and the configurations the library serves" {
    const bf: mlx.mlx_dtype = .bfloat16;
    const p = try planOf(&.{ 1, 64, 3, 128 }, &.{ 1, 64, 3, 64 }, &.{ 1, 1, 19404, 512 }, &.{ 1, 1, 19404, 64 }, &.{ 16384, 512 }, .{ bf, bf }, .{ bf, bf }, bf);
    try testing.expect(p.nj == 76);
    try testing.expectEqual(p.attnGrid(), attnGridOf(p, .base));
    try testing.expectEqual([3]u32{ 512, 76, 6 }, attnGridOf(p, .stage));
    try testing.expectEqual([3]u32{ 512, 76, 6 }, attnGridOf(p, .block));
    try testing.expectEqual([3]u32{ 512, 76, 4 }, attnGridOf(p, .rows2));
    try testing.expectEqual([3]u32{ 512, 76, 2 }, attnGridOf(p, .rows4));
    const w = try planOf(&.{ 1, 64, 16, 128 }, &.{ 1, 64, 16, 64 }, &.{ 1, 1, 300, 512 }, &.{ 1, 1, 300, 64 }, &.{ 16384, 512 }, .{ bf, bf }, .{ bf, bf }, bf);
    try testing.expectEqual([3]u32{ 512, 2, 8 }, attnGridOf(w, .rows4));
    try testing.expectEqual([3]u32{ 512, 2, 16 }, attnGridOf(w, .rows2));
    try testing.expectEqual(@as(u32, 1), rowsPerGroup(.block));
    const none: config.Config = .{};
    for ([_]u32{ 1, 2, 3, 4, 16 }) |rows| try testing.expectEqual(Attn.base, attnFor(none, rows));
    const r2: config.Config = .{ .mla_latent_rows = 2 };
    const r4b: config.Config = .{ .mla_latent_rows = 4, .mla_latent_block = true };
    const st: config.Config = .{ .mla_latent_stage = true };
    try testing.expectEqual(Attn.base, attnFor(r2, 1));
    try testing.expectEqual(Attn.rows2, attnFor(r2, 2));
    try testing.expectEqual(Attn.rows2, attnFor(r2, 16));
    try testing.expectEqual(Attn.block, attnFor(r4b, 1));
    try testing.expectEqual(Attn.rows4, attnFor(r4b, 3));
    try testing.expectEqual(Attn.stage, attnFor(st, 1));
    try testing.expectEqual(Attn.stage, attnFor(st, 4));
    for ([_]u8{ 0, 2, 4 }) |r| try testing.expect((config.Config{ .mla_latent_rows = r }).valid());
    for ([_]u8{ 1, 3, 8 }) |r| try testing.expect(!(config.Config{ .mla_latent_rows = r }).valid());
    try testing.expect(!(config.Config{ .mla_latent_stage = true, .mla_latent_block = true }).valid());
    try testing.expect((config.Config{ .mla_latent_rows = 4, .mla_latent_stage = true }).valid());
}

test "turboexl3 ling mla latent nax text: it reads every operand it binds and the constants its launch passes, the cache through its strides; the statements a row's bound, its mask, the split's sum and the partials rest on" {
    const pin = comptime support.pins(@embedFile("kernels/test/pins/mla_latent_nax.metal"));
    for (attn_nax_in) |n| try testing.expect(has(text_attn_nax, std.mem.span(n)));
    for (attn_out) |n| try testing.expect(has(text_attn_nax, std.mem.span(n)));
    for ([_][]const u8{ "H", "S", "DL", "DR", "KPT", "MT", "BKN", "SD", "CT" }) |n| try testing.expect(has(text_attn_nax, n));
    try testing.expect(has(text_attn_nax, "lat_strides") and has(text_attn_nax, "rot_strides") and !has(text_attn_nax, "_shape"));
    // the bound and the key group from position 0 (launch 2's), the step's keys, the mask, the split's sum in order, the
    // partials launch 3 merges
    for (pin[0..8]) |line| try testing.expect(has(text_attn_nax, line));
    // the header: the row ops in natural units, the finite minimum and the attribute after MLX steel's NAX tiles; the
    // close undefines it
    for (pin[8..11]) |line| try testing.expect(has(header_nax, line));
    try testing.expect(std.mem.startsWith(u8, header_nax, steel_nax));
    try testing.expect(std.mem.endsWith(u8, std.mem.trimEnd(u8, text_attn_nax, "\n"), pin[11]));
    // the merge reads pm in natural units: the header's exp is metal::exp, not attention_nax's exp2 of log2(e)-scaled scores
    try testing.expect(!has(text_attn_nax, "exp2") and !has(header_nax[steel_nax.len..], "exp2"));
}

test "turboexl3 ling mla latent nax plan: each shape's grid and threads, the configuration's text and its fallback, the calls a shape takes, the split shapes' threadgroup memory" {
    const bf: mlx.mlx_dtype = .bfloat16;
    const p = try planOf(&.{ 1, 64, 3, 128 }, &.{ 1, 64, 3, 64 }, &.{ 1, 1, 18380, 512 }, &.{ 1, 1, 18380, 64 }, &.{ 16384, 512 }, .{ bf, bf }, .{ bf, bf }, bf);
    try testing.expect(p.nj == 72);
    try testing.expectEqual([3]u32{ 128, 72, 12 }, attnGridOf(p, .nax_m1_k32));
    try testing.expectEqual([3]u32{ 256, 72, 6 }, attnGridOf(p, .nax_m2_k64));
    try testing.expectEqual([3]u32{ 512, 72, 3 }, attnGridOf(p, .nax_m4_k32));
    try testing.expectEqual([3]u32{ 128, 72, 12 }, attnGridOf(p, .nax_sd_m1_k64));
    try testing.expectEqual(@as(u32, 512), attnThreadsOf(.stage));
    try testing.expectEqual(@as(u32, 256), attnThreadsOf(.nax_sd_m2_k32));
    // every NAX enum value is a shape, every shape is listed once, launch 2 and its word-for-word variants are none
    var n_nax: usize = 0;
    for (std.enums.values(Attn)) |a| {
        if (naxOf(a)) |n| {
            n_nax += 1;
            var listed: usize = 0;
            for (nax_shapes) |b| listed += @intFromBool(b == a);
            try testing.expectEqual(@as(usize, 1), listed);
            try testing.expect(n.bk == 32 or n.bk == 64);
            try testing.expect(n.mt == 1 or n.mt == 2 or n.mt == 4);
            // a split shape's partial tiles: MT x 4 x 16 x (bk + 4) floats, under the 32 KiB a threadgroup holds
            if (n.sd) try testing.expect(n.mt * 4 * 16 * (n.bk + 4) * 4 < 32 * 1024);
            try testing.expectEqual(@as(u32, 1), rowsPerGroup(a));
        }
    }
    try testing.expectEqual(nax_shapes.len, n_nax);
    for ([_]Attn{ .base, .rows2, .rows4, .stage, .block }) |a| try testing.expect(naxOf(a) == null);
    try testing.expect(naxOf(nax_served) != null);
    // the configuration: the NAX shape at every row count, ahead of the stage; without it the stage as before
    const nax_st: config.Config = .{ .mla_latent_nax = true, .mla_latent_stage = true };
    for ([_]u32{ 1, 2, 3, 4, 16 }) |rows| {
        try testing.expectEqual(nax_served, attnFor(nax_st, rows));
        try testing.expectEqual(Attn.stage, attnServedFor(nax_st, rows));
    }
    const none: config.Config = .{};
    for ([_]u32{ 1, 4 }) |rows| try testing.expectEqual(Attn.base, attnServedFor(none, rows));
    // the calls a shape takes: a bf16 or f16 cache, the heads a whole number of its threadgroups
    for (nax_shapes) |a| try testing.expect(naxTakes(p, naxOf(a).?));
    const f16p = try planOf(&.{ 1, 64, 1, 128 }, &.{ 1, 64, 1, 64 }, &.{ 1, 1, 300, 512 }, &.{ 1, 1, 300, 64 }, &.{ 16384, 512 }, .{ .float16, .float16 }, .{ .float16, .float16 }, .float16);
    try testing.expect(naxTakes(f16p, naxOf(nax_served).?));
    const f32p = try planOf(&.{ 1, 64, 1, 128 }, &.{ 1, 64, 1, 64 }, &.{ 1, 1, 300, 512 }, &.{ 1, 1, 300, 64 }, &.{ 16384, 512 }, .{ bf, bf }, .{ .float32, .float32 }, bf);
    for (nax_shapes) |a| try testing.expect(!naxTakes(f32p, naxOf(a).?));
    const h32 = try planOf(&.{ 1, 32, 1, 128 }, &.{ 1, 32, 1, 64 }, &.{ 1, 1, 300, 512 }, &.{ 1, 1, 300, 64 }, &.{ 8192, 512 }, .{ bf, bf }, .{ bf, bf }, bf);
    try testing.expect(naxTakes(h32, naxOf(.nax_m2_k32).?));
    try testing.expect(!naxTakes(h32, naxOf(.nax_m4_k32).?));
}

test "turboexl3 ling mla latent nax: only the latent decode reads Config.mla_latent_nax, and no other Ling module reaches the latent decode (the trunk's MLA never takes it)" {
    // the trunk's MLA prompt attention, its decode GEMVs, the routed, shared and dense paths: none reads the field or
    // calls the latent decode
    const others = .{ @embedFile("mla_flash.zig"), @embedFile("kdaproj.zig"), @embedFile("kdaproj_pf.zig"), @embedFile("router.zig"), @embedFile("shared.zig"), @embedFile("tail.zig"), @embedFile("combine.zig"), @embedFile("swiglu.zig"), @embedFile("pair.zig") };
    inline for (others) |src| {
        try testing.expect(!has(src, "mla_latent_nax"));
        try testing.expect(!has(src, "decodeLatent"));
        try testing.expect(!has(src, "mla_latent.zig"));
    }
    // the API exports the latent decode as mlaLatentDecode (the plugin calls it for the MTP head alone) and never the field
    const api_src = @embedFile("api.zig");
    try testing.expect(!has(api_src, "mla_latent_nax"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, api_src, "= mla_latent.decodeLatent;"));
    // in this file the field is read once, by attnFor (its other mentions are comments)
    const needle = "c." ++ "mla_latent_nax";
    var reads: usize = 0;
    var it = std.mem.splitScalar(u8, @embedFile("mla_latent.zig"), '\n');
    while (it.next()) |line| {
        const code = std.mem.trimStart(u8, line, " ");
        if (std.mem.startsWith(u8, code, "//")) continue;
        if (has(code, needle)) reads += 1;
    }
    try testing.expectEqual(@as(usize, 1), reads);
}
