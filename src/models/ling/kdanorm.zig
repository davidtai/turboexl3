//! L5 (KDA_NORM_PREP): a KDA layer's gated norm written straight into o_proj's prepared prompt rows. The served chain
//! runs the host's fused norm-gate (mlx-serve kernels/gdn/gdn_normgate.metal at its sigmoid table, TAB 1, the sigmoid
//! gate: what the outside plugin turboexl3_ling serves with its prework), o bf16 [rows, HV·DV], and then o_proj's
//! `linear`, whose first launch prepares o (dense's `text_prep` at LANE_ORDER 0: each word to f16, times suh, H128 per
//! 128-block) for its matmul. Here one launch, `turboexl3_ling_kdanorm_prep`, is that prepare with the norm-gate's
//! statements in its load's place (kernels/kdanorm/gate.metal between text_prep's first and last pieces): a simdgroup a
//! (row, head) as in both kernels (a head is one 128-block at DV 128), the head's sum of squares in the host's own order
//! (each lane's four consecutive channels, then simd_sum), and each value the prepare loads (channel L + 32 j) made by
//! the host's per-element statements on that element. So o never reaches device memory, and xp is the prepare's words.
//! o_proj's y is then `Dense.piecePrepared` and `Dense.finishOf` over a W_hat `Dense.dequantize` decoded. That is
//! `linear`'s bytes on its matmul path over the decoded W_hat: the outside plugin's LAYER_WHAT over its kept W_hat, whose
//! `matmulPath` also excludes pg2 and an un-rotated resident trellis (that arm of `linear` is one matmul of x against its
//! resident W_std and reads no prepared rows). A call is taken only where o_proj's `linear` is on its matmul path at
//! these rows (`takes`); else error.Exl3EngineNotImplemented, nothing launched. `takes` sees o_proj's `Dense`, not its
//! trellis, so the caller's condition carries the un-rotated resident's exclusion. The design record, with the compile
//! path and the inputs on which contraction cannot move a word: README.md, L5.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");

const A = mlx.mlx_array;

// ── texts ──

/// the norm-gate's statements at head b and the prepare's lanes, each lane's four values as the prepare takes them
pub const kgate = @embedFile("kernels/kdanorm/gate.metal");

/// dense's prepare with the norm-gate in its load's place (prep_2_1): its H128 stages (prep_2_2) and store (prep_3) after
pub const text_normprep: [:0]const u8 = dense.prep_1 ++ kgate ++ dense.prep_2_2 ++ dense.prep_3;

pub const normprep_in = [_][*:0]const u8{ "y", "z", "norm_w", "eps", "sigtab", "suh" };
pub const normprep_out = [_][*:0]const u8{"xp"};

// ── the kernel: made at its first call ──

const kernel_lock = Global(std.atomic.Value(u32), "ling_kdanorm_kernel_lock", std.atomic.Value(u32).init(0));
const kernel_cache = Global(?mlx.mlx_fast_metal_kernel, "ling_kdanorm_kernel", null);

/// Test seam: a kernel of this launch's operands that a call runs in place of `turboexl3_ling_kdanorm_prep` (the red
/// checks' planted faults); null: the launch's own
pub const kernel_override = Global(?mlx.mlx_fast_metal_kernel, "ling_kdanorm_kernel_override", null);

fn kernelGet() !mlx.mlx_fast_metal_kernel {
    if (kernel_override.get().*) |planted| return planted;
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    if (kernel_cache.get().*) |k| return k;
    const k = ks.newKernel("turboexl3_ling_kdanorm_prep", &normprep_in, &normprep_out, text_normprep, decode.header, true) orelse return error.KernelCreateFailed;
    kernel_cache.get().* = k;
    return k;
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernel_cache };

// ── the call ──

/// One call's operands, as the served chain hands them to the host's norm-gate and o_proj: y [B, S, HV, DV], the KDA
/// recurrence's output (bf16 or f16; DV 128); z [B, S, z_stride] in y's dtype, the gate's rows (head h's at z_off +
/// h·DV); norm_w [DV] in y's dtype; eps an f32 scalar (0-d); sigtab [65536] in y's dtype, MLX's sigmoid of every 16-bit
/// pattern of that dtype (the host's `sigmoidTableFor`, which its norm-gate reads at TAB 1); suh, o_proj's f16 [HV·DV]
pub const Op = struct {
    y: A,
    z: A,
    z_off: u32,
    z_stride: u32,
    norm_w: A,
    eps: A,
    sigtab: A,
    suh: A,
};

/// What `shapeOk` reads of an `Op`: each operand's dtype and shape (MLX-free, so a CPU test states the rules)
pub const Shapes = struct {
    y: struct { dt: mlx.mlx_dtype, sh: []const c_int },
    z: struct { dt: mlx.mlx_dtype, sh: []const c_int },
    z_off: u32,
    z_stride: u32,
    norm_w: struct { dt: mlx.mlx_dtype, sh: []const c_int },
    eps: struct { dt: mlx.mlx_dtype, sh: []const c_int },
    sigtab: struct { dt: mlx.mlx_dtype, sh: []const c_int },
    suh: struct { dt: mlx.mlx_dtype, sh: []const c_int },
};

/// A call's geometry: its rows (B·S), heads and o's width
pub const Geometry = struct { rows: u32, hv: u32, dim: u32 };

/// The geometry of operands the launch takes, the host's norm-gate's rules and the prepare's: y [B, S, HV, DV] at DV
/// 128 in bf16 or f16, z [B, S, z_stride] with every head's DV channels inside a row (z_off + HV·DV <= z_stride),
/// norm_w [DV], sigtab [65536], all in y's dtype; eps f32 0-d; suh f16 [HV·DV]; null otherwise
pub fn shapeOk(x: Shapes) ?Geometry {
    const dt = x.y.dt;
    if (dt != .bfloat16 and dt != .float16) return null;
    if (x.z.dt != dt or x.norm_w.dt != dt or x.sigtab.dt != dt) return null;
    if (x.eps.dt != .float32 or x.eps.sh.len != 0) return null;
    if (x.suh.dt != .float16) return null;
    const y = x.y.sh;
    const z = x.z.sh;
    if (y.len != 4 or z.len != 3) return null;
    if (y[0] < 1 or y[1] < 1 or y[2] < 1 or y[3] != 128) return null;
    if (z[0] != y[0] or z[1] != y[1] or z[2] < 1 or z[2] != @as(c_int, @intCast(x.z_stride))) return null;
    const hv: u32 = @intCast(y[2]);
    const dim = hv * 128;
    if (@as(u64, x.z_off) + dim > x.z_stride) return null;
    if (x.norm_w.sh.len != 1 or x.norm_w.sh[0] != 128) return null;
    if (x.sigtab.sh.len != 1 or x.sigtab.sh[0] != 1 << 16) return null;
    if (x.suh.sh.len != 1 or x.suh.sh[0] != @as(c_int, @intCast(dim))) return null;
    const rows = @as(u64, @intCast(y[0])) * @as(u64, @intCast(y[1]));
    if (rows > std.math.maxInt(u32)) return null;
    return .{ .rows = @intCast(rows), .hv = hv, .dim = dim };
}

fn shapesOf(op: Op) Shapes {
    return .{
        .y = .{ .dt = mlx.mlx_array_dtype(op.y), .sh = mlx.getShape(op.y) },
        .z = .{ .dt = mlx.mlx_array_dtype(op.z), .sh = mlx.getShape(op.z) },
        .z_off = op.z_off,
        .z_stride = op.z_stride,
        .norm_w = .{ .dt = mlx.mlx_array_dtype(op.norm_w), .sh = mlx.getShape(op.norm_w) },
        .eps = .{ .dt = mlx.mlx_array_dtype(op.eps), .sh = mlx.getShape(op.eps) },
        .sigtab = .{ .dt = mlx.mlx_array_dtype(op.sigtab), .sh = mlx.getShape(op.sigtab) },
        .suh = .{ .dt = mlx.mlx_array_dtype(op.suh), .sh = mlx.getShape(op.suh) },
    };
}

/// Whether `kdaNormGatePrep` takes `op` for o_proj `d`: `shapeOk`, o_proj's input o's width (d.shape.in == HV·DV), and
/// o_proj's `linear` at the call's rows on its matmul path (the path whose prepared rows `Dense.piecePrepared` reads).
/// It reads `d`, not o_proj's trellis: an un-rotated resident trellis (and pg2, for LAYER_WHAT's `matmulPath`) is the
/// caller's to rule out.
pub fn takes(d: *const dense.Dense, op: Op) bool {
    const geo = shapeOk(shapesOf(op)) orelse return false;
    return d.shape.in == geo.dim and d.pathFor(geo.rows) == .matmul;
}

/// xp half [B·S, HV·DV]: o_proj's prepared rows (dense's prepare at LANE_ORDER 0) of the host's norm-gate of `op`, word
/// for word, in one launch on `s`, a GPU stream; with `d.piecePrepared` and `d.finishOf` over a decoded W_hat, o_proj's y
/// (`linear`'s bytes under the module doc's condition). Owned result.
/// error.Exl3EngineNotImplemented where it does not take the call (`takes`, or not a GPU stream), nothing launched.
pub fn kdaNormGatePrep(s: mlx.mlx_stream, d: *const dense.Dense, op: Op) !A {
    if (!mlx.streamIsGpu(s) or !takes(d, op)) return error.Exl3EngineNotImplemented;
    const geo = shapeOk(shapesOf(op)).?;
    const k = try kernelGet();
    // the prepare's launch (`Dense.prepare` at these rows: a simdgroup each 128-block of a row) with the norm-gate's
    // template arguments
    const cfg = try ks.buildConfig(.{
        .grid = .{ geo.hv * 32, geo.rows, 1 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{&[_]c_int{ @intCast(geo.rows), @intCast(geo.dim) }},
        .out_dtypes = &.{.float16},
        .template = &.{
            .{ .name = "T", .value = .{ .dtype = mlx.mlx_array_dtype(op.y) } },
            .{ .name = "DIM", .value = .{ .int = @intCast(geo.dim) } },
            .{ .name = "LANE_ORDER", .value = .{ .int = 0 } },
            .{ .name = "HV", .value = .{ .int = @intCast(geo.hv) } },
            .{ .name = "DV", .value = .{ .int = 128 } },
            .{ .name = "ZSTRIDE", .value = .{ .int = @intCast(op.z_stride) } },
            .{ .name = "ZOFF", .value = .{ .int = @intCast(op.z_off) } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var outs: [1]A = undefined;
    try ks.applyConfig(k, cfg, &.{ op.y, op.z, op.norm_w, op.eps, op.sigtab, op.suh }, &outs, s);
    return outs[0];
}

// ── tests (CPU) ──

const testing = std.testing;

/// `needle` once in `hay`
fn once(hay: []const u8, needle: []const u8) bool {
    return std.mem.count(u8, hay, needle) == 1;
}

/// each of `needles` in `hay`, each after the one before it
fn inOrder(hay: []const u8, needles: []const []const u8) bool {
    var at: usize = 0;
    for (needles) |n| {
        const i = std.mem.indexOfPos(u8, hay, at, n) orelse return false;
        at = i + n.len;
    }
    return true;
}

/// each of `needles` once in `hay`, in their order
fn onceInOrder(hay: []const u8, needles: []const []const u8) bool {
    for (needles) |n| if (!once(hay, n)) return false;
    return inOrder(hay, needles);
}

/// `text` with every `from` made `to`, into `buf`
fn swap(buf: []u8, text: []const u8, from: []const u8, to: []const u8) []const u8 {
    const n = std.mem.replacementSize(u8, text, from, to);
    _ = std.mem.replace(u8, text, from, to, buf[0..n]);
    return buf[0..n];
}

/// `hay` from `first` to the end of the first `last` after it
fn between(hay: []const u8, first: []const u8, last: []const u8) ?[]const u8 {
    const i = std.mem.indexOf(u8, hay, first) orelse return null;
    const j = std.mem.indexOfPos(u8, hay, i, last) orelse return null;
    return hay[i .. j + last.len];
}

test "turboexl3 ling kdanorm texts: the launch is dense's prepare with the norm-gate in its load's place (text_prep's first and last pieces around it); text_prep is its three pieces, its bytes kept" {
    // the served piece's, the host copy's and the prepare's statements these tests pin
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/kdanorm.metal"));
    try testing.expectEqualStrings(dense.prep_1 ++ dense.prep_2_1 ++ dense.prep_2_2 ++ dense.prep_3, dense.text_prep);
    // text_prep as it was before its cut (c2783f5's kernels/dense/prep.metal: 1085 B)
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(dense.text_prep, &h, .{});
    const hex = std.fmt.bytesToHex(h, .lower);
    try testing.expectEqual(@as(usize, 1085), dense.text_prep.len);
    try testing.expectEqualStrings("110e3a6a2eb5f59a70aca17ee2b3cd8e7bf376d24e7722d7e97ae862573822e4", &hex);
    try testing.expectEqualStrings(dense.prep_1 ++ kgate ++ dense.prep_2_2 ++ dense.prep_3, text_normprep);
    // the cut is at the load: prep_2_1 is its declaration and its load, and the gate makes the same four values
    try testing.expect(once(dense.prep_2_1, pin[24]) and once(dense.prep_2_1, pin[34]));
    try testing.expect(once(kgate, pin[25]) and once(kgate, pin[35]));
    try testing.expect(std.mem.indexOf(u8, text_normprep, pin[24]) == null);
}

test "turboexl3 ling kdanorm operands: a KDA layer's y [1, S, 64, 128], z [1, S, 8192] at 0, norm_w [128], sigtab [65536] in bf16, eps f32 0-d and o_proj's suh f16 [8192] fit; another dtype, rank, length, width or z offset does not" {
    const S = 1965;
    const y_sh = [_]c_int{ 1, S, 64, 128 };
    const z_sh = [_]c_int{ 1, S, 8192 };
    const nw_sh = [_]c_int{128};
    const tab_sh = [_]c_int{1 << 16};
    const suh_sh = [_]c_int{8192};
    const ok: Shapes = .{
        .y = .{ .dt = .bfloat16, .sh = &y_sh },
        .z = .{ .dt = .bfloat16, .sh = &z_sh },
        .z_off = 0,
        .z_stride = 8192,
        .norm_w = .{ .dt = .bfloat16, .sh = &nw_sh },
        .eps = .{ .dt = .float32, .sh = &.{} },
        .sigtab = .{ .dt = .bfloat16, .sh = &tab_sh },
        .suh = .{ .dt = .float16, .sh = &suh_sh },
    };
    const geo = shapeOk(ok) orelse return error.TestExpectedEqual;
    try testing.expectEqual(Geometry{ .rows = S, .hv = 64, .dim = 8192 }, geo);
    // f16 throughout fits too (the host's other dtype)
    var f = ok;
    f.y.dt = .float16;
    f.z.dt = .float16;
    f.norm_w.dt = .float16;
    f.sigtab.dt = .float16;
    try testing.expect(shapeOk(f) != null);
    // a wider z row with the heads inside it fits; past it does not
    var w = ok;
    const zw = [_]c_int{ 1, S, 8192 + 256 };
    w.z = .{ .dt = .bfloat16, .sh = &zw };
    w.z_stride = 8192 + 256;
    w.z_off = 256;
    try testing.expect(shapeOk(w) != null);
    w.z_off = 257;
    try testing.expect(shapeOk(w) == null);
    const y_dv64 = [_]c_int{ 1, S, 128, 64 };
    const y_r3 = [_]c_int{ S, 64, 128 };
    const z_rows = [_]c_int{ 1, S + 1, 8192 };
    const nw_64 = [_]c_int{64};
    const tab_half = [_]c_int{1 << 15};
    const suh_4096 = [_]c_int{4096};
    const eps_1 = [_]c_int{1};
    var bad: [11]Shapes = @splat(ok);
    bad[0].y.dt = .float32;
    bad[1].z.dt = .float16;
    bad[2].sigtab.dt = .float32;
    bad[3].suh.dt = .bfloat16;
    bad[4].eps = .{ .dt = .float32, .sh = &eps_1 };
    bad[5].y.sh = &y_dv64;
    bad[6].y.sh = &y_r3;
    bad[7].z.sh = &z_rows;
    bad[8].norm_w.sh = &nw_64;
    bad[9].sigtab.sh = &tab_half;
    bad[10].suh.sh = &suh_4096;
    for (bad) |b| try testing.expect(shapeOk(b) == null);
}

test "turboexl3 ling kdanorm statements: the gate is the host's norm-gate (its test-only copy at bc6c5f2c): its sum of squares statement for statement, each element's normed, z, sigmoid-table and output statements at the prepare's channel c, and the prepare's value of that output" {
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/kdanorm.metal"));
    // the host's norm-gate at bc6c5f2c, a test-only copy under its `//! copied_from:` line
    const host_copy = @embedFile("kernels/test/copy/gdn_normgate.metal");
    // both carry the copy's origin
    try testing.expect(std.mem.startsWith(u8, std.mem.trimStart(u8, kgate, " "), pin[0]));
    try testing.expect(std.mem.startsWith(u8, host_copy, pin[0]));
    // the reduction: the host's statements in its order (the copy's per-element loop opens as its sum's does), and
    // the gate's, each once
    try testing.expect(inOrder(host_copy, pin[1..9]));
    try testing.expect(onceInOrder(kgate, pin[1..9]));
    // a lane, a row and a head: the host's (its threadgroup of 32, rows on y, heads on z) and the prepare's (a
    // simdgroup a 128-block, rows on y, blocks on x), the gate's head the prepare's block
    for (pin[38..41]) |p| try testing.expect(once(host_copy, p));
    for (pin[41..44]) |p| try testing.expect(once(dense.prep_1, p));
    try testing.expect(once(kgate, pin[36]) and once(kgate, pin[37]));
    // each element's statements at channel c of head b (the host's lane * 4 + i): its x, normed, z and sigmoid
    try testing.expect(once(host_copy, pin[5]) and once(kgate, pin[31]));
    try testing.expect(std.mem.indexOf(u8, pin[1], pin[32]) != null and std.mem.startsWith(u8, pin[31][std.mem.indexOf(u8, pin[31], pin[32]).? + pin[32].len ..], pin[33]));
    var b1: [256]u8 = undefined;
    var b2: [256]u8 = undefined;
    try testing.expect(once(host_copy, pin[9]) and once(kgate, pin[10]));
    try testing.expectEqualStrings(pin[10], swap(&b2, swap(&b1, pin[9], pin[20], pin[21]), pin[22], pin[23]));
    try testing.expect(once(host_copy, pin[11]) and once(host_copy, pin[12]) and once(kgate, pin[13]));
    try testing.expect(std.mem.indexOf(u8, pin[11], pin[19]) != null);
    try testing.expectEqualStrings(pin[13], pin[28] ++ pin[19] ++ pin[29]);
    try testing.expect(once(host_copy, pin[30]) and once(kgate, pin[30]));
    try testing.expect(once(host_copy, pin[14]) and std.mem.endsWith(u8, pin[14], pin[15]) and once(kgate, pin[15]));
    // the output: the host's sigmoid gate's (SWISH 0) value, then the prepare's value of it at the prepare's index
    try testing.expect(once(host_copy, pin[16]) and std.mem.endsWith(u8, pin[16], pin[18] ++ ";"));
    try testing.expect(once(kgate, pin[17]) and std.mem.endsWith(u8, pin[17], pin[18] ++ ";"));
    for ([_][]const u8{ pin[24], pin[25] }) |v| try testing.expect(std.mem.startsWith(u8, v, pin[26]) and std.mem.indexOf(u8, v, pin[27]) != null);
}

test "turboexl3 ling kdanorm attribution: the gate's copied statements are its copied_from source's: the sum of squares from its first statement to the precise rsqrt is the test-only copy's line for line (indent aside), and the copy is mlx-serve's kernels/gdn/gdn_normgate.metal at bc6c5f2c byte for byte (that file's sha256 in mlx-serve's kernels/digests.txt)" {
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/kdanorm.metal"));
    const host_copy = @embedFile("kernels/test/copy/gdn_normgate.metal");
    // the copy: its copied_from line, then mlx-serve's file at bc6c5f2c, whose digests.txt entry (there and at main
    // 66aad949) is this sha256
    try testing.expect(std.mem.startsWith(u8, host_copy, pin[0]));
    const body = host_copy[(std.mem.indexOfScalar(u8, host_copy, '\n') orelse return error.TestExpectedEqual) + 1 ..];
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(body, &h, .{});
    const hex = std.fmt.bytesToHex(h, .lower);
    try testing.expectEqualStrings("1a0a459d1e5f0ef114a19016f9324954a3ca9d1db7ed9f4910acaa594fc07de6", &hex);
    // the served piece's copied block and the copy's, from the reduction's first statement to its last: the same lines
    // once each line's indent is trimmed, as many of them, so a statement added, dropped or changed on either side fails
    const want = between(host_copy, pin[1], pin[8]) orelse return error.TestExpectedEqual;
    const got = between(kgate, pin[1], pin[8]) orelse return error.TestExpectedEqual;
    var wi = std.mem.splitScalar(u8, want, '\n');
    var gi = std.mem.splitScalar(u8, got, '\n');
    var n: usize = 0;
    while (wi.next()) |w| : (n += 1) {
        const l = gi.next() orelse return error.TestExpectedEqual;
        try testing.expectEqualStrings(std.mem.trimStart(u8, w, " "), std.mem.trimStart(u8, l, " "));
    }
    try testing.expect(gi.next() == null);
    try testing.expectEqual(@as(usize, 9), n);
}
