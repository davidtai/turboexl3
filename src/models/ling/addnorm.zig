//! Ling-3.1-flash's residual add and RMSNorm folded into the next linears' prepare (`addNormLinears`): at a decode
//! layer's mixer input, one launch (one threadgroup of 1024 threads per row) computes h + delta (the residual, in h's
//! dtype) and its RMSNorm with w (mlx-serve's add_norm statements, MLX's fast rms_norm order), keeps the normed row in
//! threadgroup memory, runs each EXL3 member's DEC-ONE prepare on it (text_prep_s's statements: f16 round, the member's
//! suh, H128, lane order, the 128-block sums) into the joined prepared rows (member m's rows at m * ROWS, `pair.zig`'s
//! layout), and zeroes the members' arrival counters. Each member then runs its own GEMV over its prepared rows
//! (`dense.Dense.linearFromPrepared`: DEC-ONE's dec0 where DEC-ONE takes the call, else the chain's GEMV and finish), so
//! each member's columns are `linear`'s bytes of the normed rows, the residual is the add's, and the normed rows (also
//! written, for the readers that are not EXL3) are MLX's rms_norm's. `addNormPrepare` stops after the launch, for a
//! caller whose GEMV reads the prepared rows itself.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const format = @import("../../core/format.zig");
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const root = @import("../../root.zig");

/// The width the launch serves: Ling's hidden size (1024 threads of four elements, one 128-block per simdgroup)
pub const width: u32 = 4096;
/// The EXL3 members one launch prepares
pub const max_members: usize = 3;

// ── texts ──

/// one member's prepare of the normed row (a function in the launch's header)
pub const an_prep = @embedFile("kernels/addnorm/an_prep.metal");
/// the residual add and its RMSNorm, the normed row kept in threadgroup memory
pub const an_norm = @embedFile("kernels/addnorm/an_norm.metal");
/// member m's prepare of the row
pub const an_m0 = @embedFile("kernels/addnorm/an_m0.metal");
pub const an_m1 = @embedFile("kernels/addnorm/an_m1.metal");
pub const an_m2 = @embedFile("kernels/addnorm/an_m2.metal");
/// the members' arrival counters zeroed
pub const an_cnt = @embedFile("kernels/addnorm/an_cnt.metal");

/// The launch for one, two or three members
pub const text_an1: [:0]const u8 = an_norm ++ an_m0 ++ an_cnt;
pub const text_an2: [:0]const u8 = an_norm ++ an_m0 ++ an_m1 ++ an_cnt;
pub const text_an3: [:0]const u8 = an_norm ++ an_m0 ++ an_m1 ++ an_m2 ++ an_cnt;
pub const header: [:0]const u8 = an_prep;

const an_out = [_][*:0]const u8{ "H_NEW", "HN", "jxp", "jxs", "cnt" };
const an1_in = [_][*:0]const u8{ "H", "X", "W", "eps", "suh0" };
const an2_in = [_][*:0]const u8{ "H", "X", "W", "eps", "suh0", "suh1" };
const an3_in = [_][*:0]const u8{ "H", "X", "W", "eps", "suh0", "suh1", "suh2" };

// ── kernels: made at their first call ──

const kernel_lock = Global(std.atomic.Value(u32), "ling_addnorm_kernel_lock", std.atomic.Value(u32).init(0));
const kernels = Global([max_members]?mlx.mlx_fast_metal_kernel, "ling_addnorm_kernels", @splat(null));

fn kernelOf(n: usize) !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    const slot = &kernels.get().*[n - 1];
    if (slot.*) |made| return made;
    const made = switch (n) {
        1 => ks.newKernel("turboexl3_ling_addnorm1", &an1_in, &an_out, text_an1, header, true),
        2 => ks.newKernel("turboexl3_ling_addnorm2", &an2_in, &an_out, text_an2, header, true),
        3 => ks.newKernel("turboexl3_ling_addnorm3", &an3_in, &an_out, text_an3, header, true),
        else => return error.BadExl3Shape,
    } orelse return error.KernelCreateFailed;
    slot.* = made;
    return made;
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernels };

// ── the plan ──

/// A call's members: each one's `Dense`, its output width, and its arrival counters' offset and count in the launch's
/// counters (count 0: the chain's GEMV and finish take its rows, no counters)
pub const Plan = struct {
    n: usize,
    rows: usize,
    dens: [max_members]*dense.Dense = undefined,
    out: [max_members]u32 = @splat(0),
    cnt_off: [max_members]u32 = @splat(0),
    cnt_len: [max_members]u32 = @splat(0),
    /// the launch's counters: every member's back to back, at least 8 (MLX binds a smaller input in constant space)
    ncnt: u32 = 8,
};

/// The plan of a call of `rows` rows in `xd` over `members` (each as `linear` takes it, input width `width`), or null
/// where the launch does not serve (the caller keeps the add, the norm and each member's `linear`): no member or more
/// than `max_members`, rows past the decode rows, or a member whose `linear` takes neither DEC-ONE nor the chain there.
pub fn plan(members: []const root.Proj, xd: mlx.mlx_dtype, rows: usize, dec: format.Decode) !?Plan {
    if (members.len == 0 or members.len > max_members) return null;
    if (rows < 1 or rows > decode.Kernels.max_rows) return null;
    if (xd != .bfloat16 and xd != .float16) return null;
    var p: Plan = .{ .n = members.len, .rows = rows };
    var off: u32 = 0;
    for (members, 0..) |m, i| {
        const d = try root.denseOf(m, width, xd, dec);
        p.dens[i] = d;
        p.out[i] = d.shape.out;
        p.cnt_off[i] = off;
        if (d.decTakes(rows)) {
            p.cnt_len[i] = try d.decCounters(rows);
        } else if (!d.preparedTakes(rows)) return null;
        off += p.cnt_len[i];
    }
    p.ncnt = @max(off, 8);
    return p;
}

// ── the calls ──

/// What the launch makes: the residual and the normed rows (h's shape and dtype), the joined prepared rows jxp f16
/// [n * rows, width] (member m's at m * rows, lane order) and their block sums jxs f32 [n * rows, width / 128], and the
/// zeroed counters cnt u32 [plan.ncnt]. The caller frees all five.
pub const Prepared = struct {
    h: mlx.mlx_array,
    normed: mlx.mlx_array,
    jxp: mlx.mlx_array,
    jxs: mlx.mlx_array,
    cnt: mlx.mlx_array,

    pub fn deinit(self: *Prepared) void {
        inline for (.{ self.h, self.normed, self.jxp, self.jxs, self.cnt }) |a| root.freeArr(a);
        self.* = undefined;
    }
};

/// The launch alone over h, delta [..., width] (one dtype, bf16 or f16), w [width] in that dtype and eps 0-dim f32 (the
/// norm's), for `p.n` members whose suh (f16 [width], in the plan's order) it prepares with; `ncnt` counters zeroed (at
/// least `p.ncnt`: a caller whose GEMV reads the counters itself passes its own count).
pub fn addNormPrepare(s: mlx.mlx_stream, h: mlx.mlx_array, delta: mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array, suhs: []const mlx.mlx_array, rows: usize, ncnt: u32) !Prepared {
    const n = suhs.len;
    if (n == 0 or n > max_members or rows < 1 or rows > decode.Kernels.max_rows or ncnt < 8) return error.BadExl3Shape;
    const xd = mlx.mlx_array_dtype(h);
    if ((xd != .bfloat16 and xd != .float16) or mlx.mlx_array_dtype(delta) != xd or mlx.mlx_array_dtype(w) != xd) return error.BadExl3Dtype;
    if (mlx.mlx_array_dtype(eps) != .float32 or mlx.mlx_array_ndim(eps) != 0) return error.BadExl3Dtype;
    if (mlx.getShape(w).len != 1 or mlx.getShape(w)[0] != @as(c_int, @intCast(width))) return error.BadExl3Shape;
    for (suhs) |u| {
        if (mlx.getShape(u).len != 1 or mlx.getShape(u)[0] != @as(c_int, @intCast(width))) return error.BadExl3Shape;
        if (mlx.mlx_array_dtype(u) != .float16) return error.BadExl3Dtype;
    }
    const hv = try root.rowsOf(h, width, s);
    defer root.freeArr(hv.x2);
    const dv = try root.rowsOf(delta, width, s);
    defer root.freeArr(dv.x2);
    if (hv.rows != rows or dv.rows != rows) return error.BadExl3Shape;
    const tn: u32 = width / 4;
    const r: c_int = @intCast(rows);
    const nr: c_int = @intCast(n * rows);
    const cfg = try ks.buildConfig(.{
        .grid = .{ tn * @as(u32, @intCast(rows)), 1, 1 },
        .threadgroup = .{ tn, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ r, @intCast(width) }, &[_]c_int{ r, @intCast(width) }, &[_]c_int{ nr, @intCast(width) }, &[_]c_int{ nr, @intCast(width / 128) }, &[_]c_int{@intCast(ncnt)} },
        .out_dtypes = &.{ xd, xd, .float16, .float32, .uint32 },
        .template = &.{
            .{ .name = "T", .value = .{ .dtype = xd } },
            .{ .name = "D", .value = .{ .int = @intCast(width) } },
            .{ .name = "TN", .value = .{ .int = @intCast(tn) } },
            .{ .name = "ROWS", .value = .{ .int = r } },
            .{ .name = "NCNT", .value = .{ .int = @intCast(ncnt) } },
            .{ .name = "NTH", .value = .{ .int = @intCast(tn * @as(u32, @intCast(rows))) } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var ins: [4 + max_members]mlx.mlx_array = undefined;
    ins[0] = hv.x2;
    ins[1] = dv.x2;
    ins[2] = w;
    ins[3] = eps;
    for (suhs, 0..) |u, i| ins[4 + i] = u;
    var o: [5]mlx.mlx_array = undefined;
    try ks.applyConfig(try kernelOf(n), cfg, ins[0 .. 4 + n], &o, s);
    errdefer for (o) |a| root.freeArr(a);
    // the residual and the normed rows in h's shape
    var out: Prepared = .{ .h = mlx.mlx_array_new(), .normed = mlx.mlx_array_new(), .jxp = o[2], .jxs = o[3], .cnt = o[4] };
    errdefer {
        root.freeArr(out.h);
        root.freeArr(out.normed);
    }
    const hsh = mlx.getShape(h);
    try mlx.check(mlx.mlx_reshape(&out.h, o[0], hsh.ptr, hsh.len, s));
    try mlx.check(mlx.mlx_reshape(&out.normed, o[1], hsh.ptr, hsh.len, s));
    root.freeArr(o[0]);
    root.freeArr(o[1]);
    return out;
}

/// What `addNormLinears` returns: the residual and the normed rows (h's shape and dtype), and member m's output `ys[m]`
/// [..., out_m] in h's dtype (h's leading shape), for `n` members. The caller frees them (`deinit`).
pub const Out = struct {
    h: mlx.mlx_array,
    normed: mlx.mlx_array,
    ys: [max_members]mlx.mlx_array = @splat(.{ .ctx = null }),
    n: usize = 0,

    pub fn deinit(self: *Out) void {
        root.freeArr(self.h);
        root.freeArr(self.normed);
        for (self.ys[0..self.n]) |y| root.freeArr(y);
        self.* = undefined;
    }
};

/// h + delta, its RMSNorm with w (eps: 0-dim f32), and the linears `members` (each as `linear` takes it, input width
/// `width`) of the normed rows: the residual, the normed rows and each member's output, each the bytes of the composed
/// add, MLX's rms_norm and the member's `linear`. h, delta [..., width] in one dtype (bf16 or f16), w [width] in it.
/// error.Exl3EngineNotImplemented where `plan` declines (the caller runs the composed ops).
pub fn addNormLinears(s: mlx.mlx_stream, h: mlx.mlx_array, delta: mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array, members: []const root.Proj, dec: format.Decode) !Out {
    const xd = mlx.mlx_array_dtype(h);
    const hsh = mlx.getShape(h);
    if (hsh.len < 1 or hsh[hsh.len - 1] != @as(c_int, @intCast(width))) return error.BadExl3Shape;
    var rows: c_int = 1;
    for (hsh[0 .. hsh.len - 1]) |d| rows *= d;
    if (rows < 1) return error.BadExl3Shape;
    const p = (try plan(members, xd, @intCast(rows), dec)) orelse return error.Exl3EngineNotImplemented;
    var suhs: [max_members]mlx.mlx_array = undefined;
    for (members, 0..) |m, i| suhs[i] = m.suh;
    const pre = try addNormPrepare(s, h, delta, w, eps, suhs[0..p.n], p.rows, p.ncnt);
    defer {
        root.freeArr(pre.jxp);
        root.freeArr(pre.jxs);
        root.freeArr(pre.cnt);
    }
    var out: Out = .{ .h = pre.h, .normed = pre.normed };
    errdefer out.deinit();
    const ys = try linearsOfPrepared(s, &pre, p, members);
    out.ys = ys;
    out.n = p.n;
    // each member's output in h's leading shape
    for (out.ys[0..p.n], 0..) |*y, i| {
        var shp: [8]c_int = undefined;
        if (hsh.len > shp.len) return error.BadExl3Shape;
        @memcpy(shp[0 .. hsh.len - 1], hsh[0 .. hsh.len - 1]);
        shp[hsh.len - 1] = @intCast(p.out[i]);
        var yr = mlx.mlx_array_new();
        errdefer root.freeArr(yr);
        try mlx.check(mlx.mlx_reshape(&yr, y.*, &shp, hsh.len, s));
        root.freeArr(y.*);
        y.* = yr;
    }
    return out;
}

/// Each member's GEMV over its rows of `pre` (`dense.Dense.linearFromPrepared`): ys[m] [rows, out_m]
fn linearsOfPrepared(s: mlx.mlx_stream, pre: *const Prepared, p: Plan, members: []const root.Proj) ![max_members]mlx.mlx_array {
    var ys: [max_members]mlx.mlx_array = @splat(.{ .ctx = null });
    errdefer for (ys) |y| root.freeArr(y);
    const r: c_int = @intCast(p.rows);
    const st2 = [_]c_int{ 1, 1 };
    const st1 = [_]c_int{1};
    for (members, 0..) |m, i| {
        const r0: c_int = @as(c_int, @intCast(i)) * r;
        var xp = mlx.mlx_array_new();
        defer root.freeArr(xp);
        var xs = mlx.mlx_array_new();
        defer root.freeArr(xs);
        var cnt = mlx.mlx_array_new();
        defer root.freeArr(cnt);
        try mlx.check(mlx.mlx_slice(&xp, pre.jxp, &[_]c_int{ r0, 0 }, 2, &[_]c_int{ r0 + r, @intCast(width) }, 2, &st2, 2, s));
        try mlx.check(mlx.mlx_slice(&xs, pre.jxs, &[_]c_int{ r0, 0 }, 2, &[_]c_int{ r0 + r, @intCast(width / 128) }, 2, &st2, 2, s));
        // the chain reads no counters: any 8 of them stand in
        const off: c_int = @intCast(p.cnt_off[i]);
        const len: c_int = @intCast(if (p.cnt_len[i] > 0) p.cnt_len[i] else 8);
        const end: c_int = @min(off + len, @as(c_int, @intCast(p.ncnt)));
        try mlx.check(mlx.mlx_slice(&cnt, pre.cnt, &[_]c_int{if (p.cnt_len[i] > 0) off else 0}, 1, &[_]c_int{if (p.cnt_len[i] > 0) end else 8}, 1, &st1, 1, s));
        ys[i] = try p.dens[i].linearFromPrepared(s, xp, xs, cnt, m.trellis, m.svh, p.rows);
    }
    return ys;
}

// ── tests (CPU) ──

const testing = std.testing;
const support = @import("../../core/test_support.zig");

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

test "turboexl3 ling addnorm texts: the launches are the norm, the members' prepares in member order and the counters; the prepare is text_prep_s's statements on the row in threadgroup memory, the counters prep_sc's loop, the norm mlx-serve's add_norm statements" {
    try testing.expectEqualStrings(text_an1, an_norm ++ an_m0 ++ an_cnt);
    try testing.expectEqualStrings(text_an2, an_norm ++ an_m0 ++ an_m1 ++ an_cnt);
    try testing.expectEqualStrings(text_an3, an_norm ++ an_m0 ++ an_m1 ++ an_m2 ++ an_cnt);
    const pin = comptime support.pins(@embedFile("kernels/test/pins/addnorm_texts.metal"));
    // the prepare: text_prep_s's H128, rounding, lane order, store and block sum, verbatim
    for (pin[0..9]) |line| {
        try testing.expectEqual(@as(usize, 1), count(an_prep, line));
        try testing.expectEqual(@as(usize, 1), count(dense.text_prep_s, line));
    }
    // its x: the normed row in threadgroup memory where text_prep_s reads the device rows
    try testing.expectEqual(@as(usize, 1), count(an_prep, pin[9]));
    try testing.expectEqual(@as(usize, 0), count(an_prep, pin[10]));
    try testing.expectEqual(@as(usize, 1), count(dense.text_prep_s, pin[10]));
    // the counters: prep_sc's loop
    try testing.expectEqual(@as(usize, 1), count(an_cnt, pin[11]));
    try testing.expectEqual(@as(usize, 1), count(dense.text_prep_sc, pin[11]));
    // each member reads its own suh into its own rows
    for ([_][]const u8{ an_m0, an_m1, an_m2 }, pin[12..15]) |m, args| try testing.expectEqual(@as(usize, 1), count(m, args));
    // the norm: mlx-serve's add_norm (its plain arm) statement for statement
    for (pin[15..]) |line| try testing.expectEqual(@as(usize, 1), count(an_norm, line));
    // every operand and template constant the launch binds is named in its text, and no MLX shape-buffer suffix
    for ([_][:0]const u8{ text_an1, text_an2, text_an3 }, [_][]const [*:0]const u8{ &an1_in, &an2_in, &an3_in }) |txt, ins| {
        for (ins) |n| try testing.expect(count(txt, std.mem.span(n)) > 0);
        for (an_out) |n| try testing.expect(count(txt, std.mem.span(n)) > 0);
        for ([_][]const u8{ "T", "D", "TN", "ROWS", "NCNT", "NTH" }) |n| try testing.expect(count(txt, n) > 0);
        for (ins) |n| for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| {
            var buf: [24]u8 = undefined;
            try testing.expectEqual(@as(usize, 0), count(txt, std.fmt.bufPrint(&buf, "{s}{s}", .{ std.mem.span(n), suf }) catch unreachable));
        };
    }
}

test "turboexl3 ling addnorm plan: no member, more than three, rows outside 1 to 16 and float32 rows are refused before any Dense is made" {
    const none: mlx.mlx_array = .{ .ctx = null };
    const m: root.Proj = .{ .trellis = none, .suh = none, .svh = none };
    try testing.expect((try plan(&[_]root.Proj{}, .bfloat16, 1, .mul1)) == null);
    try testing.expect((try plan(&[_]root.Proj{ m, m, m, m }, .bfloat16, 1, .mul1)) == null);
    try testing.expect((try plan(&[_]root.Proj{m}, .bfloat16, 0, .mul1)) == null);
    try testing.expect((try plan(&[_]root.Proj{m}, .bfloat16, decode.Kernels.max_rows + 1, .mul1)) == null);
    try testing.expect((try plan(&[_]root.Proj{m}, .float32, 1, .mul1)) == null);
}
