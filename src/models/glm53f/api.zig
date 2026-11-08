//! GLM-5.3-Flash's host helpers (root re-exports this file as `glm53f`): the trunk linears that run at prefill rows, and
//! the pg2 table a host builds from one tile code a linear (`pg2Table`). The outside plugin turboexl3_glm53f passes the
//! codes (its TURBOEXL3_GLM53F_PG2); fused_test.zig holds every candidate tile at every linear here to the chain's words and
//! times them, which is where a host's codes come from.
const std = @import("std");
const pgemm = @import("../../core/pgemm.zig");
/// The shared expert's gate | up in one DEC-ONE launch (the plugin's TURBOEXL3_GLM53F_SHARED_GU)
pub const shared_gu = @import("shared_gu.zig");
/// Two linears of one input in one DEC-ONE launch, each at its own K split (a DSA layer's q_a and kv_a: the plugin's
/// TURBOEXL3_GLM53F_SMALLOP_FUSE)
pub const pair_ks = @import("pair_ks.zig");

/// One trunk linear at prefill rows: its name, in (K) and out (N)
pub const Linear = struct { name: []const u8, k: u32, n: u32 };

/// The trunk linears of every GLM-5.3-Flash SC pack that run at prefill rows (34 KDA layers, 11 DSA layers, the 42 MoE
/// layers' shared experts, the 3 dense layers' MLP; config.json's layer_types; a gate and its up share a shape), each (K,
/// N) once, in the order of `pg2Table`'s codes. The rate does not change a shape.
pub const trunk = [_]Linear{
    .{ .name = "kda.qkv_proj", .k = 4096, .n = 24576 },
    .{ .name = "kda.o_proj", .k = 8192, .n = 4096 },
    .{ .name = "dsa.q_a_proj", .k = 4096, .n = 1536 },
    .{ .name = "dsa.q_b_proj", .k = 1536, .n = 16384 },
    .{ .name = "dsa.kv_a_proj_with_mqa", .k = 4096, .n = 512 },
    .{ .name = "dsa.o_proj", .k = 16384, .n = 4096 },
    .{ .name = "indexer.wq_b", .k = 1536, .n = 4096 },
    .{ .name = "shared.gate_proj/up_proj", .k = 4096, .n = 2048 },
    .{ .name = "shared.down_proj", .k = 2048, .n = 4096 },
    .{ .name = "dense.gate_proj/up_proj", .k = 4096, .n = 12288 },
    .{ .name = "dense.down_proj", .k = 12288, .n = 4096 },
};

/// A tile code: '0' none (the chain), '1' to '9' `pgemm.candidates[0..9]`, 'a' to 'z' `pgemm.candidates[9..]`; null for any
/// other character or a candidate past the list
pub fn candidateOf(code: u8) ??pgemm.Tile {
    const i: usize = switch (code) {
        '0' => return @as(?pgemm.Tile, null),
        '1'...'9' => code - '1',
        'a'...'z' => code - 'a' + 9,
        else => return null,
    };
    if (i >= pgemm.candidates.len) return null;
    return @as(?pgemm.Tile, pgemm.candidates[i]);
}

/// The code of `pgemm.candidates[i]` (`candidateOf`'s inverse)
pub fn codeOf(i: usize) u8 {
    std.debug.assert(i < pgemm.candidates.len and i < 35);
    return if (i < 9) @intCast('1' + i) else @intCast('a' + i - 9);
}

/// The pg2 table `codes` names, one code a `trunk` linear in order, written to `out`: the linears whose code is not '0',
/// each at its candidate. error.Glm53fPg2Codes for another length, an unknown code or a candidate past the list.
pub fn pg2Table(codes: []const u8, out: *[trunk.len]pgemm.Entry) error{Glm53fPg2Codes}![]const pgemm.Entry {
    if (codes.len != trunk.len) return error.Glm53fPg2Codes;
    var n: usize = 0;
    for (codes, trunk) |c, l| {
        const t = candidateOf(c) orelse return error.Glm53fPg2Codes;
        if (t) |tile| {
            out[n] = .{ .k = l.k, .n = l.n, .tile = tile };
            n += 1;
        }
    }
    return out[0..n];
}

const testing = std.testing;

test "glm53f api: the trunk's shapes are distinct, every candidate's BN divides each N, and the pg2 codes name the candidates in order" {
    for (trunk, 0..) |l, i| {
        for (trunk[i + 1 ..]) |m| try testing.expect(l.k != m.k or l.n != m.n);
        for (pgemm.candidates) |t| try testing.expect(l.n % t.bn == 0);
    }
    try testing.expect(pgemm.candidates.len <= 35);
    for (0..pgemm.candidates.len) |i| try testing.expect(std.meta.eql(candidateOf(codeOf(i)).?.?, pgemm.candidates[i]));
    try testing.expect(candidateOf('0').? == null);
    try testing.expect(candidateOf('A') == null and candidateOf('-') == null);
    try testing.expect(candidateOf(codeOf(pgemm.candidates.len - 1) + 1) == null);
    var buf: [trunk.len]pgemm.Entry = undefined;
    try testing.expectEqual(@as(usize, 0), (try pg2Table("00000000000", &buf)).len);
    const all = try pg2Table("12345678912", &buf);
    try testing.expectEqual(trunk.len, all.len);
    for (all, trunk, 0..) |e, l, i| {
        try testing.expect(e.k == l.k and e.n == l.n);
        try testing.expect(std.meta.eql(e.tile, pgemm.candidates[if (i < 9) i else i - 9]));
    }
    const some = try pg2Table("20000a00000", &buf);
    try testing.expectEqual(@as(usize, 2), some.len);
    try testing.expect(some[0].k == 4096 and some[0].n == 24576 and std.meta.eql(some[0].tile, pgemm.candidates[1]));
    try testing.expect(some[1].k == 16384 and some[1].n == 4096 and std.meta.eql(some[1].tile, pgemm.candidates[9]));
    for ([_][]const u8{ "", "0000000000", "000000000000", "0000000000A", "0000000000-", "0000000000z" }) |bad|
        try testing.expectError(error.Glm53fPg2Codes, pg2Table(bad, &buf));
}

test "glm53f api: pg2's candidate tiles are admitted and distinct, and a host's table answers before pgemm's own table, which keeps every other shape" {
    for (pgemm.candidates, 0..) |t, i| {
        try testing.expect(t.admitted());
        for (pgemm.candidates[i + 1 ..]) |u| try testing.expect(!std.meta.eql(t, u));
    }
    const saved = pgemm.host_table.get().*;
    defer pgemm.host_table.get().* = saved;
    pgemm.host_table.get().* = null;
    try testing.expect(pgemm.tileOf(4096, 24576) == null);
    try testing.expect(std.meta.eql(pgemm.tileOf(8192, 5120).?, pgemm.table[1].tile));
    var buf: [trunk.len]pgemm.Entry = undefined;
    pgemm.host_table.get().* = try pg2Table("80000100000", &buf);
    try testing.expect(std.meta.eql(pgemm.tileOf(4096, 24576).?, pgemm.candidates[7]));
    try testing.expect(std.meta.eql(pgemm.tileOf(16384, 4096).?, pgemm.candidates[0]));
    try testing.expect(pgemm.tileOf(8192, 4096) == null);
    try testing.expect(std.meta.eql(pgemm.tileOf(5120, 512).?, pgemm.table[5].tile));
    const theirs = [_]pgemm.Entry{.{ .k = 8192, .n = 5120, .tile = pgemm.candidates[4] }};
    pgemm.host_table.get().* = &theirs;
    try testing.expect(std.meta.eql(pgemm.tileOf(8192, 5120).?, pgemm.candidates[4]));
}
