//! Hy4-flash's pg2 tiles (core/pgemm.zig): the tile per trunk (K, N) on Apple M5 Max, 128 GB, the host's table for pg2's
//! switch. With it on, a trunk linear at prompt rows runs its matmul and its finish as one launch at its tabled shape
//! (`Dense.linearOn`: where MLX runs its regular kernel, and with split K where MLX splits K at a device-independent
//! partition size: o_proj at the sliding layers, 16384 -> 3072, and the dense MLP's down, 13824 -> 3072), and each
//! arm with a finish of its own (gate.zig, swiglu.zig, qrope.zig) runs pg2's GEMM with that finish at its shape. Every
//! word is the chain's: pg2's GEMM gives MLX's z word for word at every tile (hy4f_test.zig's pg2arms tests at every
//! Hy4 shape and candidate tile), and its epilogue is the finish's statements.
//! A host serves it from one switch (`serve`), set at load before the first prompt, in each copy of turboexl3 it runs
//! (the quant plugin's for the plain linears, the model plugin's for the arms).
const pgemm = @import("../../core/pgemm.zig");

/// A trunk linear at prompt rows: K = in, N = out
pub const Linear = struct { name: []const u8, k: u32, n: u32 };

/// Hy4-flash's trunk linears (the 2.08 pack's K4 trunk; kv_a as stored, 640 outputs): the indexed layers' and the
/// sliding layers' attention, the latent MoE's projections and its shared expert, layer 0's dense MLP
pub const trunk = [_]Linear{
    .{ .name = "q_a", .k = 3072, .n = 1024 },
    .{ .name = "q_b.indexed", .k = 1024, .n = 8192 },
    .{ .name = "q_b.sliding", .k = 1024, .n = 16384 },
    .{ .name = "kv_a", .k = 3072, .n = 640 },
    .{ .name = "linear_gate.indexed", .k = 3072, .n = 8192 },
    .{ .name = "linear_gate.sliding", .k = 3072, .n = 16384 },
    .{ .name = "o_proj.indexed", .k = 8192, .n = 3072 },
    .{ .name = "o_proj.sliding", .k = 16384, .n = 3072 },
    .{ .name = "routed_expert_down/shared.gate_up", .k = 3072, .n = 1536 },
    .{ .name = "routed_expert_up/shared.down", .k = 1536, .n = 3072 },
    .{ .name = "dense.gate_up", .k = 3072, .n = 13824 },
    .{ .name = "dense.down", .k = 13824, .n = 3072 },
};

/// `pgemm.candidates[i]`
fn c(i: usize) pgemm.Tile {
    return pgemm.candidates[i];
}

/// The tile a trunk (K, N) on Apple M5 Max, 128 GB: the fastest tile at 3584 rows (K4, bf16 rows) at or below the
/// chain's time at 3584 and 3125 rows. A (K, N) two calls share takes the tile both gain from (3072 -> 1536: the latent
/// MoE's down and the shared expert's gate and up; 3072 -> 13824: the dense MLP's gate and up). The dense MLP's down
/// (13824 -> 3072, split K) has no entry: no tile ran at or below the chain's time there.
pub const table = [_]pgemm.Entry{
    .{ .k = 3072, .n = 1024, .tile = c(12) }, // q_a: 64x128x256 2x4
    .{ .k = 1024, .n = 8192, .tile = c(4) }, // q_b.indexed (qrope): 64x128x512 2x2
    .{ .k = 1024, .n = 16384, .tile = c(13) }, // q_b.sliding (qrope): 64x128x512 2x4
    .{ .k = 3072, .n = 640, .tile = c(4) }, // kv_a: 64x128x512 2x2
    .{ .k = 3072, .n = 8192, .tile = c(13) }, // linear_gate.indexed (gate): 64x128x512 2x4
    .{ .k = 3072, .n = 16384, .tile = c(4) }, // linear_gate.sliding (gate): 64x128x512 2x2
    .{ .k = 8192, .n = 3072, .tile = c(8) }, // o_proj.indexed: 32x256x512 1x4 swizzle 1
    .{ .k = 16384, .n = 3072, .tile = c(1) }, // o_proj.sliding (split K): 32x256x512 1x4
    .{ .k = 3072, .n = 1536, .tile = c(13) }, // latent down, shared gate and up (swiglu): 64x128x512 2x4
    .{ .k = 1536, .n = 3072, .tile = c(13) }, // latent up, shared down: 64x128x512 2x4
    .{ .k = 3072, .n = 13824, .tile = c(1) }, // dense gate and up (swiglu): 32x256x512 1x4
};

/// Whether `table` holds a tile for (k, n)
pub fn has(k: u32, n: u32) bool {
    for (table) |e| if (e.k == k and e.n == n) return true;
    return false;
}

/// The host's switch at load, before the first prompt, in this copy of turboexl3: pg2 on at `table`'s shapes, split K
/// in registers at its split shapes with `split_k`
pub fn serve(split_k: bool) void {
    pgemm.host_table.get().* = &table;
    pgemm.split_k.get().* = split_k;
    pgemm.mode.get().* = .on;
}

/// The switch off again (a host's unload, the tests' restore): pg2 off, no table, no split K
pub fn unserve() void {
    pgemm.mode.get().* = .off;
    pgemm.split_k.get().* = false;
    pgemm.host_table.get().* = null;
}

test "turboexl3 hy4f pg2arms table: each entry a trunk shape, once, its tile admitted and its N a multiple of the tile's BN" {
    const std = @import("std");
    for (table, 0..) |e, i| {
        var shapes: usize = 0;
        for (trunk) |l| shapes += @intFromBool(l.k == e.k and l.n == e.n);
        try std.testing.expect(shapes == 1);
        for (table[0..i]) |o| try std.testing.expect(o.k != e.k or o.n != e.n);
        try std.testing.expect(e.tile.admitted() and e.n % e.tile.bn == 0);
    }
    try std.testing.expect(has(16384, 3072) and !has(13824, 3072));
}
