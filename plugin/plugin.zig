//! turboexl3 as an mlx-serve quant plugin (a dylib from the plugin kit): the table is turboexl3_table's
//! (`table/table.zig`), here behind the server's quant table. It serves the blocks it claims (`table/config.zig`): a
//! fully EXL3 pack's `trunk_quant`, the trunk's dense linears, and in the reordered numerics where MLX's NAX gate holds,
//! `expert_quant`, the routed banks. The server's built-in Sushi table keeps the banks otherwise.

const std = @import("std");
const kit = @import("mlx_serve");
const tt = @import("turboexl3_table");

/// Shown at install and in Settings.
pub const version = "0.1.0";

pub const plugin = kit.sdk.Plugin{ .name = tt.table_name, .kind = .quant };

pub const table = tt.table;
pub const config = tt.config;

/// The conformance suite's block: a trunk block, claimed on every machine in either numerics.
pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    _ = a;
    return .{ .files = &.{}, .block = "{\"format\":\"exl3\",\"codebook\":\"mul1\",\"window\":16}" };
}
