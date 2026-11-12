//! The routed prefill's 128-row tile launches at 128-deep K stages (`prefill.Options.t128_bk128`, `Config.pf_t128_bk128`:
//! the plugin's TURBOEXL3_GLM53F_PF_OPTS values `tk` and `bk128.tk`), CPU only: where the option is taken and the stage
//! depth each launch then runs. Its bytes at GLM-5.3-Flash's routed geometry and every SC rate, against the served
//! options, are fused_test.zig's prefill options test (the same MMAs in the same order: the served words).
const std = @import("std");
const prefill = @import("../../core/prefill.zig");
const config = @import("../../core/config.zig");

const testing = std.testing;

/// root.pfOptions' served set with the 128-row tiles (bm 64, the mixed launch, fragment-order double-buffered stage)
const served: prefill.Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .defer_affine = true, .ncb = 4, .sintr = true, .froute = true, .alay = 2, .t128 = true };

test "turboexl3 glm53f prefill t128_bk128: the 128-row tile launches take 128-deep stages beside t128 within 32 KB, each mixed launch keeps its own depth, and the config carries it off by default" {
    try testing.expect(served.valid());
    var o = served;
    o.t128_bk128 = true;
    try testing.expect(o.valid());
    // gate|up's mixed launch at BK and down's at its stage depth; both 128-row tile launches at 128 (32 KB double-buffered)
    try testing.expectEqual(@as(u32, 128), o.t128Stage(prefill.BK));
    try testing.expectEqual(@as(u32, 128), o.t128Stage(o.downStage()));
    try testing.expectEqual(@as(u32, 32768), o.stageBytes(o.t128Stage(prefill.BK)));
    // with down's bk128 its mixed launch runs 128 deep too; off, every launch keeps the depth it is handed
    o.bk128 = true;
    try testing.expect(o.valid() and o.downStage() == 128 and o.t128Stage(o.downStage()) == 128);
    try testing.expectEqual(prefill.BK, served.t128Stage(prefill.BK));
    try testing.expectEqual(@as(u32, 128), served.t128Stage(128));
    // nothing to do without the 128-row tiles; the padded stage would be 36 KB
    var no_t = served;
    no_t.t128 = false;
    no_t.t128_bk128 = true;
    try testing.expect(!no_t.valid());
    var padded = served;
    padded.fragb = false;
    padded.alay = 0;
    try testing.expect(padded.valid());
    padded.t128_bk128 = true;
    try testing.expect(!padded.valid());
    const d: config.Config = .{};
    try testing.expect(!d.pf_t128_bk128);
    try testing.expect((config.Config{ .pf_t128_bk128 = true }).valid());
}
