//! turboexl3's test root: every file's tests. CPU tests always run; the GPU parity tests need TURBOEXL3_GPU=1 and a GPU
//! stream, the Metal-source dumps TURBOEXL3_METAL_DUMP=1. mlx-stream reaches it from its own test root (src/tests.zig); a
//! standalone build compiles it as a test root over `sdk` and `mlx_host` (README.md).
comptime {
    _ = @import("core/config.zig");
    _ = @import("core/global.zig");
    _ = @import("core/sources.zig");
    _ = @import("core/steel.zig");
    _ = @import("core/format.zig");
    _ = @import("core/spec.zig");
    _ = @import("core/lane.zig");
    _ = @import("core/ref.zig");
    _ = @import("core/unfused.zig");
    _ = @import("core/decode.zig");
    _ = @import("core/dense.zig");
    _ = @import("core/pgemm.zig");
    _ = @import("core/prefill.zig");
    _ = @import("models/qwen38/decb.zig");
    _ = @import("models/qwen38/attn_io.zig");
    _ = @import("models/qwen38/gdn_out.zig");
    _ = @import("models/qwen38/shared.zig");
    _ = @import("models/ricursive/ricursive.zig");
    _ = @import("models/ling/pair.zig");
    _ = @import("models/ling/kdaproj.zig");
    _ = @import("models/deepseek_v41/grouped.zig");
    _ = @import("root.zig");
    _ = @import("core/test_support.zig");
    _ = @import("core/gpu_test.zig");
    _ = @import("core/prefill_test.zig");
    _ = @import("models/qwen38/decb_test.zig");
    _ = @import("models/qwen38/attn_io_test.zig");
    _ = @import("models/qwen38/gdn_out_test.zig");
    _ = @import("models/deepseek_v41/clamped_test.zig");
    _ = @import("models/deepseek_v41/banked_test.zig");
    _ = @import("models/deepseek_v41/banked_grp_test.zig");
    _ = @import("models/deepseek_v41/banked_ilp_test.zig");
    _ = @import("models/deepseek_v41/deepseek_test.zig");
    _ = @import("models/deepseek_v41/deepseek_gemv_test.zig");
    _ = @import("core/topk_test.zig");
    _ = @import("core/bench.zig");
    _ = @import("models/deepseek_v41/f3_test.zig");
    _ = @import("core/chains_test.zig");
    _ = @import("models/ricursive/ricursive_test.zig");
    _ = @import("models/ling/ling_test.zig");
    _ = @import("models/ling/shared_test.zig");
    _ = @import("models/ling/combine.zig");
    _ = @import("models/ling/combine_test.zig");
    _ = @import("models/ling/addnorm.zig");
    _ = @import("models/ling/addnorm_test.zig");
    _ = @import("models/ling/router.zig");
    _ = @import("models/ling/router_test.zig");
}


// every pub declaration of the module root and of the namespaces and types it exports from this module's files, analyzed
// the way a consumer's build reaches them (Zig analyzes only what is referenced, and the tests reference what they call)
test "turboexl3 module: every pub declaration compiles" {
    @setEvalBranchQuota(1_000_000);
    refAll(@import("root.zig"), 4);
}

fn refAll(comptime T: type, comptime depth: u32) void {
    const std = @import("std");
    inline for (comptime std.meta.declarations(T)) |name| {
        const v = &@field(T, name);
        _ = v;
        if (depth > 1 and @TypeOf(@field(T, name)) == type) {
            const U = @field(T, name);
            const ours = comptime blk: {
                const n = @typeName(U);
                break :blk std.mem.eql(u8, n, "root") or std.mem.startsWith(u8, n, "root.") or std.mem.startsWith(u8, n, "core.") or std.mem.startsWith(u8, n, "models.");
            };
            switch (@typeInfo(U)) {
                .@"struct", .@"union", .@"enum", .@"opaque" => if (ours) refAll(U, depth - 1),
                else => {},
            }
        }
    }
}
