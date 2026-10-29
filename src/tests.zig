//! turboexl3's test root: every file's tests. CPU tests always run; the GPU parity tests need TURBOEXL3_GPU=1 and a GPU
//! stream, the Metal-source dumps TURBOEXL3_METAL_DUMP=1. mlx-stream reaches it from its own test root (src/tests.zig); a
//! standalone build compiles it as a test root over `sdk` and `mlx_host` (README.md).
comptime {
    _ = @import("config.zig");
    _ = @import("global.zig");
    _ = @import("sources.zig");
    _ = @import("steel.zig");
    _ = @import("format.zig");
    _ = @import("spec.zig");
    _ = @import("lane.zig");
    _ = @import("ref.zig");
    _ = @import("unfused.zig");
    _ = @import("decode.zig");
    _ = @import("dense.zig");
    _ = @import("prefill.zig");
    _ = @import("decb.zig");
    _ = @import("attn_io.zig");
    _ = @import("gdn_out.zig");
    _ = @import("root.zig");
    _ = @import("test_support.zig");
    _ = @import("gpu_test.zig");
    _ = @import("prefill_test.zig");
    _ = @import("decb_test.zig");
    _ = @import("attn_io_test.zig");
    _ = @import("gdn_out_test.zig");
    _ = @import("bench.zig");
}
