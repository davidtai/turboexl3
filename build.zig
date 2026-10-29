//! turboexl3 as a package. Its module `turboexl3` (src/root.zig) imports `std` and one module the consumer supplies,
//! `sdk`: a namespace with `mlx` (mlx-c's bindings plus `check`, `getShape` and `streamIsGpu`) and `kernel_launch`
//! (MLX custom Metal kernels and their launches: `newKernel`, `Launch`, `TemplateArg`, `buildConfig`, `applyConfig`).
//! README.md describes both; CONSUMING.md gives what every consumer supplies (the `sdk` module) and how the quant plugin
//! in plugin/ builds turboexl3 in.
//!
//!     const t3 = b.dependency("turboexl3", .{ .target = target, .optimize = optimize }).module("turboexl3");
//!     t3.addImport("sdk", sdk);
//!     consumer.addImport("turboexl3", t3);
//!
//! A second module, `turboexl3_kit` (kit/root.zig), is the `kernel_launch` and the NAX gate for an mlx-serve plugin built
//! with the plugin kit. It launches through the server's host table and falls back to mlx-c's own kernels when no table
//! is bound. Its consumer adds `sdk` (its own turboexl3 `sdk`, for `sdk.mlx`) and `mlx_serve` (the kit's root), and takes
//! `kernel_launch` from it into that `sdk`:
//!
//!     const dep = b.dependency("turboexl3", .{ .target = target, .optimize = optimize });
//!     const tkit = dep.module("turboexl3_kit");
//!     tkit.addImport("sdk", sdk);
//!     tkit.addImport("mlx_serve", kit_root);
//!     sdk.addImport("turboexl3_kit", tkit); // sdk.zig: `pub const kernel_launch = @import("turboexl3_kit").kernel_launch;`
//!
//! A third module, `turboexl3_table` (table/table.zig), is turboexl3 as an mlx-serve quant table: the quant plugin in
//! plugin/ is its dylib, and a model plugin can run it in process behind its own format seam. It imports `sdk` (its
//! `mlx`, `kernel_launch` and `quant_abi`, the quant ABI the table type comes from), `mlx_serve` (the kit's root),
//! `turboexl3` and `turboexl3_kit`.
//!
//! The tests (src/tests.zig) need a Metal device, MLX and the consumer's launcher linked, so the consumer builds them over
//! its own surface with `testModule` (they also import `mlx_host`, whose `sushi_exl3`, where present, is a second
//! reference).
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    _ = b.addModule("turboexl3", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    _ = b.addModule("turboexl3_kit", .{
        .root_source_file = b.path("kit/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    _ = b.addModule("turboexl3_table", .{
        .root_source_file = b.path("table/table.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
}

/// The test root module over the consumer's `sdk` and `mlx_host`, for a consumer's build.zig:
///
///     const t3_build = @import("turboexl3");
///     const dep = b.dependency("turboexl3", .{ .target = target, .optimize = optimize });
///     const t = b.addTest(.{ .name = "turboexl3-test", .root_module = t3_build.testModule(b, dep, target, optimize, sdk, mlx_host) });
///
/// then link what the launcher needs (MLX, the launch cache's C sources, Metal) to `t.root_module`.
pub fn testModule(b: *std.Build, dep: *std.Build.Dependency, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, sdk: *std.Build.Module, mlx_host: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = dep.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "sdk", .module = sdk }, .{ .name = "mlx_host", .module = mlx_host } },
    });
}
