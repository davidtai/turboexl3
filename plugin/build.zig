//! turboexl3 as an mlx-serve quant plugin, built with mlx-serve's plugin kit: `zig build` writes
//! zig-out/lib/libturboexl3.dylib, `zig build test` runs its MLX-free claim tests, and `mlx-serve plugin check
//! zig-out/lib/libturboexl3.dylib [--model <pack>]` runs mlx-serve's conformance suite on it (with a pack: the blocks it
//! claims there); `mlx-serve plugin install` installs it. `kit` (build.zig.zon) is the staged kit; the dylib links no MLX
//! (mlx-c resolves against the server's own at load), and its MLX pin is the kit's, which must equal the server's.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const lib = @import("mlx_serve_kit").plugin(b, .{ .name = "turboexl3" });
    // The kit's builder made the plugin's root module (`plugin.zig`, over `mlx_serve` and `mlx_c`); turboexl3's modules
    // join it over the consumer module `sdk` (sdk.zig): the library, its kit (the launcher, the NAX gate) and the table.
    const root = lib.root_module.import_table.get("plugin").?;
    const target = root.resolved_target.?;
    const optimize = root.optimize.?;
    const kit_root = root.import_table.get("mlx_serve").?;
    const dep = b.dependency("turboexl3", .{ .target = target, .optimize = optimize });
    const tkit = dep.module("turboexl3_kit");
    const sdk = b.createModule(.{
        .root_source_file = b.path("sdk.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "mlx_serve", .module = kit_root }, .{ .name = "turboexl3_kit", .module = tkit } },
    });
    tkit.addImport("sdk", sdk);
    tkit.addImport("mlx_serve", kit_root);
    const t3 = dep.module("turboexl3");
    t3.addImport("sdk", sdk);
    const table = dep.module("turboexl3_table");
    for ([_][]const u8{ "sdk", "mlx_serve", "turboexl3", "turboexl3_kit" }, [_]*std.Build.Module{ sdk, kit_root, t3, tkit }) |n, m| table.addImport(n, m);
    root.addImport("sdk", sdk);
    root.addImport("turboexl3", t3);
    root.addImport("turboexl3_table", table);
}
