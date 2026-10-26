# Consuming turboexl3 from mlx-stream

Today turboexl3 is the folder `src/turboexl3/` of mlx-stream, compiled as part of the plugin's one Zig module. Its files
import only `std`, `sdk` (mlx-stream's plugin SDK: `sdk.mlx` and `sdk.kernel_launch`), each other and, in the tests,
`mlx_host`; nothing reaches outside the folder. This page is the change that replaces the folder with this package.

## Where the change lands

mlx-stream's `build.zig` does not build the plugin's Zig modules: its `test`, `conformance` and `serve` steps run
`zig build` in an mlx-serve checkout with `-Dmlx-stream-dir=<this checkout>`, and mlx-serve's `build.zig` creates the
modules (`mlxStreamPkg`: the plugin module over `sdk/root.zig` and the host surface `mlx_host`; `addMlxStreamTests`: the test
roots). So the package is fetched by mlx-stream and wired by mlx-serve, the way mlx-serve already builds Sushi from
`lib/sushi` (`-Dsushi-dir`).

A Zig module cannot share a file with another module, so the folder and the package cannot coexist in one build: the switch
moves every turboexl3 file at once.

## mlx-stream

1. Fetch the package as a submodule at `lib/turboexl3` (the path mlx-serve reads; see below) and pin the same commit in
   `build.zig.zon` for Zig-native consumers:

   ```sh
   git submodule add https://github.com/davidtai/turboexl3 lib/turboexl3
   zig fetch --save=turboexl3 git+https://github.com/davidtai/turboexl3#<commit>
   ```

   which adds to `build.zig.zon`:

   ```zig
   .dependencies = .{
       .turboexl3 = .{
           .url = "git+https://github.com/davidtai/turboexl3#<commit>",
           .hash = "<the hash zig fetch prints>",
       },
   },
   ```

   and add `"lib"` to `.paths` so a fetched mlx-stream carries the submodule.

2. Delete `src/turboexl3/`.

3. `src/root.zig`:

   ```zig
   pub const turboexl3 = @import("turboexl3");
   ```

   (was `@import("turboexl3/root.zig")`).

4. `src/tests.zig`: drop `_ = @import("turboexl3/tests.zig");` (a test binary runs only its root module's tests; turboexl3's
   run from their own root, below).

5. `build.zig`: forward one more step to mlx-serve, beside `test` / `conformance` / `serve`:

   ```zig
   .{ .name = "turboexl3-test", .desc = "Run turboexl3's tests over this plugin's SDK (the host's turboexl3-test step)", .args = &.{"turboexl3-test"} },
   ```

6. `NOTICE`: the turboexl3 lines (src/turboexl3/steel_nax.metal under MLX, the Sushi port of src/turboexl3/format.zig) move
   to the package's NOTICE; keep a line that lib/turboexl3 is MIT, Copyright (c) 2026 David Tai.

Nothing else in the plugin changes: `sdk/kernel_launch.zig` stays the plugin SDK's launcher (the archs use it through
`src/sdk_ext/kernel_set.zig` and `metal_launch.zig`), and the host keeps calling `plugin.turboexl3.*`.

## mlx-serve (`build.zig`)

The turboexl3 checkout, beside the existing `mlxStreamDir`:

```zig
/// turboexl3 (`-Dturboexl3-dir`, else the mlx-stream checkout's lib/turboexl3). `b.option` may be declared once.
var turboexl3_dir: ?[]const u8 = null;
fn turboexl3Dir(b: *std.Build) []const u8 {
    if (turboexl3_dir == null) turboexl3_dir = b.option([]const u8, "turboexl3-dir", "turboexl3 checkout (default: <mlx-stream-dir>/lib/turboexl3)") orelse
        b.pathJoin(&.{ mlxStreamDir(b), "lib/turboexl3" });
    return turboexl3_dir.?;
}
```

In `mlxStreamPkg`, after `m` is created, the module over the same `sdk`:

```zig
    const t3 = b.createModule(.{
        .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ turboexl3Dir(b), "src/root.zig" }) },
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "sdk", .module = sdk }},
    });
    m.addImport("turboexl3", t3);
```

Its tests, beside `addMlxStreamTests`: the same host surface and the plugin's SDK, with the launch cache
(`csrc/mlx_launch_cache.inc`, compiled through `csrc/mlx_alloc_shim.cpp`) and MLX linked the way `mlxStreamPkg` links them:

```zig
fn addTurboexl3Tests(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, filter: ?[]const u8, frameworks: ?[]const u8) *std.Build.Step.Compile {
    const host = b.createModule(.{ .root_source_file = b.path("src/plugin_host.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true });
    addMlxLib(b, host);
    addExl3Module(b, host, target, optimize);
    if (frameworks) |fw| host.addFrameworkPath(.{ .cwd_relative = fw });
    for ([_][]const u8{ "IOKit", "CoreFoundation", "Foundation", "Metal", "IOSurface" }) |f| host.linkFramework(f, .{});
    const dir = mlxStreamDir(b);
    const sdk = b.createModule(.{
        .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ dir, "sdk/root.zig" }) },
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_host", .module = host }},
    });
    const t = b.createModule(.{
        .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ turboexl3Dir(b), "src/tests.zig" }) },
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "sdk", .module = sdk }, .{ .name = "mlx_host", .module = host } },
    });
    const csrc = b.pathJoin(&.{ dir, "csrc" });
    t.addCSourceFile(.{ .file = .{ .cwd_relative = b.pathJoin(&.{ csrc, "mlx_alloc_shim.cpp" }) }, .flags = &.{ "-std=c++20", "-O2", "-D_METAL_", "-DACCELERATE_NEW_LAPACK", "-fno-sanitize=all", "-Wall", "-Wno-unused-parameter", "-Wno-deprecated-declarations" } });
    t.addIncludePath(.{ .cwd_relative = csrc });
    t.addIncludePath(b.path("lib/mlx/include"));
    t.addIncludePath(b.path("lib/mlx/include/metal_cpp"));
    t.addIncludePath(b.path("lib/mlxc-src"));
    t.linkSystemLibrary("mlx", .{ .use_pkg_config = .no });
    return b.addTest(.{ .name = "turboexl3-test", .root_module = t, .filters = if (filter) |f| &.{f} else &.{} });
}
```

registered where `mlx-stream-test` is:

```zig
        const t3 = b.addRunArtifact(addTurboexl3Tests(b, target, optimize, test_filter, macos_sdk_frameworks));
        b.step("turboexl3-test", "Run turboexl3's tests over the mlx-stream plugin's SDK").dependOn(&t3.step);
```

The served binary needs nothing more: `addMlxStreamModule` goes through `mlxStreamPkg`.

## Checks after the switch (on the build box)

The build code above is written against mlx-serve's `build.zig` as of this change and has not been compiled (only a box
build can); the checks:

- `zig build` and `zig build mlx-stream-test` in mlx-serve with `-Dmlx-stream-dir`: the plugin builds against the package.
- `zig build turboexl3-test` (CPU tests), then with `TURBOEXL3_GPU=1` (GPU parity).
- The served cell's outputs are byte-identical before and after: the kernels' texts and launches do not change.
