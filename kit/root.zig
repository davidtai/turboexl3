//! `turboexl3_kit`: what an mlx-serve plugin built with the plugin kit hands turboexl3 besides its mlx-c bindings.
//! - `kernel_launch`: turboexl3's kernel API (README.md's Wiring) over the server's host table, bound by the plugin at
//!   load (`bind`): kernels from the server's kernel cache, launches through its launcher. With no table bound, mlx-c's
//!   own kernels.
//! - `nax`: MLX's NAX gate, probed without MLX, for the plugin's claim and `Config.nax`.
//!
//! The consumer gives this module two imports:
//! - `sdk`: its own turboexl3 `sdk`, for `sdk.mlx`, its mlx-c bindings. They must carry the types and functions
//!   turboexl3 uses, plus the kernel calls `kernel_launch` makes; mlx-serve's `mlx/ffi.zig` has all of them.
//! - `mlx_serve`: the plugin kit's root.
//!
//! That `sdk` takes `kernel_launch` from here: `pub const kernel_launch = @import("turboexl3_kit").kernel_launch;`.
//! The two modules import each other, which Zig allows. A consumer whose bindings are files of its own module (mlx-serve's
//! `mlx/ffi.zig`, imported by path) cannot also make them a module, so `mlx` reaches this module through `sdk`.

pub const kernel_launch = @import("kernel_launch.zig");
pub const nax = @import("nax.zig");
pub const bind = kernel_launch.bind;
/// The kit's module state (`global.zig`), for the host to register.
pub const shared_state = .{ kernel_launch.shared_state, nax.shared_state };

test {
    _ = nax;
    _ = @import("global.zig");
}
