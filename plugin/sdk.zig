//! turboexl3's consumer module (`sdk`, README.md's Wiring) inside the quant plugin:
//! - `mlx`: mlx-c as the server's libmlxc serves it (`mlx.zig`);
//! - `kernel_launch`: turboexl3_kit's, MLX custom Metal kernels from the server's kernel cache and launched by its
//!   launcher, through the host table the table's `open` binds;
//! - `quant_abi`: the kit's quant ABI, the server's table type.
//! The kit carries no MLX steel headers, so turboexl3 takes its vendored copy.

pub const mlx = @import("mlx.zig");
pub const kernel_launch = @import("turboexl3_kit").kernel_launch;
pub const quant_abi = @import("mlx_serve").abi.quant;
