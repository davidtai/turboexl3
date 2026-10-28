//! turboexl3's `kernel_launch` (README.md's Wiring) for an mlx-serve plugin. Kernels come from the server's kernel cache
//! (`Host.kernel`: one kernel object per name and text for the process) and launch through its launcher
//! (`HostGpu.launch`: MLX's built pipeline, without regenerating and hashing the source per launch; the same results).
//! Both go through the host table the plugin binds at load (`bind`). With no table bound (a server that hands its loads
//! none, or a test binary), the kernels are mlx-c's own: `mlx_fast_metal_kernel_new` and `mlx_fast_metal_kernel_apply`,
//! with `mlx_fast_metal_kernel_free` as their free (`decode.freeKernel` routes here).
//!
//! Imports: `sdk`, the consumer's turboexl3 `sdk` (its `mlx`: mlx-c's bindings, shaped as mlx-serve's `mlx/ffi.zig`
//! declares them), and `mlx_serve`, the plugin kit's root (`abi.host`, `host.digest`).

const std = @import("std");
const kit = @import("mlx_serve");
const mlx = @import("sdk").mlx;
const host_abi = kit.abi.host;
const Str = kit.abi.common.Str;

/// The server's host table, bound at load before any kernel is made and never unbound; null: mlx-c's own kernels.
var host: ?*const host_abi.Host = null;

/// Binds the server's host table, the one every model load is handed (`LoadCtx.host`). Call it before the first kernel.
pub fn bind(h: *const host_abi.Host) void {
    host = h;
}

pub const TemplateValue = union(enum) { int: i32, dtype: mlx.mlx_dtype };
pub const TemplateArg = struct { name: [:0]const u8, value: TemplateValue };

/// One launch: grid (threads), threadgroup, the outputs' shapes and dtypes, the template arguments.
pub const Launch = struct {
    grid: [3]u32,
    threadgroup: [3]u32,
    out_shapes: []const []const c_int,
    out_dtypes: []const mlx.mlx_dtype,
    template: []const TemplateArg,
};

/// The most input or output names a kernel takes here.
const max_names = 32;

/// The kernel of one text (`header` before `source`, the inputs' and outputs' names in signature order): the server's,
/// or mlx-c's own when no host table is bound; null when MLX refuses it.
pub fn newKernel(name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool) ?mlx.mlx_fast_metal_kernel {
    if (inputs.len > max_names or outputs.len > max_names) return null;
    const h = host orelse return ownKernel(name, inputs, outputs, source, header, ensure_row_contiguous);
    var ins: [max_names]Str = undefined;
    var outs: [max_names]Str = undefined;
    for (inputs, ins[0..inputs.len]) |n, *o| o.* = .of(std.mem.span(n));
    for (outputs, outs[0..outputs.len]) |n, *o| o.* = .of(std.mem.span(n));
    const src = std.mem.span(source);
    const hdr = std.mem.span(header);
    const desc: host_abi.KernelDesc = .{
        .ensure_row_contiguous = ensure_row_contiguous,
        .digest = kit.host.digest(src, hdr),
        .name = .of(std.mem.span(name)),
        .source = .of(src),
        .header = .of(hdr),
        .inputs = .of(ins[0..inputs.len]),
        .outputs = .of(outs[0..outputs.len]),
    };
    return .{ .ctx = h.kernel(h.ctx, &desc) orelse return null };
}

/// mlx-c's own kernel of one text, freed by `mlx_fast_metal_kernel_free` below.
fn ownKernel(name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool) ?mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, ensure_row_contiguous, false);
    return if (k.ctx == null) null else k;
}

/// The MLX config of one launch, handed to every launch at its geometry; the caller frees it.
pub fn buildConfig(l: Launch) error{MlxError}!mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (l.out_shapes, l.out_dtypes) |sh, dt| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @intCast(l.grid[0]), @intCast(l.grid[1]), @intCast(l.grid[2])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, @intCast(l.threadgroup[0]), @intCast(l.threadgroup[1]), @intCast(l.threadgroup[2])));
    for (l.template) |t| switch (t.value) {
        .int => |v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t.name.ptr, v)),
        .dtype => |v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, t.name.ptr, v)),
    };
    return c;
}

/// One launch of `kernel` with `config` on `stream`, through the server's launcher (mlx-c's apply with no host table
/// bound); `outs` receives the launch's outputs (new arrays the caller frees).
pub fn applyConfig(kernel: mlx.mlx_fast_metal_kernel, config: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array, outs: []mlx.mlx_array, stream: mlx.mlx_stream) error{MlxError}!void {
    const vin = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(vin);
    var vout = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vout);
    if (host) |h| {
        const v = h.gpu.launch(h.gpu.ctx, @ptrCast(&vout), kernel.ctx, .{ .ctx = vin.ctx }, .{ .ctx = config.ctx }, .{ .ctx = stream.ctx });
        if (!v.isOk()) return error.MlxError;
    } else try mlx.check(mlx.mlx_fast_metal_kernel_apply(&vout, kernel, vin, config, stream));
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, vout, i));
    }
}

/// Drops a kernel `newKernel` made (turboexl3's `decode.freeKernel`): the server's kernels stay in its cache for the
/// process, and mlx-c's own are freed.
pub fn mlx_fast_metal_kernel_free(k: mlx.mlx_fast_metal_kernel) c_int {
    if (host != null) return 0;
    return mlx.mlx_fast_metal_kernel_free(k);
}
