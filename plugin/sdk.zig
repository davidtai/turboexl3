//! turboexl3's consumer module (`sdk`, README.md's Wiring) inside the plugin: `mlx`, mlx-c as the server's libmlxc
//! serves it (`mlx.zig`), and `kernel_launch`, MLX custom Metal kernels made by the server's kernel cache (`Host.kernel`:
//! one kernel object per name and text for the process) and launched by its launcher (`HostGpu.launch`: MLX's built
//! pipeline, without regenerating and hashing the source per launch; the same results). Both go through the host table
//! the plugin's `open` binds (`bind`). The kit carries no MLX steel headers, so turboexl3 takes its vendored copy.

const std = @import("std");
const kit = @import("mlx_serve");
const host_abi = kit.abi.host;
const Str = kit.abi.common.Str;
pub const mlx = @import("mlx.zig");

/// The server's host table: one per process, bound by every `open` (the same table) before any launch.
var host: ?*const host_abi.Host = null;

pub fn bind(h: *const host_abi.Host) void {
    host = h;
}

pub const kernel_launch = struct {
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

    /// The server's kernel of one text (`header` before `source`, the inputs' and outputs' names in signature order);
    /// null before a host is bound or when MLX refuses it.
    pub fn newKernel(name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool) ?mlx.mlx_fast_metal_kernel {
        const h = host orelse return null;
        if (inputs.len > max_names or outputs.len > max_names) return null;
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

    /// One launch of `kernel` with `config` on `stream` through the server's launcher; `outs` receives the launch's
    /// outputs (new arrays the caller frees).
    pub fn applyConfig(kernel: mlx.mlx_fast_metal_kernel, config: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array, outs: []mlx.mlx_array, stream: mlx.mlx_stream) error{MlxError}!void {
        const h = host orelse return error.MlxError;
        const vin = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
        defer _ = mlx.mlx_vector_array_free(vin);
        var vout = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vout);
        const v = h.gpu.launch(h.gpu.ctx, @ptrCast(&vout), kernel.ctx, .{ .ctx = vin.ctx }, .{ .ctx = config.ctx }, .{ .ctx = stream.ctx });
        if (!v.isOk()) return error.MlxError;
        for (outs, 0..) |*o, i| {
            o.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(o, vout, i));
        }
    }
};
