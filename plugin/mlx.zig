//! The mlx-c turboexl3 calls, declared for Zig as the server's own bindings declare them (an exhaustive `mlx_dtype`, the
//! handles as one-pointer structs) and resolved at load against the server's libmlxc: one MLX in the process, its
//! error handler the server's (a failed call returns nonzero here, its message latched there). Plus `check`, `getShape`
//! and `streamIsGpu`, the helpers turboexl3's `sdk.mlx` carries.

pub const mlx_array = extern struct { ctx: ?*anyopaque = null };
pub const mlx_stream = extern struct { ctx: ?*anyopaque = null };
pub const mlx_device = extern struct { ctx: ?*anyopaque = null };
pub const mlx_vector_array = extern struct { ctx: ?*anyopaque = null };
pub const mlx_fast_metal_kernel = extern struct { ctx: ?*anyopaque = null };
pub const mlx_fast_metal_kernel_config = extern struct { ctx: ?*anyopaque = null };

pub const mlx_dtype = enum(c_int) {
    bool_ = 0,
    uint8 = 1,
    uint16 = 2,
    uint32 = 3,
    uint64 = 4,
    int8 = 5,
    int16 = 6,
    int32 = 7,
    int64 = 8,
    float16 = 9,
    float32 = 10,
    float64 = 11,
    bfloat16 = 12,
    complex64 = 13,
};

pub const mlx_device_type = enum(c_int) { cpu = 0, gpu = 1 };

// ── arrays ──
pub extern "c" fn mlx_array_new() mlx_array;
pub extern "c" fn mlx_array_free(arr: mlx_array) c_int;
pub extern "c" fn mlx_array_set(arr: *mlx_array, src: mlx_array) c_int;
pub extern "c" fn mlx_array_new_int(val: c_int) mlx_array;
pub extern "c" fn mlx_array_new_float(val: f32) mlx_array;
pub extern "c" fn mlx_array_new_data(data: ?*const anyopaque, shape: [*]const c_int, dim: c_int, dtype: mlx_dtype) mlx_array;
pub extern "c" fn mlx_array_ndim(arr: mlx_array) usize;
pub extern "c" fn mlx_array_shape(arr: mlx_array) [*]const c_int;
pub extern "c" fn mlx_array_size(arr: mlx_array) usize;
pub extern "c" fn mlx_array_itemsize(arr: mlx_array) usize;
pub extern "c" fn mlx_array_dtype(arr: mlx_array) mlx_dtype;
pub extern "c" fn mlx_array_eval(arr: mlx_array) c_int;
pub extern "c" fn mlx_array_item_bool(res: *bool, arr: mlx_array) c_int;
pub extern "c" fn mlx_array_data_int32(arr: mlx_array) ?[*]const i32;
pub extern "c" fn mlx_array_data_uint32(arr: mlx_array) ?[*]const u32;

pub extern "c" fn mlx_vector_array_new() mlx_vector_array;
pub extern "c" fn mlx_vector_array_new_data(data: [*]const mlx_array, size: usize) mlx_vector_array;
pub extern "c" fn mlx_vector_array_free(vec: mlx_vector_array) c_int;
pub extern "c" fn mlx_vector_array_get(res: *mlx_array, vec: mlx_vector_array, idx: usize) c_int;

// ── ops ──
pub extern "c" fn mlx_reshape(res: *mlx_array, a: mlx_array, shape: [*]const c_int, shape_num: usize, s: mlx_stream) c_int;
pub extern "c" fn mlx_matmul(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_add(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_zeros(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, s: mlx_stream) c_int;
pub extern "c" fn mlx_slice(res: *mlx_array, a: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) c_int;
pub extern "c" fn mlx_concatenate_axis(res: *mlx_array, arrays: mlx_vector_array, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_argsort_axis(res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_isfinite(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_all(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_clear_cache() c_int;

// ── devices ──
pub extern "c" fn mlx_stream_get_device(dev: *mlx_device, stream: mlx_stream) c_int;
pub extern "c" fn mlx_device_get_type(dtype: *mlx_device_type, dev: mlx_device) c_int;
pub extern "c" fn mlx_device_free(dev: mlx_device) c_int;

// ── custom Metal kernels: the launch configs, and mlx-c's own kernels, which `kernel_launch` makes only when no host
// table is bound (the kernels are otherwise the server's) ──
pub const mlx_vector_string = extern struct { ctx: ?*anyopaque = null };
pub extern "c" fn mlx_vector_string_new_data(data: [*]const [*:0]const u8, size: usize) mlx_vector_string;
pub extern "c" fn mlx_vector_string_free(vec: mlx_vector_string) c_int;
pub extern "c" fn mlx_fast_metal_kernel_new(name: [*:0]const u8, input_names: mlx_vector_string, output_names: mlx_vector_string, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx_fast_metal_kernel;
pub extern "c" fn mlx_fast_metal_kernel_free(cls: mlx_fast_metal_kernel) c_int;
pub extern "c" fn mlx_fast_metal_kernel_apply(outputs: *mlx_vector_array, cls: mlx_fast_metal_kernel, inputs: mlx_vector_array, config: mlx_fast_metal_kernel_config, s: mlx_stream) c_int;
pub extern "c" fn mlx_fast_metal_kernel_config_new() mlx_fast_metal_kernel_config;
pub extern "c" fn mlx_fast_metal_kernel_config_free(cls: mlx_fast_metal_kernel_config) c_int;
pub extern "c" fn mlx_fast_metal_kernel_config_add_output_arg(cls: mlx_fast_metal_kernel_config, shape: [*]const c_int, size: usize, dtype: mlx_dtype) c_int;
pub extern "c" fn mlx_fast_metal_kernel_config_set_grid(cls: mlx_fast_metal_kernel_config, g1: c_int, g2: c_int, g3: c_int) c_int;
pub extern "c" fn mlx_fast_metal_kernel_config_set_thread_group(cls: mlx_fast_metal_kernel_config, t1: c_int, t2: c_int, t3: c_int) c_int;
pub extern "c" fn mlx_fast_metal_kernel_config_add_template_arg_int(cls: mlx_fast_metal_kernel_config, name: [*:0]const u8, value: c_int) c_int;
pub extern "c" fn mlx_fast_metal_kernel_config_add_template_arg_dtype(cls: mlx_fast_metal_kernel_config, name: [*:0]const u8, dtype: mlx_dtype) c_int;

pub fn check(rc: c_int) error{MlxError}!void {
    if (rc != 0) return error.MlxError;
}

pub fn getShape(arr: mlx_array) []const c_int {
    const ndim = mlx_array_ndim(arr);
    if (ndim == 0) return &.{};
    return mlx_array_shape(arr)[0..ndim];
}

/// The stream runs on a GPU (the server is Metal here: an outside plugin loads on macOS only).
pub fn streamIsGpu(s: mlx_stream) bool {
    var dev: mlx_device = .{};
    if (mlx_stream_get_device(&dev, s) != 0) return false;
    defer _ = mlx_device_free(dev);
    var dt: mlx_device_type = .cpu;
    if (mlx_device_get_type(&dt, dev) != 0) return false;
    return dt == .gpu;
}
