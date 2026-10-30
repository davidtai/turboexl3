//! A small HIP runtime for turboexl3's AMD kernels and their tests: libamdhip64 opened at run time, code objects
//! loaded from memory, kernels launched with their arguments, device buffers. No MLX on AMD. A host with its own HIP
//! runtime (TensorFold) runs plan.zig's launches through that runtime instead and does not import this file.
const std = @import("std");

pub const Error = error{ HipUnavailable, MissingSymbol, HipFailed, BadArgs };

const Result = c_int;
const Ptr = ?*anyopaque;

const Api = struct {
    hipInit: *const fn (c_uint) callconv(.c) Result,
    hipSetDevice: *const fn (c_int) callconv(.c) Result,
    hipGetErrorString: *const fn (Result) callconv(.c) [*:0]const u8,
    hipModuleLoadData: *const fn (*Ptr, [*]const u8) callconv(.c) Result,
    hipModuleUnload: *const fn (Ptr) callconv(.c) Result,
    hipModuleGetFunction: *const fn (*Ptr, Ptr, [*:0]const u8) callconv(.c) Result,
    hipModuleLaunchKernel: *const fn (Ptr, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Ptr, ?[*]Ptr, ?[*]Ptr) callconv(.c) Result,
    hipMalloc: *const fn (*Ptr, usize) callconv(.c) Result,
    hipFree: *const fn (Ptr) callconv(.c) Result,
    hipMemcpyHtoD: *const fn (Ptr, [*]const u8, usize) callconv(.c) Result,
    hipMemcpyDtoH: *const fn ([*]u8, Ptr, usize) callconv(.c) Result,
    hipMemsetD8: *const fn (Ptr, u8, usize) callconv(.c) Result,
    hipDeviceSynchronize: *const fn () callconv(.c) Result,
    hipStreamCreate: *const fn (*Ptr) callconv(.c) Result,
    hipStreamDestroy: *const fn (Ptr) callconv(.c) Result,
    hipStreamSynchronize: *const fn (Ptr) callconv(.c) Result,
    hipStreamBeginCapture: *const fn (Ptr, c_int) callconv(.c) Result,
    hipStreamEndCapture: *const fn (Ptr, *Ptr) callconv(.c) Result,
    hipGraphInstantiate: *const fn (*Ptr, Ptr, ?*anyopaque, ?[*]u8, usize) callconv(.c) Result,
    hipGraphLaunch: *const fn (Ptr, Ptr) callconv(.c) Result,
    hipGraphExecDestroy: *const fn (Ptr) callconv(.c) Result,
    hipGraphDestroy: *const fn (Ptr) callconv(.c) Result,
};

pub const Hip = struct {
    lib: std.DynLib,
    api: Api,

    /// libamdhip64 from the loader's path, device 0.
    pub fn open() Error!Hip {
        var lib = std.DynLib.open("libamdhip64.so") catch return error.HipUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse return error.MissingSymbol;
        }
        var h: Hip = .{ .lib = lib, .api = api };
        try h.check(api.hipInit(0));
        try h.check(api.hipSetDevice(0));
        return h;
    }

    pub fn close(h: *Hip) void {
        h.lib.close();
    }

    pub fn check(h: *const Hip, r: Result) Error!void {
        if (r == 0) return;
        std.log.err("HIP error {d}: {s}", .{ r, h.api.hipGetErrorString(r) });
        return error.HipFailed;
    }

    pub fn sync(h: *const Hip) Error!void {
        try h.check(h.api.hipDeviceSynchronize());
    }

    pub fn alloc(h: *const Hip, bytes: usize) Error!Buffer {
        var p: Ptr = null;
        try h.check(h.api.hipMalloc(&p, @max(bytes, 1)));
        return .{ .h = h, .ptr = p, .len = bytes };
    }

    pub fn upload(h: *const Hip, bytes: []const u8) Error!Buffer {
        const b = try h.alloc(bytes.len);
        if (bytes.len > 0) try h.check(h.api.hipMemcpyHtoD(b.ptr, bytes.ptr, bytes.len));
        return b;
    }

    pub fn stream(h: *const Hip) Error!Stream {
        var s: Ptr = null;
        try h.check(h.api.hipStreamCreate(&s));
        return .{ .h = h, .handle = s };
    }

    pub fn load(h: *const Hip, image: []const u8) Error!Module {
        var m: Ptr = null;
        try h.check(h.api.hipModuleLoadData(&m, image.ptr));
        return .{ .h = h, .handle = m };
    }
};

pub const Buffer = struct {
    h: *const Hip,
    ptr: Ptr,
    len: usize,

    pub fn free(b: *Buffer) void {
        _ = b.h.api.hipFree(b.ptr);
        b.ptr = null;
    }

    pub fn addr(b: Buffer) u64 {
        return @intFromPtr(b.ptr);
    }

    pub fn fill(b: Buffer, value: u8) Error!void {
        try b.h.check(b.h.api.hipMemsetD8(b.ptr, value, b.len));
    }

    /// The buffer's bytes after every launch so far.
    pub fn read(b: Buffer, out: []u8) Error!void {
        if (out.len > b.len) return error.BadArgs;
        try b.h.sync();
        try b.h.check(b.h.api.hipMemcpyDtoH(out.ptr, b.ptr, out.len));
    }
};

pub const Dim = struct { x: u32, y: u32 = 1, z: u32 = 1 };

/// A stream, and capture of its launches into a graph that replays them.
pub const Stream = struct {
    h: *const Hip,
    handle: Ptr,

    pub fn destroy(s: *Stream) void {
        _ = s.h.api.hipStreamDestroy(s.handle);
    }

    pub fn sync(s: Stream) Error!void {
        try s.h.check(s.h.api.hipStreamSynchronize(s.handle));
    }

    /// Launches on the stream are recorded, not run, until `endCapture` (thread-local capture mode).
    pub fn beginCapture(s: Stream) Error!void {
        try s.h.check(s.h.api.hipStreamBeginCapture(s.handle, 1));
    }

    pub fn endCapture(s: Stream) Error!Graph {
        var g: Ptr = null;
        try s.h.check(s.h.api.hipStreamEndCapture(s.handle, &g));
        var exec: Ptr = null;
        try s.h.check(s.h.api.hipGraphInstantiate(&exec, g, null, null, 0));
        return .{ .h = s.h, .graph = g, .exec = exec };
    }
};

pub const Graph = struct {
    h: *const Hip,
    graph: Ptr,
    exec: Ptr,

    pub fn launch(g: Graph, s: Stream) Error!void {
        try g.h.check(g.h.api.hipGraphLaunch(g.exec, s.handle));
    }

    pub fn destroy(g: *Graph) void {
        _ = g.h.api.hipGraphExecDestroy(g.exec);
        _ = g.h.api.hipGraphDestroy(g.graph);
    }
};

pub const Module = struct {
    h: *const Hip,
    handle: Ptr,

    pub fn unload(m: *Module) void {
        _ = m.h.api.hipModuleUnload(m.handle);
    }

    pub fn function(m: Module, name: [:0]const u8) Error!Function {
        var f: Ptr = null;
        try m.h.check(m.h.api.hipModuleGetFunction(&f, m.handle, name.ptr));
        return .{ .h = m.h, .handle = f };
    }
};

/// A kernel's arguments as C values, packed with their alignment (hipModuleLaunchKernel's kernelParams).
pub const Args = struct {
    storage: [1024]u8 align(16) = undefined,
    offsets: [32]u16 = undefined,
    ptrs: [32]Ptr = undefined,
    used: usize = 0,
    count: usize = 0,

    pub fn add(a: *Args, value: anytype) Error!void {
        const T = @TypeOf(value);
        const at = std.mem.alignForward(usize, a.used, @alignOf(T));
        if (a.count == a.offsets.len or at + @sizeOf(T) > a.storage.len) return error.BadArgs;
        @memcpy(a.storage[at..][0..@sizeOf(T)], std.mem.asBytes(&value));
        a.offsets[a.count] = @intCast(at);
        a.count += 1;
        a.used = at + @sizeOf(T);
    }

    fn pointers(a: *Args) ?[*]Ptr {
        if (a.count == 0) return null;
        for (a.offsets[0..a.count], 0..) |off, i| a.ptrs[i] = &a.storage[off];
        return &a.ptrs;
    }
};

pub const Function = struct {
    h: *const Hip,
    handle: Ptr,

    /// One launch on the null stream.
    pub fn launch(f: Function, grid: Dim, block: Dim, shared: u32, args: *Args) Error!void {
        try f.launchOn(null, grid, block, shared, args);
    }

    /// One launch on `stream` (null: the null stream).
    pub fn launchOn(f: Function, stream: ?Stream, grid: Dim, block: Dim, shared: u32, args: *Args) Error!void {
        const s: Ptr = if (stream) |x| x.handle else null;
        try f.h.check(f.h.api.hipModuleLaunchKernel(f.handle, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared, s, args.pointers(), null));
    }
};
