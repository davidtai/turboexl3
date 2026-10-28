//! A module global (a cache, a lock, a counter, the configuration). Where the consumer's `sdk` brings the host's
//! `shared` (mlx-serve's `sdk/util/shared.zig`), it is the host's `Shared`: a server that compiles turboexl3 into
//! several objects (`-Dunits`) keeps one copy, the host's, and every object reads and writes it. Elsewhere it is a
//! plain global.

const sdk = @import("sdk");

/// `get()` is the storage, initialised to `init`; `name` is unique within turboexl3.
pub fn Global(comptime T: type, comptime name: []const u8, comptime init: T) type {
    if (@hasDecl(sdk, "shared")) return struct {
        // `Shared` hands its storage out as `*anyopaque`, which a pointer to a pointer does not coerce to: T goes in a
        // struct, so every T does
        const Box = struct { v: T };
        const S = sdk.shared.Shared(Box, "turboexl3_" ++ name, .{ .v = init });

        // the host's copy is exported wherever this type is referenced (`shared_state`)
        comptime {
            _ = S;
        }

        pub inline fn get() *T {
            return &S.get().v;
        }
    };
    return struct {
        var storage: T = init;

        pub inline fn get() *T {
            return &storage;
        }
    };
}
