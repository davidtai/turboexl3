//! A kit global (the bound host table, the NAX probe). Where the consumer's `sdk` brings the host's
//! `shared` (mlx-serve's `sdk/util/shared.zig`), it is the host's `Shared`: a server that compiles turboexl3 into
//! several objects (`-Dunits`) keeps one copy, the host's, and every object reads and writes it. Elsewhere it is a
//! plain global.

const sdk = @import("sdk");

/// `get()` is the storage, initialised to `init`; `name` is unique within turboexl3.
pub fn Global(comptime T: type, comptime name: []const u8, comptime init: T) type {
    if (@hasDecl(sdk, "shared")) return sdk.shared.Shared(T, "turboexl3_" ++ name, init);
    return struct {
        var storage: T = init;

        pub inline fn get() *T {
            return &storage;
        }
    };
}
