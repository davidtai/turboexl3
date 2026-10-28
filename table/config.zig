//! turboexl3's claim, validation and bill, MLX-free (the server's quant-block claims and its load preflight run them).
//! It claims the EXL3 blocks it serves at `by_marker`, above the server's built-in Sushi table: a fully EXL3 pack's
//! `trunk_quant` (the trunk's dense linears; no rate `k`, each linear carries its own) always, and that pack's
//! `expert_quant` (the routed banks, rate `k`) where MLX's NAX gate holds and the SwiGLU is unclamped: a fully EXL3
//! Qwen3.8 pack, the configuration its reordered numerics were measured on. An expert-only pack's banks stay Sushi's.
//! No environment variable changes a claim.

const std = @import("std");
const abi = @import("sdk").quant_abi;
const nax = @import("turboexl3_kit").nax;

/// A dense trunk's block (`trunk_quant`) or a routed bank set's (`expert_quant`).
pub const Kind = enum { trunk, bank };

pub const Block = struct {
    kind: Kind,
    codebook: Codebook = .mul1,
    window_bits: u8 = 16,
    /// A bank block's rate: halfwords per 16x16 tile (16 k).
    n: u32 = 0,

    pub const Codebook = enum { mul1, mcg };
};

pub const BlockError = error{ NotExl3, Exl3CodebookUnsupported, Exl3WindowUnsupported, Exl3RateUnsupported };

/// The widest top-k turboexl3's decode reduction holds.
pub const max_topk = 32;

/// An EXL3 block turboexl3 decodes; `NotExl3` for any other (another format, Sushi's STQ family).
pub fn parseBlock(a: std.mem.Allocator, json: []const u8) BlockError!Block {
    const o = parseObject(a, json) orelse return error.NotExl3;
    var b: Block = .{ .kind = kindOf(o) orelse return error.NotExl3 };
    if (o.get("codebook")) |c| {
        if (c != .string) return error.Exl3CodebookUnsupported;
        b.codebook = std.meta.stringToEnum(Block.Codebook, c.string) orelse return error.Exl3CodebookUnsupported;
    } else if (b.kind == .bank) return error.Exl3CodebookUnsupported;
    if (o.get("window")) |w| {
        if (w != .integer or w.integer < 8 or w.integer > 16) return error.Exl3WindowUnsupported;
        b.window_bits = @intCast(w.integer);
    }
    if (b.kind == .bank) b.n = rateOf(o.get("k").?) orelse return error.Exl3RateUnsupported;
    return b;
}

pub fn claims(block_json: abi.Str, config_json: abi.Str) callconv(.c) abi.Claim {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    return claimFor(arena.allocator(), block_json.slice(), config_json.slice(), nax.available());
}

/// The claim on one block over the NAX gate (`nax_on`).
pub fn claimFor(a: std.mem.Allocator, block_json: []const u8, config_json: []const u8, nax_on: bool) abi.Claim {
    const o = parseObject(a, block_json) orelse return .no;
    const kind = kindOf(o) orelse return .no;
    // A trunk block is turboexl3's alone (no built-in format runs a dense EXL3 linear); validation names what it lacks.
    if (kind == .trunk) return .by_marker;
    _ = parseBlock(a, block_json) catch return .no;
    const cfg = parseObject(a, config_json);
    if (!profile(cfg) or !nax_on or clamped(cfg)) return .no;
    return .by_marker;
}

pub fn validate(block_json: abi.Str, config_json: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Verdict {
    _ = dir;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const b = parseBlock(a, block_json.slice()) catch |e| return abi.Verdict.named(.unsupported, @errorName(e));
    if (b.kind == .bank) {
        const k = intField(textOf(parseObject(a, config_json.slice())), "num_experts_per_tok") orelse 0;
        if (k > max_topk) return abi.Verdict.refuse(.unsupported, "Exl3TopKExceedsReduceBank");
    }
    return .ok;
}

/// Its weights are the server's loaded arrays, as stored.
pub fn bill(block_json: abi.Str, model_dir: abi.Str, stored: u64) callconv(.c) u64 {
    _ = block_json;
    _ = model_dir;
    return stored;
}

/// A routed bank set's block declares its rate `k`; a dense trunk's does not.
fn kindOf(o: std.json.ObjectMap) ?Kind {
    const f = o.get("format") orelse return null;
    if (f != .string or !std.mem.eql(u8, f.string, "exl3")) return null;
    return if (o.get("k") != null) .bank else .trunk;
}

/// `k` as halfwords per tile: an even whole number in 16..128 (fractional `k` included).
fn rateOf(k: std.json.Value) ?u32 {
    const scaled: f64 = switch (k) {
        .integer => |i| @as(f64, @floatFromInt(i)) * 16.0,
        .float => |x| x * 16.0,
        else => return null,
    };
    const n = @round(scaled);
    if (@abs(scaled - n) > 1e-6 or n < 16 or n > 128) return null;
    const whole: u32 = @intFromFloat(n);
    return if (whole % 2 == 0) whole else null;
}

/// A fully EXL3 Qwen3.8 pack (`qwen4_exp` with a `trunk_quant`), the configuration the reordered numerics were measured
/// on.
fn profile(cfg: ?std.json.ObjectMap) bool {
    if ((cfg orelse return false).get("trunk_quant") == null) return false;
    for ([_]?std.json.ObjectMap{ cfg, textOf(cfg) }) |o| {
        const mt = (o orelse continue).get("model_type") orelse continue;
        if (mt == .string and (std.mem.eql(u8, mt.string, "qwen4_exp") or std.mem.eql(u8, mt.string, "qwen4_exp_text"))) return true;
    }
    return false;
}

/// A clamped SwiGLU (GLM's `swiglu_limit`): no turboexl3 kernel serves it.
fn clamped(cfg: ?std.json.ObjectMap) bool {
    for ([_]?std.json.ObjectMap{ cfg, textOf(cfg) }) |o| {
        const v = (o orelse continue).get("swiglu_limit") orelse continue;
        switch (v) {
            .integer => |i| if (i > 0) return true,
            .float => |x| if (x > 0) return true,
            else => {},
        }
    }
    return false;
}

fn parseObject(a: std.mem.Allocator, json: []const u8) ?std.json.ObjectMap {
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return null;
    return if (v == .object) v.object else null;
}

/// The text config of a config root: `text_config` when it nests one, else the root.
fn textOf(cfg: ?std.json.ObjectMap) ?std.json.ObjectMap {
    const o = cfg orelse return null;
    const t = o.get("text_config") orelse return o;
    return if (t == .object) t.object else o;
}

fn intField(o: ?std.json.ObjectMap, key: []const u8) ?i64 {
    const v = (o orelse return null).get(key) orelse return null;
    return if (v == .integer) v.integer else null;
}

// ── tests ──

const testing = std.testing;

test {
    _ = nax;
}

const trunk_block = "{\"format\":\"exl3\",\"codebook\":\"mul1\",\"window\":16}";
const bank_block = "{\"format\":\"exl3\",\"k\":5,\"codebook\":\"mul1\"}";
const qwen38 = "{\"model_type\":\"qwen4_exp\",\"trunk_quant\":" ++ trunk_block ++ ",\"text_config\":{\"num_experts_per_tok\":8}}";
const qwen38_experts_only = "{\"model_type\":\"qwen4_exp\",\"text_config\":{\"num_experts_per_tok\":8}}";
const glm = "{\"model_type\":\"glm5_next\",\"swiglu_limit\":7.0}";

test "turboexl3 config: a trunk block is always claimed; a bank block on a fully EXL3 Qwen3.8 pack with NAX and no clamp" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ true, false }) |on| try testing.expectEqual(abi.Claim.by_marker, claimFor(a, trunk_block, "{}", on));
    try testing.expectEqual(abi.Claim.by_marker, claimFor(a, bank_block, qwen38, true));
    try testing.expectEqual(abi.Claim.no, claimFor(a, bank_block, qwen38, false));
    // An expert-only pack's banks stay Sushi's, as do another model's; GLM's clamp is never served.
    try testing.expectEqual(abi.Claim.no, claimFor(a, bank_block, qwen38_experts_only, true));
    try testing.expectEqual(abi.Claim.no, claimFor(a, bank_block, "{\"model_type\":\"mimo_v2\",\"trunk_quant\":" ++ trunk_block ++ "}", true));
    try testing.expectEqual(abi.Claim.no, claimFor(a, bank_block, glm, true));
    // Sushi's STQ family, another format, a broken block: not turboexl3's.
    try testing.expectEqual(abi.Claim.no, claimFor(a, "{\"format\":\"stq\",\"k\":3,\"codebook\":\"std1\"}", qwen38, true));
    try testing.expectEqual(abi.Claim.no, claimFor(a, "{\"format\":\"exl3\",\"k\":3,\"codebook\":\"pair1\"}", qwen38, true));
    try testing.expectEqual(abi.Claim.no, claimFor(a, "{\"bits\":4,\"group_size\":64}", qwen38, true));
    try testing.expectEqual(abi.Claim.no, claimFor(a, "{}", "{}", true));
    try testing.expectEqual(abi.Claim.no, claimFor(a, "{", "{}", true));
}

test "turboexl3 config: validation names an unserved codebook, window, rate or top-k" {
    const dir: abi.PackDir = undefined;
    try testing.expect(validate(.of(trunk_block), .of("{}"), &dir).isOk());
    try testing.expect(validate(.of(bank_block), .of(qwen38), &dir).isOk());
    try testing.expect(validate(.of("{\"format\":\"exl3\",\"codebook\":\"mcg\",\"window\":12}"), .of("{}"), &dir).isOk());
    for ([_]struct { block: []const u8, config: []const u8, want: []const u8 }{
        .{ .block = "{\"format\":\"exl3\",\"codebook\":\"std1\"}", .config = "{}", .want = "Exl3CodebookUnsupported" },
        .{ .block = "{\"format\":\"exl3\",\"window\":7}", .config = "{}", .want = "Exl3WindowUnsupported" },
        .{ .block = "{\"format\":\"exl3\",\"k\":1.03,\"codebook\":\"mul1\"}", .config = "{}", .want = "Exl3RateUnsupported" },
        .{ .block = "{\"format\":\"exl3\",\"k\":9,\"codebook\":\"mul1\"}", .config = "{}", .want = "Exl3RateUnsupported" },
        .{ .block = bank_block, .config = "{\"num_experts_per_tok\":33}", .want = "Exl3TopKExceedsReduceBank" },
        .{ .block = "{\"bits\":4}", .config = "{}", .want = "NotExl3" },
    }) |c| try testing.expectEqualStrings(c.want, validate(.of(c.block), .of(c.config), &dir).why.slice());
}

test "turboexl3 config: a bank block's rate is its k in halfwords per tile, fractional k included" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const b = try parseBlock(a, bank_block);
    try testing.expectEqual(Kind.bank, b.kind);
    try testing.expectEqual(@as(u32, 80), b.n);
    try testing.expectEqual(@as(u32, 40), (try parseBlock(a, "{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\",\"window\":14}")).n);
    const t = try parseBlock(a, trunk_block);
    try testing.expect(t.kind == .trunk and t.n == 0 and t.window_bits == 16);
}
