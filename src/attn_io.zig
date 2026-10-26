//! A gated, QK-normed, partially rotated attention layer's EXL3 projections at prefill rows, fused into fewer passes and
//! launches, byte for byte the chain they replace (`Dense.linear` per projection plus MLX's ops). The host takes them per
//! call (MLX_SERVE_ATTN_IN, MLX_SERVE_ATTN_OUT); Qwen3.8-Flash-Next's 12 full-attention layers are the client.
//!
//! `attnIn` (ATTN-IN): q (each head's 512 columns = [query 256 | gate 256]), k and v of one input x.
//! 1. One prepare launch (`text_prep3`): x read once, xp_m = H128(f16(x) * suh_m) for the three members, natural order;
//!    each member's statements are `dense.text_prep`'s (LANE_ORDER 0) on the same f16 x value.
//! 2. The members' inner products are `linear`'s own: the register-path GEMM up to `small_rows`, dequantize + MLX's matmul
//!    above (`prefillInner`). The three GEMMs cannot share a launch: each multiplies its own rotated copy.
//! 3. One finish launch (`text_finish_attn`, 64 threads per 256-column unit and row): each simdgroup runs `text_finish`'s
//!    statements on its 128-block (f16 round, H128, x svh, f16 round, the output dtype's round), then
//!    - query heads and k heads: MLX's `rms_single_row` at axis 256 (thread t owns [4t, 4t+4), f32 squares in index order,
//!      simd_sum per simdgroup, the 32-slot second simd_sum, precise::rsqrt(sum / 256 + eps), T(w * T(x * inv))), then the
//!      NeoX rotation of the first RD dims against the host's angle rows (`ropeAngleRows`: MLX's own rope over a unit probe,
//!      so cos and sin are its floats), one rounding; these are the statements of the host's hd-256 QK-norm+RoPE kernel
//!      (`QK_NORM_ROPE_256_SOURCE`), which its parity test pins to the composed rms_norm -> transpose -> rope chain;
//!    - gate halves and v heads: the finish's values;
//!    - every output in the layout its consumer reads: q, k, v head-major [1, H, S, D] (the KV cache's chunk and the QSA
//!      core's q), the gate [1, S, H, D].
//!    The KV-cache write stays the host's `slice_update` (MLX custom kernels allocate their outputs: one cannot write into
//!    the cache buffer); it now copies a contiguous chunk instead of reading v through a transposed view.
//!
//! `linearGated` (ATTN-OUT): o_proj over T(attn * sigmoid(gate)), the product formed inside its prepare (`text_prep_gate`):
//! the sigmoid by the host's table of MLX's sigmoid over every 16-bit pattern, the multiply in the activation dtype (both
//! exact in f32 before the one rounding, as MLX's Sigmoid and Multiply), then `text_prep`'s statements; the inner product
//! and finish are `linear`'s (the finish is `Dense.finish` itself with `linear`'s config).
//!
//! Rows <= 16 (the GEMV path) and the fused-GEMM option (MLX_SERVE_TURBOEXL3_DENSE_PREFILL=fused) are refused
//! (error.Exl3EngineNotImplemented): the host keeps its chain there.
//!
//! The four-member prepare (the host's MLX_SERVE_ATTN_IDXPREP): the layer's QSA indexer projection reads the same x as
//! q|gate, k and v, so its prepare joins step 1 as a fourth member (`prepare4`, `text_prep4`: x read once for the four
//! rotated copies, every member's statements `text_prep3`'s). The indexer's inner product and finish are `linear`'s own
//! launches over its copy (`linearPrepared`: `Dense.innerOf` / `Dense.finishOf` on the path `linear` takes at those rows),
//! and `attnIn` takes the other three as given (`Epilogue.prep`) instead of launching its prepare: the bytes of the
//! indexer's `linear` and ATTN-IN's own prepare.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const decode = @import("decode.zig");
const dense = @import("dense.zig");

/// One prepare for three members: xp_m half [rows, DIM] = H128(f16(x[row]) * suh_m) per 128-block, natural order (dense
/// `text_prep` at LANE_ORDER 0, member by member on the one f16 value of x); one simdgroup per (block, row).
pub const text_prep3 =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float xh[4];
    \\    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; xh[j] = float(half(float(x[row * DIMC + i]))); }
    \\    for (uint m = 0; m < 3u; ++m) {
    \\        const device half* suh = (m == 0u) ? suh0 : ((m == 1u) ? suh1 : suh2);
    \\        device half* xp = (m == 0u) ? xp0 : ((m == 1u) ? xp1 : xp2);
    \\        float v[4];
    \\        for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = xh[j] * float(suh[i]); }
    \\        for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\        { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\        for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; xp[row * DIMC + i] = half(v[j] * HAD_SCALE); }
    \\    }
    \\
;

/// One finish for q|gate, k and v with the attention epilogue. Grid (units x 64 threads, rows); unit u < HQ: query head u
/// (q_proj blocks 4u, 4u+1), HQ <= u < 2 HQ: head u - HQ's gate half (blocks 4h+2, 4h+3), then HK k heads (k_proj blocks
/// 2h, 2h+1), then HK v heads. Simdgroup sg finishes the unit's block sg (dense `text_finish`'s statements); query and k
/// units then normalize and rotate the 256 values (the host's hd-256 QK-norm+RoPE statements). zq/zk/zv f16 [rows, width],
/// svq/svk/svv f16 [width], qw/kw OUT_T [HD], angles f32 [rows, RD] (cos | sin per row), eps f32 (0-dim).
/// oq OUT_T [1, HQ, rows, HD], og OUT_T [1, rows, HQ, HD], ok / ov OUT_T [1, HK, rows, HD].
/// (Two pieces, one text: `finish_attn_head` finishes the unit's block into y[], `finish_attn_tail` is the epilogue from
/// y[] on, which the standard-basis `text_epi_attn` repeats verbatim.)
pub const text_finish_attn: [:0]const u8 = finish_attn_head ++ finish_attn_tail;

/// `text_finish_attn` up to the finished y[] (the unit's block of its member: f16 round, H128, x svh, f16 round, OUT_T)
const finish_attn_head =
    \\    constexpr uint HDC = uint(HD);
    \\    constexpr uint QDIM = 2u * uint(HQ) * HDC;
    \\    constexpr uint KDIM = uint(HK) * HDC;
    \\    constexpr uint half_rd = uint(RD) / 2u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    threadgroup OUT_T ys[HD];
    \\    threadgroup OUT_T nrm[HD];
    \\    threadgroup float local_sums[32];
    \\    threadgroup float local_inv[1];
    \\    const uint u = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint S = threadgroups_per_grid.y;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const bool on_q = u < 2u * uint(HQ);
    \\    const bool is_g = on_q && u >= uint(HQ);
    \\    const bool is_k = !on_q && u < 2u * uint(HQ) + uint(HK);
    \\    const bool is_v = !on_q && !is_k;
    \\    const uint h = on_q ? (is_g ? u - uint(HQ) : u) : (is_k ? u - 2u * uint(HQ) : u - 2u * uint(HQ) - uint(HK));
    \\    const device half* z = on_q ? zq : (is_k ? zk : zv);
    \\    const device half* svh = on_q ? svq : (is_k ? svk : svv);
    \\    const uint dimc = on_q ? QDIM : KDIM;
    \\    const uint blk = on_q ? (4u * h + (is_g ? 2u : 0u) + sg) : (2u * h + sg);
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(z[row * dimc + blk * 128u + L + 32u * j])));
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    OUT_T y[4];
    \\    for (uint j = 0; j < 4u; ++j) { const uint o = blk * 128u + L + 32u * j; y[j] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
    \\
;

/// `text_finish_attn`'s epilogue from the finished y[] on: the gate / v stores, the q / k heads' RMSNorm and NeoX rotation
const finish_attn_tail =
    \\    const uint c0 = sg * 128u + L;
    \\    if (is_g) {
    \\        for (uint j = 0; j < 4u; ++j) og[(ulong(row) * uint(HQ) + h) * HDC + c0 + 32u * j] = y[j];
    \\    } else if (is_v) {
    \\        for (uint j = 0; j < 4u; ++j) ov[(ulong(h) * S + row) * HDC + c0 + 32u * j] = y[j];
    \\    } else {
    \\        for (uint j = 0; j < 4u; ++j) ys[c0 + 32u * j] = y[j];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        const uint base = tid * 4u;
    \\        float acc = 0.0f;
    \\        for (uint i = 0; i < 4u; ++i) {
    \\            float xv = float(ys[base + i]);
    \\            acc += xv * xv;
    \\        }
    \\        acc = simd_sum(acc);
    \\        if (sg == 0u) {
    \\            local_sums[L] = 0.0f;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        if (L == 0u) {
    \\            local_sums[sg] = acc;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        if (sg == 0u) {
    \\            acc = simd_sum(local_sums[L]);
    \\            if (L == 0u) {
    \\                local_inv[0] = metal::precise::rsqrt(acc / float(HDC) + float(eps));
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        const float inv = local_inv[0];
    \\        const device OUT_T* w = is_k ? kw : qw;
    \\        for (uint i = 0; i < 4u; ++i) {
    \\            nrm[base + i] = OUT_T(w[base + i] * OUT_T(float(ys[base + i]) * inv));
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        const device float* arow = angles + ulong(row) * uint(RD);
    \\        device OUT_T* dst = is_k ? (ok + (ulong(h) * S + row) * HDC) : (oq + (ulong(h) * S + row) * HDC);
    \\        for (uint i = 0; i < 4u; ++i) {
    \\            const uint j = base + i;
    \\            OUT_T r;
    \\            if (j >= uint(RD)) {
    \\                r = nrm[j];
    \\            } else if (j < half_rd) {
    \\                r = OUT_T(float(nrm[j]) * arow[j] - float(nrm[j + half_rd]) * arow[half_rd + j]);
    \\            } else {
    \\                r = OUT_T(float(nrm[j - half_rd]) * arow[j] + float(nrm[j]) * arow[j - half_rd]);
    \\            }
    \\            dst[j] = r;
    \\        }
    \\    }
    \\
;

/// o_proj's prepare over the gated attention output: x[row, i] = T(attn[h, row, d] * sigtab[gate(row, h, d)]) with
/// i = h HD + d (MLX's Sigmoid by its own table, MLX's Multiply in T), then dense `text_prep`'s statements (LANE_ORDER 0).
/// attn T [1, H, rows, HD] (the QSA core's head-major output), gate T at gate[row GROW + h GHEAD + GOFF + d] (q_proj's
/// [1, rows, 2 H HD] with GROW 2 H HD, GHEAD 2 HD, GOFF HD; ATTN-IN's [1, rows, H, HD] with GROW H HD, GHEAD HD, GOFF 0),
/// sigtab T [65536]; xp half [rows, DIM]; one simdgroup per (block, row).
pub const text_prep_gate =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr uint HDC = uint(HD);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint S = threadgroups_per_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint i = b * 128u + L + 32u * j;
    \\        const uint hh = i / HDC;
    \\        const uint d = i - hh * HDC;
    \\        const T av = attn[(ulong(hh) * S + row) * HDC + d];
    \\        const T gv = gate[ulong(row) * uint(GROW) + hh * uint(GHEAD) + uint(GOFF) + d];
    \\        const T xv = av * sigtab[as_type<ushort>(gv)];
    \\        v[j] = float(half(float(xv))) * float(suh[i]);
    \\    }
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; xp[row * DIMC + i] = half(v[j] * HAD_SCALE); }
    \\
;

/// `text_prep3` with a fourth member (the QSA indexer projection of the same x): xp_m half [rows, DIM] =
/// H128(f16(x[row]) * suh_m) per 128-block, natural order, for m = 0..3; each member's statements are `text_prep3`'s (so
/// dense `text_prep`'s at LANE_ORDER 0) on the one f16 value of x; one simdgroup per (block, row).
pub const text_prep4 =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float xh[4];
    \\    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; xh[j] = float(half(float(x[row * DIMC + i]))); }
    \\    for (uint m = 0; m < 4u; ++m) {
    \\        const device half* suh = (m == 0u) ? suh0 : ((m == 1u) ? suh1 : ((m == 2u) ? suh2 : suh3));
    \\        device half* xp = (m == 0u) ? xp0 : ((m == 1u) ? xp1 : ((m == 2u) ? xp2 : xp3));
    \\        float v[4];
    \\        for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = xh[j] * float(suh[i]); }
    \\        for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\        { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\        for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; xp[row * DIMC + i] = half(v[j] * HAD_SCALE); }
    \\    }
    \\
;

/// ATTN-IN's epilogue over the projections in the standard basis (the un-rotated trunk's q|gate, k and v: one plain
/// matmul each, or the columns of one joined product). `text_finish_attn` with its finish replaced by a read: unit u's
/// block of its member is y[j] = z[row ZS + ZO + blk 128 + L + 32 j] (OUT_T, the matmul's output as it is), zq / zk / zv
/// OUT_T [rows, QS / KS / VS] holding q|gate, k and v at columns QO / KO / VO (one array three times for a join); the units,
/// blocks and every statement from y[] on are `text_finish_attn`'s (`finish_attn_tail` verbatim).
pub const text_epi_attn: [:0]const u8 = epi_attn_head ++ finish_attn_tail;

const epi_attn_head =
    \\    constexpr uint HDC = uint(HD);
    \\    constexpr uint half_rd = uint(RD) / 2u;
    \\    threadgroup OUT_T ys[HD];
    \\    threadgroup OUT_T nrm[HD];
    \\    threadgroup float local_sums[32];
    \\    threadgroup float local_inv[1];
    \\    const uint u = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint S = threadgroups_per_grid.y;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const bool on_q = u < 2u * uint(HQ);
    \\    const bool is_g = on_q && u >= uint(HQ);
    \\    const bool is_k = !on_q && u < 2u * uint(HQ) + uint(HK);
    \\    const bool is_v = !on_q && !is_k;
    \\    const uint h = on_q ? (is_g ? u - uint(HQ) : u) : (is_k ? u - 2u * uint(HQ) : u - 2u * uint(HQ) - uint(HK));
    \\    const device OUT_T* z = on_q ? zq : (is_k ? zk : zv);
    \\    const uint zs = on_q ? uint(QS) : (is_k ? uint(KS) : uint(VS));
    \\    const uint zo = on_q ? uint(QO) : (is_k ? uint(KO) : uint(VO));
    \\    const uint blk = on_q ? (4u * h + (is_g ? 2u : 0u) + sg) : (2u * h + sg);
    \\    OUT_T y[4];
    \\    for (uint j = 0; j < 4u; ++j) y[j] = z[ulong(row) * zs + zo + blk * 128u + L + 32u * j];
    \\
;

/// ATTN-OUT's input in the standard basis, the row o_proj's plain matmul reads: xg T [rows, DIM] with
/// xg[row, h HD + d] = T(attn[h, row, d] * sigtab[gate(row, h, d)]), `text_prep_gate`'s product statements verbatim (MLX's
/// Sigmoid by its table, its Multiply in T) and no prepare; operands as `text_prep_gate`'s, one simdgroup per (block, row).
pub const text_gate_plain =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr uint HDC = uint(HD);
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint S = threadgroups_per_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint i = b * 128u + L + 32u * j;
    \\        const uint hh = i / HDC;
    \\        const uint d = i - hh * HDC;
    \\        const T av = attn[(ulong(hh) * S + row) * HDC + d];
    \\        const T gv = gate[ulong(row) * uint(GROW) + hh * uint(GHEAD) + uint(GOFF) + d];
    \\        const T xv = av * sigtab[as_type<ushort>(gv)];
    \\        xg[ulong(row) * DIMC + i] = xv;
    \\    }
    \\
;

const epi_in = [_][*:0]const u8{ "zq", "zk", "zv", "qw", "kw", "angles", "eps" };
const gate_plain_in = [_][*:0]const u8{ "attn", "gate", "sigtab" };
const gate_plain_out = [_][*:0]const u8{"xg"};

const prep3_in = [_][*:0]const u8{ "x", "suh0", "suh1", "suh2" };
const prep3_out = [_][*:0]const u8{ "xp0", "xp1", "xp2" };
const prep4_in = [_][*:0]const u8{ "x", "suh0", "suh1", "suh2", "suh3" };
const prep4_out = [_][*:0]const u8{ "xp0", "xp1", "xp2", "xp3" };
const fin_in = [_][*:0]const u8{ "zq", "zk", "zv", "svq", "svk", "svv", "qw", "kw", "angles", "eps" };
const fin_out = [_][*:0]const u8{ "oq", "og", "ok", "ov" };
const prep_gate_in = [_][*:0]const u8{ "attn", "gate", "sigtab", "suh" };
const prep_gate_out = [_][*:0]const u8{"xp"};

/// Threads per finish unit (two simdgroups: one 256-column head).
pub const unit_threads: u32 = 64;
/// Head widths the epilogue serves (one unit = two 128-blocks).
pub const head_dim: u32 = 256;

/// The finish launch's units for `heads` query heads and `kv_heads` k / v heads: a query and a gate unit per head, one unit
/// per k head and per v head.
pub fn finishUnits(heads: u32, kv_heads: u32) u32 {
    return 2 * heads + 2 * kv_heads;
}

/// The rotary widths the staged rotation serves: an even RD whose pair partner (j +- RD/2) stays inside the 256-wide row
/// (the host's hd-256 kernel's set).
pub fn ropeDimsServed(rd: u32) bool {
    return rd == 32 or rd == 64 or rd == 128 or rd == 256;
}

const Kernels = struct {
    prep3: mlx.mlx_fast_metal_kernel,
    fin: mlx.mlx_fast_metal_kernel,
    prep_gate: mlx.mlx_fast_metal_kernel,
};

var kernels_lock: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
var kernels: ?Kernels = null;

fn kernelsGet() !*const Kernels {
    while (kernels_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernels_lock.store(0, .release);
    if (kernels) |*k| return k;
    const prep3 = ks.newKernel("turboexl3_attn_prep3", &prep3_in, &prep3_out, text_prep3, decode.header, true) orelse return error.KernelCreateFailed;
    errdefer _ = mlx.mlx_fast_metal_kernel_free(prep3);
    const fin = ks.newKernel("turboexl3_attn_finish", &fin_in, &fin_out, text_finish_attn, decode.header, true) orelse return error.KernelCreateFailed;
    errdefer _ = mlx.mlx_fast_metal_kernel_free(fin);
    const pg = ks.newKernel("turboexl3_attn_prep_gate", &prep_gate_in, &prep_gate_out, text_prep_gate, decode.header, true) orelse return error.KernelCreateFailed;
    kernels = .{ .prep3 = prep3, .fin = fin, .prep_gate = pg };
    return &kernels.?;
}

/// The four-member prepare's kernel, made at its first call (a process that never takes it never builds it).
var prep4_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn prep4KernelGet() !mlx.mlx_fast_metal_kernel {
    while (kernels_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernels_lock.store(0, .release);
    if (prep4_kernel) |k| return k;
    const k = ks.newKernel("turboexl3_attn_prep4", &prep4_in, &prep4_out, text_prep4, decode.header, true) orelse return error.KernelCreateFailed;
    prep4_kernel = k;
    return k;
}

/// The standard-basis kernels, made at their first call (a process without the un-rotated trunk never builds them).
var std_kernels: ?[2]mlx.mlx_fast_metal_kernel = null;

fn stdKernelsGet() ![2]mlx.mlx_fast_metal_kernel {
    while (kernels_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernels_lock.store(0, .release);
    if (std_kernels) |k| return k;
    const epi = ks.newKernel("turboexl3_attn_epi_std", &epi_in, &fin_out, text_epi_attn, decode.header, true) orelse return error.KernelCreateFailed;
    errdefer _ = mlx.mlx_fast_metal_kernel_free(epi);
    const gate = ks.newKernel("turboexl3_attn_gate_std", &gate_plain_in, &gate_plain_out, text_gate_plain, decode.header, true) orelse return error.KernelCreateFailed;
    std_kernels = .{ epi, gate };
    return std_kernels.?;
}

/// One projection of a fused call: its kernels (the `Dense` `linear` uses for its shape) and the host's arrays.
pub const Member = struct {
    d: *dense.Dense,
    trellis: mlx.mlx_array,
    suh: mlx.mlx_array,
    svh: mlx.mlx_array,
};

/// What ATTN-IN's finish applies after the plain finish.
pub const Epilogue = struct {
    /// query heads: q_proj is 2 heads head_dim wide, each head's columns [query | gate]
    heads: u32,
    /// k / v heads (k_proj and v_proj kv_heads head_dim wide)
    kv_heads: u32,
    /// the rotated leading dims of each 256-wide head (32, 64, 128 or 256)
    rope_dims: u32,
    /// per-head RMSNorm weights [256] in the activation dtype
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    /// 0-dim f32: the norms' eps (the host's `rms_eps_arr`)
    eps: mlx.mlx_array,
    /// f32 [rows * rope_dims]: row p holds cos then sin at the chunk's position p (the host's `ropeAngleRows`)
    angles: mlx.mlx_array,
    /// q's, k's and v's rotated copies f16 [rows, in] as the four-member prepare (`prepare4`) wrote them beside the
    /// indexer's (the caller's); null: step 1's prepare launch
    prep: ?[3]mlx.mlx_array = null,
};

/// ATTN-IN's outputs (the caller frees each): q and k normed and rotated, v as projected, all [1, heads, rows, 256]; the
/// gate halves [1, rows, heads, 256].
pub const InOut = struct {
    q: mlx.mlx_array,
    gate: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,

    pub fn deinit(self: *InOut) void {
        inline for (.{ &self.q, &self.gate, &self.k, &self.v }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
            a.* = .{ .ctx = null };
        }
    }
};

/// Where o_proj's prepare reads the gate: element (row, head h, d) at base[row row_stride + h head_stride + off + d].
pub const Gate = struct {
    base: mlx.mlx_array,
    row_stride: u32,
    head_stride: u32,
    off: u32,
};

/// Does `Dense.linear` take `prefillInner`'s paths at `rows` (the register-path GEMM or dequantize + matmul)? Not at GEMV
/// rows, not with the fused-GEMM option, and not on register-path rows while the register-path toggle is set (its
/// per-call file check stays `linear`'s alone).
pub fn innerServes(d: *const dense.Dense, rows: usize) bool {
    if (rows <= decode.Kernels.max_rows or d.fused_prefill) return false;
    if (rows <= d.small_rows and std.c.getenv("MLX_SERVE_TURBOEXL3_DENSE_SMALL_TOGGLE") != null) return false;
    return true;
}

/// The inner product `Dense.linear` computes for `rows` prepared rows (natural order) where `innerServes`: z half
/// [rows, out], the register-path GEMM up to `small_rows`, dequantize + MLX's matmul above (the same calls `linear` makes).
pub fn prefillInner(d: *dense.Dense, s: mlx.mlx_stream, xp: mlx.mlx_array, trellis: mlx.mlx_array, rows: usize) !mlx.mlx_array {
    if (!innerServes(d, rows)) return error.Exl3EngineNotImplemented;
    if (rows <= d.small_rows) return d.smallInner(s, xp, trellis, rows);
    const W = try d.dequantize(s, trellis);
    defer _ = mlx.mlx_array_free(W);
    var inner = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inner);
    try mlx.check(mlx.mlx_matmul(&inner, xp, W, s));
    return inner;
}

fn ci(v: anytype) c_int {
    return @intCast(v);
}

/// ATTN-IN over x2 [rows, in] (the activation dtype, bf16 or f16): see the file's doc. Refuses (error.Exl3EngineNotImplemented)
/// what it does not serve, before building anything.
pub fn attnIn(s: mlx.mlx_stream, x2: mlx.mlx_array, rows: usize, q: Member, k: Member, v: Member, ep: Epilogue) !InOut {
    const dt = q.d.shape.out_dtype;
    const in_dim = q.d.shape.in;
    if (dt != .bfloat16 and dt != .float16) return error.Exl3EngineNotImplemented;
    if (k.d.shape.in != in_dim or v.d.shape.in != in_dim or k.d.shape.out_dtype != dt or v.d.shape.out_dtype != dt) return error.BadExl3Shape;
    if (ep.heads == 0 or ep.kv_heads == 0 or !ropeDimsServed(ep.rope_dims)) return error.Exl3EngineNotImplemented;
    if (q.d.shape.out != 2 * ep.heads * head_dim or k.d.shape.out != ep.kv_heads * head_dim or v.d.shape.out != ep.kv_heads * head_dim) return error.BadExl3Shape;
    if (!innerServes(q.d, rows) or !innerServes(k.d, rows) or !innerServes(v.d, rows)) return error.Exl3EngineNotImplemented;
    if (mlx.mlx_array_dtype(ep.q_norm) != dt or mlx.mlx_array_dtype(ep.k_norm) != dt or mlx.mlx_array_size(ep.q_norm) != head_dim or mlx.mlx_array_size(ep.k_norm) != head_dim) return error.Exl3EngineNotImplemented;
    if (mlx.mlx_array_dtype(ep.angles) != .float32 or mlx.mlx_array_size(ep.angles) != rows * ep.rope_dims) return error.Exl3EngineNotImplemented;
    if (mlx.mlx_array_dtype(ep.eps) != .float32 or mlx.mlx_array_ndim(ep.eps) != 0) return error.Exl3EngineNotImplemented;
    const kz = try kernelsGet();

    // 1. the three rotated copies, x read once (or, given, as `prepare4` wrote them beside the indexer's: the caller's)
    const xs = [_]c_int{ ci(rows), ci(in_dim) };
    var xp: [3]mlx.mlx_array = undefined;
    if (ep.prep) |p| {
        for (p) |a| if (mlx.mlx_array_dtype(a) != .float16 or !std.mem.eql(c_int, mlx.getShape(a), &xs)) return error.BadExl3Shape;
        xp = p;
    } else xp = try prepare3(s, x2, rows, q, k, v);
    defer if (ep.prep == null) for (xp) |a| {
        _ = mlx.mlx_array_free(a);
    };

    // 2. the members' inner products, `linear`'s calls
    const zq = try prefillInner(q.d, s, xp[0], q.trellis, rows);
    defer _ = mlx.mlx_array_free(zq);
    const zk = try prefillInner(k.d, s, xp[1], k.trellis, rows);
    defer _ = mlx.mlx_array_free(zk);
    const zv = try prefillInner(v.d, s, xp[2], v.trellis, rows);
    defer _ = mlx.mlx_array_free(zv);

    // 3. one finish with the epilogue
    const hq = ci(ep.heads);
    const hk = ci(ep.kv_heads);
    const r = ci(rows);
    const hd = ci(head_dim);
    const out_t = ks.TemplateArg{ .name = "OUT_T", .value = .{ .dtype = dt } };
    const fcfg = try ks.buildConfig(.{
        .grid = .{ finishUnits(ep.heads, ep.kv_heads) * unit_threads, @intCast(rows), 1 },
        .threadgroup = .{ unit_threads, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ 1, hq, r, hd }, &[_]c_int{ 1, r, hq, hd }, &[_]c_int{ 1, hk, r, hd }, &[_]c_int{ 1, hk, r, hd } },
        .out_dtypes = &.{ dt, dt, dt, dt },
        .template = &.{
            .{ .name = "HQ", .value = .{ .int = hq } },
            .{ .name = "HK", .value = .{ .int = hk } },
            .{ .name = "HD", .value = .{ .int = hd } },
            .{ .name = "RD", .value = .{ .int = ci(ep.rope_dims) } },
            out_t,
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(fcfg);
    var fo: [4]mlx.mlx_array = undefined;
    try ks.applyConfig(kz.fin, fcfg, &.{ zq, zk, zv, q.svh, k.svh, v.svh, ep.q_norm, ep.k_norm, ep.angles, ep.eps }, &fo, s);
    return .{ .q = fo[0], .gate = fo[1], .k = fo[2], .v = fo[3] };
}

/// Step 1's launch (`text_prep3`): q's, k's and v's rotated copies f16 [rows, in] of x2 [rows, in], each the caller's.
pub fn prepare3(s: mlx.mlx_stream, x2: mlx.mlx_array, rows: usize, q: Member, k: Member, v: Member) ![3]mlx.mlx_array {
    const kz = try kernelsGet();
    const in_dim = q.d.shape.in;
    const xs = [_]c_int{ ci(rows), ci(in_dim) };
    const pcfg = try ks.buildConfig(.{ .grid = .{ (in_dim / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{ &xs, &xs, &xs }, .out_dtypes = &.{ .float16, .float16, .float16 }, .template = &.{.{ .name = "DIM", .value = .{ .int = ci(in_dim) } }} });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(pcfg);
    var xp: [3]mlx.mlx_array = undefined;
    try ks.applyConfig(kz.prep3, pcfg, &.{ x2, q.suh, k.suh, v.suh }, &xp, s);
    return xp;
}

// ---- the four-member prepare: the indexer's prepare as ATTN-IN's fourth member ----

/// The paths of `linear` that read prepared natural-order rows (`Dense.innerOf`): the register-path GEMM and dequantize +
/// matmul; not the GEMV rows (lane order, block sums) or the fused GEMM (`prep_s`'s block sums).
pub fn preparedServes(d: *const dense.Dense, rows: usize) bool {
    const p = d.pathFor(rows);
    return p == .small or p == .matmul;
}

/// `prepare4`'s calls (the tests' proof that it ran).
pub var prep4_calls: u64 = 0;

/// Step 1's prepare with the indexer projection `ix` as a fourth member, one launch over x2 [rows, in]: xp[0..3]
/// q's, k's and v's copies (`Epilogue.prep`), xp[3] the indexer's (`linearPrepared`), each f16 [rows, in], the caller's.
/// Refuses (before any launch) what `attnIn`'s members or the indexer's `linear` path would not take as prepared rows.
pub fn prepare4(s: mlx.mlx_stream, x2: mlx.mlx_array, rows: usize, q: Member, k: Member, v: Member, ix: Member) ![4]mlx.mlx_array {
    const dt = q.d.shape.out_dtype;
    const in_dim = q.d.shape.in;
    if (dt != .bfloat16 and dt != .float16) return error.Exl3EngineNotImplemented;
    for ([_]*const dense.Dense{ k.d, v.d, ix.d }) |d| if (d.shape.in != in_dim or d.shape.out_dtype != dt) return error.BadExl3Shape;
    if (!innerServes(q.d, rows) or !innerServes(k.d, rows) or !innerServes(v.d, rows) or !preparedServes(ix.d, rows)) return error.Exl3EngineNotImplemented;
    const kern = try prep4KernelGet();
    const xs = [_]c_int{ ci(rows), ci(in_dim) };
    const pcfg = try ks.buildConfig(.{ .grid = .{ (in_dim / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{ &xs, &xs, &xs, &xs }, .out_dtypes = &.{ .float16, .float16, .float16, .float16 }, .template = &.{.{ .name = "DIM", .value = .{ .int = ci(in_dim) } }} });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(pcfg);
    var xp: [4]mlx.mlx_array = undefined;
    try ks.applyConfig(kern, pcfg, &.{ x2, q.suh, k.suh, v.suh, ix.suh }, &xp, s);
    prep4_calls +%= 1;
    return xp;
}

/// `linear`'s inner product and finish over rows already prepared (`prepare4`'s fourth copy, natural order):
/// y [rows, out] in the Dense's out dtype, by the launches `linear` makes after its prepare at these rows (`Dense.innerOf` on
/// its path, `Dense.finishOf`), so `linear`'s bytes. Refuses rows whose path reads no prepared natural-order rows.
pub fn linearPrepared(s: mlx.mlx_stream, xp: mlx.mlx_array, m: Member, rows: usize) !mlx.mlx_array {
    const path = m.d.pathFor(rows);
    if (path != .small and path != .matmul) return error.Exl3EngineNotImplemented;
    const z = try m.d.innerOf(s, xp, m.trellis, rows, path);
    defer _ = mlx.mlx_array_free(z);
    return m.d.finishOf(s, z, m.svh, rows);
}

/// ATTN-OUT: y [rows, out] = o_proj(T(attn * sigmoid(gate))) for attn [1, heads, rows, hd] (head-major) and the gate at
/// `gate`; `sigtab` T [65536] is MLX's sigmoid of every 16-bit pattern. Refuses what `prefillInner` does not serve.
pub fn linearGated(s: mlx.mlx_stream, attn: mlx.mlx_array, gate: Gate, sigtab: mlx.mlx_array, o: Member, heads: u32, hd: u32, rows: usize) !mlx.mlx_array {
    const d = o.d;
    const dt = d.shape.out_dtype;
    if (d.shape.in != heads * hd or hd % 128 != 0) return error.BadExl3Shape;
    if (!innerServes(d, rows)) return error.Exl3EngineNotImplemented;
    if (gate.off + (heads - 1) * gate.head_stride + hd > gate.row_stride or mlx.mlx_array_size(gate.base) != rows * gate.row_stride) return error.BadExl3Shape;
    const kz = try kernelsGet();
    const pcfg = try ks.buildConfig(.{
        .grid = .{ (d.shape.in / 128) * 32, @intCast(rows), 1 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{&[_]c_int{ ci(rows), ci(d.shape.in) }},
        .out_dtypes = &.{.float16},
        .template = &.{
            .{ .name = "DIM", .value = .{ .int = ci(d.shape.in) } },
            .{ .name = "HD", .value = .{ .int = ci(hd) } },
            .{ .name = "GROW", .value = .{ .int = ci(gate.row_stride) } },
            .{ .name = "GHEAD", .value = .{ .int = ci(gate.head_stride) } },
            .{ .name = "GOFF", .value = .{ .int = ci(gate.off) } },
            .{ .name = "T", .value = .{ .dtype = dt } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(pcfg);
    var xp: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(kz.prep_gate, pcfg, &.{ attn, gate.base, sigtab, o.suh }, &xp, s);
    defer _ = mlx.mlx_array_free(xp[0]);
    const z = try prefillInner(d, s, xp[0], o.trellis, rows);
    defer _ = mlx.mlx_array_free(z);
    // `Dense.finish` with `linear`'s finish config (`Dense.buildCfg(rows, false).finish`): the same pipeline
    const fcfg = try ks.buildConfig(.{ .grid = .{ (d.shape.out / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ ci(rows), ci(d.shape.out) }}, .out_dtypes = &.{dt}, .template = &.{ .{ .name = "DIM", .value = .{ .int = ci(d.shape.out) } }, .{ .name = "OUT_T", .value = .{ .dtype = dt } } } });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(fcfg);
    var y: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(d.finish, fcfg, &.{ z, o.svh }, &y, s);
    return y[0];
}

// ---- the un-rotated trunk: ATTN-IN's epilogue and ATTN-OUT's input in the standard basis (dense.zig's module comment) ----

/// One un-rotated member's rows in a product: `a` [rows, stride] in the activation dtype (row-contiguous: the member's own
/// plain matmul, or a joined product), the member at columns [off, off + its width).
pub const StdIn = struct { a: mlx.mlx_array, stride: u32, off: u32 };

/// ATTN-IN over the standard-basis projections of x [rows, in]: q|gate (each head [query | gate]), k and v as
/// `q`, `k`, `v` say (`text_epi_attn`), the outputs `attnIn`'s (q, k normed and rotated, v head-major [1, H, rows, 256], the
/// gate [1, rows, H, 256]) in `dt`, by `text_finish_attn`'s epilogue. Refuses (error.Exl3EngineNotImplemented) what it does
/// not serve before any launch: another dtype, a width the epilogue does not tile, a member outside its array.
pub fn attnInStd(s: mlx.mlx_stream, rows: usize, q: StdIn, k: StdIn, v: StdIn, ep: Epilogue, dt: mlx.mlx_dtype) !InOut {
    if (dt != .bfloat16 and dt != .float16) return error.Exl3EngineNotImplemented;
    if (ep.heads == 0 or ep.kv_heads == 0 or !ropeDimsServed(ep.rope_dims)) return error.Exl3EngineNotImplemented;
    if (rows == 0 or rows <= decode.Kernels.max_rows) return error.Exl3EngineNotImplemented;
    const widths = [3]u32{ 2 * ep.heads * head_dim, ep.kv_heads * head_dim, ep.kv_heads * head_dim };
    for ([_]StdIn{ q, k, v }, widths) |m, w| {
        const sh = mlx.getShape(m.a);
        if (mlx.mlx_array_dtype(m.a) != dt or sh.len != 2 or sh[0] != ci(rows) or sh[1] != ci(m.stride) or m.off + w > m.stride or m.off % 128 != 0) return error.Exl3EngineNotImplemented;
    }
    if (mlx.mlx_array_dtype(ep.q_norm) != dt or mlx.mlx_array_dtype(ep.k_norm) != dt or mlx.mlx_array_size(ep.q_norm) != head_dim or mlx.mlx_array_size(ep.k_norm) != head_dim) return error.Exl3EngineNotImplemented;
    if (mlx.mlx_array_dtype(ep.angles) != .float32 or mlx.mlx_array_size(ep.angles) != rows * ep.rope_dims) return error.Exl3EngineNotImplemented;
    if (mlx.mlx_array_dtype(ep.eps) != .float32 or mlx.mlx_array_ndim(ep.eps) != 0) return error.Exl3EngineNotImplemented;
    const kz = try stdKernelsGet();
    const hq = ci(ep.heads);
    const hk = ci(ep.kv_heads);
    const r = ci(rows);
    const hd = ci(head_dim);
    const fcfg = try ks.buildConfig(.{
        .grid = .{ finishUnits(ep.heads, ep.kv_heads) * unit_threads, @intCast(rows), 1 },
        .threadgroup = .{ unit_threads, 1, 1 },
        .out_shapes = &.{ &[_]c_int{ 1, hq, r, hd }, &[_]c_int{ 1, r, hq, hd }, &[_]c_int{ 1, hk, r, hd }, &[_]c_int{ 1, hk, r, hd } },
        .out_dtypes = &.{ dt, dt, dt, dt },
        .template = &.{
            .{ .name = "HQ", .value = .{ .int = hq } },
            .{ .name = "HK", .value = .{ .int = hk } },
            .{ .name = "HD", .value = .{ .int = hd } },
            .{ .name = "RD", .value = .{ .int = ci(ep.rope_dims) } },
            .{ .name = "QS", .value = .{ .int = ci(q.stride) } },
            .{ .name = "QO", .value = .{ .int = ci(q.off) } },
            .{ .name = "KS", .value = .{ .int = ci(k.stride) } },
            .{ .name = "KO", .value = .{ .int = ci(k.off) } },
            .{ .name = "VS", .value = .{ .int = ci(v.stride) } },
            .{ .name = "VO", .value = .{ .int = ci(v.off) } },
            .{ .name = "OUT_T", .value = .{ .dtype = dt } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(fcfg);
    var fo: [4]mlx.mlx_array = undefined;
    try ks.applyConfig(kz[0], fcfg, &.{ q.a, k.a, v.a, ep.q_norm, ep.k_norm, ep.angles, ep.eps }, &fo, s);
    return .{ .q = fo[0], .gate = fo[1], .k = fo[2], .v = fo[3] };
}

/// ATTN-OUT in the standard basis: y [rows, out] = T(attn * sigmoid(gate)) W_std for attn [1, heads, rows, hd]
/// (head-major), the gate at `gate` and `sigtab` (`linearGated`'s operands) and o_proj's W_std [heads hd, out] (the
/// resident table's): the gated product written once in the standard basis (`text_gate_plain`), then one MLX matmul in T.
/// Refuses what it does not serve before any launch (error.Exl3EngineNotImplemented: another dtype, decode rows).
pub fn linearGatedStd(s: mlx.mlx_stream, attn: mlx.mlx_array, gate: Gate, sigtab: mlx.mlx_array, w_std: mlx.mlx_array, heads: u32, hd: u32, rows: usize) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(attn);
    const dim = heads * hd;
    const wsh = mlx.getShape(w_std);
    if (dt != .bfloat16 and dt != .float16) return error.Exl3EngineNotImplemented;
    if (rows <= decode.Kernels.max_rows or mlx.mlx_array_dtype(w_std) != dt or mlx.mlx_array_dtype(gate.base) != dt or mlx.mlx_array_dtype(sigtab) != dt) return error.Exl3EngineNotImplemented;
    if (wsh.len != 2 or wsh[0] != ci(dim) or hd % 128 != 0) return error.BadExl3Shape;
    if (gate.off + (heads - 1) * gate.head_stride + hd > gate.row_stride or mlx.mlx_array_size(gate.base) != rows * gate.row_stride) return error.BadExl3Shape;
    const kz = try stdKernelsGet();
    const pcfg = try ks.buildConfig(.{
        .grid = .{ (dim / 128) * 32, @intCast(rows), 1 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = &.{&[_]c_int{ ci(rows), ci(dim) }},
        .out_dtypes = &.{dt},
        .template = &.{
            .{ .name = "DIM", .value = .{ .int = ci(dim) } },
            .{ .name = "HD", .value = .{ .int = ci(hd) } },
            .{ .name = "GROW", .value = .{ .int = ci(gate.row_stride) } },
            .{ .name = "GHEAD", .value = .{ .int = ci(gate.head_stride) } },
            .{ .name = "GOFF", .value = .{ .int = ci(gate.off) } },
            .{ .name = "T", .value = .{ .dtype = dt } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(pcfg);
    var xg: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(kz[1], pcfg, &.{ attn, gate.base, sigtab }, &xg, s);
    defer _ = mlx.mlx_array_free(xg[0]);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_matmul(&y, xg[0], w_std, s));
    return y;
}

// ---- CPU tests (no device) ----

const testing = std.testing;

/// The statements of `text`'s lines that contain `marker`, in order (trimmed), for comparing two texts' arithmetic.
fn linesWith(a: std.mem.Allocator, text: []const u8, marker: []const u8) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(a);
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (std.mem.indexOf(u8, line, marker) != null) try out.append(a, std.mem.trim(u8, line, " "));
    }
    return out.toOwnedSlice(a);
}

test "turboexl3 attn_io: the prepares and the finish repeat dense's prepare / finish statements (the Hadamard, the roundings)" {
    const a = testing.allocator;
    // the butterfly and the local 4-point stage, verbatim, in every text
    const fly = try linesWith(a, dense.text_prep, "simd_shuffle_xor");
    defer a.free(fly);
    const local4 = try linesWith(a, dense.text_prep, "float t0 = v[0] + v[1]");
    defer a.free(local4);
    try testing.expectEqual(@as(usize, 1), fly.len);
    try testing.expectEqual(@as(usize, 1), local4.len);
    for ([_][]const u8{ text_prep3, text_finish_attn, text_prep_gate }) |t| {
        try testing.expect(std.mem.indexOf(u8, t, fly[0]) != null);
        try testing.expect(std.mem.indexOf(u8, t, local4[0]) != null);
    }
    // the prepare's f16 input rounding and suh product, and its output rounding (LANE_ORDER 0: dst = i)
    try testing.expect(std.mem.indexOf(u8, dense.text_prep, "v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]);") != null);
    try testing.expect(std.mem.indexOf(u8, text_prep3, "xh[j] = float(half(float(x[row * DIMC + i])));") != null);
    try testing.expect(std.mem.indexOf(u8, text_prep3, "v[j] = xh[j] * float(suh[i]);") != null);
    try testing.expect(std.mem.indexOf(u8, text_prep_gate, "v[j] = float(half(float(xv))) * float(suh[i]);") != null);
    try testing.expect(std.mem.indexOf(u8, dense.text_prep, "half(v[j] * HAD_SCALE);") != null);
    try testing.expect(std.mem.indexOf(u8, text_prep3, "xp[row * DIMC + i] = half(v[j] * HAD_SCALE);") != null);
    try testing.expect(std.mem.indexOf(u8, text_prep_gate, "xp[row * DIMC + i] = half(v[j] * HAD_SCALE);") != null);
    // the finish's input and output roundings
    try testing.expect(std.mem.indexOf(u8, dense.text_finish, "v[j] = float(half(float(z[row * DIMC + b * 128u + L + 32u * j])));") != null);
    try testing.expect(std.mem.indexOf(u8, text_finish_attn, "v[j] = float(half(float(z[row * dimc + blk * 128u + L + 32u * j])));") != null);
    try testing.expect(std.mem.indexOf(u8, dense.text_finish, "OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))))") != null);
    try testing.expect(std.mem.indexOf(u8, text_finish_attn, "y[j] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))));") != null);
}

test "turboexl3 attn_io: the epilogue is the host's hd-256 norm + rope statements (MLX's rms_single_row and rope_impl roundings)" {
    // the lines of the host's QK_NORM_ROPE_256_SOURCE (transformer.zig) that carry arithmetic, with its names mapped to ours
    // (src[base + i] -> ys[base + i], T -> OUT_T, HD -> HDC, simd_lane / simd_group -> L / sg)
    const want = [_][]const u8{
        "acc += xv * xv;",
        "acc = simd_sum(acc);",
        "local_sums[L] = 0.0f;",
        "local_sums[sg] = acc;",
        "acc = simd_sum(local_sums[L]);",
        "local_inv[0] = metal::precise::rsqrt(acc / float(HDC) + float(eps));",
        "nrm[base + i] = OUT_T(w[base + i] * OUT_T(float(ys[base + i]) * inv));",
        "r = OUT_T(float(nrm[j]) * arow[j] - float(nrm[j + half_rd]) * arow[half_rd + j]);",
        "r = OUT_T(float(nrm[j - half_rd]) * arow[j] + float(nrm[j]) * arow[j - half_rd]);",
        "const uint base = tid * 4u;",
    };
    for (want) |w| try testing.expect(std.mem.indexOf(u8, text_finish_attn, w) != null);
    // the gate's product in T (MLX's Multiply), the sigmoid by table (MLX's Sigmoid of the 16-bit pattern)
    try testing.expect(std.mem.indexOf(u8, text_prep_gate, "const T xv = av * sigtab[as_type<ushort>(gv)];") != null);
}

/// The finish's unit -> (member, head, blocks) rule, as the kernel decodes it.
const Unit = struct { member: enum { q, gate, k, v }, head: u32, blk0: u32 };
fn unitOf(u: u32, hq: u32, hk: u32) Unit {
    if (u < hq) return .{ .member = .q, .head = u, .blk0 = 4 * u };
    if (u < 2 * hq) return .{ .member = .gate, .head = u - hq, .blk0 = 4 * (u - hq) + 2 };
    if (u < 2 * hq + hk) return .{ .member = .k, .head = u - 2 * hq, .blk0 = 2 * (u - 2 * hq) };
    return .{ .member = .v, .head = u - 2 * hq - hk, .blk0 = 2 * (u - 2 * hq - hk) };
}

test "turboexl3 attn_io: the finish's units cover every 128-block of q_proj, k_proj and v_proj once (Qwen3.8 and other head counts)" {
    for ([_][2]u32{ .{ 24, 2 }, .{ 16, 4 }, .{ 32, 8 }, .{ 1, 1 } }) |g| {
        const hq = g[0];
        const hk = g[1];
        var q_seen: [256]u8 = @splat(0); // zig 0.17 removed the `**` array repeat
        var k_seen: [64]u8 = @splat(0); // zig 0.17 removed the `**` array repeat
        var v_seen: [64]u8 = @splat(0); // zig 0.17 removed the `**` array repeat
        const n = finishUnits(hq, hk);
        for (0..n) |ui| {
            const un = unitOf(@intCast(ui), hq, hk);
            for (0..2) |sg| {
                const blk = un.blk0 + @as(u32, @intCast(sg));
                switch (un.member) {
                    .q, .gate => q_seen[blk] += 1,
                    .k => k_seen[blk] += 1,
                    .v => v_seen[blk] += 1,
                }
                // a query unit's blocks are the head's first 256 columns, a gate unit's the last 256
                if (un.member == .q) try testing.expect(blk / 4 == un.head and blk % 4 < 2);
                if (un.member == .gate) try testing.expect(blk / 4 == un.head and blk % 4 >= 2);
            }
        }
        for (q_seen[0 .. 4 * hq]) |c| try testing.expectEqual(@as(u8, 1), c);
        for (k_seen[0 .. 2 * hk]) |c| try testing.expectEqual(@as(u8, 1), c);
        for (v_seen[0 .. 2 * hk]) |c| try testing.expectEqual(@as(u8, 1), c);
    }
    try testing.expectEqual(@as(u32, 52), finishUnits(24, 2));
    try testing.expect(ropeDimsServed(64) and ropeDimsServed(256) and !ropeDimsServed(48) and !ropeDimsServed(0));
}

test "turboexl3 attn_io: the gated prepare's gate layouts address the q_proj gate half and ATTN-IN's gate" {
    // q_proj [rows, 2 H HD]: head h's gate is columns [2 h HD + HD, 2 h HD + 2 HD)
    const H: u32 = 24;
    const HD: u32 = 256;
    const qp: Gate = .{ .base = .{ .ctx = null }, .row_stride = 2 * H * HD, .head_stride = 2 * HD, .off = HD };
    const ai: Gate = .{ .base = .{ .ctx = null }, .row_stride = H * HD, .head_stride = HD, .off = 0 };
    for ([_]u32{ 0, 7, 23 }) |h| for ([_]u32{ 0, 1, 255 }) |d| {
        const row: u32 = 5;
        try testing.expectEqual(row * 12288 + h * 512 + 256 + d, row * qp.row_stride + h * qp.head_stride + qp.off + d);
        try testing.expectEqual(row * 6144 + h * 256 + d, row * ai.row_stride + h * ai.head_stride + ai.off + d);
    };
    try testing.expect(qp.off + (H - 1) * qp.head_stride + HD <= qp.row_stride);
    try testing.expect(ai.off + (H - 1) * ai.head_stride + HD <= ai.row_stride);
}

// ---- the four-member prepare (its GPU byte tests are attn_io_test.zig's) ----

test "turboexl3 attn_io: text_prep4 is text_prep3 with a fourth member (every arithmetic line verbatim, so dense's prepare at LANE_ORDER 0), over the names its signature binds" {
    // every line of text_prep3 but its member loop's bound and its three-way member selection
    var kept: usize = 0;
    var it = std.mem.splitScalar(u8, text_prep3, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " ");
        if (line.len == 0 or std.mem.indexOf(u8, line, "m < 3u") != null or std.mem.indexOf(u8, line, "suh0 : ") != null or std.mem.indexOf(u8, line, "xp0 : ") != null) continue;
        try testing.expect(std.mem.indexOf(u8, text_prep4, line) != null);
        kept += 1;
    }
    try testing.expect(kept >= 12);
    // the four-way loop and selection
    try testing.expect(std.mem.indexOf(u8, text_prep4, "for (uint m = 0; m < 4u; ++m) {") != null);
    try testing.expect(std.mem.indexOf(u8, text_prep4, "const device half* suh = (m == 0u) ? suh0 : ((m == 1u) ? suh1 : ((m == 2u) ? suh2 : suh3));") != null);
    try testing.expect(std.mem.indexOf(u8, text_prep4, "device half* xp = (m == 0u) ? xp0 : ((m == 1u) ? xp1 : ((m == 2u) ? xp2 : xp3));") != null);
    // the f16 input rounding, the suh product and the output rounding (LANE_ORDER 0: dst = i), as the prepares' test pins them
    try testing.expect(std.mem.indexOf(u8, text_prep4, "xh[j] = float(half(float(x[row * DIMC + i])));") != null);
    try testing.expect(std.mem.indexOf(u8, text_prep4, "v[j] = xh[j] * float(suh[i]);") != null);
    try testing.expect(std.mem.indexOf(u8, text_prep4, "xp[row * DIMC + i] = half(v[j] * HAD_SCALE);") != null);
    // every input and output named, none followed by a suffix MLX would bind a shape / strides / ndim buffer for
    for (prep4_in ++ prep4_out) |nm| {
        const name = std.mem.span(nm);
        try testing.expect(std.mem.indexOf(u8, text_prep4, name) != null);
        var buf: [32]u8 = undefined;
        for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| try testing.expect(std.mem.indexOf(u8, text_prep4, std.fmt.bufPrint(&buf, "{s}{s}", .{ name, suf }) catch unreachable) == null);
    }
}

// ---- the un-rotated trunk (its GPU checks are gpu_test.zig's) ----

test "turboexl3 attn_io un-rotated trunk texts: the standard-basis epilogue is ATTN-IN's finish epilogue verbatim behind a plain read, and the gated input is ATTN-OUT's product without the prepare" {
    const has = struct {
        fn f(hay: []const u8, needle: []const u8) bool {
            return std.mem.indexOf(u8, hay, needle) != null;
        }
    }.f;
    // ATTN-IN's own text is its two pieces, and the standard-basis epilogue ends with the same tail
    try testing.expect(std.mem.startsWith(u8, text_finish_attn, finish_attn_head) and std.mem.endsWith(u8, text_finish_attn, finish_attn_tail));
    try testing.expectEqual(finish_attn_head.len + finish_attn_tail.len, text_finish_attn.len);
    try testing.expect(std.mem.endsWith(u8, text_epi_attn, finish_attn_tail));
    // the epilogue's declarations are the finish's (units, heads, blocks) but its y[] is the product's value as it is: no
    // butterfly, no svh, no f16 round
    var it = std.mem.splitScalar(u8, finish_attn_head, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " ");
        if (line.len == 0) continue;
        const finish_only = has(line, "QDIM") or has(line, "KDIM") or has(line, "HAD_SCALE") or has(line, "svh") or has(line, "dimc") or
            has(line, "float v[4];") or has(line, "simd_shuffle_xor") or has(line, "float t0") or has(line, "const device half* z");
        if (!finish_only) try testing.expect(has(epi_attn_head, line));
    }
    try testing.expect(!has(text_epi_attn, "simd_shuffle_xor") and !has(text_epi_attn, "svh") and !has(text_epi_attn, "half("));
    try testing.expect(has(text_epi_attn, "y[j] = z[ulong(row) * zs + zo + blk * 128u + L + 32u * j];"));
    // the gated input: `text_prep_gate`'s product lines, written as T, no rotation
    for ([_][]const u8{
        "const uint hh = i / HDC;",
        "const uint d = i - hh * HDC;",
        "const T av = attn[(ulong(hh) * S + row) * HDC + d];",
        "const T gv = gate[ulong(row) * uint(GROW) + hh * uint(GHEAD) + uint(GOFF) + d];",
        "const T xv = av * sigtab[as_type<ushort>(gv)];",
    }) |l| try testing.expect(has(text_prep_gate, l) and has(text_gate_plain, l));
    try testing.expect(has(text_gate_plain, "xg[ulong(row) * DIMC + i] = xv;") and !has(text_gate_plain, "suh") and !has(text_gate_plain, "simd_shuffle_xor"));
    // every input and output named, none followed by a suffix MLX would bind a shape / strides / ndim buffer for
    const Sig = struct { txt: []const u8, names: []const [*:0]const u8 };
    for ([_]Sig{ .{ .txt = text_epi_attn, .names = &(epi_in ++ fin_out) }, .{ .txt = text_gate_plain, .names = &(gate_plain_in ++ gate_plain_out) } }) |sg| for (sg.names) |nm| {
        const name = std.mem.span(nm);
        try testing.expect(has(sg.txt, name));
        var buf: [32]u8 = undefined;
        for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| try testing.expect(!has(sg.txt, std.fmt.bufPrint(&buf, "{s}{s}", .{ name, suf }) catch unreachable));
    };
}
