//! The prefill chain (rows > DECODE_ROWS_MAX), M3: the routed slots sorted by expert, each expert's rows in 128-row M tiles,
//! and the EXL3 trellis decoded inside the GEMMs' weight loader (DIG-X's shape: one 64 x 64 weight stage decoded once into
//! threadgroup memory feeds 128 rows on the NAX matrix units, f32 accumulation). Seven launches per MoE layer:
//!
//!   argsort   MLX's argsort of the flat expert ids: sorted position p -> slot order[p] = row * K + k
//!   table     one threadgroup: each expert's run [lo, hi) of sorted positions (a binary search per expert), its M tiles,
//!             an exclusive scan over the experts; tiles[t] = (expert, first position, rows); unused tiles rows = 0
//!             (`Options.froute`: one launch instead of the two: the table kernel counts each expert's slots itself,
//!             threadgroup atomics, scans the counts into each run's start and scatters the slots there)
//!   prep      per (sorted position, 128-block of hidden): the gate and up inputs (f16 round, suh, H128, f16 round) as
//!             half [S, hidden] in sorted order, and inv[order[p]] = p
//!   gemm      gate|up: per (M tile, projection, 64-wide output tile), the K loop of 64-deep stages: decode the stage's 16
//!             trellis tiles into B^T (threadgroup memory), barrier, NAX MMA; the inner product rounded to f16
//!   mid       per (sorted position, 128-block of inter): finish gate and up (H128, svh, f16), SwiGLU in f32, the down
//!             input (f16 round, suh, H128, f16 round)
//!   gemm      down: the same kernel over the inter dim (A/B options: 128-deep stages, several output blocks per threadgroup)
//!   finish    per (row, 128-block of hidden): each slot's down inner through inv, H128, svh, f16, times its score,
//!             summed in slot order
//!
//! Rounding points are `format.project`'s; the GEMM sums in the NAX's order, so parity with the reference is a tolerance.
//! A row's bytes do not depend on where its slots land in the sorted order or the M tiles, so `froute`'s atomic scatter
//! (slots of one expert in arrival order, not slot order) changes no output; no atomics anywhere else.
//!
//! `Options.alay` stores the GEMMs' A operands (xg, xu, xd) in the A fragments' slot order (`alayPos`: per 32-element
//! block, a fragment row of a 32-deep step is one 16-byte load; `alayPos16`: per 16-element block, one 8-byte load per
//! 16-deep slice); `Options.half_rows` lets a simdgroup whose rows end within its first 16 skip its second fragment row.
//! Both move loads and MMAs, not arithmetic: identical outputs.
//!
//! `Options.t128` runs `moeT128`: the route also cuts each expert's rows into 128-row tiles (its full 128-row runs, and a
//! remainder of more than `t128_cut` rows), which the bm-128 shared-decode body multiplies in a launch of their own beside
//! the mixed launch over the rest; each launch writes its own output, and mid and finish read each row from the launch that
//! computed it. bm 128 and bm 64 run the same per-simdgroup K loop, so every output sums the same 16-deep MMAs in the same
//! order: `moe`'s bytes.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const format = @import("format.zig");
const decode = @import("decode.zig");
const lane = @import("lane.zig");

/// MLX steel's NAX tiles (NAXTile, tile_matmad_nax over mpp::tensor_ops::matmul2d; steel_nax.metal: MLX 0.32.2's headers,
/// each section marked with its source path and copyright line), then turboexl3's decode helpers.
const steel_nax = @embedFile("steel_nax.metal");
pub const header_nax: [:0]const u8 = sentinel(steel_nax ++ "using namespace mlx::steel;\n" ++ decode.header ++ metal_small_helpers ++ metal_gemm_helpers);
/// The register path's header: the NAX tiles, the decoders and its fragment fills.
pub const header_nax_small: [:0]const u8 = sentinel(steel_nax ++ "using namespace mlx::steel;\n" ++ decode.header ++ metal_small_helpers);
/// The mixed GEMM's header: both bodies' helpers.
pub const header_nax_mixed: [:0]const u8 = header_nax;

/// M tile rows, output tile columns, K stage depth of the GEMM (DIG-X's m128 text).
pub const BM: u32 = 128;
pub const BN: u32 = 64;
pub const BK: u32 = 64;

/// The tile table: one threadgroup of 1024 threads, thread e owns expert e (EXP <= 1024). tiles int32 [TMAX, 4] =
/// (expert, first sorted position, rows, 0) for the tiles in expert order, then rows = 0 to the end.
pub const text_table =
    \\    constexpr uint EC = uint(EXP);
    \\    constexpr uint BMC = uint(BMT);
    \\    const uint SC = uint(inds_shape[0]);
    \\    const uint TMAXC = (SC + BMC - 1u) / BMC + EC;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    threadgroup uint lo_s[1025];
    \\    threadgroup uint pref[32];
    \\    threadgroup uint total_s;
    \\    uint lo = SC;
    \\    if (tid < EC) {
    \\        uint a = 0u, b = SC;
    \\        while (a < b) { const uint mid = (a + b) >> 1u; if (uint(inds[order[mid]]) < tid) a = mid + 1u; else b = mid; }
    \\        lo = a;
    \\    }
    \\    lo_s[tid] = lo;
    \\    if (tid == 0u) lo_s[1024] = SC;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    uint cnt = 0u, nt = 0u;
    \\    if (tid < EC) { cnt = lo_s[tid + 1u] - lo; nt = (cnt + BMC - 1u) / BMC; }
    \\    const uint incl = simd_prefix_inclusive_sum(nt);
    \\    if (L == 31u) pref[sg] = incl;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (sg == 0u) pref[L] = simd_prefix_exclusive_sum(pref[L]);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const uint start = pref[sg] + incl - nt;
    \\    for (uint t = 0u; t < nt; ++t) {
    \\        const uint i = start + t;
    \\        tiles[4u * i] = int(tid); tiles[4u * i + 1u] = int(lo + t * BMC); tiles[4u * i + 2u] = int(min(BMC, cnt - t * BMC)); tiles[4u * i + 3u] = 0;
    \\    }
    \\    if (tid == 1023u) total_s = start + nt;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint i = total_s + tid; i < TMAXC; i += 1024u) { tiles[4u * i] = 0; tiles[4u * i + 1u] = 0; tiles[4u * i + 2u] = 0; tiles[4u * i + 3u] = 0; }
    \\
;

/// The mixed tile table (`Options.small = .mixed`): each expert's rows in BMT-row tiles, except a remainder of <= 32 * SPL rows
/// (or an expert of that few rows), which becomes SPL or fewer register-path tiles of <= 32 rows. tiles int32
/// [TBIG + SPL * EXP, 4]: the BMT-row tiles in expert order from 0, the small tiles in expert order from
/// TBIG = ceil(S / BMT) + EXP; unused entries rows = 0.
pub const text_table_mixed =
    \\    constexpr uint EC = uint(EXP);
    \\    constexpr uint BMC = uint(BMT);
    \\    constexpr uint SR = 32u * uint(SPL);
    \\    const uint SC = uint(inds_shape[0]);
    \\    const uint TBIG = (SC + BMC - 1u) / BMC + EC;
    \\    const uint TALL = TBIG + uint(SPL) * EC;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    threadgroup uint lo_s[1025];
    \\    threadgroup uint prefb[32];
    \\    threadgroup uint prefs[32];
    \\    threadgroup uint totb_s;
    \\    threadgroup uint tots_s;
    \\    uint lo = SC;
    \\    if (tid < EC) {
    \\        uint a = 0u, b = SC;
    \\        while (a < b) { const uint mid = (a + b) >> 1u; if (uint(inds[order[mid]]) < tid) a = mid + 1u; else b = mid; }
    \\        lo = a;
    \\    }
    \\    lo_s[tid] = lo;
    \\    if (tid == 0u) lo_s[1024] = SC;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    uint cnt = 0u, nb = 0u, ns = 0u, nfull = 0u, r = 0u;
    \\    if (tid < EC) {
    \\        cnt = lo_s[tid + 1u] - lo;
    \\        nfull = cnt / BMC;
    \\        r = cnt - nfull * BMC;
    \\        if (r == 0u) nb = nfull;
    \\        else if (r > SR) nb = nfull + 1u;
    \\        else { nb = nfull; ns = (r + 31u) / 32u; }
    \\    }
    \\    const uint ib = simd_prefix_inclusive_sum(nb);
    \\    const uint is = simd_prefix_inclusive_sum(ns);
    \\    if (L == 31u) { prefb[sg] = ib; prefs[sg] = is; }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (sg == 0u) { prefb[L] = simd_prefix_exclusive_sum(prefb[L]); prefs[L] = simd_prefix_exclusive_sum(prefs[L]); }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const uint sb = prefb[sg] + ib - nb;
    \\    const uint ss = prefs[sg] + is - ns;
    \\    for (uint t = 0u; t < nb; ++t) {
    \\        const uint i = sb + t;
    \\        tiles[4u * i] = int(tid); tiles[4u * i + 1u] = int(lo + t * BMC); tiles[4u * i + 2u] = int(min(BMC, cnt - t * BMC)); tiles[4u * i + 3u] = 0;
    \\    }
    \\    for (uint t = 0u; t < ns; ++t) {
    \\        const uint i = TBIG + ss + t;
    \\        tiles[4u * i] = int(tid); tiles[4u * i + 1u] = int(lo + nfull * BMC + 32u * t); tiles[4u * i + 2u] = int(min(32u, r - 32u * t)); tiles[4u * i + 3u] = 0;
    \\    }
    \\    if (tid == 1023u) { totb_s = sb + nb; tots_s = ss + ns; }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint i = totb_s + tid; i < TBIG; i += 1024u) { tiles[4u * i] = 0; tiles[4u * i + 1u] = 0; tiles[4u * i + 2u] = 0; tiles[4u * i + 3u] = 0; }
    \\    for (uint i = TBIG + tots_s + tid; i < TALL; i += 1024u) { tiles[4u * i] = 0; tiles[4u * i + 1u] = 0; tiles[4u * i + 2u] = 0; tiles[4u * i + 3u] = 0; }
    \\
;

/// The tables' per-expert binary search over argsort's order (both tables carry it verbatim).
const table_bsearch =
    \\    uint lo = SC;
    \\    if (tid < EC) {
    \\        uint a = 0u, b = SC;
    \\        while (a < b) { const uint mid = (a + b) >> 1u; if (uint(inds[order[mid]]) < tid) a = mid + 1u; else b = mid; }
    \\        lo = a;
    \\    }
    \\
;

/// `Options.froute`'s replacement for it: count each expert's slots (threadgroup atomics), an exclusive scan of the counts is
/// each expert's first sorted position, then every slot takes the next position of its expert's run: `order` is written
/// here (an output), so the launch replaces argsort + table. Within an expert the slots land in arrival order.
const froute_count =
    \\    threadgroup atomic_uint fr_cnt[1024];
    \\    threadgroup uint fr_pref[32];
    \\    atomic_store_explicit(&fr_cnt[tid], 0u, memory_order_relaxed);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint i = tid; i < SC; i += 1024u) atomic_fetch_add_explicit(&fr_cnt[uint(inds[i])], 1u, memory_order_relaxed);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const uint fr_n = atomic_load_explicit(&fr_cnt[tid], memory_order_relaxed);
    \\    const uint fr_incl = simd_prefix_inclusive_sum(fr_n);
    \\    if (L == 31u) fr_pref[sg] = fr_incl;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (sg == 0u) fr_pref[L] = simd_prefix_exclusive_sum(fr_pref[L]);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    uint lo = fr_pref[sg] + fr_incl - fr_n;
    \\    atomic_store_explicit(&fr_cnt[tid], lo, memory_order_relaxed);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint i = tid; i < SC; i += 1024u) order[atomic_fetch_add_explicit(&fr_cnt[uint(inds[i])], 1u, memory_order_relaxed)] = i;
    \\
;

/// The fused route kernels (`Options.froute`): the tables with `froute_count` for the binary search; inputs inds, outputs
/// order u32 [S] and tiles.
pub const text_froute: [:0]const u8 = sentinel(replaceOnce(text_table, table_bsearch, froute_count));
pub const text_froute_m: [:0]const u8 = sentinel(replaceOnce(text_table_mixed, table_bsearch, froute_count));

/// The gate and up inputs in sorted order: xg, xu half [S, HID] = H128(f16(x[row]) * suh_e) * 2^-3.5 rounded, per 128-block;
/// one simdgroup per (block, sorted position), BPT blocks per threadgroup; inv u32 [S]: inv[order[p]] = p. ALAY
/// (`Options.alay`): each 32-block's element j at slot `turboexl3_alay(j)` (1) or `turboexl3_alay16(j)` (2) of its block; the
/// block sums are the same.
pub const text_prep =
    \\    constexpr uint HC = uint(HID);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint BPTC = uint(BPT);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x * BPTC + simdgroup_index_in_threadgroup;
    \\    const uint p = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint src = order[p];
    \\    const uint r = src / TOPKC;
    \\    const uint e = uint(inds[src]);
    \\    if (b == 0u && L == 0u) inv[src] = p;
    \\    float vg[4], vu[4];
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint i = b * 128u + L + 32u * j;
    \\        const float xv = float(half(float(x[ulong(r) * HC + i])));
    \\        vg[j] = xv * float(suh_g[ulong(e) * HC + i]);
    \\        vu[j] = xv * float(suh_u[ulong(e) * HC + i]);
    \\    }
    \\    turboexl3_h128x4(vg, L);
    \\    turboexl3_h128x4(vu, L);
    \\    float sg = 0.0f, su = 0.0f;
    \\    const uint lo = (ALAY == 1) ? turboexl3_alay(L) : ((ALAY == 2) ? turboexl3_alay16(L) : L);
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint i = b * 128u + 32u * j + lo;
    \\        const half hg = half(vg[j] * HAD_SCALE);
    \\        const half hu = half(vu[j] * HAD_SCALE);
    \\        xg[ulong(p) * HC + i] = hg;
    \\        xu[ulong(p) * HC + i] = hu;
    \\        sg += float(hg);
    \\        su += float(hu);
    \\    }
    \\    sg = simd_sum(sg);
    \\    su = simd_sum(su);
    \\    if (L == 0u) { xsg[ulong(p) * (HC / 128u) + b] = sg; xsu[ulong(p) * (HC / 128u) + b] = su; }
    \\
;

/// gate|up finish, SwiGLU and the down input, per (128-block of inter, sorted position): xd half [S, INTER] (ALAY: in
/// `text_prep`'s A layout).
pub const text_mid =
    \\    constexpr uint IC = uint(INTER);
    \\    constexpr uint BPTC = uint(BPT);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x * BPTC + simdgroup_index_in_threadgroup;
    \\    const uint p = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint e = uint(inds[order[p]]);
    \\    float g[4], u[4];
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint i = b * 128u + L + 32u * j;
    \\        g[j] = float(zg[ulong(p) * IC + i]);
    \\        u[j] = float(zu[ulong(p) * IC + i]);
    \\    }
    \\    turboexl3_h128x4(g, L);
    \\    turboexl3_h128x4(u, L);
    \\    float d[4];
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint i = b * 128u + L + 32u * j;
    \\        const float gg = float(half(g[j] * HAD_SCALE * float(svh_g[ulong(e) * IC + i])));
    \\        const float uu = float(half(u[j] * HAD_SCALE * float(svh_u[ulong(e) * IC + i])));
    \\        const float h = (gg / (1.0f + metal::precise::exp(-gg))) * uu;
    \\        d[j] = float(half(h)) * float(suh_d[ulong(e) * IC + i]);
    \\    }
    \\    turboexl3_h128x4(d, L);
    \\    float sd = 0.0f;
    \\    const uint lo = (ALAY == 1) ? turboexl3_alay(L) : ((ALAY == 2) ? turboexl3_alay16(L) : L);
    \\    for (uint j = 0; j < 4u; ++j) { const half hd = half(d[j] * HAD_SCALE); xd[ulong(p) * IC + b * 128u + 32u * j + lo] = hd; sd += float(hd); }
    \\    sd = simd_sum(sd);
    \\    if (L == 0u) xsd[ulong(p) * (IC / 128u) + b] = sd;
    \\
;

/// The down finish and the slot reduce, per (128-block of hidden, row): y OUT_T [R, HID] = sum over the row's slots in slot
/// order of score * f16(H128(zd[inv[slot]]) * 2^-3.5 * svh_e).
pub const text_finish =
    \\    constexpr uint HC = uint(HID);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint BPTC = uint(BPT);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x * BPTC + simdgroup_index_in_threadgroup;
    \\    const uint r = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    \\    for (uint k = 0; k < TOPKC; ++k) {
    \\        const uint src = r * TOPKC + k;
    \\        const uint p = inv[src];
    \\        const uint e = uint(inds[src]);
    \\        const float w = float(scores[src]);
    \\        float v[4];
    \\        for (uint j = 0; j < 4u; ++j) v[j] = float(zd[ulong(p) * HC + b * 128u + L + 32u * j]);
    \\        turboexl3_h128x4(v, L);
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint i = b * 128u + L + 32u * j;
    \\            acc[j] += w * float(half(v[j] * HAD_SCALE * float(svh_d[ulong(e) * HC + i])));
    \\        }
    \\    }
    \\    for (uint j = 0; j < 4u; ++j) y[ulong(r) * HC + b * 128u + L + 32u * j] = OUT_T(acc[j]);
    \\
;

/// `Options.aux`: prep, mid and finish with each lane holding 4 consecutive elements of its 128-block (8-byte loads and
/// stores; `turboexl3_h128v4`, `turboexl3_sum_x4`: the same values, so identical bytes) and ASG simdgroups per threadgroup over
/// the flattened (position or row, block) items instead of one small threadgroup per position.
pub const text_prep_v =
    \\    constexpr uint HC = uint(HID);
    \\    constexpr uint NBC = HC / 128u;
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint w = threadgroup_position_in_grid.x * uint(ASG) + simdgroup_index_in_threadgroup;
    \\    if (w >= uint(order_shape[0]) * NBC) return;
    \\    const uint p = w / NBC;
    \\    const uint b = w - p * NBC;
    \\    const uint src = order[p];
    \\    const uint r = src / TOPKC;
    \\    const uint e = uint(inds[src]);
    \\    if (b == 0u && L == 0u) inv[src] = p;
    \\    const uint i0 = b * 128u + 4u * L;
    \\    typedef metal::remove_cv_t<metal::remove_addrspace_t<metal::remove_reference_t<decltype(x[0])>>> XT;
    \\    const vec<XT, 4> x4 = *reinterpret_cast<const device vec<XT, 4>*>(x + ulong(r) * HC + i0);
    \\    const half4 sg4 = *reinterpret_cast<const device half4*>(suh_g + ulong(e) * HC + i0);
    \\    const half4 su4 = *reinterpret_cast<const device half4*>(suh_u + ulong(e) * HC + i0);
    \\    float vg[4], vu[4];
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const float xv = float(half(float(x4[j])));
    \\        vg[j] = xv * float(sg4[j]);
    \\        vu[j] = xv * float(su4[j]);
    \\    }
    \\    turboexl3_h128v4(vg, L);
    \\    turboexl3_h128v4(vu, L);
    \\    const half4 hg = half4(half(vg[0] * HAD_SCALE), half(vg[1] * HAD_SCALE), half(vg[2] * HAD_SCALE), half(vg[3] * HAD_SCALE));
    \\    const half4 hu = half4(half(vu[0] * HAD_SCALE), half(vu[1] * HAD_SCALE), half(vu[2] * HAD_SCALE), half(vu[3] * HAD_SCALE));
    \\    *reinterpret_cast<device half4*>(xg + ulong(p) * HC + i0) = hg;
    \\    *reinterpret_cast<device half4*>(xu + ulong(p) * HC + i0) = hu;
    \\    const float sg = turboexl3_sum_x4<int(XSO)>(hg, L);
    \\    const float su = turboexl3_sum_x4<int(XSO)>(hu, L);
    \\    if (L == 0u) { xsg[ulong(p) * NBC + b] = sg; xsu[ulong(p) * NBC + b] = su; }
    \\
;

pub const text_mid_v =
    \\    constexpr uint IC = uint(INTER);
    \\    constexpr uint NBC = IC / 128u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint w = threadgroup_position_in_grid.x * uint(ASG) + simdgroup_index_in_threadgroup;
    \\    if (w >= uint(order_shape[0]) * NBC) return;
    \\    const uint p = w / NBC;
    \\    const uint b = w - p * NBC;
    \\    const uint e = uint(inds[order[p]]);
    \\    const uint i0 = b * 128u + 4u * L;
    \\    const half4 zg4 = *reinterpret_cast<const device half4*>(zg + ulong(p) * IC + i0);
    \\    const half4 zu4 = *reinterpret_cast<const device half4*>(zu + ulong(p) * IC + i0);
    \\    float g[4] = {float(zg4[0]), float(zg4[1]), float(zg4[2]), float(zg4[3])};
    \\    float u[4] = {float(zu4[0]), float(zu4[1]), float(zu4[2]), float(zu4[3])};
    \\    turboexl3_h128v4(g, L);
    \\    turboexl3_h128v4(u, L);
    \\    const half4 vg4 = *reinterpret_cast<const device half4*>(svh_g + ulong(e) * IC + i0);
    \\    const half4 vu4 = *reinterpret_cast<const device half4*>(svh_u + ulong(e) * IC + i0);
    \\    const half4 sd4 = *reinterpret_cast<const device half4*>(suh_d + ulong(e) * IC + i0);
    \\    float d[4];
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const float gg = float(half(g[j] * HAD_SCALE * float(vg4[j])));
    \\        const float uu = float(half(u[j] * HAD_SCALE * float(vu4[j])));
    \\        const float h = (gg / (1.0f + metal::precise::exp(-gg))) * uu;
    \\        d[j] = float(half(h)) * float(sd4[j]);
    \\    }
    \\    turboexl3_h128v4(d, L);
    \\    const half4 hd = half4(half(d[0] * HAD_SCALE), half(d[1] * HAD_SCALE), half(d[2] * HAD_SCALE), half(d[3] * HAD_SCALE));
    \\    *reinterpret_cast<device half4*>(xd + ulong(p) * IC + i0) = hd;
    \\    const float sd = turboexl3_sum_x4<int(XSO)>(hd, L);
    \\    if (L == 0u) xsd[ulong(p) * NBC + b] = sd;
    \\
;

pub const text_finish_v =
    \\    constexpr uint HC = uint(HID);
    \\    constexpr uint NBC = HC / 128u;
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint w = threadgroup_position_in_grid.x * uint(ASG) + simdgroup_index_in_threadgroup;
    \\    if (w >= (uint(inv_shape[0]) / TOPKC) * NBC) return;
    \\    const uint r = w / NBC;
    \\    const uint b = w - r * NBC;
    \\    const uint i0 = b * 128u + 4u * L;
    \\    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    \\    for (uint k = 0; k < TOPKC; ++k) {
    \\        const uint src = r * TOPKC + k;
    \\        const uint p = inv[src];
    \\        const uint e = uint(inds[src]);
    \\        const float wt = float(scores[src]);
    \\        const half4 z4 = *reinterpret_cast<const device half4*>(zd + ulong(p) * HC + i0);
    \\        float v[4] = {float(z4[0]), float(z4[1]), float(z4[2]), float(z4[3])};
    \\        turboexl3_h128v4(v, L);
    \\        const half4 sv4 = *reinterpret_cast<const device half4*>(svh_d + ulong(e) * HC + i0);
    \\        for (uint j = 0; j < 4u; ++j) acc[j] += wt * float(half(v[j] * HAD_SCALE * float(sv4[j])));
    \\    }
    \\    for (uint j = 0; j < 4u; ++j) y[ulong(r) * HC + i0 + j] = OUT_T(acc[j]);
    \\
;

/// The fused GEMM: z OUT [S, ND] = f16(x [S, KD] (sorted rows) x W_e [KD, ND]) per projection op (NOPS 1: down; 2: gate|up,
/// op 0 reads x0/w0 and writes z0, op 1 x1/w1/z1). One threadgroup of 2 BMR / 32 simdgroups (BMR / 32 x 2, 32 x 32 outputs
/// each) per (M tile, op, NCB consecutive 64-wide output blocks), the blocks one after another. Per 64-deep K sub-stage
/// simdgroup sg decodes trellis tiles sg, sg + NSG, ... of its 4 x 4 (lane L: its eight codewords at code positions 8L..8L+7,
/// the tensor-core layout's rows 2q, 2q+1, 2q+8, 2q+9 of columns c, c+8, stored as four half2 into B^T [n][k], conflict-free
/// at LDB 72, or with FRAGB as the lane's NAX fragment), and each simdgroup runs two 32-deep NAX MMAs over it (A from device,
/// B^T from threadgroup memory). A K stage, the span between barriers, is BKS / 64 sub-stages: 1, or 2 for the down GEMM
/// with `Options.bk128` (16 KB per fragment-order buffer). Double-buffered (DBUF), stage f + 1 decodes while stage f
/// multiplies, sub-stage by sub-stage; the stage sequence runs on across a threadgroup's column blocks, so block j + 1's first
/// stage decodes during block j's last. Each block's accumulator starts clear and takes its sub-stages in K order: every
/// output sums the same 16-deep MMAs in the same order at any BKS and NCB. ALAY (`Options.alay`, fragb): A rows hold each
/// 32-block (1) or 16-block (2) in the fragments' slot order, so lane q's elements start at 8q (one 16-byte load per fragment
/// row and 32-deep step, `exl3pf_afrag2`) or 4q (one 8-byte load per fragment row and 16-deep slice, `exl3pf_afrag4`)
/// instead of 2q; H16 (`Options.half_rows`, fragb): a simdgroup with at most 16 rows multiplies only its first fragment row
/// (`exl3pf_mma_top`).
pub const text_gemm =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr int KDC = int(KD);
    \\    constexpr int NDC = int(ND);
    \\    constexpr int NOPSC = int(NOPS);
    \\    constexpr int WM = int(BMR) / 32;
    \\    constexpr int NSGC = 2 * WM;
    \\    constexpr int NPR = int(WSP);
    \\    constexpr int DEC = (NPR > 0) ? NPR : NSGC;
    \\    constexpr int JOBS = 16 / DEC;
    \\    constexpr int NBUF = (DBUF != 0) ? 2 : 1;
    \\    constexpr int BNC = 64;
    \\    constexpr int BKC = 64;
    \\    constexpr int SUBS = int(BKS) / BKC;
    \\    constexpr int NCBC = int(NCB);
    \\    constexpr int SM = 32;
    \\    constexpr int SN = 32;
    \\    constexpr int LDB = BKC + 8;
    \\    constexpr int BSZH = (FRAGB != 0 && (PROBE & 16) == 0) ? BNC * BKC : BNC * LDB;
    \\    constexpr int BSZ = SUBS * BSZH;
    \\    constexpr int K16 = KDC / 16;
    \\    constexpr int N16 = NDC / 16;
    \\    constexpr int NTG = NDC / BNC / NCBC;
    \\    constexpr int STAGES = KDC / (BKC * SUBS);
    \\    threadgroup uint4 bs_mem[NBUF * BSZ * 2 / 16];
    \\    threadgroup half* Bs = reinterpret_cast<threadgroup half*>(bs_mem);
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const int g = int(threadgroup_position_in_grid.x);
    \\    const int tile = g / (NOPSC * NTG);
    \\    const int rem = g - tile * (NOPSC * NTG);
    \\    const int op = rem / NTG;
    \\    const int n0g = (rem - op * NTG) * (NCBC * BNC);
    \\    const int e = tiles[4 * tile];
    \\    const int row0 = tiles[4 * tile + 1];
    \\    const int rows = tiles[4 * tile + 2];
    \\    if (rows == 0) return;
    \\    const bool consumer = int(sg) < NSGC;
    \\    const bool producer = NPR == 0 || !consumer;
    \\    const int dsg = (NPR > 0) ? int(sg) - NSGC : int(sg);
    \\    const int tm = SM * int(sg / 2u);
    \\    const int tn = SN * int(sg % 2u);
    \\    const int sgp = consumer ? min(SM, max(0, rows - tm)) : 0;
    \\    // the expert's trellis from the threadgroup's first output column; column block cb starts 4 cb tiles further
    \\    const device uint* wb = reinterpret_cast<const device uint*>(op == 0 ? w0 : WSECOND) + (ulong(e) * ulong(K16 * N16) + ulong(n0g / 16)) * ulong(N / 2u);
    \\    const device half* xa = (op == 0 ? x0 : XSECOND) + ulong(row0 + tm) * ulong(KDC);
    \\    const uint q = L & 3u;
    \\    const uint c = L >> 2u;
    \\    const uint qb = (INTR != 0) ? exl3_int_qb<N>(L) : exl3_gen_qb<N>(L);
    \\    const uint ga = (INTR != 0) ? exl3_int_a<N>(L) : exl3_gen_a<N>(L);
    \\    const uint lp = (INTR != 0) ? exl3_int_b<N>(L) : exl3_gen_lp<N>(L);
    \\    const uint lnx = (((L >> 4u) & 1u) << 4u) | (((L >> 1u) & 1u) << 3u) | (((L >> 3u) & 1u) << 2u) | (((L >> 2u) & 1u) << 1u) | (L & 1u);
    \\    Arows arw;
    \\    {
    \\        const uint fmx = ((L >> 2u) & 4u) | ((L >> 1u) & 3u);
    \\        const uint qnx = ((L >> 2u) & 2u) | (L & 1u);
    \\        const device half* xs = xa + ((ALAY == 1) ? 8u : ((ALAY == 2) ? 4u : 2u)) * qnx;
    \\        const int lastr = max(sgp - 1, 0);
    \\        arw.r[0] = xs + ulong(min(int(fmx), lastr)) * ulong(KDC);
    \\        arw.r[1] = xs + ulong(min(int(fmx) + 8, lastr)) * ulong(KDC);
    \\        arw.r[2] = xs + ulong(min(int(fmx) + 16, lastr)) * ulong(KDC);
    \\        arw.r[3] = xs + ulong(min(int(fmx) + 24, lastr)) * ulong(KDC);
    \\    }
    \\    int boff[JOBS];
    \\    ulong woff[JOBS];
    \\    for (int jj = 0; jj < JOBS; ++jj) {
    \\        const int j = max(dsg, 0) + DEC * jj;
    \\        const int kt = j >> 2;
    \\        const int ntl = j & 3;
    \\        boff[jj] = (FRAGB != 0) ? (ntl * 4 + kt) * 256 : (ntl * 16 + int(c)) * LDB + kt * 16 + 2 * int(q);
    \\        woff[jj] = (ulong(kt) * ulong(N16) + ulong(ntl)) * ulong(N / 2u);
    \\    }
    \\    const bool full = sgp == SM;
    \\    NAXTile<float, 2, 2> Dt;
    \\    Dt.clear();
    \\    if (DBUF != 0) {
    \\        STEEL_PRAGMA_UNROLL
    \\        for (int h = 0; h < SUBS; ++h) if (producer) exl3pf_stage<N, WMASK, CB, JOBS, N16, LDB, int(PROBE), int(FRAGB), int(INTR), int(DEFER)>(wb, h, woff, boff, Bs + h * BSZH, qb, ga, lp, lnx);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    for (int cb = 0; cb < NCBC; ++cb) {
    \\        if (DBUF != 0) {
    \\            for (int s = 0; s < STAGES; ++s) {
    \\                // f: the stage's place in the threadgroup's sequence (its buffer); the next stage is s + 1 of this block or 0
    \\                // of the next
    \\                const int f = cb * STAGES + s;
    \\                const bool last = s + 1 == STAGES;
    \\                const bool more = !last || cb + 1 < NCBC;
    \\                const device uint* wn = wb + ulong(4 * (last ? cb + 1 : cb)) * ulong(N / 2u);
    \\                const int sn = last ? 0 : s + 1;
    \\                STEEL_PRAGMA_UNROLL
    \\                for (int h = 0; h < SUBS; ++h) {
    \\                    const threadgroup half* Bc = Bs + (f & 1) * BSZ + h * BSZH;
    \\                    if (MFIRST != 0 && sgp > 0 && (PROBE & 2) == 0) exl3pf_mma<LDB, KDC, int(PROBE), int(UNR), int(FRAGB), int(ALAY), int(H16)>(Dt, xa, arw, SUBS * s + h, Bc, tn, full, sgp, L);
    \\                    if (producer && more) exl3pf_stage<N, WMASK, CB, JOBS, N16, LDB, int(PROBE), int(FRAGB), int(INTR), int(DEFER)>(wn, SUBS * sn + h, woff, boff, Bs + ((f + 1) & 1) * BSZ + h * BSZH, qb, ga, lp, lnx);
    \\                    if (MFIRST == 0 && sgp > 0 && (PROBE & 2) == 0) exl3pf_mma<LDB, KDC, int(PROBE), int(UNR), int(FRAGB), int(ALAY), int(H16)>(Dt, xa, arw, SUBS * s + h, Bc, tn, full, sgp, L);
    \\                }
    \\                threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            }
    \\        } else {
    \\            for (int s = 0; s < STAGES; ++s) {
    \\                threadgroup_barrier(mem_flags::mem_threadgroup);
    \\                STEEL_PRAGMA_UNROLL
    \\                for (int h = 0; h < SUBS; ++h) exl3pf_stage<N, WMASK, CB, JOBS, N16, LDB, int(PROBE), int(FRAGB), int(INTR), int(DEFER)>(wb + ulong(4 * cb) * ulong(N / 2u), SUBS * s + h, woff, boff, Bs + h * BSZH, qb, ga, lp, lnx);
    \\                threadgroup_barrier(mem_flags::mem_threadgroup);
    \\                STEEL_PRAGMA_UNROLL
    \\                for (int h = 0; h < SUBS; ++h) if (sgp > 0 && (PROBE & 2) == 0) exl3pf_mma<LDB, KDC, int(PROBE), int(UNR), int(FRAGB), int(ALAY), int(H16)>(Dt, xa, arw, SUBS * s + h, Bs + h * BSZH, tn, full, sgp, L);
    \\            }
    \\        }
    \\        if (sgp > 0) {
    \\            if (DEFER != 0) exl3pf_affine(Dt, (op == 0 ? xs0 : XSSECOND) + ulong(row0 + tm) * ulong(NBX), int(NBX), sgp, L);
    \\            device half* zp = (op == 0 ? z0 : ZSECOND) + ulong(row0 + tm) * ulong(NDC) + (n0g + cb * BNC) + tn;
    \\            if (full) Dt.store(zp, NDC);
    \\            else Dt.store_safe(zp, NDC, short2(SN, sgp));
    \\        }
    \\        if (cb + 1 < NCBC) Dt.clear();
    \\    }
    \\
;

/// The GEMM's two halves of a K stage. `exl3pf_stage`: simdgroup sg decodes trellis tiles sg, sg + NSG, ... of the stage's
/// 4 x 4 (lane L: its eight codewords at code positions 8L..8L+7, the tensor-core layout's rows 2q, 2q+1, 2q+8, 2q+9 of
/// columns c, c+8) into B^T [n][k] as four half2, conflict-free at LDB 72. `exl3pf_mma`: two 32-deep NAX MMAs of the
/// simdgroup's 32 x 32 outputs (A from device, B^T from threadgroup memory).
const metal_gemm_helpers =
    \\struct Arows { const device half* r[4]; };
    \\template <uint N, uint WMASK, int CB, int JOBS, int N16, int LDB, int P, int FB, int IR, int DF>
    \\inline void exl3pf_stage(const device uint* wb, int s, thread const ulong* woff, thread const int* boff, threadgroup half* Bs, uint qb, uint ga, uint lp, uint lnx) {
    \\    for (int jj = 0; jj < JOBS; ++jj) {
    \\        if (P & 1) {
    \\            const half k0 = half(float(s & 7) * 0.125f);
    \\            if (FB != 0) { *reinterpret_cast<threadgroup vec<half, 8>*>(Bs + boff[jj] + lnx * 8u) = vec<half, 8>(k0); continue; }
    \\            threadgroup half* d = Bs + boff[jj];
    \\            *reinterpret_cast<threadgroup half2*>(d) = half2(k0, k0);
    \\            *reinterpret_cast<threadgroup half2*>(d + 8) = half2(k0, k0);
    \\            *reinterpret_cast<threadgroup half2*>(d + 8 * LDB) = half2(k0, k0);
    \\            *reinterpret_cast<threadgroup half2*>(d + 8 * LDB + 8) = half2(k0, k0);
    \\            continue;
    \\        }
    \\        const device uint* tw = wb + ulong(s) * ulong(4 * N16) * ulong(N / 2u) + woff[jj];
    \\        const auto ww = exl3_tile_load<N, IR>(tw, qb, ga, lp);
    \\        const half v0 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 0, IR>(ww, lp));
    \\        const half v1 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 1, IR>(ww, lp));
    \\        const half v2 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 2, IR>(ww, lp));
    \\        const half v3 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 3, IR>(ww, lp));
    \\        const half v4 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 4, IR>(ww, lp));
    \\        const half v5 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 5, IR>(ww, lp));
    \\        const half v6 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 6, IR>(ww, lp));
    \\        const half v7 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 7, IR>(ww, lp));
    \\        if ((P & 32) != 0 && FB != 0) {
    \\            // C14 cost probe (outputs wrong): a 128-point Hadamard's butterflies over the decoded weights, 3 levels in the
    \\            // lane's 8 values and 4 across lanes (the floor of folding the input rotation into the weights)
    \\            half v[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
    \\            for (int h = 1; h < 8; h <<= 1) for (int i = 0; i < 8; ++i) if ((i & h) == 0) { const half a = v[i], b = v[i + h]; v[i] = a + b; v[i + h] = a - b; }
    \\            for (ushort m = 1; m < 16; m <<= 1) for (int i = 0; i < 8; ++i) { const half o = simd_shuffle_xor(v[i], m); v[i] = (lnx & m) != 0u ? o - v[i] : v[i] + o; }
    \\            *reinterpret_cast<threadgroup vec<half, 8>*>(Bs + boff[jj] + lnx * 8u) = vec<half, 8>(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    \\            continue;
    \\        }
    \\        if (FB != 0) { *reinterpret_cast<threadgroup vec<half, 8>*>(Bs + boff[jj] + lnx * 8u) = vec<half, 8>(v0, v1, v2, v3, v4, v5, v6, v7); continue; }
    \\        threadgroup half* d = Bs + boff[jj];
    \\        *reinterpret_cast<threadgroup half2*>(d) = half2(v0, v1);
    \\        *reinterpret_cast<threadgroup half2*>(d + 8) = half2(v2, v3);
    \\        *reinterpret_cast<threadgroup half2*>(d + 8 * LDB) = half2(v4, v5);
    \\        *reinterpret_cast<threadgroup half2*>(d + 8 * LDB + 8) = half2(v6, v7);
    \\    }
    \\}
    \\// FB: the stage holds each NAX lane's B^T fragment (8 halfs, 16 bytes) contiguously, written by the lane whose plan decodes
    \\// it; the A fragments are read in the same slot order (k 2q, 2q + 1, 2q + 8, 2q + 9 of each row, the register path's rule):
    \\// two half2 per fragment row and 16-deep slice from natural-order rows, or from rows stored in that slot order (AL 1: one
    \\// 16-byte load per fragment row and 32-deep step; AL 2: one 8-byte load per fragment row and 16-deep slice). H: a simdgroup
    \\// with at most 16 rows (sgp <= 16) loads and multiplies only its first fragment row (its rows 16..31 are past the tile and
    \\// never stored)
    \\template <int LDB, int KDC, int P, int U, int FB, int AL, int H>
    \\inline void exl3pf_mma(thread NAXTile<float, 2, 2>& Dt, const device half* xa, thread const Arows& arw, int s, const threadgroup half* Bs, int tn, bool full, int sgp, uint L) {
    \\    if (FB != 0) {
    \\        const bool top = H != 0 && sgp <= 16;
    \\        STEEL_PRAGMA_UNROLL
    \\        for (int i = 0; i < 2; ++i) {
    \\            NAXTile<half, 2, 2> At;
    \\            NAXTile<half, 2, 2> Bt;
    \\            if (AL == 1) {
    \\                const int k32 = s * 64 + i * 32;
    \\                if (P & 4) At.clear();
    \\                else {
    \\                    exl3pf_afrag2(At.frag_at(0, 0), At.frag_at(0, 1), arw.r[0] + k32, arw.r[1] + k32);
    \\                    if (!top) exl3pf_afrag2(At.frag_at(1, 0), At.frag_at(1, 1), arw.r[2] + k32, arw.r[3] + k32);
    \\                }
    \\            }
    \\            STEEL_PRAGMA_UNROLL
    \\            for (int kk = 0; kk < 2; ++kk) {
    \\                const int k0 = s * 64 + i * 32 + kk * 16;
    \\                if (AL != 1) {
    \\                    if (P & 4) { At.frag_at(0, kk) = vec<half, 8>(0); At.frag_at(1, kk) = vec<half, 8>(0); }
    \\                    else if (AL == 2) { exl3pf_afrag4(At.frag_at(0, kk), arw.r[0] + k0, arw.r[1] + k0); if (!top) exl3pf_afrag4(At.frag_at(1, kk), arw.r[2] + k0, arw.r[3] + k0); }
    \\                    else { exl3pf_afrag(At.frag_at(0, kk), arw.r[0] + k0, arw.r[1] + k0); if (!top) exl3pf_afrag(At.frag_at(1, kk), arw.r[2] + k0, arw.r[3] + k0); }
    \\                }
    \\                STEEL_PRAGMA_UNROLL
    \\                for (int nn = 0; nn < 2; ++nn) {
    \\                    if (P & 8) Bt.frag_at(nn, kk) = vec<half, 8>(0);
    \\                    else Bt.frag_at(nn, kk) = *reinterpret_cast<const threadgroup vec<half, 8>*>(Bs + ((tn / 16 + nn) * 4 + 2 * i + kk) * 256 + L * 8u);
    \\                }
    \\            }
    \\            if (top) exl3pf_mma_top(Dt, At, Bt);
    \\            else tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<true>{});
    \\        }
    \\        return;
    \\    }
    \\    if (U != 0) {
    \\        NAXTile<half, 2, 2> At0, At1, Bt0, Bt1;
    \\        if (P & 4) { At0.clear(); At1.clear(); }
    \\        else if (full) { At0.load(xa + s * 64, KDC); At1.load(xa + s * 64 + 32, KDC); }
    \\        else { At0.load_safe(xa + s * 64, KDC, short2(32, sgp)); At1.load_safe(xa + s * 64 + 32, KDC, short2(32, sgp)); }
    \\        if (P & 8) { Bt0.clear(); Bt1.clear(); }
    \\        else { Bt0.template load<half, LDB, 1>(Bs + tn * LDB); Bt1.template load<half, LDB, 1>(Bs + tn * LDB + 32); }
    \\        tile_matmad_nax(Dt, At0, metal::bool_constant<false>{}, Bt0, metal::bool_constant<true>{});
    \\        tile_matmad_nax(Dt, At1, metal::bool_constant<false>{}, Bt1, metal::bool_constant<true>{});
    \\        return;
    \\    }
    \\    STEEL_PRAGMA_NO_UNROLL
    \\    for (int i = 0; i < 2; ++i) {
    \\        NAXTile<half, 2, 2> At;
    \\        NAXTile<half, 2, 2> Bt;
    \\        volatile int compiler_barrier;
    \\        if (P & 4) At.clear();
    \\        else if (full) At.load(xa + s * 64 + i * 32, KDC);
    \\        else At.load_safe(xa + s * 64 + i * 32, KDC, short2(32, sgp));
    \\        if (P & 8) Bt.clear();
    \\        else Bt.template load<half, LDB, 1>(Bs + tn * LDB + i * 32);
    \\        tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<true>{});
    \\        (void)compiler_barrier;
    \\    }
    \\}
    \\
;

/// The register-path GEMM for tiles of <= 32 rows (`Options.small`): every simdgroup is an independent work item computing a
/// 32-row x 32-column output block of one tile, no threadgroup memory and no barriers. Per 32-deep K step a lane loads its NAX
/// A fragment elements and decodes its four trellis tiles straight into its B^T fragment registers. NAX lane L (bits b4..b0)
/// holds fragment rows fm = 4 b4 + 2 b2 + b1 and fm + 8 at k-slots 4q .. 4q + 3, q = 2 b3 + b0 (BaseNAXFrag::get_coord); the
/// trellis lane plan of lane L' = (b4, b2, b1, b3, b0) decodes exactly the weights of n = fm, fm + 8 at true k = 2q + {0, 1,
/// 8, 9}, in fragment element order. So slot 4q + j stands for true k 2q + {0, 1, 8, 9}[j] in BOTH operands: the A elements
/// are read from the natural-order inputs as two half2 per row (k 2q, 2q + 1 and 2q + 8, 2q + 9), and the contraction over
/// slots is the contraction over k. Rows past the tile's count read its last row (clamped pointers, no branches); their
/// products are never stored. Threadgroups hold SGT work items. SINTR (`Options.sintr`, n % 16 == 0): the lane plan's
/// codewords through the integer-rate reader instead of the general one, set up from the same permuted lane L' (both
/// readers place that lane's first window at the same start bit, so the same words, word selects, extract offsets and
/// codewords). ALAY (`Options.alay`): the inputs hold each 32-block (1: one 16-byte load per fragment row and 32-deep step, at
/// 8q) or each 16-block (2: one 8-byte load per fragment row and 16-deep slice, at 4q) in that slot order; H16
/// (`Options.half_rows`): an item of at most 16 rows multiplies only its first fragment row.
pub const text_gemm_small =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr int KDC = int(KD);
    \\    constexpr int NDC = int(ND);
    \\    constexpr int NOPSC = int(NOPS);
    \\    constexpr int SGS = int(SGT);
    \\    constexpr int K16 = KDC / 16;
    \\    constexpr int N16 = NDC / 16;
    \\    constexpr int NB = NDC / 32;
    \\    constexpr int STEPS = KDC / 32;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const int wi = int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup);
    \\    const int tile = wi / (NOPSC * NB);
    \\    const int rem = wi - tile * (NOPSC * NB);
    \\    const int op = rem / NB;
    \\    const int n0 = (rem - op * NB) * 32;
    \\    const int rows = tiles[4 * tile + 2];
    \\    if (rows == 0) return;
    \\    const int e = tiles[4 * tile];
    \\    const int row0 = tiles[4 * tile + 1];
    \\    const uint fm = ((L >> 2u) & 4u) | ((L >> 1u) & 3u);
    \\    const uint q = ((L >> 2u) & 2u) | (L & 1u);
    \\    const uint Lp = (L & 16u) | (((L >> 2u) & 1u) << 3u) | (((L >> 1u) & 1u) << 2u) | (((L >> 3u) & 1u) << 1u) | (L & 1u);
    \\    const uint qb = (SINTR != 0) ? exl3_int_qb<N>(Lp) : exl3_gen_qb<N>(Lp);
    \\    const uint ga = (SINTR != 0) ? exl3_int_a<N>(Lp) : exl3_gen_a<N>(Lp);
    \\    const uint lp = (SINTR != 0) ? exl3_int_b<N>(Lp) : exl3_gen_lp<N>(Lp);
    \\    const device uint* wb = reinterpret_cast<const device uint*>(op == 0 ? w0 : WSECOND) + ulong(e) * ulong(K16 * N16) * ulong(NW) + ulong(n0 / 16) * ulong(NW);
    \\    const device half* xa = (op == 0 ? x0 : XSECOND) + ulong(row0) * ulong(KDC) + ((ALAY == 1) ? 8u : ((ALAY == 2) ? 4u : 2u)) * q;
    \\    const device half* ar0 = xa + ulong(min(int(fm), rows - 1)) * ulong(KDC);
    \\    const device half* ar1 = xa + ulong(min(int(fm) + 8, rows - 1)) * ulong(KDC);
    \\    const device half* ar2 = xa + ulong(min(int(fm) + 16, rows - 1)) * ulong(KDC);
    \\    const device half* ar3 = xa + ulong(min(int(fm) + 24, rows - 1)) * ulong(KDC);
    \\    // H16: an item of at most 16 rows multiplies only its first fragment row (rows 16..31 are past the tile, never stored)
    \\    const bool top = H16 != 0 && rows <= 16;
    \\    NAXTile<float, 2, 2> Dt;
    \\    Dt.clear();
    \\    for (int st = 0; st < STEPS; ++st) {
    \\        NAXTile<half, 2, 2> At;
    \\        NAXTile<half, 2, 2> Bt;
    \\        if (ALAY == 1) {
    \\            exl3pf_afrag2(At.frag_at(0, 0), At.frag_at(0, 1), ar0 + st * 32, ar1 + st * 32);
    \\            if (!top) exl3pf_afrag2(At.frag_at(1, 0), At.frag_at(1, 1), ar2 + st * 32, ar3 + st * 32);
    \\        }
    \\        STEEL_PRAGMA_UNROLL
    \\        for (int kk = 0; kk < 2; ++kk) {
    \\            const int k0 = st * 32 + kk * 16;
    \\            if (ALAY == 2) {
    \\                exl3pf_afrag4(At.frag_at(0, kk), ar0 + k0, ar1 + k0);
    \\                if (!top) exl3pf_afrag4(At.frag_at(1, kk), ar2 + k0, ar3 + k0);
    \\            } else if (ALAY != 1) {
    \\                exl3pf_afrag(At.frag_at(0, kk), ar0 + k0, ar1 + k0);
    \\                if (!top) exl3pf_afrag(At.frag_at(1, kk), ar2 + k0, ar3 + k0);
    \\            }
    \\            STEEL_PRAGMA_UNROLL
    \\            for (int nn = 0; nn < 2; ++nn) {
    \\                const device uint* tw = wb + (ulong(st * 2 + kk) * ulong(N16) + ulong(nn)) * ulong(NW);
    \\                exl3pf_bfrag<N, WMASK, CB, int(DEFER), int(SINTR)>(Bt.frag_at(nn, kk), tw, qb, ga, lp);
    \\            }
    \\        }
    \\        if (top) exl3pf_mma_top(Dt, At, Bt);
    \\        else tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<true>{});
    \\    }
    \\    if (DEFER != 0) exl3pf_affine(Dt, (op == 0 ? xs0 : XSSECOND) + ulong(row0) * ulong(NBX), int(NBX), rows, L);
    \\    device half* zp = (op == 0 ? z0 : ZSECOND) + ulong(row0) * ulong(NDC) + n0;
    \\    if (rows >= 32) Dt.store(zp, NDC);
    \\    else Dt.store_safe(zp, NDC, short2(32, rows));
    \\
;

/// The register path's fragment fills: A from two natural-order rows (k 2q, 2q + 1, 2q + 8, 2q + 9 of each) or, with ALAY,
/// from two rows in that slot order (`exl3pf_afrag2`: both 16-deep halves of a 32-deep step; `exl3pf_afrag4`: one slice),
/// B^T from one trellis tile decoded with the permuted lane plan; the first-fragment-row MMA (`exl3pf_mma_top`, H16); and the
/// codeword reader both GEMM bodies decode through (`exl3pf_cw`).
const metal_small_helpers =
    \\// the deferred MUL1 map on a 32 x 32 accumulator block: lane L holds rows fm + 8 h + 16 mm (BaseNAXFrag::get_coord); each
    \\// row's input sum is its NB per-128-block partials (prep / mid); rows past `rows` are never stored
    \\inline void exl3pf_affine(thread NAXTile<float, 2, 2>& Dt, const device float* xsr, int nb, int rows, uint L) {
    \\    const float MA = float(as_type<half>(ushort(0x1EEEu)));
    \\    const float MC = 1024.0f * MA + float(as_type<half>(ushort(0xC931u)));
    \\    const int fm = int(((L >> 2u) & 4u) | ((L >> 1u) & 3u));
    \\    for (short mm = 0; mm < 2; ++mm) {
    \\        for (short hh = 0; hh < 2; ++hh) {
    \\            const int r = fm + 8 * hh + 16 * mm;
    \\            float sx = 0.0f;
    \\            if (r < rows) for (int q = 0; q < nb; ++q) sx += xsr[ulong(r) * ulong(nb) + q];
    \\            for (short nn = 0; nn < 2; ++nn) {
    \\                for (short jj = 0; jj < 4; ++jj) Dt.frag_at(mm, nn)[hh * 4 + jj] = fma(Dt.frag_at(mm, nn)[hh * 4 + jj], MA, MC * sx);
    \\            }
    \\        }
    \\    }
    \\}
    \\// a decoded weight for the GEMMs: the codebook value, or (DF, MUL1) the codeword's byte sum t in [0, 1020], exact in f16; the
    \\// epilogue then applies w = A t + C as A acc + C sum(x) (A = half 0x1EEE, C = 1024 A + half 0xC931)
    \\template <int CB, int DF> inline half exl3pf_w(uint cw) {
    \\    if (DF != 0) return exl3_mul1_th(cw);
    \\    return exl3_decode<CB>(cw);
    \\}
    \\inline void exl3pf_afrag(thread vec<half, 8>& f, const device half* r0, const device half* r1) {
    \\    const half2 a0 = *reinterpret_cast<const device half2*>(r0);
    \\    const half2 a1 = *reinterpret_cast<const device half2*>(r0 + 8);
    \\    const half2 a2 = *reinterpret_cast<const device half2*>(r1);
    \\    const half2 a3 = *reinterpret_cast<const device half2*>(r1 + 8);
    \\    f[0] = a0.x; f[1] = a0.y; f[2] = a1.x; f[3] = a1.y; f[4] = a2.x; f[5] = a2.y; f[6] = a3.x; f[7] = a3.y;
    \\}
    \\// ALAY 1: rows stored with each 32-block in the fragments' slot order (`turboexl3_alay`), r0 / r1 at the block's half 8q of the
    \\// fragment's two rows: halfs 0..3 are slots 4q..4q + 3 of the first 16-deep slice (f0), 4..7 those of the second (f1) - the
    \\// values exl3pf_afrag reads at k0 and k0 + 16 from natural-order rows, one 16-byte load per row instead of four half2
    \\inline void exl3pf_afrag2(thread vec<half, 8>& f0, thread vec<half, 8>& f1, const device half* r0, const device half* r1) {
    \\    const vec<half, 8> a = *reinterpret_cast<const device vec<half, 8>*>(r0);
    \\    const vec<half, 8> b = *reinterpret_cast<const device vec<half, 8>*>(r1);
    \\    f0[0] = a[0]; f0[1] = a[1]; f0[2] = a[2]; f0[3] = a[3]; f0[4] = b[0]; f0[5] = b[1]; f0[6] = b[2]; f0[7] = b[3];
    \\    f1[0] = a[4]; f1[1] = a[5]; f1[2] = a[6]; f1[3] = a[7]; f1[4] = b[4]; f1[5] = b[5]; f1[6] = b[6]; f1[7] = b[7];
    \\}
    \\// ALAY 2: rows stored with each 16-block in the fragments' slot order (`turboexl3_alay16`, dense.zig's LANE_ORDER rule), r0 / r1
    \\// at the slice's half 4q of the fragment's two rows: one 8-byte load per row, each straight into one half of the fragment
    \\inline void exl3pf_afrag4(thread vec<half, 8>& f, const device half* r0, const device half* r1) {
    \\    const half4 a = *reinterpret_cast<const device half4*>(r0);
    \\    const half4 b = *reinterpret_cast<const device half4*>(r1);
    \\    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
    \\}
    \\// H16: tile_matmad_nax's multiply-accumulates for the first 16-row fragment row only (fragment row 0 of Dt, both 16-deep
    \\// slices in order): rows 0..15 take the same 16x32x16 MMAs in the same order as the full call, rows 16..31 none
    \\inline void exl3pf_mma_top(thread NAXTile<float, 2, 2>& Dt, thread NAXTile<half, 2, 2>& At, thread NAXTile<half, 2, 2>& Bt) {
    \\    STEEL_PRAGMA_UNROLL
    \\    for (short kk = 0; kk < 2; ++kk) BaseNAXFrag::mma(Dt.frag_at(0, 0), Dt.frag_at(0, 1), At.frag_at(0, kk), metal::bool_constant<false>{}, Bt.frag_at(0, kk), Bt.frag_at(1, kk), metal::bool_constant<true>{});
    \\}
    \\// codeword m of a lane: the general reader (lp = its bit offset), or (IR, n % 16 == 0) the integer-rate one (lp = its b:
    \\// one select only where window m can carry, one extract); both read the same IntWords, so the same bits
    \\template <uint N, uint WMASK, uint M, int IR> inline uint exl3pf_cw(thread const Exl3BitsM<exl3_rmode<N>()>& w, uint lp) {
    \\    return exl3_tile_cw<N, WMASK, M, IR>(w, lp);
    \\}
    \\template <uint N, uint WMASK, int CB, int DF, int IR>
    \\inline void exl3pf_bfrag(thread vec<half, 8>& f, const device uint* tw, uint qb, uint ga, uint lp) {
    \\    const auto ww = exl3_tile_load<N, IR>(tw, qb, ga, lp);
    \\    f[0] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 0, IR>(ww, lp));
    \\    f[1] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 1, IR>(ww, lp));
    \\    f[2] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 2, IR>(ww, lp));
    \\    f[3] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 3, IR>(ww, lp));
    \\    f[4] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 4, IR>(ww, lp));
    \\    f[5] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 5, IR>(ww, lp));
    \\    f[6] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 6, IR>(ww, lp));
    \\    f[7] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 7, IR>(ww, lp));
    \\}
    \\
;

/// The NAX lane L's trellis lane for the register path: L' = (b4, b2, b1, b3, b0).
pub fn smallLanePlan(l: u32) u32 {
    return (l & 16) | (((l >> 2) & 1) << 3) | (((l >> 1) & 1) << 2) | (((l >> 3) & 1) << 1) | (l & 1);
}

/// H128 of a simdgroup's 128 values, lane L holding elements L + 32 j: five shuffle stages, then the two register stages.
const metal_h128 =
    \\inline void turboexl3_h128x4(thread float* v, uint L) {
    \\  for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\  const float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3];
    \\  v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3;
    \\}
    \\// the same H128 with lane L holding elements 4L..4L+3: strides 1 and 2 in the lane, 4..64 across lanes (lane bit = stride / 4),
    \\// in turboexl3_h128x4's level order with its lower a + b / upper a - b, so the same f32 value at every element
    \\inline void turboexl3_h128v4(thread float* v, uint L) {
    \\  const float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    \\  v[0] = a0 + a2; v[1] = a1 + a3; v[2] = a0 - a2; v[3] = a1 - a3;
    \\  for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\}
    \\// a 128-block's sum in the strided kernels' order from the vector layout: lane L adds elements L, L + 32, L + 64, L + 96 (lanes
    \\// L / 4 + 8 j, component L % 4) in that order, then simd_sum: the same per-lane partials, so the same sum
    \\template <int XSO> inline float turboexl3_sum_x4(half4 h, uint L) {
    \\  // XSO 1: the lane's own 4 elements in order, then simd_sum (no shuffles; a different f32 order than the strided kernels')
    \\  if (XSO != 0) return simd_sum(((float(h[0]) + float(h[1])) + float(h[2])) + float(h[3]));
    \\  float s = 0.0f;
    \\  for (uint j = 0; j < 4u; ++j) { const half4 o = simd_shuffle(h, ushort((L >> 2u) + 8u * j)); s += float(o[L & 3u]); }
    \\  return simd_sum(s);
    \\}
    \\// `Options.alay`: the slot of element j of a 32-block in the GEMMs' A layout (`prefill.alayPos`): fragment lane q = (j >> 1) & 3
    \\// holds k 2q, 2q + 1, 2q + 8, 2q + 9 of each 16-deep slice, stored at 8q + 4 (j >> 4) + {0, 1, 2, 3} (alay 1), or at
    \\// 16 (j >> 4) + 4q + {0, 1, 2, 3} (alay 2: per 16-block, `prefill.alayPos16`)
    \\inline uint turboexl3_alay(uint j) { return 8u * ((j >> 1u) & 3u) + 4u * (j >> 4u) + ((j & 1u) | (((j >> 3u) & 1u) << 1u)); }
    \\inline uint turboexl3_alay16(uint j) { return (j & 16u) + 4u * ((j >> 1u) & 3u) + ((j & 1u) | (((j >> 3u) & 1u) << 1u)); }
    \\
;
pub const header_small: [:0]const u8 = sentinel(metal_h128 ++ decode.header);

/// `Options.alay` 1's slot of element j (0..31) of a 32-element block of an A row: the fragment lane q = (j >> 1) & 3 reads k
/// 2q, 2q + 1, 2q + 8, 2q + 9 of each 16-deep slice (the trellis lane plan's order), so those of slice j >> 4 sit at
/// 8q + 4 (j >> 4) + {0, 1, 2, 3}: a lane's 8 elements of a 32-deep step are one 16-byte run. The Metal `turboexl3_alay`.
pub fn alayPos(j: u32) u32 {
    return 8 * ((j >> 1) & 3) + 4 * (j >> 4) + ((j & 1) | (((j >> 3) & 1) << 1));
}

/// `Options.alay` 2's slot: the same order per 16-element block (16 (j >> 4) + 4q + {0, 1, 2, 3}, dense.zig's LANE_ORDER): a
/// lane's 4 elements of a 16-deep slice are one 8-byte run. The Metal `turboexl3_alay16`.
pub fn alayPos16(j: u32) u32 {
    return (j & 16) + 4 * ((j >> 1) & 3) + ((j & 1) | (((j >> 3) & 1) << 1));
}

/// A comptime string as a NUL-terminated slice (the C API's text arguments).
fn sentinel(comptime s: []const u8) [:0]const u8 {
    comptime {
        var buf: [s.len:0]u8 = undefined;
        @memcpy(buf[0..s.len], s);
        const final = buf;
        return &final;
    }
}

fn textWithSecond(comptime two: bool) [:0]const u8 {
    return textWithSecondOf(text_gemm, two);
}

fn textWithSecondOf(comptime text: []const u8, comptime two: bool) [:0]const u8 {
    comptime {
        @setEvalBranchQuota(10_000_000);
        var out: []const u8 = text;
        out = replace(out, "XSSECOND", if (two) "xs1" else "xs0");
        out = replace(out, "XSECOND", if (two) "x1" else "x0");
        out = replace(out, "WSECOND", if (two) "w1" else "w0");
        out = replace(out, "ZSECOND", if (two) "z1" else "z0");
        return sentinel(out);
    }
}

/// `replace` for the mixed GEMM's splices: exactly one occurrence, or the build fails (a missed splice once left the small
/// path indexing its work items from the wrong origin).
fn replaceOnce(comptime s: []const u8, comptime needle: []const u8, comptime with: []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(10_000_000);
        const first = std.mem.indexOf(u8, s, needle) orelse @compileError("splice not found: " ++ needle);
        if (std.mem.indexOfPos(u8, s, first + needle.len, needle) != null) @compileError("splice found twice: " ++ needle);
        return replace(s, needle, with);
    }
}

fn replace(comptime s: []const u8, comptime needle: []const u8, comptime with: []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(10_000_000);
        var out: []const u8 = "";
        var i: usize = 0;
        while (std.mem.indexOfPos(u8, s, i, needle)) |j| {
            out = out ++ s[i..j] ++ with;
            i = j + needle.len;
        }
        return out ++ s[i..];
    }
}

const gemm2_text = textWithSecond(true);
pub const gemm1_text = textWithSecond(false);
/// The mixed GEMM (`Options.small = .mixed`): one launch, threadgroups below GBIGM take a BMR-row tile (NCB column blocks of it)
/// through the shared-decode body (`text_gemm`), the rest run NSG independent register-path work items each
/// (`text_gemm_small`) over the table's small tiles. Both bounds come from the tile table's shape, so the row count compiles
/// nothing. The weight stage is declared at kernel scope (the big body's `bs_mem`, same size).
pub const text_gemm_mixed: []const u8 = blk: {
    @setEvalBranchQuota(10_000_000);
    var big: []const u8 = replaceOnce(text_gemm, "    threadgroup uint4 bs_mem[NBUF * BSZ * 2 / 16];\n", "");
    big = replaceOnce(big, "reinterpret_cast<threadgroup half*>(bs_mem)", "reinterpret_cast<threadgroup half*>(mixed_bs)");
    var small: []const u8 = replaceOnce(text_gemm_small, "const int wi = int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup);", "const int wi = (int(threadgroup_position_in_grid.x) - GBIGM) * SGS + int(simdgroup_index_in_threadgroup);");
    small = replaceOnce(small, "const int tile = wi / (NOPSC * NB);", "const int stile = wi / (NOPSC * NB);\n    const int tile = TBIGM + stile;\n    if (tile >= int(tiles_shape[0])) return;");
    small = replaceOnce(small, "const int rem = wi - tile * (NOPSC * NB);", "const int rem = wi - stile * (NOPSC * NB);");
    break :blk "    threadgroup uint4 mixed_bs[((DBUF != 0) ? 2 : 1) * (int(BKS) / 64) * 64 * ((FRAGB != 0 && (PROBE & 16) == 0) ? 64 : 72) * 2 / 16];\n" ++
        "    const int TBIGM = int(tiles_shape[0]) - int(TMAXS);\n" ++
        "    const int GBIGM = TBIGM * int(NOPS) * (int(ND) / 64 / int(NCB));\n" ++
        "    if (int(threadgroup_position_in_grid.x) < GBIGM) {\n" ++ big ++ "    } else {\n" ++ small ++ "    }\n";
};
const gemm2m_text = textWithSecondOf(text_gemm_mixed, true);
const gemm1m_text = textWithSecondOf(text_gemm_mixed, false);
const gemm2s_text = textWithSecondOf(text_gemm_small, true);
pub const gemm1s_text = textWithSecondOf(text_gemm_small, false);

// ── The 128-row tiles (`Options.t128`, `Prefill.moeT128`) ──
//
// Every expert's full 128-row runs (and, past its last one, a remainder of more than `t128_cut` rows) become 128-row tiles
// on the bm-128 shared-decode body, where one decoded 64 x 64 weight stage feeds 128 rows instead of 64; the rest of each
// expert (at most `t128_cut` rows) takes the mixed launch (64-row tiles, register-path tiles of <= 32 rows). A launch has
// one threadgroup size, so the two run as two launches, each writing its own z (an MLX output [S, N] at sorted positions,
// never written or read at the other launch's rows); the route's `xbase` [E] (each expert's first remainder position)
// tells mid and finish which z holds a row: position p of expert e is the 128-row tiles' when p < xbase[e]. Every write is
// to its own kernel's output (a launch writing another's input binding is undefined in Metal), so MLX orders the readers
// after both launches, and nothing passes between the two, so the second starts as the first drains. Both tables are
// sized by routing-independent bounds (`t128MixedRows`, `t128TileRows`) that keep the early-exit threadgroups near the
// mixed table's count. Every output sums the same 16-deep MMAs in the same K order (bm 128 and bm 64 run the same
// per-simdgroup K loop): `moe`'s bytes, for a second [S, N] output per GEMM.

/// Past an expert's full 128-row tiles, a remainder of more than this many rows takes one part-full 128-row tile (one
/// decode pass over its weights where the mixed tiles of the same rows take two, at the cost of idle simdgroups).
pub const t128_cut: u8 = 64;

/// The 128-row tiles' route (`Options.froute`'s one launch: `froute_count` verbatim, so `order` is froute's): from each
/// expert's first sorted position, 128-row tiles while 128 rows are left, then, when more than CUT remain (CU = min(CUT,
/// 127)), one part-full 128-row tile for them; whatever is left takes `text_table_mixed`'s rule (BMT-row tiles, a <= 32
/// SPL-row remainder as register-path tiles). T128 0: no 128-row tiles (the mixed table of every row). Outputs order u32
/// [S], tiles int32 [TBIG + SPL EXP, 4] (`text_table_mixed`'s layout, the mixed launch's table), tiles128 int32 [TQ, 4] and
/// xbase int32 [EXP] (each expert's first remainder position: its rows before it are the 128-row tiles', `text_mid_t128` /
/// `text_finish_t128`); TBIG and TQ are `t128MixedRows`' and `t128TileRows`' expressions (bounds that hold for any routing);
/// unused entries rows = 0.
pub const text_froute_t128: [:0]const u8 = sentinel(
    \\    constexpr uint EC = uint(EXP);
    \\    constexpr uint BMC = uint(BMT);
    \\    constexpr uint SR = 32u * uint(SPL);
    \\    constexpr uint CU = (uint(CUT) < 127u) ? uint(CUT) : 127u;
    \\    const uint SC = uint(inds_shape[0]);
    \\    constexpr uint PER = CU / BMC + ((CU % BMC > SR) ? 1u : 0u);
    \\    constexpr uint MINR = (BMC < SR + 1u) ? BMC : SR + 1u;
    \\    const uint TBIG0 = (SC + BMC - 1u) / BMC + EC;
    \\    const uint TBIG = (T128 != 0) ? max(1u, min(TBIG0, min(EC * PER, SC / MINR))) : TBIG0;
    \\    const uint TALL = TBIG + uint(SPL) * EC;
    \\    const uint NPQ = min(EC, SC / (CU + 1u));
    \\    const uint TQ = (T128 != 0) ? max(1u, NPQ + (SC - (CU + 1u) * NPQ) / 128u) : 1u;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    threadgroup uint lo_s[1025];
    \\    threadgroup uint prefq[32];
    \\    threadgroup uint prefb[32];
    \\    threadgroup uint prefs[32];
    \\    threadgroup uint totq_s;
    \\    threadgroup uint totb_s;
    \\    threadgroup uint tots_s;
    \\
++ froute_count ++
    \\    lo_s[tid] = lo;
    \\    if (tid == 0u) lo_s[1024] = SC;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    uint c0 = 0u, cnt = 0u, nq = 0u, nb = 0u, ns = 0u, nfull = 0u, r = 0u;
    \\    if (tid < EC) {
    \\        c0 = lo_s[tid + 1u] - lo;
    \\        nq = (T128 != 0) ? c0 / 128u : 0u;
    \\        cnt = c0 - 128u * nq;
    \\        if (T128 != 0 && cnt > CU) { nq += 1u; cnt = 0u; }
    \\        nfull = cnt / BMC;
    \\        r = cnt - nfull * BMC;
    \\        if (r == 0u) nb = nfull;
    \\        else if (r > SR) nb = nfull + 1u;
    \\        else { nb = nfull; ns = (r + 31u) / 32u; }
    \\    }
    \\    // the remainder's first sorted position
    \\    const uint base = lo + c0 - cnt;
    \\    if (tid < EC) xbase[tid] = int(base);
    \\    const uint iq = simd_prefix_inclusive_sum(nq);
    \\    const uint ib = simd_prefix_inclusive_sum(nb);
    \\    const uint is = simd_prefix_inclusive_sum(ns);
    \\    if (L == 31u) { prefq[sg] = iq; prefb[sg] = ib; prefs[sg] = is; }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (sg == 0u) { prefq[L] = simd_prefix_exclusive_sum(prefq[L]); prefb[L] = simd_prefix_exclusive_sum(prefb[L]); prefs[L] = simd_prefix_exclusive_sum(prefs[L]); }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const uint sq = prefq[sg] + iq - nq;
    \\    const uint sb = prefb[sg] + ib - nb;
    \\    const uint ss = prefs[sg] + is - ns;
    \\    for (uint t = 0u; t < nq; ++t) {
    \\        const uint i = sq + t;
    \\        tiles128[4u * i] = int(tid); tiles128[4u * i + 1u] = int(lo + 128u * t); tiles128[4u * i + 2u] = int(min(128u, c0 - 128u * t)); tiles128[4u * i + 3u] = 0;
    \\    }
    \\    for (uint t = 0u; t < nb; ++t) {
    \\        const uint i = sb + t;
    \\        tiles[4u * i] = int(tid); tiles[4u * i + 1u] = int(base + t * BMC); tiles[4u * i + 2u] = int(min(BMC, cnt - t * BMC)); tiles[4u * i + 3u] = 0;
    \\    }
    \\    for (uint t = 0u; t < ns; ++t) {
    \\        const uint i = TBIG + ss + t;
    \\        tiles[4u * i] = int(tid); tiles[4u * i + 1u] = int(base + nfull * BMC + 32u * t); tiles[4u * i + 2u] = int(min(32u, r - 32u * t)); tiles[4u * i + 3u] = 0;
    \\    }
    \\    if (tid == 1023u) { totq_s = sq + nq; totb_s = sb + nb; tots_s = ss + ns; }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint i = totq_s + tid; i < TQ; i += 1024u) { tiles128[4u * i] = 0; tiles128[4u * i + 1u] = 0; tiles128[4u * i + 2u] = 0; tiles128[4u * i + 3u] = 0; }
    \\    for (uint i = totb_s + tid; i < TBIG; i += 1024u) { tiles[4u * i] = 0; tiles[4u * i + 1u] = 0; tiles[4u * i + 2u] = 0; tiles[4u * i + 3u] = 0; }
    \\    for (uint i = TBIG + tots_s + tid; i < TALL; i += 1024u) { tiles[4u * i] = 0; tiles[4u * i + 1u] = 0; tiles[4u * i + 2u] = 0; tiles[4u * i + 3u] = 0; }
    \\
);

/// The 128-row tiles' mid: `text_mid` with each position's gate and up rows from the launch that computed them (zgt / zut,
/// the 128-row tiles' outputs, for p < xbase[e]; zg / zu, the mixed launch's, otherwise); one simdgroup holds one position,
/// so the choice is uniform per simdgroup. Nothing else changes, so the same bytes from the same rows.
const mid_t128_sel = "    const uint e = uint(inds[order[p]]);\n";
pub const text_mid_t128: [:0]const u8 = sentinel(replaceOnce(replaceOnce(replaceOnce(text_mid, mid_t128_sel, mid_t128_sel ++
    "    const bool qt = int(p) < xbase[e];\n    const device half* zgs = qt ? zgt : zg;\n    const device half* zus = qt ? zut : zu;\n"), "g[j] = float(zg[ulong(p) * IC + i]);", "g[j] = float(zgs[ulong(p) * IC + i]);"), "u[j] = float(zu[ulong(p) * IC + i]);", "u[j] = float(zus[ulong(p) * IC + i]);"));
/// The 128-row tiles' finish: `text_finish` with each slot's down row from the launch that computed it (zdt for p < xbase[e],
/// else zd)
const fin_t128_sel = "        const uint e = uint(inds[src]);\n";
pub const text_finish_t128: [:0]const u8 = sentinel(replaceOnce(replaceOnce(text_finish, fin_t128_sel, fin_t128_sel ++
    "        const device half* zds = (int(p) < xbase[e]) ? zdt : zd;\n"), "v[j] = float(zd[ulong(p) * HC", "v[j] = float(zds[ulong(p) * HC"));

const table_in = [_][*:0]const u8{ "inds", "order" };
const table_out = [_][*:0]const u8{"tiles"};
const froute_in = [_][*:0]const u8{"inds"};
const froute_out = [_][*:0]const u8{ "order", "tiles" };
const prep_in = [_][*:0]const u8{ "x", "inds", "order", "suh_g", "suh_u" };
const prep_out = [_][*:0]const u8{ "xg", "xu", "inv", "xsg", "xsu" };
const gemm2_in = [_][*:0]const u8{ "tiles", "x0", "x1", "w0", "w1", "xs0", "xs1" };
const gemm2_out = [_][*:0]const u8{ "z0", "z1" };
pub const gemm1_in = [_][*:0]const u8{ "tiles", "x0", "w0", "xs0" };
pub const gemm1_out = [_][*:0]const u8{"z0"};
const mid_in = [_][*:0]const u8{ "zg", "zu", "inds", "order", "svh_g", "svh_u", "suh_d" };
const mid_out = [_][*:0]const u8{ "xd", "xsd" };
const fin_in = [_][*:0]const u8{ "zd", "inds", "inv", "scores", "svh_d" };
const fin_out = [_][*:0]const u8{"y"};
/// The 128-row tiles' route outputs; mid and finish take the 128-row tiles' outputs and xbase as more inputs (both GEMMs'
/// launches keep the mixed chain's kernels, inputs and outputs)
const froute_t128_out = [_][*:0]const u8{ "order", "tiles", "tiles128", "xbase" };
const mid_t128_in = mid_in ++ [_][*:0]const u8{ "zgt", "zut", "xbase" };
const fin_t128_in = fin_in ++ [_][*:0]const u8{ "zdt", "xbase" };

pub const Geometry = struct {
    hidden: u32,
    inter: u32,
    topk: u32,
    experts: u32,
    rate: format.Rate,
    dec: format.Decode,

    pub fn admitted(g: Geometry) bool {
        return g.hidden % 128 == 0 and g.inter % 128 == 0 and g.hidden % BK == 0 and g.inter % BK == 0 and
            g.hidden % BN == 0 and g.inter % BN == 0 and g.experts >= 1 and g.experts <= 1024 and g.topk >= 1 and g.topk <= 32;
    }
};

/// The largest count of 128-blocks per threadgroup (at most 8 simdgroups) that divides `blocks`.
fn bptFor(blocks: u32) u32 {
    var b: u32 = 8;
    while (b > 1) : (b -= 1) if (blocks % b == 0) return b;
    return 1;
}

pub const Operands = struct {
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
};

/// Build-time variants of the GEMM (the benchmark's arms): M tile rows (64, 128 or 256: 2, 4 or 8 simdgroup rows of 32, two
/// simdgroups per row covering the 64-wide output tile) and a double-buffered weight stage (one barrier per K stage, the next
/// stage decoded while this one multiplies).
pub const Options = struct {
    bm: u32 = 128,
    dbuf: bool = false,
    /// .all: every M tile is <= 32 rows and runs the register-path GEMM (`text_gemm_small`), sgt work items per threadgroup
    small: Small = .off,
    sgt: u32 = 4,
    /// timing probes only (outputs are wrong), a bit mask: 1 skips the trellis read and decode (constant weights), 2 skips the
    /// MMA step entirely (the compiler then drops the unread decode too), 4 replaces the A loads by zeros, 8 the B loads;
    /// 16 (outputs still right) keeps the fragment-order stage at the padded 64 x 72 size it had before it was trimmed to 64 x 64;
    /// 32 (fragb) adds a 128-point Hadamard's butterflies to every decoded weight (C14's cost floor)
    probe: u8 = 0,
    /// producer simdgroups that only decode (0 = every simdgroup decodes and multiplies); needs dbuf
    wsp: u8 = 0,
    /// issue the stage's MMA before decoding the next stage (dbuf)
    mma_first: bool = false,
    /// both 32-deep sub-steps' operands loaded up front, no compiler barrier between them
    unroll: bool = false,
    /// the weight stage in NAX fragment order (one 16-byte store per decoded tile per lane, one 16-byte load per fragment)
    fragb: bool = false,
    /// the integer-rate codeword reader (n % 16 == 0)
    intr: bool = false,
    /// the register path (small = .all, or .mixed's <= 32-row tiles) reads its codewords with intr's integer-rate reader too
    /// (needs intr); off, it keeps the general reader, so an A/B of the two register-path readers runs in one binary. Same
    /// codewords either way: outputs identical
    sintr: bool = false,
    /// MUL1: the GEMMs multiply the codeword byte sums t and the epilogue applies w = A t + C as A acc + C sum(x)
    defer_affine: bool = false,
    /// .mixed: remainders of up to 32 * split rows take the register path as split tiles of <= 32 rows (1 or 2)
    split: u8 = 1,
    /// the down GEMM's K stages 128 deep: two 64-deep sub-stages per buffer between barriers (each decoded and multiplied by
    /// the same helpers as a 64-deep stage), so half the barriers and stage-loop turns of its K loop; 16 KB per fragment-order
    /// buffer, 32 KB double-buffered (the padded layout's 36 KB does not fit Apple's 32 KB: fragb, or single-buffered).
    /// gate|up keeps 64
    bk128: bool = false,
    /// 64-wide output blocks per down-GEMM threadgroup (1, 2 or 4; the hidden width's blocks must divide by it): each
    /// threadgroup walks ncb consecutive blocks of its M tile, the double buffer carried across them (block j + 1's first
    /// stage decoded while block j's last multiplies); gate|up keeps 1
    ncb: u8 = 1,
    /// route in one launch (the table kernel counts, scans and scatters the slots itself) instead of MLX's argsort + the
    /// table; outputs identical (see the module doc)
    froute: bool = false,
    /// prep, mid and finish in the vector layout (`text_prep_v`, ...) with aux simdgroups per threadgroup (4, 8, 16 or 32;
    /// 0 = the strided kernels); outputs identical
    aux: u8 = 0,
    /// with aux: each block's sums (the deferred map's C term) in the lane's own order instead of the strided kernels' (no
    /// shuffles; f32 sums in a different order, so outputs may differ in the last bits)
    aux_lane_sums: bool = false,
    /// the GEMMs' A operands (prep's xg / xu, mid's xd) stored in the A fragments' slot order: the lane that multiplies k 2q,
    /// 2q + 1, 2q + 8, 2q + 9 of a 16-deep slice (fragb's and the register path's rule) finds them contiguous. 1: each
    /// 32-element block in that order (`alayPos`), both slices of a 32-deep step at halfs 8q .. 8q + 7, one 16-byte load per
    /// fragment row and step (8 per 64-deep stage instead of 32 half2 loads; 4 per register-path step instead of 16); 2: each
    /// 16-element block (`alayPos16`, dense.zig's LANE_ORDER rule), one 8-byte load straight into each fragment row half (16
    /// per stage, 8 per step: no splitting of a loaded vector across two fragments). The same values in the same slots, so
    /// identical outputs. Where the shared-decode body runs it needs fragb (the padded bodies load A in natural order); its
    /// writers are the strided prep and mid (aux 0)
    alay: u8 = 0,
    /// a fragment-order shared-decode simdgroup (fragb) or a register-path item whose rows end within its first 16 loads and
    /// multiplies only that 16-row fragment row: half its 16x32x16 MMAs and A loads; the rows it skips are past the tile and
    /// never stored, the rows it keeps take the same MMAs in the same order, so identical outputs
    half_rows: bool = false,
    /// each expert's full 128-row runs (and a remainder of more than `t128_cut` rows) in 128-row tiles on the bm-128 body, a
    /// launch of their own beside the mixed launch over the rest (`Prefill.moeT128`, the notes above `t128_cut`): `moe`'s
    /// bytes. Needs bm 64, the mixed table at split 1, froute's route and the strided mid and finish (aux 0)
    t128: bool = false,

    pub const Small = enum { off, all, mixed };

    pub fn valid(o: Options) bool {
        if (o.small == .mixed and o.wsp != 0) return false;
        if (o.split != 1 and (o.split != 2 or o.small != .mixed)) return false;
        if (o.wsp != 0 and (!o.dbuf or (o.wsp != 2 and o.wsp != 4 and o.wsp != 8 and o.wsp != 16) or 2 * o.bm / 32 + o.wsp > 32)) return false;
        // both reshape the shared-decode body: nothing to do for the all-register path
        if ((o.bk128 or o.ncb != 1) and o.small == .all) return false;
        // the register path's reader: intr's (and its n % 16 == 0 check), and nothing to do where no register path runs
        if (o.sintr and (!o.intr or o.small == .off)) return false;
        if (o.ncb != 1 and o.ncb != 2 and o.ncb != 4) return false;
        if (o.aux != 0 and o.aux != 4 and o.aux != 8 and o.aux != 16 and o.aux != 32) return false;
        // the A layout and the first-fragment-row MMA live in the fragment-order body and the register path; the layout's
        // writers are the strided prep and mid
        if (o.alay > 2) return false;
        if ((o.alay != 0 or o.half_rows) and !o.fragb and o.small != .all) return false;
        if (o.alay != 0 and o.aux != 0) return false;
        // the remainder tiles ride the mixed launch (froute's route at split 1, the strided mid and finish), the 128-row tiles
        // the bm-128 shared-decode body beside bm 64's
        if (o.t128 and (o.small != .mixed or o.split != 1 or !o.froute or o.aux != 0 or o.bm != 64)) return false;
        if (o.stageBytes(o.downStage()) > 32768) return false;
        return (o.bm == 64 or o.bm == 128 or o.bm == 256) and (o.sgt == 1 or o.sgt == 2 or o.sgt == 4 or o.sgt == 8);
    }

    /// rows per M tile of the table
    pub fn tileRows(o: Options) u32 {
        return if (o.small == .all) 32 else o.bm;
    }

    /// the down GEMM's K stage depth (gate|up's is BK)
    pub fn downStage(o: Options) u32 {
        return if (o.bk128) 128 else BK;
    }

    /// threadgroup memory of the shared-decode GEMM's weight stage at K stage depth `bk` (bk / BK sub-stages per buffer),
    /// bytes, both buffers with dbuf
    pub fn stageBytes(o: Options, bk: u32) u32 {
        const sub: u32 = if (o.fragb and (o.probe & 16) == 0) BN * BK else BN * (BK + 8);
        return (if (o.dbuf) @as(u32, 2) else 1) * (bk / BK) * sub * 2;
    }
};

/// The routing of one call: sorted position -> slot (`order`), the M tiles (`tiles` int32 [tmax, 4]).
pub const Route = struct {
    order: mlx.mlx_array,
    tiles: mlx.mlx_array,
    tmax: u32,

    pub fn deinit(r: Route) void {
        _ = mlx.mlx_array_free(r.order);
        _ = mlx.mlx_array_free(r.tiles);
    }
};

/// The 128-row tiles' routing (`Prefill.routeT128`): `Route`'s order and mixed table (`tiles` over every expert's remainder
/// past its 128-row tiles, `tmax` its BMT-row tile rows, `t128MixedRows`), the 128-row tiles (`tiles128` int32 [tq, 4], tq =
/// `t128TileRows`) and `xbase` int32 [experts] (each expert's first remainder position, `ZT128`).
pub const RouteT128 = struct {
    order: mlx.mlx_array,
    tiles: mlx.mlx_array,
    tiles128: mlx.mlx_array,
    xbase: mlx.mlx_array,
    tmax: u32,
    tq: u32,

    pub fn deinit(r: RouteT128) void {
        _ = mlx.mlx_array_free(r.order);
        _ = mlx.mlx_array_free(r.tiles);
        _ = mlx.mlx_array_free(r.tiles128);
        _ = mlx.mlx_array_free(r.xbase);
    }
};

/// One GEMM's f16 inner products as its two launches wrote them, each its own MLX output at sorted positions: `m` the mixed
/// launch's (the remainder tiles' rows), `t` the 128-row tiles' launch's. Row p of expert e is `t`'s when p < xbase[e],
/// else `m`'s; neither array is written (or read) at the other's rows.
pub const ZT128 = struct {
    m: [2]mlx.mlx_array,
    t: [2]mlx.mlx_array,
    n: u8,

    pub fn deinit(x: ZT128) void {
        for (x.m[0..x.n]) |a| _ = mlx.mlx_array_free(a);
        for (x.t[0..x.n]) |a| _ = mlx.mlx_array_free(a);
    }
};

/// The 128-row tile table's rows for S slots (at least 1; 1 with the tiles off). A full tile holds 128 rows, a part-full one
/// more than cu = min(cut, 127) and an expert has at most one, so np part-full tiles leave at most (S - (cu + 1) np) / 128
/// full ones; the count is largest at np = min(experts, S / (cu + 1)). Every routing fits, and the launch's grid stays near
/// its tiles (each spare table row is a row of early-exit threadgroups). `text_froute_t128`'s TQ is the same expression.
pub fn t128TileRows(S: u32, experts: u32, cut: u8, t128: bool) u32 {
    if (!t128) return 1;
    const cu1: u32 = @as(u32, @min(cut, 127)) + 1;
    const np = @min(experts, S / cu1);
    return @max(1, np + (S - cu1 * np) / 128);
}

/// The mixed table's `bm`-row tile rows for S slots (its register-path rows follow them): ceil(S / bm) + experts without the
/// 128-row tiles; with them an expert's remainder is at most cu = min(cut, 127) rows, so at most cu / bm + (cu % bm > 32
/// split) such tiles each, every one of at least min(bm, 32 split + 1) rows (at least 1). The tables of every routing fit.
/// `text_froute_t128`'s TBIG is the same expression.
pub fn t128MixedRows(S: u32, experts: u32, bm: u32, split: u32, cut: u8, t128: bool) u32 {
    const all = (S + bm - 1) / bm + experts;
    if (!t128) return all;
    const cu: u32 = @min(cut, 127);
    const sr = 32 * split;
    const per = cu / bm + @intFromBool(cu % bm > sr);
    return @max(1, @min(all, @min(experts * per, S / @min(bm, sr + 1))));
}

/// The 128-row tiles' tables from per-expert slot counts, `text_froute_t128`'s rule on the CPU (experts in order, each run
/// from its first sorted position): the 128-row tiles, the remainder's `bm`-row tiles and its register-path tiles (split
/// 1), each as (expert, first sorted position, rows). Tests check the route kernel's tables and their cover against it.
pub const T128Tables = struct {
    t128: std.ArrayList([3]u32) = .empty,
    big: std.ArrayList([3]u32) = .empty,
    small: std.ArrayList([3]u32) = .empty,

    pub fn deinit(t: *T128Tables, a: std.mem.Allocator) void {
        t.t128.deinit(a);
        t.big.deinit(a);
        t.small.deinit(a);
    }
};

pub fn t128Tables(a: std.mem.Allocator, counts: []const u32, bm: u32, cut: u8, t128: bool) !T128Tables {
    var t: T128Tables = .{};
    errdefer t.deinit(a);
    const cu: u32 = @min(cut, 127);
    var lo: u32 = 0;
    for (counts, 0..) |c0, ei| {
        const e: u32 = @intCast(ei);
        var nq: u32 = if (t128) c0 / 128 else 0;
        var cnt = c0 - 128 * nq;
        if (t128 and cnt > cu) {
            nq += 1;
            cnt = 0;
        }
        var q: u32 = 0;
        while (q < nq) : (q += 1) try t.t128.append(a, .{ e, lo + 128 * q, @min(128, c0 - 128 * q) });
        const base = lo + c0 - cnt;
        const nfull = cnt / bm;
        const r = cnt - nfull * bm;
        const nb = nfull + @intFromBool(r > 32);
        var i: u32 = 0;
        while (i < nb) : (i += 1) try t.big.append(a, .{ e, base + bm * i, @min(bm, cnt - bm * i) });
        if (r > 0 and r <= 32) try t.small.append(a, .{ e, base + nfull * bm, r });
        lo += c0;
    }
    return t;
}

/// The 128-row tiles' kernels (`Options.t128`): the route, both GEMMs' launches (over the remainder table: the mixed launch;
/// over the 128-row tiles: the shared-decode body at bm 128; the mixed chain's texts, header, inputs and outputs), and mid
/// and finish taking each row from the launch that computed it.
pub const T128Kernels = struct {
    route: mlx.mlx_fast_metal_kernel,
    gu_mixed: mlx.mlx_fast_metal_kernel,
    gu_t128: mlx.mlx_fast_metal_kernel,
    dn_mixed: mlx.mlx_fast_metal_kernel,
    dn_t128: mlx.mlx_fast_metal_kernel,
    mid: mlx.mlx_fast_metal_kernel,
    fin: mlx.mlx_fast_metal_kernel,

    fn init() !T128Kernels {
        return .{
            .route = ks.newKernel("turboexl3_pf_t128_route", &froute_in, &froute_t128_out, text_froute_t128, header_small, true) orelse return error.KernelCreateFailed,
            .gu_mixed = ks.newKernel("turboexl3_pf_t128_gum", &gemm2_in, &gemm2_out, gemm2m_text, header_nax, true) orelse return error.KernelCreateFailed,
            .gu_t128 = ks.newKernel("turboexl3_pf_t128_gut", &gemm2_in, &gemm2_out, gemm2_text, header_nax, true) orelse return error.KernelCreateFailed,
            .dn_mixed = ks.newKernel("turboexl3_pf_t128_dnm", &gemm1_in, &gemm1_out, gemm1m_text, header_nax, true) orelse return error.KernelCreateFailed,
            .dn_t128 = ks.newKernel("turboexl3_pf_t128_dnt", &gemm1_in, &gemm1_out, gemm1_text, header_nax, true) orelse return error.KernelCreateFailed,
            .mid = ks.newKernel("turboexl3_pf_t128_mid", &mid_t128_in, &mid_out, text_mid_t128, header_small, true) orelse return error.KernelCreateFailed,
            .fin = ks.newKernel("turboexl3_pf_t128_fin", &fin_t128_in, &fin_out, text_finish_t128, header_small, true) orelse return error.KernelCreateFailed,
        };
    }

    fn deinit(k: T128Kernels) void {
        inline for (.{ k.route, k.gu_mixed, k.gu_t128, k.dn_mixed, k.dn_t128, k.mid, k.fin }) |x| _ = mlx.mlx_fast_metal_kernel_free(x);
    }
};

/// The compiled kernels of one prefill geometry and option set; launch configs are built per call (their sizes follow the
/// row count, the templates do not, so a new prompt length compiles nothing). The stages are public so the benchmark can
/// time each alone; `moe` runs them in order.
pub const Prefill = struct {
    geo: Geometry,
    opts: Options,
    table: mlx.mlx_fast_metal_kernel,
    prep_k: mlx.mlx_fast_metal_kernel,
    gemm2: mlx.mlx_fast_metal_kernel,
    gemm1: mlx.mlx_fast_metal_kernel,
    mid_k: mlx.mlx_fast_metal_kernel,
    fin: mlx.mlx_fast_metal_kernel,
    gemm2s: mlx.mlx_fast_metal_kernel,
    gemm1s: mlx.mlx_fast_metal_kernel,
    table_m: mlx.mlx_fast_metal_kernel,
    gemm2m: mlx.mlx_fast_metal_kernel,
    gemm1m: mlx.mlx_fast_metal_kernel,
    froute: mlx.mlx_fast_metal_kernel,
    froute_m: mlx.mlx_fast_metal_kernel,
    prep_v: mlx.mlx_fast_metal_kernel,
    mid_v: mlx.mlx_fast_metal_kernel,
    fin_v: mlx.mlx_fast_metal_kernel,
    /// the 128-row tiles' kernels (`moeT128`); made only with `opts.t128`
    t128_k: ?T128Kernels = null,

    pub fn init(geo: Geometry, opts: Options) !Prefill {
        if (!geo.admitted() or !opts.valid() or (opts.intr and geo.rate.n % 16 != 0) or (opts.defer_affine and geo.dec.codebook != .mul1) or
            (geo.hidden / BN) % opts.ncb != 0) return error.BadExl3Shape;
        return .{
            .geo = geo,
            .opts = opts,
            .table = ks.newKernel("turboexl3_pf_table", &table_in, &table_out, text_table, header_small, true) orelse return error.KernelCreateFailed,
            .prep_k = ks.newKernel("turboexl3_pf_prep", &prep_in, &prep_out, text_prep, header_small, true) orelse return error.KernelCreateFailed,
            .gemm2 = ks.newKernel("turboexl3_pf_gemm2", &gemm2_in, &gemm2_out, gemm2_text, header_nax, true) orelse return error.KernelCreateFailed,
            .gemm1 = ks.newKernel("turboexl3_pf_gemm1", &gemm1_in, &gemm1_out, gemm1_text, header_nax, true) orelse return error.KernelCreateFailed,
            .mid_k = ks.newKernel("turboexl3_pf_mid", &mid_in, &mid_out, text_mid, header_small, true) orelse return error.KernelCreateFailed,
            .fin = ks.newKernel("turboexl3_pf_fin", &fin_in, &fin_out, text_finish, header_small, true) orelse return error.KernelCreateFailed,
            .gemm2s = ks.newKernel("turboexl3_pf_gemm2s", &gemm2_in, &gemm2_out, gemm2s_text, header_nax_small, true) orelse return error.KernelCreateFailed,
            .gemm1s = ks.newKernel("turboexl3_pf_gemm1s", &gemm1_in, &gemm1_out, gemm1s_text, header_nax_small, true) orelse return error.KernelCreateFailed,
            .table_m = ks.newKernel("turboexl3_pf_table_m", &table_in, &table_out, text_table_mixed, header_small, true) orelse return error.KernelCreateFailed,
            .gemm2m = ks.newKernel("turboexl3_pf_gemm2m", &gemm2_in, &gemm2_out, gemm2m_text, header_nax_mixed, true) orelse return error.KernelCreateFailed,
            .gemm1m = ks.newKernel("turboexl3_pf_gemm1m", &gemm1_in, &gemm1_out, gemm1m_text, header_nax_mixed, true) orelse return error.KernelCreateFailed,
            .froute = ks.newKernel("turboexl3_pf_froute", &froute_in, &froute_out, text_froute, header_small, true) orelse return error.KernelCreateFailed,
            .froute_m = ks.newKernel("turboexl3_pf_froute_m", &froute_in, &froute_out, text_froute_m, header_small, true) orelse return error.KernelCreateFailed,
            .prep_v = ks.newKernel("turboexl3_pf_prep_v", &prep_in, &prep_out, text_prep_v, header_small, true) orelse return error.KernelCreateFailed,
            .mid_v = ks.newKernel("turboexl3_pf_mid_v", &mid_in, &mid_out, text_mid_v, header_small, true) orelse return error.KernelCreateFailed,
            .fin_v = ks.newKernel("turboexl3_pf_fin_v", &fin_in, &fin_out, text_finish_v, header_small, true) orelse return error.KernelCreateFailed,
            .t128_k = if (!opts.t128) null else try T128Kernels.init(),
        };
    }

    pub fn deinit(self: *Prefill) void {
        inline for (.{ self.table, self.prep_k, self.gemm2, self.gemm1, self.mid_k, self.fin, self.gemm2s, self.gemm1s, self.table_m, self.gemm2m, self.gemm1m, self.froute, self.froute_m, self.prep_v, self.mid_v, self.fin_v }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
        if (self.t128_k) |k| k.deinit();
    }

    fn decTempl(self: *const Prefill) [3]ks.TemplateArg {
        return .{
            .{ .name = "NHW", .value = .{ .int = @intCast(self.geo.rate.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(self.geo.dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(self.geo.dec.window.bits()) } },
        };
    }

    fn run(k: mlx.mlx_fast_metal_kernel, l: ks.Launch, ins: []const mlx.mlx_array, outs: []mlx.mlx_array, s: mlx.mlx_stream) !void {
        const c = try ks.buildConfig(l);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try ks.applyConfig(k, c, ins, outs, s);
    }

    /// argsort of the flat expert ids, then the M tile table (one threadgroup); `froute`: the one-launch route kernel.
    pub fn route(self: *Prefill, s: mlx.mlx_stream, inds: mlx.mlx_array, S: u32) !Route {
        const g = self.geo;
        const tr = self.opts.tileRows();
        const tmax: u32 = (S + tr - 1) / tr + g.experts;
        const mixed = self.opts.small == .mixed;
        const rows_t: u32 = if (mixed) tmax + self.opts.split * g.experts else tmax;
        const tmpl = [_]ks.TemplateArg{
            .{ .name = "EXP", .value = .{ .int = @intCast(g.experts) } },
            .{ .name = "BMT", .value = .{ .int = @intCast(tr) } },
            .{ .name = "SPL", .value = .{ .int = self.opts.split } },
        };
        if (self.opts.froute) {
            var outs: [2]mlx.mlx_array = undefined;
            try run(if (mixed) self.froute_m else self.froute, .{ .grid = .{ 1024, 1, 1 }, .threadgroup = .{ 1024, 1, 1 }, .out_shapes = &.{ &[_]c_int{@intCast(S)}, &[_]c_int{ @intCast(rows_t), 4 } }, .out_dtypes = &.{ .uint32, .int32 }, .template = &tmpl }, &.{inds}, &outs, s);
            return .{ .order = outs[0], .tiles = outs[1], .tmax = tmax };
        }
        var order = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(order);
        try mlx.check(mlx.mlx_argsort_axis(&order, inds, 0, s));
        var tiles: [1]mlx.mlx_array = undefined;
        try run(if (mixed) self.table_m else self.table, .{ .grid = .{ 1024, 1, 1 }, .threadgroup = .{ 1024, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows_t), 4 }}, .out_dtypes = &.{.int32}, .template = &tmpl }, &.{ inds, order }, &tiles, s);
        return .{ .order = order, .tiles = tiles[0], .tmax = tmax };
    }

    /// xg, xu half [S, hidden] in sorted order, inv u32 [S], and xsg, xsu f32 [S, hidden / 128]: each 128-block's sum of the
    /// rounded inputs (the deferred MUL1 map's C term).
    pub fn prep(self: *Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, inds: mlx.mlx_array, order: mlx.mlx_array, b: Operands, S: u32) ![5]mlx.mlx_array {
        const g = self.geo;
        const H: c_int = @intCast(g.hidden);
        const bpt = bptFor(g.hidden / 128);
        var po: [5]mlx.mlx_array = undefined;
        const sums = [_]c_int{ @intCast(S), @intCast(g.hidden / 128) };
        if (self.opts.aux != 0) {
            const asg: u32 = self.opts.aux;
            const tgs = (S * (g.hidden / 128) + asg - 1) / asg;
            try run(self.prep_v, .{ .grid = .{ tgs * 32 * asg, 1, 1 }, .threadgroup = .{ 32 * asg, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(S), H }, &[_]c_int{ @intCast(S), H }, &[_]c_int{@intCast(S)}, &sums, &sums }, .out_dtypes = &.{ .float16, .float16, .uint32, .float32, .float32 }, .template = &.{
                .{ .name = "HID", .value = .{ .int = H } },
                .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
                .{ .name = "ASG", .value = .{ .int = @intCast(asg) } },
                .{ .name = "XSO", .value = .{ .int = @intFromBool(self.opts.aux_lane_sums) } },
            } }, &.{ x, inds, order, b.gate_suh, b.up_suh }, &po, s);
            return po;
        }
        try run(self.prep_k, .{ .grid = .{ (g.hidden / 128 / bpt) * 32 * bpt, S, 1 }, .threadgroup = .{ 32 * bpt, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(S), H }, &[_]c_int{ @intCast(S), H }, &[_]c_int{@intCast(S)}, &sums, &sums }, .out_dtypes = &.{ .float16, .float16, .uint32, .float32, .float32 }, .template = &.{
            .{ .name = "HID", .value = .{ .int = H } },
            .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
            .{ .name = "BPT", .value = .{ .int = @intCast(bpt) } },
            .{ .name = "ALAY", .value = .{ .int = self.opts.alay } },
        } }, &.{ x, inds, order, b.gate_suh, b.up_suh }, &po, s);
        return po;
    }

    /// A GEMM launch over `tmax` M tiles, K stages `bks` deep and `ncb` output blocks per threadgroup; the template arguments
    /// live in the caller's `tmpl` (the launch only points at them).
    fn gemmCfg(self: *const Prefill, tmpl: *[23]ks.TemplateArg, tmax: u32, kd: u32, nd: u32, nops: u32, bks: u32, ncb: u32) ks.Launch {
        const t = self.decTempl();
        // both bodies: the A layout (`Options.alay`) and the first-fragment-row MMA (`Options.half_rows`)
        const alay_t = ks.TemplateArg{ .name = "ALAY", .value = .{ .int = self.opts.alay } };
        const h16_t = ks.TemplateArg{ .name = "H16", .value = .{ .int = @intFromBool(self.opts.half_rows) } };
        if (self.opts.small == .all) {
            const sgt = self.opts.sgt;
            tmpl.* = .{
                t[0], t[1], t[2],
                .{ .name = "KD", .value = .{ .int = @intCast(kd) } },
                .{ .name = "ND", .value = .{ .int = @intCast(nd) } },
                .{ .name = "NOPS", .value = .{ .int = @intCast(nops) } },
                .{ .name = "SGT", .value = .{ .int = @intCast(sgt) } },
                .{ .name = "DEFER", .value = .{ .int = if (self.opts.defer_affine) 1 else 0 } },
                .{ .name = "NBX", .value = .{ .int = @intCast(kd / 128) } },
                .{ .name = "SINTR", .value = .{ .int = if (self.opts.sintr) 1 else 0 } },
                alay_t, h16_t,
                undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined,
            };
            const items = tmax * nops * (nd / 32);
            return .{ .grid = .{ ((items + sgt - 1) / sgt) * 32 * sgt, 1, 1 }, .threadgroup = .{ 32 * sgt, 1, 1 }, .out_shapes = &.{}, .out_dtypes = &.{}, .template = tmpl[0..12] };
        }
        const threads: u32 = 2 * self.opts.bm + 32 * @as(u32, self.opts.wsp);
        tmpl.* = .{
            t[0], t[1], t[2],
            .{ .name = "KD", .value = .{ .int = @intCast(kd) } },
            .{ .name = "ND", .value = .{ .int = @intCast(nd) } },
            .{ .name = "NOPS", .value = .{ .int = @intCast(nops) } },
            .{ .name = "BMR", .value = .{ .int = @intCast(self.opts.bm) } },
            .{ .name = "DBUF", .value = .{ .int = if (self.opts.dbuf) 1 else 0 } },
            .{ .name = "PROBE", .value = .{ .int = self.opts.probe } },
            .{ .name = "WSP", .value = .{ .int = self.opts.wsp } },
            .{ .name = "MFIRST", .value = .{ .int = if (self.opts.mma_first) 1 else 0 } },
            .{ .name = "UNR", .value = .{ .int = if (self.opts.unroll) 1 else 0 } },
            .{ .name = "SGT", .value = .{ .int = @intCast(threads / 32) } },
            .{ .name = "TMAXS", .value = .{ .int = @intCast(self.opts.split * self.geo.experts) } },
            .{ .name = "FRAGB", .value = .{ .int = if (self.opts.fragb) 1 else 0 } },
            .{ .name = "INTR", .value = .{ .int = if (self.opts.intr) 1 else 0 } },
            .{ .name = "DEFER", .value = .{ .int = if (self.opts.defer_affine) 1 else 0 } },
            .{ .name = "NBX", .value = .{ .int = @intCast(kd / 128) } },
            .{ .name = "BKS", .value = .{ .int = @intCast(bks) } },
            .{ .name = "NCB", .value = .{ .int = @intCast(ncb) } },
            // the mixed launch's register-path half (the shared-decode body does not read it)
            .{ .name = "SINTR", .value = .{ .int = if (self.opts.sintr) 1 else 0 } },
            alay_t,
            h16_t,
        };
        // shared-decode threadgroups: ncb output blocks of one (M tile, op) each
        const big_tgs = tmax * nops * (nd / BN / ncb);
        if (self.opts.small == .mixed) {
            const nsg = threads / 32;
            const small_tgs = (self.opts.split * self.geo.experts * nops * (nd / 32) + nsg - 1) / nsg;
            return .{ .grid = .{ (big_tgs + small_tgs) * threads, 1, 1 }, .threadgroup = .{ threads, 1, 1 }, .out_shapes = &.{}, .out_dtypes = &.{}, .template = tmpl };
        }
        return .{ .grid = .{ big_tgs * threads, 1, 1 }, .threadgroup = .{ threads, 1, 1 }, .out_shapes = &.{}, .out_dtypes = &.{}, .template = tmpl };
    }

    /// the gate and up inner products, half [S, inter] each (sorted order)
    pub fn gemmGU(self: *Prefill, s: mlx.mlx_stream, r: Route, xg: mlx.mlx_array, xu: mlx.mlx_array, xsg: mlx.mlx_array, xsu: mlx.mlx_array, b: Operands, S: u32) ![2]mlx.mlx_array {
        const g = self.geo;
        var tmpl: [23]ks.TemplateArg = undefined;
        var l = self.gemmCfg(&tmpl, r.tmax, g.hidden, g.inter, 2, BK, 1);
        const I: c_int = @intCast(g.inter);
        const shape = [_]c_int{ @intCast(S), I };
        l.out_shapes = &.{ &shape, &shape };
        l.out_dtypes = &.{ .float16, .float16 };
        var z: [2]mlx.mlx_array = undefined;
        try run(switch (self.opts.small) { .all => self.gemm2s, .mixed => self.gemm2m, .off => self.gemm2 }, l, &.{ r.tiles, xg, xu, b.gate_t, b.up_t, xsg, xsu }, &z, s);
        return z;
    }

    /// gate|up finish, SwiGLU and the down input: xd half [S, inter] and xsd f32 [S, inter / 128] (its per-block sums)
    pub fn mid(self: *Prefill, s: mlx.mlx_stream, zg: mlx.mlx_array, zu: mlx.mlx_array, inds: mlx.mlx_array, order: mlx.mlx_array, b: Operands, S: u32) ![2]mlx.mlx_array {
        const g = self.geo;
        const I: c_int = @intCast(g.inter);
        const bpt = bptFor(g.inter / 128);
        var xd: [2]mlx.mlx_array = undefined;
        if (self.opts.aux != 0) {
            const asg: u32 = self.opts.aux;
            const tgs = (S * (g.inter / 128) + asg - 1) / asg;
            try run(self.mid_v, .{ .grid = .{ tgs * 32 * asg, 1, 1 }, .threadgroup = .{ 32 * asg, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(S), I }, &[_]c_int{ @intCast(S), @intCast(g.inter / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = &.{
                .{ .name = "INTER", .value = .{ .int = I } },
                .{ .name = "ASG", .value = .{ .int = @intCast(asg) } },
                .{ .name = "XSO", .value = .{ .int = @intFromBool(self.opts.aux_lane_sums) } },
            } }, &.{ zg, zu, inds, order, b.gate_svh, b.up_svh, b.down_suh }, &xd, s);
            return xd;
        }
        try run(self.mid_k, .{ .grid = .{ (g.inter / 128 / bpt) * 32 * bpt, S, 1 }, .threadgroup = .{ 32 * bpt, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(S), I }, &[_]c_int{ @intCast(S), @intCast(g.inter / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = &.{
            .{ .name = "INTER", .value = .{ .int = I } },
            .{ .name = "BPT", .value = .{ .int = @intCast(bpt) } },
            .{ .name = "ALAY", .value = .{ .int = self.opts.alay } },
        } }, &.{ zg, zu, inds, order, b.gate_svh, b.up_svh, b.down_suh }, &xd, s);
        return xd;
    }

    /// the down inner product, half [S, hidden] (sorted order); `Options.bk128` / `ncb` shape its shared-decode body
    pub fn gemmD(self: *Prefill, s: mlx.mlx_stream, r: Route, xd: mlx.mlx_array, xsd: mlx.mlx_array, b: Operands, S: u32) !mlx.mlx_array {
        const g = self.geo;
        var tmpl: [23]ks.TemplateArg = undefined;
        var l = self.gemmCfg(&tmpl, r.tmax, g.inter, g.hidden, 1, self.opts.downStage(), self.opts.ncb);
        const shape = [_]c_int{ @intCast(S), @intCast(g.hidden) };
        l.out_shapes = &.{&shape};
        l.out_dtypes = &.{.float16};
        var z: [1]mlx.mlx_array = undefined;
        try run(switch (self.opts.small) { .all => self.gemm1s, .mixed => self.gemm1m, .off => self.gemm1 }, l, &.{ r.tiles, xd, b.down_t, xsd }, &z, s);
        return z[0];
    }

    /// the down finish and the slot reduce: y [rows, hidden] in `out_dtype`
    pub fn finish(self: *Prefill, s: mlx.mlx_stream, zd: mlx.mlx_array, inds: mlx.mlx_array, inv: mlx.mlx_array, scores: mlx.mlx_array, b: Operands, rows: usize, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
        const g = self.geo;
        const H: c_int = @intCast(g.hidden);
        const bpt = bptFor(g.hidden / 128);
        var y: [1]mlx.mlx_array = undefined;
        if (self.opts.aux != 0) {
            const asg: u32 = self.opts.aux;
            const items: u32 = @intCast(rows * (g.hidden / 128));
            const tgs = (items + asg - 1) / asg;
            try run(self.fin_v, .{ .grid = .{ tgs * 32 * asg, 1, 1 }, .threadgroup = .{ 32 * asg, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{out_dtype}, .template = &.{
                .{ .name = "HID", .value = .{ .int = H } },
                .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
                .{ .name = "ASG", .value = .{ .int = @intCast(asg) } },
                .{ .name = "OUT_T", .value = .{ .dtype = out_dtype } },
            } }, &.{ zd, inds, inv, scores, b.down_svh }, &y, s);
            return y[0];
        }
        try run(self.fin, .{ .grid = .{ (g.hidden / 128 / bpt) * 32 * bpt, @intCast(rows), 1 }, .threadgroup = .{ 32 * bpt, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{out_dtype}, .template = &.{
            .{ .name = "HID", .value = .{ .int = H } },
            .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
            .{ .name = "BPT", .value = .{ .int = @intCast(bpt) } },
            .{ .name = "OUT_T", .value = .{ .dtype = out_dtype } },
        } }, &.{ zd, inds, inv, scores, b.down_svh }, &y, s);
        return y[0];
    }

    /// y [rows, hidden] in `out_dtype` for x [rows, hidden], inds / scores [rows * topk] (the host's dtypes).
    pub fn moe(self: *Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, b: Operands, rows: usize, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
        const S: u32 = @intCast(rows * self.geo.topk);
        const r = try self.route(s, inds, S);
        defer r.deinit();
        const po = try self.prep(s, x, inds, r.order, b, S);
        defer for (po) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const zgu = try self.gemmGU(s, r, po[0], po[1], po[3], po[4], b, S);
        defer for (zgu) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const xd = try self.mid(s, zgu[0], zgu[1], inds, r.order, b, S);
        defer for (xd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const zd = try self.gemmD(s, r, xd[0], xd[1], b, S);
        defer _ = mlx.mlx_array_free(zd);
        return self.finish(s, zd, inds, po[2], scores, b, rows, out_dtype);
    }

    /// The 128-row tiles' route: `froute`'s count and scatter (the same `order`), then the 128-row tiles, the mixed table over
    /// the remainders and each expert's first remainder position (`text_froute_t128`), one launch.
    pub fn routeT128(self: *Prefill, s: mlx.mlx_stream, inds: mlx.mlx_array, S: u32) !RouteT128 {
        const k = self.t128_k orelse return error.Exl3EngineNotImplemented;
        const g = self.geo;
        const tr = self.opts.tileRows();
        const tmax = t128MixedRows(S, g.experts, tr, self.opts.split, t128_cut, true);
        const rows_t: u32 = tmax + self.opts.split * g.experts;
        const tq = t128TileRows(S, g.experts, t128_cut, true);
        const tmpl = [_]ks.TemplateArg{
            .{ .name = "EXP", .value = .{ .int = @intCast(g.experts) } },
            .{ .name = "BMT", .value = .{ .int = @intCast(tr) } },
            .{ .name = "SPL", .value = .{ .int = self.opts.split } },
            .{ .name = "CUT", .value = .{ .int = t128_cut } },
            .{ .name = "T128", .value = .{ .int = 1 } },
        };
        var outs: [4]mlx.mlx_array = undefined;
        try run(k.route, .{ .grid = .{ 1024, 1, 1 }, .threadgroup = .{ 1024, 1, 1 }, .out_shapes = &.{ &[_]c_int{@intCast(S)}, &[_]c_int{ @intCast(rows_t), 4 }, &[_]c_int{ @intCast(tq), 4 }, &[_]c_int{@intCast(g.experts)} }, .out_dtypes = &.{ .uint32, .int32, .int32, .int32 }, .template = &tmpl }, &.{inds}, &outs, s);
        return .{ .order = outs[0], .tiles = outs[1], .tiles128 = outs[2], .xbase = outs[3], .tmax = tmax, .tq = tq };
    }

    /// The 128-row tiles' launch: `gemmCfg`'s shared-decode body at bm 128 (8 simdgroups, the same stage and K loop), over
    /// `tq` table rows
    fn t128Cfg(self: *const Prefill, tmpl: *[23]ks.TemplateArg, tq: u32, kd: u32, nd: u32, nops: u32, bks: u32, ncb: u32) ks.Launch {
        var p = self.*;
        p.opts.bm = 128;
        p.opts.small = .off;
        return p.gemmCfg(tmpl, tq, kd, nd, nops, bks, ncb);
    }

    /// A GEMM's outputs ([S, nd] f16 per op, `ZT128`): the mixed launch's over the remainder table, then the bm-128 launch's
    /// over the 128-row tiles (so the register-path tail at the mixed launch's end overlaps the 128-row tiles' start), each
    /// an output of its own launch: no data between the two, so no barrier between them, and MLX orders every reader after
    /// both. `ins_m` / `ins_t` are the launches' inputs.
    fn t128Gemm(self: *Prefill, s: mlx.mlx_stream, rt: RouteT128, k_m: mlx.mlx_fast_metal_kernel, k_t: mlx.mlx_fast_metal_kernel, nops: u32, kd: u32, nd: u32, bks: u32, ncb: u32, ins_m: []const mlx.mlx_array, ins_t: []const mlx.mlx_array, S: u32) !ZT128 {
        const shape = [_]c_int{ @intCast(S), @intCast(nd) };
        const shapes = [_][]const c_int{ &shape, &shape };
        const dtypes = [_]mlx.mlx_dtype{ .float16, .float16 };
        var x: ZT128 = .{ .m = undefined, .t = undefined, .n = @intCast(nops) };
        var tmpl: [23]ks.TemplateArg = undefined;
        var lm = self.gemmCfg(&tmpl, rt.tmax, kd, nd, nops, bks, ncb);
        lm.out_shapes = shapes[0..nops];
        lm.out_dtypes = dtypes[0..nops];
        try run(k_m, lm, ins_m, x.m[0..nops], s);
        errdefer for (x.m[0..nops]) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var tmpl_t: [23]ks.TemplateArg = undefined;
        var lt = self.t128Cfg(&tmpl_t, rt.tq, kd, nd, nops, bks, ncb);
        lt.out_shapes = shapes[0..nops];
        lt.out_dtypes = dtypes[0..nops];
        try run(k_t, lt, ins_t, x.t[0..nops], s);
        return x;
    }

    /// The gate and up inner products, half [S, inter] each (sorted order, `ZT128`): `gemmGU`'s per-output MMAs from the
    /// 128-row tiles and the mixed tiles.
    pub fn gemmGUT128(self: *Prefill, s: mlx.mlx_stream, rt: RouteT128, xg: mlx.mlx_array, xu: mlx.mlx_array, xsg: mlx.mlx_array, xsu: mlx.mlx_array, b: Operands, S: u32) !ZT128 {
        const k = self.t128_k orelse return error.Exl3EngineNotImplemented;
        const g = self.geo;
        return self.t128Gemm(s, rt, k.gu_mixed, k.gu_t128, 2, g.hidden, g.inter, BK, 1, &.{ rt.tiles, xg, xu, b.gate_t, b.up_t, xsg, xsu }, &.{ rt.tiles128, xg, xu, b.gate_t, b.up_t, xsg, xsu }, S);
    }

    /// `mid`'s strided launch with `text_mid_t128` (each position's rows from the launch that computed them)
    pub fn midT128(self: *Prefill, s: mlx.mlx_stream, zt: ZT128, rt: RouteT128, inds: mlx.mlx_array, b: Operands, S: u32) ![2]mlx.mlx_array {
        const k = self.t128_k orelse return error.Exl3EngineNotImplemented;
        const g = self.geo;
        const I: c_int = @intCast(g.inter);
        const bpt = bptFor(g.inter / 128);
        var xd: [2]mlx.mlx_array = undefined;
        try run(k.mid, .{ .grid = .{ (g.inter / 128 / bpt) * 32 * bpt, S, 1 }, .threadgroup = .{ 32 * bpt, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(S), I }, &[_]c_int{ @intCast(S), @intCast(g.inter / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = &.{
            .{ .name = "INTER", .value = .{ .int = I } },
            .{ .name = "BPT", .value = .{ .int = @intCast(bpt) } },
            .{ .name = "ALAY", .value = .{ .int = self.opts.alay } },
        } }, &.{ zt.m[0], zt.m[1], inds, rt.order, b.gate_svh, b.up_svh, b.down_suh, zt.t[0], zt.t[1], rt.xbase }, &xd, s);
        return xd;
    }

    /// The down inner product, half [S, hidden] (sorted order, `ZT128` with one op), as `gemmGUT128`
    pub fn gemmDT128(self: *Prefill, s: mlx.mlx_stream, rt: RouteT128, xd: mlx.mlx_array, xsd: mlx.mlx_array, b: Operands, S: u32) !ZT128 {
        const k = self.t128_k orelse return error.Exl3EngineNotImplemented;
        const g = self.geo;
        return self.t128Gemm(s, rt, k.dn_mixed, k.dn_t128, 1, g.inter, g.hidden, self.opts.downStage(), self.opts.ncb, &.{ rt.tiles, xd, b.down_t, xsd }, &.{ rt.tiles128, xd, b.down_t, xsd }, S);
    }

    /// `finish`'s strided launch with `text_finish_t128` (each slot's row from the launch that computed it)
    pub fn finishT128(self: *Prefill, s: mlx.mlx_stream, zt: ZT128, xbase: mlx.mlx_array, inds: mlx.mlx_array, inv: mlx.mlx_array, scores: mlx.mlx_array, b: Operands, rows: usize, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
        const k = self.t128_k orelse return error.Exl3EngineNotImplemented;
        const g = self.geo;
        const H: c_int = @intCast(g.hidden);
        const bpt = bptFor(g.hidden / 128);
        var y: [1]mlx.mlx_array = undefined;
        try run(k.fin, .{ .grid = .{ (g.hidden / 128 / bpt) * 32 * bpt, @intCast(rows), 1 }, .threadgroup = .{ 32 * bpt, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{out_dtype}, .template = &.{
            .{ .name = "HID", .value = .{ .int = H } },
            .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
            .{ .name = "BPT", .value = .{ .int = @intCast(bpt) } },
            .{ .name = "OUT_T", .value = .{ .dtype = out_dtype } },
        } }, &.{ zt.m[0], inds, inv, scores, b.down_svh, zt.t[0], xbase }, &y, s);
        return y[0];
    }

    /// `moe`'s y through the 128-row tiles (`routeT128`, `prep`, `gemmGUT128`, `midT128`, `gemmDT128`, `finishT128`): its
    /// bytes. Refused (error.Exl3EngineNotImplemented, nothing launched) without `opts.t128`.
    pub fn moeT128(self: *Prefill, s: mlx.mlx_stream, x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, b: Operands, rows: usize, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
        if (self.t128_k == null) return error.Exl3EngineNotImplemented;
        const S: u32 = @intCast(rows * self.geo.topk);
        const rt = try self.routeT128(s, inds, S);
        defer rt.deinit();
        const po = try self.prep(s, x, inds, rt.order, b, S);
        defer for (po) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const zgu = try self.gemmGUT128(s, rt, po[0], po[1], po[3], po[4], b, S);
        defer zgu.deinit();
        const xd = try self.midT128(s, zgu, rt, inds, b, S);
        defer for (xd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const zd = try self.gemmDT128(s, rt, xd[0], xd[1], b, S);
        defer zd.deinit();
        return self.finishT128(s, zd, rt.xbase, inds, po[2], scores, b, rows, out_dtype);
    }
};

test "turboexl3 prefill texts: the NAX header carries steel's tiles and turboexl3's decoder; the GEMM's second-op names resolve" {
    const t = std.testing;
    try t.expect(std.mem.indexOf(u8, header_nax, "struct NAXTile") != null);
    try t.expect(std.mem.indexOf(u8, header_nax, "exl3_gen_cw") != null);
    try t.expect(std.mem.indexOf(u8, gemm2_text, "XSECOND") == null);
    try t.expect(std.mem.indexOf(u8, gemm2_text, "op == 0 ? x0 : x1") != null);
    try t.expect(std.mem.indexOf(u8, gemm1_text, "op == 0 ? x0 : x0") != null);
    try t.expect(bptFor(20) == 5 and bptFor(5) == 5 and bptFor(7) == 7 and bptFor(16) == 8);
}

test "turboexl3 prefill down options: bk128 doubles the stage (32 KB double-buffered in fragment order); ncb is 1, 2 or 4" {
    const t = std.testing;
    const served: Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .defer_affine = true };
    try t.expect(served.stageBytes(served.downStage()) == 16384);
    var o = served;
    o.bk128 = true;
    try t.expect(o.valid() and o.stageBytes(o.downStage()) == 32768 and o.stageBytes(BK) == 16384);
    o.ncb = 4;
    try t.expect(o.valid());
    o.ncb = 3;
    try t.expect(!o.valid());
    try t.expect(!(Options{ .dbuf = true, .bk128 = true }).valid()); // the padded B^T sub-stages: 36 KB
    try t.expect(!(Options{ .dbuf = true, .fragb = true, .bk128 = true, .probe = 16 }).valid());
    try t.expect((Options{ .bk128 = true }).valid() and (Options{ .bk128 = true }).stageBytes(128) == 18432); // single-buffered
    try t.expect((Options{ .fragb = true, .bk128 = true, .ncb = 2 }).valid());
    try t.expect(!(Options{ .small = .all, .ncb = 2 }).valid() and !(Options{ .small = .all, .fragb = true, .bk128 = true }).valid());
}

test "turboexl3 prefill alay: a lane's A fragment elements are one aligned 16-byte (32-deep step) or 8-byte (16-deep slice) run; the writers' slot is alayPos / alayPos16" {
    const t = std.testing;
    // each a permutation of a 32-block; alayPos16 keeps each 16-block's elements in it
    var seen: [32]bool = @splat(false);
    var seen16: [32]bool = @splat(false);
    for (0..32) |j_| {
        const j: u32 = @intCast(j_);
        const p = alayPos(j);
        const p16 = alayPos16(j);
        try t.expect(p < 32 and !seen[p] and p16 < 32 and !seen16[p16] and p16 / 16 == j / 16);
        seen[p] = true;
        seen16[p16] = true;
    }
    // the fragment rule both GEMM bodies share (exl3pf_afrag; the register path's lane-plan test above): NAX lane L, q = 2 b3 +
    // b0, holds at element e of its fragment row for 16-deep slice kk the input k 16 kk + 2q + {0, 1, 8, 9}[e]. Stored at
    // alayPos that element is half 8q + 4 kk + e of the 32-block: the lane's eight elements of both slices are halfs 8q .. 8q + 7,
    // one vec<half, 8> at the row pointer's lane offset 8q (exl3pf_afrag2; 16q bytes). Stored at alayPos16 it is half
    // 16 kk + 4q + e: the slice's four are one half4 at 16 kk + 4q (exl3pf_afrag4; 8q bytes). Rows of an admitted width (a
    // multiple of 128 halfs) and the blocks (64 / 32 bytes) keep both aligned.
    const kof = [_]u32{ 0, 1, 8, 9 };
    for (0..32) |l_| {
        const l: u32 = @intCast(l_);
        const q = ((l >> 2) & 2) | (l & 1);
        for (0..2) |kk_| for (0..4) |e_| {
            const kk: u32 = @intCast(kk_);
            const e: u32 = @intCast(e_);
            const k = 16 * kk + 2 * q + kof[e];
            try t.expectEqual(8 * q + 4 * kk + e, alayPos(k));
            try t.expectEqual(16 * kk + 4 * q + e, alayPos16(k));
        };
    }
    // prep and mid write lane L's element L of each 32-block at turboexl3_alay(L) / turboexl3_alay16(L): the expressions of
    // alayPos / alayPos16
    try t.expect(std.mem.indexOf(u8, header_small, "turboexl3_alay(uint j) { return 8u * ((j >> 1u) & 3u) + 4u * (j >> 4u) + ((j & 1u) | (((j >> 3u) & 1u) << 1u)); }") != null);
    try t.expect(std.mem.indexOf(u8, header_small, "turboexl3_alay16(uint j) { return (j & 16u) + 4u * ((j >> 1u) & 3u) + ((j & 1u) | (((j >> 3u) & 1u) << 1u)); }") != null);
    for ([_][]const u8{ text_prep, text_mid }) |txt| try t.expect(std.mem.indexOf(u8, txt, "(ALAY == 1) ? turboexl3_alay(L) : ((ALAY == 2) ? turboexl3_alay16(L) : L)") != null);
    // both GEMM bodies (and so the mixed launch) read the layout and the first-fragment-row switch
    for ([_][]const u8{ gemm2m_text, gemm1m_text, gemm2s_text, gemm1s_text, gemm2_text, gemm1_text }) |txt| {
        try t.expect(std.mem.indexOf(u8, txt, "((ALAY == 1) ? 8u : ((ALAY == 2) ? 4u : 2u))") != null);
        try t.expect(std.mem.indexOf(u8, txt, "H16") != null);
    }
}

test "turboexl3 prefill alay / half_rows options: fragment order where the shared-decode body runs, the strided prep and mid" {
    const t = std.testing;
    const served: Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .sintr = true, .defer_affine = true, .ncb = 4, .froute = true };
    try t.expect(served.valid());
    for ([_]u8{ 1, 2 }) |mode| {
        var o = served;
        o.alay = mode;
        o.half_rows = true;
        try t.expect(o.valid());
        o.bm = 128;
        try t.expect(o.valid());
        // the padded bodies load A in natural order (and multiply whole fragments)
        try t.expect(!(Options{ .small = .mixed, .dbuf = true, .alay = mode }).valid());
        // the all-register path needs no stage layout
        try t.expect((Options{ .small = .all, .alay = mode, .half_rows = true }).valid());
        // the vector-layout prep / mid write natural order
        var v = served;
        v.alay = mode;
        v.aux = 8;
        try t.expect(!v.valid());
    }
    try t.expect(!(Options{ .dbuf = true, .unroll = true, .half_rows = true }).valid());
    var o3 = served;
    o3.alay = 3;
    try t.expect(!o3.valid());
    var w = served;
    w.aux = 8;
    w.half_rows = true;
    try t.expect(w.valid());
}

test "turboexl3 prefill stage plan: a sub-stage fills its own BSZH halfs; the carried double buffer feeds every block its sub-stages in K order" {
    const t = std.testing;
    // one 64-deep sub-stage (the text's plan): decoding simdgroup d of DEC takes tiles j = d + DEC jj (kt = j >> 2, ntl = j & 3);
    // fragment order: lane L writes (ntl 4 + kt) 256 + 8 lnx(L), and the MMA's lane L reads tile (tn / 16 + nn, 2 i + kk) at
    // ((tn / 16 + nn) 4 + 2 i + kk) 256 + 8 L; padded: four half2 at d, d + 8, d + 8 LDB, d + 8 LDB + 8, d = (ntl 16 + c) LDB +
    // kt 16 + 2 q. Either way the sub-stage stays inside BSZH halfs, so a 128-deep buffer holds two side by side.
    for ([_]u32{ 2, 4, 8, 16 }) |dec| {
        var writer: [512]?[3]u32 = @splat(null); // fragment slot (8 halfs) -> (ntl, kt, lnx) of its writer
        var padded_end: u32 = 0;
        for (0..dec) |d| for (0..16 / dec) |jj| {
            const j: u32 = @intCast(d + dec * jj);
            const kt = j >> 2;
            const ntl = j & 3;
            for (0..32) |l_| {
                const l: u32 = @intCast(l_);
                const lnx = (((l >> 4) & 1) << 4) | (((l >> 1) & 1) << 3) | (((l >> 3) & 1) << 2) | (((l >> 2) & 1) << 1) | (l & 1);
                const slot = ((ntl * 4 + kt) * 256 + lnx * 8) / 8;
                try t.expect(writer[slot] == null);
                writer[slot] = .{ ntl, kt, lnx };
                padded_end = @max(padded_end, (ntl * 16 + (l >> 2)) * (BK + 8) + kt * 16 + 2 * (l & 3) + 8 * (BK + 8) + 8 + 2);
            }
        };
        for (writer) |w| try t.expect(w != null);
        try t.expect(padded_end <= BN * (BK + 8));
        for ([_]u32{ 0, 32 }) |tn| for (0..2) |nn| for (0..2) |i| for (0..2) |kk| for (0..32) |l| {
            const ntl: u32 = @intCast(tn / 16 + nn);
            const kt: u32 = @intCast(2 * i + kk);
            try t.expectEqual([3]u32{ ntl, kt, @intCast(l) }, writer[((ntl * 4 + kt) * 256 + 8 * l) / 8].?);
        };
    }
    // the double-buffered sequence: NCB blocks of STAGES stages of SUBS sub-stages; the prologue decodes block 0's sub-stages
    // 0 .. SUBS - 1 into buffer 0; step f = (cb, s) multiplies buffer f & 1's sub-stages SUBS s + h and decodes the next stage's
    // (s + 1 of cb, or 0 of cb + 1) into buffer (f + 1) & 1
    for ([_]u32{ 1, 2 }) |subs| for ([_]u32{ 1, 2, 4 }) |ncb| for ([_]u32{ 640, 2560 }) |kd| {
        const stages = kd / BK / subs;
        var buf: [2][2]?[2]u32 = .{ .{ null, null }, .{ null, null } };
        for (0..subs) |h| buf[0][h] = .{ 0, @intCast(h) };
        var next_k: [4]u32 = @splat(0); // per block: the next 64-deep sub-stage its accumulator must take
        for (0..ncb) |cb_| for (0..stages) |s_| {
            const cb: u32 = @intCast(cb_);
            const s: u32 = @intCast(s_);
            const f = cb * stages + s;
            const last = s + 1 == stages;
            for (0..subs) |h_| {
                const h: u32 = @intCast(h_);
                if (!last or cb + 1 < ncb) buf[(f + 1) & 1][h] = .{ if (last) cb + 1 else cb, subs * (if (last) 0 else s + 1) + h };
                const got = buf[f & 1][h].?;
                try t.expectEqual([2]u32{ cb, subs * s + h }, got);
                try t.expectEqual(next_k[cb], got[1]);
                next_k[cb] += 1;
            }
        };
        for (next_k[0..ncb]) |k| try t.expectEqual(kd / BK, k);
    };
}

test "turboexl3 prefill small: lane L decoding the trellis plan of L' = (b4,b2,b1,b3,b0) holds its NAX B^T fragment elements in order" {
    var perm: [format.TILE_VALUES]usize = undefined;
    format.tensorCorePerm(&perm); // code position p -> row-major tile index (row = input k, column = output n)
    const kof = [_]usize{ 0, 1, 8, 9 };
    for (0..32) |l| {
        const lp = smallLanePlan(@intCast(l));
        // BaseNAXFrag::get_coord(idx): row fm + (idx >> 2) * 8, column fn + idx % 4
        const qid = l >> 2;
        const fm = (qid & 4) | ((l >> 1) & 3);
        const fnc = ((qid & 2) | (l & 1)) * 4;
        for (0..8) |idx| {
            const n_frag = fm + (idx >> 2) * 8; // B^T fragment row = output n
            const slot = fnc + idx % 4; // fragment column = k-slot
            const k_true = 2 * (slot >> 2) + kof[slot & 3];
            const rm = perm[8 * lp + idx]; // weight idx of trellis lane lp
            try std.testing.expectEqual(k_true, rm / 16);
            try std.testing.expectEqual(n_frag, rm % 16);
        }
    }
    // L -> L' is a permutation of the 32 lanes
    var seen: [32]bool = @splat(false);
    for (0..32) |l| seen[smallLanePlan(@intCast(l))] = true;
    for (seen) |b| try std.testing.expect(b);
}

test "turboexl3 prefill small sintr: at every register-path lane the integer-rate reader picks the general reader's words, halfword and extract offset (n % 16 == 0)" {
    // lane.zig's Metal readers at lane L' = smallLanePlan(L), K = 1..8: the same start bit gives the same word qb and halfword a
    // (so the same IntWords) and lp = 8 b + K mod 8; window m then selects the same h[j] and extracts at the same offset
    var prng = std.Random.DefaultPrng.init(0x5147);
    const rnd = prng.random();
    var words: [format.Rate.max_n]u16 = undefined;
    var want: [format.TILE_VALUES]u16 = undefined;
    var gi: [8]u16 = undefined;
    var gg: [8]u16 = undefined;
    var k: u32 = 1;
    while (k <= 8) : (k += 1) {
        const rate = format.Rate.fromK(k);
        const n = rate.n;
        for (0..32) |l| {
            const lpl = smallLanePlan(@intCast(l));
            const ip = lane.intPlan(rate, lpl);
            const gp = lane.genPlan(rate, lpl);
            try std.testing.expectEqual(gp.qb, ip.qb);
            try std.testing.expectEqual(gp.a, ip.a);
            try std.testing.expectEqual(gp.lp, 8 * ip.b + k % 8);
            for (0..8) |m_| {
                const m: u32 = @intCast(m_);
                // exl3_int_cw: CM = K mod 8 + K m, h[(CM >> 4) + carry] with carry = b and (CM & 15) >= 8, offset (8 b + CM) & 15
                const cm = k % 8 + k * m;
                const ji = (cm >> 4) + @intFromBool(ip.b != 0 and (cm & 15) >= 8);
                const ri = (8 * ip.b + cm) & 15;
                // exl3_gen_cw: C_m = ((m + 1) N >> 4) - (N >> 4), pos = lp + C_m, h[pos >> 4] (J or J + 1), offset pos & 15
                const pos = gp.lp + ((((m + 1) * n) >> 4) - (n >> 4));
                try std.testing.expectEqual(pos >> 4, ji);
                try std.testing.expectEqual(pos & 15, ri);
                try std.testing.expect(ji <= 5); // an IntWords index: h[i] holds halfwords a + i, a + i + 1 of the four words
            }
        }
        for (0..4) |_| {
            for (words[0..n]) |*w| w.* = rnd.int(u16);
            format.unpackTile(words[0..n], rate, &want);
            for (0..32) |l| {
                const lpl = smallLanePlan(@intCast(l));
                lane.laneCodewordsInt(words[0..n], rate, lpl, &gi);
                lane.laneCodewordsGen(words[0..n], rate, lpl, &gg);
                try std.testing.expectEqualSlices(u16, &gg, &gi);
                try std.testing.expectEqualSlices(u16, want[8 * lpl ..][0..8], &gi);
            }
        }
    }
}

/// One kernel as a standalone Metal translation unit (the CPU-side compile check, TURBOEXL3_METAL_DUMP=1): the header, the
/// template ints and dtypes as file-scope constants, and a [[kernel]] whose arguments mirror what MLX binds (inputs in order,
/// `<name>_shape` where the source reads it, then outputs, then the thread attributes the sources use). Printed between
/// @@METAL_BEGIN / @@METAL_END markers; the box splits them and runs `xcrun metal` on each without touching the GPU.
pub const MetalArg = struct { name: []const u8, ty: []const u8, out: bool = false, shape: bool = false };

pub fn printMetal(name: []const u8, header: []const u8, consts: []const u8, args: []const MetalArg, body: []const u8) void {
    std.debug.print("@@METAL_BEGIN {s}\n{s}\n{s}\n[[kernel]] void {s}(", .{ name, header, consts, name });
    var b: usize = 0;
    for (args) |arg| {
        std.debug.print("{s}{s} {s}* {s} [[buffer({d})]]", .{ if (b == 0) "" else ", ", if (arg.out) "device" else "const device", arg.ty, arg.name, b });
        b += 1;
        if (arg.shape) {
            std.debug.print(", const constant int* {s}_shape [[buffer({d})]]", .{ arg.name, b });
            b += 1;
        }
    }
    std.debug.print(", uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]) {{\n{s}\n}}\n@@METAL_END\n", .{body});
}

test "turboexl3 prefill metal sources: every kernel as a standalone translation unit (TURBOEXL3_METAL_DUMP=1)" {
    _ = std.c.getenv("TURBOEXL3_METAL_DUMP") orelse return error.SkipZigTest;
    const gemm_c = "constant constexpr int NHW = 80; constant constexpr int CODEBOOK = 0; constant constexpr int WBITS = 16; constant constexpr int NOPS = 2; constant constexpr int BMR = 128; constant constexpr int DBUF = 1; constant constexpr int PROBE = 0; constant constexpr int WSP = 0; constant constexpr int MFIRST = 0; constant constexpr int UNR = 1; constant constexpr int SGT = 8; constant constexpr int TMAXS = 512; constant constexpr int FRAGB = 1; constant constexpr int INTR = 1; constant constexpr int DEFER = 1; constant constexpr int NBX = 20; constant constexpr int KD = 2560; constant constexpr int ND = 640; constant constexpr int BKS = 64; constant constexpr int NCB = 1; constant constexpr int SINTR = 0; constant constexpr int ALAY = 0; constant constexpr int H16 = 0;";
    // the register path through the integer-rate reader (j = Options.sintr), with the deferred MUL1 map and without it
    const gemm_j = comptime replaceOnce(gemm_c, "int SINTR = 0;", "int SINTR = 1;");
    const gemm_j0 = comptime replaceOnce(gemm_j, "int DEFER = 1;", "int DEFER = 0;");
    // the A layouts (l = Options.alay 1, l2 = alay 2) and the first-fragment-row MMA (e = Options.half_rows), each alone and
    // with e, on the served gate|up launch (bm 64, 4 simdgroups) and at bm 128; with the A-load probe (p4: the layout's zero
    // fill)
    const gemm_j64 = comptime replaceOnce(replaceOnce(gemm_j, "int BMR = 128;", "int BMR = 64;"), "int SGT = 8;", "int SGT = 4;");
    const gemm_jl = comptime replaceOnce(gemm_j64, "int ALAY = 0;", "int ALAY = 1;");
    const gemm_je = comptime replaceOnce(gemm_j64, "int H16 = 0;", "int H16 = 1;");
    const gemm_jle = comptime replaceOnce(gemm_jl, "int H16 = 0;", "int H16 = 1;");
    const gemm_jle128 = comptime replaceOnce(replaceOnce(gemm_c, "int ALAY = 0;", "int ALAY = 1;"), "int H16 = 0;", "int H16 = 1;");
    const gemm_jle_p4 = comptime replaceOnce(gemm_jle, "int PROBE = 0;", "int PROBE = 4;");
    const gemm_jl2e = comptime replaceOnce(gemm_je, "int ALAY = 0;", "int ALAY = 2;");
    const g2 = [_]MetalArg{ .{ .name = "tiles", .ty = "int", .shape = true }, .{ .name = "x0", .ty = "half" }, .{ .name = "x1", .ty = "half" }, .{ .name = "w0", .ty = "uint16_t" }, .{ .name = "w1", .ty = "uint16_t" }, .{ .name = "xs0", .ty = "float" }, .{ .name = "xs1", .ty = "float" }, .{ .name = "z0", .ty = "half", .out = true }, .{ .name = "z1", .ty = "half", .out = true } };
    const g1 = [_]MetalArg{ .{ .name = "tiles", .ty = "int", .shape = true }, .{ .name = "x0", .ty = "half" }, .{ .name = "w0", .ty = "uint16_t" }, .{ .name = "xs0", .ty = "float" }, .{ .name = "z0", .ty = "half", .out = true } };
    printMetal("turboexl3_pf_gemm2", header_nax, gemm_c, &g2, gemm2_text);
    printMetal("turboexl3_pf_gemm1", header_nax, gemm_c, &g1, gemm1_text);
    printMetal("turboexl3_pf_gemm2s", header_nax_small, gemm_c, &g2, gemm2s_text);
    printMetal("turboexl3_pf_gemm2m", header_nax_mixed, gemm_c, &g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm1m", header_nax_mixed, gemm_c, &g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm2s_j", header_nax_small, gemm_j, &g2, gemm2s_text);
    printMetal("turboexl3_pf_gemm2s_j0", header_nax_small, gemm_j0, &g2, gemm2s_text);
    printMetal("turboexl3_pf_gemm2m_j", header_nax_mixed, gemm_j, &g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm2m_jl", header_nax_mixed, gemm_jl, &g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm2m_je", header_nax_mixed, gemm_je, &g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm2m_jle", header_nax_mixed, gemm_jle, &g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm2m_jle_p4", header_nax_mixed, gemm_jle_p4, &g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm2_le128", header_nax, gemm_jle128, &g2, gemm2_text);
    printMetal("turboexl3_pf_gemm2s_jle", header_nax_small, gemm_jle, &g2, gemm2s_text);
    printMetal("turboexl3_pf_gemm2m_jl2e", header_nax_mixed, gemm_jl2e, &g2, gemm2m_text);
    printMetal("turboexl3_pf_gemm2s_jl2e", header_nax_small, gemm_jl2e, &g2, gemm2s_text);
    // the down GEMM at Qwen's geometry (K 640, N 2560): the served stage (bm 64, fragment order), 128-deep stages (k), several
    // column blocks per threadgroup (n<c>), and the single-buffered padded body at bm 128 (both sub-stage and block loops there);
    // the all-register down GEMM and the served launch (n4) with the register path's integer-rate reader (j)
    const down_c = "constant constexpr int NHW = 80; constant constexpr int CODEBOOK = 0; constant constexpr int WBITS = 16; constant constexpr int NOPS = 1; constant constexpr int PROBE = 0; constant constexpr int WSP = 0; constant constexpr int MFIRST = 0; constant constexpr int TMAXS = 512; constant constexpr int INTR = 1; constant constexpr int NBX = 5; constant constexpr int KD = 640; constant constexpr int ND = 2560; constant constexpr int SINTR = 0; constant constexpr int ALAY = 0; constant constexpr int H16 = 0;";
    const served = down_c ++ " constant constexpr int BMR = 64; constant constexpr int SGT = 4; constant constexpr int DBUF = 1; constant constexpr int UNR = 1; constant constexpr int FRAGB = 1; constant constexpr int DEFER = 1;";
    const served_j = comptime replaceOnce(served, "int SINTR = 0;", "int SINTR = 1;");
    const served_jle = comptime replaceOnce(replaceOnce(served_j, "int ALAY = 0;", "int ALAY = 1;"), "int H16 = 0;", "int H16 = 1;");
    printMetal("turboexl3_pf_gemm1m_n4_jle", header_nax_mixed, served_jle ++ " constant constexpr int BKS = 64; constant constexpr int NCB = 4;", &g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1s_jle", header_nax_small, served_jle ++ " constant constexpr int BKS = 64; constant constexpr int NCB = 1;", &g1, gemm1s_text);
    const served_jl2e = comptime replaceOnce(served_jle, "int ALAY = 1;", "int ALAY = 2;");
    printMetal("turboexl3_pf_gemm1m_n4_jl2e", header_nax_mixed, served_jl2e ++ " constant constexpr int BKS = 64; constant constexpr int NCB = 4;", &g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1s", header_nax_small, served ++ " constant constexpr int BKS = 64; constant constexpr int NCB = 1;", &g1, gemm1s_text);
    printMetal("turboexl3_pf_gemm1s_j", header_nax_small, served_j ++ " constant constexpr int BKS = 64; constant constexpr int NCB = 1;", &g1, gemm1s_text);
    printMetal("turboexl3_pf_gemm1m_n4_j", header_nax_mixed, served_j ++ " constant constexpr int BKS = 64; constant constexpr int NCB = 4;", &g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1m_down", header_nax_mixed, served ++ " constant constexpr int BKS = 64; constant constexpr int NCB = 1;", &g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1m_k", header_nax_mixed, served ++ " constant constexpr int BKS = 128; constant constexpr int NCB = 1;", &g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1m_n2", header_nax_mixed, served ++ " constant constexpr int BKS = 64; constant constexpr int NCB = 2;", &g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1m_n4", header_nax_mixed, served ++ " constant constexpr int BKS = 64; constant constexpr int NCB = 4;", &g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1m_kn2", header_nax_mixed, served ++ " constant constexpr int BKS = 128; constant constexpr int NCB = 2;", &g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1m_kn4", header_nax_mixed, served ++ " constant constexpr int BKS = 128; constant constexpr int NCB = 4;", &g1, gemm1m_text);
    printMetal("turboexl3_pf_gemm1_kn2_sb", header_nax, down_c ++ " constant constexpr int BMR = 128; constant constexpr int SGT = 8; constant constexpr int DBUF = 0; constant constexpr int UNR = 0; constant constexpr int FRAGB = 0; constant constexpr int DEFER = 0; constant constexpr int BKS = 128; constant constexpr int NCB = 2;", &g1, gemm1_text);
    const tbl = [_]MetalArg{ .{ .name = "inds", .ty = "int", .shape = true }, .{ .name = "order", .ty = "uint" }, .{ .name = "tiles", .ty = "int", .out = true } };
    printMetal("turboexl3_pf_table", header_small, "constant constexpr int EXP = 512; constant constexpr int BMT = 128;", &tbl, text_table);
    printMetal("turboexl3_pf_table_m", header_small, "constant constexpr int EXP = 512; constant constexpr int BMT = 128; constant constexpr int SPL = 2;", &tbl, text_table_mixed);
    const frt = [_]MetalArg{ .{ .name = "inds", .ty = "uint", .shape = true }, .{ .name = "order", .ty = "uint", .out = true }, .{ .name = "tiles", .ty = "int", .out = true } };
    printMetal("turboexl3_pf_froute", header_small, "constant constexpr int EXP = 512; constant constexpr int BMT = 32; constant constexpr int SPL = 1;", &frt, text_froute);
    printMetal("turboexl3_pf_froute_m", header_small, "constant constexpr int EXP = 512; constant constexpr int BMT = 64; constant constexpr int SPL = 1;", &frt, text_froute_m);
    const prep = [_]MetalArg{ .{ .name = "x", .ty = "bfloat" }, .{ .name = "inds", .ty = "int" }, .{ .name = "order", .ty = "uint" }, .{ .name = "suh_g", .ty = "half" }, .{ .name = "suh_u", .ty = "half" }, .{ .name = "xg", .ty = "half", .out = true }, .{ .name = "xu", .ty = "half", .out = true }, .{ .name = "inv", .ty = "uint", .out = true }, .{ .name = "xsg", .ty = "float", .out = true }, .{ .name = "xsu", .ty = "float", .out = true } };
    printMetal("turboexl3_pf_prep", header_small, "constant constexpr int HID = 2560; constant constexpr int TOPK = 10; constant constexpr int BPT = 5; constant constexpr int ALAY = 0;", &prep, text_prep);
    printMetal("turboexl3_pf_prep_l", header_small, "constant constexpr int HID = 2560; constant constexpr int TOPK = 10; constant constexpr int BPT = 5; constant constexpr int ALAY = 1;", &prep, text_prep);
    printMetal("turboexl3_pf_prep_l2", header_small, "constant constexpr int HID = 2560; constant constexpr int TOPK = 10; constant constexpr int BPT = 5; constant constexpr int ALAY = 2;", &prep, text_prep);
    const prep_v = [_]MetalArg{ .{ .name = "x", .ty = "bfloat" }, .{ .name = "inds", .ty = "uint" }, .{ .name = "order", .ty = "uint", .shape = true }, .{ .name = "suh_g", .ty = "half" }, .{ .name = "suh_u", .ty = "half" }, .{ .name = "xg", .ty = "half", .out = true }, .{ .name = "xu", .ty = "half", .out = true }, .{ .name = "inv", .ty = "uint", .out = true }, .{ .name = "xsg", .ty = "float", .out = true }, .{ .name = "xsu", .ty = "float", .out = true } };
    printMetal("turboexl3_pf_prep_v", header_small, "constant constexpr int HID = 2560; constant constexpr int TOPK = 10; constant constexpr int ASG = 8; constant constexpr int XSO = 1;", &prep_v, text_prep_v);
    const mid = [_]MetalArg{ .{ .name = "zg", .ty = "half" }, .{ .name = "zu", .ty = "half" }, .{ .name = "inds", .ty = "int" }, .{ .name = "order", .ty = "uint" }, .{ .name = "svh_g", .ty = "half" }, .{ .name = "svh_u", .ty = "half" }, .{ .name = "suh_d", .ty = "half" }, .{ .name = "xd", .ty = "half", .out = true }, .{ .name = "xsd", .ty = "float", .out = true } };
    printMetal("turboexl3_pf_mid", header_small, "constant constexpr int INTER = 640; constant constexpr int BPT = 5; constant constexpr int ALAY = 0;", &mid, text_mid);
    printMetal("turboexl3_pf_mid_l", header_small, "constant constexpr int INTER = 640; constant constexpr int BPT = 5; constant constexpr int ALAY = 1;", &mid, text_mid);
    printMetal("turboexl3_pf_mid_l2", header_small, "constant constexpr int INTER = 640; constant constexpr int BPT = 5; constant constexpr int ALAY = 2;", &mid, text_mid);
    const mid_v = [_]MetalArg{ .{ .name = "zg", .ty = "half" }, .{ .name = "zu", .ty = "half" }, .{ .name = "inds", .ty = "uint" }, .{ .name = "order", .ty = "uint", .shape = true }, .{ .name = "svh_g", .ty = "half" }, .{ .name = "svh_u", .ty = "half" }, .{ .name = "suh_d", .ty = "half" }, .{ .name = "xd", .ty = "half", .out = true }, .{ .name = "xsd", .ty = "float", .out = true } };
    printMetal("turboexl3_pf_mid_v", header_small, "constant constexpr int INTER = 640; constant constexpr int ASG = 8; constant constexpr int XSO = 0;", &mid_v, text_mid_v);
    const fin = [_]MetalArg{ .{ .name = "zd", .ty = "half" }, .{ .name = "inds", .ty = "int" }, .{ .name = "inv", .ty = "uint" }, .{ .name = "scores", .ty = "float" }, .{ .name = "svh_d", .ty = "half" }, .{ .name = "y", .ty = "bfloat", .out = true } };
    printMetal("turboexl3_pf_fin", header_small, "constant constexpr int HID = 2560; constant constexpr int TOPK = 10; constant constexpr int BPT = 5; typedef bfloat OUT_T;", &fin, text_finish);
    const fin_v = [_]MetalArg{ .{ .name = "zd", .ty = "half" }, .{ .name = "inds", .ty = "uint" }, .{ .name = "inv", .ty = "uint", .shape = true }, .{ .name = "scores", .ty = "float" }, .{ .name = "svh_d", .ty = "half" }, .{ .name = "y", .ty = "bfloat", .out = true } };
    printMetal("turboexl3_pf_fin_v", header_small, "constant constexpr int HID = 2560; constant constexpr int TOPK = 10; constant constexpr int ASG = 8; typedef bfloat OUT_T;", &fin_v, text_finish_v);
}

test "turboexl3 prefill 128-row tiles texts: both GEMMs' launches are the mixed chain's kernels; mid and finish change only where they read a row; the route is froute's count and scatter with the 128-row rule" {
    const t = std.testing;
    const has = struct {
        fn f(h: []const u8, n: []const u8) bool {
            return std.mem.indexOf(u8, h, n) != null;
        }
    }.f;
    const count = struct {
        fn f(h: []const u8, n: []const u8) usize {
            return std.mem.count(u8, h, n);
        }
    }.f;
    // nothing in the chain writes through an input binding: every launch writes only its own outputs
    for ([_][]const u8{ text_froute_t128, text_mid_t128, text_finish_t128, gemm2m_text, gemm2_text, gemm1m_text, gemm1_text }) |x| try t.expect(!has(x, "const_cast"));
    // mid: text_mid with the row choice after the expert and the two loads through it, nothing else
    try t.expect(count(text_mid_t128, "const bool qt = int(p) < xbase[e];") == 1 and count(text_mid_t128, "const device half* zgs = qt ? zgt : zg;") == 1 and count(text_mid_t128, "const device half* zus = qt ? zut : zu;") == 1);
    try t.expect(has(text_mid_t128, "g[j] = float(zgs[ulong(p) * IC + i]);") and has(text_mid_t128, "u[j] = float(zus[ulong(p) * IC + i]);") and !has(text_mid_t128, "float(zg[") and !has(text_mid_t128, "float(zu["));
    {
        const back = comptime replace(replace(replace(text_mid_t128, "    const bool qt = int(p) < xbase[e];\n    const device half* zgs = qt ? zgt : zg;\n    const device half* zus = qt ? zut : zu;\n", ""), "float(zgs[", "float(zg["), "float(zus[", "float(zu[");
        try t.expectEqualStrings(text_mid, back);
    }
    // finish: text_finish with the slot's row choice and its load through it
    try t.expect(count(text_finish_t128, "const device half* zds = (int(p) < xbase[e]) ? zdt : zd;") == 1 and has(text_finish_t128, "v[j] = float(zds[ulong(p) * HC") and !has(text_finish_t128, "float(zd["));
    {
        const back = comptime replace(replace(text_finish_t128, "        const device half* zds = (int(p) < xbase[e]) ? zdt : zd;\n", ""), "float(zds[", "float(zd[");
        try t.expectEqualStrings(text_finish, back);
    }
    // the choice reads the rows before an expert's first remainder position from the 128-row tiles' outputs
    try t.expect(mid_t128_in.len == mid_in.len + 3 and fin_t128_in.len == fin_in.len + 2);
    for (mid_in, mid_t128_in[0..mid_in.len]) |p, q| try t.expect(std.mem.eql(u8, std.mem.span(p), std.mem.span(q)));
    for (fin_in, fin_t128_in[0..fin_in.len]) |p, q| try t.expect(std.mem.eql(u8, std.mem.span(p), std.mem.span(q)));
    try t.expect(std.mem.eql(u8, std.mem.span(mid_t128_in[mid_in.len + 2]), "xbase") and std.mem.eql(u8, std.mem.span(fin_t128_in[fin_in.len + 1]), "xbase"));
    // the route: froute's count and scatter verbatim, the 128-row table's bound as t128TileRows computes it, the mixed rule,
    // each expert's first remainder position
    try t.expect(has(text_froute_t128, froute_count));
    try t.expect(has(text_froute_t128, "const uint NPQ = min(EC, SC / (CU + 1u));") and has(text_froute_t128, "const uint TQ = (T128 != 0) ? max(1u, NPQ + (SC - (CU + 1u) * NPQ) / 128u) : 1u;"));
    try t.expect(has(text_froute_t128, "constexpr uint PER = CU / BMC + ((CU % BMC > SR) ? 1u : 0u);") and has(text_froute_t128, "constexpr uint MINR = (BMC < SR + 1u) ? BMC : SR + 1u;"));
    try t.expect(has(text_froute_t128, "const uint TBIG = (T128 != 0) ? max(1u, min(TBIG0, min(EC * PER, SC / MINR))) : TBIG0;"));
    try t.expect(has(text_froute_t128, "constexpr uint CU = (uint(CUT) < 127u) ? uint(CUT) : 127u;"));
    try t.expect(has(text_froute_t128, "if (T128 != 0 && cnt > CU) { nq += 1u; cnt = 0u; }"));
    try t.expect(has(text_froute_t128, "if (r == 0u) nb = nfull;") and has(text_froute_t128, "else if (r > SR) nb = nfull + 1u;") and has(text_froute_t128, "else { nb = nfull; ns = (r + 31u) / 32u; }"));
    try t.expect(has(text_froute_t128, "const uint base = lo + c0 - cnt;\n    if (tid < EC) xbase[tid] = int(base);"));
    try t.expect(froute_t128_out.len == 4 and std.mem.eql(u8, std.mem.span(froute_t128_out[3]), "xbase"));
}

test "turboexl3 prefill 128-row tiles tables: every sorted position in exactly one tile; 128-row tiles hold full runs or one remainder past the cut; the mixed tiles hold what is left; the route's bounds hold" {
    const t = std.testing;
    const a = t.allocator;
    var prng = std.Random.DefaultPrng.init(0xE96E);
    const rnd = prng.random();
    var counts: [512]u32 = undefined;
    for (0..24) |ci| {
        for (&counts, 0..) |*c, e| c.* = switch (ci % 6) {
            0 => rnd.uintLessThan(u32, 200),
            // skewed: a few hot experts, many cold ones
            1 => if (rnd.uintLessThan(u32, 8) == 0) rnd.uintLessThan(u32, 3000) else rnd.uintLessThan(u32, 40),
            2 => @intCast(e % 301),
            // every count on a 128-row tile's cut edges, and on the 64- and 32-row ones
            3 => @intCast((e % 4) * 128 + 63 + (e % 3)),
            4 => @intCast((e % 3) * 64 + 31 + (e % 4)),
            else => if (e % 7 == 0) 0 else rnd.uintLessThan(u32, 600),
        };
        var S: u32 = 0;
        for (counts) |c| S += c;
        for ([_]u8{ 0, 1, 32, 63, 64, 65, 96, 127, 200 }) |cut| for ([_]bool{ true, false }) |on| {
            var tb = try t128Tables(a, &counts, 64, cut, on);
            defer tb.deinit(a);
            const cover = try a.alloc(u8, S);
            defer a.free(cover);
            @memset(cover, 0);
            const cu: u32 = @min(cut, 127);
            for (tb.t128.items, 0..) |q, i| {
                try t.expect(q[2] >= 1 and q[2] <= 128 and (q[2] == 128 or q[2] > cu));
                // a part-full one is its expert's last 128-row tile
                if (q[2] < 128 and i + 1 < tb.t128.items.len) try t.expect(tb.t128.items[i + 1][0] != q[0]);
                for (q[1]..q[1] + q[2]) |p| cover[p] += 1;
            }
            for (tb.big.items) |q| {
                try t.expect(q[2] >= 1 and q[2] <= 64 and (q[2] == 64 or q[2] > 32));
                for (q[1]..q[1] + q[2]) |p| cover[p] += 1;
            }
            for (tb.small.items) |q| {
                try t.expect(q[2] >= 1 and q[2] <= 32);
                for (q[1]..q[1] + q[2]) |p| cover[p] += 1;
            }
            for (cover) |v| try t.expectEqual(@as(u8, 1), v);
            // the route's output bounds: the 128-row table (`t128TileRows`, `text_froute_t128`'s TQ) and the mixed one's
            // (`t128MixedRows`, TBIG; never above the mixed table without the 128-row tiles)
            try t.expect(tb.t128.items.len <= t128TileRows(S, 512, cut, on));
            try t.expect(tb.big.items.len <= t128MixedRows(S, 512, 64, 1, cut, on) and tb.small.items.len <= 512);
            try t.expect(t128MixedRows(S, 512, 64, 1, cut, on) <= (S + 63) / 64 + 512);
            if (!on) try t.expectEqual(@as(usize, 0), tb.t128.items.len);
            // with the 128-row tiles on, the mixed tiles take each expert's rows past its full 128-row tiles when no more
            // than the cut (an expert with a part-full 128-row tile has none), and nothing else
            if (on) {
                var mixed: [512]u32 = @splat(0);
                for (tb.big.items) |m| mixed[m[0]] += m[2];
                for (tb.small.items) |m| mixed[m[0]] += m[2];
                for (counts, mixed) |c, mx| try t.expect(mx <= cu and mx == (if (c % 128 > cu) 0 else c % 128));
            }
        };
    }
}

test "turboexl3 prefill 128-row tiles options: t128 needs bm 64, the mixed table at split 1, froute and aux 0; the tables' row bounds" {
    const t = std.testing;
    const served: Options = .{ .bm = 64, .small = .mixed, .dbuf = true, .unroll = true, .fragb = true, .intr = true, .defer_affine = true, .ncb = 4, .sintr = true, .froute = true, .alay = 2 };
    var o = served;
    o.t128 = true;
    try t.expect(o.valid());
    var v = o;
    v.bm = 128;
    try t.expect(!v.valid());
    v = o;
    v.froute = false;
    try t.expect(!v.valid());
    v = o;
    v.split = 2;
    try t.expect(!v.valid());
    v = o;
    v.small = .off;
    try t.expect(!v.valid());
    v = o;
    v.alay = 0;
    v.aux = 8;
    try t.expect(!v.valid());
    try t.expectEqual(@as(u32, 1), t128TileRows(13280, 512, 64, false));
    try t.expectEqual(@as(u32, 81920 / 128), t128TileRows(81920, 512, 127, true));
    // 512 part-full tiles of 65 rows, then full ones: 512 + (40960 - 33280) / 128
    try t.expectEqual(@as(u32, 572), t128TileRows(40960, 512, 64, true));
    try t.expectEqual(@as(u32, 892), t128TileRows(81920, 512, 64, true));
    try t.expectEqual(@as(u32, 204), t128TileRows(13280, 512, 64, true));
    try t.expectEqual(@as(u32, 828), t128TileRows(40960, 512, 0, true));
    try t.expectEqual(@as(u32, 1), t128TileRows(10, 512, 64, true));
    // the mixed table: all of it without the 128-row tiles; with them one tile per expert at cut 64 (two at 127, none at
    // 32), at most S / 33 of them
    try t.expectEqual(@as(u32, 1152), t128MixedRows(40960, 512, 64, 1, 64, false));
    try t.expectEqual(@as(u32, 512), t128MixedRows(81920, 512, 64, 1, 64, true));
    try t.expectEqual(@as(u32, 402), t128MixedRows(13280, 512, 64, 1, 64, true));
    try t.expectEqual(@as(u32, 1024), t128MixedRows(40960, 512, 64, 1, 127, true));
    try t.expectEqual(@as(u32, 1), t128MixedRows(40960, 512, 64, 1, 32, true));
    try t.expectEqual(@as(u32, 9), t128MixedRows(310, 512, 64, 1, 64, true));
}

test "turboexl3 prefill 128-row tiles metal sources: the route, both GEMMs' launches (both bodies), mid and finish as standalone translation units (TURBOEXL3_METAL_DUMP=1)" {
    _ = std.c.getenv("TURBOEXL3_METAL_DUMP") orelse return error.SkipZigTest;
    const frt = [_]MetalArg{ .{ .name = "inds", .ty = "uint", .shape = true }, .{ .name = "order", .ty = "uint", .out = true }, .{ .name = "tiles", .ty = "int", .out = true }, .{ .name = "tiles128", .ty = "int", .out = true }, .{ .name = "xbase", .ty = "int", .out = true } };
    printMetal("turboexl3_pf_t128_route", header_small, "constant constexpr int EXP = 512; constant constexpr int BMT = 64; constant constexpr int SPL = 1; constant constexpr int CUT = 64; constant constexpr int T128 = 1;", &frt, text_froute_t128);
    // the served gate|up launch (bm 64, fragment order, A layout 2, the register path's integer-rate reader) and its bm-128
    // body; the served down launch (ncb 4) and its bm-128 body; the mixed chain's kernels' arguments
    const gu = "constant constexpr int NHW = 80; constant constexpr int CODEBOOK = 0; constant constexpr int WBITS = 16; constant constexpr int NOPS = 2; constant constexpr int DBUF = 1; constant constexpr int PROBE = 0; constant constexpr int WSP = 0; constant constexpr int MFIRST = 0; constant constexpr int UNR = 1; constant constexpr int TMAXS = 512; constant constexpr int FRAGB = 1; constant constexpr int INTR = 1; constant constexpr int DEFER = 1; constant constexpr int NBX = 20; constant constexpr int KD = 2560; constant constexpr int ND = 640; constant constexpr int BKS = 64; constant constexpr int NCB = 1; constant constexpr int SINTR = 1; constant constexpr int ALAY = 2; constant constexpr int H16 = 0;";
    const dn = "constant constexpr int NHW = 80; constant constexpr int CODEBOOK = 0; constant constexpr int WBITS = 16; constant constexpr int NOPS = 1; constant constexpr int DBUF = 1; constant constexpr int PROBE = 0; constant constexpr int WSP = 0; constant constexpr int MFIRST = 0; constant constexpr int UNR = 1; constant constexpr int TMAXS = 512; constant constexpr int FRAGB = 1; constant constexpr int INTR = 1; constant constexpr int DEFER = 1; constant constexpr int NBX = 5; constant constexpr int KD = 640; constant constexpr int ND = 2560; constant constexpr int BKS = 64; constant constexpr int NCB = 4; constant constexpr int SINTR = 1; constant constexpr int ALAY = 2; constant constexpr int H16 = 0;";
    const m64 = " constant constexpr int BMR = 64; constant constexpr int SGT = 4;";
    const m128 = " constant constexpr int BMR = 128; constant constexpr int SGT = 8;";
    const g2 = [_]MetalArg{ .{ .name = "tiles", .ty = "int", .shape = true }, .{ .name = "x0", .ty = "half" }, .{ .name = "x1", .ty = "half" }, .{ .name = "w0", .ty = "uint16_t" }, .{ .name = "w1", .ty = "uint16_t" }, .{ .name = "xs0", .ty = "float" }, .{ .name = "xs1", .ty = "float" }, .{ .name = "z0", .ty = "half", .out = true }, .{ .name = "z1", .ty = "half", .out = true } };
    const g1 = [_]MetalArg{ .{ .name = "tiles", .ty = "int", .shape = true }, .{ .name = "x0", .ty = "half" }, .{ .name = "w0", .ty = "uint16_t" }, .{ .name = "xs0", .ty = "float" }, .{ .name = "z0", .ty = "half", .out = true } };
    printMetal("turboexl3_pf_t128_gum", header_nax, gu ++ m64, &g2, gemm2m_text);
    printMetal("turboexl3_pf_t128_gut", header_nax, gu ++ m128, &g2, gemm2_text);
    printMetal("turboexl3_pf_t128_dnm", header_nax, dn ++ m64, &g1, gemm1m_text);
    printMetal("turboexl3_pf_t128_dnt", header_nax, dn ++ m128, &g1, gemm1_text);
    const mid = [_]MetalArg{ .{ .name = "zg", .ty = "half" }, .{ .name = "zu", .ty = "half" }, .{ .name = "inds", .ty = "int" }, .{ .name = "order", .ty = "uint" }, .{ .name = "svh_g", .ty = "half" }, .{ .name = "svh_u", .ty = "half" }, .{ .name = "suh_d", .ty = "half" }, .{ .name = "zgt", .ty = "half" }, .{ .name = "zut", .ty = "half" }, .{ .name = "xbase", .ty = "int" }, .{ .name = "xd", .ty = "half", .out = true }, .{ .name = "xsd", .ty = "float", .out = true } };
    printMetal("turboexl3_pf_t128_mid", header_small, "constant constexpr int INTER = 640; constant constexpr int BPT = 5; constant constexpr int ALAY = 2;", &mid, text_mid_t128);
    const fin = [_]MetalArg{ .{ .name = "zd", .ty = "half" }, .{ .name = "inds", .ty = "int" }, .{ .name = "inv", .ty = "uint" }, .{ .name = "scores", .ty = "float" }, .{ .name = "svh_d", .ty = "half" }, .{ .name = "zdt", .ty = "half" }, .{ .name = "xbase", .ty = "int" }, .{ .name = "y", .ty = "bfloat", .out = true } };
    printMetal("turboexl3_pf_t128_fin", header_small, "constant constexpr int HID = 2560; constant constexpr int TOPK = 10; constant constexpr int BPT = 5; typedef bfloat OUT_T;", &fin, text_finish_t128);
}
