//! A dense EXL3 linear (one trellis, no experts): y [rows, out] = finish(innerGemv(prepare(x))) with exllamav3's rounding
//! points, for the trunk of a fully EXL3 model (attention / GDN projections, shared expert, lm_head, MTP). Decode rows
//! (<= DECODE_ROWS_MAX) run prepare (f16 round, suh, H128), a multi-row trellis GEMV that decodes every weight once for up
//! to 8 rows (split-K partial planes) and finish (sum the planes, H128, svh, f16 round); MLX_SERVE_TURBOEXL3_DENSE=rowslot keeps
//! the row-slot kernels of `unfused.zig` (one decode per row) for A/B. Rows up to `small_rows` (default 32: a prefill's
//! snapshot tail) run the expert chain's register-path GEMM with one op (`prefill.gemm1s_text`): each 32-row tile decodes
//! the trellis once straight into the matrix units' B registers, no f16 W_hat. Wider calls dequantize the trellis to an f16
//! W_hat [in, out] once, run MLX's matmul on the prepared rows (f32 accumulation, f16 output = the finish's first rounding)
//! and finish. Output in x's dtype. Held to `format.project` per row.
//! DEC-ONE (MLX_SERVE_DEC_ONE, on unless "0"): decode rows in one GEMV launch instead of three. The threadgroup that
//! arrives last at its (128-wide output block, row chunk) runs the finish there (an integer arrival counter: no float
//! atomics, the K slices still summed in slice order), and the prepare runs inside the launch (threadgroup memory) or in
//! its own launch (MLX_SERVE_DEC_ONE_PREP): the three-launch chain's bytes either way.
//! The resident trunk (TRUNKRES; the host's MLX_SERVE_EXL3_TRUNK=resident): prefill rows take an admitted trellis's W_hat
//! from the resident table (`residentArm`), decoded by the same launch once per load and kept, so every call's bytes stay
//! the per-call path's.
//! The un-rotated trunk (TRUNKUNROT; the host's MLX_SERVE_EXL3_TRUNK=bf16, opt-in, a reordering): the table holds an
//! admitted trellis's linear out of the Hadamard basis instead (`unrotCols` / `unrotRows`, `residentInstall`), and prefill
//! rows past `unrot_min_rows` are one MLX matmul of x against it in x's dtype (`residentPlain`).
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const format = @import("format.zig");
const decode = @import("decode.zig");
const unfused = @import("unfused.zig");
const prefill = @import("prefill.zig");

/// prepare for a dense linear: xp half [rows, IN] = H128(f16(x[row]) * suh) per 128-block; LANE_ORDER 1 writes the GEMV's
/// lane order within each 16-block, 0 the natural order the matmul path needs; one simdgroup per (block, row); x any float dtype
pub const text_prep =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]); }
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint i = b * 128u + L + 32u * j;
    \\        const uint r = i & 15u;
    \\        const uint q = (r & 7u) >> 1u;
    \\        const uint m = (r & 1u) | ((r >> 3u) << 1u);
    \\        const uint dst = (LANE_ORDER != 0) ? ((i & ~15u) + 4u * q + m) : i;
    \\        xp[row * DIMC + dst] = half(v[j] * HAD_SCALE);
    \\    }
    \\
;

/// text_prep (lane order for the multi-row GEMV, natural for the fused prefill GEMM) plus xs f32 [rows, IN/128], the sum of
/// each 128-block's rounded values (what a deferred MUL1 affine map needs: sum_i x_i (A t_i + C) = A sum_i x_i t_i + C sum_i x_i)
pub const text_prep_s =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr uint NB = DIMC / 128u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]); }
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    float sum = 0.0f;
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint i = b * 128u + L + 32u * j;
    \\        const uint r = i & 15u;
    \\        const uint q = (r & 7u) >> 1u;
    \\        const uint m = (r & 1u) | ((r >> 3u) << 1u);
    \\        const half hv = half(v[j] * HAD_SCALE);
    \\        xp[row * DIMC + ((LANE_ORDER != 0) ? ((i & ~15u) + 4u * q + m) : i)] = hv;
    \\        sum += float(hv);
    \\    }
    \\    sum = simd_sum(sum);
    \\    if (L == 0u) xs[row * NB + b] = sum;
    \\
;

/// finish for a dense linear: y OUT_T [rows, OUT] = f16(H128(f16(z)) * svh) per 128-block; z f32 (decode) or f16 (prefill)
pub const text_finish =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(z[row * DIMC + b * 128u + L + 32u * j])));
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    for (uint j = 0; j < 4u; ++j) { const uint o = b * 128u + L + 32u * j; y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
    \\
;

/// text_finish over a call whose inner product arrives as up to four row pieces zp0..zp3 (`Dense.finishPieces`; f16
/// [rows_p, DIM]; piece p holds the call's rows [R_p, R_{p+1}), R_0 = 0, R1..R3 0-dim ints, an unused piece's bound at the
/// call's rows): row `row` reads its piece's row, then text_finish's statements, so y OUT_T [rows, DIM] is the one-call
/// finish's bit for bit
pub const text_finish_pieces =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint r1 = uint(R1), r2 = uint(R2), r3 = uint(R3);
    \\    const device half* z = (row >= r3) ? zp3 : ((row >= r2) ? zp2 : ((row >= r1) ? zp1 : zp0));
    \\    const uint pr = row - ((row >= r3) ? r3 : ((row >= r2) ? r2 : ((row >= r1) ? r1 : 0u)));
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(z[pr * DIMC + b * 128u + L + 32u * j])));
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    for (uint j = 0; j < 4u; ++j) { const uint o = b * 128u + L + 32u * j; y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
    \\
;

/// dequantize: W half [IN, OUT] from the trellis [IN/16, OUT/16, n]; one simdgroup per tile, lane L decodes its 8 weights:
/// weight m of lane L is row 2 (L & 3) + {0, 1, 8, 9}[m & 3], column (L >> 2) + 8 (m >> 2) of the tile (the GEMV's layout)
pub const text_dequant =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint INC = uint(IN);
    \\    constexpr uint OUTC = uint(OUT);
    \\    constexpr uint OUT_TILES = OUTC / 16u;
    \\    const uint tile_id = threadgroup_position_in_grid.x * 4u + simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    if (tile_id >= (INC / 16u) * OUT_TILES) return;
    \\    const uint tk = tile_id / OUT_TILES;
    \\    const uint tn = tile_id - tk * OUT_TILES;
    \\    const device uint* tile = (const device uint*)(w) + ulong(tile_id) * ulong(NW);
    \\    const uint qb = exl3_gen_qb<N>(L);
    \\    const uint ga = exl3_gen_a<N>(L);
    \\    const uint lp = exl3_gen_lp<N>(L);
    \\    const auto ww = exl3_tile_load<N, 0>(tile, qb, ga, lp);
    \\    uint cw[8];
    \\    cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
    \\    cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
    \\    for (uint m = 0; m < 8u; ++m) {
    \\        const uint r = turboexl3_row_of(L, m & 3u);
    \\        const uint c = (L >> 2u) + 8u * (m >> 2u);
    \\        W[ulong(tk * 16u + r) * ulong(OUTC) + tn * 16u + c] = exl3_decode<CB>(cw[m]);
    \\    }
    \\
;

/// multi-row GEMV: every weight is decoded once per threadgroup and applied to RB rows. z f32 [KS * ROWS, OUT]: K-slice
/// ks's partial sums of row r at z[ks * ROWS + r]. One 128-thread threadgroup per (BT output tiles, row chunk of RB rows,
/// K slice); the four simdgroups split the slice's input tiles; 2 BT RB f32 accumulators per lane. A row's summation order
/// (tile order per lane, the quad fold, simdgroups 0..3, then the slices in order in the finish) depends on KS alone, so a
/// row's bytes do not change with how many rows share the launch.
pub const text_dgemv =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint INC = uint(IN);
    \\    constexpr uint OUTC = uint(OUT);
    \\    constexpr uint IN_TILES = INC / 16u;
    \\    constexpr uint OUT_TILES = OUTC / 16u;
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr uint BTC = uint(BT);
    \\    constexpr uint RBC = uint(RB);
    \\    constexpr uint ROWSC = uint(ROWS);
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint r0 = threadgroup_position_in_grid.y * RBC;
    \\    const uint ksl = threadgroup_position_in_grid.z;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    threadgroup float red[4][RBC * 16u * BTC];
    \\    const uint qb = exl3_gen_qb<N>(L);
    \\    const uint ga = exl3_gen_a<N>(L);
    \\    const uint lp = exl3_gen_lp<N>(L);
    \\    float a[RBC][BTC][2];
    \\    for (uint r = 0; r < RBC; ++r) for (uint t = 0; t < BTC; ++t) { a[r][t][0] = 0.0f; a[r][t][1] = 0.0f; }
    \\    constexpr uint PER = IN_TILES / (KSC * 4u);
    \\    const uint tk0 = (ksl * 4u + sg) * PER;
    \\    const device uint* wbase = (const device uint*)(w);
    \\    const device half* xbase = xp + 4u * (L & 3u);   // lane order: the quad's four rows of each tile
    \\    for (uint tk = tk0; tk < tk0 + PER; ++tk) {
    \\        float xr[RBC][4];
    \\        for (uint r = 0; r < RBC; ++r) {
    \\            // a chunk's spare rows (ROWS not a multiple of RB) reread the last row; their sums are never written
    \\            const uint row = min(r0 + r, ROWSC - 1u);
    \\            const half4 x4 = *reinterpret_cast<const device half4*>(xbase + ulong(row) * ulong(INC) + tk * 16u);
    \\            xr[r][0] = float(x4.x); xr[r][1] = float(x4.y); xr[r][2] = float(x4.z); xr[r][3] = float(x4.w);
    \\        }
    \\        for (uint t = 0; t < BTC; ++t) {
    \\            const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * BTC + t) * ulong(NW);
    \\            const auto ww = exl3_tile_load<N, 0>(tile, qb, ga, lp);
    \\            uint cw[8];
    \\            cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
    \\            cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
    \\            for (uint m = 0; m < 8u; ++m) {
    \\                // DEFER: MUL1's byte sum t; the finish applies A and C (C times the row's input sum)
    \\                const float wv = (DEFER != 0) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
    \\                for (uint r = 0; r < RBC; ++r) a[r][t][m >> 2] = fma(xr[r][m & 3u], wv, a[r][t][m >> 2]);
    \\            }
    \\        }
    \\    }
    \\    for (uint r = 0; r < RBC; ++r) turboexl3_fold<BTC>(a[r], &red[sg][r * 16u * BTC], L);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint i = tid; i < RBC * 16u * BTC; i += 128u) {
    \\        const uint r = i / (16u * BTC);
    \\        const uint row = r0 + r;
    \\        if (row < ROWSC) z[(ulong(ksl) * ulong(ROWSC) + row) * ulong(OUTC) + blk * (16u * BTC) + (i - r * 16u * BTC)] = ((red[0][i] + red[1][i]) + red[2][i]) + red[3][i];
    \\    }
    \\
;

/// text_dfinish up to the 128-point butterfly (decb.zig's finish variants reuse these statements verbatim)
pub const dfinish_v =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr uint ROWSC = uint(ROWS);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    constexpr uint INB = uint(INBLK);
    \\    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    \\    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    \\    float xsum = 0.0f;
    \\    if (DEFER != 0) for (uint q = 0; q < INB; ++q) xsum += xs[row * INB + q];
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint o = b * 128u + L + 32u * j;
    \\        float acc = z[ulong(row) * ulong(DIMC) + o];
    \\        for (uint k = 1u; k < KSC; ++k) acc += z[(ulong(k) * ulong(ROWSC) + row) * ulong(DIMC) + o];
    \\        if (DEFER != 0) acc = fma(acc, MUL1_A, MUL1_C * xsum);
    \\        v[j] = float(half(acc));
    \\    }
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\
;
/// text_dfinish's output: x HAD_SCALE x svh, f16, OUT_T
pub const dfinish_store =
    \\    for (uint j = 0; j < 4u; ++j) { const uint o = b * 128u + L + 32u * j; y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
    \\
;
/// finish of the multi-row GEMV: y OUT_T [ROWS, OUT] = f16(H128(f16(sum of the KS planes, in slice order)) * svh)
pub const text_dfinish = dfinish_v ++ dfinish_store;

// ---- DEC-ONE: one launch per dense linear at decode rows ----
//
// text_dgemv's GEMV (same grid, same per-lane tile order, quad fold and simdgroup sum into the KS partial planes z), then
// the threadgroup that arrives last at its (128-wide output block, row chunk) runs text_dfinish's arithmetic for that block.
// Arrival: one integer counter per (block, chunk), zeroed by a launch before this one (prep_sc, or dec_zero with the prepare
// inside); device-scope seq_cst fences around one relaxed fetch_add per threadgroup and the partials through coherent(device)
// pointers (form 6's idiom); nothing waits. With the prepare inside (text_dec1), each threadgroup prepares its K
// slice of its row chunk into threadgroup memory with text_prep_s's arithmetic, and the first column group of each output
// block publishes the slice's 128-block input sums (xsx) for the deferred MUL1 map; text_dec0 reads prep_sc's xp / xs. The
// value-setting lines are copied from text_dgemv / text_prep_s / text_dfinish (a CPU test holds them equal), so each
// rounding happens where the three-launch chain rounds it.

/// DEC-ONE's constants and GEMV setup (text_dgemv's, plus the output block of this threadgroup's columns)
pub const dec_head =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint INC = uint(IN);
    \\    constexpr uint OUTC = uint(OUT);
    \\    constexpr uint IN_TILES = INC / 16u;
    \\    constexpr uint OUT_TILES = OUTC / 16u;
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr uint BTC = uint(BT);
    \\    constexpr uint RBC = uint(RB);
    \\    constexpr uint ROWSC = uint(ROWS);
    \\    constexpr uint NCHC = (ROWSC + RBC - 1u) / RBC;   // row chunks
    \\    constexpr uint NB = INC / 128u;                   // input 128-blocks
    \\    constexpr uint GC = 8u / BTC;                     // column groups per 128-wide output block
    \\    constexpr uint ARR = GC * KSC;                    // arrivals per (output block, row chunk)
    \\    constexpr uint PER = IN_TILES / (KSC * 4u);
    \\    constexpr uint SPAN = 64u * PER;                  // a K slice's inputs
    \\    constexpr uint SPB = SPAN / 128u;                 // its 128-blocks (the prepare inside takes SPAN % 128 == 0)
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint chunk = threadgroup_position_in_grid.y;
    \\    const uint r0 = chunk * RBC;
    \\    const uint ksl = threadgroup_position_in_grid.z;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint ob = blk / GC;
    \\    threadgroup float red[4][RBC * 16u * BTC];
    \\    threadgroup uint arrived_last;
    \\    device TURBOEXL3_COH float* zc = (device TURBOEXL3_COH float*)z;
    \\
;

/// DEC-ONE with the prepare inside: the threadgroup's K slice of its row chunk into threadgroup memory (text_prep_s's
/// arithmetic, lane order, spare rows from the chunk's last row as text_dgemv reads them), one simdgroup per (128-block, row)
pub const dec_prep1 =
    \\    threadgroup half xt[RBC * SPAN];
    \\    device TURBOEXL3_COH float* xsc = (device TURBOEXL3_COH float*)xsx;
    \\    for (uint pi = sg; pi < SPB * RBC; pi += 4u) {
    \\        const uint rr = pi / SPB;
    \\        const uint b = ksl * SPB + (pi - rr * SPB);
    \\        const uint row = min(r0 + rr, ROWSC - 1u);
    \\        float v[4];
    \\        for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(float(x[row * INC + i]))) * float(suh[i]); }
    \\        for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\        { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\        float sum = 0.0f;
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint i = b * 128u + L + 32u * j;
    \\            const uint r = i & 15u;
    \\            const uint q = (r & 7u) >> 1u;
    \\            const uint m = (r & 1u) | ((r >> 3u) << 1u);
    \\            const half hv = half(v[j] * HAD_SCALE);
    \\            xt[rr * SPAN + ((i & ~15u) + 4u * q + m) - ksl * SPAN] = hv;
    \\            sum += float(hv);
    \\        }
    \\        sum = simd_sum(sum);
    \\        // the deferred map's input sums: the block's first column group publishes them (it arrives like every other)
    \\        if (DEFER != 0 && (blk % GC) == 0u && L == 0u && r0 + rr < ROWSC) xsc[((ob * NCHC + chunk) * RBC + rr) * NB + b] = sum;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
;

/// the GEMV's input loads: prep_sc's rows in device memory (text_dgemv's) ...
pub const dec_gemv0 =
    \\    const uint qb = exl3_gen_qb<N>(L);
    \\    const uint ga = exl3_gen_a<N>(L);
    \\    const uint lp = exl3_gen_lp<N>(L);
    \\    float a[RBC][BTC][2];
    \\    for (uint r = 0; r < RBC; ++r) for (uint t = 0; t < BTC; ++t) { a[r][t][0] = 0.0f; a[r][t][1] = 0.0f; }
    \\    const uint tk0 = (ksl * 4u + sg) * PER;
    \\    const device uint* wbase = (const device uint*)(w);
    \\    const device half* xbase = xp + 4u * (L & 3u);   // lane order: the quad's four rows of each tile
    \\    for (uint tk = tk0; tk < tk0 + PER; ++tk) {
    \\        float xr[RBC][4];
    \\        for (uint r = 0; r < RBC; ++r) {
    \\            const uint row = min(r0 + r, ROWSC - 1u);
    \\            const half4 x4 = *reinterpret_cast<const device half4*>(xbase + ulong(row) * ulong(INC) + tk * 16u);
    \\            xr[r][0] = float(x4.x); xr[r][1] = float(x4.y); xr[r][2] = float(x4.z); xr[r][3] = float(x4.w);
    \\        }
    \\
;

/// ... or the slice prepared in threadgroup memory
pub const dec_gemv1 =
    \\    const uint qb = exl3_gen_qb<N>(L);
    \\    const uint ga = exl3_gen_a<N>(L);
    \\    const uint lp = exl3_gen_lp<N>(L);
    \\    float a[RBC][BTC][2];
    \\    for (uint r = 0; r < RBC; ++r) for (uint t = 0; t < BTC; ++t) { a[r][t][0] = 0.0f; a[r][t][1] = 0.0f; }
    \\    const uint tk0 = (ksl * 4u + sg) * PER;
    \\    const device uint* wbase = (const device uint*)(w);
    \\    const uint tks = ksl * 4u * PER;   // the slice's first input tile
    \\    for (uint tk = tk0; tk < tk0 + PER; ++tk) {
    \\        float xr[RBC][4];
    \\        for (uint r = 0; r < RBC; ++r) {
    \\            const half4 x4 = *reinterpret_cast<const threadgroup half4*>(xt + r * SPAN + (tk - tks) * 16u + 4u * (L & 3u));
    \\            xr[r][0] = float(x4.x); xr[r][1] = float(x4.y); xr[r][2] = float(x4.z); xr[r][3] = float(x4.w);
    \\        }
    \\
;

/// text_dgemv's tile loop, quad fold and simdgroup sum (the z plane values), then the arrival
pub const dec_gemv_arrive =
    \\        for (uint t = 0; t < BTC; ++t) {
    \\            const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * BTC + t) * ulong(NW);
    \\            const auto ww = exl3_tile_load<N, 0>(tile, qb, ga, lp);
    \\            uint cw[8];
    \\            cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
    \\            cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
    \\            for (uint m = 0; m < 8u; ++m) {
    \\                // DEFER: MUL1's byte sum t; the finish applies A and C (C times the row's input sum)
    \\                const float wv = (DEFER != 0) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
    \\                for (uint r = 0; r < RBC; ++r) a[r][t][m >> 2] = fma(xr[r][m & 3u], wv, a[r][t][m >> 2]);
    \\            }
    \\        }
    \\    }
    \\    for (uint r = 0; r < RBC; ++r) turboexl3_fold<BTC>(a[r], &red[sg][r * 16u * BTC], L);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint i = tid; i < RBC * 16u * BTC; i += 128u) {
    \\        const uint r = i / (16u * BTC);
    \\        const uint row = r0 + r;
    \\        if (row < ROWSC) zc[(ulong(ksl) * ulong(ROWSC) + row) * ulong(OUTC) + blk * (16u * BTC) + (i - r * 16u * BTC)] = ((red[0][i] + red[1][i]) + red[2][i]) + red[3][i];
    \\    }
    \\    // arrival: this threadgroup's stores fenced device-wide before its one count; the count that reaches ARR finishes
    \\    // the output block here, every other threadgroup is done (nothing waits)
    \\    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\    threadgroup_barrier(mem_flags::mem_device);
    \\    if (tid == 0u) {
    \\        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\        const uint old = atomic_fetch_add_explicit((device atomic_uint*)cnt + (ob * NCHC + chunk), 1u, memory_order_relaxed);
    \\        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\        arrived_last = (old == ARR - 1u) ? 1u : 0u;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (arrived_last == 0u) return;
    \\    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\    // text_dfinish for this output block, simdgroup rr for the chunk's row rr
    \\    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    \\    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    \\    for (uint rr = sg; rr < RBC; rr += 4u) {
    \\        const uint row = r0 + rr;
    \\        if (row >= ROWSC) break;
    \\        float xsum = 0.0f;
    \\
;

/// the finish's input sums: prep_sc's (text_dfinish's loop) ...
pub const dec_xsum0 =
    \\        if (DEFER != 0) for (uint q = 0; q < NB; ++q) xsum += xs[row * NB + q];
    \\
;

/// ... or the ones the output block's first column groups published
pub const dec_xsum1 =
    \\        if (DEFER != 0) for (uint q = 0; q < NB; ++q) xsum += xsc[((ob * NCHC + chunk) * RBC + rr) * NB + q];
    \\
;

/// text_dfinish's planes-in-slice-order sum, deferred map, f16 round and H128 (decb.zig's finish variants reuse them)
pub const dec_finish_v =
    \\        float v[4];
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint o = ob * 128u + L + 32u * j;
    \\            float acc = zc[ulong(row) * ulong(OUTC) + o];
    \\            for (uint k = 1u; k < KSC; ++k) acc += zc[(ulong(k) * ulong(ROWSC) + row) * ulong(OUTC) + o];
    \\            if (DEFER != 0) acc = fma(acc, MUL1_A, MUL1_C * xsum);
    \\            v[j] = float(half(acc));
    \\        }
    \\        for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\        { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\
;
/// dec_finish's output (svh and f16 round) and the end of the finish's row loop
pub const dec_finish_store =
    \\        for (uint j = 0; j < 4u; ++j) { const uint o = ob * 128u + L + 32u * j; y[row * OUTC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
    \\    }
    \\
;
/// text_dfinish's planes-in-slice-order sum, deferred map, f16 round, H128, svh and f16 round
const dec_finish = dec_finish_v ++ dec_finish_store;

/// DEC-ONE over prep_sc's rows: y OUT_T [ROWS, OUT], z f32 [KS * ROWS, OUT] (the partial planes, scratch)
pub const text_dec0: [:0]const u8 = dec_head ++ dec_gemv0 ++ dec_gemv_arrive ++ dec_xsum0 ++ dec_finish;
/// DEC-ONE with the prepare inside: y, z, and xsx f32 (the published input sums, scratch)
pub const text_dec1: [:0]const u8 = dec_head ++ dec_prep1 ++ dec_gemv1 ++ dec_gemv_arrive ++ dec_xsum1 ++ dec_finish;

/// text_prep_s (lane order) that also zeroes the arrival counters of the DEC-ONE launch after it
pub const text_prep_sc: [:0]const u8 = text_prep_s ++
    \\    // DEC-ONE: the arrival counters of the GEMV launch that follows, one per (output block, row chunk)
    \\    for (uint g = (row * NB + b) * 32u + L; g < uint(NCNT); g += uint(NTH)) cnt[g] = 0u;
    \\
;

/// the arrival counters for DEC-ONE with the prepare inside: a launch nothing it reads depends on (svh is a weight), so
/// MLX needs no barrier before it and it runs beside the previous op; the barrier DEC-ONE's own input needs covers it
pub const text_dec_zero =
    \\    const uint i = thread_position_in_grid.x;
    \\    if (i < uint(NCNT)) cnt[i] = 0u;
    \\
;

const dec0_in = [_][*:0]const u8{ "xp", "xs", "w", "svh", "cnt" };
const dec0_out = [_][*:0]const u8{ "y", "z" };
const dec1_in = [_][*:0]const u8{ "x", "suh", "w", "svh", "cnt" };
const dec1_out = [_][*:0]const u8{ "y", "z", "xsx" };
const prep_sc_out = [_][*:0]const u8{ "xp", "xs", "cnt" };
const dec_zero_in = [_][*:0]const u8{"svh"};
const dec_zero_out = [_][*:0]const u8{"cnt"};

const prep_in = [_][*:0]const u8{ "x", "suh" };
const prep_out = [_][*:0]const u8{"xp"};
const gemv_in = [_][*:0]const u8{ "xp", "slots", "w" };
const gemv_out = [_][*:0]const u8{"z"};
const finish_in = [_][*:0]const u8{ "z", "svh" };
const finish_out = [_][*:0]const u8{"y"};
const dequant_in = [_][*:0]const u8{"w"};
const dgemv_in = [_][*:0]const u8{ "xp", "w" };
const dfinish_in = [_][*:0]const u8{ "z", "svh", "xs" };
const prep_s_out = [_][*:0]const u8{ "xp", "xs" };
const dequant_out = [_][*:0]const u8{"W"};
const finishp_in = [_][*:0]const u8{ "zp0", "zp1", "zp2", "zp3", "svh", "R1", "R2", "R3" };

pub const Shape = struct { in: u32, out: u32, rate: format.Rate, dec: format.Decode, out_dtype: mlx.mlx_dtype };

/// A/B aid: MLX_SERVE_TURBOEXL3_DENSE_SMALL_TOGGLE=<path>: while <path> exists the register path is off (dequantize + matmul),
/// so one server alternates the two per request; read only on calls the register path would take (17..small_rows rows).
var small_toggle: ?[*:0]const u8 = null;
var small_toggle_read = false;
extern "c" fn access(path: [*:0]const u8, mode: c_int) c_int;
fn smallToggledOff() bool {
    if (!small_toggle_read) {
        small_toggle_read = true;
        small_toggle = std.c.getenv("MLX_SERVE_TURBOEXL3_DENSE_SMALL_TOGGLE");
        if (small_toggle) |t| std.debug.print("[turboexl3] dense register-path toggle: dequantize + matmul while {s} exists\n", .{std.mem.sliceTo(t, 0)});
    }
    const t = small_toggle orelse return false;
    return access(t, 0) == 0;
}

/// MLX_SERVE_DEC_ONE's value: unset, empty or anything but "0" on (the default: the three-launch chain's bytes either way),
/// "0" off.
pub fn decOneOn(raw: ?[]const u8) bool {
    const v = raw orelse return true;
    return !std.mem.eql(u8, v, "0");
}

/// DEC-ONE's switch, MLX_SERVE_DEC_ONE read once (`decOneOn`)
var dec_env_read = false;
var dec_env_on = false;
var dec_said = false;
fn decEnvOn() bool {
    if (!dec_env_read) {
        dec_env_read = true;
        dec_env_on = decOneOn(if (std.c.getenv("MLX_SERVE_DEC_ONE")) |v| std.mem.span(v) else null);
    }
    return dec_env_on;
}
fn envRows(name: [*:0]const u8, default: u32) u32 {
    const v = std.c.getenv(name) orelse return default;
    return @min(@as(u32, @intCast(decode.Kernels.max_rows)), std.fmt.parseInt(u32, std.mem.span(v), 10) catch default);
}

/// The kernels of one dense linear shape; configs per row count (decode) plus the prefill set.
pub const Dense = struct {
    shape: Shape,
    prep: mlx.mlx_fast_metal_kernel,
    gemv: mlx.mlx_fast_metal_kernel,
    finish: mlx.mlx_fast_metal_kernel,
    dequant: mlx.mlx_fast_metal_kernel,
    dgemv: mlx.mlx_fast_metal_kernel,
    dfinish: mlx.mlx_fast_metal_kernel,
    prep_s: mlx.mlx_fast_metal_kernel,
    /// MUL1 packs: the GEMV sums x times the codeword's byte sum and the finish applies the affine map once per output
    /// (MLX_SERVE_TURBOEXL3_DENSE_DEFER=0 decodes every weight to its f16 value instead, the reference's rounding)
    defer_affine: bool,
    /// K slices of the multi-row GEMV: fixed per shape (so a row's summation order is too)
    ks_split: u32,
    /// MLX_SERVE_TURBOEXL3_DENSE=rowslot: the per-row-slot GEMV (every row decodes the trellis again), for A/B
    rowslot: bool,
    /// prefill rows through the fused GEMM (`prefill.zig`'s one-op text: the trellis decoded per 64-deep stage into threadgroup
    /// memory, M tiles of the single "expert") when MLX_SERVE_TURBOEXL3_DENSE_PREFILL=fused; the default stays dequantize + MLX
    /// matmul, which is faster on every trunk shape (one decode per call beats one per 128-row tile). Shapes need in and out multiples of 64.
    gemm_pf: mlx.mlx_fast_metal_kernel,
    fused_prefill: bool,
    /// the fused GEMM's options: the expert chain's served set (`root.pfOptions`) where the rate and codebook allow it;
    /// MLX_SERVE_TURBOEXL3_DENSE_PF_LEGACY=1 keeps the pre-port set (128-row tiles, one buffer, general reader, f16 decode)
    pf: PfOpts,
    /// rows above the GEMV's max_rows and up to this take the register-path GEMM (`smallInner`); MLX_SERVE_TURBOEXL3_DENSE_SMALL
    /// =<rows> (0 = off, up to small_rows_cap), default 32
    small_rows: u32,
    gemm_rs: mlx.mlx_fast_metal_kernel,
    /// per row count: the register path's prep / finish configs, its launch config and tile table
    small: [small_rows_cap + 1]?Small = @splat(null),
    xs_dummy: mlx.mlx_array = .{ .ctx = null },
    cfg: [decode.Kernels.max_rows + 1]?Cfg = @splat(null),
    dequant_cfg: ?mlx.mlx_fast_metal_kernel_config = null,
    /// DEC-ONE (`decEnvOn`: on unless MLX_SERVE_DEC_ONE=0): decode calls of up to `dec_rows` rows
    /// (MLX_SERVE_DEC_ONE_ROWS, default 4) run the GEMV with the finish in each output block's last-arriving threadgroup.
    /// The prepare runs inside that launch too (one GEMV launch, beside a counter-zeroing launch that waits on nothing) for
    /// calls of up to `dec_prep_rows` rows (MLX_SERVE_DEC_ONE_PREP, default 1) or of up to `dec_prep_out` outputs at any
    /// decode width (MLX_SERVE_DEC_ONE_PREP_OUT, default 1024: the small, latency-bound linears), where its threadgroup
    /// buffer fits; otherwise in prep_sc's launch (two launches). The defaults because every column group of a call redoes
    /// the rotation of its K slice (RB / (8 BT) rotations per tile decode, on an ALU-bound GEMV) and the slice's rows take
    /// up to 20 KB of threadgroup memory at four rows. Bytes are the three-launch chain's in every arm.
    dec_on: bool,
    dec_rows: u32,
    dec_prep_rows: u32,
    dec_prep_out: u32,
    dec0: mlx.mlx_fast_metal_kernel,
    dec1: mlx.mlx_fast_metal_kernel,
    prep_sc: mlx.mlx_fast_metal_kernel,
    dec_zero: mlx.mlx_fast_metal_kernel,
    dec_cfg: [decode.Kernels.max_rows + 1]?DecCfg = @splat(null),
    /// the finish over row pieces (`text_finish_pieces`, `finishPieces`): made on first use, so a process that never
    /// splits a call never builds it
    finish_pieces: ?mlx.mlx_fast_metal_kernel = null,

    pub const Cfg = struct { prep: mlx.mlx_fast_metal_kernel_config, gemv: mlx.mlx_fast_metal_kernel_config, finish: mlx.mlx_fast_metal_kernel_config };
    /// DEC-ONE at one row count: `prep` is prep_sc's config (prepare outside) or dec_zero's (inside); `gemv` dec0's or dec1's
    pub const DecCfg = struct { prep_in: bool, prep: mlx.mlx_fast_metal_kernel_config, gemv: mlx.mlx_fast_metal_kernel_config };
    const Small = struct { io: Cfg, gemm: mlx.mlx_fast_metal_kernel_config, tiles: mlx.mlx_array };
    pub const small_rows_cap: u32 = 128;

    pub const PfOpts = struct {
        bm: u32 = 64,
        dbuf: bool = true,
        unroll: bool = true,
        fragb: bool = true,
        intr: bool = true,
        defer_affine: bool = true,

        pub const legacy: PfOpts = .{ .bm = 128, .dbuf = false, .unroll = false, .fragb = false, .intr = false, .defer_affine = false };

        /// the served set, less what this shape's rate or codebook cannot take
        pub fn forShape(shape: Shape) PfOpts {
            if (std.c.getenv("MLX_SERVE_TURBOEXL3_DENSE_PF_LEGACY")) |v| if (v[0] == '1') return legacy;
            return .{ .intr = shape.rate.n % 16 == 0, .defer_affine = shape.dec.codebook == .mul1 };
        }
    };

    /// M tiles the fused GEMM's host-built table holds; a longer call takes dequantize + matmul
    pub const pf_tiles_cap: usize = 512;

    /// rows sharing one decode per threadgroup (a call's further rows take more chunks); MLX_SERVE_TURBOEXL3_DENSE_RB=1..8 pins
    /// the cap for tuning (more rows per decode means more accumulators and input registers per lane)
    pub fn maxRb() u32 {
        if (std.c.getenv("MLX_SERVE_TURBOEXL3_DENSE_RB")) |v| {
            const k = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
            if (k >= 1 and k <= 8) return k;
        }
        return 4;
    }

    /// output tiles per threadgroup for RB rows: 2 BT RB f32 accumulators per lane (8 at 1-2 rows, 4 RB at 3-4, 2 RB beyond);
    /// MLX_SERVE_TURBOEXL3_DENSE_BT=1|2|4 pins it for tuning
    pub fn btFor(rb: u32) u32 {
        if (std.c.getenv("MLX_SERVE_TURBOEXL3_DENSE_BT")) |v| switch (v[0]) {
            '1' => return 1,
            '2' => return 2,
            '4' => return 4,
            else => {},
        };
        return if (rb == 1) 4 else if (rb <= 4) 2 else 1;
    }

    /// the smallest K split that puts 160 threadgroups (4 per core of a 40-core GPU) in flight at one row (BT 4);
    /// MLX_SERVE_TURBOEXL3_DENSE_KS pins it when it divides the input tiles
    pub fn ksFor(in: u32, out: u32) u32 {
        const q = (in / 16) / 4; // KS divides this: each of the 4 simdgroups of a slice gets IN_TILES / (4 KS) tiles
        if (std.c.getenv("MLX_SERVE_TURBOEXL3_DENSE_KS")) |v| {
            const k = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
            if (k >= 1 and q % k == 0) return k;
        }
        const groups = (out / 16) / 4;
        // MLX_SERVE_TURBOEXL3_DENSE_PERMAX=<tiles>: at most this many input tiles per simdgroup (its K loop is one dependent
        // chain of loads, the latency floor at verify widths); still the shape's own split, so every row count sums alike
        const permax: u32 = if (std.c.getenv("MLX_SERVE_TURBOEXL3_DENSE_PERMAX")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0 else 0;
        var d: u32 = 1;
        while (d <= q) : (d += 1) {
            if (q % d != 0) continue;
            if (permax != 0 and q / d > permax) continue;
            if (groups * d >= 160) return d;
        }
        return q;
    }

    /// whether a decode with `dec` takes the deferred MUL1 map (MLX_SERVE_TURBOEXL3_DENSE_DEFER=0 turns it off)
    pub fn deferFor(dec: format.Decode) bool {
        if (dec.codebook != .mul1) return false;
        const v = std.c.getenv("MLX_SERVE_TURBOEXL3_DENSE_DEFER") orelse return true;
        return v[0] != '0';
    }

    pub fn init(shape: Shape) !Dense {
        if (shape.in % 128 != 0 or shape.out % 128 != 0 or (shape.in / 16) % 4 != 0 or shape.in == 0 or shape.out == 0) return error.BadExl3Shape;
        return .{
            .shape = shape,
            .prep = ks.newKernel("turboexl3_dense_prep", &prep_in, &prep_out, text_prep, decode.header, true) orelse return error.KernelCreateFailed,
            .gemv = ks.newKernel("turboexl3_dense_gemv", &gemv_in, &gemv_out, unfused.text_gemv, decode.header, true) orelse return error.KernelCreateFailed,
            .finish = ks.newKernel("turboexl3_dense_finish", &finish_in, &finish_out, text_finish, decode.header, true) orelse return error.KernelCreateFailed,
            .dequant = ks.newKernel("turboexl3_dense_dequant", &dequant_in, &dequant_out, text_dequant, decode.header, true) orelse return error.KernelCreateFailed,
            .dgemv = ks.newKernel("turboexl3_dense_dgemv", &dgemv_in, &gemv_out, text_dgemv, decode.header, true) orelse return error.KernelCreateFailed,
            .dfinish = ks.newKernel("turboexl3_dense_dfinish", &dfinish_in, &finish_out, text_dfinish, decode.header, true) orelse return error.KernelCreateFailed,
            .prep_s = ks.newKernel("turboexl3_dense_prep_s", &prep_in, &prep_s_out, text_prep_s, decode.header, true) orelse return error.KernelCreateFailed,
            .defer_affine = deferFor(shape.dec),
            .ks_split = ksFor(shape.in, shape.out),
            .rowslot = blk: {
                const v = std.c.getenv("MLX_SERVE_TURBOEXL3_DENSE") orelse break :blk false;
                break :blk std.mem.eql(u8, std.mem.span(v), "rowslot");
            },
            .gemm_pf = ks.newKernel("turboexl3_dense_pf_gemm", &prefill.gemm1_in, &prefill.gemm1_out, prefill.gemm1_text, prefill.header_nax, true) orelse return error.KernelCreateFailed,
            .fused_prefill = shape.in % 64 == 0 and shape.out % 64 == 0 and blk: {
                const v = std.c.getenv("MLX_SERVE_TURBOEXL3_DENSE_PREFILL") orelse break :blk false;
                break :blk std.mem.eql(u8, std.mem.span(v), "fused");
            },
            .pf = PfOpts.forShape(shape),
            .small_rows = blk: {
                const v = std.c.getenv("MLX_SERVE_TURBOEXL3_DENSE_SMALL") orelse break :blk 32;
                break :blk @min(small_rows_cap, std.fmt.parseInt(u32, std.mem.span(v), 10) catch 32);
            },
            .gemm_rs = ks.newKernel("turboexl3_dense_rs_gemm", &prefill.gemm1_in, &prefill.gemm1_out, prefill.gemm1s_text, prefill.header_nax_small, true) orelse return error.KernelCreateFailed,
            .dec_on = decEnvOn(),
            .dec_rows = envRows("MLX_SERVE_DEC_ONE_ROWS", 4),
            .dec_prep_rows = envRows("MLX_SERVE_DEC_ONE_PREP", 1),
            .dec_prep_out = blk: {
                const v = std.c.getenv("MLX_SERVE_DEC_ONE_PREP_OUT") orelse break :blk 1024;
                break :blk std.fmt.parseInt(u32, std.mem.span(v), 10) catch 1024;
            },
            .dec0 = ks.newKernel("turboexl3_dense_dec0", &dec0_in, &dec0_out, text_dec0, decode.header5, true) orelse return error.KernelCreateFailed,
            .dec1 = ks.newKernel("turboexl3_dense_dec1", &dec1_in, &dec1_out, text_dec1, decode.header5, true) orelse return error.KernelCreateFailed,
            .prep_sc = ks.newKernel("turboexl3_dense_prep_sc", &prep_in, &prep_sc_out, text_prep_sc, decode.header, true) orelse return error.KernelCreateFailed,
            .dec_zero = ks.newKernel("turboexl3_dense_dec_zero", &dec_zero_in, &dec_zero_out, text_dec_zero, decode.header, true) orelse return error.KernelCreateFailed,
        };
    }

    /// Whether DEC-ONE's prepare can run inside its launch for this geometry: the K slice whole 128-blocks (PER even), and the
    /// slice's prepared rows (RB x SPAN halves) beside the reduction buffer within Metal's 32 KB of threadgroup memory.
    pub fn decPrepFits(shape: Shape, ks_split: u32, rb: u32, bt: u32) bool {
        const per = (shape.in / 16) / (ks_split * 4);
        const span = 64 * per;
        if (span == 0 or span % 128 != 0) return false;
        return rb * span * 2 + 4 * rb * 16 * bt * 4 + 16 <= 32768;
    }

    /// DEC-ONE's configs for `rows` rows (cached): the GEMV launch in text_dgemv's grid (so its tiles, K slices and row chunks)
    pub fn decCfgs(self: *Dense, rows: usize) !DecCfg {
        if (rows == 0 or rows > decode.Kernels.max_rows) return error.BadExl3Shape;
        if (self.dec_cfg[rows]) |c| return c;
        const r: c_int = @intCast(rows);
        const rows_u: u32 = @intCast(rows);
        const IN: c_int = @intCast(self.shape.in);
        const OUT: c_int = @intCast(self.shape.out);
        const t = self.templ();
        const rb: u32 = @min(rows_u, maxRb());
        const bt = btFor(rb);
        if (8 % bt != 0) return error.BadExl3Shape;
        const chunks: u32 = (rows_u + rb - 1) / rb;
        const ksv: c_int = @intCast(self.ks_split);
        const nob = self.shape.out / 128;
        const nb = self.shape.in / 128;
        // at least 8 counters: MLX binds an input of fewer than 8 elements in the constant address space (no atomics there)
        const ncnt: u32 = @max(nob * chunks, 8);
        const inside = (rows_u <= self.dec_prep_rows or self.shape.out <= self.dec_prep_out) and decPrepFits(self.shape, self.ks_split, rb, bt);
        const tmpl = [_]ks.TemplateArg{
            t[0], t[1], t[2],
            .{ .name = "IN", .value = .{ .int = IN } },
            .{ .name = "OUT", .value = .{ .int = OUT } },
            .{ .name = "KS", .value = .{ .int = ksv } },
            .{ .name = "BT", .value = .{ .int = @intCast(bt) } },
            .{ .name = "RB", .value = .{ .int = @intCast(rb) } },
            .{ .name = "ROWS", .value = .{ .int = r } },
            .{ .name = "DEFER", .value = .{ .int = if (self.defer_affine) 1 else 0 } },
            .{ .name = "OUT_T", .value = .{ .dtype = self.shape.out_dtype } },
        };
        const grid = [3]u32{ (self.shape.out / 16 / bt) * 128, chunks, self.ks_split };
        const ncnt_shape = [_]c_int{@intCast(ncnt)};
        const y_shape = [_]c_int{ r, OUT };
        const z_shape = [_]c_int{ ksv * r, OUT };
        var c: DecCfg = undefined;
        if (inside) {
            const nx: u32 = if (self.defer_affine) @max(nob * chunks * rb * nb, 8) else 8;
            const prep = try ks.buildConfig(.{ .grid = .{ ((ncnt + 127) / 128) * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&ncnt_shape}, .out_dtypes = &.{.uint32}, .template = &.{.{ .name = "NCNT", .value = .{ .int = @intCast(ncnt) } }} });
            errdefer _ = mlx.mlx_fast_metal_kernel_config_free(prep);
            c = .{ .prep_in = true, .prep = prep, .gemv = try ks.buildConfig(.{ .grid = grid, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &y_shape, &z_shape, &[_]c_int{@intCast(nx)} }, .out_dtypes = &.{ self.shape.out_dtype, .float32, .float32 }, .template = &tmpl }) };
        } else {
            const nth: u32 = nb * 32 * rows_u;
            const prep = try ks.buildConfig(.{ .grid = .{ nb * 32, rows_u, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{ &[_]c_int{ r, IN }, &[_]c_int{ r, @intCast(nb) }, &ncnt_shape }, .out_dtypes = &.{ .float16, .float32, .uint32 }, .template = &.{
                .{ .name = "DIM", .value = .{ .int = IN } },
                .{ .name = "LANE_ORDER", .value = .{ .int = 1 } },
                .{ .name = "NCNT", .value = .{ .int = @intCast(ncnt) } },
                .{ .name = "NTH", .value = .{ .int = @intCast(nth) } },
            } });
            errdefer _ = mlx.mlx_fast_metal_kernel_config_free(prep);
            c = .{ .prep_in = false, .prep = prep, .gemv = try ks.buildConfig(.{ .grid = grid, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &y_shape, &z_shape }, .out_dtypes = &.{ self.shape.out_dtype, .float32 }, .template = &tmpl }) };
        }
        self.dec_cfg[rows] = c;
        return c;
    }

    /// Whether DEC-ONE takes this call (the switch and the row cap; the row-slot A/B path keeps its kernels)
    pub fn decTakes(self: *const Dense, rows: usize) bool {
        return self.dec_on and !self.rowslot and rows >= 1 and rows <= self.dec_rows;
    }

    /// The prepare a DEC-ONE call of `rows` rows takes: inside its GEMV launch or in prep_sc's
    pub fn decPrepInside(self: *Dense, rows: usize) !bool {
        return (try self.decCfgs(rows)).prep_in;
    }

    /// DEC-ONE: y [rows, out] in `shape.out_dtype`, the three-launch chain's bytes
    fn decLinear(self: *Dense, s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        const c = try self.decCfgs(rows);
        if (!dec_said) {
            dec_said = true;
            std.debug.print("[turboexl3] DEC-ONE engaged: one GEMV launch per dense linear at decode rows <= {d}\n", .{self.dec_rows});
        }
        if (c.prep_in) {
            const cnt = try one(self.dec_zero, c.prep, &.{svh}, s);
            defer _ = mlx.mlx_array_free(cnt);
            var o: [3]mlx.mlx_array = undefined;
            try ks.applyConfig(self.dec1, c.gemv, &.{ x, suh, trellis, svh, cnt }, &o, s);
            _ = mlx.mlx_array_free(o[1]);
            _ = mlx.mlx_array_free(o[2]);
            return o[0];
        }
        var po: [3]mlx.mlx_array = undefined;
        try ks.applyConfig(self.prep_sc, c.prep, &.{ x, suh }, &po, s);
        defer for (po) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var o: [2]mlx.mlx_array = undefined;
        try ks.applyConfig(self.dec0, c.gemv, &.{ po[0], po[1], trellis, svh, po[2] }, &o, s);
        _ = mlx.mlx_array_free(o[1]);
        return o[0];
    }

    pub fn deinit(self: *Dense) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep, cc.gemv, cc.finish }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        if (self.dequant_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        for (&self.small) |*c| if (c.*) |cc| {
            inline for (.{ cc.io.prep, cc.io.finish, cc.gemm }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            _ = mlx.mlx_array_free(cc.tiles);
            c.* = null;
        };
        if (self.xs_dummy.ctx != null) _ = mlx.mlx_array_free(self.xs_dummy);
        for (&self.dec_cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep, cc.gemv }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        inline for (.{ self.prep, self.gemv, self.finish, self.dequant, self.dgemv, self.dfinish, self.prep_s, self.gemm_pf, self.gemm_rs, self.dec0, self.dec1, self.prep_sc, self.dec_zero }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
        if (self.finish_pieces) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
        self.finish_pieces = null;
    }

    pub fn templ(self: *const Dense) [3]ks.TemplateArg {
        return .{
            .{ .name = "NHW", .value = .{ .int = @intCast(self.shape.rate.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(self.shape.dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(self.shape.dec.window.bits()) } },
        };
    }

    /// prepare and finish configs for `rows` rows (any count); the GEMV config only up to max_rows
    pub fn cfgs(self: *Dense, rows: usize) !Cfg {
        if (rows == 0 or rows > decode.Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const c = if (self.rowslot) try self.buildCfg(rows, true) else try self.buildMultiCfg(rows);
        self.cfg[rows] = c;
        return c;
    }

    /// prepare (lane order), the multi-row GEMV over row chunks of RB = min(rows, 8) and its KS-plane finish
    fn buildMultiCfg(self: *const Dense, rows: usize) !Cfg {
        const r: c_int = @intCast(rows);
        const IN: c_int = @intCast(self.shape.in);
        const OUT: c_int = @intCast(self.shape.out);
        const t = self.templ();
        const rb: u32 = @min(@as(u32, @intCast(rows)), maxRb());
        const bt = btFor(rb);
        const chunks: u32 = (@as(u32, @intCast(rows)) + rb - 1) / rb;
        const ksv: c_int = @intCast(self.ks_split);
        const out_t = ks.TemplateArg{ .name = "OUT_T", .value = .{ .dtype = self.shape.out_dtype } };
        const defer_t = ks.TemplateArg{ .name = "DEFER", .value = .{ .int = if (self.defer_affine) 1 else 0 } };
        return .{
            .prep = try ks.buildConfig(.{ .grid = .{ (self.shape.in / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{ &[_]c_int{ r, IN }, &[_]c_int{ r, @intCast(self.shape.in / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = &.{ .{ .name = "DIM", .value = .{ .int = IN } }, .{ .name = "LANE_ORDER", .value = .{ .int = 1 } } } }),
            .gemv = try ks.buildConfig(.{ .grid = .{ (self.shape.out / 16 / bt) * 128, chunks, self.ks_split }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ ksv * r, OUT }}, .out_dtypes = &.{.float32}, .template = &.{ t[0], t[1], t[2], .{ .name = "IN", .value = .{ .int = IN } }, .{ .name = "OUT", .value = .{ .int = OUT } }, .{ .name = "KS", .value = .{ .int = ksv } }, .{ .name = "BT", .value = .{ .int = @intCast(bt) } }, .{ .name = "RB", .value = .{ .int = @intCast(rb) } }, .{ .name = "ROWS", .value = .{ .int = r } }, defer_t } }),
            .finish = try ks.buildConfig(.{ .grid = .{ (self.shape.out / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ r, OUT }}, .out_dtypes = &.{self.shape.out_dtype}, .template = &.{ .{ .name = "DIM", .value = .{ .int = OUT } }, out_t, .{ .name = "KS", .value = .{ .int = ksv } }, .{ .name = "ROWS", .value = .{ .int = r } }, .{ .name = "INBLK", .value = .{ .int = @intCast(self.shape.in / 128) } }, defer_t } }),
        };
    }

    fn buildCfg(self: *const Dense, rows: usize, with_gemv: bool) !Cfg {
        const r: c_int = @intCast(rows);
        const IN: c_int = @intCast(self.shape.in);
        const OUT: c_int = @intCast(self.shape.out);
        const t = self.templ();
        const out_t = ks.TemplateArg{ .name = "OUT_T", .value = .{ .dtype = self.shape.out_dtype } };
        return .{
            .prep = try ks.buildConfig(.{ .grid = .{ (self.shape.in / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ r, IN }}, .out_dtypes = &.{.float16}, .template = &.{ .{ .name = "DIM", .value = .{ .int = IN } }, .{ .name = "LANE_ORDER", .value = .{ .int = if (with_gemv) 1 else 0 } } } }),
            .gemv = if (with_gemv) try ks.buildConfig(.{ .grid = .{ (self.shape.out / 128) * 128, @intCast(rows), 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ r, OUT }}, .out_dtypes = &.{.float32}, .template = &.{ t[0], t[1], t[2], .{ .name = "IN", .value = .{ .int = IN } }, .{ .name = "OUT", .value = .{ .int = OUT } } } }) else undefined,
            .finish = try ks.buildConfig(.{ .grid = .{ (self.shape.out / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ r, OUT }}, .out_dtypes = &.{self.shape.out_dtype}, .template = &.{ .{ .name = "DIM", .value = .{ .int = OUT } }, out_t } }),
        };
    }

    fn one(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, c, ins, &outs, s);
        return outs[0];
    }

    /// W half [in, out] decoded from the trellis (the prefill operand); the caller frees it. On an armed resident table,
    /// for an admitted trellis, a new reference to the table's kept W_hat (`residentTake`): this launch's bytes, decoded
    /// once.
    pub fn dequantize(self: *Dense, s: mlx.mlx_stream, trellis: mlx.mlx_array) !mlx.mlx_array {
        if (Resident.armed.load(.acquire) and !resident_bypass) if (try residentTake(self, s, trellis)) |w| return w;
        return self.decodeTrellis(s, trellis);
    }

    /// The dequantize launch: a fresh W half [in, out] from the trellis (the un-rotation reads it too).
    pub fn decodeTrellis(self: *Dense, s: mlx.mlx_stream, trellis: mlx.mlx_array) !mlx.mlx_array {
        if (self.dequant_cfg == null) {
            const tiles = (self.shape.in / 16) * (self.shape.out / 16);
            const t = self.templ();
            self.dequant_cfg = try ks.buildConfig(.{ .grid = .{ ((tiles + 3) / 4) * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(self.shape.in), @intCast(self.shape.out) }}, .out_dtypes = &.{.float16}, .template = &.{ t[0], t[1], t[2], .{ .name = "IN", .value = .{ .int = @intCast(self.shape.in) } }, .{ .name = "OUT", .value = .{ .int = @intCast(self.shape.out) } } } });
        }
        return one(self.dequant, self.dequant_cfg.?, &.{trellis}, s);
    }

    /// The fused GEMM over prepared rows: z half [rows, out] = xp [rows, in] x W (decoded per stage from the trellis), the
    /// rows in `pf.bm`-row M tiles of the one "expert" (a host-built tile table: the row count is known here). `xs` f32
    /// [rows, in / 128] (prep_s's block sums) when `pf.defer_affine`: the epilogue's C sum(x).
    pub fn fusedInner(self: *Dense, s: mlx.mlx_stream, xp: mlx.mlx_array, xs: ?mlx.mlx_array, trellis: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        const o = self.pf;
        const bm: usize = o.bm;
        const tmax: usize = (rows + bm - 1) / bm;
        if (tmax > pf_tiles_cap or (o.defer_affine and xs == null)) return error.BadExl3Shape;
        var tbuf: [4 * pf_tiles_cap]i32 = @splat(0);
        for (0..tmax) |t| {
            tbuf[4 * t] = 0;
            tbuf[4 * t + 1] = @intCast(t * bm);
            tbuf[4 * t + 2] = @intCast(@min(bm, rows - t * bm));
            tbuf[4 * t + 3] = 0;
        }
        // at least two rows (8 ints): MLX binds an input of fewer than 8 elements in the constant address space
        const tiles = mlx.mlx_array_new_data(&tbuf, &[_]c_int{ @intCast(@max(tmax, 2)), 4 }, 2, .int32);
        if (tiles.ctx == null) return error.MlxError;
        defer _ = mlx.mlx_array_free(tiles);
        const t = self.templ();
        const tmpl = [_]ks.TemplateArg{
            t[0], t[1], t[2],
            .{ .name = "KD", .value = .{ .int = @intCast(self.shape.in) } },
            .{ .name = "ND", .value = .{ .int = @intCast(self.shape.out) } },
            .{ .name = "NOPS", .value = .{ .int = 1 } },
            .{ .name = "BMR", .value = .{ .int = @intCast(bm) } },
            .{ .name = "DBUF", .value = .{ .int = @intFromBool(o.dbuf) } },
            .{ .name = "PROBE", .value = .{ .int = 0 } },
            .{ .name = "WSP", .value = .{ .int = 0 } },
            .{ .name = "MFIRST", .value = .{ .int = 0 } },
            .{ .name = "UNR", .value = .{ .int = @intFromBool(o.unroll) } },
            .{ .name = "FRAGB", .value = .{ .int = @intFromBool(o.fragb) } },
            .{ .name = "INTR", .value = .{ .int = @intFromBool(o.intr) } },
            .{ .name = "DEFER", .value = .{ .int = @intFromBool(o.defer_affine) } },
            .{ .name = "NBX", .value = .{ .int = @intCast(self.shape.in / 128) } },
            .{ .name = "BKS", .value = .{ .int = @intCast(prefill.BK) } },
            .{ .name = "NCB", .value = .{ .int = 1 } },
            // the expert chain's A layout and first-fragment-row MMA (`prefill.Options.alay` / `half_rows`) off: this prep writes
            // natural order
            .{ .name = "ALAY", .value = .{ .int = 0 } },
            .{ .name = "H16", .value = .{ .int = 0 } },
        };
        const zshape = [_]c_int{ @intCast(rows), @intCast(self.shape.out) };
        const cfg = try ks.buildConfig(.{ .grid = .{ @as(u32, @intCast(tmax)) * (self.shape.out / prefill.BN) * 2 * @as(u32, @intCast(bm)), 1, 1 }, .threadgroup = .{ 2 * @as(u32, @intCast(bm)), 1, 1 }, .out_shapes = &.{&zshape}, .out_dtypes = &.{.float16}, .template = &tmpl });
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        if (xs) |sums| return one(self.gemm_pf, cfg, &.{ tiles, xp, trellis, sums }, s);
        var xs_dummy = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xs_dummy);
        try mlx.check(mlx.mlx_zeros(&xs_dummy, &[_]c_int{8}, 1, .float32, s));
        return one(self.gemm_pf, cfg, &.{ tiles, xp, trellis, xs_dummy }, s);
    }

    /// The register path's configs for `rows` rows (cached): prep in natural order, finish, the GEMM launch over 32-row tiles of
    /// the one "expert" (tile t = rows 32t..; the item count is a multiple of SGT since out % 128 == 0) and its tile table.
    fn smallFor(self: *Dense, s: mlx.mlx_stream, rows: usize) !Small {
        if (rows == 0 or rows > small_rows_cap) return error.BadExl3Shape;
        if (self.small[rows]) |c| return c;
        const io = try self.buildCfg(rows, false);
        errdefer {
            _ = mlx.mlx_fast_metal_kernel_config_free(io.prep);
            _ = mlx.mlx_fast_metal_kernel_config_free(io.finish);
        }
        const tmax: usize = (rows + 31) / 32;
        var tbuf: [4 * (small_rows_cap / 32)]i32 = @splat(0);
        for (0..tmax) |t| {
            tbuf[4 * t] = 0;
            tbuf[4 * t + 1] = @intCast(t * 32);
            tbuf[4 * t + 2] = @intCast(@min(32, rows - t * 32));
            tbuf[4 * t + 3] = 0;
        }
        // at least two rows (8 ints): MLX binds an input of fewer than 8 elements in the constant address space
        const tiles = mlx.mlx_array_new_data(&tbuf, &[_]c_int{ @intCast(@max(tmax, 2)), 4 }, 2, .int32);
        if (tiles.ctx == null) return error.MlxError;
        errdefer _ = mlx.mlx_array_free(tiles);
        try mlx.check(mlx.mlx_array_eval(tiles));
        if (self.xs_dummy.ctx == null) {
            var z = mlx.mlx_array_new();
            // 8 floats: a smaller input would be bound as `constant`, and the GEMM's (DEFER-off) affine call takes `device`
            try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{8}, 1, .float32, s));
            try mlx.check(mlx.mlx_array_eval(z));
            self.xs_dummy = z;
        }
        const t = self.templ();
        const sgt: u32 = 4;
        const tmpl = [_]ks.TemplateArg{
            t[0], t[1], t[2],
            .{ .name = "KD", .value = .{ .int = @intCast(self.shape.in) } },
            .{ .name = "ND", .value = .{ .int = @intCast(self.shape.out) } },
            .{ .name = "NOPS", .value = .{ .int = 1 } },
            .{ .name = "SGT", .value = .{ .int = @intCast(sgt) } },
            .{ .name = "DEFER", .value = .{ .int = 0 } },
            .{ .name = "NBX", .value = .{ .int = @intCast(self.shape.in / 128) } },
            .{ .name = "SINTR", .value = .{ .int = if (self.shape.rate.n % 16 == 0) 1 else 0 } },
            // natural-order prep, whole fragments (`prefill.Options.alay` / `half_rows` off)
            .{ .name = "ALAY", .value = .{ .int = 0 } },
            .{ .name = "H16", .value = .{ .int = 0 } },
        };
        const items: u32 = @as(u32, @intCast(tmax)) * (self.shape.out / 32);
        const zshape = [_]c_int{ @intCast(rows), @intCast(self.shape.out) };
        const gemm = try ks.buildConfig(.{ .grid = .{ ((items + sgt - 1) / sgt) * 32 * sgt, 1, 1 }, .threadgroup = .{ 32 * sgt, 1, 1 }, .out_shapes = &.{&zshape}, .out_dtypes = &.{.float16}, .template = &tmpl });
        const c: Small = .{ .io = io, .gemm = gemm, .tiles = tiles };
        self.small[rows] = c;
        return c;
    }

    /// The register-path inner product z half [rows, out] of prepared rows (natural order): every weight decoded once per
    /// 32-row tile into the B fragments (f16 values: DEFER off), f32 accumulation, rounded to f16 like the matmul path.
    pub fn smallInner(self: *Dense, s: mlx.mlx_stream, xp: mlx.mlx_array, trellis: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        const c = try self.smallFor(s, rows);
        return one(self.gemm_rs, c.gemm, &.{ c.tiles, xp, trellis, self.xs_dummy }, s);
    }

    /// What `linear` runs at `rows` rows: its branches in its order, the decode GEMV (prep_s, dgemv, dfinish), the register-path
    /// GEMM, the fused GEMM (MLX_SERVE_TURBOEXL3_DENSE_PREFILL=fused) or dequantize + MLX matmul; the register path's toggle file
    /// read once, as one `linear` call reads it.
    pub const Path = enum { gemv, small, fused, matmul };

    pub fn pathFor(self: *const Dense, rows: usize) Path {
        if (rows <= decode.Kernels.max_rows) return .gemv;
        if (rows <= self.small_rows and !self.fused_prefill and !smallToggledOff()) return .small;
        if (self.fused_prefill and (rows + self.pf.bm - 1) / self.pf.bm <= pf_tiles_cap) return .fused;
        return .matmul;
    }

    /// `linear`'s inner product on `path` (.small or .matmul) of prepared rows xp half [rows, in] (natural order: `text_prep`
    /// with LANE_ORDER 0, what both paths read): z half [rows, out] by the launches `linear` makes there (the register path's
    /// cached config and tile table; the dequantized W_hat and MLX's matmul), so the same bytes. Other paths refuse.
    pub fn innerOf(self: *Dense, s: mlx.mlx_stream, xp: mlx.mlx_array, trellis: mlx.mlx_array, rows: usize, path: Path) !mlx.mlx_array {
        switch (path) {
            .small => return self.smallInner(s, xp, trellis, rows),
            .matmul => {
                const W = try self.dequantize(s, trellis);
                defer _ = mlx.mlx_array_free(W);
                var inner = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(inner);
                try mlx.check(mlx.mlx_matmul(&inner, xp, W, s));
                return inner;
            },
            .gemv, .fused => return error.Exl3EngineNotImplemented,
        }
    }

    /// `linear`'s prepare at prefill rows (the register path's and the matmul's: natural order), xp half [rows, in]; with
    /// `finishOf` and `innerOf` the stage references of the byte tests
    pub fn prepare(self: *const Dense, s: mlx.mlx_stream, x: mlx.mlx_array, suh: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        const c = try self.buildCfg(rows, false);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c.prep);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c.finish);
        return one(self.prep, c.prep, &.{ x, suh }, s);
    }

    /// `linear`'s finish at prefill rows of an inner product z half [rows, out]: y [rows, out] in `shape.out_dtype`
    pub fn finishOf(self: *const Dense, s: mlx.mlx_stream, z: mlx.mlx_array, svh: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        const c = try self.buildCfg(rows, false);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c.prep);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c.finish);
        return one(self.finish, c.finish, &.{ z, svh }, s);
    }

    /// `linearOn`'s launches up to its finish (the host's fused GDN norm-gate runs from there): the inner product z half
    /// [rows, out] of x [rows, in] its finish would read, the register path's prepare and GEMM on `s`; on the matmul path
    /// the prepare on `s`, the dequantize and the matmul on `rs`. The same configs and operands as `linearOn`, so the same
    /// bytes. The GEMV's rows and the fused-GEMM option refuse (error.Exl3EngineNotImplemented) before any launch.
    pub fn innerOn(self: *Dense, s: mlx.mlx_stream, rs: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        switch (self.pathFor(rows)) {
            .small => {
                const c = try self.smallFor(s, rows);
                const xp = try one(self.prep, c.io.prep, &.{ x, suh }, s);
                defer _ = mlx.mlx_array_free(xp);
                return one(self.gemm_rs, c.gemm, &.{ c.tiles, xp, trellis, self.xs_dummy }, s);
            },
            .matmul => {
                const c = try self.buildCfg(rows, false);
                defer _ = mlx.mlx_fast_metal_kernel_config_free(c.prep);
                defer _ = mlx.mlx_fast_metal_kernel_config_free(c.finish);
                const xp = try one(self.prep, c.prep, &.{ x, suh }, s);
                defer _ = mlx.mlx_array_free(xp);
                const W = try self.dequantize(rs, trellis);
                defer _ = mlx.mlx_array_free(W);
                var inner = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(inner);
                try mlx.check(mlx.mlx_matmul(&inner, xp, W, rs));
                return inner;
            },
            .gemv, .fused => return error.Exl3EngineNotImplemented,
        }
    }

    /// The most row pieces a call splits into (the host's GDN overlap; `finishPieces` reads up to four)
    pub const max_pieces: usize = 4;

    /// Whether `linear` takes the matmul path at the whole call's rows and at every piece's (`rows`: the pieces'
    /// row counts, 2..max_pieces of them), the path whose launches `pieceInner` and `finishPieces` split
    pub fn piecesServe(self: *const Dense, rows: []const usize) bool {
        if (rows.len < 2 or rows.len > max_pieces) return false;
        var total: usize = 0;
        for (rows) |r| {
            if (r == 0 or self.pathFor(r) != .matmul) return false;
            total += r;
        }
        return self.pathFor(total) == .matmul;
    }

    /// One piece's inner product z half [rows, out]: `linearOn`'s prepare of x [rows, in] (its config at these rows)
    /// and its matmul against W (`dequantize`'s, made once for every piece of the call), both on `s`
    pub fn pieceInner(self: *Dense, s: mlx.mlx_stream, x: mlx.mlx_array, W: mlx.mlx_array, suh: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        if (self.pathFor(rows) != .matmul) return error.Exl3EngineNotImplemented;
        const c = try self.buildCfg(rows, false);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c.prep);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c.finish);
        const xp = try one(self.prep, c.prep, &.{ x, suh }, s);
        defer _ = mlx.mlx_array_free(xp);
        var inner = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(inner);
        try mlx.check(mlx.mlx_matmul(&inner, xp, W, s));
        return inner;
    }

    /// One piece's inner product z half [rows, out] from rows the host prepared (xp half [rows, in],
    /// natural order: what `innerOf` reads on the matmul path, as `linearPrepared` takes it): `pieceInner`'s matmul against W
    /// without its prepare, on `s`
    pub fn piecePrepared(self: *const Dense, s: mlx.mlx_stream, xp: mlx.mlx_array, W: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        if (self.pathFor(rows) != .matmul) return error.Exl3EngineNotImplemented;
        var inner = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(inner);
        try mlx.check(mlx.mlx_matmul(&inner, xp, W, s));
        return inner;
    }

    /// `linear`'s finish over a call whose inner product arrives as 2..max_pieces row pieces (f16 [rows_i, out], in
    /// row order), one launch (`text_finish_pieces`: each row reads its own inner-product row): y [sum rows_i, out] in
    /// `shape.out_dtype`. With `pieceInner` per piece that is `linear` of the concatenated rows bit for bit (with
    /// `piecePrepared`, `linearPrepared` of the concatenated prepared rows) wherever MLX's matmul runs the whole call's kernel
    /// at every piece's row count (the caller's check: the host's `gdnOverlapGemmSame`).
    pub fn finishPieces(self: *Dense, s: mlx.mlx_stream, inners: []const mlx.mlx_array, svh: mlx.mlx_array) !mlx.mlx_array {
        const n = inners.len;
        if (n < 2 or n > max_pieces) return error.BadExl3Shape;
        const OUT: c_int = @intCast(self.shape.out);
        // bounds[i] = the call's first row of piece i
        var bounds: [max_pieces + 1]usize = @splat(0);
        for (inners, 0..) |z, i| {
            const sh = mlx.getShape(z);
            if (sh.len != 2 or sh[0] < 1 or sh[1] != OUT or mlx.mlx_array_dtype(z) != .float16) return error.BadExl3Shape;
            bounds[i + 1] = bounds[i] + @as(usize, @intCast(sh[0]));
        }
        const rows = bounds[n];
        if (self.finish_pieces == null) self.finish_pieces = ks.newKernel("turboexl3_dense_finish_pieces", &finishp_in, &finish_out, text_finish_pieces, decode.header, true) orelse return error.KernelCreateFailed;
        // R1..R3: the pieces' first rows; an unused piece's bound is the call's rows (no row reads it) and its input piece 0
        var bound_args: [3]mlx.mlx_array = undefined;
        for (&bound_args, 1..) |*a, p| a.* = mlx.mlx_array_new_int(@intCast(if (p < n) bounds[p] else rows));
        defer for (bound_args) |a| {
            _ = mlx.mlx_array_free(a);
        };
        // `linear`'s finish launch (`buildCfg(rows, false).finish`): its grid, threadgroup, output and template arguments
        const cfg = try ks.buildConfig(.{ .grid = .{ (self.shape.out / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), OUT }}, .out_dtypes = &.{self.shape.out_dtype}, .template = &.{ .{ .name = "DIM", .value = .{ .int = OUT } }, .{ .name = "OUT_T", .value = .{ .dtype = self.shape.out_dtype } } } });
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const ins = [_]mlx.mlx_array{ inners[0], inners[1], if (n > 2) inners[2] else inners[0], if (n > 3) inners[3] else inners[0], svh, bound_args[0], bound_args[1], bound_args[2] };
        return one(self.finish_pieces.?, cfg, &ins, s);
    }

    /// y [rows, out] in `shape.out_dtype` for x [rows, in] (any float dtype); the caller frees y.
    pub fn linear(self: *Dense, s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        return self.linearOn(s, s, x, trellis, suh, svh, rows);
    }

    /// `linear` with the prefill rows' inner product (dequantize + matmul, or the fused GEMM) and finish on `rs`, the prepare
    /// on `s` (`rs` the host's side stream: its launches start once the prepared rows exist). The same launches, operands
    /// and graph as `linear`, only the stream each launch is recorded on, so the same bytes. Rows the GEMV or the register
    /// path take run every launch on `s`.
    pub fn linearOn(self: *Dense, s: mlx.mlx_stream, rs: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        if (rows <= decode.Kernels.max_rows) {
            if (self.decTakes(rows)) return self.decLinear(s, x, trellis, suh, svh, rows);
            const c = try self.cfgs(rows);
            if (!self.rowslot) {
                var po: [2]mlx.mlx_array = undefined;
                try ks.applyConfig(self.prep_s, c.prep, &.{ x, suh }, &po, s);
                defer for (po) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                const z = try one(self.dgemv, c.gemv, &.{ po[0], trellis }, s);
                defer _ = mlx.mlx_array_free(z);
                return one(self.dfinish, c.finish, &.{ z, svh, po[1] }, s);
            }
            const xp = try one(self.prep, c.prep, &.{ x, suh }, s);
            defer _ = mlx.mlx_array_free(xp);
            // every row-slot is row r of expert 0: a zero slot per row
            var slots = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(slots);
            try mlx.check(mlx.mlx_zeros(&slots, &[_]c_int{@intCast(rows)}, 1, .uint32, s));
            const z = try one(self.gemv, c.gemv, &.{ xp, slots, trellis }, s);
            defer _ = mlx.mlx_array_free(z);
            return one(self.finish, c.finish, &.{ z, svh }, s);
        }
        if (rows <= self.small_rows and !self.fused_prefill and !smallToggledOff()) {
            const c = try self.smallFor(s, rows);
            const xp = try one(self.prep, c.io.prep, &.{ x, suh }, s);
            defer _ = mlx.mlx_array_free(xp);
            const z = try one(self.gemm_rs, c.gemm, &.{ c.tiles, xp, trellis, self.xs_dummy }, s);
            defer _ = mlx.mlx_array_free(z);
            return one(self.finish, c.io.finish, &.{ z, svh }, s);
        }
        // an un-rotated linear's rows past `unrot_min_rows` are one matmul of x against its resident W_std in x's dtype, on
        // the inner product's stream; null keeps the chain below (its W_hat decoded per call)
        if (rows > unrot_min_rows) if (try residentPlain(rs, x, trellis)) |y| return y;
        // prefill: prepared rows (natural order) x the trellis, f32 accumulation, f16 inner = the finish's first rounding
        const c = try self.buildCfg(rows, false);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c.prep);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c.finish);
        if (self.fused_prefill and (rows + self.pf.bm - 1) / self.pf.bm <= pf_tiles_cap) {
            if (self.pf.defer_affine) {
                const cs = try ks.buildConfig(.{ .grid = .{ (self.shape.in / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(rows), @intCast(self.shape.in) }, &[_]c_int{ @intCast(rows), @intCast(self.shape.in / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = &.{ .{ .name = "DIM", .value = .{ .int = @intCast(self.shape.in) } }, .{ .name = "LANE_ORDER", .value = .{ .int = 0 } } } });
                defer _ = mlx.mlx_fast_metal_kernel_config_free(cs);
                var po: [2]mlx.mlx_array = undefined;
                try ks.applyConfig(self.prep_s, cs, &.{ x, suh }, &po, s);
                defer for (po) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                const z = try self.fusedInner(rs, po[0], po[1], trellis, rows);
                defer _ = mlx.mlx_array_free(z);
                return one(self.finish, c.finish, &.{ z, svh }, rs);
            }
            const xp = try one(self.prep, c.prep, &.{ x, suh }, s);
            defer _ = mlx.mlx_array_free(xp);
            const z = try self.fusedInner(rs, xp, null, trellis, rows);
            defer _ = mlx.mlx_array_free(z);
            return one(self.finish, c.finish, &.{ z, svh }, rs);
        }
        const xp = try one(self.prep, c.prep, &.{ x, suh }, s);
        defer _ = mlx.mlx_array_free(xp);
        const W = try self.dequantize(rs, trellis);
        defer _ = mlx.mlx_array_free(W);
        var inner = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inner);
        try mlx.check(mlx.mlx_matmul(&inner, xp, W, rs));
        return one(self.finish, c.finish, &.{ inner, svh }, rs);
    }
};

// ---- The resident trunk (memory for compute; the host's MLX_SERVE_EXL3_TRUNK=resident or bf16): W_hat kept ----
//
// Prefill rows (`pathFor` .matmul: `linearOn`, `innerOf`, attn_io's `prefillInner`) decode the whole trellis to an f16 W_hat
// on every call, so every chunk decodes every tile of every trunk linear again and writes its 2 B per weight. Armed with the
// host's admitted trellises (`residentArm`), `dequantize` decodes an admitted trellis once, by its own launch and config,
// evaluates that W_hat on the calling thread and keeps it; every later call gets a new reference to the kept array. The
// launch writes each weight from its own trellis tile alone (no reduction, no atomics, every element once), the trellis is a
// weight nothing writes, and W_hat is only ever read (the matmul's B operand; the table's own reference keeps MLX from
// donating its buffer), so the kept W_hat is the per-call one byte for byte, and so is every output. Decode rows (the GEMV,
// DEC-ONE) and the register path never call `dequantize`. Evaluated at once rather than kept lazy: a lazy array stays bound to
// the stream it was recorded on (MLX streams are per thread), and a graph that only builds a linear (the MTP history pass's
// dropped outputs) would leave it pending. A W_hat that cannot be kept (its eval fails, its size is not the admitted bytes)
// leaves its trellis on the per-call path and its bytes leave the bill. The host bills what is admitted but not kept yet
// (`residentPending`) in its GPU ceiling; `residentRelease` (its unload) drops every kept array.

/// Test seam (the resident-trunk tests' reference arm): the per-call path while the table stays armed.
pub var resident_bypass: bool = false;

/// One trunk linear's f16 W_hat bytes from its trellis shape [in/16, out/16, n]; 0 for any other shape.
pub fn residentBytesOf(shape: []const c_int) u64 {
    if (shape.len != 3 or shape[0] <= 0 or shape[1] <= 0) return 0;
    return @as(u64, @intCast(shape[0])) * 16 * @as(u64, @intCast(shape[1])) * 16 * 2;
}

const Resident = struct {
    /// twice the host registry's 2048 trellises (`Exl3TrunkRegistry.CAP`): every one admissible at half load
    pub const cap: usize = 4096;
    const cap_bits: u6 = 12;
    /// `unrot`: the un-rotated trunk's W_std kept in the W_hat's place (`residentInstall`)
    const State = enum(u8) { empty, admitted, busy, kept, failed, unrot };
    /// `unrot`: `w` is the linear's W_std (a column view of its group's joined W_std for a member of a join),
    /// `jw` that joined W_std (null ctx for a group of one), `col` the member's first column in it, `group` the group's id.
    const Slot = struct { key: usize = 0, bytes: u64 = 0, state: State = .empty, w: mlx.mlx_array = .{ .ctx = null }, jw: mlx.mlx_array = .{ .ctx = null }, col: u32 = 0, group: u32 = 0 };

    /// The open-addressed slots (linear probing from the key's golden-ratio hash, at most half full) and their bytes.
    const Table = struct {
        slots: [cap]Slot = @splat(.{}),
        modules: u32 = 0,
        kept: u32 = 0,
        /// the bytes of every admitted W_hat that has not failed, and of those kept
        planned: u64 = 0,
        held: u64 = 0,
        reuses: u64 = 0,
        /// the un-rotated trunk: slots holding W_std (counted in `kept`, their bytes in `held`), the groups installed and
        /// those of more than one member, the plain matmuls served and the rotated-basis calls that met an un-rotated slot
        unrot: u32 = 0,
        groups: u32 = 0,
        joined: u32 = 0,
        plain: u64 = 0,
        misses: u64 = 0,

        fn home(key: usize) usize {
            return (key *% 0x9E3779B97F4A7C15) >> @as(u6, @intCast(@bitSizeOf(usize) - @as(usize, cap_bits)));
        }

        fn find(t: *Table, key: usize) ?*Slot {
            var i = home(key);
            for (0..cap) |_| {
                const sl = &t.slots[i];
                if (sl.state == .empty) return null;
                if (sl.key == key) return sl;
                i = (i + 1) % cap;
            }
            return null;
        }

        /// `key` admitted with its W_hat's bytes (a repeat is ignored); error.ResidentTableFull past half the slots.
        fn admit(t: *Table, key: usize, bytes: u64) !void {
            if (key == 0 or bytes == 0) return error.BadExl3Shape;
            if (t.find(key) != null) return;
            if (t.modules >= cap / 2) return error.ResidentTableFull;
            var i = home(key);
            while (t.slots[i].state != .empty) i = (i + 1) % cap;
            t.slots[i] = .{ .key = key, .bytes = bytes, .state = .admitted };
            t.modules += 1;
            t.planned += bytes;
        }

        /// A busy slot's W_hat kept: its bytes move from pending to held.
        fn keep(t: *Table, sl: *Slot, w: mlx.mlx_array) void {
            sl.w = w;
            sl.state = .kept;
            t.kept += 1;
            t.held += sl.bytes;
        }

        /// A busy slot's W_hat not kept: the per-call path from now on, its bytes off the bill.
        fn fail(t: *Table, sl: *Slot) void {
            sl.state = .failed;
            t.planned -|= sl.bytes;
        }

        /// An admitted or kept slot's W_std installed (the table owns `w` and `jw` from here): a kept W_hat is
        /// dropped, its bytes staying held; an admitted slot's bytes move from pending to held, as `keep` moves them.
        fn unrotate(t: *Table, sl: *Slot, w: mlx.mlx_array, jw: mlx.mlx_array, col: u32, group: u32) void {
            if (sl.state == .kept) {
                if (sl.w.ctx != null) _ = mlx.mlx_array_free(sl.w);
            } else {
                t.kept += 1;
                t.held += sl.bytes;
            }
            sl.* = .{ .key = sl.key, .bytes = sl.bytes, .state = .unrot, .w = w, .jw = jw, .col = col, .group = group };
            t.unrot += 1;
        }

        fn pending(t: *const Table) u64 {
            return t.planned -| t.held;
        }
    };

    var lock: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
    var armed: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);
    /// some slot holds a W_std (`residentPlain` asks the table only then)
    var unrot_any: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);
    var table: Table = .{};
    var failed_said = false;
    var miss_said = false;
    var dtype_said = false;

    fn acquire() void {
        while (lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn unlock() void {
        lock.store(0, .release);
    }
};

/// Arms the resident table: admit `trellises` (the host's handles, borrowed: the model holds them) with each one's W_hat
/// bytes (in x out x 2, checked against its shape), replacing any earlier set (its kept arrays dropped). Nothing is decoded
/// here.
/// error.ResidentTableFull past half the table's slots, error.BadExl3Shape for a handle that is not a trellis of its bytes,
/// error.Exl3EngineNotImplemented under the fused dense GEMM (MLX_SERVE_TURBOEXL3_DENSE_PREFILL=fused: its prefill rows never
/// dequantize, so nothing would be kept and the bill would stay pending); on an error nothing stays admitted.
pub fn residentArm(trellises: []const mlx.mlx_array, bytes: []const u64) !void {
    if (trellises.len != bytes.len) return error.BadExl3Shape;
    residentRelease();
    if (std.c.getenv("MLX_SERVE_TURBOEXL3_DENSE_PREFILL")) |v| if (std.mem.eql(u8, std.mem.span(v), "fused")) return error.Exl3EngineNotImplemented;
    Resident.acquire();
    defer Resident.unlock();
    errdefer Resident.table = .{};
    for (trellises, bytes) |t, b| {
        const p = t.ctx orelse return error.BadExl3Shape;
        if (residentBytesOf(mlx.getShape(t)) != b) return error.BadExl3Shape;
        try Resident.table.admit(@intFromPtr(p), b);
    }
    Resident.armed.store(Resident.table.modules > 0, .release);
}

/// Every kept W_hat (and W_std) dropped and the table disarmed (the host's unload; idempotent). A graph that already holds
/// a kept array keeps it alive until it is done with it.
pub fn residentRelease() void {
    Resident.armed.store(false, .release);
    Resident.unrot_any.store(false, .release);
    Resident.acquire();
    defer Resident.unlock();
    for (&Resident.table.slots) |*sl| switch (sl.state) {
        .kept => _ = mlx.mlx_array_free(sl.w),
        // an un-rotated member's W_std and its reference to the group's joined W_std
        .unrot => {
            _ = mlx.mlx_array_free(sl.w);
            if (sl.jw.ctx != null) _ = mlx.mlx_array_free(sl.jw);
        },
        else => {},
    };
    Resident.table = .{};
}

/// The W_hat bytes admitted but not kept yet, which the host bills in its GPU ceiling ahead of the allocation (0 when
/// disarmed).
pub fn residentPending() u64 {
    if (!Resident.armed.load(.acquire)) return 0;
    Resident.acquire();
    defer Resident.unlock();
    return Resident.table.pending();
}

/// The resident table's counts (its tests and logs): trellises admitted and kept, their W_hat bytes, calls a kept W_hat
/// served; of the un-rotated trunk: slots holding W_std (in `kept`), groups installed and joined, plain matmuls served,
/// rotated-basis calls that met an un-rotated slot (each decoded its W_hat per call).
pub const ResidentStats = struct { modules: u32, kept: u32, planned: u64, held: u64, reuses: u64, unrot: u32 = 0, groups: u32 = 0, joined: u32 = 0, plain: u64 = 0, misses: u64 = 0 };

pub fn residentStats() ResidentStats {
    Resident.acquire();
    defer Resident.unlock();
    const t = &Resident.table;
    return .{ .modules = t.modules, .kept = t.kept, .planned = t.planned, .held = t.held, .reuses = t.reuses, .unrot = t.unrot, .groups = t.groups, .joined = t.joined, .plain = t.plain, .misses = t.misses };
}

/// Whether the table keeps `trellis`'s W_hat.
pub fn residentKeeps(trellis: mlx.mlx_array) bool {
    const p = trellis.ctx orelse return false;
    Resident.acquire();
    defer Resident.unlock();
    const sl = Resident.table.find(@intFromPtr(p)) orelse return false;
    return sl.state == .kept;
}

/// `dequantize` on an armed table: a new reference to an admitted trellis's kept W_hat, decoded by `decodeTrellis` and
/// evaluated at its first call; null where the table keeps none (not admitted, failed, or being kept by another thread).
/// An un-rotated slot holds no W_hat, so a rotated-basis call on it (a call site on the rotated chain) decodes per call:
/// counted and said once, the bytes right, the time the per-call path's.
fn residentTake(d: *Dense, s: mlx.mlx_stream, trellis: mlx.mlx_array) !?mlx.mlx_array {
    const key = @intFromPtr(trellis.ctx orelse return null);
    const bytes = claim: {
        Resident.acquire();
        defer Resident.unlock();
        const sl = Resident.table.find(key) orelse return null;
        switch (sl.state) {
            .kept => {
                var w = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(w);
                try mlx.check(mlx.mlx_array_set(&w, sl.w));
                Resident.table.reuses += 1;
                return w;
            },
            .admitted => {
                sl.state = .busy;
                break :claim sl.bytes;
            },
            .unrot => {
                Resident.table.misses += 1;
                if (!Resident.miss_said) {
                    Resident.miss_said = true;
                    std.debug.print("[turboexl3] un-rotated trunk: a rotated-basis call on a {d} x {d} linear decodes its W_hat per call\n", .{ d.shape.in, d.shape.out });
                }
                return null;
            },
            else => return null,
        }
    };
    // outside the lock: the per-call launch, evaluated on this thread (its graph is the trellis alone)
    const w = d.decodeTrellis(s, trellis) catch |e| {
        residentFail(key, e);
        return null;
    };
    if (@as(u64, mlx.mlx_array_size(w)) * 2 != bytes or mlx.mlx_array_eval(w) != 0) {
        _ = mlx.mlx_array_free(w);
        residentFail(key, error.Exl3ResidentNotKept);
        return null;
    }
    Resident.acquire();
    defer Resident.unlock();
    // released (or re-armed) meanwhile: this call's own W_hat, as the per-call path hands it out
    const sl = Resident.table.find(key) orelse return w;
    if (sl.state != .busy) return w;
    var out = mlx.mlx_array_new();
    if (mlx.mlx_array_set(&out, w) != 0) {
        _ = mlx.mlx_array_free(out);
        Resident.table.fail(sl);
        return w;
    }
    Resident.table.keep(sl, w);
    return out;
}

/// A W_hat that could not be kept: its trellis stays on the per-call path and its bytes leave the host's bill; said once.
fn residentFail(key: usize, err: anyerror) void {
    Resident.acquire();
    defer Resident.unlock();
    if (Resident.table.find(key)) |sl| if (sl.state == .busy) Resident.table.fail(sl);
    if (Resident.failed_said) return;
    Resident.failed_said = true;
    std.debug.print("[turboexl3] resident trunk: a W_hat could not be kept ({s}); its linear dequantizes per call\n", .{@errorName(err)});
}

// ---- The un-rotated trunk (the host's MLX_SERVE_EXL3_TRUNK=bf16, opt-in; a reordering): out of the Hadamard basis ----
//
// `linear`'s prefill rows compute y = finish(f16(prep(x)) W_hat). prep is per 128-block of x: f16(H128(f16(x) * suh)), where
// `text_prep`'s lane L holds block element L + 32 j, the five xor-shuffle stages run over the lane bits and the four-point
// stage over j, so H128[i][k] = (-1)^popcount(i & k) / sqrt(128) in the block's natural index order (Sylvester order:
// symmetric, its own inverse); finish is per 128-block of the output: f16(H128(f16(z)) * svh) (`text_finish`, the same
// butterfly). With H = blockdiag(H128): y = x diag(suh) H W_hat H diag(svh), so the linear in the standard basis is
//   W_std = diag(suh) H W_hat H diag(svh):   W_std[i, o] = suh[i] * svh[o] * (H' W_hat H')[i, o] / 128
// with H' the unscaled +-1 transform, each 128 x 128 tile of W_hat mapping to the same tile of W_std. `unrotCols` runs the +-1
// butterfly along each row's 128-column blocks of the f16 W_hat into an f32 copy, `unrotRows` along each column's 128-row
// blocks (through threadgroup memory: coalesced loads and stores), times 1/128 (exact), suh and svh, in f32, and rounds once
// to the activation dtype (bf16 on the Qwen3.8 pack), the dtype every trunk linear's x already has. A prefill call is then
// one MLX matmul y = x W_std in that dtype (`residentPlain`): no prepare, no finish, no cast; one rounding of each weight
// and of each output where the chain rounds x's rotated copy, the inner product and the finish to f16 (a reordering: the
// same linear map, other roundings).
// `residentInstall` puts a group's W_std in its trellises' slots, replacing the W_hat (the same bytes, in x out x 2): a group
// of one is its member's W_std, a group of up to `max_join` members reading one x (the host's: GDN in_proj_qkv | in_proj_z,
// QSA q | k | v | indexer, the shared expert's gate | up) one joined W_std [in, sum of outs] whose members are column views of
// it (evaluated: zero-copy views on its buffer with its row stride, which MLX's matmul takes as B without a copy), so the
// host multiplies a join once and any member alone, and nothing is resident twice. Calls of up to `unrot_min_rows` rows
// (decode, verify, the register path's snapshot tail) never read the table and keep the trellis paths byte for byte; a
// rotated-basis call on an un-rotated slot (`dequantize`: a call site on the rotated chain) decodes its W_hat per call.

/// The widest call that keeps the trellis paths on an un-rotated linear: decode and verify rows (the GEMV, DEC-ONE) and the
/// register path's rows (`small_rows`' default) never meet the un-rotated table.
pub const unrot_min_rows: usize = 32;
/// The most members of one un-rotated join (QSA's q | k | v | indexer).
pub const max_join: usize = 4;

/// Un-rotation pass 1: r f32 [rows, DIM] = the unscaled +-1 H128 of each 128-column block of each row of w (f16 W_hat
/// [rows, DIM]): `text_finish`'s layout and butterfly, one simdgroup per (block, row)
pub const text_unrot_cols =
    \\    constexpr uint DIMC = uint(DIM);
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) v[j] = float(w[ulong(row) * DIMC + b * 128u + L + 32u * j]);
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    for (uint j = 0; j < 4u; ++j) r[ulong(row) * DIMC + b * 128u + L + 32u * j] = v[j];
    \\
;

/// Un-rotation pass 2: W OUT_T [rows, DIM] = OUT_T(((H' r)[i, o] / 128) * suh[i] * svh[o]) for pass 1's r, one rounding.
/// One 256-thread threadgroup per (128-row block, 32-column strip): the strip read into threadgroup memory row by row
/// (coalesced; 33-float rows keep a column's reads on distinct banks), simdgroup sg transforming the strip's columns
/// sg + 8 q with `text_prep`'s layout and butterfly (lane L holding rows L + 32 j), the strip written back row by row
pub const text_unrot_rows =
    \\    constexpr uint DIMC = uint(DIM);
    \\    threadgroup float t[128u * 33u];
    \\    const uint cs = threadgroup_position_in_grid.x;
    \\    const uint rb = threadgroup_position_in_grid.y;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    for (uint e = tid; e < 4096u; e += 256u) t[(e >> 5u) * 33u + (e & 31u)] = r[ulong(rb * 128u + (e >> 5u)) * DIMC + cs * 32u + (e & 31u)];
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint q = 0; q < 4u; ++q) {
    \\        const uint c = sg + 8u * q;
    \\        float v[4];
    \\        for (uint j = 0; j < 4u; ++j) v[j] = t[(L + 32u * j) * 33u + c];
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\        for (uint j = 0; j < 4u; ++j) t[(L + 32u * j) * 33u + c] = v[j];
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint e = tid; e < 4096u; e += 256u) {
    \\        const uint i = rb * 128u + (e >> 5u);
    \\        const uint o = cs * 32u + (e & 31u);
    \\        W[ulong(i) * DIMC + o] = OUT_T(t[(e >> 5u) * 33u + (e & 31u)] * 0.0078125f * float(suh[i]) * float(svh[o]));
    \\    }
    \\
;

const unrot_cols_in = [_][*:0]const u8{"w"};
const unrot_cols_out = [_][*:0]const u8{"r"};
const unrot_rows_in = [_][*:0]const u8{ "r", "suh", "svh" };
const unrot_rows_out = [_][*:0]const u8{"W"};

/// The two un-rotation kernels, made at their first use (a process with the switch off never builds them).
var unrot_kernels_lock: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
var unrot_kernels: ?[2]mlx.mlx_fast_metal_kernel = null;

fn unrotKernels() ![2]mlx.mlx_fast_metal_kernel {
    while (unrot_kernels_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer unrot_kernels_lock.store(0, .release);
    if (unrot_kernels) |k| return k;
    const cols = ks.newKernel("turboexl3_unrot_cols", &unrot_cols_in, &unrot_cols_out, text_unrot_cols, decode.header, true) orelse return error.KernelCreateFailed;
    errdefer _ = mlx.mlx_fast_metal_kernel_free(cols);
    const rows = ks.newKernel("turboexl3_unrot_rows", &unrot_rows_in, &unrot_rows_out, text_unrot_rows, decode.header, true) orelse return error.KernelCreateFailed;
    unrot_kernels = .{ cols, rows };
    return unrot_kernels.?;
}

/// Un-rotation pass 1 (lazy): r f32 [in, out] of an f16 W_hat [in, out] (`Dense.decodeTrellis`'s), in and out multiples
/// of 128; the caller frees r.
pub fn unrotCols(s: mlx.mlx_stream, w_hat: mlx.mlx_array) !mlx.mlx_array {
    const sh = mlx.getShape(w_hat);
    if (sh.len != 2 or sh[0] <= 0 or sh[1] <= 0 or @mod(sh[1], 128) != 0 or mlx.mlx_array_dtype(w_hat) != .float16) return error.BadExl3Shape;
    const k = try unrotKernels();
    const in: u32 = @intCast(sh[0]);
    const out: u32 = @intCast(sh[1]);
    const c = try ks.buildConfig(.{ .grid = .{ (out / 128) * 32, in, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ sh[0], sh[1] }}, .out_dtypes = &.{.float32}, .template = &.{.{ .name = "DIM", .value = .{ .int = sh[1] } }} });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    var r: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(k[0], c, &.{w_hat}, &r, s);
    return r[0];
}

/// Un-rotation pass 2 (lazy): W_std [in, out] in `act` (bf16 or f16: two bytes a weight, the W_hat's) from pass 1's r f32
/// [in, out], suh f16 [in] and svh f16 [out], in a multiple of 128 and out of 32; the caller frees it.
pub fn unrotRows(s: mlx.mlx_stream, r: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, act: mlx.mlx_dtype) !mlx.mlx_array {
    const sh = mlx.getShape(r);
    if (sh.len != 2 or sh[0] <= 0 or sh[1] <= 0 or @mod(sh[0], 128) != 0 or @mod(sh[1], 32) != 0 or mlx.mlx_array_dtype(r) != .float32) return error.BadExl3Shape;
    if (mlx.mlx_array_size(suh) != @as(usize, @intCast(sh[0])) or mlx.mlx_array_size(svh) != @as(usize, @intCast(sh[1]))) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(suh) != .float16 or mlx.mlx_array_dtype(svh) != .float16) return error.BadExl3Dtype;
    if (act != .bfloat16 and act != .float16) return error.BadExl3Dtype;
    const k = try unrotKernels();
    const in: u32 = @intCast(sh[0]);
    const out: u32 = @intCast(sh[1]);
    const c = try ks.buildConfig(.{ .grid = .{ (out / 32) * 256, in / 128, 1 }, .threadgroup = .{ 256, 1, 1 }, .out_shapes = &.{&[_]c_int{ sh[0], sh[1] }}, .out_dtypes = &.{act}, .template = &.{
        .{ .name = "DIM", .value = .{ .int = sh[1] } },
        .{ .name = "OUT_T", .value = .{ .dtype = act } },
    } });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    var w: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(k[1], c, &.{ r, suh, svh }, &w, s);
    return w[0];
}

/// Both un-rotation passes over one W_hat, lazily (tests).
pub fn unrotWeight(s: mlx.mlx_stream, w_hat: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, act: mlx.mlx_dtype) !mlx.mlx_array {
    const r = try unrotCols(s, w_hat);
    defer _ = mlx.mlx_array_free(r);
    return unrotRows(s, r, suh, svh, act);
}

/// Whether `trellis` is admitted with a W_hat slot the un-rotation may take (admitted or kept, not being kept by another
/// thread, failed or un-rotated already).
pub fn residentUnrotAdmits(trellis: mlx.mlx_array) bool {
    const p = trellis.ctx orelse return false;
    Resident.acquire();
    defer Resident.unlock();
    const sl = Resident.table.find(@intFromPtr(p)) orelse return false;
    return sl.state == .admitted or sl.state == .kept;
}

/// One un-rotated group's W_std into its members' slots (`trellises`, in column order): `ws[i]` member i's W_std
/// (evaluated; a column view of `jw` for a join), `jw` the joined W_std (null for a group of one), `cols[i]` member i's
/// first column in it. The table owns `ws` and `jw` on success (each member's slot a reference to `jw`); on an error
/// (a member not admitted, a W_std that is not the member's bytes) nothing is installed and the caller keeps them.
pub fn residentInstall(trellises: []const mlx.mlx_array, ws: []const mlx.mlx_array, jw: ?mlx.mlx_array, cols: []const u32) !void {
    const n = trellises.len;
    if (n == 0 or n > max_join or ws.len != n or cols.len != n or (n > 1) != (jw != null)) return error.BadExl3Shape;
    Resident.acquire();
    defer Resident.unlock();
    var slots: [max_join]*Resident.Slot = undefined;
    for (trellises, ws, 0..) |t, w, i| {
        const sl = Resident.table.find(@intFromPtr(t.ctx orelse return error.BadExl3Shape)) orelse return error.Exl3ResidentNotAdmitted;
        if (sl.state != .admitted and sl.state != .kept) return error.Exl3ResidentNotAdmitted;
        // two bytes a weight, the W_hat's: the bill and the memory stay the table's
        if (mlx.mlx_array_itemsize(w) != 2 or @as(u64, mlx.mlx_array_size(w)) * 2 != sl.bytes) return error.BadExl3Shape;
        slots[i] = sl;
    }
    // every member's reference to the joined W_std, made before any slot changes
    var jrefs: [max_join]mlx.mlx_array = @splat(.{ .ctx = null });
    if (jw) |j| for (jrefs[0..n]) |*r| {
        r.* = mlx.mlx_array_new();
        if (mlx.mlx_array_set(r, j) != 0) {
            for (jrefs[0..n]) |a| if (a.ctx != null) {
                _ = mlx.mlx_array_free(a);
            };
            return error.MlxError;
        }
    };
    Resident.table.groups += 1;
    if (n > 1) Resident.table.joined += 1;
    for (slots[0..n], ws, cols, jrefs[0..n]) |sl, w, c, r| Resident.table.unrotate(sl, w, r, c, Resident.table.groups);
    // the caller's handle on the joined W_std is the table's now too: drop it, the members' references keep the buffer
    if (jw) |j| _ = mlx.mlx_array_free(j);
    Resident.unrot_any.store(true, .release);
}

/// Whether the table holds `trellis`'s W_std.
pub fn residentUnrot(trellis: mlx.mlx_array) bool {
    if (!Resident.unrot_any.load(.acquire)) return false;
    const p = trellis.ctx orelse return false;
    Resident.acquire();
    defer Resident.unlock();
    const sl = Resident.table.find(@intFromPtr(p)) orelse return false;
    return sl.state == .unrot;
}

/// A new reference to `trellis`'s W_std [in, out] (the caller frees it), null where the table holds none.
pub fn residentStd(trellis: mlx.mlx_array) ?mlx.mlx_array {
    return stdTake(trellis, false);
}

fn stdTake(trellis: mlx.mlx_array, plain: bool) ?mlx.mlx_array {
    if (!Resident.unrot_any.load(.acquire)) return null;
    const p = trellis.ctx orelse return null;
    Resident.acquire();
    defer Resident.unlock();
    const sl = Resident.table.find(@intFromPtr(p)) orelse return null;
    if (sl.state != .unrot) return null;
    var w = mlx.mlx_array_new();
    if (mlx.mlx_array_set(&w, sl.w) != 0) {
        _ = mlx.mlx_array_free(w);
        return null;
    }
    if (plain) Resident.table.plain += 1;
    return w;
}

/// An un-rotated member's group: `w` a new reference to the group's joined W_std (the member's own for a group
/// of one; the caller frees it), the member's columns [col, col + width) there, the group's id.
pub const StdJoined = struct { w: mlx.mlx_array, col: u32, width: u32, group: u32 };

pub fn residentJoined(trellis: mlx.mlx_array) ?StdJoined {
    if (!Resident.unrot_any.load(.acquire)) return null;
    const p = trellis.ctx orelse return null;
    Resident.acquire();
    defer Resident.unlock();
    const sl = Resident.table.find(@intFromPtr(p)) orelse return null;
    if (sl.state != .unrot) return null;
    const wsh = mlx.getShape(sl.w);
    if (wsh.len != 2) return null;
    var w = mlx.mlx_array_new();
    if (mlx.mlx_array_set(&w, if (sl.jw.ctx != null) sl.jw else sl.w) != 0) {
        _ = mlx.mlx_array_free(w);
        return null;
    }
    return .{ .w = w, .col = sl.col, .width = @intCast(wsh[1]), .group = sl.group };
}

/// `linearOn`'s un-rotated arm: y [rows, out] = x W_std, one MLX matmul on `s` in x's dtype, for an un-rotated
/// `trellis`; null where the table holds no W_std for it, or x's dtype is not the W_std's (said once: no cast is added, the
/// chain serves the call with its W_hat decoded per call).
fn residentPlain(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array) !?mlx.mlx_array {
    const w = stdTake(trellis, true) orelse return null;
    defer _ = mlx.mlx_array_free(w);
    if (mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(w)) {
        if (!Resident.dtype_said) {
            Resident.dtype_said = true;
            std.debug.print("[turboexl3] un-rotated trunk: x is {s}, its W_std {s}; the rotated chain serves the call\n", .{ @tagName(mlx.mlx_array_dtype(x)), @tagName(mlx.mlx_array_dtype(w)) });
        }
        return null;
    }
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_matmul(&y, x, w, s));
    return y;
}

// ---- The un-rotation's CPU checks: the kernels' arithmetic ported statement for statement against an f64 reference ----

/// f32 to f16 and back (round to nearest even), as Metal's half() rounds.
fn cpuF16(x: f32) f32 {
    return @floatCast(@as(f16, @floatCast(x)));
}

/// f32 to bf16 and back (round to nearest even on the 16 dropped bits; finite values), as Metal's and MLX's bfloat round.
fn cpuBf16(x: f32) f32 {
    const b: u32 = @bitCast(x);
    return @bitCast((b +% (0x7FFF + ((b >> 16) & 1))) & 0xFFFF_0000);
}

/// The 128-point butterfly of `text_prep` / `text_finish` / `text_unrot_cols` / `text_unrot_rows` on the CPU, statement for
/// statement: v[L][j] holds block element L + 32 j; each xor-shuffle stage reads every partner before it writes (as
/// `simd_shuffle_xor` does), then the four-point stage over j; f32.
fn cpuButterfly(v: *[32][4]f32) void {
    for (0..5) |st| {
        const bit: usize = @as(usize, 1) << @intCast(st);
        var nv: [32][4]f32 = undefined;
        for (0..32) |l| for (0..4) |j| {
            const p = v[l ^ bit][j];
            nv[l][j] = if (l & bit != 0) p - v[l][j] else v[l][j] + p;
        };
        v.* = nv;
    }
    for (v) |*lv| {
        const t0 = lv[0] + lv[1];
        const t1 = lv[0] - lv[1];
        const t2 = lv[2] + lv[3];
        const t3 = lv[2] - lv[3];
        lv[0] = t0 + t2;
        lv[1] = t1 + t3;
        lv[2] = t0 - t2;
        lv[3] = t1 - t3;
    }
}

/// Sylvester's H128 sign of (i, k) in a block's natural index order: (-1)^popcount(i & k).
fn h128Sign(i: usize, k: usize) f64 {
    return if (@popCount(i & k) & 1 == 0) 1.0 else -1.0;
}

const cpu_had_scale: f32 = 0.08838834764831845;

/// `text_prep` (LANE_ORDER 0) on one row x [in] (bf16-exact values), suh [in]: xp [in] (f16 values).
fn cpuPrep(x: []const f32, suh: []const f32, xp: []f32) void {
    for (0..x.len / 128) |b| {
        var v: [32][4]f32 = undefined;
        for (0..32) |l| for (0..4) |j| {
            const i = b * 128 + l + 32 * j;
            v[l][j] = cpuF16(x[i]) * suh[i];
        };
        cpuButterfly(&v);
        for (0..32) |l| for (0..4) |j| {
            xp[b * 128 + l + 32 * j] = cpuF16(v[l][j] * cpu_had_scale);
        };
    }
}

/// `text_finish` on one row's f16 inner product z [out] and svh [out], OUT_T = bf16: y [out].
fn cpuFinish(z: []const f32, svh: []const f32, y: []f32) void {
    for (0..z.len / 128) |b| {
        var v: [32][4]f32 = undefined;
        for (0..32) |l| for (0..4) |j| {
            v[l][j] = cpuF16(z[b * 128 + l + 32 * j]);
        };
        cpuButterfly(&v);
        for (0..32) |l| for (0..4) |j| {
            const o = b * 128 + l + 32 * j;
            y[o] = cpuBf16(cpuF16(v[l][j] * cpu_had_scale * svh[o]));
        };
    }
}

/// `text_unrot_cols` then `text_unrot_rows` (OUT_T bf16) on a W_hat [in, out] row-major (f16 values): W_std [in, out].
fn cpuUnrot(a: std.mem.Allocator, w_hat: []const f32, suh: []const f32, svh: []const f32, in: usize, out: usize, w_std: []f32) !void {
    const r = try a.alloc(f32, in * out);
    defer a.free(r);
    for (0..in) |row| for (0..out / 128) |b| {
        var v: [32][4]f32 = undefined;
        for (0..32) |l| for (0..4) |j| {
            v[l][j] = w_hat[row * out + b * 128 + l + 32 * j];
        };
        cpuButterfly(&v);
        for (0..32) |l| for (0..4) |j| {
            r[row * out + b * 128 + l + 32 * j] = v[l][j];
        };
    };
    for (0..in / 128) |rb| for (0..out) |o| {
        var v: [32][4]f32 = undefined;
        for (0..32) |l| for (0..4) |j| {
            v[l][j] = r[(rb * 128 + l + 32 * j) * out + o];
        };
        cpuButterfly(&v);
        for (0..32) |l| for (0..4) |j| {
            const i = rb * 128 + l + 32 * j;
            w_std[i * out + o] = cpuBf16(v[l][j] * 0.0078125 * suh[i] * svh[o]);
        };
    };
}

/// The f64 reference of one 128 x 128 tile (row block rb, column block cb) of W_std = diag(suh) H W_hat H diag(svh) for a
/// row-major W_hat [_, out] (f16 values as f32), H the scaled blockwise H128 built from its signs (the GPU tests' sampled tiles).
pub fn refUnrotTile(w_hat: []const f32, suh: []const f32, svh: []const f32, out: usize, rb: usize, cb: usize, tile: *[128 * 128]f64) void {
    var t: [128 * 128]f64 = undefined;
    for (0..128) |a| for (0..128) |o| {
        var acc: f64 = 0;
        for (0..128) |k| acc += @as(f64, w_hat[(rb * 128 + a) * out + cb * 128 + k]) * h128Sign(k, o);
        t[a * 128 + o] = acc;
    };
    for (0..128) |i| for (0..128) |o| {
        var acc: f64 = 0;
        for (0..128) |k| acc += h128Sign(i, k) * t[k * 128 + o];
        tile[i * 128 + o] = acc / 128.0 * @as(f64, suh[rb * 128 + i]) * @as(f64, svh[cb * 128 + o]);
    };
}

/// The f64 reference W_ref [in, out] = diag(suh) H W_hat H diag(svh), H the scaled blockwise H128 built from its signs.
fn refUnrot(a: std.mem.Allocator, w_hat: []const f32, suh: []const f32, svh: []const f32, in: usize, out: usize, w_ref: []f64) !void {
    // T = W_hat H (per 128-column block), then H T (per 128-row block)
    const t = try a.alloc(f64, in * out);
    defer a.free(t);
    for (0..in) |i| for (0..out) |o| {
        const b = o / 128 * 128;
        var acc: f64 = 0;
        for (0..128) |k| acc += @as(f64, w_hat[i * out + b + k]) * h128Sign(k, o - b);
        t[i * out + o] = acc;
    };
    for (0..in) |i| for (0..out) |o| {
        const b = i / 128 * 128;
        var acc: f64 = 0;
        for (0..128) |k| acc += h128Sign(i - b, k) * t[(b + k) * out + o];
        w_ref[i * out + o] = acc / 128.0 * @as(f64, suh[i]) * @as(f64, svh[o]);
    };
}

/// One bf16 step at |v|'s binade (the spacing of bf16 values around v: 2^(e - 7) for 2^e <= |v| < 2^(e + 1); 2^-133 under
/// the normal range).
pub fn bf16Step(v: f64) f64 {
    const m: f32 = @floatCast(@abs(v));
    const e: i32 = @intCast((@as(u32, @bitCast(m)) >> 23) & 0xFF);
    if (e == 0) return 0x1p-133;
    return @bitCast(@as(u64, @intCast(e - 127 - 7 + 1023)) << 52);
}

/// The un-rotation's element bar (tests): the f32 un-rotation's error bound in f32 epsilons (2^-23) of an element's terms
/// (`unrotElemOk`): two 7-stage butterflies and the two scale products round at most 16 u = 8 eps of the terms' absolute
/// sum to first order; the bar is twice that.
pub const unrot_terms_eps: f64 = 16.0;

/// The un-rotation's element bar (tests) for W_std's element `got` against the f64 reference `ref`: within one bf16 step of
/// `ref`, or within half a bf16 step of `got` (its own rounding) plus `unrot_terms_eps` f32 epsilons of `terms`, the
/// absolute sum of the element's terms (|suh[i] svh[o]| / 128 times its 128 x 128 tile's sum of |W_hat|,
/// `unrotTileAbsSum`). The second arm is the element whose terms cancel to near zero: the f32 butterflies round relative
/// to the partial sums, not to that small result, so a bf16 step of the result is no bound there.
pub fn unrotElemOk(got: f64, ref: f64, terms: f64) bool {
    const d = @abs(got - ref);
    return d <= bf16Step(ref) or d <= 0.5 * bf16Step(got) + unrot_terms_eps * 0x1p-23 * terms;
}

/// The sum of |W_hat| over the 128 x 128 tile (row block rb, column block cb) of a row-major W_hat [_, out] (tests).
pub fn unrotTileAbsSum(w_hat: []const f32, out: usize, rb: usize, cb: usize) f64 {
    var s: f64 = 0;
    for (0..128) |i| for (0..128) |o| {
        s += @abs(@as(f64, w_hat[(rb * 128 + i) * out + cb * 128 + o]));
    };
    return s;
}

test "turboexl3 dense un-rotated trunk: the dense butterfly is Sylvester's H128 in the block's natural index (lane L, register j = element L + 32 j)" {
    var prng = std.Random.DefaultPrng.init(0x7_2B07);
    const rnd = prng.random();
    for (0..4) |_| {
        var x: [128]f32 = undefined;
        for (&x) |*e| e.* = rnd.floatNorm(f32);
        var v: [32][4]f32 = undefined;
        for (0..32) |l| for (0..4) |j| {
            v[l][j] = x[l + 32 * j];
        };
        cpuButterfly(&v);
        for (0..128) |i| {
            var want: f64 = 0;
            for (0..128) |k| want += h128Sign(i, k) * @as(f64, x[k]);
            const got: f64 = v[i % 32][i / 32];
            try std.testing.expect(@abs(got - want) <= 1e-4 * (1.0 + @abs(want)));
        }
    }
    // H128 / sqrt(128) is its own inverse: the butterfly twice is 128 x the identity
    var x: [128]f32 = undefined;
    for (&x, 0..) |*e, i| e.* = @floatFromInt(@as(i32, @intCast(i % 7)) - 3);
    var v: [32][4]f32 = undefined;
    for (0..32) |l| for (0..4) |j| {
        v[l][j] = x[l + 32 * j];
    };
    cpuButterfly(&v);
    cpuButterfly(&v);
    for (0..128) |i| try std.testing.expectEqual(128.0 * x[i], v[i % 32][i / 32]);
}

test "turboexl3 dense un-rotated trunk: the un-rotation equals diag(suh) H W_hat H diag(svh) in f64 within one bf16 rounding, and x W_std is the chain's linear (prep, f16 inner product, finish) within the stated bars" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x7_2B08);
    const rnd = prng.random();
    // two 128-blocks each way, a prefill row count past the register path
    const in: usize = 256;
    const out: usize = 384;
    const rows: usize = 33;
    const w_hat = try a.alloc(f32, in * out);
    defer a.free(w_hat);
    // f16 W_hat values in the trellis codebook's range; suh / svh f16 (random signs times scales, as the packs hold them)
    for (w_hat) |*e| e.* = cpuF16(rnd.floatNorm(f32) * 1.5);
    const suh = try a.alloc(f32, in);
    defer a.free(suh);
    for (suh) |*e| e.* = cpuF16((if (rnd.boolean()) @as(f32, 1.0) else -1.0) * (0.5 + rnd.float(f32)));
    const svh = try a.alloc(f32, out);
    defer a.free(svh);
    for (svh) |*e| e.* = cpuF16((if (rnd.boolean()) @as(f32, 1.0) else -1.0) * (0.5 + rnd.float(f32)) / 16.0);
    const w_std = try a.alloc(f32, in * out);
    defer a.free(w_std);
    try cpuUnrot(a, w_hat, suh, svh, in, out, w_std);
    const w_ref = try a.alloc(f64, in * out);
    defer a.free(w_ref);
    try refUnrot(a, w_hat, suh, svh, in, out, w_ref);
    // the GPU tests' sampled-tile reference is this one
    const tile = try a.create([128 * 128]f64);
    defer a.destroy(tile);
    refUnrotTile(w_hat, suh, svh, out, 1, 2, tile);
    for (0..128) |i| for (0..128) |o| try std.testing.expectApproxEqAbs(w_ref[(128 + i) * out + 256 + o], tile[i * 128 + o], 1e-9);
    // every element within one bf16 step of the f64 value (round to nearest is half a step; f32 arithmetic before it can
    // land a tie the other way) or, where its terms cancel to near zero, within the f32 bound of its terms (`unrotElemOk`;
    // the count past one step reported), nearly all exactly bf16(W_ref)
    const tile_abs = try a.alloc(f64, (in / 128) * (out / 128));
    defer a.free(tile_abs);
    for (0..in / 128) |rb| for (0..out / 128) |cb| {
        tile_abs[rb * (out / 128) + cb] = unrotTileAbsSum(w_hat, out, rb, cb);
    };
    var off_round: usize = 0;
    var over_step: usize = 0;
    for (w_std, w_ref, 0..) |g, r, e| {
        const i = e / out;
        const o = e % out;
        const terms = @abs(@as(f64, suh[i]) * @as(f64, svh[o])) / 128.0 * tile_abs[(i / 128) * (out / 128) + o / 128];
        try std.testing.expect(unrotElemOk(g, r, terms));
        if (@abs(@as(f64, g) - r) > bf16Step(r)) over_step += 1;
        if (g != cpuBf16(@floatCast(r))) off_round += 1;
    }
    try std.testing.expect(off_round * 1000 <= w_std.len);
    // y three ways for bf16-exact x: the chain (prep, the f16-out matmul with f32 accumulation, finish to bf16), the plain
    // matmul over W_std (bf16 out of an f32 accumulation) and the exact x W_ref (f64)
    const x = try a.alloc(f32, rows * in);
    defer a.free(x);
    for (x) |*e| e.* = cpuBf16(rnd.floatNorm(f32));
    const xp = try a.alloc(f32, in);
    defer a.free(xp);
    const z = try a.alloc(f32, out);
    defer a.free(z);
    const yc = try a.alloc(f32, out);
    defer a.free(yc);
    var se_pc: f64 = 0;
    var se_pe: f64 = 0;
    var se_ce: f64 = 0;
    var ss: f64 = 0;
    var max_pc: f64 = 0;
    var max_c: f64 = 0;
    for (0..rows) |rr| {
        const xr = x[rr * in ..][0..in];
        cpuPrep(xr, suh, xp);
        for (0..out) |o| {
            var acc: f32 = 0;
            for (0..in) |i| acc += xp[i] * w_hat[i * out + o];
            z[o] = cpuF16(acc);
        }
        cpuFinish(z, svh, yc);
        for (0..out) |o| {
            var acc: f32 = 0;
            var exact: f64 = 0;
            for (0..in) |i| {
                acc += xr[i] * w_std[i * out + o];
                exact += @as(f64, xr[i]) * w_ref[i * out + o];
            }
            const yp: f64 = cpuBf16(acc);
            const c: f64 = yc[o];
            se_pc += (yp - c) * (yp - c);
            se_pe += (yp - exact) * (yp - exact);
            se_ce += (c - exact) * (c - exact);
            ss += exact * exact;
            max_pc = @max(max_pc, @abs(yp - c));
            max_c = @max(max_c, @abs(c));
        }
    }
    const rms_pc = @sqrt(se_pc / ss);
    const rms_pe = @sqrt(se_pe / ss);
    const rms_ce = @sqrt(se_ce / ss);
    std.debug.print("TURBOEXL3_UNROT_CPU rows={d} in={d} out={d} w_off_round={d}/{d} w_over_1_step={d} (info: within the f32 bound of their terms) rel_rms plain_vs_chain={e:.3} plain_vs_exact={e:.3} chain_vs_exact={e:.3} max_abs plain_vs_chain={e:.3} (bf16 step at max|y| {e:.3})\n", .{ rows, in, out, off_round, w_std.len, over_step, rms_pc, rms_pe, rms_ce, max_pc, bf16Step(max_c) });
    // the bars the GPU tests hold the device to: plain vs chain rel RMS <= 2^-7.5, max |diff| <= 2 bf16 steps
    // of max|y|'s binade; the plain path within 2x the chain's own distance from the exact linear, plus 2^-10
    try std.testing.expect(rms_pc <= 0x1p-7 / std.math.sqrt2);
    try std.testing.expect(max_pc <= 2.0 * bf16Step(max_c));
    try std.testing.expect(rms_pe <= 2.0 * rms_ce + 0x1p-10);
}

test "turboexl3 dense un-rotated trunk texts: the un-rotation kernels repeat the dense butterfly verbatim, name their signatures, bind no MLX shape buffers" {
    const t = std.testing;
    const has = struct {
        fn f(hay: []const u8, needle: []const u8) bool {
            return std.mem.indexOf(u8, hay, needle) != null;
        }
    }.f;
    const fly = "for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }";
    const local4 = "{ float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }";
    for ([_][]const u8{ text_prep, text_finish, text_unrot_cols, text_unrot_rows }) |txt| {
        try t.expect(has(txt, fly));
        try t.expect(has(txt, local4));
    }
    // pass 1 reads the block's element L + 32 j of the row, as `text_finish` reads z; pass 2 the column's rows L + 32 j
    try t.expect(has(text_finish, "z[row * DIMC + b * 128u + L + 32u * j]"));
    try t.expect(has(text_unrot_cols, "w[ulong(row) * DIMC + b * 128u + L + 32u * j]"));
    try t.expect(has(text_unrot_rows, "v[j] = t[(L + 32u * j) * 33u + c];"));
    // the one rounding: 1/128, suh, svh in f32, then the output dtype
    try t.expect(has(text_unrot_rows, "OUT_T(t[(e >> 5u) * 33u + (e & 31u)] * 0.0078125f * float(suh[i]) * float(svh[o]))"));
    const Sig = struct { txt: []const u8, names: []const [*:0]const u8 };
    for ([_]Sig{ .{ .txt = text_unrot_cols, .names = &(unrot_cols_in ++ unrot_cols_out) }, .{ .txt = text_unrot_rows, .names = &(unrot_rows_in ++ unrot_rows_out) } }) |sg| for (sg.names) |nm| {
        const name = std.mem.span(nm);
        try t.expect(has(sg.txt, name));
        var buf: [32]u8 = undefined;
        for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| try t.expect(!has(sg.txt, std.fmt.bufPrint(&buf, "{s}{s}", .{ name, suf }) catch unreachable));
    };
}

test "turboexl3 dense un-rotated trunk table: an install replaces admitted or kept slots (bytes held once), refuses a member it cannot take, and release drops it" {
    const t = std.testing;
    const tab = try t.allocator.create(Resident.Table);
    defer t.allocator.destroy(tab);
    tab.* = .{};
    const base: usize = 0x6000_0000_0000;
    for (0..4) |i| try tab.admit(base + 64 * i, 1000 * (i + 1));
    const total: u64 = 1000 + 2000 + 3000 + 4000;
    try t.expectEqual(total, tab.pending());
    // slot 0 kept a W_hat first (its bytes held), then both are un-rotated as one group: held once each, nothing pending
    const s0 = tab.find(base).?;
    s0.state = .busy;
    tab.keep(s0, .{ .ctx = null });
    tab.groups += 1;
    tab.unrotate(s0, .{ .ctx = null }, .{ .ctx = null }, 0, tab.groups);
    tab.unrotate(tab.find(base + 64).?, .{ .ctx = null }, .{ .ctx = null }, 128, tab.groups);
    try t.expectEqual(@as(u32, 2), tab.kept);
    try t.expectEqual(@as(u32, 2), tab.unrot);
    try t.expectEqual(@as(u64, 3000), tab.held);
    try t.expectEqual(total - 3000, tab.pending());
    try t.expectEqual(Resident.State.unrot, tab.find(base + 64).?.state);
    try t.expectEqual(@as(u32, 128), tab.find(base + 64).?.col);
    // the process table: nothing un-rotated, so no plain arm, no W_std, no join
    try t.expect(!residentUnrot(.{ .ctx = @ptrFromInt(base) }));
    try t.expect(residentStd(.{ .ctx = @ptrFromInt(base) }) == null);
    try t.expect(residentJoined(.{ .ctx = @ptrFromInt(base) }) == null);
    try t.expectError(error.BadExl3Shape, residentInstall(&.{}, &.{}, null, &.{}));
    try t.expect(unrot_min_rows == 32 and max_join == 4);
}

test "turboexl3 dense resident trunk: the table admits each trellis once, bills what is admitted but not kept, refuses past half its slots" {
    const t = std.testing;
    const tab = try t.allocator.create(Resident.Table);
    defer t.allocator.destroy(tab);
    tab.* = .{};
    // the Qwen3.8-Flash-Next set's size (322 trellises), keys spaced like heap handles
    const base: usize = 0x6000_0000_0000;
    var total: u64 = 0;
    for (0..322) |i| {
        const out_tiles: c_int = @intCast(32 + 8 * (i % 5));
        const b = residentBytesOf(&.{ 160, out_tiles, 112 });
        try tab.admit(base + 48 * i, b);
        total += b;
    }
    try t.expectEqual(@as(u32, 322), tab.modules);
    try t.expectEqual(total, tab.planned);
    try t.expectEqual(total, tab.pending());
    // a repeat is ignored; every key is found, an unknown one is not
    try tab.admit(base, 1);
    try t.expectEqual(@as(u32, 322), tab.modules);
    try t.expectEqual(total, tab.planned);
    for (0..322) |i| try t.expect(tab.find(base + 48 * i) != null);
    try t.expect(tab.find(base + 48 * 322) == null);
    // kept: its bytes move from pending to held; failed: its bytes leave the bill
    const ka = tab.find(base).?;
    ka.state = .busy;
    tab.keep(ka, .{ .ctx = null });
    const kb = tab.find(base + 48).?;
    kb.state = .busy;
    tab.fail(kb);
    try t.expectEqual(@as(u32, 1), tab.kept);
    try t.expectEqual(ka.bytes, tab.held);
    try t.expectEqual(total - kb.bytes, tab.planned);
    try t.expectEqual(total - kb.bytes - ka.bytes, tab.pending());
    // at most half the slots; a zero key or byte count is no trellis
    const full = try t.allocator.create(Resident.Table);
    defer t.allocator.destroy(full);
    full.* = .{};
    for (0..Resident.cap / 2) |i| try full.admit(0x1000 + 16 * i, 2);
    try t.expectError(error.ResidentTableFull, full.admit(0x1000 + 16 * (Resident.cap / 2), 2));
    try t.expectError(error.BadExl3Shape, full.admit(0, 2));
    try t.expectError(error.BadExl3Shape, full.admit(0x10, 0));
    // a trellis shape's W_hat bytes, in x out x 2 (the GDN qkv projection; the lm_head the host leaves out); other ranks none
    try t.expectEqual(@as(u64, 52_428_800), residentBytesOf(&.{ 160, 640, 112 }));
    try t.expectEqual(@as(u64, 1_271_398_400), residentBytesOf(&.{ 160, 15520, 96 }));
    try t.expectEqual(@as(u64, 0), residentBytesOf(&.{ 160, 640 }));
    // the process's table is disarmed until the host arms it: nothing pending
    try t.expectEqual(@as(u64, 0), residentPending());
}

test "turboexl3 dense texts: inputs read, outputs written, the dequant layout names the GEMV's row/column rule" {
    const t = std.testing;
    try t.expect(std.mem.indexOf(u8, text_prep, "xp[row * DIMC") != null);
    try t.expect(std.mem.indexOf(u8, text_finish, "y[row * DIMC + o] = OUT_T(") != null);
    try t.expect(std.mem.indexOf(u8, text_dequant, "turboexl3_row_of(L, m & 3u)") != null);
    try t.expect(std.mem.indexOf(u8, text_dequant, "(L >> 2u) + 8u * (m >> 2u)") != null);
}

test "turboexl3 dense one-launch decode (DEC-ONE) texts: every value-setting line is the three-launch chain's, the signatures' names are used, no MLX shape buffers" {
    const t = std.testing;
    const has = struct {
        fn f(hay: []const u8, needle: []const u8) bool {
            return std.mem.indexOf(u8, hay, needle) != null;
        }
    }.f;
    // the GEMV (text_dgemv): weights, FMA order, quad fold, simdgroup sum
    const gemv_lines = [_][]const u8{
        "const auto ww = exl3_tile_load<N, 0>(tile, qb, ga, lp);",
        "const float wv = (DEFER != 0) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));",
        "for (uint r = 0; r < RBC; ++r) a[r][t][m >> 2] = fma(xr[r][m & 3u], wv, a[r][t][m >> 2]);",
        "for (uint r = 0; r < RBC; ++r) turboexl3_fold<BTC>(a[r], &red[sg][r * 16u * BTC], L);",
        "= ((red[0][i] + red[1][i]) + red[2][i]) + red[3][i];",
        "const half4 x4 = *reinterpret_cast<const device half4*>(xbase + ulong(row) * ulong(INC) + tk * 16u);",
        "xr[r][0] = float(x4.x); xr[r][1] = float(x4.y); xr[r][2] = float(x4.z); xr[r][3] = float(x4.w);",
    };
    for (gemv_lines) |l| {
        try t.expect(has(text_dgemv, l));
        try t.expect(has(text_dec0, l));
    }
    for (gemv_lines[0..5]) |l| try t.expect(has(text_dec1, l));
    // the finish (text_dfinish): planes in slice order, deferred map, rounding, H128, svh
    const fin_lines = [_][]const u8{
        "acc += z",
        "if (DEFER != 0) acc = fma(acc, MUL1_A, MUL1_C * xsum);",
        "v[j] = float(half(acc));",
        "for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }",
        "{ float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }",
        "OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))));",
        "const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));",
        "const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));",
        "for (uint k = 1u; k < KSC; ++k) acc += z",
    };
    for (fin_lines) |l| {
        try t.expect(has(text_dfinish, l));
        try t.expect(has(text_dec0, l));
        try t.expect(has(text_dec1, l));
    }
    try t.expect(has(text_dfinish, "if (DEFER != 0) for (uint q = 0; q < INB; ++q) xsum += xs[row * INB + q];"));
    try t.expect(has(text_dec0, "if (DEFER != 0) for (uint q = 0; q < NB; ++q) xsum += xs[row * NB + q];"));
    // the prepare inside (text_prep_s): rounding, suh, H128, scale, lane order, block sum
    const prep_lines = [_][]const u8{
        "v[j] = float(half(float(x[row * ",
        "]))) * float(suh[i]); }",
        "for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }",
        "const uint q = (r & 7u) >> 1u;",
        "const uint m = (r & 1u) | ((r >> 3u) << 1u);",
        "const half hv = half(v[j] * HAD_SCALE);",
        "sum += float(hv);",
        "sum = simd_sum(sum);",
        "((i & ~15u) + 4u * q + m)",
    };
    for (prep_lines) |l| {
        try t.expect(has(text_prep_s, l));
        try t.expect(has(text_dec1, l));
    }
    try t.expect(has(text_prep_sc, text_prep_s));
    // arrival: one relaxed count per threadgroup between device fences; the partials through coherent pointers
    for ([_][:0]const u8{ text_dec0, text_dec1 }) |txt| {
        try t.expect(has(txt, "atomic_fetch_add_explicit((device atomic_uint*)cnt + (ob * NCHC + chunk), 1u, memory_order_relaxed)"));
        try t.expect(has(txt, "device TURBOEXL3_COH float* zc = (device TURBOEXL3_COH float*)z;"));
        try t.expect(!has(txt, "atomic<float>") and !has(txt, "atomic_float"));
    }
    // MLX adds a shape / strides / ndim buffer for any input name followed by those suffixes in the source
    // `unused`: an input the kernel takes only for ordering (dec_zero's svh: a materialized weight, so the zeroing waits on nothing)
    const Sig = struct { txt: []const u8, names: []const [*:0]const u8, unused: []const u8 = "" };
    const sigs = [_]Sig{
        .{ .txt = text_dec0, .names = &(dec0_in ++ dec0_out) },
        .{ .txt = text_dec1, .names = &(dec1_in ++ dec1_out) },
        .{ .txt = text_prep_sc, .names = &(prep_in ++ prep_sc_out) },
        .{ .txt = text_dec_zero, .names = &(dec_zero_in ++ dec_zero_out), .unused = "svh" },
    };
    for (sigs) |sg| for (sg.names) |n| {
        const name = std.mem.span(n);
        if (!std.mem.eql(u8, name, sg.unused)) try t.expect(has(sg.txt, name));
        var buf: [32]u8 = undefined;
        for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| try t.expect(!has(sg.txt, std.fmt.bufPrint(&buf, "{s}{s}", .{ name, suf }) catch unreachable));
    };
    // the prepare inside fits the trunk's decode geometries (K slices of whole 128-blocks, <= 32 KB at RB 4, BT 2)
    const sh = Shape{ .in = 2560, .out = 12288, .rate = format.Rate.fromN(112).?, .dec = .mul1, .out_dtype = .bfloat16 };
    try t.expect(Dense.decPrepFits(sh, 1, 4, 2));
    try t.expect(Dense.decPrepFits(.{ .in = 6144, .out = 2560, .rate = sh.rate, .dec = .mul1, .out_dtype = .bfloat16 }, 4, 4, 2));
    try t.expect(!Dense.decPrepFits(.{ .in = 6144, .out = 2560, .rate = sh.rate, .dec = .mul1, .out_dtype = .bfloat16 }, 1, 4, 2));
    try t.expect(!Dense.decPrepFits(sh, 8, 4, 2)); // PER 5: a slice ends inside a 128-block
}

test "turboexl3 dense DEC-ONE switch: unset or empty on, 0 off, anything else on" {
    const t = std.testing;
    try t.expect(decOneOn(null));
    try t.expect(decOneOn(""));
    try t.expect(!decOneOn("0"));
    try t.expect(decOneOn("1"));
    try t.expect(decOneOn("yes"));
}
