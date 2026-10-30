// turboexl3 on HIP: the routed experts at prefill rows (docs/KERNELS.md section 3) and the dense GEMM above 16 rows
// (section 4): route -> prep -> GEMM (gate + up) -> mid -> GEMM (down) -> fin. The chain of src/kernels/prefill/
// (pf_froute, pf_prep, pf_gemm2 / pf_gemm1, pf_mid, pf_fin) with the matrix product on gfx11's
// v_wmma_f32_16x16x16_f16 in place of NAX: each 32-deep K step decodes its 2 x 8 trellis tiles into LDS as f16
// (the shared-decode body), then two WMMA a 16 x 16 accumulator, K ascending: 64-k chains from zero, folded into the
// accumulator on the VALU (F-K6). A pair's sums do not depend on where its slot lands in the sorted order, the tile it
// shares, or the pass of 64 rows it falls in.
//
// The rounding profiles are decode.hip's (PROF 0 format.project's, PROF 1 GLM's); the finishes are routed.hpp's.

#include "routed.hpp"

namespace {

using namespace t3;

constexpr uint32_t kRouteThreads = 1024;

// One block, 1024 threads (an expert a thread, experts <= 1024): the routed pairs (slot ids < experts) sorted by expert,
// in pair order within an expert (a stable counting sort). order[p] = the pair at sorted position p; inv[rk] = the
// position of pair rk (-1 for a skipped slot); items[3 i] = (expert, first position, count) of at most `bm` positions,
// an expert's tiles in order, `cap` item slots (dead ones count 0); counts[0] = live items, counts[1] = positions.
__device__ void route_kernel(const int* slots, uint32_t rk_count, uint32_t experts,
                                                              uint32_t bm, int* order, int* inv, int* items,
                                                              uint32_t cap, int* counts) {
    __shared__ uint32_t scan_c[kRouteThreads];
    __shared__ uint32_t scan_t[kRouteThreads];
    const uint32_t e = threadIdx.x;
    uint32_t c = 0;
    if (e < experts)
        for (uint32_t rk = 0; rk < rk_count; ++rk) c += static_cast<uint32_t>(slots[rk]) == e;
    const uint32_t tiles = (c + bm - 1u) / bm;
    scan_c[e] = c;
    scan_t[e] = tiles;
    __syncthreads();
    for (uint32_t off = 1; off < kRouteThreads; off <<= 1) {
        const uint32_t vc = e >= off ? scan_c[e - off] : 0u, vt = e >= off ? scan_t[e - off] : 0u;
        __syncthreads();
        scan_c[e] += vc;
        scan_t[e] += vt;
        __syncthreads();
    }
    const uint32_t start = scan_c[e] - c, item0 = scan_t[e] - tiles;
    if (e < experts) {
        uint32_t j = 0;
        for (uint32_t rk = 0; rk < rk_count; ++rk)
            if (static_cast<uint32_t>(slots[rk]) == e) {
                order[start + j] = static_cast<int>(rk);
                inv[rk] = static_cast<int>(start + j);
                ++j;
            }
        for (uint32_t t = 0; t < tiles; ++t) {
            items[3u * (item0 + t)] = static_cast<int>(e);
            items[3u * (item0 + t) + 1u] = static_cast<int>(start + t * bm);
            items[3u * (item0 + t) + 2u] = static_cast<int>(min(bm, c - t * bm));
        }
    }
    for (uint32_t rk = threadIdx.x; rk < rk_count; rk += kRouteThreads) {
        const int s = slots[rk];
        if (s < 0 || static_cast<uint32_t>(s) >= experts) inv[rk] = -1;
    }
    const uint32_t live = scan_t[kRouteThreads - 1u];
    for (uint32_t i = live + threadIdx.x; i < cap; i += kRouteThreads) {
        items[3u * i] = 0;
        items[3u * i + 1u] = 0;
        items[3u * i + 2u] = 0;
    }
    if (threadIdx.x == 0) {
        counts[0] = static_cast<int>(live);
        counts[1] = static_cast<int>(scan_c[kRouteThreads - 1u]);
    }
}

// Grid (hidden / 128, positions), 32 threads: each sorted position's gate and up inputs (natural order: the GEMM's A),
// xg / xu f16 [P][hidden], and their 128-block sums. Positions past counts[1] are skipped. With `shared` (every routed
// expert's gate suh the same, and every up suh: a checkpoint's shared input), grid (hidden / 128, rows): each row's
// inputs once, with expert 0's suh, xg / xu [rows][hidden]; the GEMM and mid read row order[p] / topk (their `xdiv`).
template <int DT, int PROF>
__device__ void prep(const void* x, const int* order, const int* slots, const int* counts, const uint16_t* gate_suh,
                     const uint16_t* up_suh, uint16_t* xg, uint16_t* xu, float* sg, float* su, uint32_t hidden,
                     uint32_t topk, int shared) {
    const uint32_t b = blockIdx.x, p = blockIdx.y, lane = threadIdx.x;  // p: a sorted position, or a row (shared)
    size_t e = 0;
    uint32_t row = p;
    if (!shared) {
        if (p >= static_cast<uint32_t>(counts[1])) return;
        const uint32_t rk = static_cast<uint32_t>(order[p]);
        e = static_cast<size_t>(slots[rk]);
        row = rk / topk;
    }
    float vg[4], vu[4];
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t i = b * 128u + lane + 32u * j;
        float xr = load_act<DT>(x, static_cast<size_t>(row) * hidden + i);
        if constexpr (PROF == 0) xr = round_half(xr);
        vg[j] = xr * half_value(gate_suh[e * hidden + i]);
        vu[j] = xr * half_value(up_suh[e * hidden + i]);
    }
    h128(vg, lane);
    h128(vu, lane);
    float sgv = 0.0f, suv = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t i = b * 128u + lane + 32u * j;
        const uint16_t hg = half_bits(vg[j] * kHadScale), hu = half_bits(vu[j] * kHadScale);
        xg[static_cast<size_t>(p) * hidden + i] = hg;
        xu[static_cast<size_t>(p) * hidden + i] = hu;
        sgv += half_value(hg);
        suv += half_value(hu);
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        sgv += __shfl_xor(sgv, m, 32);
        suv += __shfl_xor(suv, m, 32);
    }
    if (lane == 0u) {
        sg[static_cast<size_t>(p) * (hidden / 128u) + b] = sgv;
        su[static_cast<size_t>(p) * (hidden / 128u) + b] = suv;
    }
}

constexpr uint32_t kBM = 64, kBK = 32, kLK = kBK + 8;  // rows a pass, K a step, halves an LDS row (padded)

typedef _Float16 h16v __attribute__((ext_vector_type(16)));
typedef float f8v __attribute__((ext_vector_type(8)));

// Grid (N / 128, item slots, mats), 256 threads (8 waves: 2 along m x 4 along n). A block owns one item (an expert's
// tile of at most 64 positions) and one 128-column block: z[(mat P + p) N + column] = X[p] W_q for the item's
// positions p (f32; MUL1's t under the deferred map). Weights: TABLE 1 (the routed bank), W0 and W1 are u64 [E] tables
// of each expert's trellis address (routed.hpp expert_trellis); TABLE 0 (a dense matrix, item expert 0), the trellis.
// X's row of position p: p (xdiv 0), or order[p] / xdiv (prep's `shared`).
// The sum over K: FOLD > 0, each group of FOLD K steps (FOLD x 32 k) is one WMMA chain from zero, and the groups add
// into the accumulator on the VALU (IEEE fp32), in K order. gfx11's WMMA fp32 adds are not IEEE (they truncate), so
// one chain over K drifts with K; 64-k groups (FOLD 2) keep each chain short. FOLD 0: one chain over K (the reference
// the tests measure the fold against). Either way, one order for every row: a row's bits do not depend on its pass.
template <int CB, int DEFER, int TABLE, int FOLD>
__device__ void gemm(const uint16_t* X0, const uint16_t* X1, const void* W0, const void* W1, const int* items,
                     float* z, uint32_t k, uint32_t n_out, uint32_t positions, uint32_t n, uint32_t mask,
                     const int* order, uint32_t xdiv) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
    const uint32_t nb = blockIdx.x, it = blockIdx.y, mat = blockIdx.z;
    const uint32_t e = static_cast<uint32_t>(items[3u * it]), first = static_cast<uint32_t>(items[3u * it + 1u]);
    const uint32_t count = static_cast<uint32_t>(items[3u * it + 2u]);
    if (count == 0u) return;
    __shared__ __attribute__((aligned(16))) _Float16 As[2][kBM * kLK];
    __shared__ __attribute__((aligned(16))) _Float16 Bs[2][128 * kLK];
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t wm = warp & 1u, wn = warp >> 1, l16 = lane & 15u;
    const uint32_t nt_all = n_out / 16u, iters = k / kBK, nw = n / 2u;
    const uint16_t* X = mat ? X1 : X0;
    const void* W = mat ? W1 : W0;
    const uint32_t* T = TABLE ? expert_trellis(static_cast<const uint64_t*>(W), e) : static_cast<const uint32_t*>(W);
    const uint32_t arow = threadIdx.x >> 2, akc = threadIdx.x & 3u;
    const uint32_t base = lane_base(n, lane);
    uint32_t off[8];
#pragma unroll
    for (uint32_t m = 0; m < 8u; ++m) off[m] = lane_off(n, lane, m, base);
    for (uint32_t h0 = 0; h0 < count; h0 += kBM) {
        const uint32_t rows = count - h0 < kBM ? count - h0 : kBM;
        const bool alive = arow < rows;
        const uint32_t pa = first + h0 + arow;
        const size_t xi = !alive ? 0u : xdiv ? static_cast<uint32_t>(order[pa]) / xdiv : pa;
        const uint16_t* xa = X + xi * k + akc * 8u;
        uint4 ra;
        LaneWords rb[2];
        auto gload = [&](uint32_t s) {
            ra = alive ? *reinterpret_cast<const uint4*>(xa + s * kBK) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
            for (uint32_t u = 0; u < 2u; ++u) {
                const uint32_t q = 2u * warp + u, kk = q >> 3, nt = q & 7u;
                rb[u] = lane_words(T + (static_cast<size_t>(s * 2u + kk) * nt_all + nb * 8u + nt) * nw, n, base);
            }
        };
        auto sstore = [&](uint32_t buf) {
            *reinterpret_cast<uint4*>(&As[buf][arow * kLK + akc * 8u]) = ra;
#pragma unroll
            for (uint32_t u = 0; u < 2u; ++u) {
                const uint32_t q = 2u * warp + u, kk = q >> 3, nt = q & 7u;
#pragma unroll
                for (uint32_t m = 0; m < 8u; ++m) {
                    const uint32_t cw = lane_cw(rb[u], off[m], mask);
                    const uint16_t wv = DEFER ? half_bits(static_cast<float>(mul1_t(cw))) : decode<CB>(cw);
                    const uint32_t r = 2u * (lane & 3u) + (m & 1u) + 8u * ((m >> 1) & 1u);  // k within the tile
                    const uint32_t c = (lane >> 2) + 8u * (m >> 2);                           // n within the tile
                    Bs[buf][(nt * 16u + c) * kLK + kk * 16u + r] = __builtin_bit_cast(_Float16, wv);
                }
            }
        };
        const f8v zero = f8v{0, 0, 0, 0, 0, 0, 0, 0};
        f8v acc[2][2];
        f8v d[2][2];  // FOLD: the group's WMMA chain
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i)
#pragma unroll
            for (uint32_t j = 0; j < 2u; ++j) acc[i][j] = d[i][j] = zero;
        gload(0);
        sstore(0);
        __syncthreads();
        for (uint32_t s = 0; s < iters; ++s) {
            const uint32_t buf = s & 1u;
            if (s + 1u < iters) gload(s + 1u);
#pragma unroll
            for (uint32_t kk = 0; kk < 2u; ++kk) {
                h16v af[2], bf[2];
#pragma unroll
                for (uint32_t i = 0; i < 2u; ++i) af[i] = *reinterpret_cast<const h16v*>(&As[buf][(wm * 32u + i * 16u + l16) * kLK + kk * 16u]);
#pragma unroll
                for (uint32_t j = 0; j < 2u; ++j) bf[j] = *reinterpret_cast<const h16v*>(&Bs[buf][(wn * 32u + j * 16u + l16) * kLK + kk * 16u]);
#pragma unroll
                for (uint32_t i = 0; i < 2u; ++i)
#pragma unroll
                    for (uint32_t j = 0; j < 2u; ++j) {
                        if constexpr (FOLD > 0) {
                            const bool start = kk == 0u && s % FOLD == 0u;
                            d[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[i], bf[j], start ? zero : d[i][j]);
                        } else {
                            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[i], bf[j], acc[i][j]);
                        }
                    }
            }
            if constexpr (FOLD > 0) {
                if ((s + 1u) % FOLD == 0u || s + 1u == iters) {
#pragma unroll
                    for (uint32_t i = 0; i < 2u; ++i)
#pragma unroll
                        for (uint32_t j = 0; j < 2u; ++j)
#pragma unroll
                            for (uint32_t q = 0; q < 8u; ++q) acc[i][j][q] = acc[i][j][q] + d[i][j][q];
                }
            }
            if (s + 1u < iters) sstore(buf ^ 1u);
            __syncthreads();
        }
        // accumulator element r of lane l: row 2 r + l / 16, column l % 16 of its 16 x 16 block
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i)
#pragma unroll
            for (uint32_t r = 0; r < 8u; ++r) {
                const uint32_t m = wm * 32u + i * 16u + 2u * r + (lane >> 4);
                if (m >= rows) continue;
                float* o = z + (static_cast<size_t>(mat) * positions + first + h0 + m) * n_out + nb * 128u + wn * 32u + l16;
#pragma unroll
                for (uint32_t j = 0; j < 2u; ++j) o[j * 16u] = acc[i][j][r];
            }
        __syncthreads();
    }
#else
    __builtin_trap();
#endif
}

// Grid (inter / 128, positions), 128 threads: each sorted position's gate and up finish, SwiGLU and the down's prepare
// (natural order: the down GEMM's A), with its block sums. The input's block sums xsg / xsu: position p (xdiv 0), or
// row order[p] / xdiv (prep's `shared`).
template <int PROF>
__device__ void mid(const float* z, const int* order, const int* slots, const int* counts, const uint16_t* gate_svh,
                    const uint16_t* up_svh, const uint16_t* down_suh, const float* xsg, const float* xsu, uint16_t* hd,
                    float* hs, uint32_t hidden, uint32_t inter, uint32_t positions, int defer, float limit,
                    uint32_t xdiv) {
    __shared__ float xch[128];
    const uint32_t blk = blockIdx.x, p = blockIdx.y, tid = threadIdx.x;
    if (p >= static_cast<uint32_t>(counts[1])) return;
    const size_t e = static_cast<size_t>(slots[order[p]]);
    const uint32_t nb = hidden / 128u;
    const size_t xr = xdiv ? static_cast<uint32_t>(order[p]) / xdiv : p;
    float sgx = 0.0f, sux = 0.0f;
    if (defer)
        for (uint32_t q = 0; q < nb; ++q) {
            sgx += xsg[xr * nb + q];
            sux += xsu[xr * nb + q];
        }
    // the gate's z at plane 0, the up's at plane 1 (the GEMM's mat), one slice each
    finish_gate_up<PROF>(z, z + static_cast<size_t>(positions) * inter, p, positions, 1u, e, gate_svh, up_svh, down_suh,
                         sgx, sux, hd, hs, p, inter, blk, defer, limit, false, xch, tid);
}

// Grid (hidden / 128, rows), 128 threads: each row's slots in order through inv: PROF 0 the down finish times its score,
// summed into y [rows][hidden] (DT); PROF 1 each routed slot's finish as f32 ys [RK][hidden].
template <int DT, int PROF>
__device__ void fin(const float* zd, const int* inv, const int* slots, const float* scores, const uint16_t* down_svh,
                    const float* hsx, void* y, uint32_t hidden, uint32_t inter, uint32_t topk, uint32_t positions,
                    int defer) {
    __shared__ float xch[128];
    const uint32_t blk = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    const uint32_t o = blk * 128u + tid;
    const uint32_t nbi = inter / 128u;
    float acc = 0.0f;
    for (uint32_t k = 0; k < topk; ++k) {
        const uint32_t rk = row * topk + k;
        const int p = inv[rk];
        if (p < 0) continue;  // uniform over the block
        float hsum = 0.0f;
        if (defer)
            for (uint32_t q = 0; q < nbi; ++q) hsum += hsx[static_cast<size_t>(p) * nbi + q];
        const float v = finish_down<PROF>(zd, static_cast<size_t>(p), positions, 1u, hsum, defer, down_svh,
                                          static_cast<size_t>(slots[rk]), hidden, o, xch, tid);
        if constexpr (PROF == 0) acc = __builtin_fmaf(scores[rk], v, acc);
        else static_cast<float*>(y)[static_cast<size_t>(rk) * hidden + o] = v;
    }
    if constexpr (PROF == 0) store_act<DT>(y, static_cast<size_t>(row) * hidden + o, acc);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kRouteThreads) t3_pf_route(const int* slots, uint32_t rk_count,
                                                                       uint32_t experts, uint32_t bm, int* order,
                                                                       int* inv, int* items, uint32_t cap, int* counts) {
    route_kernel(slots, rk_count, experts, bm, order, inv, items, cap, counts);
}

#define T3_PF_PREP(PNAME, PROF, DNAME, DT)                                                                           \
    extern "C" __global__ void __launch_bounds__(32) t3_pf_prep_##PNAME##_##DNAME(                                  \
        const void* x, const int* order, const int* slots, const int* counts, const uint16_t* gate_suh,              \
        const uint16_t* up_suh, uint16_t* xg, uint16_t* xu, float* sg, float* su, uint32_t hidden, uint32_t topk,    \
        int shared) {                                                                                                \
        prep<DT, PROF>(x, order, slots, counts, gate_suh, up_suh, xg, xu, sg, su, hidden, topk, shared);             \
    }
T3_PF_PREP(exl3, 0, f32, kF32)
T3_PF_PREP(exl3, 0, f16, kF16)
T3_PF_PREP(exl3, 0, bf16, kBf16)
T3_PF_PREP(glm, 1, bf16, kBf16)
#undef T3_PF_PREP

// t3_pf_gemm_*: the routed bank (u64 trellis tables); t3_pf_dgemm_*: one dense matrix (its trellis). Both fold 64-k
// groups (FOLD 2); t3_pf_gemm_chain_mcg is the one-chain form, for the tests' before / after.
#define T3_PF_GEMM(KIND, TABLE, WT, FOLD, NAME, CB, DEFER)                                                           \
    extern "C" __global__ void __launch_bounds__(256) t3_pf_##KIND##_##NAME(                                         \
        const uint16_t* X0, const uint16_t* X1, const WT* W0, const WT* W1, const int* items, float* z, uint32_t k,  \
        uint32_t n_out, uint32_t positions, uint32_t n, uint32_t mask, const int* order, uint32_t xdiv) {            \
        gemm<CB, DEFER, TABLE, FOLD>(X0, X1, W0, W1, items, z, k, n_out, positions, n, mask, order, xdiv);           \
    }
T3_PF_GEMM(gemm, 1, uint64_t, 2, mcg, kMcg, 0)
T3_PF_GEMM(gemm, 1, uint64_t, 2, mul1, kMul1, 0)
T3_PF_GEMM(gemm, 1, uint64_t, 2, mul1d, kMul1, 1)
T3_PF_GEMM(dgemm, 0, uint32_t, 2, mcg, kMcg, 0)
T3_PF_GEMM(dgemm, 0, uint32_t, 2, mul1, kMul1, 0)
T3_PF_GEMM(dgemm, 0, uint32_t, 2, mul1d, kMul1, 1)
T3_PF_GEMM(gemm_chain, 1, uint64_t, 0, mcg, kMcg, 0)
#undef T3_PF_GEMM

#define T3_PF_MID(PNAME, PROF)                                                                                       \
    extern "C" __global__ void __launch_bounds__(128) t3_pf_mid_##PNAME(                                            \
        const float* z, const int* order, const int* slots, const int* counts, const uint16_t* gate_svh,             \
        const uint16_t* up_svh, const uint16_t* down_suh, const float* xsg, const float* xsu, uint16_t* hd,          \
        float* hs, uint32_t hidden, uint32_t inter, uint32_t positions, int defer, float limit, uint32_t xdiv) {     \
        mid<PROF>(z, order, slots, counts, gate_svh, up_svh, down_suh, xsg, xsu, hd, hs, hidden, inter, positions,   \
                  defer, limit, xdiv);                                                                               \
    }
T3_PF_MID(exl3, 0)
T3_PF_MID(glm, 1)
#undef T3_PF_MID

#define T3_PF_FIN(PNAME, PROF, DNAME, DT)                                                                            \
    extern "C" __global__ void __launch_bounds__(128) t3_pf_fin_##PNAME##_##DNAME(                                  \
        const float* zd, const int* inv, const int* slots, const float* scores, const uint16_t* down_svh,            \
        const float* hsx, void* y, uint32_t hidden, uint32_t inter, uint32_t topk, uint32_t positions, int defer) {  \
        fin<DT, PROF>(zd, inv, slots, scores, down_svh, hsx, y, hidden, inter, topk, positions, defer);              \
    }
T3_PF_FIN(exl3, 0, f32, kF32)
T3_PF_FIN(exl3, 0, f16, kF16)
T3_PF_FIN(exl3, 0, bf16, kBf16)
T3_PF_FIN(glm, 1, f32, kF32)
#undef T3_PF_FIN
