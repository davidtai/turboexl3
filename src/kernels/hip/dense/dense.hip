// turboexl3 on HIP: the dense projection's GEMV chain (docs/KERNELS.md section 4): prep_s, dgemv (and the lean GEMV),
// dfinish. HIP forms of src/kernels/dense/prep_s.metal, dgemv_1/_x/_2/_m/_3.metal (dgemv_j for the lean text) and
// dfinish_v/_store.metal: the same partition (K slices fixed per shape, four parts a slice, planes in slice order), the
// same rounding points (format.project's) and the deferred MUL1 map. The rate n and the codeword mask are launch values.

#include "common.hpp"

namespace {

using namespace t3;

// Grid (IN / 128, rows), 32 threads: xp = f16(H128(f16(x) suh)) in lane order (or natural order), and each 128-block's
// sum of xp (the deferred map's input sums).
template <int DT>
__device__ void prep_s(const void* x, const uint16_t* suh, uint16_t* xp, float* xs, uint32_t in, int order) {
    const uint32_t b = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    float v[4];
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t i = b * 128u + lane + 32u * j;
        v[j] = round_half(load_act<DT>(x, static_cast<size_t>(row) * in + i)) * half_value(suh[i]);
    }
    h128(v, lane);
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t i = b * 128u + lane + 32u * j;
        const uint16_t hv = half_bits(v[j] * kHadScale);
        xp[static_cast<size_t>(row) * in + (order ? lane_order(i) : i)] = hv;
        sum += half_value(hv);
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) sum += __shfl_xor(sum, m, 32);
    if (lane == 0u) xs[static_cast<size_t>(row) * (in / 128u) + b] = sum;
}

// output tiles a threadgroup for RB rows (Dense.btFor): 4 at one row, 2 at 2-4, 1 beyond
template <uint32_t RB>
constexpr uint32_t bt_for() {
    return RB == 1u ? 4u : (RB <= 4u ? 2u : 1u);
}

// Grid (OUT / 16 / BT, chunks, KS), 128 threads (4 waves). A threadgroup owns BT output tiles, a chunk of RB rows and
// K slice ks; wave sg takes PER = IN_TILES / (4 KS) input tiles of it. Lane L folds weight m of each tile into
// accumulator m >> 2 with the row's input element m & 3 (a quad's four rows of the tile, lane order), in input-tile
// order, m = 0..7. z[(ks ROWS + row) OUT + column] = the four waves' sums, wave 0 first. LEAN: the codeword loop taken by
// input element (dgemv_j.metal): each accumulator gets the same FMAs in the same order, one input a row live.
template <int CB, int DEFER, uint32_t RB, int LEAN>
__device__ void dgemv(const uint32_t* w, const uint16_t* xp, float* z, uint32_t in, uint32_t out, uint32_t rows,
                      uint32_t n, uint32_t mask) {
    constexpr uint32_t BT = bt_for<RB>();
    __shared__ float red[4][RB * 16u * BT];
    const uint32_t blk = blockIdx.x, r0 = blockIdx.y * RB, ksl = blockIdx.z, ks = gridDim.z;
    const uint32_t sg = threadIdx.x >> 5, lane = threadIdx.x & 31u, tid = threadIdx.x;
    const uint32_t in_tiles = in / 16u, out_tiles = out / 16u, nw = n / 2u;
    const uint32_t per = in_tiles / (ks * 4u);
    const uint32_t tk0 = (ksl * 4u + sg) * per;
    const uint32_t base = lane_base(n, lane);
    uint32_t off[8];
#pragma unroll
    for (uint32_t m = 0; m < 8u; ++m) off[m] = lane_off(n, lane, m, base);
    float a[RB][BT][2];
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r)
#pragma unroll
        for (uint32_t t = 0; t < BT; ++t) a[r][t][0] = a[r][t][1] = 0.0f;
    const uint16_t* xbase = xp + 4u * (lane & 3u);
    for (uint32_t tk = tk0; tk < tk0 + per; ++tk) {
        float xr[RB][4];
        if constexpr (LEAN == 0) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) {
                // a chunk's spare rows reread the last row; their sums are never written
                const uint32_t row = min(r0 + r, rows - 1u);
                const uint16_t* xq = xbase + static_cast<size_t>(row) * in + tk * 16u;
#pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) xr[r][j] = half_value(xq[j]);
            }
        }
#pragma unroll
        for (uint32_t t = 0; t < BT; ++t) {
            const uint32_t* tile = w + (static_cast<size_t>(tk) * out_tiles + blk * BT + t) * nw;
            const LaneWords ww = lane_words(tile, n, base);
            uint32_t cw[8];
#pragma unroll
            for (uint32_t m = 0; m < 8u; ++m) cw[m] = lane_cw(ww, off[m], mask);
            if constexpr (LEAN == 0) {
#pragma unroll
                for (uint32_t m = 0; m < 8u; ++m) {
                    const float wv = weight<CB, DEFER>(cw[m]);
#pragma unroll
                    for (uint32_t r = 0; r < RB; ++r) a[r][t][m >> 2] = __builtin_fmaf(xr[r][m & 3u], wv, a[r][t][m >> 2]);
                }
            } else {
#pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) {
                    const float w0 = weight<CB, DEFER>(cw[j]);
                    const float w1 = weight<CB, DEFER>(cw[j + 4u]);
#pragma unroll
                    for (uint32_t r = 0; r < RB; ++r) {
                        const float xv = half_value(xbase[static_cast<size_t>(min(r0 + r, rows - 1u)) * in + tk * 16u + j]);
                        a[r][t][0] = __builtin_fmaf(xv, w0, a[r][t][0]);
                        a[r][t][1] = __builtin_fmaf(xv, w1, a[r][t][1]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) fold<BT>(a[r], &red[sg][r * 16u * BT], lane);
    __syncthreads();
    for (uint32_t i = tid; i < RB * 16u * BT; i += 128u) {
        const uint32_t r = i / (16u * BT);
        const uint32_t row = r0 + r;
        if (row < rows)
            z[(static_cast<size_t>(ksl) * rows + row) * out + blk * (16u * BT) + (i - r * 16u * BT)] =
                ((red[0][i] + red[1][i]) + red[2][i]) + red[3][i];
    }
}

// Grid (OUT / 128, rows), 32 threads: the planes summed in slice order, the deferred map (A acc + C sum x), f16, the
// output H128, svh, f16, then y in its dtype.
template <int DT>
__device__ void dfinish(const float* z, const float* xs, const uint16_t* svh, void* y, uint32_t out, uint32_t rows,
                        uint32_t ks, uint32_t inblk, int defer) {
    const uint32_t b = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    float xsum = 0.0f;
    if (defer)
        for (uint32_t q = 0; q < inblk; ++q) xsum += xs[static_cast<size_t>(row) * inblk + q];
    float v[4];
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t o = b * 128u + lane + 32u * j;
        float acc = z[static_cast<size_t>(row) * out + o];
        for (uint32_t k = 1u; k < ks; ++k) acc += z[(static_cast<size_t>(k) * rows + row) * out + o];
        if (defer) acc = __builtin_fmaf(acc, mul1_a(), mul1_c() * xsum);
        v[j] = round_half(acc);
    }
    h128(v, lane);
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t o = b * 128u + lane + 32u * j;
        store_act<DT>(y, static_cast<size_t>(row) * out + o, round_half(v[j] * kHadScale * half_value(svh[o])));
    }
}

}  // namespace

#define T3_PREP_S(NAME, DT)                                                                                          \
    extern "C" __global__ void __launch_bounds__(32) t3_dense_prep_s_##NAME(const void* x, const uint16_t* suh,    \
                                                                           uint16_t* xp, float* xs, uint32_t in,   \
                                                                           int order) {                            \
        prep_s<DT>(x, suh, xp, xs, in, order);                                                                     \
    }
T3_PREP_S(f32, kF32)
T3_PREP_S(f16, kF16)
T3_PREP_S(bf16, kBf16)
#undef T3_PREP_S

// t3_dense_dgemv_<codebook>_<rows>: mcg, mul1 (the map off), mul1d (the deferred map); rows 1-5 on the dgemv text,
// 6-8 on the lean text
#define T3_DGEMV(NAME, CB, DEFER, RB, LEAN)                                                                          \
    extern "C" __global__ void __launch_bounds__(128) t3_dense_dgemv_##NAME##_##RB(                                 \
        const uint32_t* w, const uint16_t* xp, float* z, uint32_t in, uint32_t out, uint32_t rows, uint32_t n,       \
        uint32_t mask) {                                                                                             \
        dgemv<CB, DEFER, RB, LEAN>(w, xp, z, in, out, rows, n, mask);                                                \
    }
#define T3_DGEMV_ROWS(NAME, CB, DEFER) \
    T3_DGEMV(NAME, CB, DEFER, 1, 0)    \
    T3_DGEMV(NAME, CB, DEFER, 2, 0)    \
    T3_DGEMV(NAME, CB, DEFER, 3, 0)    \
    T3_DGEMV(NAME, CB, DEFER, 4, 0)    \
    T3_DGEMV(NAME, CB, DEFER, 5, 0)    \
    T3_DGEMV(NAME, CB, DEFER, 6, 1)    \
    T3_DGEMV(NAME, CB, DEFER, 7, 1)    \
    T3_DGEMV(NAME, CB, DEFER, 8, 1)
T3_DGEMV_ROWS(mcg, kMcg, 0)
T3_DGEMV_ROWS(mul1, kMul1, 0)
T3_DGEMV_ROWS(mul1d, kMul1, 1)
#undef T3_DGEMV_ROWS
#undef T3_DGEMV

// the lean text at 1-5 rows too, for the test that holds it to dgemv's bytes
#define T3_LEAN(NAME, CB, DEFER, RB)                                                                                 \
    extern "C" __global__ void __launch_bounds__(128) t3_dense_lean_##NAME##_##RB(                                  \
        const uint32_t* w, const uint16_t* xp, float* z, uint32_t in, uint32_t out, uint32_t rows, uint32_t n,       \
        uint32_t mask) {                                                                                             \
        dgemv<CB, DEFER, RB, 1>(w, xp, z, in, out, rows, n, mask);                                                   \
    }
T3_LEAN(mcg, kMcg, 0, 2)
T3_LEAN(mcg, kMcg, 0, 5)
T3_LEAN(mul1d, kMul1, 1, 5)
#undef T3_LEAN

#define T3_DFINISH(NAME, DT)                                                                                         \
    extern "C" __global__ void __launch_bounds__(32) t3_dense_dfinish_##NAME(                                       \
        const float* z, const float* xs, const uint16_t* svh, void* y, uint32_t out, uint32_t rows, uint32_t ks,     \
        uint32_t inblk, int defer) {                                                                                 \
        dfinish<DT>(z, xs, svh, y, out, rows, ks, inblk, defer);                                                     \
    }
T3_DFINISH(f32, kF32)
T3_DFINISH(f16, kF16)
T3_DFINISH(bf16, kBf16)
#undef T3_DFINISH
