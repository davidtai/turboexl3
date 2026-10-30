// turboexl3 on HIP: the routed experts at decode rows, form 3 (docs/KERNELS.md section 1): prep2 -> gu3 -> fin3 ->
// down3 -> dfin3. HIP forms of src/kernels/decode/prep2_1/_2.metal, gu3_*.metal, fin3.metal, down3_*.metal and
// dfin3.metal with the same partition: gate/up over hidden / 16 input tiles in KS x 2 parts (wave = projection x half),
// down over inter / 16 tiles in KS2 x 4 parts, planes added in slice order. One row-slot a threadgroup (GROUP 1) and one
// K slice a threadgroup (SPT 1): launch shape only, the sums are the partition's.
//
// Two rounding profiles, chosen at compile time (a piece set each, not a text substitution):
//   PROF 0, format.project's (turboexl3's packs): x rounds to f16 before suh, z rounds to f16 before each finish,
//     SwiGLU in f32, y = sum over slots of score x f16(finish), in the output dtype.
//   PROF 1, GLM's (TensorFold's GLM family and its CUDA path): x times suh in f32, z stays f32, gate and up round to
//     bf16 and are limited, silu rounds to bf16, the product rounds to bf16; each routed slot's down finish is stored as
//     f32 [row-slot][hidden] with no score (the host combines the slots and the shared expert). A slot whose expert id
//     is >= `experts` (the shared expert) is skipped and its output left as it is.

#include "routed.hpp"

namespace {

using namespace t3;

// Grid (hidden / 128, RK), 32 threads: each routed slot's gate and up inputs, xg / xu f16 [RK][hidden] in lane order,
// and each 128-block's sums sg / su (the deferred map).
template <int DT, int PROF>
__device__ void prep2(const void* x, const int* slots, const uint16_t* gate_suh, const uint16_t* up_suh, uint16_t* xg,
                      uint16_t* xu, float* sg, float* su, uint32_t hidden, uint32_t topk, uint32_t experts) {
    const uint32_t b = blockIdx.x, rk = blockIdx.y, lane = threadIdx.x;
    const int es = slots[rk];
    if (es < 0 || static_cast<uint32_t>(es) >= experts) return;
    const size_t e = static_cast<size_t>(es);
    const uint32_t row = rk / topk;
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
        const size_t o = static_cast<size_t>(rk) * hidden + lane_order(i);
        const uint16_t hg = half_bits(vg[j] * kHadScale);
        const uint16_t hu = half_bits(vu[j] * kHadScale);
        xg[o] = hg;
        xu[o] = hu;
        sgv += half_value(hg);
        suv += half_value(hu);
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        sgv += __shfl_xor(sgv, m, 32);
        suv += __shfl_xor(suv, m, 32);
    }
    if (lane == 0u) {
        sg[static_cast<size_t>(rk) * (hidden / 128u) + b] = sgv;
        su[static_cast<size_t>(rk) * (hidden / 128u) + b] = suv;
    }
}

constexpr uint32_t kBT = 4;  // output tiles a GEMV threadgroup

// Grid (inter / 16 / BT, RK, KS), 128 threads: waves 0, 1 the gate's two halves of slice ks, 2, 3 the up's.
// zg / zu f32 [KS][RK][inter].
template <int CB, int DEFER>
__device__ void gu3(const uint16_t* xg, const uint16_t* xu, const int* slots, const uint32_t* gate_t,
                    const uint32_t* up_t, float* zg, float* zu, uint32_t hidden, uint32_t inter, uint32_t rk_count,
                    uint32_t experts, uint32_t n, uint32_t mask) {
    __shared__ float red[4][16u * kBT];
    const uint32_t blk = blockIdx.x, rk = blockIdx.y, kz = blockIdx.z, ks = gridDim.z;
    const uint32_t sg = threadIdx.x >> 5, lane = threadIdx.x & 31u, tid = threadIdx.x;
    const int es = slots[rk];
    if (es < 0 || static_cast<uint32_t>(es) >= experts) return;
    const uint32_t proj = sg >> 1, khalf = sg & 1u;
    const uint32_t in_tiles = hidden / 16u, out_tiles = inter / 16u;
    const uint32_t* wbase = (proj == 0u ? gate_t : up_t) + static_cast<size_t>(es) * in_tiles * out_tiles * (n / 2u);
    const uint16_t* xrow = (proj == 0u ? xg : xu) + 4u * (lane & 3u) + static_cast<size_t>(rk) * hidden;
    const uint32_t per = in_tiles / (ks * 2u);
    const uint32_t tk0 = (kz * 2u + khalf) * per;
    float a[kBT][2];
    gemv_tiles<CB, DEFER, kBT>(wbase, xrow, tk0, tk0 + per, blk, out_tiles, n, mask, lane, a);
    fold<kBT>(a, red[sg], lane);
    __syncthreads();
    if (tid < 16u * kBT) {
        const size_t o = (static_cast<size_t>(kz) * rk_count + rk) * inter + blk * (16u * kBT) + tid;
        zg[o] = red[0][tid] + red[1][tid];
        zu[o] = red[2][tid] + red[3][tid];
    }
}

// Grid (inter / 128, RK), 128 threads (one output each): the planes in slice order, the deferred map, the gate and up
// finishes, SwiGLU (the profile's), then the down's prepare into hd f16 [RK][inter] (lane order) and its block sums hs.
template <int PROF>
__device__ void fin3(const float* zg, const float* zu, const int* slots, const uint16_t* gate_svh,
                     const uint16_t* up_svh, const uint16_t* down_suh, const float* xsg, const float* xsu, uint16_t* hd,
                     float* hs, uint32_t hidden, uint32_t inter, uint32_t rk_count, uint32_t ks, uint32_t experts,
                     int defer, float limit) {
    __shared__ float xch[128];
    const uint32_t blk = blockIdx.x, rk = blockIdx.y, tid = threadIdx.x;
    const int es = slots[rk];
    if (es < 0 || static_cast<uint32_t>(es) >= experts) return;
    const uint32_t nb = hidden / 128u;
    float sgx = 0.0f, sux = 0.0f;
    if (defer)
        for (uint32_t q = 0; q < nb; ++q) {
            sgx += xsg[static_cast<size_t>(rk) * nb + q];
            sux += xsu[static_cast<size_t>(rk) * nb + q];
        }
    finish_gate_up<PROF>(zg, zu, rk, rk_count, ks, static_cast<size_t>(es), gate_svh, up_svh, down_suh, sgx, sux, hd, hs,
                         rk, inter, blk, defer, limit, true, xch, tid);
}

// Grid (hidden / 16 / BT, RK, KS2), 128 threads: wave sg takes part sg of slice ks; zd f32 [KS2][RK][hidden].
template <int CB, int DEFER>
__device__ void down3(const uint16_t* hd, const int* slots, const uint32_t* down_t, float* zd, uint32_t hidden,
                      uint32_t inter, uint32_t rk_count, uint32_t experts, uint32_t n, uint32_t mask) {
    __shared__ float red[4][16u * kBT];
    const uint32_t blk = blockIdx.x, rk = blockIdx.y, kz = blockIdx.z, ks = gridDim.z;
    const uint32_t sg = threadIdx.x >> 5, lane = threadIdx.x & 31u, tid = threadIdx.x;
    const int es = slots[rk];
    if (es < 0 || static_cast<uint32_t>(es) >= experts) return;
    const uint32_t in_tiles = inter / 16u, out_tiles = hidden / 16u;
    const uint32_t* wbase = down_t + static_cast<size_t>(es) * in_tiles * out_tiles * (n / 2u);
    const uint16_t* xrow = hd + 4u * (lane & 3u) + static_cast<size_t>(rk) * inter;
    const uint32_t per = in_tiles / (ks * 4u);
    const uint32_t tk0 = (kz * 4u + sg) * per;
    float a[kBT][2];
    gemv_tiles<CB, DEFER, kBT>(wbase, xrow, tk0, tk0 + per, blk, out_tiles, n, mask, lane, a);
    fold<kBT>(a, red[sg], lane);
    __syncthreads();
    if (tid < 16u * kBT)
        zd[(static_cast<size_t>(kz) * rk_count + rk) * hidden + blk * (16u * kBT) + tid] =
            ((red[0][tid] + red[1][tid]) + red[2][tid]) + red[3][tid];
}

// Grid (hidden / 128, rows), 128 threads (one output each). PROF 0: for each slot in order, its planes, the deferred
// map, f16, H128, svh, f16, then fma with its score; y [rows][hidden] in DT. PROF 1: each routed slot's down finish
// stored as f32 ys [RK][hidden] (no f16 rounding, no score).
template <int DT, int PROF>
__device__ void dfin3(const float* zd, const int* slots, const float* scores, const uint16_t* down_svh, const float* hsx,
                      void* y, uint32_t hidden, uint32_t inter, uint32_t topk, uint32_t rk_count, uint32_t ks,
                      uint32_t experts, int defer) {
    __shared__ float xch[128];
    const uint32_t blk = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    const uint32_t o = blk * 128u + tid;
    const uint32_t nbi = inter / 128u;
    float acc = 0.0f;
    for (uint32_t k = 0; k < topk; ++k) {
        const uint32_t rk = row * topk + k;
        const int es = slots[rk];
        if (es < 0 || static_cast<uint32_t>(es) >= experts) continue;  // uniform over the block
        float hsum = 0.0f;
        if (defer)
            for (uint32_t q = 0; q < nbi; ++q) hsum += hsx[static_cast<size_t>(rk) * nbi + q];
        const float v = finish_down<PROF>(zd, rk, rk_count, ks, hsum, defer, down_svh, static_cast<size_t>(es), hidden, o, xch, tid);
        if constexpr (PROF == 0) acc = __builtin_fmaf(scores[rk], v, acc);
        else static_cast<float*>(y)[static_cast<size_t>(rk) * hidden + o] = v;
    }
    if constexpr (PROF == 0) store_act<DT>(y, static_cast<size_t>(row) * hidden + o, acc);
}

}  // namespace

// ---- entry points: t3_dec_<kernel>_<profile: exl3 | glm>[_<dtype>] and t3_dec_<gemv>_<codebook: mcg | mul1 | mul1d>

#define T3_PREP2(PNAME, PROF, DNAME, DT)                                                                             \
    extern "C" __global__ void __launch_bounds__(32) t3_dec_prep2_##PNAME##_##DNAME(                                \
        const void* x, const int* slots, const uint16_t* gate_suh, const uint16_t* up_suh, uint16_t* xg,             \
        uint16_t* xu, float* sg, float* su, uint32_t hidden, uint32_t topk, uint32_t experts) {                     \
        prep2<DT, PROF>(x, slots, gate_suh, up_suh, xg, xu, sg, su, hidden, topk, experts);                         \
    }
T3_PREP2(exl3, 0, f32, kF32)
T3_PREP2(exl3, 0, f16, kF16)
T3_PREP2(exl3, 0, bf16, kBf16)
T3_PREP2(glm, 1, bf16, kBf16)
#undef T3_PREP2

#define T3_GU3(NAME, CB, DEFER)                                                                                      \
    extern "C" __global__ void __launch_bounds__(128) t3_dec_gu3_##NAME(                                            \
        const uint16_t* xg, const uint16_t* xu, const int* slots, const uint32_t* gate_t, const uint32_t* up_t,      \
        float* zg, float* zu, uint32_t hidden, uint32_t inter, uint32_t rk_count, uint32_t experts, uint32_t n,      \
        uint32_t mask) {                                                                                             \
        gu3<CB, DEFER>(xg, xu, slots, gate_t, up_t, zg, zu, hidden, inter, rk_count, experts, n, mask);              \
    }
T3_GU3(mcg, kMcg, 0)
T3_GU3(mul1, kMul1, 0)
T3_GU3(mul1d, kMul1, 1)
#undef T3_GU3

#define T3_FIN3(PNAME, PROF)                                                                                         \
    extern "C" __global__ void __launch_bounds__(128) t3_dec_fin3_##PNAME(                                          \
        const float* zg, const float* zu, const int* slots, const uint16_t* gate_svh, const uint16_t* up_svh,        \
        const uint16_t* down_suh, const float* xsg, const float* xsu, uint16_t* hd, float* hs, uint32_t hidden,      \
        uint32_t inter, uint32_t rk_count, uint32_t ks, uint32_t experts, int defer, float limit) {                   \
        fin3<PROF>(zg, zu, slots, gate_svh, up_svh, down_suh, xsg, xsu, hd, hs, hidden, inter, rk_count, ks,         \
                   experts, defer, limit);                                                                           \
    }
T3_FIN3(exl3, 0)
T3_FIN3(glm, 1)
#undef T3_FIN3

#define T3_DOWN3(NAME, CB, DEFER)                                                                                    \
    extern "C" __global__ void __launch_bounds__(128) t3_dec_down3_##NAME(                                          \
        const uint16_t* hd, const int* slots, const uint32_t* down_t, float* zd, uint32_t hidden, uint32_t inter,    \
        uint32_t rk_count, uint32_t experts, uint32_t n, uint32_t mask) {                                            \
        down3<CB, DEFER>(hd, slots, down_t, zd, hidden, inter, rk_count, experts, n, mask);                          \
    }
T3_DOWN3(mcg, kMcg, 0)
T3_DOWN3(mul1, kMul1, 0)
T3_DOWN3(mul1d, kMul1, 1)
#undef T3_DOWN3

#define T3_DFIN3(PNAME, PROF, DNAME, DT)                                                                             \
    extern "C" __global__ void __launch_bounds__(128) t3_dec_dfin3_##PNAME##_##DNAME(                               \
        const float* zd, const int* slots, const float* scores, const uint16_t* down_svh, const float* hsx, void* y, \
        uint32_t hidden, uint32_t inter, uint32_t topk, uint32_t rk_count, uint32_t ks, uint32_t experts,            \
        int defer) {                                                                                                 \
        dfin3<DT, PROF>(zd, slots, scores, down_svh, hsx, y, hidden, inter, topk, rk_count, ks, experts, defer);    \
    }
T3_DFIN3(exl3, 0, f32, kF32)
T3_DFIN3(exl3, 0, f16, kF16)
T3_DFIN3(exl3, 0, bf16, kBf16)
T3_DFIN3(glm, 1, f32, kF32)
#undef T3_DFIN3
